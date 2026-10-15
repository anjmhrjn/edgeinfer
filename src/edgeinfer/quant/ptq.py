"""FP32 ONNX -> INT8 QDQ ONNX with ORT `quantize_static` (D-015, D-016).

The output is content-addressed like the FP32 export:
    artifact_id = sha256 of (FP32 artifact id, quant config, calibration id,
                             excluded nodes, quantizer versions)
and lives at artifacts/onnx/<model>/<model>__int8__<id[:12]>.onnx with a sidecar.
Exclusions are resolved to node names before hashing, so two lists that keep the same
nodes in float give the same artifact.

After quantizing, the placement is checked against what was asked for: every
quantized node must read Q/DQ'd activations and weights, every excluded node must
read none. An INT8 model that silently isn't one would make every accuracy number
about it meaningless.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantFormat, QuantType, quantize_static

from edgeinfer.config import ModelConfig
from edgeinfer.export.onnx_export import (
    META_PREFIX,
    ONNX_DIR,
    canonical_json,
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from edgeinfer.quant import QuantError
from edgeinfer.quant.calib import ORT_METHODS
from edgeinfer.quant.config import Exclusions, GroupRules, QuantConfig
from edgeinfer.quant.groups import assign_groups, resolve_exclusions

PRECISION = "int8"
QUANTIZER = "onnxruntime.quantization.quantize_static"
_QTYPES = {"int8": QuantType.QInt8, "uint8": QuantType.QUInt8}


def _versions() -> dict[str, str]:
    return {"onnxruntime": ort.__version__, "onnx": onnx.__version__}


def quant_artifact_id(
    fp32_id: str, qcfg: QuantConfig, calibration_id: str, excluded: list[str]
) -> str:
    settings = qcfg.to_dict()
    settings.pop("name")  # a label; the settings are what matter
    key = {
        "fp32_artifact_id": fp32_id,
        "quant": settings,
        "calibration_id": calibration_id,
        "excluded_nodes": excluded,
        "quantizer": QUANTIZER,
        "versions": _versions(),
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()


def _ort_kwargs(qcfg: QuantConfig, cache: Path, excluded: list[str]) -> dict[str, Any]:
    ops = list(qcfg.op_types)
    return {
        "quant_format": QuantFormat.QDQ,
        "per_channel": qcfg.per_channel,
        "activation_type": _QTYPES[qcfg.activation_type],
        "weight_type": _QTYPES[qcfg.weight_type],
        "op_types_to_quantize": ops,
        "nodes_to_exclude": excluded,
        "calibrate_method": ORT_METHODS[qcfg.calibration.method],
        "calibration_cache_path": str(cache),
        "extra_options": {
            "ActivationSymmetric": qcfg.activation_symmetric,
            "WeightSymmetric": qcfg.weight_symmetric,
            "QuantizeBias": qcfg.quantize_bias,
            "OpTypesToExcludeOutputQuantization": [] if qcfg.quantize_outputs else ops,
        },
    }


def check_placement(
    model: onnx.ModelProto, op_types: tuple[str, ...], excluded: list[str]
) -> dict[str, Any]:
    """Placement stats; raises QUANT_PLACEMENT_MISMATCH if a node that should be
    quantized reads a float input, or an excluded node reads a dequantized one."""
    producer = {o: n.op_type for n in model.graph.node for o in n.output}
    excluded_set = set(excluded)
    quantized: Counter[str] = Counter()
    kept: Counter[str] = Counter()
    wrong = []
    for node in model.graph.node:
        if node.op_type not in op_types:
            continue
        # Data and weight inputs; a Conv bias (input 2) stays float by design (D-016).
        dq = [producer.get(i) == "DequantizeLinear" for i in node.input[:2]]
        if node.name in excluded_set:
            kept[node.op_type] += 1
            if any(dq):
                wrong.append(f"{node.name} is excluded but reads a dequantized input")
        else:
            quantized[node.op_type] += 1
            if not all(dq):
                wrong.append(f"{node.name} should be quantized but reads a float input")
    if wrong:
        raise QuantError("QUANT_PLACEMENT_MISMATCH", f"{len(wrong)} nodes: {wrong[:5]}")
    ops = Counter(n.op_type for n in model.graph.node)
    return {
        "quantized_nodes": dict(sorted(quantized.items())),
        "excluded_nodes": dict(sorted(kept.items())),
        "n_quantize_linear": ops["QuantizeLinear"],
        "n_dequantize_linear": ops["DequantizeLinear"],
    }


def quantize(
    fp32_path: Path,
    cfg: ModelConfig,
    qcfg: QuantConfig,
    calibration: tuple[Path, dict[str, Any]],
    rules: GroupRules,
    exclusions: Exclusions,
    out_root: Path = ONNX_DIR,
) -> tuple[Path, dict[str, Any]]:
    """Quantize with a calibration cache. Returns (path, sidecar); reuses an existing
    artifact with the same id."""
    fp32_meta = load_sidecar(fp32_path)
    if fp32_meta["precision"] != "fp32":
        raise QuantError(
            "QUANT_INPUT_NOT_FP32",
            f"{fp32_path.name} has precision {fp32_meta['precision']!r}; quantize the FP32 export",
        )
    cache, cal_meta = calibration
    if cal_meta["fp32_artifact_id"] != fp32_meta["artifact_id"]:
        raise QuantError("QUANT_CALIB_MISMATCH", f"{cache.name} was calibrated on another model")
    fp32 = onnx.load(str(fp32_path))
    groups = assign_groups(fp32, rules, qcfg.op_types)
    excluded = resolve_exclusions(exclusions, groups, fp32)
    art_id = quant_artifact_id(fp32_meta["artifact_id"], qcfg, cal_meta["calibration_id"], excluded)
    path = out_root / cfg.name / f"{cfg.name}__{PRECISION}__{art_id[:12]}.onnx"
    if path.is_file() and sidecar_path(path).is_file():
        return path, load_sidecar(path)

    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    root_logger = logging.getLogger()
    level = root_logger.level
    try:
        # quantize_static logs a generic "consider pre-processing" warning on every call;
        # the model is already shape-inferred and optimized by the exporter.
        root_logger.setLevel(logging.ERROR)
        quantize_static(str(fp32_path), str(tmp), None, **_ort_kwargs(qcfg, cache, excluded))
        root_logger.setLevel(level)
        model = onnx.load(str(tmp))
        placement = check_placement(model, qcfg.op_types, excluded)
        onnx.helper.set_model_props(model, {
            **{p.key: p.value for p in model.metadata_props},
            META_PREFIX + "artifact_id": art_id,
            META_PREFIX + "parent_artifact_id": fp32_meta["artifact_id"],
            META_PREFIX + "precision": PRECISION,
        })  # fmt: skip
        onnx.save(model, str(tmp))
        tmp.replace(path)
    finally:
        root_logger.setLevel(level)
        tmp.unlink(missing_ok=True)

    group_of = {n: g for g, nodes in groups.items() for n in nodes}
    meta = {
        "artifact_id": art_id,
        "kind": "onnx",
        "model": cfg.name,
        "precision": PRECISION,
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "parent_artifact_id": fp32_meta["artifact_id"],
        "quantizer": QUANTIZER,
        "quant_config": qcfg.to_dict(),
        "calibration": {k: cal_meta[k] for k in
                        ("calibration_id", "manifest", "limit", "n_images", "method", "options")},
        "exclusions": exclusions.to_dict(),
        "excluded_nodes": [{"node": n, "group": group_of.get(n)} for n in excluded],
        "placement": placement,
        "versions": _versions(),
        "contract": fp32_meta["contract"],
        "quantize_seconds": round(time.perf_counter() - t0, 2),
    }  # fmt: skip
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta
