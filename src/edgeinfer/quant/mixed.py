"""INT8 Q/DQ ONNX with FP32 float parts -> INT8 Q/DQ with FP16 float parts (D-031).

TensorRT 11 networks are strongly typed (D-024): every layer runs in the type its ONNX
model gives it. The layers an INT8 Q/DQ model from `quant/ptq.py` leaves unquantized
(excluded groups, LayerNorm, softmax, Resize, ...) would therefore run in FP32. ModelOpt's
`convert_to_f16`, the step its own quantizer runs after inserting Q/DQ, converts everything
outside the Q/DQ pairs to FP16, including the Q/DQ scales. INT8 weights, zero points and
the placement are unchanged, so an `int8_fp16` model keeps its parent's calibration and
exclusions, and C3's ORT results still describe its quantization.

The conversion runs in the ModelOpt environment as a subprocess (tools/modelopt/, D-024).
The output is content-addressed: its id hashes the parent INT8 artifact id, the converter
settings and tools/modelopt/uv.lock.

FP16 scales: a scale below FP16's smallest normal value (6.1e-5) is stored as a subnormal
with fewer significant bits. The sidecar records how many there are and the largest
relative change of any scale, so a rounding problem shows up as a number, not a guess.

ORT CPU can't run these models with its graph optimizations on (it fuses MatMul+Add into
QGemm, which rejects FP16 scales, and mis-handles DeepLab), so `ort_cpu` runs `int8_fp16`
with optimizations off. That is a check of the conversion; TensorRT is the real runtime.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnx import numpy_helper

from edgeinfer.artifacts import (
    META_PREFIX,
    ONNX_DIR,
    canonical_json,
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from edgeinfer.config import ModelConfig
from edgeinfer.quant import QuantError
from edgeinfer.quant.fp16 import (
    MODELOPT_PROJECT,
    Runner,
    precision_summary,
    run_modelopt_script,
    toolchain_id,
)
from edgeinfer.quant.ptq import PRECISION as INT8
from edgeinfer.quant.ptq import check_placement

PRECISION = "int8_fp16"
CONVERTER = "modelopt.onnx.autocast.convert.convert_to_f16"
RUNNER = MODELOPT_PROJECT / "qdq_fp16.py"
SETTINGS = {"low_precision_type": "fp16", "keep_io_types": True}  # I/O contract unchanged
FP16_MIN_NORMAL = float(np.finfo(np.float16).tiny)
# Sidecar fields copied from the parent INT8 model: the quantization itself is unchanged.
INHERITED = ("quantizer", "quant_config", "calibration", "exclusions", "excluded_nodes")


def run_qdq_fp16_subprocess(args: dict[str, Any]) -> dict[str, Any]:
    """Run tools/modelopt/qdq_fp16.py in the ModelOpt environment; returns its summary."""
    return run_modelopt_script(RUNNER, args, "convert_to_f16", "QUANT_FP16_CONVERT_FAILED")


def int8_fp16_artifact_id(int8_id: str) -> str:
    key = {
        "int8_artifact_id": int8_id,
        "converter": CONVERTER,
        "settings": SETTINGS,
        "modelopt_lock_sha256": toolchain_id(),
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()


def _scales(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    """Q/DQ scale values by node output name (stable across the conversion)."""
    inits = {t.name: t for t in model.graph.initializer}
    out = {}
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear") and node.input[1] in inits:
            out[node.output[0]] = numpy_helper.to_array(inits[node.input[1]]).astype(np.float64)
    return out


def scale_rounding(parent: onnx.ModelProto, converted: onnx.ModelProto) -> dict[str, Any]:
    """How much converting the Q/DQ scales to FP16 changed them."""
    before, after = _scales(parent), _scales(converted)
    if before.keys() != after.keys():
        missing = sorted(before.keys() ^ after.keys())
        raise QuantError(
            "QUANT_FP16_PLACEMENT_CHANGED",
            f"Q/DQ nodes differ after the FP16 conversion ({len(missing)}): {missing[:5]}",
        )
    n = subnormal = 0
    worst, where = 0.0, ""
    for name, s in before.items():
        rel = float(np.max(np.abs(after[name] - s) / s))
        n += s.size
        subnormal += int((s < FP16_MIN_NORMAL).sum())
        if rel > worst:
            worst, where = rel, name
    return {"n_values": n, "subnormal_in_fp16": subnormal, "max_rel_change": worst,
            "max_rel_change_at": where}  # fmt: skip


def convert_int8_fp16(
    int8_path: Path,
    cfg: ModelConfig,
    out_root: Path = ONNX_DIR,
    runner: Runner = run_qdq_fp16_subprocess,
) -> tuple[Path, dict[str, Any]]:
    """Convert an INT8 Q/DQ model's float parts to FP16. Returns (path, sidecar); reuses an
    existing artifact with the same id."""
    parent = load_sidecar(int8_path)
    if parent["precision"] != INT8:
        raise QuantError(
            "QUANT_INPUT_NOT_INT8",
            f"{int8_path.name} has precision {parent['precision']!r}; convert an INT8 Q/DQ "
            "model from `edgeinfer quantize`",
        )
    art_id = int8_fp16_artifact_id(parent["artifact_id"])
    path = out_root / cfg.name / f"{cfg.name}__{PRECISION}__{art_id[:12]}.onnx"
    if path.is_file() and sidecar_path(path).is_file():
        return path, load_sidecar(path)

    t0 = time.perf_counter()
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    parent_model = onnx.load(str(int8_path))
    try:
        summary = runner({"onnx_path": str(int8_path.resolve()),
                          "output_path": str(tmp.resolve()),
                          "keep_io_types": SETTINGS["keep_io_types"]})  # fmt: skip
        model = onnx.load(str(tmp))
        # convert_to_f16 drops metadata_props; the I/O contract must survive.
        props = {p.key: p.value for p in parent_model.metadata_props}
        onnx.helper.set_model_props(model, {
            **props,
            META_PREFIX + "artifact_id": art_id,
            META_PREFIX + "parent_artifact_id": parent["artifact_id"],
            META_PREFIX + "precision": PRECISION,
        })  # fmt: skip
        excluded = [e["node"] for e in parent["excluded_nodes"]]
        placement = check_placement(model, tuple(parent["quant_config"]["op_types"]), excluded)
        if placement["quantized_nodes"] != parent["placement"]["quantized_nodes"]:
            raise QuantError(
                "QUANT_FP16_PLACEMENT_CHANGED",
                f"quantized nodes {placement['quantized_nodes']} after the FP16 conversion, "
                f"{parent['placement']['quantized_nodes']} before",
            )
        rounding = scale_rounding(parent_model, model)
        floats = precision_summary(model)
        onnx.save(model, str(tmp))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

    meta = {
        "artifact_id": art_id,
        "kind": "onnx",
        "model": cfg.name,
        "precision": PRECISION,
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "parent_artifact_id": parent["artifact_id"],
        "fp32_artifact_id": parent["parent_artifact_id"],
        "converter": CONVERTER,
        "converter_settings": SETTINGS,
        **{k: parent[k] for k in INHERITED},
        "placement": placement | {"unshared_inputs": parent["placement"].get("unshared_inputs")},
        "float_precision": floats,
        "scale_rounding": rounding,
        "versions": {
            **parent["versions"],
            **{f"modelopt_env.{k}": v for k, v in summary["versions"].items()},
            "modelopt_env.lock_sha256": toolchain_id(),
        },
        "contract": parent["contract"],
        "convert_seconds": round(time.perf_counter() - t0, 2),
        "converter_log": summary.get("log", []),
    }
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta
