"""PyTorch -> ONNX export (FP32, static batch-1 shape).

The exported file is content-addressed (design principle 4):
    artifact_id = sha256 of (weights id, model config, export settings, exporter versions)
and lives at artifacts/onnx/<model>/<model>__fp32__<id[:12]>.onnx with a
`.meta.json` sidecar. The model's I/O contract (input spec, preprocessing,
postprocessing, classes, stride) is embedded in the ONNX `metadata_props`, so the
file describes itself to other runtimes and the validator can check a config
against the model it is used with.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch

from edgeinfer.artifacts import (
    CONTRACT_KEY,
    INPUT_NAME,
    META_PREFIX,
    ONNX_DIR,
    OUTPUT_NAME,
    canonical_json,
    io_contract,
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from edgeinfer.config import ModelConfig
from edgeinfer.models.registry import build_model, weights_id
from edgeinfer.reports import package_versions

OPSET = 20  # D-013
EXPORT_SETTINGS = {
    "exporter": "torch.onnx.export(dynamo=True)",
    "opset": OPSET,
    "optimize": True,
    "external_data": False,
    "static_shape": True,
}
# Node metadata the exporter adds that contains absolute source paths. It would leak
# the local checkout path and make the bytes depend on where the repo lives.
_STRIP_NODE_METADATA = ("pkg.torch.onnx.stack_trace",)

# Post-export sanity bound, relative to max |torch logits|. Only catches a broken
# export; parity tolerances (configs/tolerances.yaml) are the real check.
_SANITY_REL = 1e-3


class ExportError(RuntimeError):
    pass


def _exporter_versions() -> dict[str, str]:
    v = package_versions("torch", "onnx", "onnxscript")
    return {k: v[k] for k in ("torch", "onnx", "onnxscript")}


def artifact_id(cfg: ModelConfig) -> str:
    key = {
        "weights_id": weights_id(cfg),
        "model_config": asdict(cfg),
        "export": EXPORT_SETTINGS,
        "versions": _exporter_versions(),
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()


def artifact_path(cfg: ModelConfig, out_root: Path = ONNX_DIR) -> Path:
    return out_root / cfg.name / f"{cfg.name}__fp32__{artifact_id(cfg)[:12]}.onnx"


def _strip_and_annotate(model: onnx.ModelProto, cfg: ModelConfig, art_id: str) -> None:
    for node in model.graph.node:
        keep = [p for p in node.metadata_props if p.key not in _STRIP_NODE_METADATA]
        del node.metadata_props[:]
        node.metadata_props.extend(keep)
    props = {
        CONTRACT_KEY: canonical_json(io_contract(cfg)),
        META_PREFIX + "artifact_id": art_id,
        META_PREFIX + "weights_id": weights_id(cfg),
    }
    onnx.helper.set_model_props(model, props)


def _sanity_check(path: Path, net: torch.nn.Module, cfg: ModelConfig) -> dict[str, float]:
    import onnxruntime as ort

    shape = (1, 3, cfg.input.height, cfg.input.width)
    x = np.random.default_rng(0).standard_normal(shape).astype(np.float32)
    with torch.inference_mode():
        ref = net(torch.from_numpy(x)).numpy()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    out = sess.run([OUTPUT_NAME], {INPUT_NAME: x})[0]
    if out.shape != ref.shape:
        raise ExportError(f"ONNX output shape {out.shape} != torch output shape {ref.shape}")
    max_abs = float(np.abs(out - ref).max())
    ref_scale = float(np.abs(ref).max())
    if not max_abs <= _SANITY_REL * max(ref_scale, 1.0):
        raise ExportError(
            f"post-export check failed: ORT vs torch max abs error {max_abs:.3g} "
            f"(max |logit| {ref_scale:.3g}); the export is wrong"
        )
    return {"sanity_max_abs": max_abs, "sanity_max_abs_ref": ref_scale}


def export_model(
    cfg: ModelConfig, out_root: Path = ONNX_DIR, force: bool = False
) -> tuple[Path, dict[str, Any]]:
    """Export `cfg` to ONNX; returns (path, sidecar). Reuses an existing artifact
    unless `force`."""
    art_id = artifact_id(cfg)
    path = artifact_path(cfg, out_root)
    if path.exists() and not force:
        return path, load_sidecar(path)

    net = build_model(cfg).to("cpu").float()
    x = torch.zeros(1, 3, cfg.input.height, cfg.input.width)
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Write to a temp file in the same directory; only a checked export gets the final name.
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        torch.onnx.export(
            net,
            (x,),
            str(tmp),
            dynamo=True,
            opset_version=OPSET,
            input_names=[INPUT_NAME],
            output_names=[OUTPUT_NAME],
            external_data=False,
            optimize=True,
        )
        model = onnx.load(str(tmp))
        _strip_and_annotate(model, cfg, art_id)
        onnx.save(model, str(tmp))
        sanity = _sanity_check(tmp, net, cfg)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)

    meta = {
        "artifact_id": art_id,
        "kind": "onnx",
        "model": cfg.name,
        "precision": "fp32",
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "weights_id": weights_id(cfg),
        "export": EXPORT_SETTINGS,
        "versions": _exporter_versions(),
        "contract": io_contract(cfg),
        "export_seconds": round(time.perf_counter() - t0, 2),
        **sanity,
    }
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta
