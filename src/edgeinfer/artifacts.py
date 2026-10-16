"""Artifact identity, sidecars and the model I/O contract, without PyTorch.

Everything that runs on the GPU instance (ORT reference, validator, TensorRT backend,
performance probe) imports from here instead of from `export.onnx_export`, which
needs torch. Keeping torch out of those processes keeps a second CUDA runtime out of
the process that runs TensorRT (D-020).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from edgeinfer.config import ModelConfig
from edgeinfer.reports import REPO_ROOT

ONNX_DIR = REPO_ROOT / "artifacts" / "onnx"

INPUT_NAME = "input"
OUTPUT_NAME = "logits"

META_PREFIX = "edgeinfer."
CONTRACT_KEY = META_PREFIX + "contract"


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def io_contract(cfg: ModelConfig) -> dict[str, Any]:
    """What a runtime needs to feed the model and interpret its output.

    Normalized through JSON (tuples -> lists) so it compares equal to the embedded copy."""
    contract = {
        "model": cfg.name,
        "num_classes": cfg.num_classes,
        "output_stride": cfg.output_stride,
        "input_name": INPUT_NAME,
        "output_name": OUTPUT_NAME,
        "input": asdict(cfg.input),
        "preprocess": asdict(cfg.preprocess),
        "postprocess": asdict(cfg.postprocess),
    }
    return json.loads(canonical_json(contract))


def sidecar_path(path: Path) -> Path:
    return path.with_suffix(".meta.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_sidecar(path: Path) -> dict[str, Any]:
    """Sidecar of an artifact, checked against the file it describes."""
    meta_path = sidecar_path(path)
    if not meta_path.is_file():
        raise FileNotFoundError(f"no sidecar for {path} (expected {meta_path})")
    meta = json.loads(meta_path.read_text())
    if meta["sha256"] != sha256_file(path):
        raise ValueError(f"{path} does not match the sha256 in {meta_path}; file was modified")
    return meta


def artifact_identity(path: Path) -> dict[str, str]:
    """artifact_id and precision of a model file: from its sidecar, or by content for
    derived files without one (e.g. a modified copy made by a test)."""
    try:
        meta = load_sidecar(path)
    except FileNotFoundError:
        return {"artifact_id": f"sha256:{sha256_file(path)}", "precision": "unknown"}
    return {"artifact_id": meta["artifact_id"], "precision": meta["precision"]}
