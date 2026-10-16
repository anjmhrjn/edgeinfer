"""FP32 ONNX -> FP16 ONNX with ModelOpt AutoCast, for TensorRT 11 strongly typed builds (D-024).

TensorRT 11 removed the FP16 builder flag: every layer runs in the type the ONNX model
gives it, so an FP16 engine needs an FP16 model. AutoCast converts every node it judges
safe and keeps a node in FP32 if one of its weights exceeds `init_max` or its output
exceeds `data_max` on the calibration images (a reference run on ORT CPU).

ModelOpt pins onnx/onnxruntime versions older than this project's, so it lives in its own
environment (tools/modelopt/, own uv.lock) and runs as a subprocess. This module never
imports it.

The FP16 file is content-addressed like the other artifacts: its id hashes the FP32
artifact id, the FP16 config, the calibration images and tools/modelopt/uv.lock.
"""

from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import yaml

from edgeinfer.artifacts import (
    INPUT_NAME,
    META_PREFIX,
    ONNX_DIR,
    canonical_json,
    load_sidecar,
    sha256_file,
    sidecar_path,
)
from edgeinfer.config import CONFIG_DIR, ConfigError, ModelConfig, _take
from edgeinfer.data.cityscapes import iter_samples
from edgeinfer.data.manifests import ManifestEntry, format_manifest
from edgeinfer.preprocess import preprocess
from edgeinfer.quant import QuantError
from edgeinfer.quant.calib import CALIB_DIR, train_entries
from edgeinfer.reports import REPO_ROOT

PRECISION = "fp16"
MODELOPT_PROJECT = REPO_ROOT / "tools" / "modelopt"
RUNNER = MODELOPT_PROJECT / "autocast_fp16.py"
_FP16, _FP32 = onnx.TensorProto.FLOAT16, onnx.TensorProto.FLOAT


@dataclass(frozen=True)
class Fp16Calibration:
    manifest: str  # train split only
    limit: int | None


@dataclass(frozen=True)
class Fp16Config:
    name: str
    low_precision_type: str
    keep_io_types: bool
    data_max: float
    init_max: float
    nodes_to_exclude: tuple[str, ...]
    op_types_to_exclude: tuple[str, ...]
    calibration: Fp16Calibration

    @classmethod
    def from_dict(cls, d: Any, where: str) -> Fp16Config:
        d = _take(cls, d, where)
        cal = _take(Fp16Calibration, d["calibration"], f"{where}.calibration")
        if d["low_precision_type"] != "fp16":
            raise ConfigError(f"{where}.low_precision_type: only 'fp16' (the T4 has no BF16)")
        if not isinstance(d["keep_io_types"], bool):
            raise ConfigError(f"{where}.keep_io_types: expected true or false")
        for key in ("data_max", "init_max"):
            if not isinstance(d[key], int | float) or d[key] <= 0:
                raise ConfigError(f"{where}.{key}: expected a positive number, got {d[key]!r}")
        for key in ("nodes_to_exclude", "op_types_to_exclude"):
            if not isinstance(d[key], list) or not all(isinstance(s, str) for s in d[key]):
                raise ConfigError(f"{where}.{key}: expected a list of names")
        if cal["limit"] is not None and (not isinstance(cal["limit"], int) or cal["limit"] <= 0):
            raise ConfigError(f"{where}.calibration.limit: expected a positive int or null")
        return cls(
            name=str(d["name"]),
            low_precision_type=d["low_precision_type"],
            keep_io_types=d["keep_io_types"],
            data_max=float(d["data_max"]),
            init_max=float(d["init_max"]),
            nodes_to_exclude=tuple(d["nodes_to_exclude"]),
            op_types_to_exclude=tuple(d["op_types_to_exclude"]),
            calibration=Fp16Calibration(manifest=str(cal["manifest"]), limit=cal["limit"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return json.loads(canonical_json(asdict(self)))


def load_fp16_config(name_or_path: str | Path) -> Fp16Config:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / "precision" / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"FP16 config not found: {path}")
    return Fp16Config.from_dict(yaml.safe_load(path.read_text()), where=path.name)


def toolchain_id() -> str:
    """Hash of tools/modelopt/uv.lock, which pins ModelOpt and everything it runs on."""
    return sha256_file(MODELOPT_PROJECT / "uv.lock")


def fp16_artifact_id(fp32_id: str, fcfg: Fp16Config, entries: list[ManifestEntry]) -> str:
    key = {
        "fp32_artifact_id": fp32_id,
        "fp16_config": fcfg.to_dict(),
        "calibration_sha256": hashlib.sha256(format_manifest(entries).encode()).hexdigest(),
        "modelopt_lock_sha256": toolchain_id(),
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()


def write_calibration_inputs(
    cfg: ModelConfig, entries: list[ManifestEntry], root: Path, out_dir: Path
) -> Path:
    """Preprocessed calibration inputs, one NPZ per image (AutoCast's multi-batch format).
    Derived from Cityscapes: they live under artifacts/ and are never committed."""
    done = out_dir / "COMPLETE"
    if done.is_file():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(iter_samples(root, entries)):
        np.savez(out_dir / f"{i:04d}.npz", **{INPUT_NAME: preprocess(sample.image, cfg)})
    done.write_text(format_manifest(entries))
    return out_dir


Runner = Callable[[dict[str, Any]], dict[str, Any]]


def run_autocast_subprocess(args: dict[str, Any]) -> dict[str, Any]:
    """Run tools/modelopt/autocast_fp16.py in the ModelOpt environment; returns its summary."""
    cmd = ["uv", "run", "--frozen", "--project", str(MODELOPT_PROJECT), "python", str(RUNNER),
           json.dumps(args)]  # fmt: skip
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}  # not the main venv
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, check=False)
    if proc.returncode != 0:
        tail = "\n".join(proc.stderr.strip().splitlines()[-15:])
        raise QuantError("FP16_AUTOCAST_FAILED", f"AutoCast exited {proc.returncode}:\n{tail}")
    # ModelOpt logs to stdout; the runner's last line is its JSON summary. The log says
    # which nodes AutoCast kept in FP32 and why, so it's kept with the artifact.
    *log, last = proc.stdout.strip().splitlines()
    sys.stderr.write("\n".join(log) + "\n" + proc.stderr)
    return json.loads(last) | {"log": log}


def precision_summary(model: onnx.ModelProto) -> dict[str, Any]:
    """Compute nodes by the type of their first output, after type inference. Cast nodes
    are counted apart; nodes producing integers (shape arithmetic) aren't compute."""
    inferred = onnx.shape_inference.infer_shapes(model)
    types = {
        vi.name: vi.type.tensor_type.elem_type
        for vi in [*inferred.graph.value_info, *inferred.graph.input, *inferred.graph.output]
    }
    types |= {t.name: t.data_type for t in model.graph.initializer}
    fp16: Counter = Counter()
    fp32: list[str] = []
    casts = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            casts += 1
            continue
        dtype = types.get(node.output[0]) if node.output else None
        if dtype == _FP16:
            fp16[node.op_type] += 1
        elif dtype == _FP32:
            fp32.append(node.name or node.output[0])
    return {"fp16_nodes": sum(fp16.values()), "fp16_by_op": dict(sorted(fp16.items())),
            "fp32_nodes": fp32, "cast_nodes": casts}  # fmt: skip


def _children_peak_rss_bytes() -> int:
    peak = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def convert_fp16(
    fp32_path: Path,
    cfg: ModelConfig,
    fcfg: Fp16Config,
    root: Path,
    out_root: Path = ONNX_DIR,
    calib_root: Path = CALIB_DIR,
    runner: Runner = run_autocast_subprocess,
) -> tuple[Path, dict[str, Any]]:
    """Convert an FP32 export to FP16. Returns (path, sidecar); reuses an existing artifact
    with the same id."""
    fp32_meta = load_sidecar(fp32_path)
    if fp32_meta["precision"] != "fp32":
        raise QuantError(
            "QUANT_INPUT_NOT_FP32",
            f"{fp32_path.name} has precision {fp32_meta['precision']!r}; convert the FP32 export",
        )
    entries = train_entries(fcfg.calibration.manifest, fcfg.calibration.limit)
    art_id = fp16_artifact_id(fp32_meta["artifact_id"], fcfg, entries)
    path = out_root / cfg.name / f"{cfg.name}__{PRECISION}__{art_id[:12]}.onnx"
    if path.is_file() and sidecar_path(path).is_file():
        return path, load_sidecar(path)

    t0 = time.perf_counter()
    calib_dir = write_calibration_inputs(
        cfg, entries, root, calib_root / cfg.name / f"fp16__{art_id[:12]}"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".onnx", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        summary = runner({
            "onnx_path": str(fp32_path.resolve()), "output_path": str(tmp.resolve()),
            "low_precision_type": fcfg.low_precision_type, "keep_io_types": fcfg.keep_io_types,
            "data_max": fcfg.data_max, "init_max": fcfg.init_max,
            "nodes_to_exclude": list(fcfg.nodes_to_exclude),
            "op_types_to_exclude": list(fcfg.op_types_to_exclude),
            "calibration_data": str(calib_dir.resolve()),
        })  # fmt: skip
        model = onnx.load(str(tmp))
        # AutoCast drops metadata_props; the I/O contract must survive the conversion.
        fp32_props = {p.key: p.value for p in onnx.load(str(fp32_path)).metadata_props}
        onnx.helper.set_model_props(model, {
            **fp32_props,
            META_PREFIX + "artifact_id": art_id,
            META_PREFIX + "parent_artifact_id": fp32_meta["artifact_id"],
            META_PREFIX + "precision": PRECISION,
        })  # fmt: skip
        placement = precision_summary(model)
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
        "parent_artifact_id": fp32_meta["artifact_id"],
        "converter": "modelopt.onnx.autocast.convert_to_mixed_precision",
        "fp16_config": fcfg.to_dict(),
        "calibration": {
            "manifest": fcfg.calibration.manifest,
            "limit": fcfg.calibration.limit,
            "n_images": len(entries),
        },
        "placement": placement,
        # The subprocess runs its own onnx/onnxruntime; prefixed so a report never confuses
        # them with the main environment's.
        "versions": {
            **{f"modelopt_env.{k}": v for k, v in summary["versions"].items()},
            "modelopt_env.lock_sha256": toolchain_id(),
        },
        "contract": fp32_meta["contract"],
        "autocast_seconds": summary["seconds"],
        "convert_seconds": round(time.perf_counter() - t0, 2),
        "autocast_peak_rss_bytes": _children_peak_rss_bytes(),
        "autocast_log": summary.get("log", []),
    }
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta
