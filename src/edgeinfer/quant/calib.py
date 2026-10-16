"""Calibration: activation ranges of the FP32 model on the calibration manifest.

ORT's histogram calibrators (entropy, percentile) keep every intermediate output of
everything fed to one `collect_data` call in memory before building histograms. At
512x1024, SegFormer-B0's intermediates exceed 1 GB per image, so 200 images can't go
through one call on a 16 GB machine. We feed `chunk_images` images per call; the
histogram collector merges across calls.

The result is ORT's own calibration-cache JSON (`save_tensors_data`), which
`quantize_static(calibration_cache_path=...)` loads instead of calibrating again. The
cache is content-addressed and gitignored: tensor ranges are derived from Cityscapes
images and are never committed.
"""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.calibrate import create_calibrator, save_tensors_data

from edgeinfer.artifacts import INPUT_NAME, canonical_json, load_sidecar
from edgeinfer.config import ModelConfig
from edgeinfer.data.cityscapes import iter_samples
from edgeinfer.data.manifests import ManifestEntry, format_manifest, load_manifest
from edgeinfer.preprocess import preprocess
from edgeinfer.quant import QuantError
from edgeinfer.quant.config import CalibrationSpec, QuantConfig
from edgeinfer.reports import REPO_ROOT

CALIB_DIR = REPO_ROOT / "artifacts" / "calib"
CALIBRATION_SPLIT = "train"

ORT_METHODS = {
    "minmax": CalibrationMethod.MinMax,
    "entropy": CalibrationMethod.Entropy,
    "percentile": CalibrationMethod.Percentile,
}


def calibration_entries(spec: CalibrationSpec) -> list[ManifestEntry]:
    """The calibration images. Anything outside the train split is refused: calibrating
    on val would tune the quantizer on the images it is evaluated on."""
    entries = load_manifest(spec.manifest)
    if spec.limit is not None:
        if spec.limit > len(entries):
            raise QuantError(
                "QUANT_CALIB_TOO_SMALL",
                f"calibration.limit {spec.limit} > {len(entries)} images in {spec.manifest}",
            )
        entries = entries[: spec.limit]
    bad = [e for e in entries if e.split != CALIBRATION_SPLIT]
    if bad:
        raise QuantError(
            "QUANT_CALIB_ON_EVAL_SPLIT",
            f"calibration manifest {spec.manifest!r} has {len(bad)}/{len(entries)} images "
            f"outside the {CALIBRATION_SPLIT} split (first: {bad[0]}); never calibrate on "
            "evaluation data",
        )
    return entries


def _calib_options(qcfg: QuantConfig) -> dict[str, Any]:
    spec = qcfg.calibration
    opts: dict[str, Any] = {"symmetric": qcfg.activation_symmetric}
    for key in ("num_bins", "num_quantized_bins", "percentile"):
        if getattr(spec, key) is not None:
            opts[key] = getattr(spec, key)
    return opts


def calibration_id(fp32_id: str, entries: list[ManifestEntry], qcfg: QuantConfig) -> str:
    spec = qcfg.calibration
    key = {
        "fp32_artifact_id": fp32_id,
        "manifest_sha256": hashlib.sha256(format_manifest(entries).encode()).hexdigest(),
        "method": spec.method,
        "options": _calib_options(qcfg),
        "op_types": list(qcfg.op_types),
        "onnxruntime": ort.__version__,
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()


class _ChunkReader(CalibrationDataReader):
    """Yields at most `n` preprocessed images from a shared sample iterator."""

    def __init__(self, samples: Iterator, cfg: ModelConfig, n: int) -> None:
        self._samples, self._cfg, self._left = samples, cfg, n

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self._left == 0:
            return None
        sample = next(self._samples, None)
        if sample is None:
            return None
        self._left -= 1
        return {INPUT_NAME: preprocess(sample.image, self._cfg)}


def _peak_rss_bytes() -> int:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss if sys.platform == "darwin" else rss * 1024  # Linux reports KiB


def calibrate(
    fp32_path: Path, cfg: ModelConfig, qcfg: QuantConfig, root: Path, out_root: Path = CALIB_DIR
) -> tuple[Path, dict[str, Any]]:
    """Calibrate (or reuse a cached calibration). Returns (cache path, sidecar)."""
    entries = calibration_entries(qcfg.calibration)
    fp32_id = load_sidecar(fp32_path)["artifact_id"]
    cal_id = calibration_id(fp32_id, entries, qcfg)
    path = out_root / cfg.name / f"{cfg.name}__{qcfg.calibration.method}__{cal_id[:12]}.json"
    meta_path = path.with_suffix(".meta.json")
    if path.is_file() and meta_path.is_file():
        return path, json.loads(meta_path.read_text())

    spec = qcfg.calibration
    t0 = time.perf_counter()
    samples = iter_samples(root, entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="calib.", dir=path.parent) as tmp:
        calibrator = create_calibrator(
            fp32_path,
            op_types_to_calibrate=list(qcfg.op_types),
            augmented_model_path=str(Path(tmp) / "augmented.onnx"),
            calibrate_method=ORT_METHODS[spec.method],
            extra_options=_calib_options(qcfg),
        )
        done = 0
        while done < len(entries):
            n = min(spec.chunk_images, len(entries) - done)
            calibrator.collect_data(_ChunkReader(samples, cfg, n))
            done += n
            print(f"  calibration: {done}/{len(entries)} images", file=sys.stderr)
        ranges = calibrator.compute_data()
        del calibrator
    save_tensors_data(ranges, path)
    meta = {
        "calibration_id": cal_id,
        "file": path.name,
        "fp32_artifact_id": fp32_id,
        "manifest": Path(spec.manifest).stem,
        "limit": spec.limit,
        "n_images": len(entries),
        "method": spec.method,
        "options": _calib_options(qcfg),
        "op_types": list(qcfg.op_types),
        "n_tensors": len(ranges.data),
        "versions": {"onnxruntime": ort.__version__},
        # Dev-machine figures, for sizing chunk_images only.
        "seconds": round(time.perf_counter() - t0, 1),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta
