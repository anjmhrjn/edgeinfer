"""Evaluate a backend on a manifest: preprocess -> infer -> postprocess -> confusion matrix."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Iterable
from typing import Any

from edgeinfer.backends.base import Backend
from edgeinfer.config import ModelConfig
from edgeinfer.data.cityscapes import Sample
from edgeinfer.data.labels import CLASS_NAMES
from edgeinfer.eval.miou import ConfusionMatrix
from edgeinfer.postprocess import postprocess
from edgeinfer.preprocess import preprocess


def evaluate(
    cfg: ModelConfig, backend: Backend, samples: Iterable[Sample], total: int | None = None
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Returns (metrics, findings). Predictions are scored at full label resolution (D-004)."""
    cm = ConfusionMatrix(cfg.num_classes)
    n = 0
    infer_seconds = 0.0
    start = time.perf_counter()
    for sample in samples:
        x = preprocess(sample.image, cfg)
        t0 = time.perf_counter()
        logits = backend.infer(x)
        infer_seconds += time.perf_counter() - t0
        pred = postprocess(logits, *sample.label.shape)
        cm.update(pred, sample.label)
        n += 1
        if n % 50 == 0 or n == total:
            print(f"  {n}/{total or '?'} images, running mIoU {cm.miou():.4f}", file=sys.stderr)
    if n == 0:
        raise ValueError("no samples evaluated")
    wall = time.perf_counter() - start

    iou = cm.iou()
    findings = [
        {
            "severity": "warn",
            "code": "EVAL_CLASS_ABSENT",
            "message": f"class '{name}' absent from labels and predictions; excluded from mIoU",
            "where": name,
        }
        for name, v in zip(CLASS_NAMES, iou, strict=True)
        if math.isnan(v)
    ]
    metrics = {
        "miou": cm.miou(),
        "pixel_accuracy": cm.pixel_accuracy(),
        "per_class_iou": {
            name: (None if math.isnan(v) else float(v))
            for name, v in zip(CLASS_NAMES, iou, strict=True)
        },
        "n_images": n,
        # Dev-machine timings only; latency budgets are measured on the T4 (C4).
        "dev_timing_s": {"infer_mean": infer_seconds / n, "wall_total": wall},
    }
    return metrics, findings
