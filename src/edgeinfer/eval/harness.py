"""Evaluate a backend on a manifest: preprocess -> infer -> postprocess -> confusion matrix."""

from __future__ import annotations

import math
import sys
import time
from collections.abc import Iterable, Mapping
from typing import Any

from edgeinfer.backends.base import Backend
from edgeinfer.config import ModelConfig
from edgeinfer.data.cityscapes import Sample
from edgeinfer.data.labels import CLASS_NAMES
from edgeinfer.eval.miou import ConfusionMatrix
from edgeinfer.postprocess import postprocess
from edgeinfer.preprocess import preprocess


def _scores(cm: ConfusionMatrix, n: int) -> dict[str, Any]:
    if n == 0:  # a subset none of whose images were evaluated (--limit)
        return {"miou": None, "pixel_accuracy": None, "per_class_iou": None, "n_images": 0}
    return {
        "miou": cm.miou(),
        "pixel_accuracy": cm.pixel_accuracy(),
        "per_class_iou": {
            name: (None if math.isnan(v) else float(v))
            for name, v in zip(CLASS_NAMES, cm.iou(), strict=True)
        },
        "n_images": n,
    }


def evaluate(
    cfg: ModelConfig,
    backend: Backend,
    samples: Iterable[Sample],
    total: int | None = None,
    subsets: Mapping[str, set[str]] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Returns (metrics, findings). Predictions are scored at full label resolution (D-004).

    `subsets` (name -> image ids as "split/id") also scores those images on their own, in
    the same pass: one run over the full val split gives full val and the held-out images
    (C5), each with its own confusion matrix, so no image is inferred twice."""
    cm = ConfusionMatrix(cfg.num_classes)
    sub_cms = {name: ConfusionMatrix(cfg.num_classes) for name in subsets or {}}
    sub_n = dict.fromkeys(sub_cms, 0)
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
        for name, ids in (subsets or {}).items():
            if str(sample.entry) in ids:
                sub_cms[name].update(pred, sample.label)
                sub_n[name] += 1
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
        **_scores(cm, n),
        # Dev-machine timings only; latency budgets are measured on the T4 (C4).
        "dev_timing_s": {"infer_mean": infer_seconds / n, "wall_total": wall},
    }
    if subsets:
        metrics["subsets"] = {name: _scores(sub_cms[name], sub_n[name]) for name in sub_cms}
    return metrics, findings
