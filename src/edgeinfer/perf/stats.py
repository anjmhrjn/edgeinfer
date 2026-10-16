"""Latency statistics (D-022).

Percentiles use the nearest-rank definition: p is the smallest observed value such that
at least p% of the samples are at or below it. Every reported percentile is a latency
that actually happened, never an interpolation between two, and p99 is never optimistic
by interpolating toward a smaller sample.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

PERCENTILES = (50, 90, 99)


def nearest_rank(sorted_values: Sequence[float], p: float) -> float:
    if not sorted_values:
        raise ValueError("no samples")
    if not 0 < p <= 100:
        raise ValueError(f"percentile must be in (0, 100], got {p}")
    rank = math.ceil(p / 100 * len(sorted_values))
    return float(sorted_values[rank - 1])


def summarize(samples_ms: Sequence[float]) -> dict[str, float | int]:
    """n, mean, std (sample), min, p50/p90/p99, max, all in milliseconds."""
    values = sorted(float(v) for v in samples_ms)
    if not values:
        raise ValueError("no samples")
    arr = np.asarray(values)
    out: dict[str, float | int] = {
        "n": len(values),
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(values) > 1 else 0.0,
        "min": values[0],
    }
    for p in PERCENTILES:
        out[f"p{p}"] = nearest_rank(values, p)
    out["max"] = values[-1]
    return out
