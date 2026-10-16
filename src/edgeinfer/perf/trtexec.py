"""Summarize `trtexec --exportTimes` output (C4 task 5: cross-check the probe's compute span).

trtexec (TensorRT 11.3) times the engine with no host<->device copies, spin-wait
synchronization and CUDA graphs, all on by default. Its compute time is therefore a floor
for the probe's compute span, which launches each inference with a plain `execute_async_v3`
between two CUDA events. Each record in the export is one iteration.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from edgeinfer.perf.stats import summarize

# Export key -> span name in the report.
SPANS = {"computeMs": "compute", "latencyMs": "latency", "enqueueMs": "enqueue"}


def summarize_times(records: Sequence[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Per-span latency statistics (nearest-rank percentiles, D-022) from export records."""
    if not records:
        raise ValueError("trtexec exported no timing records")
    if "computeMs" not in records[0]:
        raise ValueError(f"no computeMs in trtexec timing records; keys: {sorted(records[0])}")
    return {
        span: summarize([float(r[key]) for r in records])
        for key, span in SPANS.items()
        if key in records[0]
    }


def compare_compute(
    trtexec_spans: dict[str, dict[str, float | int]], probe_spans: dict[str, dict[str, float | int]]
) -> dict[str, float]:
    """Probe compute over trtexec compute at p50 and p99 (>= 1 expected: trtexec is a floor)."""
    t, p = trtexec_spans["compute"], probe_spans["compute"]
    return {f"probe_over_trtexec_{q}": float(p[q]) / float(t[q]) for q in ("p50", "p99")}
