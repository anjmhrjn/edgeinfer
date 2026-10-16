"""trtexec timing summary (C4 task 5)."""

import pytest

from edgeinfer.perf.trtexec import compare_compute, summarize_times


def _records(n):
    return [
        {"computeMs": 1.0 + i / 100, "latencyMs": 1.5 + i / 100, "enqueueMs": 0.1} for i in range(n)
    ]


def test_summarize_uses_nearest_rank_per_span():
    spans = summarize_times(_records(100))
    assert set(spans) == {"compute", "latency", "enqueue"}
    assert spans["compute"]["n"] == 100
    assert spans["compute"]["p50"] == pytest.approx(1.49)  # 50th of 1.00..1.99
    assert spans["compute"]["p99"] == pytest.approx(1.98)
    assert spans["compute"]["max"] == pytest.approx(1.99)


def test_missing_compute_or_empty_export_is_an_error():
    with pytest.raises(ValueError, match="no timing records"):
        summarize_times([])
    with pytest.raises(ValueError, match="no computeMs"):
        summarize_times([{"latencyMs": 1.0}])


def test_compare_compute_is_probe_over_trtexec():
    t = {"compute": {"p50": 2.0, "p99": 4.0}}
    p = {"compute": {"p50": 3.0, "p99": 4.0}}
    assert compare_compute(t, p) == {"probe_over_trtexec_p50": 1.5, "probe_over_trtexec_p99": 1.0}
