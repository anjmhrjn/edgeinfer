"""Pareto front and the pre-registered final-engine rule (C5)."""

from edgeinfer.analyze.pareto import pareto_front, select_final, sensitivity_points


def _p(label, ms, delta):
    return {"label": label, "latency_ms": ms, "delta_points": delta}


def test_pareto_front():
    pts = [_p("fp32", 20.0, 0.0), _p("fp16", 6.0, 0.0), _p("int8", 4.0, -0.3),
           _p("int8_bad", 4.5, -0.5), _p("mixed", 4.6, -0.1), _p("same", 6.0, 0.0)]  # fmt: skip
    front = [p["label"] for p in pareto_front(pts)]
    # fp32 is dominated by fp16; int8_bad by int8; equal points both stay.
    assert front == ["int8", "mixed", "fp16", "same"]


def _f(label, p99, drop_full=0.2, drop_heldout=0.2, nodes=60):
    return {"label": label, "p99_ms": p99, "drop_full": drop_full, "drop_heldout": drop_heldout,
            "quantized_nodes": nodes}  # fmt: skip


def test_fastest_eligible_wins():
    choice, why = select_final([_f("fp16", 10.0, 0, 0, 0), _f("int8", 8.0)], 1.0)
    assert choice["label"] == "int8" and why["reason"] == "fastest eligible"


def test_ineligible_configs_are_skipped():
    choice, why = select_final([_f("fp16", 10.0, 0, 0, 0), _f("int8", 8.0, drop_heldout=1.2)],
                               1.0)  # fmt: skip
    assert choice["label"] == "fp16" and why["ineligible"] == ["int8"]


def test_a_tie_goes_to_fewer_quantized_nodes():
    # int8 is 3% faster than fp16: within build-to-build variation, so fp16 is chosen.
    choice, why = select_final([_f("fp16", 10.0, 0, 0, 0), _f("int8", 9.7)], 1.0)
    assert choice["label"] == "fp16" and why["ties"] == ["fp16", "int8"]
    assert why["reason"] == "tied with the fastest, fewer quantized nodes"
    # Among INT8 configurations too: the mixed one keeps more in float.
    choice, _ = select_final([_f("int8", 8.0, nodes=60), _f("mixed", 8.2, nodes=57)], 1.0)
    assert choice["label"] == "mixed"


def test_nothing_eligible():
    choice, why = select_final([_f("int8", 8.0, drop_full=1.5)], 1.0)
    assert choice is None and "no finalist" in why["reason"]


def test_sensitivity_points():
    report = {"metrics": {
        "full_int8": {"delta_points": -0.5},
        "groups": [{"group": "a", "except_delta": -0.1}, {"group": "b", "except_delta": -0.45}],
        "greedy": [{"excluded": ["a"], "delta": -0.1}, {"excluded": ["a", "b"], "delta": -0.05}],
        "engines": [{"excluded": [], "artifact_id": "e0"}, {"excluded": ["a"], "artifact_id": "ea"},
                    {"excluded": ["b"], "artifact_id": "eb"},
                    {"excluded": ["a", "b"], "artifact_id": "eab"}],
    }}  # fmt: skip
    pts = {p["label"]: p for p in sensitivity_points(report)}
    assert set(pts) == {"full INT8", "a in FP16", "b in FP16", "2 groups in FP16"}
    assert pts["2 groups in FP16"]["engine_id"] == "eab"
    assert pts["a in FP16"]["delta_points"] == -0.1  # greedy step 1 is the same engine
