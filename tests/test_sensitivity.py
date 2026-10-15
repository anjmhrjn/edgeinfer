"""Ranking and greedy-selection logic of the sensitivity analyzer, on a stub scorer."""

import pytest

from edgeinfer.analyze.sensitivity import Score, run_sensitivity

GROUPS = ["a", "b", "c", "d"]
COST = {"a": 0.1, "b": 1.5, "c": 0.02, "d": 0.6}  # mIoU points lost when the group is INT8


def stub(cost=COST, interaction=0.0):
    """Additive model: the drop is the sum of the costs of the quantized groups, plus
    `interaction` when both b and d are quantized."""
    calls = []

    def score(excluded):
        calls.append(tuple(excluded))
        quantized = [g for g in cost if g not in excluded]
        drop = sum(cost[g] for g in quantized)
        if {"b", "d"} <= set(quantized):
            drop += interaction
        return Score(0.7 - drop / 100, -drop, 1.0 - drop / 100, f"id-{len(calls)}")

    return score, calls


def _codes(findings):
    return {f["code"] for f in findings}


def test_ranking_and_greedy_suggestion():
    score, _ = stub()
    full = score(())
    metrics, findings = run_sensitivity(
        GROUPS, score, full=full, noise_floor=0.05, budget=1.0, margin=0.2
    )
    assert metrics["ranking"] == ["b", "d", "a", "c"]
    rows = {r["group"]: r for r in metrics["groups"]}
    assert rows["b"]["recovery"] == pytest.approx(1.5)
    assert rows["b"]["only_delta"] == pytest.approx(-1.5)
    assert not rows["c"]["distinguishable"]  # 0.02 < noise floor 0.05
    # full drop 2.22 > target 0.8; excluding b leaves 0.72 <= 0.8
    assert metrics["suggested_exclusions"] == ["b"]
    assert metrics["suggested_delta_points"] == pytest.approx(-0.72)
    assert _codes(findings) == {"SENS_EXCLUSION_SUGGESTED"}


def test_margin_forces_more_exclusions():
    score, _ = stub()
    metrics, _ = run_sensitivity(GROUPS, score, full=score(()), noise_floor=0.0,
                                 budget=1.0, margin=0.5)  # fmt: skip
    assert metrics["suggested_exclusions"] == ["b", "d"]  # 0.72 > 0.5, then 0.12


def test_full_int8_within_budget_needs_no_exclusions():
    score, calls = stub({g: 0.1 for g in GROUPS})
    metrics, findings = run_sensitivity(GROUPS, score, full=score(()), noise_floor=0.0,
                                        budget=1.0, margin=0.0)  # fmt: skip
    assert metrics["suggested_exclusions"] == [] and metrics["greedy"] == []
    assert _codes(findings) == {"SENS_FULL_INT8_WITHIN_BUDGET"}
    assert len(calls) == 1 + 2 * len(GROUPS)  # full + only + except, no greedy runs


def test_non_additive_effects_are_flagged():
    score, _ = stub(interaction=1.0)  # b and d together cost an extra point
    metrics, findings = run_sensitivity(GROUPS, score, full=score(()), noise_floor=0.0,
                                        budget=1.0, margin=0.5)  # fmt: skip
    # Leave-one-out credits the interaction to both b and d, so the sum of their
    # recoveries (+0.88 predicted) overestimates excluding both (-0.12 measured).
    assert metrics["suggested_exclusions"] == ["b", "d"]
    last = metrics["greedy"][-1]
    assert last["delta"] == pytest.approx(-0.12)
    assert last["predicted_additive"] == pytest.approx(0.88)
    assert "SENS_NON_ADDITIVE" in _codes(findings)


def test_unreachable_budget_is_an_error():
    def score(excluded):  # a loss outside every group: nothing recovers it
        return Score(0.6, -3.0, 0.9, "x")

    metrics, findings = run_sensitivity(GROUPS, score, full=score(()), noise_floor=0.0,
                                        budget=1.0, margin=0.0)  # fmt: skip
    assert len(metrics["suggested_exclusions"]) == len(GROUPS)
    assert "SENS_BUDGET_UNREACHABLE" in _codes(findings)


def test_single_mode_still_ranks():
    score, _ = stub()
    metrics, _ = run_sensitivity(GROUPS, score, full=score(()), noise_floor=0.0,
                                 budget=1.0, margin=0.0, modes=("only",))  # fmt: skip
    assert metrics["ranking"] == ["b", "d", "a", "c"]
    assert "recovery" not in metrics["groups"][0]
    with pytest.raises(ValueError, match="unknown modes"):
        run_sensitivity(GROUPS, score, full=score(()), noise_floor=0.0, budget=1.0,
                        margin=0.0, modes=("loo",))  # fmt: skip
