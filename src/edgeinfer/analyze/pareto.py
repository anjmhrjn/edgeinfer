"""Latency-accuracy trade-off of the configurations tried in C5, and the final-engine rule.

Pure functions over plain dicts, so the plot script (scripts/pareto.py) and the tests
share them. A point is one engine: `latency_ms` (lower is better) and `delta_points`
(mIoU change vs FP32 on the same images and backend; higher is better, 0 for FP32).

The final-engine rule was fixed before any TensorRT INT8 result existed (C5 plan,
2026-10-16), so the choice can't be fitted to the outcome:
  1. Eligible: within the accuracy budget on full val AND on the held-out images.
  2. Fastest by the probe's `infer` p99, the budgeted span (D-026).
  3. Latencies within `tie_rel` of the fastest count as ties: build-to-build variation of
     one engine is ~3.4% (D-029), so smaller gaps aren't evidence of a faster model.
  4. A tie goes to the configuration with the fewest quantized nodes, i.e. closest to FP16
     (less quantization error for the same speed). FP16 has none, so if no INT8 engine is
     faster than FP16 beyond the tie margin, FP16 is the final engine.
"""

from __future__ import annotations

from typing import Any

TIE_REL = 0.035
N_FINALISTS = 3  # INT8 engines per model that get the full probe and full-val eval (C5 plan)

Point = dict[str, Any]


def pareto_front(points: list[Point]) -> list[Point]:
    """Points not dominated by any other (no other point is at least as fast and at least
    as accurate, and strictly better in one), sorted by latency."""
    front = []
    for p in points:
        dominated = any(
            q["latency_ms"] <= p["latency_ms"]
            and q["delta_points"] >= p["delta_points"]
            and (q["latency_ms"] < p["latency_ms"] or q["delta_points"] > p["delta_points"])
            for q in points
        )
        if not dominated:
            front.append(p)
    return sorted(front, key=lambda p: (p["latency_ms"], -p["delta_points"]))


def select_finalists(
    points: list[Point], target_points: float, n: int = N_FINALISTS
) -> tuple[list[Point], dict[str, Any]]:
    """The INT8 engines worth the expensive checks (D-035; probe ~15 min, full val ~7 min).
    Each point has `label`, `latency_ms` (trtexec compute p50 at the clock lock) and
    `delta_points` on the sensitivity images. Kept: within the subset target (the same
    margin greedy selection uses, D-017), on the Pareto front of those, the `n` fastest.
    A point off the front is slower and less accurate than another candidate, so it can't
    win the final rule unless trtexec misranks it, which C4 measured at 0.7-2.3%.
    Returns (finalists sorted by latency, the reasoning)."""
    within = [p for p in points if p["delta_points"] >= -target_points]
    front = pareto_front(within)
    finalists = front[:n]
    why = {
        "target_drop_points": target_points,
        "n": n,
        "outside_target": sorted(p["label"] for p in points if p not in within),
        "off_front": sorted(p["label"] for p in within if p not in front),
        "front_not_taken": [p["label"] for p in front[n:]],
    }
    return finalists, why


def select_final(
    finalists: list[Point], budget_points: float, tie_rel: float = TIE_REL
) -> tuple[Point | None, dict[str, Any]]:
    """Apply the final-engine rule. Each finalist has `label`, `p99_ms`, `drop_full`,
    `drop_heldout` (points lost vs FP32, >= 0 is a loss) and `quantized_nodes`.
    Returns (choice or None, the reasoning, step by step)."""

    def within(f: Point) -> bool:
        return f["drop_full"] <= budget_points and f["drop_heldout"] <= budget_points

    eligible = [f for f in finalists if within(f)]
    why: dict[str, Any] = {
        "budget_points": budget_points,
        "tie_rel": tie_rel,
        "ineligible": sorted(f["label"] for f in finalists if f not in eligible),
    }
    if not eligible:
        return None, why | {"reason": "no finalist is within the accuracy budget"}
    fastest = min(eligible, key=lambda f: f["p99_ms"])
    limit = fastest["p99_ms"] * (1 + tie_rel)
    ties = [f for f in eligible if f["p99_ms"] <= limit]
    choice = min(ties, key=lambda f: (f["quantized_nodes"], f["p99_ms"], f["label"]))
    why |= {
        "fastest": fastest["label"],
        "fastest_p99_ms": fastest["p99_ms"],
        "tie_limit_ms": limit,
        "ties": sorted(f["label"] for f in ties),
        "reason": (
            "fastest eligible"
            if choice is fastest
            else "tied with the fastest, fewer quantized nodes"
        ),
    }
    return choice, why


def sensitivity_points(report: dict[str, Any]) -> list[Point]:
    """One point per engine a TensorRT sensitivity report scored (latency still to add):
    full INT8, each leave-one-out configuration and each greedy step. `delta_points` is on
    the report's scoring images, against the FP32 engine on the same images."""
    m = report["metrics"]
    engines = {tuple(sorted(e["excluded"])): e["artifact_id"] for e in m["engines"]}
    deltas: dict[tuple[str, ...], tuple[str, float]] = {
        (): ("full INT8", m["full_int8"]["delta_points"])
    }
    for row in m["groups"]:
        deltas[(row["group"],)] = (f"{row['group']} in FP16", row["except_delta"])
    for step in m["greedy"]:
        key = tuple(sorted(step["excluded"]))
        deltas.setdefault(key, (f"{len(key)} groups in FP16", step["delta"]))
    return [{"label": label, "kind": "int8_fp16", "excluded": list(key),
             "engine_id": engines[key], "delta_points": delta}
            for key, (label, delta) in deltas.items()]  # fmt: skip
