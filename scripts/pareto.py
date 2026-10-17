"""Latency vs accuracy per model, from reports (C5 task 4).

Every point is one TensorRT engine:
  y  mIoU change vs the FP32 engine, in points, on the TensorRT sensitivity report's
     scoring images (sens_val100), so all points share one image set and one reference;
  x  trtexec GPU compute p50 at the locked clock (D-027), matched by engine artifact id.
Points come from the TensorRT sensitivity report (our placement, INT8+FP16) and from eval
reports on the same images given with --point (FP32/FP16 baselines, C3 INT8+FP32, the
placement pair: ours and ModelOpt's, both min/max). Points without a trtexec report are
listed as unmeasured, not drawn.

    uv run --group plots python scripts/pareto.py --model segformer_b0 \\
        --sensitivity artifacts/reports/sensitivity__segformer_b0__trt__int8_fp16__....json \\
        --point fp32:baseline:<eval report> --point fp16:baseline:<eval report> \\
        --point "INT8+FP32 (C3)":int8_fp32:<eval report> \\
        --point "ModelOpt placement":modelopt:<eval report>

Writes docs/figures/pareto__<model>.png and a `pareto` report (points, front, sources).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgeinfer.analyze.pareto import pareto_front, sensitivity_points
from edgeinfer.config import load_budgets
from edgeinfer.reports import REPO_ROOT, REPORTS_DIR, git_state, make_report, write_report

FIGURES = REPO_ROOT / "docs" / "figures"
# Reference palette (dataviz skill, light mode): three categorical slots at most for a
# scatter, so every INT8 kind gets one; baselines are neutral ink with direct labels. Our
# min/max placement shares our placement's hue, hollow, so the pair reads as one family.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#a3a29c"
BLUE = "#2a78d6"
KINDS = {  # kind -> (legend label, color, filled)
    "int8_fp16": ("INT8+FP16, our placement", BLUE, True),
    "minmax_ours": ("INT8+FP16, our placement, min/max", BLUE, False),
    "modelopt": ("INT8+FP16, ModelOpt placement, min/max", "#eb6834", True),
    "int8_fp32": ("INT8+FP32 (C3 model as is)", "#1baf7a", True),
    "baseline": ("FP32 / FP16", INK_2, True),
}


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def collect(args: argparse.Namespace) -> tuple[list[dict], list[dict], dict]:
    sens = _load(args.sensitivity)
    fp32_miou = sens["metrics"]["fp32_miou"]
    images = sens["inputs"]["manifest"]
    points = sensitivity_points(sens)
    for p in points:
        p["sources"] = [Path(args.sensitivity).name]
    for spec in args.point:
        label, kind, path = spec.split(":", 2)
        ev = _load(path)
        if ev["inputs"]["manifest"] != images or ev["model"] != args.model:
            raise SystemExit(f"{path}: {ev['model']} on {ev['inputs']['manifest']}, "
                             f"expected {args.model} on {images}")  # fmt: skip
        points.append({"label": label, "kind": kind, "engine_id": ev["artifact_id"],
                       "delta_points": (ev["metrics"]["miou"] - fp32_miou) * 100,
                       "sources": [Path(path).name]})  # fmt: skip
    # The newest passing report per engine at the lock: C4 and C5 both timed the FP16 engine,
    # and every point should come from the same session where there is a choice.
    newest: dict[str, tuple[str, float, str]] = {}
    for path in REPORTS_DIR.glob(f"trtexec__{args.model}__*.json"):
        r = _load(str(path))
        if r["inputs"].get("gpu_clock_mhz") != args.clock_mhz or r.get("git_dirty"):
            continue
        if r["passed"] and r["created_at"] > newest.get(r["artifact_id"], ("",))[0]:
            p50 = r["metrics"]["spans_ms"]["compute"]["p50"]
            newest[r["artifact_id"]] = (r["created_at"], p50, path.name)
    timings = {aid: (p50, name) for aid, (_, p50, name) in newest.items()}
    drawn, unmeasured = [], []
    for p in points:
        if p["engine_id"] in timings:
            p["latency_ms"], src = timings[p["engine_id"]]
            p["sources"].append(src)
            drawn.append(p)
        else:
            unmeasured.append(p)
    return drawn, unmeasured, {"fp32_miou": fp32_miou, "images": images}


def plot(model: str, points: list[dict], front: list[dict], target: float, budget: float,
         out: Path) -> None:  # fmt: skip
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color="#e6e5e0", linewidth=0.6)
    ax.set_axisbelow(True)
    for y, name in ((-budget, f"budget -{budget:g} (full val)"),
                    (-target, f"subset target -{target:g}")):  # fmt: skip
        ax.axhline(y, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.annotate(name, (1.0, y), xycoords=("axes fraction", "data"), xytext=(-2, 3),
                    textcoords="offset points", ha="right", fontsize=7, color=INK_2)  # fmt: skip
    # The view covers the region where the configurations differ. FP32 (far slower) and
    # points far below the budget would squeeze it into a corner, so they are drawn as edge
    # markers carrying their real values; the report keeps every point as measured.
    lowest = min(p["delta_points"] for p in points)
    floor = min(max(lowest - 0.15, -(budget + 0.6)), -budget - 0.15)  # budget line in view
    others = [p["latency_ms"] for p in points if p["label"] != "FP32"]
    xmax = max(others) * 1.08
    xmin = min(others) * 0.95
    view = [p for p in points if p["latency_ms"] <= xmax and p["delta_points"] >= floor]
    ymax = max(0.0, *(p["delta_points"] for p in view)) + 0.08
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(floor, ymax)
    fx = [p["latency_ms"] for p in front if p in view]
    fy = [p["delta_points"] for p in front if p in view]
    ax.plot(fx, fy, color=MUTED, linewidth=1.5, zorder=1, label="Pareto front")
    stacked: dict[str, list[int]] = {}
    for kind, (name, color, filled) in KINDS.items():
        pts = [p for p in points if p["kind"] == kind]
        if not pts:
            continue
        style = {"color": color if filled else SURFACE, "edgecolors": SURFACE if filled else color,
                 "linewidths": 1.5, "zorder": 3}  # fmt: skip
        inside = [p for p in pts if p in view]
        ax.scatter([p["latency_ms"] for p in inside], [p["delta_points"] for p in inside],
                   s=36, label=name, **style)  # fmt: skip
        for p in pts:
            if p in view:
                continue
            right = p["latency_ms"] > xmax
            xy = (xmax if right else p["latency_ms"], max(p["delta_points"], floor))
            ax.scatter(*xy, s=44, marker=">" if right else "v", clip_on=False, **style)
            # Edge labels stack upwards, one line per off-scale point on the same edge.
            n = stacked.setdefault("right" if right else "bottom", [0])
            ax.annotate(f"{p['label']}: {p['latency_ms']:.2f} ms, {p['delta_points']:+.2f}",
                        xy, xytext=(-6, -14 - 10 * n[0]) if right else (6, 4 + 10 * n[0]),
                        textcoords="offset points", ha="right" if right else "left",
                        fontsize=7, color=INK)  # fmt: skip
            n[0] += 1
    for p in view:  # label the front and the baselines only
        if p in front or p["kind"] == "baseline":
            ax.annotate(p["label"], (p["latency_ms"], p["delta_points"]), xytext=(5, 4),
                        textcoords="offset points", fontsize=7, color=INK)  # fmt: skip
    ax.set_xlabel("trtexec GPU compute p50, ms (T4, 990 MHz)", fontsize=8, color=INK_2)
    ax.set_ylabel("mIoU change vs FP32, points (sens_val100)", fontsize=8, color=INK_2)
    ax.set_title(f"{model}: latency vs accuracy, TensorRT 11", fontsize=10, color=INK,
                 loc="left")  # fmt: skip
    ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="center right")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)


def plot_report(report: dict, out: Path) -> None:
    """Redraw a figure from its `pareto` report alone (scripts/make_results.py --plots)."""
    points = report["metrics"]["points"]
    by_label = {p["label"]: p for p in points}
    front = [by_label[label] for label in report["metrics"]["front"]]
    t = report["thresholds"]
    plot(report["model"], points, front, t["target_drop_points"], t["miou_drop_budget_points"], out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--sensitivity", required=True, help="TensorRT sensitivity report")
    ap.add_argument("--point", action="append", default=[], help="LABEL:KIND:EVAL_REPORT")
    ap.add_argument("--clock-mhz", type=int, default=990, help="trtexec reports at this lock")
    ap.add_argument("--out", help="report path (default: artifacts/reports/pareto__...)")
    args = ap.parse_args()
    points, unmeasured, ref = collect(args)
    front = pareto_front(points)
    sens = _load(args.sensitivity)
    budget = float(load_budgets()["accuracy"]["miou_drop_max_points"])
    target = sens["thresholds"]["target_drop_points"]
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")
    fig = FIGURES / f"pareto__{args.model}.png"
    plot(args.model, points, front, target, budget, fig)
    report = make_report(
        tool="pareto", model=args.model,
        backend_meta={"backend": "trt", "precision": "mixed", "device": "report data",
                      "artifact_id": "none", "versions": {}},
        inputs={"sensitivity": Path(args.sensitivity).name, "points": args.point,
                "clock_mhz": args.clock_mhz, **ref},
        thresholds={"miou_drop_budget_points": budget, "target_drop_points": target},
        metrics={"points": points, "front": [p["label"] for p in front],
                 "unmeasured": [p["label"] for p in unmeasured], "figure": fig.name},
        findings=[], passed=True,
    )  # fmt: skip
    out = Path(args.out) if args.out else REPORTS_DIR / f"pareto__{args.model}__{sha[:7]}.json"
    write_report(report, out)
    print(f"{fig} ({len(points)} points, front {[p['label'] for p in front]}, "
          f"unmeasured {[p['label'] for p in unmeasured]}) -> {out}")  # fmt: skip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
