"""docs/results.md and the README's results table, from reports only (C7, D-041).

Reads configs/results.yaml, which names the report behind every row, and the reports it
names. Refuses a missing report, a report written from a dirty tree, and a row whose
reports measured different artifacts, so a published number can always be traced to one
clean report.

    uv run python scripts/make_results.py            # write docs/results.md and README.md
    uv run python scripts/make_results.py --check    # fail if either differs (CI)
    uv run --group plots python scripts/make_results.py --plots   # also redraw the figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from edgeinfer.config import load_budgets
from edgeinfer.reports import REPO_ROOT, REPORTS_DIR

MANIFEST = REPO_ROOT / "configs" / "results.yaml"
RESULTS = REPO_ROOT / "docs" / "results.md"
README = REPO_ROOT / "README.md"
FIGURES = REPO_ROOT / "docs" / "figures"
START, END = "<!-- results:start -->", "<!-- results:end -->"

# trtexec layer types, grouped as in the C5 profile table: the matrix layers are the only
# ops our Q/DQ placement quantizes.
LAYER_GROUPS = {
    "Matrix layers (conv, GEMM)": ("gemm", "CaskConvolution", "CaskGemmConvolution", "fusion"),
    "`kgen` (TensorRT-generated kernels)": ("kgen",),
    "Reformat": ("Reformat",),
    "Resize, pooling": ("Resize", "CaskPooling"),
    "Other": ("NoOp", "signal", "wait"),
}


class ResultsError(Exception):
    pass


class Reports:
    """Loads the named reports, checking each is clean, and remembers which were used."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._cache: dict[str, dict[str, Any]] = {}

    def __call__(self, name: str) -> dict[str, Any]:
        if name not in self._cache:
            path = self.root / name
            if not path.is_file():
                raise ResultsError(f"{name}: no such report in {self.root}")
            report = json.loads(path.read_text())
            if report.get("git_dirty") is not False:
                raise ResultsError(f"{name}: written from a dirty tree; it can't back a number")
            self._cache[name] = report
        return self._cache[name]


def _same_artifact(reports: Reports, names: list[str], where: str, allowed: str = "") -> None:
    """The reports must have measured the same artifact (id and bytes), unless the manifest
    says why not (`different_artifacts: <reason>`)."""
    if allowed:
        return
    ids = {reports(n)["artifact_id"] for n in names}
    shas = {reports(n)["inputs"]["artifact_sha256"] for n in names
            if "artifact_sha256" in reports(n)["inputs"]}  # fmt: skip
    if len(ids) > 1:
        raise ResultsError(f"{where}: the reports measured different artifacts {sorted(ids)}")
    if len(shas) > 1:
        raise ResultsError(f"{where}: same artifact id but different bytes (sha256"
                           f" {', '.join(sorted(h[:12] for h in shas))}): a rebuild")  # fmt: skip


def _expect(report: dict[str, Any], name: str, **expected: Any) -> None:
    for key, want in expected.items():
        got = report["inputs"].get(key, report["metrics"].get(key))
        if got != want:
            raise ResultsError(f"{name}: {key} is {got!r}, expected {want!r}")


def _link(name: str, base: str) -> str:
    return f"[{name}]({base}artifacts/reports/{name})"


def _pts(x: float) -> str:
    """A change in points, with a true minus sign; a change that rounds to zero has none."""
    return "0.000" if round(x, 3) == 0 else f"{x:+.3f}".replace("-", "−")


def _notes(rows: list[dict[str, Any]]) -> list[str]:
    notes = [f"- {r['label']}: {r['note']}" for r in rows if r.get("note")]
    notes += [f"- {r['label']}: {r['different_artifacts']}" for r in rows
              if r.get("different_artifacts")]  # fmt: skip
    return ["", "Notes:", *notes] if notes else []


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return out + ["| " + " | ".join(r) + " |" for r in rows]


def main_rows(m: dict, reports: Reports) -> dict[str, list[dict[str, Any]]]:
    """The main table's values, per model; the first row of each model is FP32."""
    budgets = load_budgets()
    held_out = m["held_out"]
    out: dict[str, list[dict[str, Any]]] = {}
    for model, rows in m["main"].items():
        ref = None
        out[model] = []
        for row in rows:
            names = [row[k] for k in ("eval", "perf", "parity") if k in row]
            _same_artifact(reports, names, f"{model} {row['label']}",
                           row.get("different_artifacts", ""))  # fmt: skip
            ev, perf = reports(row["eval"]), reports(row["perf"])
            _expect(ev, row["eval"], manifest="eval_val", input_hw=[512, 1024], n_images=500)
            if perf["inputs"]["perf_config"]["gpu_clock_mhz"] != 990:
                raise ResultsError(f"{row['perf']}: not measured at the 990 MHz lock (D-027)")
            spans = perf["metrics"]["spans_ms"]
            v = {
                "label": row["label"],
                "final": bool(row.get("final")),
                "miou": ev["metrics"]["miou"] * 100,
                "held_out": ev["metrics"]["subsets"][held_out]["miou"] * 100,
                "p50": spans["infer"]["p50"],
                "p99": spans["infer"]["p99"],
                "e2e_p99": spans["e2e"]["p99"],
                "memory_mib": perf["metrics"]["device_used_mib_max"],
                "engine_mib": perf["metrics"]["artifact_size_bytes"] / 2**20,
                "sources": names,
                "note": row.get("note"),
                "different_artifacts": row.get("different_artifacts"),
            }
            if ref is None:
                ref, v["delta"], v["delta_held_out"], v["gates"] = v, None, None, "reference"
            else:
                v["delta"] = v["miou"] - ref["miou"]
                v["delta_held_out"] = v["held_out"] - ref["held_out"]
                failed = [g for g, ok in (
                    ("accuracy", -v["delta"] <= budgets["accuracy"]["miou_drop_max_points"]),
                    ("perf", perf["passed"]),
                    ("parity", reports(row["parity"])["passed"] if "parity" in row else False),
                ) if not ok]  # fmt: skip
                v["gates"] = "pass" if not failed else "fail: " + ", ".join(failed)
            out[model].append(v)
    return out


def _main_line(v: dict[str, Any]) -> list[str]:
    label = f"**{v['label']} (final)**" if v["final"] else v["label"]
    delta = "—" if v["delta"] is None else _pts(v["delta"])
    held = "—" if v["delta_held_out"] is None else _pts(v["delta_held_out"])
    return [label, f"{v['miou']:.3f}", delta, held, f"{v['p50']:.3f}", f"{v['p99']:.3f}",
            f"{v['e2e_p99']:.1f}", f"{v['memory_mib']:.0f}", f"{v['engine_mib']:.1f}",
            v["gates"]]  # fmt: skip


MAIN_HEADER = ["Engine", "mIoU", "vs FP32", "held out vs FP32", "`infer` p50 (ms)",
               "`infer` p99 (ms)", "e2e p99 (ms)", "GPU memory (MiB)", "engine file (MiB)",
               "gates"]  # fmt: skip


def render_results(m: dict, reports: Reports) -> str:
    budgets = load_budgets()
    names = m["models"]
    main = main_rows(m, reports)
    env_report = reports(next(iter(m["main"].values()))[0]["perf"])
    env = f"{env_report['env']['device']}, TensorRT {env_report['env']['versions']['tensorrt']}"
    base = "../"
    lines = [
        "# Results",
        "",
        "<!-- Generated by scripts/make_results.py from configs/results.yaml. Don't edit by"
        " hand: change the manifest or the reports, then regenerate. -->",
        "",
        "Every number on this page comes from a committed JSON report in `artifacts/reports/`,"
        " linked under its table. Each report records its git SHA and environment.",
        "",
        f"Budgets (`configs/budgets.yaml`): `infer` p99 ≤ {budgets['latency']['p99_ms_max']} ms,"
        f" mIoU drop ≤ {budgets['accuracy']['miou_drop_max_points']} point vs FP32 on full val,"
        f" GPU memory ≤ {budgets['memory']['device_used_mib_max']} MiB.",
        "",
        "## 1. TensorRT on the T4",
        "",
        f"{env}, batch 1. Accuracy on Cityscapes full val (500 images) at 512×1024, in mIoU"
        f" points; *held out* is `{m['held_out']}`, the 400 images not used to choose the mixed"
        " configurations (D-017). Latency is the probe at the 990 MHz clock lock (D-027):"
        " 20 warmup + 3 × 400 frames. The budget gates the engine's `infer` span (D-026); e2e"
        " adds the Python pre- and post-processing at full label resolution (D-028), which"
        " isn't part of the engine.",
    ]
    for model, rows in main.items():
        lines += [
            "",
            f"### {names[model]}",
            "",
            *_table(MAIN_HEADER, [_main_line(v) for v in rows]),
        ]
        lines += _notes(rows)
        lines += ["", "Sources:"] + [
            f"- {v['label']}: " + " · ".join(_link(n, base) for n in v["sources"]) for v in rows
        ]

    lines += ["", "## 2. Accuracy across harnesses", "",
              "The same full val through each step of the toolchain. The first change is the"
              " input resolution, 1024×2048 → 512×1024, the deployment size; each later step"
              " changes only the runtime."]  # fmt: skip
    for model, rows in m["accuracy_chain"].items():
        table, prev = [], None
        for row in rows:
            ev = reports(row["eval"])
            miou = ev["metrics"]["miou"] * 100
            table.append([row["label"], f"{miou:.3f}", "—" if prev is None else _pts(miou - prev)])
            prev = miou
        lines += ["", f"### {names[model]}", "", *_table(["Harness", "mIoU", "vs previous"], table)]
        lines += ["", "Sources: " + " · ".join(_link(r["eval"], base) for r in rows)]

    lines += ["", "## 3. INT8 on ONNX Runtime CPU", "",
              "C3's post-training quantization, explicit Q/DQ, calibrated on 200 train images"
              " (`calib_train200`). Full val at 512×1024, vs the ONNX Runtime FP32"
              " model."]  # fmt: skip
    for model, rows in m["ort_int8"].items():
        ref = reports(rows[0]["eval"])["metrics"]["miou"] * 100
        table = []
        for i, row in enumerate(rows):
            miou = reports(row["eval"])["metrics"]["miou"] * 100
            table.append([row["label"], f"{miou:.3f}", "—" if i == 0 else _pts(miou - ref)])
        lines += ["", f"### {names[model]}", "", *_table(["Model", "mIoU", "vs FP32"], table)]
        lines += _notes(rows)
        lines += ["", "Sources: " + " · ".join(_link(r["eval"], base) for r in rows)]

    lines += ["", "## 4. Latency vs accuracy, and the final engines", "",
              "Each point is one TensorRT engine: mIoU change on `sens_val100` against trtexec"
              " GPU compute p50 at 990 MHz. The finalists then got the full probe and full val,"
              " and the rule fixed before any INT8 result (D-033) chose the final engine: the"
              " lowest `infer` p99 among engines passing every gate, with differences under"
              " the tie band counted as ties, and ties going to fewer quantized"
              " nodes."]  # fmt: skip
    for model, entry in m["pareto"].items():
        final = reports(entry["final"])
        reports(entry["pareto"])
        why = final["metrics"]["why"]
        label = {**{r["label"]: r["label"] for r in final["metrics"]["rows"]}, **entry["labels"]}
        table = [[f"**{label[r['label']]}**" if r["label"] == final["metrics"]["final"]
                  else label[r["label"]],
                  f"{r['p99_ms']:.3f}", _pts(-r["drop_full"]), _pts(-r["drop_heldout"]),
                  str(r["quantized_nodes"]),
                  "pass" if r["perf_passed"] and r["parity_passed"] else "fail"]
                 for r in final["metrics"]["rows"]]  # fmt: skip
        lines += [
            "",
            f"### {names[model]}",
            "",
            f"![{names[model]}: latency vs accuracy](figures/pareto__{model}.png)",
            "",
            *_table(
                [
                    "Candidate",
                    "`infer` p99 (ms)",
                    "full val vs FP32",
                    "held out vs FP32",
                    "quantized nodes",
                    "perf + parity",
                ],
                table,
            ),
            "",
            f"Final: **{label[final['metrics']['final']]}** ({why['reason']}; tie band"
            f" {why['tie_rel'] * 100:.1f}%, up to {why['tie_limit_ms']:.3f} ms).",
            "",
            f"Sources: {_link(entry['pareto'], base)} · {_link(entry['final'], base)}",
        ]

    lines += ["", "## 5. Where the time goes: FP16 vs INT8", "",
              "Sum of per-layer medians by layer type (trtexec `--dumpProfile`, 990 MHz), in ms."
              " SegFormer runs layers on several streams in parallel, so its sums exceed the"
              " wall-clock compute time. INT8 only speeds up the matrix layers; the rest sets"
              " the ceiling (Amdahl's law)."]  # fmt: skip
    for model, rows in m["profile"].items():
        profs = [reports(r["trtexec"]) for r in rows]
        types = {t for p in profs for t in p["metrics"]["profile"]["by_type"]}
        unknown = types - {t for ts in LAYER_GROUPS.values() for t in ts}
        if unknown:
            raise ResultsError(f"{model}: layer types {sorted(unknown)} not in LAYER_GROUPS")
        table = []
        for group, members in LAYER_GROUPS.items():
            ms = [sum(p["metrics"]["profile"]["by_type"].get(t, {"ms": 0.0})["ms"] for t in members)
                  for p in profs]  # fmt: skip
            if any(ms):
                table.append([group, *(f"{x:.3f}" for x in ms)])
        totals = [p["metrics"]["profile"]["total_median_ms"] for p in profs]
        compute = [p["metrics"]["spans_ms"]["compute"]["p50"] for p in profs]
        table.append(["Sum of layers", *(f"{x:.3f}" for x in totals)])
        table.append(["trtexec compute p50", *(f"{x:.3f}" for x in compute)])
        lines += ["", f"### {names[model]}", "",
                  *_table(["", *(r["label"] for r in rows)], table),
                  "", f"Speed-up of compute p50: {compute[0] / compute[-1]:.2f}×.",
                  "", "Sources: " + " · ".join(_link(r["trtexec"], base)
                                               for r in rows)]  # fmt: skip

    lines += ["", "## 6. Gated deployment", "",
              "`edgeinfer deploy run --recipe configs/deploy/<name>.yaml` (D-037 to D-040)."
              " *Seconds* is the sum of the stage times in the deploy report. A rerun skips"
              " every producing stage whose content-addressed artifact exists and reuses every"
              " passing check whose inputs and code are unchanged."]  # fmt: skip
    table = []
    for row in m["deploy"]:
        d = reports(row["deploy"])
        stages = d["metrics"]["stages"]
        ran = [s for s, e in stages.items() if e["status"] == "ran" and s != "publish"]
        if d["passed"]:
            p = d["metrics"]["published"]
            result = f"v{p['version']} → `{p['alias']}`" + (
                "" if p["new_version"] else " (already)"
            )
        else:
            stop = next(f for f in d["findings"] if f["code"].startswith("DEPLOY_"))
            codes = list(dict.fromkeys(c.split(": ")[-1] for c in stop["message"].split("; ")))
            result = f"blocked at {stop['where']}: {', '.join(codes)}"
        secs = sum(e.get("seconds") or 0 for e in stages.values())
        table.append([row["label"], result, ", ".join(ran) or "—", f"{secs:.1f}"])
    lines += ["", *_table(["Run", "Result", "Stages that ran", "Seconds"], table)]
    if m.get("rollback"):
        rb = reports(m["rollback"])
        lines += ["", f"Rollback: {rb['findings'][0]['message']}, after checking the target"
                  f" engine's sha256 on the instance ({_link(m['rollback'], base)})."]  # fmt: skip
    lines += ["", "Sources: " + " · ".join(_link(r["deploy"], base) for r in m["deploy"])]

    lines += ["", "### The same engine measured twice", "",
              "C5's selection run and C6's perf gate, on the same engine bytes.", ""]  # fmt: skip
    table = []
    for row in m["rerun"]:
        _same_artifact(reports, [row["first"], row["second"]], row["label"])
        a, b = (reports(row[k])["metrics"]["spans_ms"]["infer"]["p99"] for k in ("first", "second"))
        change = f"{(b - a) / a * 100:+.1f}%".replace("-", "−")
        table.append([row["label"], f"{a:.3f}", f"{b:.3f}", change])
    lines += [*_table(["Engine", "C5 `infer` p99 (ms)", "C6 `infer` p99 (ms)", "change"], table),
              "", "Sources: " + " · ".join(f"{_link(r['first'], base)} · {_link(r['second'], base)}"
                                           for r in m["rerun"])]  # fmt: skip
    return "\n".join(lines) + "\n"


def render_readme_table(m: dict, reports: Reports) -> str:
    main = main_rows(m, reports)
    names = m["models"]
    rows = []
    for pick in m["readme"]:
        (v,) = [v for v in main[pick["model"]] if v["label"] == pick["label"]]
        label = f"**{v['label']} (final)**" if v["final"] else v["label"]
        rows.append([names[pick["model"]], label, f"{v['miou']:.3f}",
                     "—" if v["delta"] is None else _pts(v["delta"]), f"{v['p99']:.3f}",
                     f"{v['memory_mib']:.0f}", f"{v['engine_mib']:.1f}"])  # fmt: skip
    header = ["Model", "Engine", "mIoU", "vs FP32", "`infer` p99 (ms)", "GPU memory (MiB)",
              "engine file (MiB)"]  # fmt: skip
    return "\n".join([
        START,
        "<!-- Generated by scripts/make_results.py; don't edit by hand. -->",
        *_table(header, rows),
        "",
        "TensorRT 11.3 on a T4 at the 990 MHz clock lock, batch 1; Cityscapes full val at"
        " 512×1024. Sources for every cell: [docs/results.md](docs/results.md).",
        END,
    ])  # fmt: skip


def splice(text: str, table: str) -> str:
    if text.count(START) != 1 or text.count(END) != 1:
        raise ResultsError(f"README needs exactly one {START} ... {END} block")
    head, rest = text.split(START)
    return head + table + rest.split(END, 1)[1]


def redraw(m: dict, reports: Reports, figures: Path) -> list[Path]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from pareto import plot_report

    out = []
    for model, entry in m["pareto"].items():
        path = figures / f"pareto__{model}.png"
        plot_report(reports(entry["pareto"]), path)
        out.append(path)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="fail if the outputs aren't current")
    ap.add_argument("--plots", action="store_true", help="also redraw docs/figures (plots group)")
    ap.add_argument("--manifest", type=Path, default=MANIFEST)
    ap.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--readme", type=Path, default=README)
    args = ap.parse_args(argv)
    m = yaml.safe_load(args.manifest.read_text())
    reports = Reports(args.reports_dir)
    try:
        outputs = {
            args.results: render_results(m, reports),
            args.readme: splice(args.readme.read_text(), render_readme_table(m, reports)),
        }
    except ResultsError as e:
        print(f"error: {e}")
        return 2
    stale = [p for p, text in outputs.items() if not p.is_file() or p.read_text() != text]
    if args.check:
        for p in stale:
            print(f"{p} is out of date: run scripts/make_results.py")
        return 1 if stale else 0
    for p in stale:
        p.write_text(outputs[p])
        print(f"wrote {p}")
    if args.plots:
        for p in redraw(m, reports, FIGURES):
            print(f"drew {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
