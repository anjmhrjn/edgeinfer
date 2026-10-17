"""C5 GPU phases, run inside the TensorRT container (started by scripts/c5_runs.sh):
    scripts/remote.sh run 'python scripts/c5_gpu.py <phase>'

Every model and engine path comes from a named, committed report (quantize, candidates,
build), never from "the newest file in a directory", so each step says which artifact it
used. Exit status 1 from an edgeinfer command is a failed check with its report written
(a result to read); any other failure stops the phase.

Session 1 (accuracy):
  build     engines for the INT8 configurations outside the sensitivity search: the C3 Q/DQ
            models as they are (INT8+FP32), their INT8+FP16 conversions (D-031), and the
            placement pair (ours vs ModelOpt, both min/max; D-015)
  sub       mIoU on sens_val100 for those engines and the C4 FP32/FP16 engines: one image
            set and one reference for every Pareto point
  parity    golden parity for the C3 and INT8+FP16 engines
  sens      TensorRT sensitivity over the top-5 candidates (D-032), margin 0.7 (D-017)

Session 2 (latency and confirmation), in this order; trtexec, perf, profile and nsys need
the SM clock locked on the host first (c5_runs.sh does it, D-027):
  trtexec   compute latency at the lock for every engine with a sens_val100 score: the
            Pareto x axis (D-033)
  finalists per model, the INT8 engines of our placement worth the expensive checks:
            within the subset target, on the Pareto front, the 3 fastest (pareto.py:
            select_finalists, D-035); engines that failed golden parity are left out
  perf      the probe (the latency and memory gates) for the finalists and FP16
  eval      full val for the finalists, FP16 and FP32, scoring val_minus_sens100 (held out)
            and sens_val100 in the same pass
  parity    (as above) also covers the finalists
  final     the D-033 rule over the finalists and FP16, among engines that pass golden (D-035)
            parity and the probe's gates; also the best INT8 engine by the same rule
  profile   trtexec per-layer profile of FP16 and the best INT8 engine (why INT8 < 2x)
  nsys      Nsight Systems trace of SegFormer's best INT8 engine (C4 traced its FP16 one)

Engines are named by label in report file names: the C3/D-031 labels above, `fp32`/`fp16`,
and `sens_<groups in FP16>` (`sens_full` for none) for engines the sensitivity run built.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from edgeinfer.analyze.pareto import select_final, select_finalists, sensitivity_points
from edgeinfer.config import load_budgets
from edgeinfer.reports import REPO_ROOT, REPORTS_DIR, git_state, make_report, write_report

MODELS = os.environ.get("MODELS", "segformer_b0 deeplabv3plus_mbv2").split()
SENS_MARGIN = "0.7"  # D-017: the measured subset-vs-held-out gap
CLOCK = "990MHz"  # configs/perf.yaml gpu_clock_mhz (D-027); trtexec_report.py checks the lock
HELD_OUT, SENS_IMAGES = "val_minus_sens100", "sens_val100"
# Engines of our pipeline that can become the final engine. The min/max placement pair is
# a comparison (D-015), not a deployment candidate.
POOL_KINDS = ("int8_fp16", "int8_fp32")
ENGINES_DIR = REPO_ROOT / "artifacts" / "engines"


def _sha() -> str:
    sha, dirty = git_state()
    if dirty:
        sys.exit("uncommitted changes: reports from this tree can't back published numbers")
    return sha[:7]


def report(name: str) -> dict:
    path = REPORTS_DIR / name
    if not path.is_file():
        sys.exit(f"missing report {name}: run the Mac phase that writes it, then sync")
    r = json.loads(path.read_text())
    if r.get("git_dirty"):
        sys.exit(f"{name} was written from a dirty tree; it can't be used")
    return r


def one(pattern: str) -> str:
    """The single report matching `pattern` (several would make the choice ambiguous)."""
    found = sorted(p.name for p in REPORTS_DIR.glob(pattern))
    if len(found) != 1:
        sys.exit(f"expected exactly one report matching {pattern}, found {found}")
    return found[0]


def onnx_of(quantize_report: str) -> Path:
    r = report(quantize_report)
    return REPO_ROOT / "artifacts" / "onnx" / r["model"] / r["metrics"]["file"]


def engine_of(build_report: str) -> Path:
    r = report(build_report)
    return ENGINES_DIR / r["model"] / r["metrics"]["file"]


def run(*argv: str) -> int:
    print("+ " + " ".join(argv), flush=True)
    rc = subprocess.run(list(argv), check=False).returncode
    if rc == 1:
        print(f"GATE FAILED (report written): {' '.join(argv[:2])}", flush=True)
    elif rc != 0:
        sys.exit(rc)
    return rc


def edgeinfer(*argv: str) -> int:
    return run("edgeinfer", *argv)


def int8_models(m: str) -> dict[str, str]:
    """label -> quantize report of the INT8 models built outside the sensitivity search.
    The placement pair is included once both of its reports exist (ModelOpt's calibration
    runs for hours on the Mac, so it can arrive a session later)."""
    models = {
        "int8_c3_none": f"quantize__{m}__int8_percentile__none__cb60f28.json",
        "int8_c3_sens": f"quantize__{m}__int8_percentile__sens__da584fa.json",
        "int8fp16_none": one(f"quantize__{m}__int8_percentile_fp16__none__*.json"),
        "int8fp16_sens": one(f"quantize__{m}__int8_percentile_fp16__sens__*.json"),
    }
    unfused = latest(f"quantize__{m}__int8_percentile_fp16__unfused__*.json")
    if unfused:  # SegFormer: the nodes TensorRT doesn't fuse kept in FP16 (D-034)
        models["int8fp16_unfused"] = unfused
    pair = {f"int8fp16_minmax_{tool}": f"quantize__{m}__int8_minmax_fp16__{tool}__*.json"
            for tool in ("ort", "modelopt")}  # fmt: skip
    if all(any(REPORTS_DIR.glob(pattern)) for pattern in pair.values()):
        models |= {label: one(pattern) for label, pattern in pair.items()}
    else:
        print(f"{m}: placement pair not complete yet, skipped", flush=True)
    return models


def latest(pattern: str) -> str | None:
    """The newest clean report matching `pattern`, by creation time; None if there is none.
    Phases run at different SHAs within a session, so reports are found by what they
    describe, not by the SHA in their name."""
    found = [(json.loads(p.read_text()), p.name) for p in REPORTS_DIR.glob(pattern)]
    found = [(r["created_at"], name) for r, name in found if not r.get("git_dirty")]
    return max(found)[1] if found else None


def built(m: str) -> dict[str, str]:
    """label -> latest build report, for the labels of int8_models(m) that are built."""
    reports = {label: latest(f"build__{m}__{label}__*.json") for label in int8_models(m)}
    return {label: rep for label, rep in reports.items() if rep}


def baselines(m: str) -> dict[str, str]:
    """label -> build report of the C4 engines (D-025)."""
    return {"fp32": f"build__{m}__fp32__d2bf208.json", "fp16": f"build__{m}__fp16__d2bf208.json"}


def build(sha: str) -> None:
    for m in MODELS:
        for label, qrep in int8_models(m).items():
            if latest(f"build__{m}__{label}__*.json"):
                continue  # built in an earlier run
            edgeinfer("build", "--model", m, "--artifact", str(onnx_of(qrep)),
                      "--out", str(REPORTS_DIR / f"build__{m}__{label}__{sha}.json"))  # fmt: skip


def engines(m: str) -> dict[str, Path]:
    out = {label: engine_of(rep) for label, rep in baselines(m).items()}
    out |= {label: engine_of(rep) for label, rep in built(m).items()}
    return out


def sub(sha: str) -> None:
    for m in MODELS:
        for label, eng in engines(m).items():
            if latest(f"eval__{m}__trt__{label}__sens_val100__512x1024__*.json"):
                continue  # evaluated in an earlier run
            out = REPORTS_DIR / f"eval__{m}__trt__{label}__sens_val100__512x1024__{sha}.json"
            edgeinfer("eval", "--model", m, "--backend", "trt", "--artifact", str(eng),
                      "--manifest", "sens_val100", "--out", str(out))  # fmt: skip


def parity(sha: str) -> None:
    for m in MODELS:
        targets = {label: eng for label, eng in engines(m).items()
                   if label.startswith(("int8_c3", "int8fp16_none", "int8fp16_sens",
                                        "int8fp16_unf"))}  # fmt: skip
        fin = latest(f"finalists__{m}__*.json")
        if fin:  # session 2: the finalists too
            targets |= {f["label"]: REPO_ROOT / f["engine"]
                        for f in report(fin)["metrics"]["finalists"]}  # fmt: skip
        for label, eng in targets.items():
            if latest(f"parity__{m}__trt__{label}__golden_val32__golden__*.json"):
                continue  # checked in an earlier run
            out = REPORTS_DIR / f"parity__{m}__trt__{label}__golden_val32__golden__{sha}.json"
            edgeinfer("parity", "--model", m, "--backend", "trt", "--artifact", str(eng),
                      "--inputs", "golden", "--out", str(out))  # fmt: skip


def sens(sha: str) -> None:
    for m in MODELS:
        cands = latest(f"candidates__{m}__int8_percentile__top5__*.json")  # newest set
        if cands is None:
            sys.exit(f"no candidates report for {m}")
        out = REPORTS_DIR / f"sensitivity__{m}__trt__int8_fp16__sens_val100__{sha}.json"
        edgeinfer("sensitivity", "--model", m, "--candidates", str(REPORTS_DIR / cands),
                  "--backend", "trt", "--ref-artifact", str(engine_of(baselines(m)["fp32"])),
                  "--margin", SENS_MARGIN, "--out", str(out))  # fmt: skip


# --- session 2 -------------------------------------------------------------------------


def engine_file(m: str, artifact_id: str) -> Path:
    """The engine with this artifact id (file names end in the id's first 12 characters)."""
    found = sorted(ENGINES_DIR.joinpath(m).glob(f"{m}__*__{artifact_id[:12]}.engine"))
    if len(found) != 1:
        sys.exit(f"expected one engine with id {artifact_id[:12]} for {m}, found {found}")
    return found[0]


def kind_of(label: str) -> str:
    if label in ("fp32", "fp16"):
        return "baseline"
    if label.startswith("int8_c3"):
        return "int8_fp32"
    if label.startswith("int8fp16_minmax"):
        return "modelopt" if label.endswith("modelopt") else "minmax_ours"
    return "int8_fp16"


def sens_label(excluded: list[str]) -> str:
    return "sens_" + ("+".join(sorted(excluded)) if excluded else "full")


def sens_report(m: str) -> dict:
    name = latest(f"sensitivity__{m}__trt__int8_fp16__sens_val100__*.json")
    if name is None:
        sys.exit(f"no TensorRT sensitivity report for {m}: run the sens phase")
    return report(name) | {"_name": name}


def points(m: str) -> list[dict]:
    """Every engine with a sens_val100 score: label, kind, engine_id, delta_points (vs the
    FP32 engine, points), one per engine: an engine reached under two labels (the same
    artifact id, e.g. SegFormer's C3 model with its empty exclusion list, or a sensitivity
    configuration equal to a named one) keeps the first, sub phase labels before `sens_`."""
    sens = sens_report(m)
    fp32 = sens["metrics"]["fp32_miou"]
    pts, seen = [], set()
    for label in baselines(m) | built(m):
        name = latest(f"eval__{m}__trt__{label}__sens_val100__512x1024__*.json")
        if name is None:
            sys.exit(f"{m} {label} has no sens_val100 eval: run the sub phase")
        ev = report(name)
        if ev["artifact_id"] in seen:
            continue
        seen.add(ev["artifact_id"])
        delta = (ev["metrics"]["miou"] - fp32) * 100
        pts.append({"label": label, "kind": kind_of(label), "engine_id": ev["artifact_id"],
                    "delta_points": delta, "source": name})  # fmt: skip
    for p in sensitivity_points(sens):
        if p["engine_id"] not in seen:
            seen.add(p["engine_id"])
            pts.append(p | {"label": sens_label(p["excluded"]), "source": sens["_name"]})
    return pts


def timings(m: str) -> dict[str, tuple[float, str]]:
    """engine artifact id -> (trtexec compute p50 at the lock, report), from session 2's
    passing trtexec reports (newest per engine)."""
    found: dict[str, tuple[str, float, str]] = {}
    for path in REPORTS_DIR.glob(f"trtexec__{m}__*__c5__{CLOCK}__*.json"):
        r = json.loads(path.read_text())
        if r.get("git_dirty") or not r["passed"]:
            continue
        p50 = r["metrics"]["spans_ms"]["compute"]["p50"]
        if r["artifact_id"] not in found or r["created_at"] > found[r["artifact_id"]][0]:
            found[r["artifact_id"]] = (r["created_at"], p50, path.name)
    return {aid: (p50, name) for aid, (_, p50, name) in found.items()}


def trtexec(sha: str) -> None:
    for m in MODELS:
        done = timings(m)
        for p in points(m):
            if p["engine_id"] in done:
                continue  # measured in an earlier run
            out = REPORTS_DIR / f"trtexec__{m}__{p['label']}__c5__{CLOCK}__{sha}.json"
            run("python", "scripts/trtexec_report.py", "--engine",
                str(engine_file(m, p["engine_id"])), "--out", str(out))  # fmt: skip


def parity_failed(m: str, label: str) -> bool:
    name = latest(f"parity__{m}__trt__{label}__golden_val32__golden__*.json")
    return name is not None and not report(name)["passed"]


def _summary_report(tool: str, m: str, inputs: dict, metrics: dict) -> dict:
    return make_report(
        tool=tool, model=m,
        backend_meta={"backend": "trt", "precision": "mixed", "device": "report data",
                      "artifact_id": "none", "versions": {}},
        inputs=inputs, thresholds={}, metrics=metrics, findings=[], passed=True,
    )  # fmt: skip


def finalists(sha: str) -> None:
    for m in MODELS:
        sens = sens_report(m)
        target = sens["thresholds"]["target_drop_points"]
        lat = timings(m)
        pool, unmeasured = [], []
        for p in points(m):
            if p["kind"] not in POOL_KINDS or parity_failed(m, p["label"]):
                continue
            if p["engine_id"] not in lat:
                unmeasured.append(p["label"])
                continue
            pool.append(p | {"latency_ms": lat[p["engine_id"]][0],
                             "engine": str(engine_file(m, p["engine_id"]).relative_to(REPO_ROOT)),
                             "latency_source": lat[p["engine_id"]][1]})  # fmt: skip
        if unmeasured:
            sys.exit(f"{m}: no trtexec report for {unmeasured}: run the trtexec phase")
        chosen, why = select_finalists(pool, target)
        metrics = {"finalists": chosen, "pool": pool, "selection": why}
        out = REPORTS_DIR / f"finalists__{m}__{sha}.json"
        write_report(_summary_report("finalists", m, {"sensitivity": sens["_name"]}, metrics), out)
        print(f"{m}: finalists {[f['label'] for f in chosen]} -> {out.name}", flush=True)


def checked(m: str) -> dict[str, Path]:
    """label -> engine for the engines that get the probe and full val: finalists + FP16."""
    fin = latest(f"finalists__{m}__*.json")
    if fin is None:
        sys.exit(f"no finalists report for {m}: run the finalists phase")
    out = {f["label"]: REPO_ROOT / f["engine"] for f in report(fin)["metrics"]["finalists"]}
    return out | {"fp16": engine_of(baselines(m)["fp16"])}


def perf(sha: str) -> None:
    for m in MODELS:
        for label, eng in checked(m).items():
            if latest(f"perf__{m}__trt__{label}__c5__*.json"):
                continue  # measured in an earlier run
            out = REPORTS_DIR / f"perf__{m}__trt__{label}__c5__{sha}.json"
            edgeinfer("perf", "--model", m, "--backend", "trt", "--artifact", str(eng),
                      "--out", str(out))  # fmt: skip


def full_eval(m: str, label: str) -> str | None:
    """Session 2's full-val report of an engine (with the held-out subset), if any."""
    return latest(f"eval__{m}__trt__{label}__eval_val__c5__512x1024__*.json")


def eval_(sha: str) -> None:
    for m in MODELS:
        targets = checked(m) | {"fp32": engine_of(baselines(m)["fp32"])}
        for label, eng in targets.items():
            if full_eval(m, label):
                continue  # evaluated in an earlier run
            out = REPORTS_DIR / f"eval__{m}__trt__{label}__eval_val__c5__512x1024__{sha}.json"
            edgeinfer("eval", "--model", m, "--backend", "trt", "--artifact", str(eng),
                      "--manifest", "eval_val", "--subset", HELD_OUT, "--subset", SENS_IMAGES,
                      "--out", str(out))  # fmt: skip


def quantized_nodes(eng: Path) -> int:
    """Quantized Conv/MatMul/Gemm nodes in the engine's ONNX model (0 for FP16)."""
    import onnx

    from edgeinfer.artifacts import load_sidecar
    from edgeinfer.backends.trt_layers import quantized_onnx_nodes

    meta = load_sidecar(eng)
    pid = meta["parent_artifact_id"]
    found = sorted((REPO_ROOT / "artifacts" / "onnx" / meta["model"]).glob(f"*__{pid[:12]}.onnx"))
    if len(found) != 1:
        sys.exit(f"expected one ONNX model with id {pid[:12]}, found {found}")
    return len(quantized_onnx_nodes(onnx.load(str(found[0]), load_external_data=False)))


def need(name: str | None, what: str) -> dict:
    if name is None:
        sys.exit(f"missing {what}: run its phase first")
    return report(name) | {"_name": name}


def final(sha: str) -> None:
    budget = float(load_budgets()["accuracy"]["miou_drop_max_points"])
    for m in MODELS:
        ref = need(full_eval(m, "fp32"), f"{m} fp32 full-val eval")
        rows, gate_failed = [], []
        for label, eng in checked(m).items():
            pf = need(latest(f"perf__{m}__trt__{label}__c5__*.json"), f"{m} {label} perf")
            ev = need(full_eval(m, label), f"{m} {label} full-val eval")
            par = latest(f"parity__{m}__trt__{label}__golden_val32__golden__*.json")
            if label != "fp16" and par is None:
                sys.exit(f"{m} {label} has no golden parity report: run the parity phase")
            row = {
                "label": label, "engine": eng.name,
                "p99_ms": pf["metrics"]["spans_ms"]["infer"]["p99"],
                "drop_full": (ref["metrics"]["miou"] - ev["metrics"]["miou"]) * 100,
                "drop_heldout": (ref["metrics"]["subsets"][HELD_OUT]["miou"]
                                 - ev["metrics"]["subsets"][HELD_OUT]["miou"]) * 100,
                "quantized_nodes": quantized_nodes(eng),
                "perf_passed": pf["passed"],
                "parity_passed": None if par is None else report(par)["passed"],
                "sources": [pf["_name"], ev["_name"], par],
            }  # fmt: skip
            # The final engine must also pass the probe's gates and golden parity (C5 exit
            # criteria). FP16's golden parity is C4's (D-030).
            if not row["perf_passed"] or row["parity_passed"] is False:
                gate_failed.append(label)
            else:
                rows.append(row)
        choice, why = select_final(rows, budget)
        int8 = [r for r in rows if r["label"] != "fp16"]
        best_int8, why_int8 = select_final(int8, budget) if int8 else (None, {"reason": "none"})
        metrics = {
            "final": choice and choice["label"], "why": why | {"gate_failed": gate_failed},
            "best_int8": best_int8 and best_int8["label"], "why_int8": why_int8,
            "rows": rows, "fp32_eval": ref["_name"],
        }  # fmt: skip
        out = REPORTS_DIR / f"final__{m}__{sha}.json"
        write_report(_summary_report("final", m, {"budget_points": budget}, metrics), out)
        print(f"{m}: final {metrics['final']}, best INT8 {metrics['best_int8']} -> {out.name}",
              flush=True)  # fmt: skip


def best_int8(m: str) -> tuple[str, Path] | None:
    fin = need(latest(f"final__{m}__*.json"), f"{m} final report")
    label = fin["metrics"]["best_int8"]
    if label is None:
        return None
    row = next(r for r in fin["metrics"]["rows"] if r["label"] == label)
    return label, ENGINES_DIR / m / row["engine"]


def profile(sha: str) -> None:
    for m in MODELS:
        targets = {"fp16": engine_of(baselines(m)["fp16"])}
        if (b := best_int8(m)) is not None:
            targets[b[0]] = b[1]
        for label, eng in targets.items():
            if latest(f"trtexec__{m}__{label}__c5profile__{CLOCK}__*.json"):
                continue  # profiled in an earlier run
            out = REPORTS_DIR / f"trtexec__{m}__{label}__c5profile__{CLOCK}__{sha}.json"
            run("python", "scripts/trtexec_report.py", "--engine", str(eng), "--profile",
                "--out", str(out))  # fmt: skip


def nsys(sha: str) -> None:
    m = "segformer_b0"  # the transformer; C4 traced its FP16 engine
    b = best_int8(m)
    if b is None:
        sys.exit(f"{m} has no INT8 engine within the budget to trace")
    if latest(f"nsys__{m}__*__{sha}.json") or latest(f"perf__{m}__trt__*__nsys__{sha}.json"):
        return  # traced at this commit
    run("python", "scripts/nsys_report.py", "--model", m, "--engine", str(b[1]))


PHASES = {"build": build, "sub": sub, "parity": parity, "sens": sens, "trtexec": trtexec,
          "finalists": finalists, "perf": perf, "eval": eval_, "final": final,
          "profile": profile, "nsys": nsys}  # fmt: skip


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=sorted(PHASES))
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    PHASES[args.phase](_sha())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
