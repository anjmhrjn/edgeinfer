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
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from edgeinfer.reports import REPO_ROOT, REPORTS_DIR, git_state

MODELS = os.environ.get("MODELS", "segformer_b0 deeplabv3plus_mbv2").split()
SENS_MARGIN = "0.7"  # D-017: the measured subset-vs-held-out gap


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
    return REPO_ROOT / "artifacts" / "engines" / r["model"] / r["metrics"]["file"]


def edgeinfer(*argv: str) -> int:
    print("+ edgeinfer " + " ".join(argv), flush=True)
    rc = subprocess.run(["edgeinfer", *argv], check=False).returncode
    if rc == 1:
        print(f"GATE FAILED (report written): edgeinfer {argv[0]}", flush=True)
    elif rc != 0:
        sys.exit(rc)
    return rc


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
    dw = latest(f"quantize__{m}__int8_percentile_fp16__dwfp16__*.json")
    if dw:  # SegFormer: depthwise convs in FP16 (D-034)
        models["int8fp16_dwfp16"] = dw
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
        for label, eng in engines(m).items():
            if latest(f"parity__{m}__trt__{label}__golden_val32__golden__*.json"):
                continue  # checked in an earlier run
            if label.startswith(("int8_c3", "int8fp16_none", "int8fp16_sens", "int8fp16_dw")):
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


PHASES = {"build": build, "sub": sub, "parity": parity, "sens": sens}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("phase", choices=sorted(PHASES))
    args = ap.parse_args()
    os.chdir(REPO_ROOT)
    PHASES[args.phase](_sha())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
