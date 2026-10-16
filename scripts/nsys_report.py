"""Nsight Systems trace of the end-to-end path, summarized into an `nsys` report (C4 task 6).

Profiles a short `edgeinfer perf` run, captured only between the probe's
cudaProfilerStart/Stop (`--cuda-profiler-range`), so engine loading isn't in it. The
.nsys-rep stays under artifacts/nsys/ (not committed) for the Nsight Systems GUI; the report
records GPU kernel, memory-copy and CUDA API totals and the GPU's busy share of a frame.
Runs in the GPU container:
    scripts/remote.sh run 'python scripts/nsys_report.py --model <name> --engine <path>'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from edgeinfer.artifacts import load_sidecar
from edgeinfer.backends.trt import runtime_fingerprint
from edgeinfer.perf.nsys import summarize_table
from edgeinfer.reports import REPO_ROOT, REPORTS_DIR, git_state, make_report, write_report

WARMUP, ITERATIONS = 10, 50  # one repeat: enough frames for steady-state totals
# Flags checked against `nsys profile --help` of Nsight Systems 2026.5.1 on 2026-10-16.
NSYS_FLAGS = ["--trace=cuda", "--sample=none", "--cpuctxsw=none",
              "--capture-range=cudaProfilerApi", "--capture-range-end=stop"]  # fmt: skip
REPORTS = ("cuda_gpu_kern_sum", "cuda_gpu_mem_time_sum", "cuda_api_sum")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--engine", required=True)
    args = ap.parse_args()
    engine = Path(args.engine)
    meta = load_sidecar(engine)
    sha, _ = git_state()
    tag = f"{meta['model']}__{meta['precision']}__{sha[:7]}"
    trace = REPO_ROOT / "artifacts" / "nsys" / f"nsys__{tag}"
    trace.parent.mkdir(parents=True, exist_ok=True)
    perf_out = (
        REPORTS_DIR / f"perf__{meta['model']}__trt__{meta['precision']}__nsys__{sha[:7]}.json"
    )

    profile = subprocess.run(
        ["nsys", "profile", *NSYS_FLAGS, f"--output={trace}", "--force-overwrite=true",
         "edgeinfer", "perf", "--model", args.model, "--backend", "trt", "--artifact", str(engine),
         "--warmup", str(WARMUP), "--iterations", str(ITERATIONS), "--repeats", "1",
         "--cuda-profiler-range", "--out", str(perf_out)],
        check=False,
    )  # fmt: skip
    # perf exits 1 when a check fails but still writes its report; anything else is a crash.
    if profile.returncode not in (0, 1) or not perf_out.is_file():
        print(f"error: profiled perf run failed (exit {profile.returncode})")
        return 2

    summaries = {}
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["nsys", "stats", "--report", ",".join(REPORTS), "--format", "csv",
             "--output", f"{tmp}/s", f"{trace}.nsys-rep"],
            check=True, stdout=subprocess.DEVNULL,
        )  # fmt: skip
        for name in REPORTS:
            summaries[name] = summarize_table(Path(f"{tmp}/s_{name}.csv").read_text(), name)

    perf = json.loads(perf_out.read_text())
    frames = WARMUP + ITERATIONS
    kernel_ms = summaries["cuda_gpu_kern_sum"]["total_ms"] / frames
    copy_ms = summaries["cuda_gpu_mem_time_sum"]["total_ms"] / frames
    e2e_ms = perf["metrics"]["spans_ms"]["e2e"]["mean"]
    metrics = {
        "frames_captured": frames,
        "gpu_kernel_ms_per_frame": kernel_ms,
        "gpu_copy_ms_per_frame": copy_ms,
        "e2e_mean_ms": e2e_ms,
        # Share of a frame the GPU spends on kernels or copies (they don't overlap: one
        # stream). The rest is CPU work: preprocessing, postprocessing, Python.
        "gpu_busy_share": (kernel_ms + copy_ms) / e2e_ms,
        "kernels": summaries["cuda_gpu_kern_sum"],
        "copies": summaries["cuda_gpu_mem_time_sum"],
        "cuda_api": summaries["cuda_api_sum"],
    }
    runtime = runtime_fingerprint()
    report = make_report(
        tool="nsys", model=meta["model"],
        backend_meta={"backend": "trt", "precision": meta["precision"],
                      "artifact_id": meta["artifact_id"], "artifact_sha256": meta["sha256"],
                      "device": f"{runtime['gpu_name']} (sm_{runtime['compute_capability']})",
                      "versions": {"tensorrt": runtime["tensorrt"],
                                   "cuda_driver": str(runtime["cuda_driver"]),
                                   "cuda_runtime": str(runtime["cuda_runtime"])}},
        inputs={"artifact": engine.name, "nsys_flags": NSYS_FLAGS, "warmup": WARMUP,
                "iterations": ITERATIONS, "perf_report": perf_out.name,
                "trace": f"artifacts/nsys/{trace.name}.nsys-rep (not committed)",
                "note": "profiling adds overhead; latency numbers come from perf reports"},
        metrics=metrics,
        findings=[f for f in perf["findings"] if f["code"].startswith(("PERF_CLOCK", "PERF_GPU"))],
        passed=perf["passed"],
    )  # fmt: skip
    out = write_report(report, REPORTS_DIR / f"nsys__{tag}.json")
    print(
        f"  GPU per frame: kernels {kernel_ms:.3f} ms, copies {copy_ms:.3f} ms; "
        f"e2e mean {e2e_ms:.1f} ms; GPU busy {metrics['gpu_busy_share']:.1%}"
    )
    for k in metrics["kernels"]["top"][:5]:
        print(f"  {k['share']:6.1%}  {k['total_ms'] / frames:7.3f} ms/frame  {k['name'][:90]}")
    print(f"{'PASS' if report['passed'] else 'FAIL'} -> {out}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
