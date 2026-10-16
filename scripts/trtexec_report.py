"""Run trtexec on an engine and write a `trtexec` report (C4 task 5).

trtexec's own output records no git SHA or environment, so this wraps it in the report
envelope; the raw per-iteration times stay out of artifacts/reports/. Runs in the GPU
container:
    scripts/remote.sh run 'python scripts/trtexec_report.py --engine <path> --perf-report <path>'
--perf-report (a perf report for the same engine) adds the probe/trtexec compute ratio.
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
from edgeinfer.perf.probe import load_perf_config
from edgeinfer.perf.telemetry import GpuSampler, clock_findings
from edgeinfer.perf.trtexec import compare_compute, summarize_times
from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

# Flags checked against `trtexec --help` of TensorRT 11.3.0.99 on 2026-10-16. warmUp is in
# milliseconds; duration=0 makes the iteration count the only stopping rule.
TRTEXEC_FLAGS = ["--warmUp=2000", "--iterations=1000", "--duration=0"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--perf-report", help="perf report of the same engine to compare against")
    ap.add_argument(
        "--clock-mhz",
        type=int,
        help="expected SM clock lock for clock searches (default: perf.yaml)",
    )
    args = ap.parse_args()
    engine = Path(args.engine)
    meta = load_sidecar(engine)

    probe = None
    if args.perf_report:
        probe = json.loads(Path(args.perf_report).read_text())
        if probe["inputs"].get("artifact") != engine.name:
            print(f"error: {args.perf_report} measured {probe['inputs'].get('artifact')}, "
                  f"not {engine.name}")  # fmt: skip
            return 2
    lock = args.clock_mhz if args.clock_mhz is not None else load_perf_config().gpu_clock_mhz
    if probe is not None and probe["inputs"]["perf_config"].get("gpu_clock_mhz") != lock:
        print(f"error: {args.perf_report} was measured with gpu_clock_mhz "
              f"{probe['inputs']['perf_config'].get('gpu_clock_mhz')}, not {lock}")  # fmt: skip
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "times.json"
        cmd = ["trtexec", f"--loadEngine={engine}", *TRTEXEC_FLAGS, f"--exportTimes={times}"]
        # Clocks are sampled through the whole run, warmup included (D-027).
        with GpuSampler() as sampler:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        spans = summarize_times(json.loads(times.read_text()))

    findings = clock_findings([sampler.summary] if sampler.summary else [], lock)
    if sampler.error:
        findings.append({"severity": "warn", "code": "PERF_TELEMETRY_UNAVAILABLE",
                         "message": f"GPU clocks not sampled: {sampler.error}"})  # fmt: skip
    passed = not any(f["severity"] == "error" for f in findings)
    metrics: dict = {"spans_ms": spans, "telemetry_during": sampler.summary}
    if probe is not None:
        metrics |= compare_compute(spans, probe["metrics"]["spans_ms"])
    runtime = runtime_fingerprint()
    sha, _ = git_state()
    report = make_report(
        tool="trtexec", model=meta["model"],
        backend_meta={"backend": "trtexec", "precision": meta["precision"],
                      "artifact_id": meta["artifact_id"],
                      "device": f"{runtime['gpu_name']} (sm_{runtime['compute_capability']})",
                      "versions": {"tensorrt": runtime["tensorrt"],
                                   "cuda_driver": str(runtime["cuda_driver"]),
                                   "cuda_runtime": str(runtime["cuda_runtime"])}},
        inputs={"artifact": engine.name, "trtexec_flags": TRTEXEC_FLAGS, "gpu_clock_mhz": lock,
                "perf_report": Path(args.perf_report).name if args.perf_report else None},
        metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    clock = f"{lock}MHz" if lock is not None else "unlocked"
    out = REPORTS_DIR / f"trtexec__{meta['model']}__{meta['precision']}__{clock}__{sha[:7]}.json"
    write_report(report, out)
    for span, s in spans.items():
        print(f"  {span:8s} p50 {s['p50']:8.3f}  p99 {s['p99']:8.3f}  max {s['max']:8.3f} ms")
    if probe is not None:
        print(f"  probe/trtexec compute: p50 {metrics['probe_over_trtexec_p50']:.3f}, "
              f"p99 {metrics['probe_over_trtexec_p99']:.3f}")  # fmt: skip
    for f in findings:
        print(f"  {f['severity'].upper():5s} {f['code']}: {f['message']}")
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
