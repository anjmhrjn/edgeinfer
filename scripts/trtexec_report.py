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
from edgeinfer.perf.trtexec import compare_compute, summarize_times
from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

# Flags checked against `trtexec --help` of TensorRT 11.3.0.99 on 2026-10-16. warmUp is in
# milliseconds; duration=0 makes the iteration count the only stopping rule.
TRTEXEC_FLAGS = ["--warmUp=2000", "--iterations=1000", "--duration=0"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--engine", required=True)
    ap.add_argument("--perf-report", help="perf report of the same engine to compare against")
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

    with tempfile.TemporaryDirectory() as tmp:
        times = Path(tmp) / "times.json"
        cmd = ["trtexec", f"--loadEngine={engine}", *TRTEXEC_FLAGS, f"--exportTimes={times}"]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
        spans = summarize_times(json.loads(times.read_text()))

    metrics: dict = {"spans_ms": spans}
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
        inputs={"artifact": engine.name, "trtexec_flags": TRTEXEC_FLAGS,
                "perf_report": Path(args.perf_report).name if args.perf_report else None},
        metrics=metrics,
    )  # fmt: skip
    out = REPORTS_DIR / f"trtexec__{meta['model']}__{meta['precision']}__{sha[:7]}.json"
    write_report(report, out)
    for span, s in spans.items():
        print(f"  {span:8s} p50 {s['p50']:8.3f}  p99 {s['p99']:8.3f}  max {s['max']:8.3f} ms")
    if probe is not None:
        print(f"  probe/trtexec compute: p50 {metrics['probe_over_trtexec_p50']:.3f}, "
              f"p99 {metrics['probe_over_trtexec_p99']:.3f}")  # fmt: skip
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
