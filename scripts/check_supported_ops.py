"""Check configs/supported_ops.yaml against the real TensorRT ONNX parser (C4 task 4).

The list was seeded from onnx-tensorrt's docs (scripts/seed_supported_ops.py) and is
provisional until this check passes on the pinned TensorRT. Runs in the GPU container:
    scripts/remote.sh run 'python scripts/check_supported_ops.py'
Writes an `ops_check` report; exits 1 if an op the list calls supported is rejected.
"""

from __future__ import annotations

import sys

from edgeinfer.backends.trt import _import_runtime, runtime_fingerprint
from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report
from edgeinfer.validate.onnx_validator import load_supported_ops


def main() -> int:
    trt, _ = _import_runtime()
    supported = load_supported_ops()
    logger = trt.Logger(trt.Logger.WARNING)
    network = trt.Builder(logger).create_network()
    parser = trt.OnnxParser(network, logger)
    listed_yes = sorted(supported["ops"])
    listed_no = sorted(supported.get("unsupported", []))
    rejected = [op for op in listed_yes if not parser.supports_operator(op)]
    accepted = [op for op in listed_no if parser.supports_operator(op)]
    findings = [
        {"severity": "error", "code": "OPS_LISTED_BUT_REJECTED", "where": op,
         "message": f"{op} is listed as supported but the TensorRT parser rejects it"}
        for op in rejected
    ] + [
        {"severity": "warn", "code": "OPS_UNLISTED_BUT_ACCEPTED", "where": op,
         "message": f"{op} is listed as unsupported but the TensorRT parser accepts it"}
        for op in accepted
    ]  # fmt: skip
    runtime = runtime_fingerprint()
    sha, _ = git_state()
    report = make_report(
        tool="ops_check", model="n/a",
        backend_meta={"backend": "tensorrt-onnx-parser", "precision": "n/a",
                      "artifact_id": supported["source_commit"],
                      "device": f"{runtime['gpu_name']} (sm_{runtime['compute_capability']})",
                      "versions": {"tensorrt": runtime["tensorrt"]}},
        inputs={"supported_ops": supported["source"],
                "listed_tensorrt_version": supported["tensorrt_version"]},
        metrics={"checked": len(listed_yes) + len(listed_no), "rejected": rejected,
                 "accepted_unlisted": accepted},
        findings=findings, passed=not rejected,
    )  # fmt: skip
    out = write_report(report, REPORTS_DIR / f"ops_check__trt{runtime['tensorrt']}__{sha[:7]}.json")
    print(f"{len(rejected)} rejected, {len(accepted)} unlisted but accepted -> {out}")
    return 0 if not rejected else 1


if __name__ == "__main__":
    sys.exit(main())
