"""Measure legitimate FP32 variation, as evidence for the FP32 parity tolerances (D-014).

For each model, on the golden images:
  - ORT FP32 vs golden under different session settings (all graph optimizations,
    none, one thread): how much do valid FP32 executions differ?
  - per-layer drift between ORT with all optimizations and with none
  - near-tie pixels: fraction of stride-4 pixels whose top-2 logit margin is below
    2x a candidate max-abs tolerance (only those can change class)
Writes artifacts/reports/tolerance_study__<model>__<sha>.json.

Usage: uv run python scripts/tolerance_study.py <model> [--images 4] [--margin-tol 1e-3]
"""

from __future__ import annotations

import argparse

import numpy as np
import onnx
import onnxruntime as ort

from edgeinfer.analyze.parity import _cosine, compare_logits
from edgeinfer.backends.ort_cpu import OrtCpuBackend
from edgeinfer.config import load_model_config
from edgeinfer.data.manifests import load_manifest
from edgeinfer.export.onnx_export import artifact_path
from edgeinfer.golden import GOLDEN_DIR, verify_golden
from edgeinfer.onnx_utils import layer_groups, layer_tensors
from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report


def _variant(opt_level: ort.GraphOptimizationLevel | None = None, threads: int | None = None):
    class Variant(OrtCpuBackend):
        @staticmethod
        def _options() -> ort.SessionOptions:
            opts = OrtCpuBackend._options()
            if opt_level is not None:
                opts.graph_optimization_level = opt_level
            if threads is not None:
                opts.intra_op_num_threads = threads
            return opts

    return Variant()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--images", type=int, default=4, help="images for the session variants")
    ap.add_argument("--margin-tol", type=float, default=1e-3)
    args = ap.parse_args()

    cfg = load_model_config(args.model)
    art = artifact_path(cfg)
    ids = [e.image_id for e in load_manifest("golden_val32")]
    golden_dir = GOLDEN_DIR / cfg.name
    if not verify_golden(cfg.name, golden_dir, ids):
        raise SystemExit("golden outputs could not be verified against committed hashes")

    variants = {
        "ort_enable_all_4t": _variant(),
        "ort_disable_all_4t": _variant(ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        "ort_enable_all_1t": _variant(threads=1),
    }
    for v in variants.values():
        v.load(art, cfg)

    whole: dict[str, dict[str, float]] = {k: {"max_abs": 0.0, "one_minus_cos": 0.0}
                                          for k in variants}  # fmt: skip
    near_tie = []
    for i, image_id in enumerate(ids):
        with np.load(golden_dir / f"{image_id}.npz") as z:
            x, g = z["input"], z["logits"]
        top2 = np.sort(g[0], axis=0)[-2:]
        near_tie.append(float(((top2[1] - top2[0]) < 2 * args.margin_tol).mean()))
        if i < args.images:
            for name, backend in variants.items():
                r = compare_logits(g, backend.infer(x))
                whole[name]["max_abs"] = max(whole[name]["max_abs"], r["max_abs"])
                whole[name]["one_minus_cos"] = max(whole[name]["one_minus_cos"], 1 - r["cosine"])

    model = onnx.load(str(art))
    names = layer_tensors(model)
    with np.load(golden_dir / f"{ids[0]}.npz") as z:
        x = z["input"]
    a, b = variants["ort_enable_all_4t"], variants["ort_disable_all_4t"]
    rel, one_minus_cos = [], 0.0
    for group in layer_groups(model, names, 1 << 30):
        ra, rb = a.layer_outputs(x, group), b.layer_outputs(x, group)
        for n in group:
            scale = float(np.abs(ra[n]).max())
            diff = float(np.abs(ra[n].astype(np.float64) - rb[n]).max())
            rel.append(diff / scale if scale else 0.0)
            one_minus_cos = max(one_minus_cos, 1 - _cosine(ra[n], rb[n]))

    sha, _ = git_state()
    meta = a.metadata()
    report = make_report(
        tool="tolerance_study",
        model=cfg.name,
        backend_meta=meta,
        inputs={
            "manifest": "golden_val32",
            "variant_images": args.images,
            "margin_tol": args.margin_tol,
            "artifact": art.name,
        },
        metrics={
            "whole_model_vs_golden": whole,
            "per_layer_enable_vs_disable_all": {
                "n_tensors": len(rel),
                "rel_max_abs_median": float(np.median(rel)),
                "rel_max_abs_max": float(np.max(rel)),
                "one_minus_cos_max": one_minus_cos,
            },
            "near_tie_fraction": {"per_image_max": max(near_tie), "mean": float(np.mean(near_tie))},
        },
    )
    out = write_report(report, REPORTS_DIR / f"tolerance_study__{cfg.name}__{sha[:7]}.json")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
