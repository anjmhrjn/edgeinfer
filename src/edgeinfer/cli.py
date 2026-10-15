"""Command-line entry point: `edgeinfer <command> ...`."""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_manifests(args: argparse.Namespace) -> int:
    from edgeinfer.data.cityscapes import cityscapes_root, list_image_ids
    from edgeinfer.data.manifests import MANIFEST_SPECS, build_manifest, format_manifest

    root = cityscapes_root(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_by_split: dict[str, list[str]] = {}
    built: dict[str, list] = {}
    for spec in MANIFEST_SPECS.values():  # in order: an excluded manifest comes first
        if spec.split not in ids_by_split:
            ids_by_split[spec.split] = list_image_ids(root, spec.split)
        excluded = built[spec.exclude] if spec.exclude else None
        built[spec.name] = build_manifest(spec, ids_by_split[spec.split], excluded)
        text = format_manifest(built[spec.name])
        path = out_dir / f"{spec.name}.txt"
        # Manifests are frozen once committed; a silent change would invalidate every report.
        if path.exists() and path.read_text() != text and not args.force:
            print(f"error: {path} exists and differs; use --force to overwrite")
            return 1
        path.write_text(text)
        print(f"{path}: {text.count(chr(10))} images from {spec.split}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from dataclasses import replace
    from itertools import islice

    from edgeinfer.backends import get_backend
    from edgeinfer.config import load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root, iter_samples
    from edgeinfer.data.manifests import load_manifest
    from edgeinfer.eval.harness import evaluate
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    if args.height or args.width:
        if not (args.height and args.width):
            print("error: --height and --width must be given together")
            return 2
        cfg = replace(cfg, input=replace(cfg.input, height=args.height, width=args.width))
    entries = load_manifest(args.manifest)
    if args.limit:
        entries = entries[: args.limit]
    root = cityscapes_root(args.root)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")

    backend = get_backend(args.backend)
    backend.load(Path(args.artifact) if args.artifact else None, cfg)
    h, w = cfg.input.height, cfg.input.width
    print(f"eval {cfg.name} on {args.backend}, {len(entries)} images at {h}x{w}")
    metrics, findings = evaluate(
        cfg, backend, islice(iter_samples(root, entries), len(entries)), total=len(entries)
    )
    report = make_report(
        tool="eval",
        model=cfg.name,
        backend_meta=backend.metadata(),
        inputs={
            "manifest": Path(args.manifest).stem,
            "artifact": Path(args.artifact).name if args.artifact else None,
            "limit": args.limit,
            "input_hw": [h, w],
            "eval_resolution": "label (full)",
        },
        metrics=metrics,
        findings=findings,
    )
    limit_tag = f"__first{args.limit}" if args.limit else ""
    name = f"eval__{cfg.name}__{args.backend}__{Path(args.manifest).stem}__{h}x{w}{limit_tag}"
    out = Path(args.out) if args.out else REPORTS_DIR / f"{name}__{sha[:7]}.json"
    write_report(report, out)
    print(f"mIoU {metrics['miou']:.4f} over {metrics['n_images']} images -> {out}")
    return 0


def cmd_golden(args: argparse.Namespace) -> int:
    from edgeinfer.backends import get_backend
    from edgeinfer.config import load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root, iter_samples
    from edgeinfer.data.manifests import load_manifest
    from edgeinfer.golden import GOLDEN_DIR, HASH_DIR, format_hashes, write_golden
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    entries = load_manifest(args.manifest)
    root = cityscapes_root(args.root)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; these outputs must not be the committed golden set")
    backend = get_backend("torch_ref")
    backend.load(None, cfg)
    out_dir = Path(args.out) if args.out else GOLDEN_DIR / cfg.name
    print(f"golden {cfg.name}: {len(entries)} images -> {out_dir}")
    hashes = write_golden(cfg, backend, iter_samples(root, entries), out_dir)
    text = format_hashes(hashes)

    hash_path = Path(args.hashes) if args.hashes else HASH_DIR / f"{cfg.name}.sha256"
    if hash_path.exists() and hash_path.read_text() != text:
        # Committed hashes are the reference. A mismatch means the outputs changed
        # (code, weights, library or thread-count differences) and needs a look.
        if not args.force:
            print(f"error: outputs differ from committed {hash_path}; investigate, or --force")
            return 1
        print(f"warning: overwriting differing hashes in {hash_path} (--force)")
    hash_path.parent.mkdir(parents=True, exist_ok=True)
    hash_path.write_text(text)
    report = make_report(
        tool="golden",
        model=cfg.name,
        backend_meta=backend.metadata(),
        inputs={
            "manifest": Path(args.manifest).stem,
            "input_hw": [cfg.input.height, cfg.input.width],
        },
        metrics={"n_images": len(hashes), "hash_file": str(hash_path.name)},
    )
    write_report(report, REPORTS_DIR / f"golden__{cfg.name}__{sha[:7]}.json")
    print(f"wrote {len(hashes)} files; hashes in {hash_path}")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    import platform
    from collections import Counter

    import onnx

    from edgeinfer.config import load_model_config
    from edgeinfer.export.onnx_export import ONNX_DIR, export_model
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")
    out_root = Path(args.out) if args.out else ONNX_DIR
    path, meta = export_model(cfg, out_root, force=args.force)
    ops = Counter(n.op_type for n in onnx.load(str(path)).graph.node)
    report = make_report(
        tool="export",
        model=cfg.name,
        backend_meta={
            "backend": "onnx",
            "precision": "fp32",
            "device": f"cpu ({platform.machine()})",
            "artifact_id": meta["artifact_id"],
            "versions": meta["versions"],
        },
        inputs={"weights_id": meta["weights_id"], "export": meta["export"]},
        metrics={
            "file": meta["file"],
            "sha256": meta["sha256"],
            "size_bytes": meta["size_bytes"],
            "n_nodes": sum(ops.values()),
            "op_counts": dict(sorted(ops.items())),
            "sanity_max_abs": meta["sanity_max_abs"],
            "sanity_max_abs_ref": meta["sanity_max_abs_ref"],
        },
    )
    report_dir = Path(args.report_dir) if args.report_dir else REPORTS_DIR
    out = write_report(report, report_dir / f"export__{cfg.name}__{sha[:7]}.json")
    print(f"{path} ({meta['size_bytes'] / 1e6:.1f} MB, {sum(ops.values())} nodes) -> {out}")
    return 0


def _print_findings(findings: list[dict[str, str]]) -> None:
    for f in findings:
        where = f" [{f['where']}]" if f.get("where") else ""
        print(f"  {f['severity'].upper():5} {f['code']}{where}: {f['message']}")


def cmd_validate(args: argparse.Namespace) -> int:
    import platform

    from edgeinfer.config import load_model_config
    from edgeinfer.export.onnx_export import artifact_identity
    from edgeinfer.reports import (
        REPORTS_DIR,
        git_state,
        make_report,
        package_versions,
        write_report,
    )
    from edgeinfer.validate.onnx_validator import (
        load_supported_ops,
        supported_ops_ref,
        validate_onnx,
    )

    cfg = load_model_config(args.model)
    artifact = Path(args.artifact)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")
    supported = load_supported_ops()
    metrics, findings = validate_onnx(artifact, cfg, supported)
    passed = metrics["n_errors"] == 0
    report = make_report(
        tool="validate",
        model=cfg.name,
        backend_meta={
            "backend": "onnx",
            "device": f"cpu ({platform.machine()})",
            **artifact_identity(artifact),
            "versions": package_versions("onnx"),
        },
        inputs={"artifact": artifact.name, "config": Path(args.model).name},
        thresholds={"supported_ops": supported_ops_ref(supported)},
        metrics=metrics,
        findings=findings,
        passed=passed,
    )
    out = Path(args.out) if args.out else REPORTS_DIR / f"validate__{cfg.name}__{sha[:7]}.json"
    write_report(report, out)
    _print_findings(findings)
    status = "PASS" if passed else "FAIL"
    print(f"{status}: {metrics['n_errors']} errors, {metrics['n_warnings']} warnings -> {out}")
    return 0 if passed else 1


def cmd_parity(args: argparse.Namespace) -> int:
    import os
    import platform

    import onnx

    from edgeinfer.analyze.parity import (
        load_tolerances,
        run_layer_parity,
        run_parity,
        tolerances_dict,
    )
    from edgeinfer.backends import get_backend
    from edgeinfer.config import load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root, iter_samples
    from edgeinfer.data.manifests import load_manifest
    from edgeinfer.golden import GOLDEN_DIR, verify_golden
    from edgeinfer.onnx_utils import layer_groups, layer_tensors
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    entries = load_manifest(args.manifest)
    if args.limit:
        entries = entries[: args.limit]
    mode = "npz" if args.from_npz else args.inputs
    if mode != "npz" and not args.artifact:
        print("error: --artifact is required unless --from-npz is given")
        return 2
    if args.per_layer and mode == "npz":
        print("error: --per-layer needs a backend; it can't use --from-npz")
        return 2
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")

    golden_dir = Path(args.golden_dir) if args.golden_dir else GOLDEN_DIR / cfg.name
    verified = verify_golden(cfg.name, golden_dir, [e.image_id for e in entries])
    # The dataset is needed in images mode and optional otherwise (labels -> mIoU delta).
    root = None
    if mode == "images" or args.root or os.environ.get("CITYSCAPES_ROOT"):
        root = cityscapes_root(args.root)
    samples = iter_samples(root, entries) if root else None

    backend = None
    if mode == "npz":
        backend_meta = {
            "backend": args.backend_name,
            "precision": args.precision or "unknown",
            "device": "external",
            "artifact_id": args.artifact_id or "unknown",
        }
    else:
        backend = get_backend(args.backend)
        backend.load(Path(args.artifact), cfg)
        backend_meta = backend.metadata()
    precision = args.precision or backend_meta["precision"]
    tol = load_tolerances(precision)

    print(f"parity {cfg.name}: {args.backend if backend else 'npz'} ({precision}), {mode} mode, "
          f"{len(entries)} images")  # fmt: skip
    metrics, findings, pairs = run_parity(
        cfg, entries, golden_dir, tol, mode,
        backend=backend, samples=samples, npz_dir=Path(args.from_npz) if args.from_npz else None,
    )  # fmt: skip
    if not verified:
        findings.append({
            "severity": "info", "code": "PARITY_GOLDEN_UNVERIFIED",
            "message": f"no committed hashes checked for {golden_dir}",
        })  # fmt: skip

    inputs = {
        "manifest": Path(args.manifest).stem, "limit": args.limit, "mode": mode,
        "artifact": Path(args.artifact).name if args.artifact else None,
        "config": Path(args.model).name, "golden_verified": verified,
        "tolerances": precision,
    }  # fmt: skip
    if args.per_layer:
        assert backend is not None
        ref_path = Path(args.ref_artifact) if args.ref_artifact else Path(args.artifact)
        ref = get_backend("ort_cpu")
        ref.load(ref_path, cfg)
        ref_model = onnx.load(str(ref_path))
        ops = {n.output[0]: n.op_type for n in ref_model.graph.node}
        names = layer_tensors(ref_model)
        if Path(args.artifact).suffix == ".onnx":
            produced = {o for n in onnx.load(str(args.artifact)).graph.node for o in n.output}
            names = [n for n in names if n in produced]
        groups = layer_groups(ref_model, names, args.layer_group_mb << 20)
        print(f"  per-layer: {len(names)} tensors in {len(groups)} groups, "
              f"{min(args.layer_images, len(pairs))} images")  # fmt: skip
        layer_metrics, layer_findings = run_layer_parity(
            ref, backend, [(n, ops[n]) for n in names], groups, pairs[: args.layer_images], tol
        )
        metrics["per_layer"] = layer_metrics
        findings += layer_findings
        inputs["ref_artifact"] = ref_path.name
        inputs["ref_artifact_id"] = ref.metadata()["artifact_id"]

    passed = not any(f["severity"] == "error" for f in findings)
    report = make_report(
        tool="parity", model=cfg.name, backend_meta=backend_meta, inputs=inputs,
        thresholds=tolerances_dict(tol), metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    if args.out:
        out = Path(args.out)
    else:
        tag = f"{backend_meta['backend']}__{precision}__{Path(args.manifest).stem}__{mode}"
        tag += "__layers" if args.per_layer else ""
        tag += f"__first{args.limit}" if args.limit else ""
        out = REPORTS_DIR / f"parity__{cfg.name}__{tag}__{sha[:7]}.json"
    write_report(report, out)
    _print_findings(findings)
    m = metrics
    print(f"  logits max abs {m['logits_max_abs']}, cosine min {m['logits_cosine_min']}, "
          f"argmax agreement min {m['argmax_agreement_min']}")  # fmt: skip
    if "miou_delta_points" in m:
        print(f"  mIoU {m['miou_reference'] * 100:.3f} -> {m['miou_candidate'] * 100:.3f} "
              f"({m['miou_delta_points']:+.4f} points)")  # fmt: skip
    print(f"{'PASS' if passed else 'FAIL'} -> {out} ({platform.machine()})")
    return 0 if passed else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="edgeinfer")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("manifests", help="generate data/manifests/*.txt from the dataset")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--out", default="data/manifests")
    p.add_argument("--force", action="store_true", help="overwrite manifests that differ")
    p.set_defaults(func=cmd_manifests)

    p = sub.add_parser("eval", help="mIoU of a model/backend on a manifest; writes a JSON report")
    p.add_argument("--model", required=True, help="config name, e.g. segformer_b0")
    p.add_argument("--backend", default="torch_ref")
    p.add_argument("--artifact", help="model file for the backend (e.g. ONNX for ort_cpu)")
    p.add_argument("--manifest", default="eval_val")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--height", type=int, help="override the config input height")
    p.add_argument("--width", type=int, help="override the config input width")
    p.add_argument("--limit", type=int, help="first N images only (smoke runs)")
    p.add_argument("--out", help="report path (default: artifacts/reports/...)")
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("golden", help="FP32 reference inputs + logits for parity (torch_ref)")
    p.add_argument("--model", required=True)
    p.add_argument("--manifest", default="golden_val32")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--out", help="output dir (default: artifacts/golden/<model>)")
    p.add_argument("--hashes", help="hash file (default: data/golden_hashes/<model>.sha256)")
    p.add_argument("--force", action="store_true", help="overwrite differing committed hashes")
    p.set_defaults(func=cmd_golden)

    p = sub.add_parser("export", help="PyTorch -> ONNX (FP32, static shape); writes a report")
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--out", help="artifact root (default: artifacts/onnx)")
    p.add_argument("--report-dir", help="report directory (default: artifacts/reports)")
    p.add_argument("--force", action="store_true", help="re-export even if the artifact exists")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("validate", help="check an ONNX model against its config; writes a report")
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--artifact", required=True, help="ONNX file")
    p.add_argument("--out", help="report path (default: artifacts/reports/...)")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("parity", help="compare a backend's outputs with the FP32 golden outputs")
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--backend", default="ort_cpu")
    p.add_argument("--artifact", help="model file for the backend")
    p.add_argument("--inputs", choices=("images", "golden"), default="images",
                   help="images: preprocess raw images (end to end, needs the dataset); "
                        "golden: feed the stored golden inputs (model only)")  # fmt: skip
    p.add_argument("--from-npz", help="compare logits from an external runtime (<image_id>.npz)")
    p.add_argument("--backend-name", default="external", help="backend label for --from-npz")
    p.add_argument("--artifact-id", help="artifact id for --from-npz reports")
    p.add_argument("--precision", help="tolerance set (default: the backend's precision)")
    p.add_argument("--manifest", default="golden_val32")
    p.add_argument("--golden-dir", help="default: artifacts/golden/<model>")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--limit", type=int, help="first N images only (smoke runs)")
    p.add_argument("--per-layer", action="store_true", help="also compare intermediate tensors")
    p.add_argument("--ref-artifact", help="per-layer reference ONNX (default: --artifact)")
    p.add_argument("--layer-images", type=int, default=4, help="images used for per-layer")
    p.add_argument("--layer-group-mb", type=int, default=1024,
                   help="max MB of intermediate tensors fetched per run")  # fmt: skip
    p.add_argument("--out", help="report path (default: artifacts/reports/...)")
    p.set_defaults(func=cmd_parity)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
