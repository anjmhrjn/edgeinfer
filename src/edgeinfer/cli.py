"""Command-line entry point: `edgeinfer <command> ...`."""

from __future__ import annotations

import argparse
import contextlib
import traceback
from collections.abc import Iterator
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

    from edgeinfer.artifacts import ONNX_DIR
    from edgeinfer.config import load_model_config
    from edgeinfer.export.onnx_export import export_model
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

    from edgeinfer.artifacts import artifact_identity
    from edgeinfer.config import load_model_config
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


def cmd_quantize(args: argparse.Namespace) -> int:
    import platform

    from edgeinfer.artifacts import artifact_identity
    from edgeinfer.config import load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root
    from edgeinfer.quant import QuantError
    from edgeinfer.quant.calib import CALIB_DIR, calibrate, calibration_entries
    from edgeinfer.quant.config import (
        load_exclusions,
        load_group_rules,
        load_quant_config,
        no_exclusions,
    )
    from edgeinfer.quant.mixed import convert_int8_fp16, quantize_modelopt
    from edgeinfer.quant.ptq import PRECISION, quantize
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    if args.tool == "modelopt" and (args.float_dtype != "fp16" or args.exclusions):
        print("error: --tool modelopt writes INT8+FP16 models with ModelOpt's own placement: "
              "pass --float-dtype fp16 and no --exclusions")  # fmt: skip
        return 2
    cfg = load_model_config(args.model)
    qcfg = load_quant_config(args.quant_config)
    rules = load_group_rules(cfg.name, args.groups)
    exclusions = (load_exclusions(args.exclusions, cfg.name) if args.exclusions
                  else no_exclusions(cfg.name))  # fmt: skip
    fp32_path = Path(args.artifact)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")

    inputs = {
        "artifact": fp32_path.name, "config": Path(args.model).name, "quant_config": qcfg.to_dict(),
        "exclusions": exclusions.to_dict(), "groups": Path(args.groups).name if args.groups
        else f"{cfg.name}.yaml", "float_dtype": args.float_dtype, "tool": args.tool,
    }  # fmt: skip
    backend_meta = {"backend": "onnxruntime.quantization", "precision": PRECISION,
                    "device": f"cpu ({platform.machine()})", "versions": {}}  # fmt: skip
    metrics: dict = {}
    findings = []
    try:
        calibration_entries(qcfg.calibration)  # refuse a bad manifest before loading data
        out_root = Path(args.out_root) if args.out_root else None
        if args.tool == "modelopt":
            path, meta = quantize_modelopt(
                fp32_path, cfg, qcfg, cityscapes_root(args.root),
                **({"out_root": out_root} if out_root else {}),
                calib_root=Path(args.calib_dir) if args.calib_dir else CALIB_DIR,
            )  # fmt: skip
            backend_meta |= {"backend": "modelopt.onnx.quantization",
                             "precision": meta["precision"], "artifact_id": meta["artifact_id"],
                             "versions": meta["versions"]}  # fmt: skip
            metrics = {k: meta[k] for k in ("file", "sha256", "size_bytes", "parent_artifact_id",
                                            "calibration", "modelopt_settings", "placement",
                                            "float_precision", "quantize_seconds",
                                            "modelopt_peak_rss_bytes")}  # fmt: skip
            print(f"{path} ({meta['size_bytes'] / 1e6:.1f} MB), ModelOpt placement "
                  f"{meta['placement']['quantized_nodes']}")  # fmt: skip
        else:
            cal = calibrate(fp32_path, cfg, qcfg, cityscapes_root(args.root),
                            Path(args.calib_dir) if args.calib_dir else CALIB_DIR)  # fmt: skip
            path, meta = quantize(fp32_path, cfg, qcfg, cal, rules, exclusions,
                                  **({"out_root": out_root} if out_root else {}))  # fmt: skip
            extra = {}
            if args.float_dtype == "fp16":
                # TensorRT 11 runs unquantized layers in the model's float type (D-024, D-031).
                int8_id = meta["artifact_id"]
                path, meta = convert_int8_fp16(path, cfg, **({"out_root": out_root} if out_root
                                                             else {}))  # fmt: skip
                backend_meta["precision"] = meta["precision"]
                extra = {"int8_artifact_id": int8_id, "float_precision": meta["float_precision"],
                         "scale_rounding": meta["scale_rounding"]}  # fmt: skip
            backend_meta |= {"artifact_id": meta["artifact_id"], "versions": meta["versions"]}
            keys = ("file", "sha256", "size_bytes", "parent_artifact_id", "calibration",
                    "placement", "excluded_nodes")  # fmt: skip
            metrics = {k: meta[k] for k in keys} | extra
            metrics["calibration_dev"] = {k: cal[1][k] for k in
                                          ("seconds", "peak_rss_bytes", "n_tensors")}  # fmt: skip
            print(f"{path} ({meta['size_bytes'] / 1e6:.1f} MB), quantized "
                  f"{meta['placement']['quantized_nodes']}, excluded "
                  f"{meta['placement']['excluded_nodes'] or 'none'}")  # fmt: skip
    except QuantError as e:
        backend_meta["artifact_id"] = "none"
        findings.append({"severity": "error", "code": e.code, "message": str(e)})
    if findings:
        # The FP32 input is still identified, so a failed report says what it refused.
        inputs["artifact_id"] = artifact_identity(fp32_path)["artifact_id"]

    passed = not findings
    report = make_report(
        tool="quantize", model=cfg.name, backend_meta=backend_meta, inputs=inputs,
        metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    if args.out:
        out = Path(args.out)
    else:
        report_dir = Path(args.report_dir) if args.report_dir else REPORTS_DIR
        label = qcfg.name + ("_fp16" if args.float_dtype == "fp16" else "")
        label += "_modelopt" if args.tool == "modelopt" else ""
        out = report_dir / f"quantize__{cfg.name}__{label}__{exclusions.name}__{sha[:7]}.json"
    write_report(report, out)
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


def cmd_fp16(args: argparse.Namespace) -> int:
    import platform

    from edgeinfer.artifacts import artifact_identity
    from edgeinfer.config import load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root
    from edgeinfer.quant import QuantError
    from edgeinfer.quant.calib import CALIB_DIR
    from edgeinfer.quant.fp16 import PRECISION, convert_fp16, load_fp16_config
    from edgeinfer.reports import (
        REPORTS_DIR,
        git_state,
        make_report,
        package_versions,
        write_report,
    )

    cfg = load_model_config(args.model)
    fcfg = load_fp16_config(args.fp16_config)
    fp32_path = Path(args.artifact)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")
    inputs = {"artifact": fp32_path.name, "config": Path(args.model).name,
              "fp16_config": fcfg.to_dict()}  # fmt: skip
    # The main environment's onnx re-saves the model with its contract restored.
    backend_meta = {"backend": "modelopt.onnx.autocast", "precision": PRECISION,
                    "device": f"cpu ({platform.machine()})",
                    "versions": package_versions("onnx")}  # fmt: skip
    metrics: dict = {}
    findings = []
    try:
        path, meta = convert_fp16(
            fp32_path, cfg, fcfg, cityscapes_root(args.root),
            **({"out_root": Path(args.out_root)} if args.out_root else {}),
            calib_root=Path(args.calib_dir) if args.calib_dir else CALIB_DIR,
        )  # fmt: skip
        backend_meta["artifact_id"] = meta["artifact_id"]
        backend_meta["versions"] |= meta["versions"]
        metrics = {k: meta[k] for k in ("file", "sha256", "size_bytes", "parent_artifact_id",
                                        "calibration", "placement", "autocast_seconds",
                                        "autocast_peak_rss_bytes")}  # fmt: skip
        pl = meta["placement"]
        print(f"{path} ({meta['size_bytes'] / 1e6:.1f} MB): {pl['fp16_nodes']} nodes FP16, "
              f"{len(pl['fp32_nodes'])} kept FP32, {pl['cast_nodes']} casts")  # fmt: skip
    except QuantError as e:
        backend_meta["artifact_id"] = "none"
        inputs["artifact_id"] = artifact_identity(fp32_path)["artifact_id"]
        findings.append({"severity": "error", "code": e.code, "message": str(e)})
    passed = not findings
    report = make_report(
        tool="fp16", model=cfg.name, backend_meta=backend_meta, inputs=inputs,
        metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    out = (Path(args.out) if args.out
           else REPORTS_DIR / f"fp16__{cfg.name}__{fcfg.name}__{sha[:7]}.json")  # fmt: skip
    write_report(report, out)
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


def cmd_sensitivity(args: argparse.Namespace) -> int:
    import platform

    import onnxruntime as ort
    import yaml

    from edgeinfer.analyze.sensitivity import (
        NON_ADDITIVE_MIN_POINTS,
        SubsetScorer,
        run_sensitivity,
    )
    from edgeinfer.backends.ort_cpu import THREADS
    from edgeinfer.config import load_budgets, load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root, iter_samples
    from edgeinfer.data.manifests import load_manifest
    from edgeinfer.quant.calib import CALIB_DIR, calibrate
    from edgeinfer.quant.config import load_group_rules, load_quant_config
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    if args.candidates:
        return _sensitivity_prebuilt(args)
    if not (args.artifact and args.quant_config):
        print("error: --artifact and --quant-config are required (or --candidates)")
        return 2
    cfg = load_model_config(args.model)
    qcfg = load_quant_config(args.quant_config)
    rules = load_group_rules(cfg.name, args.groups)
    budget = float(load_budgets()["accuracy"]["miou_drop_max_points"])
    modes = tuple(args.modes.split(","))
    entries = load_manifest(args.manifest)
    if args.limit:
        entries = entries[: args.limit]
    if args.noise_threads == THREADS:
        print(f"error: --noise-threads must differ from the scoring thread count ({THREADS})")
        return 2
    root = cityscapes_root(args.root)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")

    fp32_path = Path(args.artifact)
    cal = calibrate(fp32_path, cfg, qcfg, root,
                    Path(args.calib_dir) if args.calib_dir else CALIB_DIR)  # fmt: skip
    print(f"sensitivity {cfg.name} ({qcfg.name}): {len(rules.names)} groups, modes {modes}, "
          f"{len(entries)} images from {Path(args.manifest).stem}")  # fmt: skip
    scorer = SubsetScorer(cfg, fp32_path, qcfg, cal, rules, iter_samples(root, entries),
                          Path(args.out_root) if args.out_root else None)  # fmt: skip
    full = scorer.score(())
    noise = scorer.score((), threads=args.noise_threads)
    floor = abs(noise.delta_points - full.delta_points)
    print(
        f"  FP32 mIoU {scorer.fp32_miou * 100:.3f}; full INT8 {full.delta_points:+.3f} points "
        f"({args.noise_threads} threads: {noise.delta_points:+.3f}; noise floor {floor:.3f})"
    )
    metrics, findings = run_sensitivity(
        rules.names, scorer.score, full=full, noise_floor=floor, budget=budget,
        margin=args.margin, modes=modes,
    )  # fmt: skip
    metrics = {"n_images": len(entries), "fp32_miou": scorer.fp32_miou,
               "fp32_artifact_id": scorer.fp32_artifact_id,
               "noise_run": {"threads": args.noise_threads, "delta_points": noise.delta_points},
               **metrics}  # fmt: skip
    passed = not any(f["severity"] == "error" for f in findings)
    report = make_report(
        tool="sensitivity", model=cfg.name,
        backend_meta={"backend": "ort_cpu", "precision": "int8", "threads": THREADS,
                      "device": f"cpu ({platform.machine()})", "artifact_id": full.artifact_id,
                      "versions": {"onnxruntime": ort.__version__}},
        inputs={"artifact": fp32_path.name, "quant_config": qcfg.to_dict(),
                "calibration_id": cal[1]["calibration_id"],
                "groups": Path(args.groups).name if args.groups else f"{cfg.name}.yaml",
                "manifest": Path(args.manifest).stem, "limit": args.limit, "modes": list(modes)},
        thresholds={"miou_drop_budget_points": budget, "margin_points": args.margin,
                    "target_drop_points": budget - args.margin,
                    "non_additive_min_points": NON_ADDITIVE_MIN_POINTS},
        metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    if args.out:
        out = Path(args.out)
    else:
        report_dir = Path(args.report_dir) if args.report_dir else REPORTS_DIR
        tag = f"{cfg.name}__{qcfg.name}__{Path(args.manifest).stem}"
        tag += f"__first{args.limit}" if args.limit else ""
        out = report_dir / f"sensitivity__{tag}__{sha[:7]}.json"
    write_report(report, out)
    if args.write_exclusions:
        exc_path = Path(args.write_exclusions)
        exc_path.parent.mkdir(parents=True, exist_ok=True)
        header = (
            f"# Suggested by `edgeinfer sensitivity` ({out.name}): groups kept in float so\n"
            f"# the drop on {Path(args.manifest).stem} is within the budget minus margin.\n"
        )
        exc_path.write_text(header + yaml.safe_dump({
            "name": exc_path.stem, "model": cfg.name,
            "groups": metrics["suggested_exclusions"], "nodes": [],
        }, sort_keys=False))  # fmt: skip
        print(f"  exclusions -> {exc_path}")
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


def _sensitivity_prebuilt(args: argparse.Namespace) -> int:
    """`edgeinfer sensitivity --candidates`: leave-one-out and greedy over configurations
    generated on the Mac, built and scored on `--backend` (TensorRT; D-032)."""
    import json
    from dataclasses import replace

    from edgeinfer.analyze.candidates import CandidateError, CandidateSet
    from edgeinfer.analyze.sensitivity import (
        NON_ADDITIVE_MIN_POINTS,
        PrebuiltScorer,
        run_sensitivity,
    )
    from edgeinfer.backends import get_backend
    from edgeinfer.backends.trt import build_engine
    from edgeinfer.backends.trt_meta import ENGINE_DIR, BackendError, load_build_config
    from edgeinfer.config import load_budgets, load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root, iter_samples
    from edgeinfer.data.manifests import load_manifest
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    if not args.ref_artifact:
        print(
            "error: --ref-artifact (the FP32 model on this backend) is required with --candidates"
        )
        return 2
    cands_path = Path(args.candidates)
    cands_report = json.loads(cands_path.read_text())
    cands = CandidateSet(cands_report)
    if cands.model != cfg.name:
        print(f"error: {cands_path.name} is for {cands.model}, not {cfg.name}")
        return 2
    bcfg = load_build_config(args.build_config)
    budget = float(load_budgets()["accuracy"]["miou_drop_max_points"])
    entries = load_manifest(args.manifest)
    if args.limit:
        entries = entries[: args.limit]
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")

    def build(onnx_path: Path) -> tuple[Path, dict]:
        return build_engine(onnx_path, cfg, bcfg)

    # Noise floor: the full INT8 model built again, cold (no timing cache) and separately,
    # so the builder chooses its kernels anew (D-029, D-032).
    noise_bcfg = replace(bcfg, timing_cache=False)

    def rebuild(onnx_path: Path) -> tuple[Path, dict]:
        return build_engine(
            onnx_path, cfg, noise_bcfg, out_root=ENGINE_DIR.parent / "engines_noise"
        )

    ref = get_backend(args.backend)
    ref.load(Path(args.ref_artifact), cfg)
    groups = cands.groups
    print(f"sensitivity {cfg.name} on {args.backend} ({args.precision}): top {len(groups)} "
          f"groups {groups}, leave-one-out, {len(entries)} images from "
          f"{Path(args.manifest).stem}")  # fmt: skip
    findings = []
    metrics: dict = {}
    try:
        scorer = PrebuiltScorer(cfg, ref, cands, build, iter_samples(cityscapes_root(args.root),
                                entries), args.backend, args.precision)  # fmt: skip
        full = scorer.score(())
        noise = scorer.score((), build=rebuild)
        floor = abs(noise.delta_points - full.delta_points)
        print(f"  FP32 mIoU {scorer.fp32_miou * 100:.3f}; full INT8 {full.delta_points:+.3f} "
              f"(cold rebuild {noise.delta_points:+.3f}; noise floor {floor:.3f})")  # fmt: skip
        metrics, findings = run_sensitivity(
            groups, scorer.score, full=full, noise_floor=floor, budget=budget,
            margin=args.margin, modes=("except",),
        )  # fmt: skip
        metrics = {"n_images": len(entries), "fp32_miou": scorer.fp32_miou,
                   "fp32_artifact_id": scorer.fp32_artifact_id,
                   "noise_run": {"cold_rebuild": True, "artifact_id": noise.artifact_id,
                                 "delta_points": noise.delta_points},
                   **metrics,
                   "engines": [{"excluded": list(k), "artifact_id": m["artifact_id"],
                                "sha256": m["sha256"], "build_seconds": m["build_seconds"]}
                               for k, m in sorted(scorer.built.items())]}  # fmt: skip
    except (CandidateError, BackendError) as e:
        findings.append({"severity": "error", "code": e.code, "message": str(e)})
    passed = not any(f["severity"] == "error" for f in findings)
    ref_meta = ref.metadata()
    report = make_report(
        tool="sensitivity", model=cfg.name,
        backend_meta={**ref_meta, "precision": args.precision,
                      "artifact_id": metrics.get("full_int8", {}).get("artifact_id", "none")},
        inputs={"candidates": cands_path.name, "ref_artifact": Path(args.ref_artifact).name,
                "ref_artifact_id": ref_meta["artifact_id"], "build_config": bcfg.to_dict(),
                "manifest": Path(args.manifest).stem, "limit": args.limit,
                "modes": ["except"], "groups": groups},
        thresholds={"miou_drop_budget_points": budget, "margin_points": args.margin,
                    "target_drop_points": budget - args.margin,
                    "non_additive_min_points": NON_ADDITIVE_MIN_POINTS},
        metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    if args.out:
        out = Path(args.out)
    else:
        report_dir = Path(args.report_dir) if args.report_dir else REPORTS_DIR
        tag = f"{cfg.name}__{args.backend}__{args.precision}__{Path(args.manifest).stem}"
        tag += f"__first{args.limit}" if args.limit else ""
        out = report_dir / f"sensitivity__{tag}__{sha[:7]}.json"
    write_report(report, out)
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


def cmd_candidates(args: argparse.Namespace) -> int:
    """Every subset of the top-N ORT sensitivity groups, quantized and converted to INT8+FP16
    for a TensorRT sensitivity run (D-032)."""
    import json
    import platform

    from edgeinfer.analyze.candidates import CandidateError, generate, top_groups
    from edgeinfer.config import load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root
    from edgeinfer.quant import QuantError
    from edgeinfer.quant.calib import CALIB_DIR, calibrate
    from edgeinfer.quant.config import load_group_rules, load_quant_config
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    qcfg = load_quant_config(args.quant_config)
    rules = load_group_rules(cfg.name, args.groups)
    fp32_path = Path(args.artifact)
    ranking_path = Path(args.ranking)
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")
    inputs = {"artifact": fp32_path.name, "quant_config": qcfg.to_dict(),
              "ranking_report": ranking_path.name, "top": args.top}  # fmt: skip
    metrics: dict = {}
    findings = []
    try:
        groups = top_groups(json.loads(ranking_path.read_text()), cfg.name, args.top)
        cal = calibrate(fp32_path, cfg, qcfg, cityscapes_root(args.root),
                        Path(args.calib_dir) if args.calib_dir else CALIB_DIR)  # fmt: skip
        print(f"candidates {cfg.name}: {2 ** len(groups)} configurations of {groups}")
        roots = {"out_root": Path(args.out_root)} if args.out_root else {}
        configs = generate(fp32_path, cfg, qcfg, cal, rules, groups, **roots)
        metrics = {"groups": groups, "n_configs": len(configs),
                   "calibration_id": cal[1]["calibration_id"], "configs": configs}  # fmt: skip
    except (CandidateError, QuantError) as e:
        findings.append({"severity": "error", "code": e.code, "message": str(e)})
    passed = not findings
    report = make_report(
        tool="candidates", model=cfg.name,
        backend_meta={"backend": "onnxruntime.quantization+modelopt", "precision": "int8_fp16",
                      "device": f"cpu ({platform.machine()})", "artifact_id": "none",
                      "versions": {}},
        inputs=inputs, metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    name = f"candidates__{cfg.name}__{qcfg.name}__top{args.top}__{sha[:7]}.json"
    out = Path(args.out) if args.out else REPORTS_DIR / name
    write_report(report, out)
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


def cmd_build(args: argparse.Namespace) -> int:
    import json

    import onnx

    from edgeinfer.artifacts import artifact_identity
    from edgeinfer.backends.trt import build_engine, runtime_fingerprint
    from edgeinfer.backends.trt_layers import layer_summary, quantized_onnx_nodes
    from edgeinfer.backends.trt_meta import BackendError, layer_datatypes, load_build_config
    from edgeinfer.config import load_model_config
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    bcfg = load_build_config(args.build_config)
    onnx_path = Path(args.artifact)
    debug = [n for n in (args.debug_outputs or "").split(",") if n]
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")
    runtime = runtime_fingerprint()  # TRT_UNAVAILABLE here: exit 2, no report
    device = f"cuda:0 ({runtime['gpu_name']}, sm_{runtime['compute_capability'].replace('.', '')})"
    onnx_id = artifact_identity(onnx_path)
    backend_meta = {
        "backend": "trt",
        "precision": onnx_id["precision"],
        "device": device,
        "artifact_id": "none",
        "versions": {"tensorrt": runtime["tensorrt"]},
    }
    inputs = {"artifact": onnx_path.name, "onnx_artifact_id": onnx_id["artifact_id"],
              "build_config": bcfg.to_dict(), "debug_outputs": debug}  # fmt: skip
    metrics: dict = {}
    findings = []
    try:
        roots = {"out_root": Path(args.out_root)} if args.out_root else {}
        path, meta = build_engine(onnx_path, cfg, bcfg, debug, **roots)
        backend_meta["artifact_id"] = meta["artifact_id"]
        keys = ("file", "sha256", "size_bytes", "parent_artifact_id", "runtime", "io",
                "num_layers", "layers_file", "device_memory_bytes", "timing_cache",
                "build_seconds")  # fmt: skip
        metrics = {k: meta[k] for k in keys}
        # What the engine actually runs in (strong typing should match the ONNX model).
        layers = json.loads((path.parent / meta["layers_file"]).read_text())
        metrics["layer_output_datatypes"] = layer_datatypes(layers)
        # INT8 engines: did every quantized ONNX node become an INT8 layer (Q/DQ fused)?
        quantized = (quantized_onnx_nodes(onnx.load(str(onnx_path), load_external_data=False))
                     if onnx_id["precision"] in ("int8", "int8_fp16") else None)  # fmt: skip
        metrics["layers"], layer_findings = layer_summary(layers, quantized)
        findings += layer_findings
        print(f"{path} ({meta['size_bytes'] / 1e6:.1f} MB, {meta['num_layers']} layers, "
              f"built in {meta['build_seconds']} s); layers {metrics['layers']['by_precision']}, "
              f"{metrics['layers']['reformat_layers']} reformats")  # fmt: skip
    except BackendError as e:
        findings.append({"severity": "error", "code": e.code, "message": str(e)})
    passed = not any(f["severity"] == "error" for f in findings)
    report = make_report(tool="build", model=cfg.name, backend_meta=backend_meta, inputs=inputs,
                         metrics=metrics, findings=findings, passed=passed)  # fmt: skip
    tag = f"{cfg.name}__{onnx_id['precision']}" + ("__debug" if debug else "")
    out = Path(args.out) if args.out else REPORTS_DIR / f"build__{tag}__{sha[:7]}.json"
    write_report(report, out)
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
    return 0 if passed else 1


@contextlib.contextmanager
def _cuda_profiler_range() -> Iterator[None]:
    """cudaProfilerStart/Stop around the block: the capture range for nsys (C4 task 6)."""
    from edgeinfer.backends.trt import _import_runtime, _ok

    _, cudart = _import_runtime()
    _ok(cudart, cudart.cudaProfilerStart())
    try:
        yield
    finally:
        _ok(cudart, cudart.cudaProfilerStop())


def cmd_perf(args: argparse.Namespace) -> int:
    from dataclasses import asdict

    import numpy as np

    from edgeinfer.backends import get_backend
    from edgeinfer.config import ConfigError, load_budgets, load_model_config
    from edgeinfer.data.cityscapes import cityscapes_root, iter_samples
    from edgeinfer.data.manifests import load_manifest
    from edgeinfer.perf.probe import (
        PerfConfig,
        latency_budget,
        load_perf_config,
        memory_budget,
        run_probe,
    )
    from edgeinfer.reports import REPORTS_DIR, git_state, make_report, write_report

    cfg = load_model_config(args.model)
    try:
        pcfg = load_perf_config(args.perf_config)
        overrides = {
            "warmup_iterations": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "manifest": args.manifest, "max_images": args.max_images,
        }  # fmt: skip
        merged = asdict(pcfg) | {k: v for k, v in overrides.items() if v is not None}
        pcfg = PerfConfig.from_dict(merged, where="perf config + CLI overrides")
        budgets = load_budgets()
        latency_budget(budgets)
        memory_budget(budgets)
    except ConfigError as e:
        print(f"error: {e}")
        return 2
    sha, dirty = git_state()
    if dirty:
        print("warning: uncommitted changes; this report must not back any published number")

    if args.synthetic:
        h, w = (int(v) for v in args.synthetic.lower().split("x"))
        rng = np.random.default_rng(0)
        frames = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(pcfg.max_images)]
        source = f"synthetic {h}x{w}"
    else:
        entries = load_manifest(pcfg.manifest)[: pcfg.max_images]
        frames = [s.image for s in iter_samples(cityscapes_root(args.root), entries)]
        source = Path(pcfg.manifest).stem

    backend = get_backend(args.backend)
    backend.load(Path(args.artifact) if args.artifact else None, cfg)
    meta = backend.metadata()
    print(f"perf {cfg.name} on {args.backend} ({meta['precision']}): {pcfg.warmup_iterations} "
          f"warmup + {pcfg.repeats} x {pcfg.iterations} iterations, frames: {source}")  # fmt: skip
    if args.cuda_profiler_range and args.backend != "trt":
        print("error: --cuda-profiler-range needs --backend trt")
        return 2
    profiler = _cuda_profiler_range() if args.cuda_profiler_range else contextlib.nullcontext()
    try:
        with profiler:
            metrics, findings, passed = run_probe(cfg, backend, frames, pcfg, budgets)
    except ConfigError as e:
        print(f"error: {e}")
        return 2
    if metrics:
        size = Path(args.artifact).stat().st_size if args.artifact else None
        metrics["artifact_size_bytes"] = size
        if "device_memory" in meta:
            metrics["device_memory"] = meta["device_memory"]
    report = make_report(
        tool="perf", model=cfg.name, backend_meta=meta,
        inputs={"frames": source, "frame_hw": list(frames[0].shape[:2]),
                "artifact": Path(args.artifact).name if args.artifact else None,
                "perf_config": asdict(pcfg), "gc": "enabled; collected before each repeat"},
        thresholds={"latency": budgets.get("latency"), "memory": budgets.get("memory")},
        metrics=metrics, findings=findings, passed=passed,
    )  # fmt: skip
    name = f"perf__{cfg.name}__{meta['backend']}__{meta['precision']}"
    out = Path(args.out) if args.out else REPORTS_DIR / f"{name}__{sha[:7]}.json"
    write_report(report, out)
    for span, s in metrics.get("spans_ms", {}).items():
        print(f"  {span:12s} p50 {s['p50']:9.3f}  p90 {s['p90']:9.3f}  p99 {s['p99']:9.3f}  "
              f"max {s['max']:9.3f} ms")  # fmt: skip
    if metrics:
        print(f"  throughput {metrics['throughput_fps']:.2f} frames/s, "
              f"repeat p99 spread {metrics['repeat_p99_spread']:.1%}")  # fmt: skip
    _print_findings(findings)
    print(f"{'PASS' if passed else 'FAIL'} -> {out}")
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

    p = sub.add_parser("quantize", help="FP32 ONNX -> INT8 QDQ ONNX (ORT PTQ); writes a report")
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--artifact", required=True, help="FP32 ONNX file (from `edgeinfer export`)")
    p.add_argument("--quant-config", required=True, help="configs/quant/<name>.yaml, or a path")
    p.add_argument("--exclusions", help="groups/nodes kept in float (configs/quant/exclusions/)")
    p.add_argument("--groups", help="group rules (default: configs/quant/groups/<model>.yaml)")
    p.add_argument("--tool", choices=("ort", "modelopt"), default="ort",
                   help="ort: our placement (D-016); modelopt: ModelOpt's placement and "
                        "calibrator, for comparison (D-015; needs --float-dtype fp16)")  # fmt: skip
    p.add_argument("--float-dtype", choices=("fp32", "fp16"), default="fp32",
                   help="type of everything outside the Q/DQ pairs; fp16 converts the INT8 "
                        "model with ModelOpt (precision int8_fp16, D-031)")  # fmt: skip
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--calib-dir", help="calibration cache root (default: artifacts/calib)")
    p.add_argument("--out-root", help="artifact root (default: artifacts/onnx)")
    p.add_argument("--report-dir", help="report directory (default: artifacts/reports)")
    p.add_argument("--out", help="report path")
    p.set_defaults(func=cmd_quantize)

    p = sub.add_parser("fp16", help="FP32 ONNX -> FP16 ONNX (ModelOpt AutoCast) for "
                                    "strongly typed TensorRT builds; writes a report")  # fmt: skip
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--artifact", required=True, help="FP32 ONNX file (from `edgeinfer export`)")
    p.add_argument("--fp16-config", default="fp16_autocast",
                   help="configs/precision/<name>.yaml, or a path")  # fmt: skip
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--calib-dir", help="calibration input root (default: artifacts/calib)")
    p.add_argument("--out-root", help="artifact root (default: artifacts/onnx)")
    p.add_argument("--out", help="report path")
    p.set_defaults(func=cmd_fp16)

    p = sub.add_parser("sensitivity", help="rank layer groups by INT8 accuracy impact; "
                                           "suggest an exclusion list")  # fmt: skip
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--artifact", help="FP32 ONNX file (ORT mode)")
    p.add_argument("--quant-config", help="configs/quant/<name>.yaml, or a path (ORT mode)")
    p.add_argument("--candidates", help="candidates report: score its prebuilt configurations "
                                        "on --backend instead (TensorRT mode, D-032)")  # fmt: skip
    p.add_argument("--backend", default="trt", help="with --candidates: backend to score on")
    p.add_argument("--ref-artifact", help="with --candidates: the FP32 model on that backend")
    p.add_argument("--precision", default="int8_fp16", choices=("int8", "int8_fp16"),
                   help="with --candidates: the model of each configuration to build")  # fmt: skip
    p.add_argument("--build-config", default="build", help="with --candidates: configs/<name>.yaml")
    p.add_argument("--groups", help="group rules (default: configs/quant/groups/<model>.yaml)")
    p.add_argument("--manifest", default="sens_val100", help="scoring images")
    p.add_argument("--limit", type=int, help="first N images only (smoke runs)")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument(
        "--modes",
        default="only,except",
        help="comma-separated: only (one-at-a-time), except (leave-one-out)",
    )
    p.add_argument(
        "--margin",
        type=float,
        default=0.0,
        help="points kept below the budget on the scoring set (subset vs full val)",
    )
    p.add_argument(
        "--noise-threads",
        type=int,
        default=1,
        help="thread count for the noise-floor rerun of full INT8",
    )
    p.add_argument("--write-exclusions", help="write the suggested exclusion list (YAML) here")
    p.add_argument("--calib-dir", help="calibration cache root (default: artifacts/calib)")
    p.add_argument("--out-root", help="artifact root for the configurations' models")
    p.add_argument("--report-dir", help="report directory (default: artifacts/reports)")
    p.add_argument("--out", help="report path")
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("candidates", help="every subset of the top-N sensitivity groups as "
                                          "INT8+FP16 models, for a TensorRT run")  # fmt: skip
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--artifact", required=True, help="FP32 ONNX file")
    p.add_argument("--quant-config", required=True, help="configs/quant/<name>.yaml, or a path")
    p.add_argument("--ranking", required=True, help="ORT sensitivity report to take the top from")
    p.add_argument("--top", type=int, default=5, help="number of groups (2^N configurations)")
    p.add_argument("--groups", help="group rules (default: configs/quant/groups/<model>.yaml)")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--calib-dir", help="calibration cache root (default: artifacts/calib)")
    p.add_argument("--out-root", help="artifact root (default: artifacts/onnx)")
    p.add_argument("--out", help="report path")
    p.set_defaults(func=cmd_candidates)

    p = sub.add_parser("build", help="ONNX -> TensorRT engine with its metadata sidecar; "
                                     "writes a report (GPU container only)")  # fmt: skip
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--artifact", required=True, help="ONNX file (FP32, FP16 or INT8 Q/DQ)")
    p.add_argument("--build-config", default="build", help="configs/<name>.yaml, or a path")
    p.add_argument(
        "--debug-outputs",
        help="comma-separated tensors to mark as outputs (a debug engine for per-layer "
        "parity; never benchmarked)",
    )
    p.add_argument("--out-root", help="engine root (default: artifacts/engines)")
    p.add_argument("--out", help="report path")
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("perf", help="latency distribution per stage and throughput; "
                                    "writes a JSON report")  # fmt: skip
    p.add_argument("--model", required=True, help="config name or path")
    p.add_argument("--backend", default="ort_cpu")
    p.add_argument("--artifact", help="model file for the backend")
    p.add_argument("--perf-config", help="probe settings (default: configs/perf.yaml)")
    p.add_argument("--manifest", help="frames to cycle through (default: from the perf config)")
    p.add_argument(
        "--synthetic",
        metavar="HxW",
        help="random frames of this size instead of dataset images (no dataset needed)",
    )
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--warmup", type=int, help="override warmup_iterations")
    p.add_argument("--iterations", type=int, help="override iterations per repeat")
    p.add_argument("--repeats", type=int, help="override repeats")
    p.add_argument("--max-images", type=int, help="override max_images")
    p.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="bracket warmup and measurement with cudaProfilerStart/Stop, so "
        "`nsys profile --capture-range=cudaProfilerApi` leaves out engine loading (trt only)",
    )
    p.add_argument("--out", help="report path (default: artifacts/reports/...)")
    p.set_defaults(func=cmd_perf)

    return parser


# Exit codes: 0 checks passed, 1 checks failed (report written), 2 usage or config error,
# 3 the tool crashed (traceback on stderr, no report). Scripts treat 1 as a result to read
# and anything else as a reason to stop; Python's own exit code for an uncaught
# exception is 1, which would read as a failed check.
EXIT_ENVIRONMENT = 2  # same code as usage/configuration errors (docs/report-codes.md)
EXIT_CRASH = 3


def main(argv: list[str] | None = None) -> int:
    from edgeinfer.backends.trt_meta import BackendError

    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except BackendError as e:
        # The backend can't load or run here (no TensorRT, an engine for another runtime):
        # an environment error with a report code, not a crash.
        print(f"error: {e.code}: {e}")
        return EXIT_ENVIRONMENT
    except Exception:  # noqa: BLE001 - top level: re-reported, not silenced
        traceback.print_exc()
        return EXIT_CRASH


if __name__ == "__main__":
    raise SystemExit(main())
