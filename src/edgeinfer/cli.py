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
    for spec in MANIFEST_SPECS.values():
        if spec.split not in ids_by_split:
            ids_by_split[spec.split] = list_image_ids(root, spec.split)
        text = format_manifest(build_manifest(spec, ids_by_split[spec.split]))
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
    backend.load(None, cfg)
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
    p.add_argument("--manifest", default="eval_val")
    p.add_argument("--root", help="Cityscapes root (default: $CITYSCAPES_ROOT)")
    p.add_argument("--height", type=int, help="override the config input height")
    p.add_argument("--width", type=int, help="override the config input width")
    p.add_argument("--limit", type=int, help="first N images only (smoke runs)")
    p.add_argument("--out", help="report path (default: artifacts/reports/...)")
    p.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
