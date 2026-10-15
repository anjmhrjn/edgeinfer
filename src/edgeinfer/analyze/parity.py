"""Parity analyzer: compare a backend's outputs with the FP32 golden outputs.

Whole-model mode compares logits per golden image. Three input modes:
    images   preprocess the raw image with the given config (end to end). The fresh
             input is also compared to the golden input, so a preprocessing fault is
             reported as such, with a diagnosis of which kind.
    golden   feed the stored golden input (model only).
    npz      logits produced elsewhere (e.g. the C++ runtime), one .npz per image.

Per-layer mode compares intermediate tensors by ONNX name between a reference backend
(ORT FP32 on the FP32 ONNX, D-012) and the candidate, and reports the first tensor in
graph order whose drift exceeds tolerance.

Read-only: inputs are never modified. Thresholds come from configs/tolerances.yaml.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from edgeinfer.backends.base import Backend
from edgeinfer.config import CONFIG_DIR, ModelConfig
from edgeinfer.data.cityscapes import Sample
from edgeinfer.data.manifests import ManifestEntry
from edgeinfer.eval.miou import ConfusionMatrix
from edgeinfer.postprocess import postprocess
from edgeinfer.preprocess import preprocess

TOLERANCES_PATH = CONFIG_DIR / "tolerances.yaml"
MODES = ("images", "golden", "npz")
TOP_K = 10

Finding = dict[str, str]


@dataclass(frozen=True)
class Tolerances:
    """None = reported but not gated (e.g. elementwise drift under INT8, D-018)."""

    input_max_abs: float | None  # preprocessed input vs golden input (images mode)
    logits_max_abs: float | None
    logits_cosine_min: float | None
    argmax_agreement_min: float | None  # fraction of stride-4 pixels with the same class
    miou_delta_max: float | None  # |candidate - reference| in mIoU points, golden images
    layer_cosine_min: float | None
    layer_rel_max_abs: float | None  # max|a-b| / max|ref| per tensor


def load_tolerances(precision: str, path: Path = TOLERANCES_PATH) -> Tolerances:
    doc = yaml.safe_load(path.read_text())
    if precision not in doc:
        raise KeyError(f"{path.name}: no tolerances for precision {precision!r}")
    d = doc[precision]
    names = {f.name for f in fields(Tolerances)}
    if set(d) != names:
        raise ValueError(
            f"{path.name}[{precision}]: expected keys {sorted(names)}, got {sorted(d)}"
        )
    return Tolerances(**{k: None if v is None else float(v) for k, v in d.items()})


def _f(severity: str, code: str, message: str, where: str = "") -> Finding:
    out = {"severity": severity, "code": code, "message": message}
    if where:
        out["where"] = where
    return out


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a64, b64 = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a64), np.linalg.norm(b64)
    if na == 0 or nb == 0:
        return 1.0 if na == nb else 0.0
    return float(np.dot(a64, b64) / (na * nb))


def compare_logits(ref: np.ndarray, cand: np.ndarray) -> dict[str, Any]:
    """Elementwise and class-level agreement of two 1xCxhxw logit tensors."""
    if ref.shape != cand.shape:
        raise ValueError(f"shape mismatch: reference {ref.shape}, candidate {cand.shape}")
    nonfinite = int((~np.isfinite(cand)).sum())
    if nonfinite:
        return {"nonfinite": nonfinite, "max_abs": None, "mean_abs": None, "cosine": None,
                "argmax_agreement": float((ref.argmax(1) == cand.argmax(1)).mean())}  # fmt: skip
    diff = np.abs(cand.astype(np.float64) - ref.astype(np.float64))
    return {
        "nonfinite": 0,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "cosine": _cosine(ref, cand),
        "argmax_agreement": float((ref.argmax(1) == cand.argmax(1)).mean()),
    }


def diagnose_input(x: np.ndarray, golden: np.ndarray) -> str:
    """Name the likely preprocessing fault behind an input mismatch.

    Per-channel correlation is unaffected by per-channel scale and offset, so it
    separates channel order (a permutation), normalization (same content, different
    affine map) and everything else (resize, or a different image)."""
    xs = x[0].reshape(3, -1).astype(np.float64)
    gs = golden[0].reshape(3, -1).astype(np.float64)
    corr = np.corrcoef(np.concatenate([xs, gs]))[:3, 3:]  # corr[i, j]: x channel i vs golden j
    perm = [int(j) for j in corr.argmax(axis=1)]
    matched = all(corr[i, perm[i]] > 0.99999 for i in range(3))
    if matched and perm == [2, 1, 0]:
        return "channels are reversed relative to the golden input: check input.channel_order"
    if matched and perm == [0, 1, 2]:
        return ("same image content with a different per-channel scale/offset: check "
                "preprocess.scale, mean and std")  # fmt: skip
    if matched:
        return f"channels are permuted ({perm}) relative to the golden input"
    return "spatial content differs: check input size, resize, or that it is the same image"


def _agg(values: list[float | None], fn: Callable[[list[float]], float]) -> float | None:
    vals = [v for v in values if v is not None]
    return fn(vals) if vals else None


def run_parity(
    cfg: ModelConfig,
    entries: list[ManifestEntry],
    golden_dir: Path,
    tol: Tolerances,
    mode: str,
    *,
    backend: Backend | None = None,
    samples: Iterable[Sample] | None = None,
    npz_dir: Path | None = None,
) -> tuple[dict[str, Any], list[Finding], list[tuple[np.ndarray, np.ndarray]]]:
    """Whole-model parity. Returns (metrics, findings, input pairs).

    `samples` (image + label per entry, in order) are required in images mode and
    optional otherwise; with them, mIoU on the golden images is compared too. The
    returned (golden input, candidate input) pairs feed per-layer mode."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if mode in ("images", "golden") and backend is None:
        raise ValueError(f"mode {mode!r} needs a backend")
    if mode == "npz" and npz_dir is None:
        raise ValueError("mode 'npz' needs npz_dir")
    if mode == "images" and samples is None:
        raise ValueError("mode 'images' needs the dataset (samples)")
    sample_iter = iter(samples) if samples is not None else None

    per_image: list[dict[str, Any]] = []
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    cm_ref = ConfusionMatrix(cfg.num_classes)
    cm_cand = ConfusionMatrix(cfg.num_classes)
    label_agree: list[float] = []
    input_bad: list[tuple[str, float, str]] = []
    for entry in entries:
        path = golden_dir / f"{entry.image_id}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"golden output missing: {path} (run `edgeinfer golden`)")
        with np.load(path) as z:
            gx, gl = z["input"], z["logits"]
        sample = next(sample_iter) if sample_iter is not None else None
        if sample is not None and sample.entry != entry:
            raise ValueError(f"samples out of order: {sample.entry} vs {entry}")
        row: dict[str, Any] = {"image_id": entry.image_id}
        if mode == "images":
            assert sample is not None and backend is not None
            x = preprocess(sample.image, cfg)
            if x.shape != gx.shape:
                in_diff = math.inf
            else:
                in_diff = float(np.abs(x.astype(np.float64) - gx).max())
            row["input_max_abs"] = None if math.isinf(in_diff) else in_diff
            if tol.input_max_abs is not None and in_diff > tol.input_max_abs:
                why = (f"shape {x.shape} vs golden {gx.shape}" if math.isinf(in_diff)
                       else diagnose_input(x, gx))  # fmt: skip
                input_bad.append((entry.image_id, in_diff, why))
            cand = backend.infer(x)
            pairs.append((gx, x))
        elif mode == "golden":
            assert backend is not None
            cand = backend.infer(gx)
            pairs.append((gx, gx))
        else:
            assert npz_dir is not None
            cand_path = npz_dir / f"{entry.image_id}.npz"
            if not cand_path.is_file():
                raise FileNotFoundError(f"candidate output missing: {cand_path}")
            with np.load(cand_path) as z:
                if "logits" not in z:
                    raise KeyError(f"{cand_path}: expected an array named 'logits'")
                cand = z["logits"]
        row.update(compare_logits(gl, cand))
        if sample is not None:
            h, w = sample.label.shape
            ref_pred, cand_pred = postprocess(gl, h, w), postprocess(cand, h, w)
            cm_ref.update(ref_pred, sample.label)
            cm_cand.update(cand_pred, sample.label)
            label_agree.append(float((ref_pred == cand_pred).mean()))
            row["argmax_agreement_label_res"] = label_agree[-1]
        per_image.append(row)
    if not per_image:
        raise ValueError("no images compared")

    def worst(key: str, fn) -> tuple[float | None, str]:
        rows = [r for r in per_image if r.get(key) is not None]
        if not rows:
            return None, ""
        r = fn(rows, key=lambda r: r[key])
        return r[key], r["image_id"]

    max_abs, max_abs_at = worst("max_abs", max)
    cos_min, cos_at = worst("cosine", min)
    agree_min, agree_at = worst("argmax_agreement", min)
    metrics: dict[str, Any] = {
        "n_images": len(per_image),
        "mode": mode,
        "logits_max_abs": max_abs,
        "logits_mean_abs": _agg([r["mean_abs"] for r in per_image], lambda v: sum(v) / len(v)),
        "logits_cosine_min": cos_min,
        "argmax_agreement_min": agree_min,
        "argmax_agreement_mean": sum(r["argmax_agreement"] for r in per_image) / len(per_image),
    }
    findings: list[Finding] = []

    if mode == "images":
        metrics["input_max_abs"] = _agg([r["input_max_abs"] for r in per_image], max)
        if input_bad:
            image_id, diff, why = max(input_bad, key=lambda t: t[1])
            findings.append(_f(
                "error", "PARITY_INPUT_MISMATCH",
                f"preprocessed input differs from the golden input on {len(input_bad)}/"
                f"{len(per_image)} images (max abs {diff:.4g} > {tol.input_max_abs:g}); {why}",
                image_id,
            ))  # fmt: skip

    nonfinite = [r for r in per_image if r["nonfinite"]]
    if nonfinite:
        findings.append(_f(
            "error", "PARITY_OUTPUT_NONFINITE",
            f"candidate logits contain NaN/Inf on {len(nonfinite)}/{len(per_image)} images "
            f"({nonfinite[0]['nonfinite']} values on the first)", nonfinite[0]["image_id"],
        ))  # fmt: skip
    if None not in (max_abs, tol.logits_max_abs) and max_abs > tol.logits_max_abs:
        findings.append(_f(
            "error", "PARITY_OUTPUT_DRIFT",
            f"logits max abs error {max_abs:.4g} > {tol.logits_max_abs:g}", max_abs_at,
        ))  # fmt: skip
    if None not in (cos_min, tol.logits_cosine_min) and cos_min < tol.logits_cosine_min:
        findings.append(_f(
            "error", "PARITY_OUTPUT_DRIFT",
            f"logits cosine similarity {cos_min:.8f} < {tol.logits_cosine_min}", cos_at,
        ))  # fmt: skip
    if None not in (agree_min, tol.argmax_agreement_min) and (agree_min < tol.argmax_agreement_min):
        findings.append(_f(
            "error", "PARITY_ARGMAX_DISAGREEMENT",
            f"argmax agreement {agree_min:.6f} < {tol.argmax_agreement_min} "
            f"({(1 - agree_min) * 100:.3f}% of stride-{cfg.output_stride} pixels changed class)",
            agree_at,
        ))  # fmt: skip

    if label_agree:
        miou_ref, miou_cand = cm_ref.miou(), cm_cand.miou()
        delta = (miou_cand - miou_ref) * 100
        metrics.update({
            "argmax_agreement_label_res_min": min(label_agree),
            "miou_reference": miou_ref,
            "miou_candidate": miou_cand,
            "miou_delta_points": delta,
        })  # fmt: skip
        if tol.miou_delta_max is not None and abs(delta) > tol.miou_delta_max:
            findings.append(_f(
                "error", "PARITY_MIOU_DELTA",
                f"mIoU on {len(per_image)} golden images changed by {delta:+.3f} points "
                f"({miou_ref * 100:.2f} -> {miou_cand * 100:.2f}); limit ±{tol.miou_delta_max:g}",
                "miou",
            ))  # fmt: skip
    else:
        findings.append(_f(
            "info", "PARITY_MIOU_SKIPPED",
            "no labels available (dataset root not given), so the mIoU delta wasn't computed",
        ))  # fmt: skip
    metrics["per_image"] = per_image
    return metrics, findings, pairs


def run_layer_parity(
    ref: Backend,
    cand: Backend,
    names: list[tuple[str, str]],
    groups: list[list[str]],
    pairs: list[tuple[np.ndarray, np.ndarray]],
    tol: Tolerances,
) -> tuple[dict[str, Any], list[Finding]]:
    """Per-layer parity over `names` ((tensor, op type) in graph order), fetched in
    memory-bounded `groups`. Each pair is (reference input, candidate input)."""
    if not pairs:
        raise ValueError("no inputs for per-layer parity")
    stats: dict[str, dict[str, float]] = {n: {"cosine_min": 1.0, "rel_max_abs": 0.0}
                                          for n, _ in names}  # fmt: skip
    for rx, cx in pairs:
        for group in groups:
            r_out = ref.layer_outputs(rx, group)
            c_out = cand.layer_outputs(cx, group)
            for name in group:
                r, c = r_out[name], c_out[name]
                scale = float(np.abs(r).max())
                diff = float(np.abs(c.astype(np.float64) - r).max())
                rel = diff / scale if scale > 0 else (0.0 if diff == 0 else math.inf)
                s = stats[name]
                s["cosine_min"] = min(s["cosine_min"], _cosine(r, c))
                s["rel_max_abs"] = max(s["rel_max_abs"], rel)
            del r_out, c_out

    rows = [{"index": i, "tensor": n, "op": op, **stats[n]} for i, (n, op) in enumerate(names)]
    for row in rows:
        if math.isinf(row["rel_max_abs"]):
            row["rel_max_abs"] = None  # JSON has no inf; means reference was all zeros

    def over(row: dict[str, Any]) -> bool:
        if tol.layer_cosine_min is not None and row["cosine_min"] < tol.layer_cosine_min:
            return True
        if tol.layer_rel_max_abs is None:
            return False
        return row["rel_max_abs"] is None or row["rel_max_abs"] > tol.layer_rel_max_abs

    failing = [r for r in rows if over(r)]
    worst = sorted(rows, key=lambda r: r["cosine_min"])[:TOP_K]
    metrics = {
        "n_tensors": len(rows),
        "n_images": len(pairs),
        "n_over_tolerance": len(failing),
        "first_over_tolerance": failing[0] if failing else None,
        "worst_by_cosine": worst,
        "tensors": rows,
    }
    findings: list[Finding] = []
    if failing:
        first = failing[0]
        rel = "n/a" if first["rel_max_abs"] is None else f"{first['rel_max_abs']:.4g}"
        findings.append(_f(
            "error", "PARITY_LAYER_DRIFT",
            f"first tensor over tolerance (#{first['index']} of {len(rows)} in graph order): "
            f"{first['op']} output {first['tensor']!r}, cosine {first['cosine_min']:.6f}, "
            f"rel max abs {rel}; {len(failing)} tensors over tolerance in total",
            first["tensor"],
        ))  # fmt: skip
    return metrics, findings


def tolerances_dict(tol: Tolerances) -> dict[str, float]:
    return asdict(tol)
