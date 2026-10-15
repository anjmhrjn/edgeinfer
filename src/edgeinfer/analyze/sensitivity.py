"""Sensitivity analyzer: which layer groups cost accuracy under INT8, and which to keep
in float to meet the accuracy budget (D-017).

Every configuration is a real quantized model, built with the same calibration cache
and exclusion mechanism as a deployable one (quant/ptq.py), and scored on a fixed
image subset against the FP32 model on the same images:

    full INT8        every group quantized
    only:<g>         only group g quantized (every other group excluded)
    except:<g>       every group but g quantized (leave-one-out)
    greedy:<k>       the k groups with the largest leave-one-out recovery excluded

Execution noise floor: the full INT8 model scored again with a different ORT thread
count. Quantization turns tiny FP32 rounding differences into whole quantization steps,
so mIoU moves a little even when nothing meaningful changed; group effects within this
floor aren't distinguishable from noise.

Greedy selection excludes groups in leave-one-out order and scores each cumulative
configuration for real, without assuming the effects add up, until the drop is within
budget - margin. The margin covers the gap between the scoring subset and full val.

Read-only: inputs are never modified; quantized models are new content-addressed files.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from edgeinfer.backends import get_backend
from edgeinfer.config import ModelConfig
from edgeinfer.data.cityscapes import Sample
from edgeinfer.eval.miou import ConfusionMatrix
from edgeinfer.postprocess import postprocess
from edgeinfer.preprocess import preprocess
from edgeinfer.quant.config import Exclusions, GroupRules, QuantConfig
from edgeinfer.quant.ptq import quantize

Finding = dict[str, str]
MODES = ("only", "except")
# Greedy steps whose measured delta differs from the sum of single-group recoveries by
# more than this (or 2x the noise floor, if larger) are flagged as non-additive.
NON_ADDITIVE_MIN_POINTS = 0.1


@dataclass(frozen=True)
class Score:
    miou: float
    delta_points: float  # (candidate - FP32) mIoU points on the same images; < 0 is a drop
    argmax_agreement: float  # mean over images, native-stride pixels, vs FP32
    artifact_id: str


def _f(severity: str, code: str, message: str, where: str = "") -> Finding:
    out = {"severity": severity, "code": code, "message": message}
    if where:
        out["where"] = where
    return out


def run_sensitivity(
    groups: list[str],
    score: Callable[[tuple[str, ...]], Score],
    *,
    full: Score,
    noise_floor: float,
    budget: float,
    margin: float,
    modes: tuple[str, ...] = MODES,
) -> tuple[dict[str, Any], list[Finding]]:
    """Rank `groups` and suggest an exclusion list. `score(excluded_groups)` returns the
    score of the model with those groups kept in float."""
    if not groups:
        raise ValueError("no groups")
    unknown = set(modes) - set(MODES)
    if unknown:
        raise ValueError(f"unknown modes {sorted(unknown)} (available: {MODES})")
    target = budget - margin  # max allowed drop on the scoring subset, in points
    rows: list[dict[str, Any]] = []
    for i, g in enumerate(groups):
        row: dict[str, Any] = {"group": g}
        if "only" in modes:
            s = score(tuple(x for x in groups if x != g))
            row |= {"only_delta": s.delta_points, "only_agreement": s.argmax_agreement}
        if "except" in modes:
            s = score((g,))
            row |= {"except_delta": s.delta_points, "except_agreement": s.argmax_agreement,
                    "recovery": s.delta_points - full.delta_points}  # fmt: skip
        effect = row.get("recovery", -row.get("only_delta", 0.0))
        row["distinguishable"] = abs(effect) > noise_floor
        rows.append(row)
        print(f"  [{i + 1}/{len(groups)}] {g}: " + ", ".join(
            f"{k} {v:+.3f}" for k, v in row.items() if k.endswith(("delta", "recovery"))
        ), file=sys.stderr)  # fmt: skip

    def key(r: dict[str, Any]) -> tuple[float, str]:
        return (-r.get("recovery", -r.get("only_delta", 0.0)), r["group"])

    ranking = [r["group"] for r in sorted(rows, key=key)]
    findings: list[Finding] = []
    steps: list[dict[str, Any]] = []
    suggestion: list[str] = []
    final = full
    if -full.delta_points <= target:
        findings.append(_f(
            "info", "SENS_FULL_INT8_WITHIN_BUDGET",
            f"full INT8 drops {-full.delta_points:.3f} points on the scoring set, within "
            f"{target:g} (budget {budget:g} - margin {margin:g}); nothing needs to stay in float",
        ))  # fmt: skip
    else:
        recovery = {r["group"]: r.get("recovery", -r.get("only_delta", 0.0)) for r in rows}
        for g in ranking:
            suggestion.append(g)
            s = score(tuple(suggestion))
            predicted = full.delta_points + sum(recovery[x] for x in suggestion)
            steps.append({"excluded": list(suggestion), "delta": s.delta_points,
                          "predicted_additive": predicted, "agreement": s.argmax_agreement,
                          "artifact_id": s.artifact_id})  # fmt: skip
            final = s
            print(f"  greedy {len(suggestion)}: {s.delta_points:+.3f}", file=sys.stderr)
            if -s.delta_points <= target:
                break
        if -final.delta_points <= target:
            findings.append(_f(
                "info", "SENS_EXCLUSION_SUGGESTED",
                f"keeping {len(suggestion)}/{len(groups)} groups in float brings the drop "
                f"from {-full.delta_points:.3f} to {-final.delta_points:.3f} points "
                f"(target {target:g}): {suggestion}", ",".join(suggestion),
            ))  # fmt: skip
        else:
            findings.append(_f(
                "error", "SENS_BUDGET_UNREACHABLE",
                f"even with every group in float the drop is {-final.delta_points:.3f} points "
                f"> target {target:g}; the loss isn't in the grouped (quantizable) layers",
            ))  # fmt: skip
        last = steps[-1]
        gap = abs(last["delta"] - last["predicted_additive"])
        if gap > max(2 * noise_floor, NON_ADDITIVE_MIN_POINTS):
            findings.append(_f(
                "warn", "SENS_NON_ADDITIVE",
                f"excluding {len(suggestion)} groups gives {last['delta']:+.3f} points; the sum "
                f"of their single-group recoveries predicted {last['predicted_additive']:+.3f}",
            ))  # fmt: skip

    metrics = {
        "full_int8": asdict(full),
        "noise_floor_points": noise_floor,
        "groups": rows,
        "ranking": ranking,
        "n_distinguishable": sum(r["distinguishable"] for r in rows),
        "greedy": steps,
        "suggested_exclusions": suggestion,
        "suggested_delta_points": final.delta_points,
    }
    return metrics, findings


@dataclass
class _Cached:
    x: np.ndarray
    label: np.ndarray


class SubsetScorer:
    """Quantize-and-score on a fixed image subset, with inputs preprocessed once and the
    FP32 reference computed once."""

    def __init__(
        self,
        cfg: ModelConfig,
        fp32_path: Path,
        qcfg: QuantConfig,
        calibration: tuple[Path, dict[str, Any]],
        rules: GroupRules,
        samples: Iterable[Sample],
        out_root: Path | None = None,
    ) -> None:
        self.cfg, self.fp32_path, self.qcfg = cfg, fp32_path, qcfg
        self.calibration, self.rules = calibration, rules
        self.out_root = out_root
        self.data = [_Cached(preprocess(s.image, cfg), s.label) for s in samples]
        if not self.data:
            raise ValueError("no images to score on")
        ref = get_backend("ort_cpu")
        ref.load(fp32_path, cfg)
        self.fp32_argmax: list[np.ndarray] = []
        cm = ConfusionMatrix(cfg.num_classes)
        for d in self.data:
            logits = ref.infer(d.x)
            self.fp32_argmax.append(logits.argmax(1))
            cm.update(postprocess(logits, *d.label.shape), d.label)
        self.fp32_miou = cm.miou()
        self.fp32_artifact_id = ref.metadata()["artifact_id"]

    def build(self, excluded: tuple[str, ...]) -> tuple[Path, dict[str, Any]]:
        exc = Exclusions("sensitivity", self.cfg.name, tuple(excluded), ())
        kwargs = {"out_root": self.out_root} if self.out_root else {}
        return quantize(self.fp32_path, self.cfg, self.qcfg, self.calibration, self.rules,
                        exc, **kwargs)  # fmt: skip

    def score(self, excluded: tuple[str, ...], threads: int | None = None) -> Score:
        path, meta = self.build(excluded)
        backend = get_backend("ort_cpu", **({"threads": threads} if threads else {}))
        backend.load(path, self.cfg)
        cm = ConfusionMatrix(self.cfg.num_classes)
        agree = []
        for d, ref_argmax in zip(self.data, self.fp32_argmax, strict=True):
            logits = backend.infer(d.x)
            agree.append(float((logits.argmax(1) == ref_argmax).mean()))
            cm.update(postprocess(logits, *d.label.shape), d.label)
        miou = cm.miou()
        return Score(miou, (miou - self.fp32_miou) * 100, float(np.mean(agree)),
                     meta["artifact_id"])  # fmt: skip
