"""Sensitivity configurations generated ahead of a TensorRT run (D-032).

On TensorRT, sensitivity is re-run for the top-N groups of the ORT ranking (D-017) only.
The configurations are quantized and converted to INT8+FP16 on the Mac, because the
FP16 conversion runs in the ModelOpt environment (D-031) and GPU sessions are for running,
not preparing. Every subset of the N groups is generated (2^N models, a few seconds each),
so the greedy search can follow whatever order TensorRT's leave-one-out gives without a
trip back to the Mac.

The list is written as a `candidates` report. The GPU side reads it, looks a configuration
up by its set of excluded groups, and checks that the file it finds is the artifact the
report names.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from pathlib import Path
from typing import Any

from edgeinfer.artifacts import ONNX_DIR, load_sidecar
from edgeinfer.config import ModelConfig
from edgeinfer.quant import QuantError
from edgeinfer.quant.config import Exclusions, GroupRules, QuantConfig
from edgeinfer.quant.mixed import convert_int8_fp16
from edgeinfer.quant.ptq import quantize
from edgeinfer.validate.onnx_validator import validate_onnx

PRECISIONS = ("int8", "int8_fp16")
Convert = Callable[[Path, ModelConfig, Path], tuple[Path, dict[str, Any]]]


class CandidateError(ValueError):
    """A configuration can't be found or doesn't match. `code` is the finding code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def top_groups(ranking_report: dict[str, Any], model: str, n: int) -> list[str]:
    """The first n groups of an ORT sensitivity report's ranking (largest recovery first)."""
    if ranking_report.get("tool") != "sensitivity" or ranking_report.get("model") != model:
        raise CandidateError(
            "SENS_RANKING_MISMATCH",
            f"the ranking must come from a sensitivity report for {model}, got tool "
            f"{ranking_report.get('tool')!r} for {ranking_report.get('model')!r}",
        )
    ranking = ranking_report["metrics"]["ranking"]
    if not 0 < n <= len(ranking):
        raise CandidateError("SENS_RANKING_MISMATCH", f"--top {n}: the ranking has {len(ranking)}")
    return list(ranking[:n])


def subsets(groups: list[str]) -> list[tuple[str, ...]]:
    """Every subset of `groups`, smallest first, each in `groups` order."""
    return [c for k in range(len(groups) + 1) for c in itertools.combinations(groups, k)]


def _entry(path: Path, meta: dict[str, Any]) -> dict[str, Any]:
    return {"file": path.name, "artifact_id": meta["artifact_id"]}


def generate(
    fp32_path: Path,
    cfg: ModelConfig,
    qcfg: QuantConfig,
    calibration: tuple[Path, dict[str, Any]],
    rules: GroupRules,
    groups: list[str],
    out_root: Path = ONNX_DIR,
    convert: Convert = convert_int8_fp16,
) -> list[dict[str, Any]]:
    """Quantize every subset of `groups` kept in float, convert each to INT8+FP16 and
    validate it. Existing artifacts with the same ids are reused."""
    unknown = sorted(set(groups) - set(rules.names))
    if unknown:
        raise QuantError("QUANT_UNKNOWN_EXCLUSION", f"unknown groups {unknown}")
    configs = []
    for excluded in subsets(groups):
        exc = Exclusions("sensitivity", cfg.name, excluded, ())
        int8_path, int8_meta = quantize(fp32_path, cfg, qcfg, calibration, rules, exc, out_root)
        path, meta = convert(int8_path, cfg, out_root)
        _, findings = validate_onnx(path, cfg)
        errors = [f for f in findings if f["severity"] == "error"]
        if errors:
            raise CandidateError(
                "SENS_CANDIDATE_INVALID",
                f"{path.name} (excluded {list(excluded)}) fails validation: "
                f"{[f['code'] for f in errors]}",
            )
        configs.append({
            "excluded": list(excluded),
            "int8": _entry(int8_path, int8_meta),
            "int8_fp16": _entry(path, meta),
            "scale_rounding": meta["scale_rounding"],
        })  # fmt: skip
        print(f"  [{len(configs)}] excluded {list(excluded) or 'nothing'}: {path.name}")
    return configs


class CandidateSet:
    """The configurations of a `candidates` report, looked up by excluded groups."""

    def __init__(self, report: dict[str, Any], onnx_root: Path = ONNX_DIR) -> None:
        if report.get("tool") != "candidates":
            raise CandidateError(
                "SENS_RANKING_MISMATCH", f"not a candidates report (tool {report.get('tool')!r})"
            )
        self.model = report["model"]
        self.groups: list[str] = report["metrics"]["groups"]
        self.dir = onnx_root / self.model
        self._by_key = {frozenset(c["excluded"]): c for c in report["metrics"]["configs"]}

    def path(self, excluded: tuple[str, ...], precision: str = "int8_fp16") -> Path:
        """The ONNX file of a configuration; refuses anything the report didn't list or a
        file that isn't the listed artifact."""
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        config = self._by_key.get(frozenset(excluded))
        if config is None:
            raise CandidateError(
                "SENS_CANDIDATE_MISSING",
                f"no generated configuration keeps {sorted(excluded)} in float; generate it on "
                f"the Mac (`edgeinfer candidates`) for groups {self.groups}",
            )
        entry = config[precision]
        path = self.dir / entry["file"]
        if not path.is_file():
            raise CandidateError(
                "SENS_CANDIDATE_MISSING",
                f"{path} is listed but not on this machine; push it (scripts/c5_runs.sh push)",
            )
        found = load_sidecar(path)["artifact_id"]
        if found != entry["artifact_id"]:
            raise CandidateError(
                "SENS_CANDIDATE_MISMATCH",
                f"{path.name} is artifact {found[:12]}, the report lists "
                f"{entry['artifact_id'][:12]}",
            )
        return path
