"""Quantization configs (configs/quant/), loaded strictly like model configs.

    configs/quant/<name>.yaml                    how to quantize (format, dtypes, calibration)
    configs/quant/groups/<model>.yaml            node -> layer group rules (sensitivity, exclusions)
    configs/quant/exclusions/<model>__<x>.yaml   groups/nodes kept in float

Everything that changes the quantized model lives in these files, so a fault (a wrong
dtype, a swapped calibration manifest, an exclusion list) can be injected by config alone.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

from edgeinfer.config import CONFIG_DIR, ConfigError, _choice, _take

QUANT_DIR = CONFIG_DIR / "quant"

QTYPES = ("int8", "uint8")
METHODS = ("minmax", "entropy", "percentile")
OP_TYPES = ("Conv", "MatMul", "Gemm")
# Histogram parameters each calibration method takes (D-016, D-019).
_METHOD_PARAMS = {
    "minmax": set(),
    "entropy": {"num_bins", "num_quantized_bins"},
    "percentile": {"num_bins", "percentile"},
}


@dataclass(frozen=True)
class CalibrationSpec:
    manifest: str  # manifest name or path; must be from the train split
    limit: int | None  # first N images of the manifest (calibration-size study)
    method: str
    num_bins: int | None
    num_quantized_bins: int | None
    percentile: float | None
    chunk_images: int  # images per collection pass; bounds calibration memory

    @classmethod
    def from_dict(cls, d: Any, where: str) -> CalibrationSpec:
        d = _take(cls, d, where)
        method = _choice(d["method"], METHODS, f"{where}.method")
        given = {k for k in ("num_bins", "num_quantized_bins", "percentile") if d[k] is not None}
        if given != _METHOD_PARAMS[method]:
            raise ConfigError(
                f"{where}: method {method!r} takes {sorted(_METHOD_PARAMS[method]) or 'no'} "
                f"histogram parameters, got {sorted(given) or 'none'} (set the others to null)"
            )
        for key in ("limit", "num_bins", "num_quantized_bins", "chunk_images"):
            if d[key] is not None and (not isinstance(d[key], int) or d[key] <= 0):
                raise ConfigError(f"{where}.{key}: expected a positive int, got {d[key]!r}")
        if d["percentile"] is not None and not 50 < d["percentile"] <= 100:
            raise ConfigError(f"{where}.percentile: expected (50, 100], got {d['percentile']!r}")
        return cls(
            manifest=str(d["manifest"]),
            limit=d["limit"],
            method=method,
            num_bins=d["num_bins"],
            num_quantized_bins=d["num_quantized_bins"],
            percentile=None if d["percentile"] is None else float(d["percentile"]),
            chunk_images=d["chunk_images"],
        )


@dataclass(frozen=True)
class QuantConfig:
    name: str
    activation_type: str
    weight_type: str
    activation_symmetric: bool
    weight_symmetric: bool
    per_channel: bool
    op_types: tuple[str, ...]
    quantize_outputs: bool
    quantize_bias: bool
    calibration: CalibrationSpec

    @classmethod
    def from_dict(cls, d: Any, where: str = "quant config") -> QuantConfig:
        d = _take(cls, d, where)
        for key in ("activation_symmetric", "weight_symmetric", "per_channel",
                    "quantize_outputs", "quantize_bias"):  # fmt: skip
            if not isinstance(d[key], bool):
                raise ConfigError(f"{where}.{key}: expected true/false, got {d[key]!r}")
        ops = d["op_types"]
        if not isinstance(ops, list) or not ops:
            raise ConfigError(f"{where}.op_types: expected a non-empty list")
        return cls(
            name=str(d["name"]),
            activation_type=_choice(d["activation_type"], QTYPES, f"{where}.activation_type"),
            weight_type=_choice(d["weight_type"], QTYPES, f"{where}.weight_type"),
            activation_symmetric=d["activation_symmetric"],
            weight_symmetric=d["weight_symmetric"],
            per_channel=d["per_channel"],
            op_types=tuple(_choice(o, OP_TYPES, f"{where}.op_types") for o in ops),
            quantize_outputs=d["quantize_outputs"],
            quantize_bias=d["quantize_bias"],
            calibration=CalibrationSpec.from_dict(d["calibration"], f"{where}.calibration"),
        )

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["op_types"] = list(self.op_types)
        return d


def _path(name_or_path: str | Path, subdir: Path) -> Path:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = subdir / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return path


def load_quant_config(name_or_path: str | Path) -> QuantConfig:
    path = _path(name_or_path, QUANT_DIR)
    return QuantConfig.from_dict(yaml.safe_load(path.read_text()), where=path.name)


@dataclass(frozen=True)
class GroupRule:
    name: str
    match: re.Pattern[str]  # full match against a node's innermost module scope


@dataclass(frozen=True)
class GroupRules:
    model: str
    rules: tuple[GroupRule, ...]

    @classmethod
    def from_dict(cls, d: Any, where: str) -> GroupRules:
        d = _take(cls, d, where)
        if not isinstance(d["rules"], list) or not d["rules"]:
            raise ConfigError(f"{where}.rules: expected a non-empty list")
        rules = []
        for i, r in enumerate(d["rules"]):
            r = _take(GroupRule, r, f"{where}.rules[{i}]")
            try:
                pattern = re.compile(r["match"])
            except re.error as e:
                raise ConfigError(f"{where}.rules[{i}].match: {e}") from e
            rules.append(GroupRule(str(r["name"]), pattern))
        names = [r.name for r in rules]
        dupes = sorted({n for n in names if names.count(n) > 1})
        if dupes:
            raise ConfigError(f"{where}: duplicate group names {dupes}")
        return cls(model=str(d["model"]), rules=tuple(rules))

    @property
    def names(self) -> list[str]:
        return [r.name for r in self.rules]

    def group_of(self, scope: str) -> str | None:
        """First rule whose pattern fully matches `scope`."""
        return next((r.name for r in self.rules if r.match.fullmatch(scope)), None)


def load_group_rules(model: str, path: str | Path | None = None) -> GroupRules:
    p = _path(path or model, QUANT_DIR / "groups")
    rules = GroupRules.from_dict(yaml.safe_load(p.read_text()), where=p.name)
    if rules.model != model:
        raise ConfigError(f"{p.name}: rules are for model {rules.model!r}, not {model!r}")
    return rules


@dataclass(frozen=True)
class Exclusions:
    """Layer groups and/or individual nodes to keep in float."""

    name: str
    model: str
    groups: tuple[str, ...]
    nodes: tuple[str, ...]

    @classmethod
    def from_dict(cls, d: Any, where: str) -> Exclusions:
        d = _take(cls, d, where)
        for key in ("groups", "nodes"):
            if not isinstance(d[key], list) or not all(isinstance(v, str) for v in d[key]):
                raise ConfigError(f"{where}.{key}: expected a list of names")
        return cls(str(d["name"]), str(d["model"]), tuple(d["groups"]), tuple(d["nodes"]))

    def to_dict(self) -> dict[str, Any]:
        return {f.name: list(v) if isinstance(v := getattr(self, f.name), tuple) else v
                for f in fields(self)}  # fmt: skip

    @property
    def empty(self) -> bool:
        return not self.groups and not self.nodes


def no_exclusions(model: str) -> Exclusions:
    return Exclusions("none", model, (), ())


def load_exclusions(path: str | Path, model: str) -> Exclusions:
    p = _path(path, QUANT_DIR / "exclusions")
    exc = Exclusions.from_dict(yaml.safe_load(p.read_text()), where=p.name)
    if exc.model != model:
        raise ConfigError(f"{p.name}: exclusions are for model {exc.model!r}, not {model!r}")
    return exc
