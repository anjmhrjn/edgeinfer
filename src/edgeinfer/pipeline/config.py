"""Deployment recipes (configs/deploy/) and gates (configs/gates.yaml), loaded strictly.

A recipe says what to deploy: model config, precision, how the precision is made, build
settings and target backend. Everything that changes the deployed artifact is in the
recipe or the files it names, so a fault can be injected by configuration alone.

The gates file says which checks a deployment must pass and on which images. It holds no
numbers: thresholds come from configs/budgets.yaml and configs/tolerances.yaml (D-039).
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from edgeinfer.artifacts import canonical_json
from edgeinfer.config import CONFIG_DIR, ConfigError, _choice, _take

DEPLOY_DIR = CONFIG_DIR / "deploy"

PRECISIONS = ("fp32", "fp16", "int8", "int8_fp16")
INT8_PRECISIONS = ("int8", "int8_fp16")
TARGETS = ("trt", "ort_cpu")
PARITY_INPUTS = ("images", "golden")


def _path(name_or_path: str | Path, subdir: Path) -> Path:
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = subdir / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path}")
    return path


@dataclass(frozen=True)
class QuantSpec:
    config: str  # configs/quant/<name>.yaml, or a path
    exclusions: str | None  # configs/quant/exclusions/<name>.yaml, or a path; null: none

    @classmethod
    def from_dict(cls, d: Any, where: str) -> QuantSpec:
        d = _take(cls, d, where)
        return cls(
            config=str(d["config"]),
            exclusions=None if d["exclusions"] is None else str(d["exclusions"]),
        )


@dataclass(frozen=True)
class Recipe:
    name: str  # the file's stem; names the deploy report
    model: str  # configs/models/<name>.yaml, or a path (e.g. a fault config)
    precision: str
    quant: QuantSpec | None  # int8 / int8_fp16 only
    fp16: str | None  # configs/precision/<name>.yaml; fp16 only
    build: str | None  # configs/<name>.yaml builder settings; trt only
    target: str
    gates: str  # configs/<name>.yaml, or a path

    @classmethod
    def from_dict(cls, d: Any, name: str, where: str) -> Recipe:
        if isinstance(d, dict) and "name" in d:
            raise ConfigError(f"{where}: a recipe's name is its file name; remove the 'name' key")
        d = _take(cls, {**d, "name": name} if isinstance(d, dict) else d, where)
        precision = _choice(d["precision"], PRECISIONS, f"{where}.precision")
        target = _choice(d["target"], TARGETS, f"{where}.target")
        # Each precision-specific field is given exactly when it's used, so a recipe can't
        # carry a setting that silently does nothing.
        needs = {
            "quant": precision in INT8_PRECISIONS,
            "fp16": precision == "fp16",
            "build": target == "trt",
        }
        for key, needed in needs.items():
            if needed and d[key] is None:
                raise ConfigError(f"{where}.{key}: required for precision {precision!r} "
                                  f"on target {target!r}")  # fmt: skip
            if not needed and d[key] is not None:
                raise ConfigError(f"{where}.{key}: not used for precision {precision!r} "
                                  f"on target {target!r}; set it to null")  # fmt: skip
        return cls(
            name=name,
            model=str(d["model"]),
            precision=precision,
            quant=None if d["quant"] is None else QuantSpec.from_dict(d["quant"], f"{where}.quant"),
            fp16=None if d["fp16"] is None else str(d["fp16"]),
            build=None if d["build"] is None else str(d["build"]),
            target=target,
            gates=str(d["gates"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def recipe_hash(recipe: Recipe) -> str:
    """Hash of the recipe's settings (not of the files it names: their effect is in the
    artifact ids)."""
    settings = recipe.to_dict()
    settings.pop("name")
    return hashlib.sha256(canonical_json(settings).encode()).hexdigest()


def load_recipe(name_or_path: str | Path) -> Recipe:
    path = _path(name_or_path, DEPLOY_DIR)
    return Recipe.from_dict(yaml.safe_load(path.read_text()), name=path.stem, where=path.name)


@dataclass(frozen=True)
class ParityGate:
    manifest: str
    inputs: str  # images: preprocess raw images, so preprocessing faults are caught

    @classmethod
    def from_dict(cls, d: Any, where: str) -> ParityGate:
        d = _take(cls, d, where)
        return cls(manifest=str(d["manifest"]),
                   inputs=_choice(d["inputs"], PARITY_INPUTS, f"{where}.inputs"))  # fmt: skip


@dataclass(frozen=True)
class AccuracyGate:
    manifest: str
    precisions: tuple[str, ...]  # precisions gated on accuracy against FP32

    @classmethod
    def from_dict(cls, d: Any, where: str) -> AccuracyGate:
        d = _take(cls, d, where)
        if not isinstance(d["precisions"], list):
            raise ConfigError(f"{where}.precisions: expected a list")
        precisions = tuple(_choice(p, PRECISIONS, f"{where}.precisions") for p in d["precisions"])
        if "fp32" in precisions:
            raise ConfigError(f"{where}.precisions: fp32 is the reference, not a candidate")
        return cls(manifest=str(d["manifest"]), precisions=precisions)


@dataclass(frozen=True)
class PerfGate:
    perf_config: str  # configs/<name>.yaml probe settings, or a path

    @classmethod
    def from_dict(cls, d: Any, where: str) -> PerfGate:
        d = _take(cls, d, where)
        return cls(perf_config=str(d["perf_config"]))


@dataclass(frozen=True)
class GatesConfig:
    parity: ParityGate
    accuracy: AccuracyGate
    perf: PerfGate

    @classmethod
    def from_dict(cls, d: Any, where: str = "gates") -> GatesConfig:
        d = _take(cls, d, where)
        return cls(
            parity=ParityGate.from_dict(d["parity"], f"{where}.parity"),
            accuracy=AccuracyGate.from_dict(d["accuracy"], f"{where}.accuracy"),
            perf=PerfGate.from_dict(d["perf"], f"{where}.perf"),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_gates(name_or_path: str | Path = "gates") -> GatesConfig:
    path = _path(name_or_path, CONFIG_DIR)
    return GatesConfig.from_dict(yaml.safe_load(path.read_text()), where=path.name)
