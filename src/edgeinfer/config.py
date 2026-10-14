"""Model configs (configs/models/*.yaml), loaded strictly.

Every preprocessing detail lives in the config so that other runtimes (ORT,
TensorRT, the C++ runtime) can reproduce it exactly. Unknown or missing keys
are errors: a typo must never fall back to a silent default.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"

RESIZE_MODES = ("bilinear",)
CHANNEL_ORDERS = ("RGB", "BGR")
LAYOUTS = ("NCHW",)
DTYPES = ("float32",)


class ConfigError(ValueError):
    pass


def _take(cls: type, data: Any, where: str, optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """Check that `data` has exactly the fields of dataclass `cls` (minus optional ones)."""
    if not isinstance(data, dict):
        raise ConfigError(f"{where}: expected a mapping, got {type(data).__name__}")
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    missing = names - set(data) - set(optional)
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)}")
    if missing:
        raise ConfigError(f"{where}: missing keys {sorted(missing)}")
    return data


def _choice(value: Any, allowed: tuple[str, ...], where: str) -> str:
    if value not in allowed:
        raise ConfigError(f"{where}: {value!r} not in {list(allowed)}")
    return value


@dataclass(frozen=True)
class InputSpec:
    height: int
    width: int
    layout: str
    dtype: str
    channel_order: str

    @classmethod
    def from_dict(cls, d: Any, where: str) -> InputSpec:
        d = _take(cls, d, where)
        for key in ("height", "width"):
            if not isinstance(d[key], int) or d[key] <= 0:
                raise ConfigError(f"{where}.{key}: expected a positive int, got {d[key]!r}")
        return cls(
            height=d["height"],
            width=d["width"],
            layout=_choice(d["layout"], LAYOUTS, f"{where}.layout"),
            dtype=_choice(d["dtype"], DTYPES, f"{where}.dtype"),
            channel_order=_choice(d["channel_order"], CHANNEL_ORDERS, f"{where}.channel_order"),
        )


@dataclass(frozen=True)
class PreprocessSpec:
    resize: str
    scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]

    @classmethod
    def from_dict(cls, d: Any, where: str) -> PreprocessSpec:
        d = _take(cls, d, where)
        for key in ("mean", "std"):
            if not (isinstance(d[key], list) and len(d[key]) == 3):
                raise ConfigError(f"{where}.{key}: expected 3 numbers, got {d[key]!r}")
        if any(s <= 0 for s in d["std"]) or d["scale"] <= 0:
            raise ConfigError(f"{where}: scale and std must be positive")
        return cls(
            resize=_choice(d["resize"], RESIZE_MODES, f"{where}.resize"),
            scale=float(d["scale"]),
            mean=tuple(float(v) for v in d["mean"]),
            std=tuple(float(v) for v in d["std"]),
        )


@dataclass(frozen=True)
class PostprocessSpec:
    upsample: str

    @classmethod
    def from_dict(cls, d: Any, where: str) -> PostprocessSpec:
        d = _take(cls, d, where)
        return cls(upsample=_choice(d["upsample"], RESIZE_MODES, f"{where}.upsample"))


@dataclass(frozen=True)
class ModelConfig:
    name: str
    architecture: str
    weights: dict[str, str]  # source-specific keys, checked by the model registry
    num_classes: int
    output_stride: int
    input: InputSpec
    preprocess: PreprocessSpec
    postprocess: PostprocessSpec
    backbone_output_stride: int | None = None

    @classmethod
    def from_dict(cls, d: Any, where: str = "config") -> ModelConfig:
        d = _take(cls, d, where, optional=("backbone_output_stride",))
        if not isinstance(d["weights"], dict) or "license" not in d["weights"]:
            raise ConfigError(f"{where}.weights: expected a mapping with a 'license' key")
        return cls(
            name=d["name"],
            architecture=d["architecture"],
            weights={k: str(v) for k, v in d["weights"].items()},
            num_classes=int(d["num_classes"]),
            output_stride=int(d["output_stride"]),
            input=InputSpec.from_dict(d["input"], f"{where}.input"),
            preprocess=PreprocessSpec.from_dict(d["preprocess"], f"{where}.preprocess"),
            postprocess=PostprocessSpec.from_dict(d["postprocess"], f"{where}.postprocess"),
            backbone_output_stride=d.get("backbone_output_stride"),
        )


def load_model_config(name_or_path: str | Path) -> ModelConfig:
    """Load by model name (configs/models/<name>.yaml) or by path."""
    path = Path(name_or_path)
    if path.suffix not in (".yaml", ".yml"):
        path = CONFIG_DIR / "models" / f"{name_or_path}.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"model config not found: {path}")
    return ModelConfig.from_dict(yaml.safe_load(path.read_text()), where=path.name)
