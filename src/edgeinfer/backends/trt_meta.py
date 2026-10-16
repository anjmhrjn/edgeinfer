"""TensorRT engine identity, metadata sidecar and runtime compatibility (D-025).

Pure Python, no TensorRT import, so it is tested on the Mac. A serialized engine is only
valid for the TensorRT version and GPU it was built for, so every engine has a sidecar
describing what it was built with, and the `trt` backend refuses to load an engine whose
sidecar doesn't match the runtime: a mismatch would fail at deserialization at best,
and silently run different kernels at worst.

The sidecar is also the engine's self-description for the downstream C++ runtime:
I/O names, shapes and dtypes, precision, TensorRT version, and the preprocessing contract.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from edgeinfer.artifacts import canonical_json
from edgeinfer.config import CONFIG_DIR, ConfigError, _choice, _take
from edgeinfer.reports import REPO_ROOT

ENGINE_DIR = REPO_ROOT / "artifacts" / "engines"
PROFILING_VERBOSITY = ("layer_names_only", "detailed", "none")


class BackendError(RuntimeError):
    """A backend can't load an artifact in this environment. `code` is a report code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class BuildConfig:
    name: str
    workspace_mib: int
    builder_optimization_level: int
    timing_cache: bool
    profiling_verbosity: str

    @classmethod
    def from_dict(cls, d: Any, where: str) -> BuildConfig:
        d = _take(cls, d, where)
        if not isinstance(d["workspace_mib"], int) or d["workspace_mib"] <= 0:
            raise ConfigError(f"{where}.workspace_mib: expected a positive int")
        level = d["builder_optimization_level"]
        if not isinstance(level, int) or not 0 <= level <= 5:
            raise ConfigError(f"{where}.builder_optimization_level: expected 0..5, got {level!r}")
        if not isinstance(d["timing_cache"], bool):
            raise ConfigError(f"{where}.timing_cache: expected true or false")
        _choice(d["profiling_verbosity"], PROFILING_VERBOSITY, f"{where}.profiling_verbosity")
        return cls(**d)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_build_config(name_or_path: str | Path = "build") -> BuildConfig:
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    return BuildConfig.from_dict(yaml.safe_load(path.read_text()), where=path.name)


def engine_artifact_id(
    onnx_id: str, bcfg: BuildConfig, runtime: dict[str, Any], debug_outputs: list[str]
) -> str:
    """Hash of everything the engine depends on: the ONNX model, the builder settings,
    the TensorRT version and the GPU, and any outputs marked for debugging."""
    key = {
        "onnx_artifact_id": onnx_id,
        "build_config": bcfg.to_dict(),
        "tensorrt": runtime["tensorrt"],
        "gpu": runtime["gpu_name"],
        "compute_capability": runtime["compute_capability"],
        "debug_outputs": sorted(debug_outputs),
    }
    return hashlib.sha256(canonical_json(key).encode()).hexdigest()


def engine_path(
    model: str, precision: str, art_id: str, debug: bool, root: Path = ENGINE_DIR
) -> Path:
    tag = "__debug" if debug else ""
    return root / model / f"{model}__{precision}{tag}__{art_id[:12]}.engine"


# Runtime fields that must match exactly, and ones that are recorded and only warned about.
# A plan is built for one TensorRT version and one GPU architecture; the CUDA driver can be
# newer than at build time (backward compatible), but a difference is still worth knowing.
MUST_MATCH = ("tensorrt", "compute_capability", "gpu_name")
SHOULD_MATCH = ("cuda_driver", "cuda_runtime")


def compare_runtime(built: dict[str, Any], current: dict[str, Any]) -> tuple[list[str], list[str]]:
    """(errors, warnings) comparing an engine's build runtime with the current one."""
    errors = [f"{k}: engine built with {built.get(k)!r}, runtime has {current.get(k)!r}"
              for k in MUST_MATCH if built.get(k) != current.get(k)]  # fmt: skip
    warnings = [f"{k}: engine built with {built.get(k)!r}, runtime has {current.get(k)!r}"
                for k in SHOULD_MATCH if built.get(k) != current.get(k)]  # fmt: skip
    return errors, warnings


def check_engine_runtime(meta: dict[str, Any], current: dict[str, Any]) -> list[str]:
    """Raise TRT_ENGINE_MISMATCH if the engine can't run here; return warnings otherwise."""
    errors, warnings = compare_runtime(meta["runtime"], current)
    if errors:
        raise BackendError(
            "TRT_ENGINE_MISMATCH",
            f"{meta['file']} was built for a different runtime ({'; '.join(errors)}); "
            "rebuild it with `edgeinfer build` on this machine",
        )
    return warnings


def check_io(io: list[dict[str, Any]], contract: dict[str, Any], num_classes: int) -> list[str]:
    """Problems with an engine's I/O against the model's contract: one float32 NCHW input
    and one float32 logits output with the contract's names and sizes."""
    problems = []
    by_name = {t["name"]: t for t in io}
    inp, out = by_name.get(contract["input_name"]), by_name.get(contract["output_name"])
    spec = contract["input"]
    if inp is None or inp["mode"] != "input":
        name = contract["input_name"]
        problems.append(f"no input tensor {name!r} (engine I/O: {sorted(by_name)})")
    else:
        expected = [1, 3, spec["height"], spec["width"]]
        if inp["shape"] != expected or inp["dtype"] != "float32":
            problems.append(f"input {inp['name']}: {inp['shape']} {inp['dtype']}, "
                            f"expected {expected} float32")  # fmt: skip
    if out is None or out["mode"] != "output":
        problems.append(f"no output tensor {contract['output_name']!r}")
    elif out["dtype"] != "float32" or out["shape"][:2] != [1, num_classes]:
        problems.append(f"output {out['name']}: {out['shape']} {out['dtype']}, "
                        f"expected [1, {num_classes}, h, w] float32")  # fmt: skip
    return problems
