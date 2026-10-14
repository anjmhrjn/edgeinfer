"""The Backend protocol shared by torch_ref, ort_cpu and trt."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

import numpy as np

from edgeinfer.config import ModelConfig


class Backend(Protocol):
    name: str  # "torch_ref" | "ort_cpu" | "trt"

    def load(self, artifact: Path | None, cfg: ModelConfig) -> None:
        """Load a model artifact. torch_ref takes None and builds from the config's weights."""
        ...

    def infer(self, x: np.ndarray) -> np.ndarray:
        """NCHW float32 in, logits out (1 x classes x H/stride x W/stride)."""
        ...

    def layer_outputs(self, x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]: ...

    def metadata(self) -> dict[str, Any]:
        """Precision, versions, device, artifact id."""
        ...
