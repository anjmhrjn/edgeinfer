"""The Backend protocol shared by torch_ref, ort_cpu and trt, and the optional
TimedBackend protocol for backends that can time their own device stages (D-022)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from edgeinfer.config import ModelConfig

# Device stages a TimedBackend reports, in execution order.
DEVICE_STAGES = ("h2d", "compute", "d2h")


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


@runtime_checkable
class TimedBackend(Backend, Protocol):
    """A backend that can time its own stages on the device (trt: CUDA events).

    Kept separate from Backend so the Backend contract stays what downstream consumers
    rely on, and backends without device stages (torch_ref, ort_cpu) don't fake them.
    """

    def infer_timed(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        """Like infer(), plus milliseconds per stage in DEVICE_STAGES, measured on the
        device. The call returns only after the output is on the host, so the device
        stages always fit inside the host-measured duration of the call."""
        ...
