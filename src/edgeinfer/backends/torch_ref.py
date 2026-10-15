"""FP32 PyTorch reference on CPU. This is the ground truth other backends are compared to."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers

from edgeinfer.config import ModelConfig
from edgeinfer.models.registry import build_model, weights_id


class TorchRefBackend:
    name = "torch_ref"

    def __init__(self) -> None:
        self._model: torch.nn.Module | None = None
        self._cfg: ModelConfig | None = None

    def load(self, artifact: Path | None, cfg: ModelConfig) -> None:
        if artifact is not None:
            raise ValueError(
                "torch_ref builds the model from the config's weights; pass artifact=None"
            )
        # CPU, not MPS, is the reference: deterministic and closer to how other runtimes compute.
        torch.use_deterministic_algorithms(True)
        self._model = build_model(cfg).to("cpu").float()
        self._cfg = cfg

    def infer(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("call load() first")
        if x.dtype != np.float32 or x.ndim != 4:
            raise ValueError(f"expected NCHW float32, got {x.shape} {x.dtype}")
        with torch.inference_mode():
            return self._model(torch.from_numpy(x)).numpy()

    def layer_outputs(self, x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        # D-012: per-layer tensors are keyed by ONNX tensor names, which PyTorch modules
        # don't have. The per-layer reference is the FP32 ONNX model run in ORT.
        raise NotImplementedError(
            "torch_ref has no per-layer outputs (D-012); use ort_cpu on the FP32 ONNX model "
            "as the per-layer reference"
        )

    def metadata(self) -> dict[str, Any]:
        if self._cfg is None:
            raise RuntimeError("call load() first")
        return {
            "backend": self.name,
            "precision": "fp32",
            "device": f"cpu ({platform.machine()})",
            "artifact_id": weights_id(self._cfg),
            "threads": torch.get_num_threads(),
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__},
        }
