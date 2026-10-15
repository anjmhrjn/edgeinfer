"""ONNX Runtime on CPU: runs the exported ONNX model on the Mac (and in CI)."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from edgeinfer.config import ModelConfig
from edgeinfer.export.onnx_export import INPUT_NAME, OUTPUT_NAME, artifact_identity
from edgeinfer.onnx_utils import layer_groups, with_outputs

# Same thread count as the torch_ref golden run (C1).
THREADS = 4
# layer_outputs() splits requests so that one session returns at most this many bytes.
# SegFormer-B0's intermediate tensors at 512x1024 add up to more than 1 GB per image.
LAYER_CHUNK_BYTES = 1 << 30


class OrtCpuBackend:
    name = "ort_cpu"

    def __init__(self) -> None:
        self._model: onnx.ModelProto | None = None
        self._session: ort.InferenceSession | None = None
        self._meta: dict[str, Any] = {}
        self._layer_sessions: dict[tuple[str, ...], list[ort.InferenceSession]] = {}

    @staticmethod
    def _options() -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = THREADS
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def _session_for(self, model: onnx.ModelProto) -> ort.InferenceSession:
        return ort.InferenceSession(
            model.SerializeToString(), self._options(), providers=["CPUExecutionProvider"]
        )

    def load(self, artifact: Path | None, cfg: ModelConfig) -> None:
        if artifact is None:
            raise ValueError("ort_cpu needs an ONNX artifact (--artifact path/to/model.onnx)")
        artifact = Path(artifact)
        identity = artifact_identity(artifact)
        self._model = onnx.load(str(artifact))
        self._session = self._session_for(self._model)
        self._layer_sessions = {}
        self._meta = {
            "artifact": artifact.name,
            "artifact_id": identity["artifact_id"],
            "precision": identity["precision"],
        }

    def infer(self, x: np.ndarray) -> np.ndarray:
        if self._session is None:
            raise RuntimeError("call load() first")
        if x.dtype != np.float32 or x.ndim != 4:
            raise ValueError(f"expected NCHW float32, got {x.shape} {x.dtype}")
        return self._session.run([OUTPUT_NAME], {INPUT_NAME: x})[0]

    def _chunks(self, names: list[str]) -> list[list[str]]:
        assert self._model is not None
        return layer_groups(self._model, names, LAYER_CHUNK_BYTES)

    def layer_outputs(self, x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        """Intermediate tensors by ONNX name. Runs a copy of the model with the
        tensors added as graph outputs, in memory-bounded chunks."""
        if self._model is None:
            raise RuntimeError("call load() first")
        key = tuple(names)
        if key not in self._layer_sessions:  # sessions are reused across images
            self._layer_sessions[key] = [
                self._session_for(with_outputs(self._model, chunk)) for chunk in self._chunks(names)
            ]
        out: dict[str, np.ndarray] = {}
        for sess in self._layer_sessions[key]:
            wanted = [o.name for o in sess.get_outputs() if o.name in set(names)]
            out.update(zip(wanted, sess.run(wanted, {INPUT_NAME: x}), strict=True))
        return out

    def metadata(self) -> dict[str, Any]:
        if self._session is None:
            raise RuntimeError("call load() first")
        return {
            "backend": self.name,
            "precision": self._meta["precision"],
            "device": f"cpu ({platform.machine()})",
            "artifact_id": self._meta["artifact_id"],
            "artifact": self._meta["artifact"],
            "threads": THREADS,
            "session": {"graph_optimization_level": "ORT_ENABLE_ALL", "providers": ["CPU"]},
            "versions": {"onnxruntime": ort.__version__, "onnx": onnx.__version__},
        }
