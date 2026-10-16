"""ONNX Runtime on CPU: runs the exported ONNX model on the Mac (and in CI)."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort

from edgeinfer.artifacts import INPUT_NAME, OUTPUT_NAME, artifact_identity
from edgeinfer.config import ModelConfig
from edgeinfer.onnx_utils import layer_groups, with_outputs

# Same thread count as the torch_ref golden run (C1).
THREADS = 4
# layer_outputs() splits requests so that one session returns at most this many bytes.
# SegFormer-B0's intermediate tensors at 512x1024 add up to more than 1 GB per image.
LAYER_CHUNK_BYTES = 1 << 30
# ORT 1.31's CPU graph optimizations break INT8 Q/DQ models with FP16 scales: they fuse
# MatMul+Add into QGemm, which rejects FP16 scales, and run DeepLab's wrongly (D-031).
# Those models run unoptimized; the result is a check of the conversion, not a benchmark.
UNOPTIMIZED_PRECISIONS = ("int8_fp16",)


class OrtCpuBackend:
    name = "ort_cpu"

    def __init__(self, threads: int = THREADS) -> None:
        self._threads = threads
        self._model: onnx.ModelProto | None = None
        self._session: ort.InferenceSession | None = None
        self._meta: dict[str, Any] = {}
        self._layer_sessions: dict[tuple[str, ...], list[ort.InferenceSession]] = {}
        self._opt_level = "ORT_ENABLE_ALL"

    def _options(self) -> ort.SessionOptions:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = self._threads
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = getattr(ort.GraphOptimizationLevel, self._opt_level)
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
        self._opt_level = ("ORT_DISABLE_ALL" if identity["precision"] in UNOPTIMIZED_PRECISIONS
                           else "ORT_ENABLE_ALL")  # fmt: skip
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
            "threads": self._threads,
            "session": {"graph_optimization_level": self._opt_level, "providers": ["CPU"]},
            "versions": {"onnxruntime": ort.__version__, "onnx": onnx.__version__},
        }
