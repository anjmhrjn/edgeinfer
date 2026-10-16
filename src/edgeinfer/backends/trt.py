"""TensorRT engine builder and `trt` backend (D-025).

Runs only where TensorRT and cuda-bindings are installed (the NGC container; D-020, D-021).
Both are imported lazily, so the rest of edgeinfer imports anywhere and a missing runtime
is a specific error (TRT_UNAVAILABLE), not an ImportError.

Inference path, one stream, all copies through pinned host buffers:

    host input -> [ev0] H2D [ev1] execute_async_v3 [ev2] D2H [ev3] -> sync on ev3 -> host output

The events give the device stages for the performance probe (TimedBackend). The call
returns only after ev3 completes, so the stages always fit inside the host-measured call.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from edgeinfer.artifacts import load_sidecar, sha256_file, sidecar_path
from edgeinfer.backends.base import DEVICE_STAGES
from edgeinfer.backends.trt_meta import (
    ENGINE_DIR,
    BackendError,
    BuildConfig,
    check_engine_runtime,
    check_io,
    engine_artifact_id,
    engine_path,
)
from edgeinfer.config import ModelConfig


def _import_runtime() -> tuple[Any, Any]:
    try:
        import tensorrt as trt
        from cuda.bindings import runtime as cudart
    except ImportError as e:
        raise BackendError(
            "TRT_UNAVAILABLE",
            f"TensorRT or cuda-bindings isn't importable here ({e}); TensorRT runs in the GPU "
            "container: scripts/remote.sh run '<edgeinfer command>'",
        ) from e
    return trt, cudart


def _ok(cudart: Any, result: Any) -> Any:
    """cuda-bindings calls return (error, *values); raise on error, return the values."""
    err, *values = result if isinstance(result, tuple) else (result,)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"CUDA call failed: {err}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def runtime_fingerprint(device: int = 0) -> dict[str, Any]:
    """What an engine is tied to: TensorRT version and GPU, plus CUDA versions."""
    trt, cudart = _import_runtime()
    props = _ok(cudart, cudart.cudaGetDeviceProperties(device))
    name = props.name.decode() if isinstance(props.name, bytes) else str(props.name)
    return {
        "tensorrt": trt.__version__,
        "gpu_name": name.rstrip("\x00").strip(),
        "compute_capability": f"{props.major}.{props.minor}",
        "cuda_driver": _ok(cudart, cudart.cudaDriverGetVersion()),
        "cuda_runtime": _ok(cudart, cudart.cudaRuntimeGetVersion()),
    }


def _io_spec(trt: Any, engine: Any) -> list[dict[str, Any]]:
    io = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        mode = engine.get_tensor_mode(name)
        io.append({
            "name": name,
            "mode": "input" if mode == trt.TensorIOMode.INPUT else "output",
            "shape": [int(d) for d in engine.get_tensor_shape(name)],
            "dtype": np.dtype(trt.nptype(engine.get_tensor_dtype(name))).name,
        })  # fmt: skip
    return io


def build_engine(
    onnx_path: Path,
    cfg: ModelConfig,
    bcfg: BuildConfig,
    debug_outputs: list[str] | None = None,
    out_root: Path = ENGINE_DIR,
) -> tuple[Path, dict[str, Any]]:
    """Build a TensorRT engine from an ONNX artifact. Returns (path, sidecar); reuses an
    existing engine with the same id. The engine's precision is the ONNX model's: TensorRT 11
    networks are strongly typed (D-024)."""
    trt, _ = _import_runtime()
    debug_outputs = list(debug_outputs or [])
    onnx_meta = load_sidecar(onnx_path)
    runtime = runtime_fingerprint()
    art_id = engine_artifact_id(onnx_meta["artifact_id"], bcfg, runtime, debug_outputs)
    path = engine_path(cfg.name, onnx_meta["precision"], art_id, bool(debug_outputs), out_root)
    if path.is_file() and sidecar_path(path).is_file():
        return path, load_sidecar(path)

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()  # strongly typed by default in TensorRT 11
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise BackendError(
            "BUILD_PARSE_FAILED", f"ONNX parser rejected {onnx_path.name}: {errors[:5]}"
        )

    if debug_outputs:
        # Marking intermediate tensors as outputs keeps them from being fused away, so this
        # engine isn't the deployed one: it exists for per-layer parity only.
        tensors = {}
        for i in range(network.num_layers):
            layer = network.get_layer(i)
            for j in range(layer.num_outputs):
                tensors[layer.get_output(j).name] = layer.get_output(j)
        missing = [n for n in debug_outputs if n not in tensors]
        if missing:
            raise BackendError("BUILD_UNKNOWN_DEBUG_OUTPUT",
                               f"no tensors named {missing[:5]} in the parsed network")  # fmt: skip
        for name in debug_outputs:
            if not tensors[name].is_network_output:
                network.mark_output(tensors[name])

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, bcfg.workspace_mib << 20)
    config.builder_optimization_level = bcfg.builder_optimization_level
    config.profiling_verbosity = {
        "layer_names_only": trt.ProfilingVerbosity.LAYER_NAMES_ONLY,
        "detailed": trt.ProfilingVerbosity.DETAILED,
        "none": trt.ProfilingVerbosity.NONE,
    }[bcfg.profiling_verbosity]
    cache_path = out_root / (
        f"timing_cache__trt{runtime['tensorrt']}__sm{runtime['compute_capability']}.cache"
    )
    cache_in = b""
    if bcfg.timing_cache:
        cache_in = cache_path.read_bytes() if cache_path.is_file() else b""
        config.set_timing_cache(config.create_timing_cache(cache_in), ignore_mismatch=False)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise BackendError("BUILD_FAILED", f"TensorRT couldn't build {onnx_path.name}; see its log")
    build_seconds = time.perf_counter() - t0
    if bcfg.timing_cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(bytes(config.get_timing_cache().serialize()))

    engine = trt.Runtime(logger).deserialize_cuda_engine(serialized)
    io = _io_spec(trt, engine)
    problems = check_io(io, onnx_meta["contract"], cfg.num_classes)
    if problems:
        raise BackendError("BUILD_IO_MISMATCH", "; ".join(problems))

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(suffix=".engine", dir=path.parent)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        tmp.write_bytes(bytes(serialized))
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)
    inspector = engine.create_engine_inspector()
    layers = json.loads(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
    layers_path = path.with_suffix(".layers.json")
    layers_path.write_text(json.dumps(layers, indent=1) + "\n")

    meta = {
        "artifact_id": art_id,
        "kind": "engine",
        "model": cfg.name,
        "precision": onnx_meta["precision"],
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "parent_artifact_id": onnx_meta["artifact_id"],
        "build_config": bcfg.to_dict(),
        "runtime": runtime,
        "io": io,
        "contract": onnx_meta["contract"],
        "debug_outputs": debug_outputs,
        "num_layers": engine.num_layers,
        "layers_file": layers_path.name,
        "device_memory_bytes": int(engine.device_memory_size_v2),
        "timing_cache": {
            "used": bcfg.timing_cache,
            "file": cache_path.name if bcfg.timing_cache else None,
            "sha256_before_build": hashlib.sha256(cache_in).hexdigest() if cache_in else None,
        },
        "build_seconds": round(build_seconds, 2),
    }
    sidecar_path(path).write_text(json.dumps(meta, indent=2) + "\n")
    return path, meta


class TrtBackend:
    """Runs a serialized TensorRT engine on one GPU. Implements Backend and TimedBackend."""

    name = "trt"

    def __init__(self, device: int = 0) -> None:
        self._device = device
        self._meta: dict[str, Any] = {}
        self._warnings: list[str] = []
        self._allocations: list[tuple[str, int]] = []  # ("device"|"host", pointer)

    def load(self, artifact: Path | None, cfg: ModelConfig) -> None:
        trt, cudart = _import_runtime()
        self._trt, self._cudart = trt, cudart
        if artifact is None:
            raise ValueError("trt needs an engine (--artifact path/to/model.engine)")
        artifact = Path(artifact)
        meta = load_sidecar(artifact)
        if meta.get("kind") != "engine":
            raise ValueError(f"{artifact.name} isn't an engine; build one with `edgeinfer build`")
        self._runtime = runtime_fingerprint(self._device)
        self._warnings = check_engine_runtime(meta, self._runtime)
        problems = check_io(meta["io"], meta["contract"], cfg.num_classes)
        if problems:
            raise BackendError("TRT_IO_MISMATCH", "; ".join(problems))

        _ok(cudart, cudart.cudaSetDevice(self._device))
        free_before, _ = _ok(cudart, cudart.cudaMemGetInfo())
        logger = trt.Logger(trt.Logger.WARNING)
        self._engine = trt.Runtime(logger).deserialize_cuda_engine(artifact.read_bytes())
        if self._engine is None:
            raise BackendError("TRT_DESERIALIZE_FAILED", f"TensorRT couldn't load {artifact.name}")
        self._context = self._engine.create_execution_context()
        self._stream = _ok(cudart, cudart.cudaStreamCreate())
        self._events = [_ok(cudart, cudart.cudaEventCreate()) for _ in range(4)]

        self._host: dict[str, np.ndarray] = {}
        self._device_ptr: dict[str, int] = {}
        self._inputs, self._outputs = [], []
        io_bytes = 0
        for t in meta["io"]:
            dtype, shape = np.dtype(t["dtype"]), tuple(t["shape"])
            nbytes = int(np.prod(shape)) * dtype.itemsize
            dptr = int(_ok(cudart, cudart.cudaMalloc(nbytes)))
            self._allocations.append(("device", dptr))
            hptr = int(_ok(cudart, cudart.cudaMallocHost(nbytes)))  # pinned: async copies
            self._allocations.append(("host", hptr))
            buf = (ctypes.c_byte * nbytes).from_address(hptr)
            self._host[t["name"]] = np.frombuffer(buf, dtype=dtype).reshape(shape)
            self._device_ptr[t["name"]] = dptr
            if not self._context.set_tensor_address(t["name"], dptr):
                raise RuntimeError(f"set_tensor_address failed for {t['name']}")
            (self._inputs if t["mode"] == "input" else self._outputs).append(t["name"])
            io_bytes += nbytes
        free_after, _ = _ok(cudart, cudart.cudaMemGetInfo())
        self._input_name = meta["contract"]["input_name"]
        self._output_name = meta["contract"]["output_name"]
        self._meta = meta
        self._memory = {
            "engine_activation_bytes": meta["device_memory_bytes"],
            "io_bytes": io_bytes,
            "engine_file_bytes": meta["size_bytes"],
            # Everything this load took on the device: weights, activations, I/O buffers.
            "device_used_delta_bytes": int(free_before - free_after),
        }

    def _run(
        self, x: np.ndarray, outputs: list[str]
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        cudart = self._cudart
        host_in = self._host[self._input_name]
        if x.shape != host_in.shape or x.dtype != host_in.dtype:
            raise ValueError(f"expected {host_in.shape} {host_in.dtype}, got {x.shape} {x.dtype}")
        np.copyto(host_in, x)
        h2d, d2h = (
            cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
            cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
        )
        ev = self._events
        _ok(cudart, cudart.cudaEventRecord(ev[0], self._stream))
        for name in self._inputs:
            host = self._host[name]
            copy = cudart.cudaMemcpyAsync(
                self._device_ptr[name], host.ctypes.data, host.nbytes, h2d, self._stream
            )
            _ok(cudart, copy)
        _ok(cudart, cudart.cudaEventRecord(ev[1], self._stream))
        if not self._context.execute_async_v3(int(self._stream)):
            raise RuntimeError("execute_async_v3 failed")
        _ok(cudart, cudart.cudaEventRecord(ev[2], self._stream))
        for name in outputs:
            host = self._host[name]
            copy = cudart.cudaMemcpyAsync(
                host.ctypes.data, self._device_ptr[name], host.nbytes, d2h, self._stream
            )
            _ok(cudart, copy)
        _ok(cudart, cudart.cudaEventRecord(ev[3], self._stream))
        _ok(cudart, cudart.cudaEventSynchronize(ev[3]))  # outputs are on the host after this
        ms = [float(_ok(cudart, cudart.cudaEventElapsedTime(ev[i], ev[i + 1]))) for i in range(3)]
        # Copies: the pinned buffers are reused by the next call.
        return {n: self._host[n].copy() for n in outputs}, dict(zip(DEVICE_STAGES, ms, strict=True))

    def infer(self, x: np.ndarray) -> np.ndarray:
        return self._run(x, [self._output_name])[0][self._output_name]

    def infer_timed(self, x: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        outs, ms = self._run(x, [self._output_name])
        return outs[self._output_name], ms

    def layer_outputs(self, x: np.ndarray, names: list[str]) -> dict[str, np.ndarray]:
        missing = [n for n in names if n not in self._outputs]
        if missing:
            raise ValueError(
                f"{len(missing)} tensors aren't outputs of this engine (first: {missing[:3]}); "
                "build a debug engine with `edgeinfer build --debug-outputs ...`"
            )
        return self._run(x, list(names))[0]

    def metadata(self) -> dict[str, Any]:
        if not self._meta:
            raise RuntimeError("call load() first")
        rt = self._runtime
        sm = rt["compute_capability"].replace(".", "")
        return {
            "backend": self.name,
            "precision": self._meta["precision"],
            "device": f"cuda:{self._device} ({rt['gpu_name']}, sm_{sm})",
            "artifact_id": self._meta["artifact_id"],
            "artifact": self._meta["file"],
            "debug_outputs": self._meta["debug_outputs"],
            "device_memory": self._memory,
            "runtime_warnings": self._warnings,
            "versions": {
                "tensorrt": rt["tensorrt"],
                "cuda_driver": str(rt["cuda_driver"]),
                "cuda_runtime": str(rt["cuda_runtime"]),
            },
        }

    def close(self) -> None:
        cudart = getattr(self, "_cudart", None)
        if cudart is None:
            return
        for kind, ptr in self._allocations:
            (cudart.cudaFree if kind == "device" else cudart.cudaFreeHost)(ptr)
        self._allocations = []
