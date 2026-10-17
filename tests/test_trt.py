"""TensorRT build and inference on the tiny fixture model (D-025). Runs on the GPU instance
only: scripts/remote.sh run 'uv run --no-sync pytest -m gpu'."""

import json

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from edgeinfer.artifacts import sidecar_path
from edgeinfer.backends.base import DEVICE_STAGES
from edgeinfer.backends.trt import TrtBackend, build_engine, runtime_fingerprint
from edgeinfer.backends.trt_meta import BackendError, load_build_config
from edgeinfer.config import load_model_config
from tests.conftest import TINY_CONFIG

pytestmark = pytest.mark.gpu
CFG = load_model_config(TINY_CONFIG)


@pytest.fixture(scope="module")
def engine(tiny_onnx, tmp_path_factory):
    return build_engine(tiny_onnx, CFG, load_build_config(), out_root=tmp_path_factory.mktemp("e"))


@pytest.fixture
def x():
    return np.random.default_rng(0).standard_normal((1, 3, 32, 64)).astype(np.float32)


def _loaded(path):
    backend = TrtBackend()
    backend.load(path, CFG)
    return backend


def test_runtime_is_the_t4():
    rt = runtime_fingerprint()
    assert rt["compute_capability"] == "7.5", rt


def test_fp32_engine_matches_ort(engine, tiny_onnx, x):
    path, meta = engine
    assert meta["precision"] == "fp32" and meta["debug_outputs"] == []
    ref = ort.InferenceSession(str(tiny_onnx)).run(None, {"input": x})[0]
    backend = _loaded(path)
    got = backend.infer(x)
    assert got.shape == ref.shape and got.dtype == np.float32
    assert np.abs(got - ref).max() < 1e-3 * max(1.0, float(np.abs(ref).max()))
    backend.close()


def test_rebuild_reuses_the_engine(engine, tiny_onnx):
    path, _ = engine
    again, _ = build_engine(tiny_onnx, CFG, load_build_config(), out_root=path.parents[1])
    assert again == path


def test_device_stages_fit_inside_the_call(engine, x):
    import time

    backend = _loaded(engine[0])
    for _ in range(20):
        t0 = time.perf_counter()
        _, ms = backend.infer_timed(x)
        host_ms = (time.perf_counter() - t0) * 1e3
        assert set(ms) == set(DEVICE_STAGES) and all(v >= 0 for v in ms.values())
        assert sum(ms.values()) <= host_ms + 0.05
    meta = backend.metadata()
    assert meta["device_memory"]["device_used_delta_bytes"] > 0
    assert "T4" in meta["device"]
    backend.close()


def test_mismatched_engine_is_refused(engine, tmp_path):
    path, meta = engine
    copy = tmp_path / path.name
    copy.write_bytes(path.read_bytes())
    meta = {**meta, "runtime": {**meta["runtime"], "tensorrt": "10.0.0"}}
    sidecar_path(copy).write_text(json.dumps(meta))
    with pytest.raises(BackendError) as e:
        _loaded(copy)
    assert e.value.code == "TRT_ENGINE_MISMATCH"


def test_debug_engine_exposes_intermediate_tensors(tiny_onnx, tmp_path, x):
    model = onnx.load(str(tiny_onnx))
    name = model.graph.node[0].output[0]  # first intermediate tensor in graph order
    path, meta = build_engine(tiny_onnx, CFG, load_build_config(), debug_outputs=[name],
                              out_root=tmp_path)  # fmt: skip
    assert "__debug__" in path.name and meta["debug_outputs"] == [name]
    backend = _loaded(path)
    assert backend.metadata()["debug_outputs"] == [name]
    out = backend.layer_outputs(x, [name])
    assert set(out) == {name}
    with pytest.raises(ValueError, match="debug engine"):
        backend.layer_outputs(x, ["not_a_tensor"])
    backend.close()


def test_implicit_int8_is_gone():
    """D-031: TensorRT 11 removes calibrator-based (implicit) INT8; explicit Q/DQ only."""
    import tensorrt as trt

    assert not hasattr(trt, "IInt8Calibrator")
    assert not hasattr(trt, "IInt8EntropyCalibrator2")
    assert not hasattr(trt.BuilderFlag, "INT8")


def test_int8_engine_fuses_qdq(fake_cityscapes, tiny_onnx, tmp_path, x):
    """An INT8 Q/DQ model builds into INT8 layers, and the layer summary can place every
    quantized ONNX node in the real inspector output. Not every node fuses: in this fixture
    a DQ shared by a Conv and a Concat leaves the Conv in FP32 with a separate weight Scale
    layer (first T4 run, C5), which the summary reports as BUILD_QDQ_UNFUSED."""
    from dataclasses import replace

    from edgeinfer.backends.trt_layers import layer_summary, quantized_onnx_nodes
    from edgeinfer.quant.calib import calibrate
    from edgeinfer.quant.config import load_group_rules, load_quant_config, no_exclusions
    from edgeinfer.quant.ptq import quantize
    from tests.test_quant import TINY_GROUPS, _manifest

    q = load_quant_config("int8_percentile")
    calib = _manifest(fake_cityscapes, tmp_path, "train")
    q = replace(q, calibration=replace(q.calibration, manifest=str(calib)))
    cal = calibrate(tiny_onnx, CFG, q, fake_cityscapes, tmp_path / "calib")
    rules = load_group_rules("tiny_fixture", TINY_GROUPS)
    int8, _ = quantize(tiny_onnx, CFG, q, cal, rules, no_exclusions(CFG.name), tmp_path / "onnx")
    path, meta = build_engine(int8, CFG, load_build_config(), out_root=tmp_path / "engines")
    assert meta["precision"] == "int8"
    layers = json.loads((path.parent / meta["layers_file"]).read_text())
    summary, findings = layer_summary(layers, quantized_onnx_nodes(onnx.load(str(int8))))
    print(json.dumps(summary, indent=1))  # the real inspector output, for the session log
    q = summary["quantized_onnx_nodes"]
    assert summary["by_precision"].get("int8", 0) > 0
    assert q["unmapped"] == [] and q["int8"] > len(q["not_int8"]), q
    assert {f["code"] for f in findings} == ({"BUILD_QDQ_UNFUSED"} if q["not_int8"] else set())
    ref = ort.InferenceSession(str(int8)).run(None, {"input": x})[0]
    got = _loaded(path).infer(x)
    # INT8 kernels round differently from ORT's; a loose bound catches a broken build.
    assert np.abs(got - ref).max() < 0.25 * max(1.0, float(np.abs(ref).max()))
