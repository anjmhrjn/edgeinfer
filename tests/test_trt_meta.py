"""Engine identity, sidecar checks and runtime compatibility (D-025). No TensorRT needed."""

import sys
from dataclasses import replace

import pytest

from edgeinfer.backends.base import TimedBackend
from edgeinfer.backends.trt import TrtBackend
from edgeinfer.backends.trt_meta import (
    BackendError,
    BuildConfig,
    check_engine_runtime,
    check_io,
    engine_artifact_id,
    engine_path,
    layer_datatypes,
    load_build_config,
)
from edgeinfer.config import ConfigError, load_model_config
from tests.conftest import TINY_CONFIG

RUNTIME = {"tensorrt": "11.3.0.1", "gpu_name": "Tesla T4", "compute_capability": "7.5",
           "cuda_driver": 13020, "cuda_runtime": 13040}  # fmt: skip
CONTRACT = {"input_name": "input", "output_name": "logits",
            "input": {"height": 32, "width": 64}}  # fmt: skip
GOOD_IO = [
    {"name": "input", "mode": "input", "shape": [1, 3, 32, 64], "dtype": "float32"},
    {"name": "logits", "mode": "output", "shape": [1, 19, 8, 16], "dtype": "float32"},
]


def test_shipped_build_config_loads():
    bcfg = load_build_config()
    assert bcfg.builder_optimization_level == 3 and bcfg.timing_cache


@pytest.mark.parametrize(
    "change",
    [{"builder_optimization_level": 6}, {"workspace_mib": 0}, {"timing_cache": "yes"},
     {"profiling_verbosity": "verbose"}, {"fp16": True}],
)  # fmt: skip
def test_build_config_is_strict(change):
    with pytest.raises(ConfigError):
        BuildConfig.from_dict({**load_build_config().to_dict(), **change}, where="test")


def test_engine_id_depends_on_everything_the_engine_depends_on():
    bcfg = load_build_config()
    base = engine_artifact_id("onnx1", bcfg, RUNTIME, [])
    variants = [
        engine_artifact_id("onnx2", bcfg, RUNTIME, []),
        engine_artifact_id("onnx1", replace(bcfg, builder_optimization_level=5), RUNTIME, []),
        engine_artifact_id("onnx1", bcfg, {**RUNTIME, "tensorrt": "11.4.0.1"}, []),
        engine_artifact_id("onnx1", bcfg, {**RUNTIME, "compute_capability": "8.6"}, []),
        engine_artifact_id("onnx1", bcfg, RUNTIME, ["a"]),
    ]
    assert len({base, *variants}) == 6
    # The CUDA driver isn't part of the id: a driver update doesn't invalidate an engine.
    assert engine_artifact_id("onnx1", bcfg, {**RUNTIME, "cuda_driver": 13050}, []) == base
    # Debug output order doesn't matter.
    assert engine_artifact_id("o", bcfg, RUNTIME, ["a", "b"]) == engine_artifact_id(
        "o", bcfg, RUNTIME, ["b", "a"]
    )


def test_debug_engines_are_named_apart(tmp_path):
    assert "__debug__" in engine_path("m", "fp16", "ab" * 32, True, tmp_path).name
    assert "debug" not in engine_path("m", "fp16", "ab" * 32, False, tmp_path).name


def test_engine_for_another_runtime_is_refused():
    meta = {"file": "m.engine", "runtime": RUNTIME}
    assert check_engine_runtime(meta, dict(RUNTIME)) == []
    warnings = check_engine_runtime(meta, {**RUNTIME, "cuda_driver": 13050})
    assert len(warnings) == 1 and "cuda_driver" in warnings[0]
    for key, value in [("tensorrt", "11.4.0.1"), ("compute_capability", "8.6"),
                       ("gpu_name", "NVIDIA A10G")]:  # fmt: skip
        with pytest.raises(BackendError) as e:
            check_engine_runtime(meta, {**RUNTIME, key: value})
        assert e.value.code == "TRT_ENGINE_MISMATCH" and key in str(e.value)


def test_check_io():
    assert check_io(GOOD_IO, CONTRACT, 19) == []
    half_input = [{**GOOD_IO[0], "dtype": "float16"}, GOOD_IO[1]]
    assert "input" in check_io(half_input, CONTRACT, 19)[0]
    wrong_shape = [{**GOOD_IO[0], "shape": [1, 3, 64, 64]}, GOOD_IO[1]]
    assert check_io(wrong_shape, CONTRACT, 19)
    assert "no output" in check_io(GOOD_IO[:1], CONTRACT, 19)[0]
    assert check_io(GOOD_IO, CONTRACT, 21)  # class count differs


def test_trt_backend_without_tensorrt_is_a_specific_error(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "tensorrt", None)  # import fails, even in the container
    backend = TrtBackend()
    assert isinstance(backend, TimedBackend)
    with pytest.raises(BackendError) as e:
        backend.load(tmp_path / "x.engine", load_model_config(TINY_CONFIG))
    assert e.value.code == "TRT_UNAVAILABLE" and "remote.sh" in str(e.value)


def test_cli_without_tensorrt_exits_2_with_the_code(tiny_onnx, monkeypatch, capsys):
    from edgeinfer.cli import main

    monkeypatch.setitem(sys.modules, "tensorrt", None)
    argv = ["build", "--model", str(TINY_CONFIG), "--artifact", str(tiny_onnx)]
    assert main(argv) == 2
    assert "TRT_UNAVAILABLE" in capsys.readouterr().out
    argv = ["perf", "--model", str(TINY_CONFIG), "--backend", "trt", "--artifact", "x.engine",
            "--synthetic", "64x128"]  # fmt: skip
    assert main(argv) == 2


def test_layer_datatypes():
    layers = {"Layers": [{"Outputs": [{"Datatype": "Half"}, {"Datatype": "Half"}]},
                         {"Outputs": [{"Datatype": "Float"}]}]}  # fmt: skip
    assert layer_datatypes(layers) == {"Float": 1, "Half": 2}
    with pytest.raises(ValueError, match="detailed"):
        layer_datatypes({"Layers": ["conv1", "relu1"]})
