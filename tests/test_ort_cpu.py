import numpy as np
import onnx
import pytest

from edgeinfer.artifacts import sha256_file
from edgeinfer.backends import get_backend, ort_cpu
from edgeinfer.config import load_model_config
from edgeinfer.onnx_utils import layer_tensors
from tests.conftest import TINY_CONFIG


@pytest.fixture
def x():
    return np.random.default_rng(1).standard_normal((1, 3, 32, 64)).astype(np.float32)


def _ort(path):
    backend = get_backend("ort_cpu")
    backend.load(path, load_model_config(TINY_CONFIG))
    return backend


def test_ort_matches_torch(tiny_onnx, x):
    cfg = load_model_config(TINY_CONFIG)
    ref = get_backend("torch_ref")
    ref.load(None, cfg)
    want = ref.infer(x)
    got = _ort(tiny_onnx).infer(x)
    assert got.shape == want.shape
    assert np.abs(got - want).max() < 1e-5 * np.abs(want).max()


def test_ort_metadata_reads_sidecar(tiny_onnx):
    meta = _ort(tiny_onnx).metadata()
    assert meta["backend"] == "ort_cpu" and meta["precision"] == "fp32"
    assert len(meta["artifact_id"]) == 64 and meta["threads"] == ort_cpu.THREADS


def test_ort_requires_an_artifact():
    with pytest.raises(ValueError, match="needs an ONNX artifact"):
        get_backend("ort_cpu").load(None, load_model_config(TINY_CONFIG))


def test_layer_outputs_names_shapes_and_read_only(tiny_onnx, x):
    before = sha256_file(tiny_onnx)
    backend = _ort(tiny_onnx)
    names = layer_tensors(onnx.load(str(tiny_onnx)))
    assert len(names) >= 8
    outs = backend.layer_outputs(x, names)
    assert set(outs) == set(names)
    assert all(v.dtype == np.float32 and np.isfinite(v).all() for v in outs.values())
    # Asking for intermediate outputs must not change the final output.
    np.testing.assert_array_equal(backend.infer(x), _ort(tiny_onnx).infer(x))
    assert sha256_file(tiny_onnx) == before


def test_layer_outputs_chunking_gives_same_tensors(tiny_onnx, x, monkeypatch):
    names = layer_tensors(onnx.load(str(tiny_onnx)))
    whole = _ort(tiny_onnx).layer_outputs(x, names)
    monkeypatch.setattr(ort_cpu, "LAYER_CHUNK_BYTES", 4096)
    backend = _ort(tiny_onnx)
    assert len(backend._chunks(names)) > 1
    chunked = backend.layer_outputs(x, names)
    for name in names:
        np.testing.assert_array_equal(whole[name], chunked[name])


def test_layer_outputs_unknown_name(tiny_onnx, x):
    with pytest.raises(KeyError, match="not node outputs"):
        _ort(tiny_onnx).layer_outputs(x, ["no_such_tensor"])


def test_torch_ref_has_no_layer_outputs(x):
    ref = get_backend("torch_ref")
    ref.load(None, load_model_config(TINY_CONFIG))
    with pytest.raises(NotImplementedError, match="D-012"):
        ref.layer_outputs(x, ["relu"])
