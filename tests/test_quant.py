"""INT8 PTQ on the tiny fixture model and the fake dataset (runs in CI)."""

import json
from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import numpy_helper
from onnxruntime.quantization import CalibrationMethod
from onnxruntime.quantization.calibrate import create_calibrator

from edgeinfer.backends import get_backend
from edgeinfer.config import load_model_config
from edgeinfer.data.cityscapes import iter_samples, list_image_ids
from edgeinfer.data.manifests import ManifestSpec, build_manifest, format_manifest
from edgeinfer.export.onnx_export import load_sidecar
from edgeinfer.preprocess import preprocess
from edgeinfer.quant import QuantError
from edgeinfer.quant.calib import calibrate
from edgeinfer.quant.config import Exclusions, load_group_rules, load_quant_config, no_exclusions
from edgeinfer.quant.ptq import quantize
from tests.conftest import FIXTURES, TINY_CONFIG

TINY_GROUPS = FIXTURES / "tiny_fixture_groups.yaml"


def _manifest(root, tmp_path, split):
    path = tmp_path / f"{split}.txt"
    ids = list_image_ids(root, split)
    path.write_text(format_manifest(build_manifest(ManifestSpec("m", split, None, None), ids)))
    return path


@pytest.fixture
def setup(fake_cityscapes, tmp_path, tiny_onnx):
    cfg = load_model_config(TINY_CONFIG)
    q = load_quant_config("int8_entropy")
    calib = _manifest(fake_cityscapes, tmp_path, "train")
    q = replace(q, calibration=replace(q.calibration, manifest=str(calib)))
    rules = load_group_rules("tiny_fixture", TINY_GROUPS)
    return cfg, q, rules, fake_cityscapes, tmp_path, tiny_onnx


def _run(setup, exclusions=None, q=None):
    cfg, q0, rules, root, tmp, fp32 = setup
    q = q or q0
    cal = calibrate(fp32, cfg, q, root, tmp / "calib")
    return quantize(fp32, cfg, q, cal, rules, exclusions or no_exclusions(cfg.name), tmp / "onnx")


def _qdq(model):
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    qdq_ops = ("QuantizeLinear", "DequantizeLinear")
    return [(n, inits) for n in model.graph.node if n.op_type in qdq_ops]


def test_int8_model_is_tensorrt_format(setup):
    path, meta = _run(setup)
    model = onnx.load(str(path))
    qdq = _qdq(model)
    assert qdq
    for node, inits in qdq:
        zp = inits[node.input[2]]
        assert zp.dtype == np.int8 and not zp.any(), node.name  # signed, zero point 0
    assert meta["placement"]["quantized_nodes"] == {"Conv": 4, "MatMul": 4}
    assert meta["precision"] == "int8" and meta["excluded_nodes"] == []
    props = {p.key: p.value for p in model.metadata_props}
    assert props["edgeinfer.artifact_id"] == meta["artifact_id"]
    assert "edgeinfer.contract" in props  # the I/O contract survives quantization


def test_quantize_is_idempotent_and_deterministic(setup):
    path, meta = _run(setup)
    mtime = path.stat().st_mtime_ns
    again, _ = _run(setup)
    assert again == path and path.stat().st_mtime_ns == mtime  # reused, not rewritten
    path.unlink()
    path.with_suffix(".meta.json").unlink()
    rebuilt, meta2 = _run(setup)
    assert rebuilt == path and meta2["sha256"] == meta["sha256"]


def test_exclusions_keep_nodes_in_float(setup):
    full, _ = _run(setup)
    path, meta = _run(setup, Exclusions("no_attn", "tiny_fixture", ("attn",), ()))
    assert path != full
    assert meta["placement"]["excluded_nodes"] == {"MatMul": 4}
    assert {e["group"] for e in meta["excluded_nodes"]} == {"attn"}


def test_calibrating_on_val_is_refused(setup):
    cfg, q, _, root, tmp, fp32 = setup
    val = _manifest(root, tmp, "val")
    bad = replace(q, calibration=replace(q.calibration, manifest=str(val)))
    with pytest.raises(QuantError, match="never calibrate on evaluation data") as e:
        calibrate(fp32, cfg, bad, root, tmp / "calib")
    assert e.value.code == "QUANT_CALIB_ON_EVAL_SPLIT"


def test_calibration_cache_is_reused(setup):
    cfg, q, _, root, tmp, fp32 = setup
    path, meta = calibrate(fp32, cfg, q, root, tmp / "calib")
    assert meta["n_images"] == 6 and json.loads(path.read_text())
    mtime = path.stat().st_mtime_ns
    assert calibrate(fp32, cfg, q, root, tmp / "calib")[0].stat().st_mtime_ns == mtime


def test_quantizing_a_quantized_model_is_refused(setup):
    cfg, q, rules, _, tmp, _ = setup
    path, _ = _run(setup)
    with pytest.raises(QuantError) as e:
        quantize(path, cfg, q, ({}, {}), rules, no_exclusions(cfg.name), tmp / "onnx")
    assert e.value.code == "QUANT_INPUT_NOT_FP32"


def test_ort_runs_the_int8_model(setup):
    cfg, *_, root, _, _ = setup
    path, _ = _run(setup)
    backend = get_backend("ort_cpu")
    backend.load(path, cfg)
    assert backend.metadata()["precision"] == "int8"
    assert backend.metadata()["artifact_id"] == load_sidecar(path)["artifact_id"]
    sample = next(iter_samples(root, build_manifest(
        ManifestSpec("m", "val", None, None), list_image_ids(root, "val"))))  # fmt: skip
    assert np.isfinite(backend.infer(preprocess(sample.image, cfg))).all()


def test_ort_entropy_defaults_degenerate_to_the_full_range(tiny_onnx, tmp_path):
    """Pins the finding behind num_bins: 2048 in configs/quant/int8_entropy.yaml. With ORT's
    defaults (128 bins, 128 quantized bins) the KL search has one candidate threshold,
    the full histogram range, so 'entropy' is minmax."""
    rng = np.random.default_rng(0)
    data = [rng.standard_normal((1, 3, 32, 64)).astype(np.float32) * 3 for _ in range(3)]

    def ranges(method, opts):
        cal = create_calibrator(tiny_onnx, ["Conv", "MatMul"], str(tmp_path / f"{method}.onnx"),
                                calibrate_method=method, extra_options=opts)  # fmt: skip
        it = iter(data)
        cal.collect_data(type("R", (), {"get_next": lambda s: (
            None if (x := next(it, None)) is None else {"input": x})})())  # fmt: skip
        return {k: float(np.max(np.abs(v.range_value))) for k, v in cal.compute_data().data.items()}

    default = ranges(CalibrationMethod.Entropy, {"symmetric": True})
    tuned = ranges(CalibrationMethod.Entropy, {"symmetric": True, "num_bins": 2048,
                                               "num_quantized_bins": 128})  # fmt: skip
    minmax = ranges(CalibrationMethod.MinMax, {"symmetric": True})
    # Default "entropy" thresholds sit at the histogram edge, i.e. the observed max abs.
    assert all(np.isclose(default[k], minmax[k], rtol=1e-6) for k in minmax)
    # With 2048 bins the KL search clips at least some tensors below their max.
    assert any(tuned[k] < 0.99 * minmax[k] for k in minmax)
