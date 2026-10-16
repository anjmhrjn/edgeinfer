"""INT8 Q/DQ -> INT8 Q/DQ with FP16 float parts (D-031).

Most tests replace ModelOpt with a fake runner, so they run in CI without the ModelOpt
environment. Tests marked `modelopt` run the real `convert_to_f16` in tools/modelopt/.
"""

import json
import shutil
from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from edgeinfer.artifacts import load_sidecar
from edgeinfer.backends import get_backend
from edgeinfer.cli import main
from edgeinfer.config import load_model_config
from edgeinfer.quant import QuantError
from edgeinfer.quant.config import load_group_rules, load_quant_config
from edgeinfer.quant.mixed import convert_int8_fp16, int8_fp16_artifact_id, scale_rounding
from edgeinfer.validate.onnx_validator import validate_onnx
from tests.conftest import TINY_CONFIG
from tests.test_quant import TINY_GROUPS, _manifest, _run


@pytest.fixture
def setup(fake_cityscapes, tmp_path, tiny_onnx):
    """As in test_quant: tiny fixture, entropy INT8 config calibrated on the fake train split."""
    cfg = load_model_config(TINY_CONFIG)
    q = load_quant_config("int8_entropy")
    calib = _manifest(fake_cityscapes, tmp_path, "train")
    q = replace(q, calibration=replace(q.calibration, manifest=str(calib)))
    rules = load_group_rules("tiny_fixture", TINY_GROUPS)
    return cfg, q, rules, fake_cityscapes, tmp_path, tiny_onnx


class FakeConvert:
    """Stands in for the subprocess: copies the model and drops its metadata, as
    convert_to_f16 does. `drop_qdq` removes the first Q/DQ pair to fake a broken conversion."""

    def __init__(self, drop_qdq=False):
        self.calls = []
        self.drop_qdq = drop_qdq

    def __call__(self, args):
        self.calls.append(args)
        model = onnx.load(args["onnx_path"])
        del model.metadata_props[:]
        if self.drop_qdq:
            nodes = list(model.graph.node)
            q = next(n for n in nodes if n.op_type == "QuantizeLinear")
            dq = next(n for n in nodes if n.input[:1] == [q.output[0]])
            for n in nodes:
                n.input[:] = [q.input[0] if i == dq.output[0] else i for i in n.input]
            model.graph.node.remove(q)
            model.graph.node.remove(dq)
        onnx.save(model, args["output_path"])
        return {"versions": {"nvidia-modelopt": "fake"}, "seconds": 0.0}


def test_int8_fp16_artifact_inherits_the_quantization(setup):
    int8_path, int8_meta = _run(setup)
    cfg, *_, tmp, _ = setup
    fake = FakeConvert()
    path, meta = convert_int8_fp16(int8_path, cfg, tmp / "onnx", runner=fake)
    assert fake.calls[0]["keep_io_types"] is True
    assert meta["precision"] == "int8_fp16" and path.name.startswith("tiny_fixture__int8_fp16__")
    assert meta["artifact_id"] == int8_fp16_artifact_id(int8_meta["artifact_id"])
    assert meta["parent_artifact_id"] == int8_meta["artifact_id"]
    assert meta["fp32_artifact_id"] == int8_meta["parent_artifact_id"]
    for key in ("quant_config", "calibration", "exclusions", "excluded_nodes"):
        assert meta[key] == int8_meta[key]
    assert meta["placement"]["quantized_nodes"] == int8_meta["placement"]["quantized_nodes"]
    assert meta["scale_rounding"]["max_rel_change"] == 0.0  # the fake changes nothing
    assert meta["versions"]["modelopt_env.nvidia-modelopt"] == "fake"
    props = {p.key: p.value for p in onnx.load(str(path)).metadata_props}
    assert props["edgeinfer.artifact_id"] == meta["artifact_id"]
    assert props["edgeinfer.precision"] == "int8_fp16"
    assert "edgeinfer.contract" in props  # restored after the converter dropped it
    # Same id -> reused, converter not called again.
    again, _ = convert_int8_fp16(int8_path, cfg, tmp / "onnx", runner=fake)
    assert again == path and len(fake.calls) == 1


def test_only_int8_models_are_converted(setup):
    cfg, *_, tmp, fp32 = setup
    with pytest.raises(QuantError) as e:
        convert_int8_fp16(fp32, cfg, tmp / "onnx", runner=FakeConvert())
    assert e.value.code == "QUANT_INPUT_NOT_INT8"


def test_a_conversion_that_changes_placement_is_refused(setup):
    int8_path, _ = _run(setup)
    cfg, *_, tmp, _ = setup
    with pytest.raises(QuantError) as e:
        convert_int8_fp16(int8_path, cfg, tmp / "onnx", runner=FakeConvert(drop_qdq=True))
    assert e.value.code == "QUANT_PLACEMENT_MISMATCH"
    assert not list((tmp / "onnx" / cfg.name).glob("*int8_fp16*"))  # nothing left behind


def _dq_model(scale, dtype):
    scale = np.asarray(scale, dtype=dtype)
    nodes = [helper.make_node("DequantizeLinear", ["w", "s", "z"], ["y"], axis=0, name="dq")]
    inits = [
        numpy_helper.from_array(np.ones((scale.size, 1), np.int8), "w"),
        numpy_helper.from_array(scale, "s"),
        numpy_helper.from_array(np.zeros(scale.size, np.int8), "z"),
    ]
    out_type = TensorProto.FLOAT16 if dtype == np.float16 else TensorProto.FLOAT
    graph = helper.make_graph(nodes, "g", [], [helper.make_tensor_value_info("y", out_type, None)],
                              inits)  # fmt: skip
    return helper.make_model(graph)


def test_scale_rounding_counts_fp16_subnormals():
    scales = [0.5, 1e-3, 2e-6]  # 2e-6 is below FP16's smallest normal (6.1e-5)
    stats = scale_rounding(_dq_model(scales, np.float32), _dq_model(scales, np.float16))
    assert stats["n_values"] == 3 and stats["subnormal_in_fp16"] == 1
    assert stats["max_rel_change_at"] == "y"
    expected = max(abs(float(np.float16(s)) - s) / s for s in scales)
    assert stats["max_rel_change"] == pytest.approx(expected)
    assert stats["max_rel_change"] > 1e-3  # the subnormal loses precision


def test_ort_runs_int8_fp16_unoptimized(setup):
    int8_path, _ = _run(setup)
    cfg, *_, tmp, _ = setup
    path, _ = convert_int8_fp16(int8_path, cfg, tmp / "onnx", runner=FakeConvert())
    backend = get_backend("ort_cpu")
    backend.load(path, cfg)
    assert backend.metadata()["session"]["graph_optimization_level"] == "ORT_DISABLE_ALL"
    backend.load(int8_path, cfg)
    assert backend.metadata()["session"]["graph_optimization_level"] == "ORT_ENABLE_ALL"


@pytest.mark.modelopt
def test_real_conversion_on_tiny_model(setup):
    """The real convert_to_f16 through `edgeinfer quantize --float-dtype fp16`, then the
    validator and ORT against the INT8 parent."""
    if shutil.which("uv") is None:
        pytest.fail("uv is needed to run the ModelOpt environment")
    cfg, q, _, root, tmp, fp32 = setup
    qpath = tmp / "q.yaml"
    qpath.write_text(json.dumps(q.to_dict()))  # JSON is valid YAML
    out = tmp / "report.json"
    argv = ["quantize", "--model", str(TINY_CONFIG), "--artifact", str(fp32),
            "--quant-config", str(qpath), "--groups", str(TINY_GROUPS),
            "--float-dtype", "fp16", "--root", str(root), "--calib-dir", str(tmp / "calib"),
            "--out-root", str(tmp / "onnx"), "--out", str(out)]  # fmt: skip
    assert main(argv) == 0
    report = json.loads(out.read_text())
    assert report["precision"] == "int8_fp16"
    metrics = report["metrics"]
    assert metrics["float_precision"]["fp32_nodes"] == []
    assert metrics["scale_rounding"]["max_rel_change"] < 1e-3
    path = tmp / "onnx" / cfg.name / metrics["file"]
    int8_path = next(p for p in (tmp / "onnx" / cfg.name).glob("*__int8__*.onnx"))
    model = onnx.load(str(path))
    inits = {t.name: t for t in model.graph.initializer}
    scale_types = {inits[n.input[1]].data_type for n in model.graph.node
                   if n.op_type == "DequantizeLinear"}  # fmt: skip
    assert scale_types == {TensorProto.FLOAT16}
    assert [i.type.tensor_type.elem_type for i in model.graph.input] == [TensorProto.FLOAT]
    assert load_sidecar(path)["parent_artifact_id"] == load_sidecar(int8_path)["artifact_id"]
    assert main(["validate", "--model", str(TINY_CONFIG), "--artifact", str(path),
                 "--out", str(tmp / "validate.json")]) == 0  # fmt: skip
    x = np.random.default_rng(0).standard_normal((1, 3, 32, 64)).astype(np.float32)
    ref, got = get_backend("ort_cpu"), get_backend("ort_cpu")
    ref.load(int8_path, cfg)
    got.load(path, cfg)
    a, b = ref.infer(x), got.infer(x)
    assert b.dtype == np.float32  # keep_io_types
    assert np.abs(b - a).max() < 5e-2 * max(1.0, float(np.abs(a).max()))


def test_validator_flags_fp32_compute_in_an_int8_fp16_model(setup):
    """The fake converter leaves the float parts in FP32: TensorRT would build them in FP32."""
    int8_path, _ = _run(setup)
    cfg, *_, tmp, _ = setup
    path, _ = convert_int8_fp16(int8_path, cfg, tmp / "onnx", runner=FakeConvert())
    _, findings = validate_onnx(path, cfg)
    codes = {f["code"]: f["severity"] for f in findings}
    assert codes.get("VALIDATE_FLOAT_PRECISION") == "error"
    # The INT8 parent is FP32 outside the Q/DQ by design: no finding.
    _, findings = validate_onnx(int8_path, cfg)
    assert "VALIDATE_FLOAT_PRECISION" not in {f["code"] for f in findings}
