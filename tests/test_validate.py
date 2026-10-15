"""One test per validator finding code, each on a small hand-built graph."""

from dataclasses import replace

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from edgeinfer.cli import main
from edgeinfer.config import load_model_config
from edgeinfer.export.onnx_export import CONTRACT_KEY, canonical_json, io_contract
from edgeinfer.validate.onnx_validator import validate_onnx
from tests.conftest import TINY_CONFIG

# Input 1x3x8x16 -> 1x1 conv with stride 4 -> 1x19x2x4 logits (output stride 4).
CFG = replace(
    load_model_config(TINY_CONFIG),
    input=replace(load_model_config(TINY_CONFIG).input, height=8, width=16),
)


def _graph_model(
    nodes=None, inits=None, out_shape=(1, 19, 2, 4), opset=20, contract=True, in_name="input"
):
    w = numpy_helper.from_array(np.full((19, 3, 1, 1), 0.1, np.float32), "w")
    nodes = nodes or [helper.make_node("Conv", [in_name, "w"], ["logits"], strides=[4, 4])]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info(in_name, TensorProto.FLOAT, [1, 3, 8, 16])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, list(out_shape))],
        [w, *(inits or [])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    if contract:
        helper.set_model_props(model, {CONTRACT_KEY: canonical_json(io_contract(CFG))})
    return model


def _codes(tmp_path, model, cfg=CFG):
    path = tmp_path / "m.onnx"
    onnx.save(model, str(path))
    _, findings = validate_onnx(path, cfg)
    return {f["code"] for f in findings}


def test_clean_model_passes(tmp_path):
    assert _codes(tmp_path, _graph_model()) == set()


def test_exported_fixture_passes(tiny_onnx):
    metrics, findings = validate_onnx(tiny_onnx, load_model_config(TINY_CONFIG))
    assert findings == [] and metrics["n_errors"] == 0
    assert metrics["op_counts"]["LayerNormalization"] == 1


def test_checker_failure(tmp_path):
    bad = _graph_model(nodes=[helper.make_node("Conv", ["input", "missing"], ["logits"])])
    assert "VALIDATE_CHECKER_FAILED" in _codes(tmp_path, bad)


def test_shape_inference_failure(tmp_path):
    # Declared output shape contradicts what the conv produces.
    assert "VALIDATE_SHAPE_INFERENCE_FAILED" in _codes(
        tmp_path, _graph_model(out_shape=(1, 19, 3, 4))
    )


def test_dynamic_dim(tmp_path):
    model = _graph_model()
    model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch"
    assert "VALIDATE_DYNAMIC_DIM" in _codes(tmp_path, model)


def test_opset_out_of_range(tmp_path):
    assert "VALIDATE_OPSET_UNSUPPORTED" in _codes(tmp_path, _graph_model(opset=8))


def test_unsupported_op(tmp_path):
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["c"], strides=[4, 4]),
        helper.make_node("Det", ["c"], ["d"]),  # listed as unsupported by the parser
        helper.make_node("Identity", ["c"], ["logits"]),
    ]
    assert "VALIDATE_UNSUPPORTED_OP" in _codes(tmp_path, _graph_model(nodes=nodes))


@pytest.mark.parametrize(
    "attrs",
    [{"antialias": 1}, {"mode": "cubic"}, {"coordinate_transformation_mode": "tf_crop_and_resize"}],
)
def test_resize_restrictions(tmp_path, attrs):
    scales = numpy_helper.from_array(np.array([1, 1, 0.25, 0.25], np.float32), "scales")
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["c"], strides=[1, 1]),
        helper.make_node("Resize", ["c", "", "scales"], ["logits"], **attrs),
    ]
    model = _graph_model(nodes=nodes, inits=[scales])
    assert "VALIDATE_OP_RESTRICTION" in _codes(tmp_path, model)


def test_reduce_axes_must_be_initializer(tmp_path):
    nodes = [
        helper.make_node("Conv", ["input", "w"], ["c"], strides=[4, 4]),
        helper.make_node("Shape", ["c"], ["s"]),
        helper.make_node("Slice", ["s", "zero", "one"], ["axes"]),
        helper.make_node("ReduceMean", ["c", "axes"], ["r"], keepdims=1),
        helper.make_node("Add", ["c", "r"], ["logits"]),
    ]
    inits = [
        numpy_helper.from_array(np.array([0], np.int64), "zero"),
        numpy_helper.from_array(np.array([1], np.int64), "one"),
    ]
    assert "VALIDATE_OP_RESTRICTION" in _codes(tmp_path, _graph_model(nodes=nodes, inits=inits))


def test_asymmetric_quantization_zero_point(tmp_path):
    nodes = [
        helper.make_node("QuantizeLinear", ["input", "s", "zp"], ["q"]),
        helper.make_node("DequantizeLinear", ["q", "s", "zp"], ["dq"]),
        helper.make_node("Conv", ["dq", "w"], ["logits"], strides=[4, 4]),
    ]
    inits = [
        numpy_helper.from_array(np.array(0.1, np.float32), "s"),
        numpy_helper.from_array(np.array(3, np.int8), "zp"),
    ]
    assert "VALIDATE_OP_RESTRICTION" in _codes(tmp_path, _graph_model(nodes=nodes, inits=inits))


def test_io_mismatch(tmp_path):
    assert "VALIDATE_IO_MISMATCH" in _codes(tmp_path, _graph_model(in_name="pixel_values"))
    other = replace(CFG, num_classes=20)
    assert "VALIDATE_IO_MISMATCH" in _codes(tmp_path, _graph_model(), cfg=other)


@pytest.mark.parametrize(("value", "code"), [
    (np.nan, "VALIDATE_INITIALIZER_NONFINITE"),
    (np.inf, "VALIDATE_INITIALIZER_NONFINITE"),
    (1e6, "VALIDATE_INITIALIZER_FP16_RANGE"),
])  # fmt: skip
def test_weight_checks(tmp_path, value, code):
    model = _graph_model()
    w = numpy_helper.to_array(model.graph.initializer[0]).copy()
    w[0, 0, 0, 0] = value
    model.graph.initializer[0].CopyFrom(numpy_helper.from_array(w, "w"))
    assert code in _codes(tmp_path, model)


def test_metadata_missing(tmp_path):
    assert "VALIDATE_METADATA_MISSING" in _codes(tmp_path, _graph_model(contract=False))


def test_channel_order_swap_is_a_preprocessing_mismatch(tmp_path):
    bgr = replace(CFG, input=replace(CFG.input, channel_order="BGR"))
    path = tmp_path / "m.onnx"
    onnx.save(_graph_model(), str(path))
    _, findings = validate_onnx(path, bgr)
    (f,) = findings
    assert f["code"] == "VALIDATE_PREPROC_MISMATCH" and f["where"] == "input.channel_order"
    assert "'RGB'" in f["message"] and "'BGR'" in f["message"]


def test_validate_cli_exit_codes(tmp_path, tiny_onnx):
    out = tmp_path / "r.json"
    assert main(["validate", "--model", str(TINY_CONFIG), "--artifact", str(tiny_onnx),
                 "--out", str(out)]) == 0  # fmt: skip
    bgr = tmp_path / "bgr.yaml"
    bgr.write_text(TINY_CONFIG.read_text().replace("channel_order: RGB", "channel_order: BGR"))
    assert main(["validate", "--model", str(bgr), "--artifact", str(tiny_onnx),
                 "--out", str(out)]) == 1  # fmt: skip
