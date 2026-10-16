"""Engine layer summaries from hand-written inspector JSON (format from the TensorRT 11 docs,
"Engine Inspector"); checked against real engine files in the first C5 GPU session."""

import pytest
from onnx import TensorProto, helper

from edgeinfer.backends.trt_layers import layer_summary, profile_summary, quantized_onnx_nodes


def _t(dtype, fmt="(Linear) Row major linear format"):
    return {"Name": "t", "Dimensions": [1, 8, 4, 4], "Datatype": dtype, "Format": fmt}


def _layer(name, ltype, ins, outs, consts=(), onnx_names=()):
    meta = "\x1f".join(f"[ONNX Layer: {n}]" for n in onnx_names)
    return {"Name": name, "LayerType": ltype, "Inputs": [_t(d) for d in ins],
            "Constants": [_t(d) for d in consts], "Outputs": [_t(d) for d in outs],
            "TacticName": "x", "StreamId": 0, "Metadata": meta}  # fmt: skip


DOC = {"Layers": [
    _layer("Reformatting CopyNode for Input Tensor 0 to conv0", "Reformat", ["Float"], ["Int8"]),
    _layer("conv0", "CaskConvolution", ["Int8"], ["Half"], ["Int8", "Float"], ["node_Conv_0"]),
    _layer("conv1", "CaskConvolution", ["Half"], ["Half"], ["Half"], ["node_Conv_1"]),
    _layer("ln", "kgen", ["Half"], ["Half"], [], ["node_LayerNorm_2"]),
    _layer("reformat_out", "Reformat", ["Half"], ["Float"]),
    "a layer reported by name only",
]}  # fmt: skip


def test_layer_summary_counts_and_fusion_check():
    metrics, findings = layer_summary(DOC, ["node_Conv_0", "node_Conv_1", "node_MatMul_9"])
    assert metrics["n_layers"] == 6 and metrics["layers_without_detail"] == 1
    assert metrics["by_precision"] == {"int8": 1, "Half": 2}
    assert metrics["reformat_layers"] == 2
    assert metrics["reformats"] == {"Float->Int8": 1, "Half->Float": 1}
    q = metrics["quantized_onnx_nodes"]
    assert q == {
        "expected": 3,
        "int8": 1,
        "not_int8": ["node_Conv_1"],
        "unmapped": ["node_MatMul_9"],
    }
    codes = {f["code"]: f for f in findings}
    assert codes["BUILD_QDQ_UNFUSED"]["severity"] == "warn"
    assert codes["BUILD_QDQ_UNFUSED"]["where"] == "node_Conv_1"
    assert codes["BUILD_LAYER_INFO_UNMAPPED"]["severity"] == "info"


def test_no_fusion_check_without_quantized_nodes():
    metrics, findings = layer_summary(DOC)
    assert "quantized_onnx_nodes" not in metrics and findings == []


def test_quantized_onnx_nodes():
    nodes = [
        helper.make_node("DequantizeLinear", ["xq", "s", "z"], ["x"], name="dq_x"),
        helper.make_node("Conv", ["x", "w"], ["y"], name="conv_q"),
        helper.make_node("Conv", ["y", "w"], ["y2"], name="conv_float"),
    ]
    graph = helper.make_graph(
        nodes, "g", [], [helper.make_tensor_value_info("y2", TensorProto.FLOAT, None)]
    )
    assert quantized_onnx_nodes(helper.make_model(graph)) == ["conv_q"]


def test_profile_summary_adds_up():
    profile = [{"count": 50},
               {"name": "conv0", "averageMs": 1.1, "medianMs": 1.0, "percentage": 50},
               {"name": "conv1", "averageMs": 0.6, "medianMs": 0.5, "percentage": 25},
               {"name": "Reformatting CopyNode for Input Tensor 0 to conv0", "medianMs": 0.25},
               {"name": "not in the inspector file", "medianMs": 0.25}]  # fmt: skip
    s = profile_summary(profile, DOC)
    assert s["total_median_ms"] == pytest.approx(2.0)
    assert {k: v["ms"] for k, v in s["by_kind"].items()} == {
        "int8": 1.0,
        "fp16": 0.5,
        "reformat": 0.25,
        "unknown": 0.25,
    }
    assert sum(v["share"] for v in s["by_kind"].values()) == pytest.approx(1.0)
    assert s["by_type"]["CaskConvolution"]["ms"] == pytest.approx(1.5)
