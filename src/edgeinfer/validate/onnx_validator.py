"""Check an exported ONNX model before it is built into an engine.

Each check emits findings with a stable code (docs/report-codes.md). The model is
only read, never modified. Checks:
    structure     onnx.checker, strict shape inference, static shapes
    TensorRT      opset range, op support, coded op restrictions (configs/supported_ops.yaml)
    Q/DQ          signed INT8, per-channel weights, float bias (D-016)
    I/O           input/output names, dtypes, shapes vs the model config
    weights       NaN/Inf, values outside the FP16 range
    contract      config preprocessing/input/postprocessing vs the copy embedded at export
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import yaml
from onnx import numpy_helper, shape_inference

from edgeinfer.config import CONFIG_DIR, ModelConfig
from edgeinfer.export.onnx_export import CONTRACT_KEY, INPUT_NAME, OUTPUT_NAME, io_contract

SUPPORTED_OPS_PATH = CONFIG_DIR / "supported_ops.yaml"
FP16_MAX = 65504.0

# Coded subset of the parser restrictions in supported_ops.yaml.
_RESIZE_MODES = {"nearest", "linear"}
_RESIZE_COORD_MODES = {
    "half_pixel", "pytorch_half_pixel", "tf_half_pixel_for_nn", "asymmetric", "align_corners",
}  # fmt: skip
_AXES_INPUT_OPS = {"ReduceMean", "ReduceSum", "ReduceMax", "ReduceMin", "ReduceProd"}
_CONTRACT_PREPROC_KEYS = ("input", "preprocess", "postprocess")

Finding = dict[str, str]


def load_supported_ops(path: Path = SUPPORTED_OPS_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text())


def _finding(severity: str, code: str, message: str, where: str = "") -> Finding:
    f = {"severity": severity, "code": code, "message": message}
    if where:
        f["where"] = where
    return f


def _dims(vi: onnx.ValueInfoProto) -> list[int | str]:
    return [d.dim_param or d.dim_value for d in vi.type.tensor_type.shape.dim]


def _dtype(vi: onnx.ValueInfoProto) -> str:
    return onnx.TensorProto.DataType.Name(vi.type.tensor_type.elem_type).lower()


def _attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}


def _check_structure(model: onnx.ModelProto) -> tuple[onnx.ModelProto | None, list[Finding]]:
    findings: list[Finding] = []
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        findings.append(_finding("error", "VALIDATE_CHECKER_FAILED", str(e).splitlines()[0]))
    try:
        inferred = shape_inference.infer_shapes(model, strict_mode=True)
    except shape_inference.InferenceError as e:
        findings.append(
            _finding("error", "VALIDATE_SHAPE_INFERENCE_FAILED", str(e).splitlines()[0])
        )
        return None, findings
    g = inferred.graph
    for vi in [*g.input, *g.output]:
        if any(not isinstance(d, int) or d <= 0 for d in _dims(vi)):
            findings.append(_finding(
                "warn", "VALIDATE_DYNAMIC_DIM",
                f"graph I/O {vi.name!r} has non-static shape {_dims(vi)}; the engine is "
                "built for one fixed shape", vi.name,
            ))  # fmt: skip
    io_names = {vi.name for vi in [*g.input, *g.output]}
    dynamic = [vi.name for vi in g.value_info if vi.name not in io_names
               and any(not isinstance(d, int) or d <= 0 for d in _dims(vi))]  # fmt: skip
    if dynamic:
        findings.append(_finding(
            "warn", "VALIDATE_DYNAMIC_DIM",
            f"{len(dynamic)} intermediate tensors have non-static shapes, e.g. {dynamic[:3]}",
            dynamic[0],
        ))  # fmt: skip
    return inferred, findings


def _check_ops(model: onnx.ModelProto, supported: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []
    lo, hi = supported["opset"]["min"], supported["opset"]["max"]
    src = f"TensorRT {supported['tensorrt_version']} parser ({supported['status']} list)"
    for imp in model.opset_import:
        if imp.domain in ("", "ai.onnx") and not lo <= imp.version <= hi:
            findings.append(_finding(
                "error", "VALIDATE_OPSET_UNSUPPORTED",
                f"opset {imp.version} is outside {lo}-{hi} supported by the {src}", "opset",
            ))  # fmt: skip
    counts = Counter((n.domain or "", n.op_type) for n in model.graph.node)
    for (domain, op), n in sorted(counts.items()):
        if domain not in ("", "ai.onnx"):
            findings.append(_finding(
                "error", "VALIDATE_UNSUPPORTED_OP",
                f"{n}x {domain}::{op}: custom-domain ops need a TensorRT plugin", op,
            ))  # fmt: skip
        elif op not in supported["ops"]:
            why = ("listed as unsupported" if op in supported["unsupported"]
                   else "not in the operator list")  # fmt: skip
            findings.append(_finding(
                "error", "VALIDATE_UNSUPPORTED_OP", f"{n}x {op}: {why} for the {src}", op,
            ))  # fmt: skip
    return findings


def _check_restrictions(model: onnx.ModelProto) -> list[Finding]:
    """The parser restrictions that can be checked statically and matter for our models."""
    inits = {t.name: t for t in model.graph.initializer}
    used = {i for n in model.graph.node for i in n.input} | {o.name for o in model.graph.output}
    findings: list[Finding] = []

    def bad(node: onnx.NodeProto, msg: str) -> None:
        where = node.name or node.output[0]
        findings.append(
            _finding("error", "VALIDATE_OP_RESTRICTION", f"{node.op_type} {where}: {msg}", where)
        )

    for node in model.graph.node:
        a = _attrs(node)
        if node.op_type == "Resize":
            mode = a.get("mode", b"nearest").decode()
            coord = a.get("coordinate_transformation_mode", b"half_pixel").decode()
            policy = a.get("keep_aspect_ratio_policy", b"stretch").decode()
            if a.get("antialias", 0):
                bad(node, "antialias is not supported")
            if mode not in _RESIZE_MODES:
                bad(node, f"mode {mode!r} is not supported")
            if coord not in _RESIZE_COORD_MODES:
                bad(node, f"coordinate_transformation_mode {coord!r} is not supported")
            if policy != "stretch":
                bad(node, f"keep_aspect_ratio_policy {policy!r} is not supported")
        elif node.op_type in _AXES_INPUT_OPS and len(node.input) > 1 and node.input[1]:
            if node.input[1] not in inits:
                bad(node, "axes must be an initializer")
        elif node.op_type in ("QuantizeLinear", "DequantizeLinear") and len(node.input) > 2:
            zp = inits.get(node.input[2])
            if zp is None or np.any(numpy_helper.to_array(zp) != 0):
                bad(node, "zero point must be an all-zero initializer (symmetric quantization)")
        elif node.op_type == "LayerNormalization":
            if any(o and o in used for o in node.output[1:]):
                bad(node, "only the first output (Y) is supported")
    return findings


def _check_qdq(model: onnx.ModelProto) -> tuple[list[Finding], dict[str, Any]]:
    """Q/DQ format for TensorRT explicit quantization (D-016): activations and weights in
    signed INT8 (zero points are checked in _check_restrictions), weights per-channel,
    bias not quantized."""
    inits = {t.name: t for t in model.graph.initializer}
    consumers: dict[str, list[tuple[onnx.NodeProto, int]]] = {}
    for node in model.graph.node:
        for i, name in enumerate(node.input):
            consumers.setdefault(name, []).append((node, i))

    def qtype(node: onnx.NodeProto) -> str:
        """Quantized dtype of a Q output / DQ input. ONNX default without a zero point: uint8."""
        if node.op_type == "DequantizeLinear" and node.input[0] in inits:
            return onnx.TensorProto.DataType.Name(inits[node.input[0]].data_type)
        if len(node.input) > 2 and node.input[2] in inits:
            return onnx.TensorProto.DataType.Name(inits[node.input[2]].data_type)
        out = _attrs(node).get("output_dtype", 0)
        return onnx.TensorProto.DataType.Name(out) if out else "UINT8"

    findings: list[Finding] = []
    wrong: Counter[str] = Counter()
    first_wrong: dict[str, str] = {}
    per_tensor_weights: list[str] = []
    n_q = n_dq = 0
    for node in model.graph.node:
        if node.op_type not in ("QuantizeLinear", "DequantizeLinear"):
            continue
        n_q += node.op_type == "QuantizeLinear"
        n_dq += node.op_type == "DequantizeLinear"
        dtype = qtype(node)
        uses = consumers.get(node.output[0], [])
        if dtype == "INT32" and all(c.op_type in ("Conv", "Gemm") and i == 2 for c, i in uses):
            findings.append(_finding(
                "warn", "VALIDATE_QDQ_BIAS",
                f"{node.name}: bias is quantized to INT32; the TensorRT format keeps bias in "
                "float (D-016)", node.name,
            ))  # fmt: skip
            continue
        if dtype != "INT8":
            wrong[dtype] += 1
            first_wrong.setdefault(dtype, node.name or node.output[0])
        is_weight = node.op_type == "DequantizeLinear" and node.input[0] in inits
        scale = inits.get(node.input[1])
        if (
            is_weight
            and scale is not None
            and numpy_helper.to_array(scale).size == 1
            and any(c.op_type in ("Conv", "MatMul", "Gemm") and i == 1 for c, i in uses)
        ):
            per_tensor_weights.append(node.name or node.output[0])  # fmt: skip
    for dtype, n in sorted(wrong.items()):
        findings.append(_finding(
            "error", "VALIDATE_QDQ_DTYPE",
            f"{n} Q/DQ nodes use {dtype}; TensorRT explicit quantization needs signed INT8 "
            "(check activation_type/weight_type in the quant config)", first_wrong[dtype],
        ))  # fmt: skip
    if per_tensor_weights:
        findings.append(_finding(
            "warn", "VALIDATE_QDQ_WEIGHT_PER_TENSOR",
            f"{len(per_tensor_weights)} weights are quantized per-tensor; per-channel keeps "
            "small-magnitude output channels accurate", per_tensor_weights[0],
        ))  # fmt: skip
    return findings, {"n_quantize_linear": n_q, "n_dequantize_linear": n_dq}


def _check_io(graph: onnx.GraphProto, cfg: ModelConfig) -> list[Finding]:
    h, w, s = cfg.input.height, cfg.input.width, cfg.output_stride
    expected = {
        "input": (INPUT_NAME, "float", [1, 3, h, w]),
        "output": (OUTPUT_NAME, "float", [1, cfg.num_classes, h // s, w // s]),
    }
    findings: list[Finding] = []
    for kind, values in (("input", list(graph.input)), ("output", list(graph.output))):
        name, dtype, shape = expected[kind]
        got = [(vi.name, _dtype(vi), _dims(vi)) for vi in values]
        if got != [(name, dtype, shape)]:
            findings.append(_finding(
                "error", "VALIDATE_IO_MISMATCH",
                f"model {kind}s {got}; config {cfg.name} expects [{(name, dtype, shape)}]", kind,
            ))  # fmt: skip
    return findings


def _float_tensors(model: onnx.ModelProto):
    for t in model.graph.initializer:
        yield t.name, t
    for node in model.graph.node:
        if node.op_type == "Constant":
            for a in node.attribute:
                if a.name == "value":
                    yield node.output[0], a.t


def _check_weights(model: onnx.ModelProto) -> tuple[list[Finding], dict[str, Any]]:
    findings: list[Finding] = []
    n_params = 0
    max_abs = 0.0
    for name, t in _float_tensors(model):
        arr = numpy_helper.to_array(t)
        if not np.issubdtype(arr.dtype, np.floating):
            continue
        n_params += arr.size
        finite = np.isfinite(arr)
        if not finite.all():
            findings.append(_finding(
                "error", "VALIDATE_INITIALIZER_NONFINITE",
                f"{name}: {int((~finite).sum())} of {arr.size} values are NaN/Inf", name,
            ))  # fmt: skip
        if arr.size and finite.any():
            peak = float(np.abs(arr[finite]).max())
            max_abs = max(max_abs, peak)
            if peak > FP16_MAX:
                findings.append(_finding(
                    "warn", "VALIDATE_INITIALIZER_FP16_RANGE",
                    f"{name}: max |w| {peak:.4g} exceeds FP16 max {FP16_MAX}; it would "
                    "overflow in an FP16 engine", name,
                ))  # fmt: skip
    return findings, {"n_float_params": int(n_params), "max_abs_weight": max_abs}


def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(d, dict):
        out: dict[str, Any] = {}
        for k, v in d.items():
            out.update(_flatten(v, f"{prefix}{k}."))
        return out
    return {prefix.rstrip("."): d}


def _check_contract(model: onnx.ModelProto, cfg: ModelConfig) -> list[Finding]:
    props = {p.key: p.value for p in model.metadata_props}
    if CONTRACT_KEY not in props:
        return [_finding(
            "error", "VALIDATE_METADATA_MISSING",
            f"no {CONTRACT_KEY!r} metadata; can't check the config against the model "
            "(export with `edgeinfer export`)", CONTRACT_KEY,
        )]  # fmt: skip
    embedded = _flatten(json.loads(props[CONTRACT_KEY]))
    current = _flatten(io_contract(cfg))
    findings: list[Finding] = []
    for key in sorted(set(embedded) | set(current)):
        if embedded.get(key) != current.get(key):
            code = ("VALIDATE_PREPROC_MISMATCH" if key.split(".")[0] in _CONTRACT_PREPROC_KEYS
                    else "VALIDATE_IO_MISMATCH")  # fmt: skip
            findings.append(_finding(
                "error", code,
                f"{key}: model was exported with {embedded.get(key)!r}, config has "
                f"{current.get(key)!r}", key,
            ))  # fmt: skip
    return findings


def validate_onnx(
    path: Path, cfg: ModelConfig, supported: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[Finding]]:
    """Returns (metrics, findings). The model passes if no finding has severity error."""
    supported = supported or load_supported_ops()
    model = onnx.load(str(path))
    inferred, findings = _check_structure(model)
    findings += _check_ops(model, supported)
    findings += _check_restrictions(model)
    qdq_findings, qdq_metrics = _check_qdq(model)
    findings += qdq_findings
    findings += _check_io((inferred or model).graph, cfg)
    weight_findings, weight_metrics = _check_weights(model)
    findings += weight_findings
    findings += _check_contract(model, cfg)

    ops = Counter(n.op_type for n in model.graph.node)
    metrics = {
        "opset": {imp.domain or "ai.onnx": imp.version for imp in model.opset_import},
        "n_nodes": len(model.graph.node),
        "op_counts": dict(sorted(ops.items())),
        "inputs": {vi.name: _dims(vi) for vi in model.graph.input},
        "outputs": {vi.name: _dims(vi) for vi in model.graph.output},
        **weight_metrics,
        **qdq_metrics,
        "n_errors": sum(f["severity"] == "error" for f in findings),
        "n_warnings": sum(f["severity"] == "warn" for f in findings),
    }
    return metrics, findings


def supported_ops_ref(supported: dict[str, Any]) -> dict[str, Any]:
    """What the op checks were run against, for the report's thresholds block."""
    return {k: supported[k] for k in ("tensorrt_version", "source_commit", "status", "opset")}
