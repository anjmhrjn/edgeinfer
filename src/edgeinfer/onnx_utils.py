"""Small ONNX graph helpers shared by the ORT backend, the validator and parity.

Intermediate tensors are identified by their ONNX tensor names, the key shared by
every backend (ORT now, TensorRT debug engines in C4).
"""

from __future__ import annotations

import copy
from math import prod

import onnx
from onnx import helper, shape_inference

# Ops whose outputs per-layer parity looks at by default: the compute layers and the
# points where branches merge. Elementwise/reshape glue between them adds tensors
# without adding information.
LAYER_OPS = frozenset(
    {"Conv", "MatMul", "Gemm", "LayerNormalization", "Softmax", "Resize", "Add", "Concat"}
)


def layer_tensors(model: onnx.ModelProto, ops: frozenset[str] = LAYER_OPS) -> list[str]:
    """Outputs of `ops` nodes, in graph (topological) order."""
    graph_outputs = {o.name for o in model.graph.output}
    return [
        n.output[0]
        for n in model.graph.node
        if n.op_type in ops and n.output[0] not in graph_outputs
    ]


def tensor_types(model: onnx.ModelProto) -> dict[str, onnx.TypeProto]:
    """Type (dtype + shape) of every node output, from strict shape inference."""
    inferred = shape_inference.infer_shapes(model, strict_mode=True)
    g = inferred.graph
    return {vi.name: vi.type for vi in [*g.value_info, *g.output, *g.input]}


def tensor_nbytes(t: onnx.TypeProto) -> int:
    tt = t.tensor_type
    dims = [d.dim_value for d in tt.shape.dim]
    if any(d <= 0 for d in dims):
        raise ValueError(f"tensor has a non-static shape {dims}")
    return prod(dims) * helper.tensor_dtype_to_np_dtype(tt.elem_type).itemsize


def with_outputs(model: onnx.ModelProto, names: list[str]) -> onnx.ModelProto:
    """A copy of `model` that also outputs the named intermediate tensors.
    The original is not modified (analysis tools are read-only)."""
    produced = {o for n in model.graph.node for o in n.output}
    unknown = [n for n in names if n not in produced]
    if unknown:
        raise KeyError(f"not node outputs in this model: {unknown[:5]}")
    types = tensor_types(model)
    out = copy.deepcopy(model)
    existing = {o.name for o in out.graph.output}
    for name in names:
        if name not in existing:
            out.graph.output.append(helper.make_value_info(name, types[name]))
    return out
