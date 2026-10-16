"""What a built engine actually runs: layer precisions, reformats and Q/DQ fusion (C5).

Reads the engine inspector's JSON (`<engine>.layers.json`, written by `edgeinfer build`
with detailed profiling verbosity) and, optionally, a trtexec per-layer profile. Pure
Python: it runs anywhere, so it's tested on the Mac with hand-written layer files.

Inspector format (TensorRT 11 docs, "Engine Inspector"): `{"Layers": [...]}`; each layer
has `Name`, `LayerType`, `Inputs` / `Constants` / `Outputs` (tensors with `Datatype` and
`Format`) and `Metadata`, which lists the ONNX nodes fused into the layer as
`[ONNX Layer: <name>]` joined by U+001F. NVIDIA says new keys may appear, so unknown keys
are ignored and a layer reported by name only (lower verbosity) is counted as unknown.

Q/DQ fusion check: with explicit quantization, TensorRT fuses each DequantizeLinear into
the Conv/MatMul that reads it and runs that layer on INT8 data. A quantized ONNX node whose
engine layer has no INT8 input or constant ran in float with the Q/DQ executed as separate
scale operations: accuracy as planned, speed not (`BUILD_QDQ_UNFUSED`).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any

import onnx

ONNX_LAYER = re.compile(r"\[ONNX Layer: ([^\]]+)\]")
INT8 = "Int8"
REFORMAT_TYPES = {"Reformat"}
QUANTIZED_OP_TYPES = ("Conv", "MatMul", "Gemm")

Finding = dict[str, str]


def _layers(doc: Any) -> list[Any]:
    return doc.get("Layers", []) if isinstance(doc, dict) else list(doc)


def _dtypes(layer: dict[str, Any], key: str) -> list[str]:
    return [str(t.get("Datatype", "?")) for t in layer.get(key, []) if isinstance(t, dict)]


def is_reformat(layer: dict[str, Any]) -> bool:
    name = str(layer.get("Name", ""))
    return layer.get("LayerType") in REFORMAT_TYPES or name.startswith("Reformatting CopyNode")


def uses_int8(layer: dict[str, Any]) -> bool:
    """An INT8 input or weight: the layer computes on INT8 data (fused Q/DQ)."""
    return INT8 in _dtypes(layer, "Inputs") + _dtypes(layer, "Constants")


def onnx_nodes(layer: dict[str, Any]) -> list[str]:
    return ONNX_LAYER.findall(str(layer.get("Metadata", "")))


def quantized_onnx_nodes(model: onnx.ModelProto) -> list[str]:
    """Conv/MatMul/Gemm nodes whose data input comes from a DequantizeLinear."""
    producer = {o: n.op_type for n in model.graph.node for o in n.output}
    return [n.name for n in model.graph.node
            if n.op_type in QUANTIZED_OP_TYPES and n.input
            and producer.get(n.input[0]) == "DequantizeLinear"]  # fmt: skip


def layer_summary(doc: Any, quantized: list[str] | None = None) -> tuple[dict, list[Finding]]:
    """Counts by layer type and precision, reformat layers, and the Q/DQ fusion check for
    the ONNX nodes in `quantized`."""
    by_type: Counter[str] = Counter()
    precision: Counter[str] = Counter()
    reformats: Counter[str] = Counter()
    int8_nodes: set[str] = set()
    float_nodes: set[str] = set()
    unknown = 0
    for layer in _layers(doc):
        if not isinstance(layer, dict):
            unknown += 1
            continue
        by_type[str(layer.get("LayerType", "?"))] += 1
        if is_reformat(layer):
            src, dst = _dtypes(layer, "Inputs")[:1], _dtypes(layer, "Outputs")[:1]
            reformats[f"{(src or ['?'])[0]}->{(dst or ['?'])[0]}"] += 1
            continue
        if uses_int8(layer):
            precision["int8"] += 1
            int8_nodes.update(onnx_nodes(layer))
        else:
            out = _dtypes(layer, "Outputs")
            precision[out[0] if out else "?"] += 1
            float_nodes.update(onnx_nodes(layer))
    metrics: dict[str, Any] = {
        "n_layers": len(_layers(doc)),
        "by_type": dict(by_type.most_common()),
        "by_precision": dict(precision.most_common()),
        "reformat_layers": sum(reformats.values()),
        "reformats": dict(reformats.most_common()),
        "layers_without_detail": unknown,
    }
    findings: list[Finding] = []
    if quantized is not None:
        not_int8 = [n for n in quantized if n not in int8_nodes and n in float_nodes]
        unmapped = [n for n in quantized if n not in int8_nodes and n not in float_nodes]
        metrics["quantized_onnx_nodes"] = {"expected": len(quantized),
                                           "int8": len(quantized) - len(not_int8) - len(unmapped),
                                           "not_int8": not_int8, "unmapped": unmapped}  # fmt: skip
        if not_int8:
            findings.append({
                "severity": "warn", "code": "BUILD_QDQ_UNFUSED",
                "message": f"{len(not_int8)}/{len(quantized)} quantized ONNX nodes run in an "
                           "engine layer without INT8 inputs (Q/DQ not fused)",
                "where": not_int8[0],
            })  # fmt: skip
        if unmapped:
            findings.append({
                "severity": "info", "code": "BUILD_LAYER_INFO_UNMAPPED",
                "message": f"{len(unmapped)}/{len(quantized)} quantized ONNX nodes aren't named "
                           "in any engine layer's metadata, so their fusion can't be checked",
                "where": unmapped[0],
            })  # fmt: skip
    return metrics, findings


def layer_kind(layer: dict[str, Any] | None) -> str:
    """Coarse bucket for attributing time: why INT8 isn't 2x FP16 (C5 task 5)."""
    if layer is None:
        return "unknown"
    if is_reformat(layer):
        return "reformat"
    if uses_int8(layer):
        return "int8"
    out = _dtypes(layer, "Outputs")
    return {"Half": "fp16", "Float": "fp32"}.get(out[0] if out else "", "other")


def profile_summary(profile: list[dict[str, Any]], doc: Any) -> dict[str, Any]:
    """trtexec --exportProfile per-layer times, summed by layer kind (int8 / fp16 / fp32 /
    reformat) and by layer type. Profile entries without a matching engine layer count as
    'unknown' rather than being dropped, so the parts always add up to the total."""
    layers = {str(x.get("Name")): x for x in _layers(doc) if isinstance(x, dict)}
    by_kind: Counter[str] = Counter()
    by_type: Counter[str] = Counter()
    total = 0.0
    for entry in profile:
        if "name" not in entry:
            continue  # the leading {"count": N}
        ms = float(entry.get("medianMs", entry.get("averageMs", 0.0)))
        layer = layers.get(str(entry["name"]))
        by_kind[layer_kind(layer)] += ms
        by_type[str(layer.get("LayerType", "?")) if layer else "unknown"] += ms
        total += ms

    def share(c: Counter[str]) -> dict[str, dict[str, float]]:
        return {k: {"ms": round(v, 4), "share": round(v / total, 4) if total else 0.0}
                for k, v in c.most_common()}  # fmt: skip

    return {"total_median_ms": round(total, 4), "by_kind": share(by_kind),
            "by_type": share(by_type)}  # fmt: skip
