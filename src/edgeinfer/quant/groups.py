"""Map the quantizable nodes of an FP32 ONNX model to layer groups.

A node's group comes from its innermost module scope, which the dynamo exporter records
in the `pkg.torch.onnx.name_scopes` node metadata (e.g.
`model.segformer.stages.2.blocks.1.mlp.fc1`), matched against configs/quant/groups/.
Groups are computed on the FP32 model; the quantizer keeps node names, so the same
names select nodes in every QDQ model derived from it.
"""

from __future__ import annotations

import ast

import onnx

from edgeinfer.quant import QuantError
from edgeinfer.quant.config import Exclusions, GroupRules

SCOPES_KEY = "pkg.torch.onnx.name_scopes"


def node_scope(node: onnx.NodeProto) -> str:
    """Innermost module path of a node ('' for nodes at the top level)."""
    for prop in node.metadata_props:
        if prop.key == SCOPES_KEY:
            scopes = ast.literal_eval(prop.value)  # e.g. ['', 'model', ..., 'view_2']
            return scopes[-2] if len(scopes) >= 2 else ""
    raise QuantError(
        "QUANT_GROUP_UNMATCHED",
        f"node {node.name!r} has no {SCOPES_KEY} metadata; groups need a dynamo-exported model",
    )


def assign_groups(
    model: onnx.ModelProto, rules: GroupRules, op_types: tuple[str, ...]
) -> dict[str, list[str]]:
    """{group: [node names in graph order]} over the nodes of `op_types`. Every such node
    must match a rule, and every rule must match a node."""
    groups: dict[str, list[str]] = {name: [] for name in rules.names}
    unmatched = []
    for node in model.graph.node:
        if node.op_type not in op_types:
            continue
        scope = node_scope(node)
        group = rules.group_of(scope)
        if group is None:
            unmatched.append(f"{node.name} ({scope or '<top level>'})")
        else:
            groups[group].append(node.name)
    if unmatched:
        raise QuantError(
            "QUANT_GROUP_UNMATCHED",
            f"{len(unmatched)} quantizable nodes match no group rule: {unmatched[:5]}",
        )
    empty = [g for g, nodes in groups.items() if not nodes]
    if empty:
        raise QuantError("QUANT_GROUP_UNMATCHED", f"group rules match no node: {empty}")
    return groups


def resolve_exclusions(
    exclusions: Exclusions, groups: dict[str, list[str]], model: onnx.ModelProto
) -> list[str]:
    """Node names to keep in float, in graph order. Unknown names are errors: a typo must
    never silently quantize a layer that was meant to stay in float."""
    unknown_groups = [g for g in exclusions.groups if g not in groups]
    names = {n.name for n in model.graph.node}
    unknown_nodes = [n for n in exclusions.nodes if n not in names]
    if unknown_groups or unknown_nodes:
        raise QuantError(
            "QUANT_UNKNOWN_EXCLUSION",
            f"exclusions {exclusions.name!r}: unknown groups {unknown_groups} "
            f"(known: {list(groups)}), unknown nodes {unknown_nodes}",
        )
    wanted = {n for g in exclusions.groups for n in groups[g]} | set(exclusions.nodes)
    return [n.name for n in model.graph.node if n.name in wanted]
