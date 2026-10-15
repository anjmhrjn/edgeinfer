import onnx
import pytest
import yaml

from edgeinfer.config import ConfigError, load_model_config
from edgeinfer.export.onnx_export import artifact_path
from edgeinfer.quant import QuantError
from edgeinfer.quant.config import (
    QUANT_DIR,
    Exclusions,
    QuantConfig,
    load_group_rules,
    load_quant_config,
)
from edgeinfer.quant.groups import assign_groups, resolve_exclusions
from tests.conftest import FIXTURES

TINY_GROUPS = FIXTURES / "tiny_fixture_groups.yaml"
OPS = ("Conv", "MatMul", "Gemm")


@pytest.mark.parametrize("method", ["minmax", "entropy", "percentile"])
def test_shipped_quant_configs_are_tensorrt_format(method):
    q = load_quant_config(f"int8_{method}")
    assert q.calibration.method == method
    assert (q.activation_type, q.weight_type) == ("int8", "int8")
    assert q.activation_symmetric and q.weight_symmetric and q.per_channel
    assert not q.quantize_outputs and not q.quantize_bias
    assert q.calibration.manifest == "calib_train200"


def _raw():
    return yaml.safe_load((QUANT_DIR / "int8_entropy.yaml").read_text())


def test_entropy_defaults_are_not_left_to_ort():
    q = load_quant_config("int8_entropy")
    assert (q.calibration.num_bins, q.calibration.num_quantized_bins) == (2048, 128)


def test_method_parameters_must_match_the_method():
    raw = _raw()
    raw["calibration"]["percentile"] = 99.9
    with pytest.raises(ConfigError, match="takes"):
        QuantConfig.from_dict(raw)
    raw = _raw()
    raw["calibration"]["method"] = "minmax"
    with pytest.raises(ConfigError, match="minmax"):
        QuantConfig.from_dict(raw)


def test_unknown_keys_and_bad_values_are_errors():
    raw = _raw()
    raw["per_chanel"] = True
    with pytest.raises(ConfigError, match="unknown keys"):
        QuantConfig.from_dict(raw)
    raw = _raw()
    raw["activation_type"] = "int4"
    with pytest.raises(ConfigError, match="activation_type"):
        QuantConfig.from_dict(raw)
    raw = _raw()
    raw["op_types"] = ["Conv", "LayerNormalization"]
    with pytest.raises(ConfigError, match="op_types"):
        QuantConfig.from_dict(raw)


def test_tiny_groups_cover_every_quantizable_node(tiny_onnx):
    model = onnx.load(str(tiny_onnx))
    groups = assign_groups(model, load_group_rules("tiny_fixture", TINY_GROUPS), OPS)
    assert {g: len(n) for g, n in groups.items()} == {"stem": 2, "attn": 4, "head": 2}


def test_unmatched_node_is_an_error(tiny_onnx, tmp_path):
    rules = tmp_path / "g.yaml"
    rules.write_text("model: tiny_fixture\nrules:\n  - {name: stem, match: 'stem\\..*'}\n")
    with pytest.raises(QuantError, match="match no group rule") as e:
        assign_groups(onnx.load(str(tiny_onnx)), load_group_rules("tiny_fixture", rules), OPS)
    assert e.value.code == "QUANT_GROUP_UNMATCHED"


def test_rules_for_another_model_are_rejected():
    with pytest.raises(ConfigError, match="not 'segformer_b0'"):
        load_group_rules("segformer_b0", TINY_GROUPS)


def test_exclusions_resolve_to_nodes_and_reject_typos(tiny_onnx):
    model = onnx.load(str(tiny_onnx))
    groups = assign_groups(model, load_group_rules("tiny_fixture", TINY_GROUPS), OPS)
    exc = Exclusions("x", "tiny_fixture", ("attn",), ())
    assert resolve_exclusions(exc, groups, model) == groups["attn"]
    typo = Exclusions("x", "tiny_fixture", ("atn",), ())
    with pytest.raises(QuantError, match="unknown groups") as e:
        resolve_exclusions(typo, groups, model)
    assert e.value.code == "QUANT_UNKNOWN_EXCLUSION"


@pytest.mark.weights
@pytest.mark.parametrize(
    ("name", "n_groups", "n_nodes"), [("segformer_b0", 18, 88), ("deeplabv3plus_mbv2", 10, 60)]
)
def test_real_model_groups_cover_every_quantizable_node(name, n_groups, n_nodes):
    path = artifact_path(load_model_config(name))
    assert path.is_file(), f"{path} missing; run `edgeinfer export --model {name}`"
    groups = assign_groups(onnx.load(str(path)), load_group_rules(name), OPS)
    assert len(groups) == n_groups
    assert sum(len(v) for v in groups.values()) == n_nodes
