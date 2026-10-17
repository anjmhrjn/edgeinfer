import pytest
import yaml

from edgeinfer.config import ConfigError, load_model_config
from edgeinfer.pipeline.config import (
    DEPLOY_DIR,
    GatesConfig,
    Recipe,
    load_gates,
    load_recipe,
    recipe_hash,
)
from edgeinfer.quant.config import load_exclusions, load_quant_config
from edgeinfer.quant.fp16 import load_fp16_config

RECIPES = sorted(DEPLOY_DIR.rglob("*.yaml"))


@pytest.mark.parametrize("path", RECIPES, ids=lambda p: p.stem)
def test_shipped_recipes_load_and_name_existing_configs(path):
    r = load_recipe(path)
    assert r.name == path.stem
    cfg = load_model_config(r.model)
    if r.quant:
        load_quant_config(r.quant.config)
        if r.quant.exclusions:
            assert load_exclusions(r.quant.exclusions, cfg.name).model == cfg.name
    if r.fp16:
        load_fp16_config(r.fp16)
    load_gates(r.gates)


def test_shipped_gates_load():
    g = load_gates()
    assert g.parity.inputs == "images"  # preprocessing faults must reach the parity gate
    assert set(g.accuracy.precisions) == {"int8", "int8_fp16"}


def _raw(name="deeplabv3plus_mbv2"):
    return yaml.safe_load((DEPLOY_DIR / f"{name}.yaml").read_text())


def test_unknown_key_is_an_error():
    raw = _raw() | {"targte": "trt"}
    with pytest.raises(ConfigError, match="unknown keys"):
        Recipe.from_dict(raw, "r", "r.yaml")


def test_name_key_is_an_error():
    with pytest.raises(ConfigError, match="file name"):
        Recipe.from_dict(_raw() | {"name": "other"}, "r", "r.yaml")


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"quant": None}, "quant: required"),  # an INT8 recipe without quant settings
        ({"fp16": "fp16_autocast"}, "fp16: not used"),  # a setting that would do nothing
        ({"build": None}, "build: required"),  # trt needs builder settings
        ({"target": "ort_cpu"}, "build: not used"),
        ({"precision": "int4"}, "precision"),
    ],
)
def test_fields_must_match_precision_and_target(change, match):
    with pytest.raises(ConfigError, match=match):
        Recipe.from_dict(_raw() | change, "r", "r.yaml")


def test_recipe_hash_follows_settings_not_name():
    a = Recipe.from_dict(_raw(), "a", "a.yaml")
    b = Recipe.from_dict(_raw(), "b", "b.yaml")
    assert recipe_hash(a) == recipe_hash(b)
    raw = _raw()
    raw["quant"]["exclusions"] = None
    assert recipe_hash(Recipe.from_dict(raw, "a", "a.yaml")) != recipe_hash(a)


def test_fp32_is_not_an_accuracy_candidate():
    raw = load_gates().to_dict()
    raw["accuracy"]["precisions"] = ["fp32", "int8"]
    with pytest.raises(ConfigError, match="reference"):
        GatesConfig.from_dict(raw)
