import pytest
import yaml

from edgeinfer.config import CONFIG_DIR, ConfigError, ModelConfig, load_model_config

MODELS = ("segformer_b0", "deeplabv3plus_mbv2")


@pytest.mark.parametrize("name", MODELS)
def test_shipped_configs_load(name):
    cfg = load_model_config(name)
    assert cfg.name == name
    assert cfg.num_classes == 19
    assert (cfg.input.height, cfg.input.width) == (512, 1024)
    assert cfg.weights["license"]


def _raw(name="segformer_b0"):
    return yaml.safe_load((CONFIG_DIR / "models" / f"{name}.yaml").read_text())


def test_unknown_key_is_an_error():
    raw = _raw()
    raw["preprocess"]["meen"] = [0, 0, 0]
    with pytest.raises(ConfigError, match="unknown keys"):
        ModelConfig.from_dict(raw)


def test_missing_key_is_an_error():
    raw = _raw()
    del raw["preprocess"]["std"]
    with pytest.raises(ConfigError, match="missing keys"):
        ModelConfig.from_dict(raw)


def test_invalid_choice_is_an_error():
    raw = _raw()
    raw["input"]["channel_order"] = "rgb"
    with pytest.raises(ConfigError, match="channel_order"):
        ModelConfig.from_dict(raw)
