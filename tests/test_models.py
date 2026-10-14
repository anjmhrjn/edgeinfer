import numpy as np
import pytest
import torch

from edgeinfer.backends import get_backend
from edgeinfer.config import load_model_config
from edgeinfer.models.deeplabv3plus import DeepLabV3PlusLogits
from edgeinfer.models.weights import fetch_url_weights, sha256_file


def test_deeplab_architecture_outputs_stride4_logits():
    # Random init, no checkpoint: checks the assembled graph without downloads.
    model = DeepLabV3PlusLogits(num_classes=19, backbone_output_stride=16).eval()
    with torch.inference_mode():
        out = model(torch.zeros(1, 3, 64, 128))
    assert out.shape == (1, 19, 16, 32)


def test_weights_hash_mismatch_is_an_error(tmp_path):
    (tmp_path / "w.pth").write_bytes(b"not the right weights")
    with pytest.raises(ValueError, match="does not match"):
        fetch_url_weights("https://example.invalid/w.pth", "w.pth", "0" * 64, cache_dir=tmp_path)
    assert sha256_file(tmp_path / "w.pth") != "0" * 64


def test_unknown_backend():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("tensorflow")


@pytest.mark.weights
@pytest.mark.parametrize("name", ["segformer_b0", "deeplabv3plus_mbv2"])
def test_pretrained_models_load_and_run(name):
    cfg = load_model_config(name)
    backend = get_backend("torch_ref")
    backend.load(None, cfg)
    x = np.zeros((1, 3, cfg.input.height, cfg.input.width), dtype=np.float32)
    out = backend.infer(x)
    stride = cfg.output_stride
    assert out.shape == (1, 19, cfg.input.height // stride, cfg.input.width // stride)
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    assert backend.metadata()["precision"] == "fp32"
