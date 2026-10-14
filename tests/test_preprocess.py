from dataclasses import replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from edgeinfer.config import load_model_config
from edgeinfer.postprocess import postprocess
from edgeinfer.preprocess import preprocess
from edgeinfer.resize import resize_bilinear


@pytest.mark.parametrize(
    ("in_hw", "out_hw"),
    [((8, 16), (4, 8)), ((7, 13), (5, 3)), ((4, 6), (16, 24)), ((3, 5), (11, 7))],
)
def test_resize_matches_torch_half_pixel_bilinear(in_hw, out_hw):
    x = np.random.default_rng(0).standard_normal((2, *in_hw)).astype(np.float32)
    expected = F.interpolate(
        torch.from_numpy(x)[None], size=out_hw, mode="bilinear", align_corners=False
    )[0].numpy()
    np.testing.assert_allclose(resize_bilinear(x, *out_hw), expected, rtol=0, atol=1e-5)


def test_exact_2x_downscale_is_2x2_block_mean():
    x = np.arange(16, dtype=np.float32).reshape(1, 4, 4)
    expected = x.reshape(1, 2, 2, 2, 2).mean(axis=(2, 4))
    np.testing.assert_allclose(resize_bilinear(x, 2, 2), expected, atol=1e-6)


@pytest.fixture
def cfg():
    base = load_model_config("segformer_b0")
    return replace(base, input=replace(base.input, height=1, width=2))


def test_preprocess_known_input(cfg):
    # 2x4 image, each 2x2 block constant -> 2x downscale keeps the block values exactly.
    img = np.zeros((2, 4, 3), dtype=np.uint8)
    img[:, :2] = (255, 0, 0)
    img[:, 2:] = (0, 255, 0)
    x = preprocess(img, cfg)
    assert x.shape == (1, 3, 1, 2)
    assert x.dtype == np.float32
    mean, std = np.array(cfg.preprocess.mean), np.array(cfg.preprocess.std)
    np.testing.assert_allclose(x[0, :, 0, 0], (np.array([1, 0, 0]) - mean) / std, atol=1e-6)
    np.testing.assert_allclose(x[0, :, 0, 1], (np.array([0, 1, 0]) - mean) / std, atol=1e-6)


def test_bgr_config_swaps_channels(cfg):
    img = np.random.default_rng(0).integers(0, 256, (2, 4, 3), dtype=np.uint8)
    cfg = replace(cfg, preprocess=replace(cfg.preprocess, mean=(0.0,) * 3, std=(1.0,) * 3))
    bgr = replace(cfg, input=replace(cfg.input, channel_order="BGR"))
    np.testing.assert_array_equal(preprocess(img, cfg)[0, ::-1], preprocess(img, bgr)[0])


def test_preprocess_rejects_wrong_input(cfg):
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 4, 3), dtype=np.float32), cfg)
    with pytest.raises(ValueError):
        preprocess(np.zeros((2, 4), dtype=np.uint8), cfg)


def test_postprocess_upsamples_logits_then_argmax():
    logits = np.zeros((1, 3, 2, 2), dtype=np.float32)
    logits[0, 2, 0, 0] = 5.0  # class 2 wins top-left
    logits[0, 1, 1, 1] = 5.0  # class 1 wins bottom-right
    pred = postprocess(logits, 8, 8)
    assert pred.shape == (8, 8)
    assert pred.dtype == np.uint8
    assert pred[0, 0] == 2
    assert pred[7, 7] == 1
