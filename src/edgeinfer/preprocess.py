"""Config-driven preprocessing: the single source of truth for model input.

uint8 HxWx3 RGB image -> float32 1x3xHxW tensor, in this order:
    1. cast to float32 (resizing in float avoids uint8 rounding)
    2. channel order (RGB as loaded; BGR swaps channels)
    3. resize to config input size (see edgeinfer.resize for the exact definition)
    4. divide by `scale`
    5. subtract mean, divide by std (per channel, in the config's channel order)
    6. HWC -> NCHW
"""

from __future__ import annotations

import numpy as np

from edgeinfer.config import ModelConfig
from edgeinfer.resize import resize_bilinear


def preprocess(image: np.ndarray, cfg: ModelConfig) -> np.ndarray:
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 uint8 RGB, got {image.shape} {image.dtype}")
    pp, spec = cfg.preprocess, cfg.input
    x = image.astype(np.float32).transpose(2, 0, 1)  # CHW
    if spec.channel_order == "BGR":
        x = x[::-1]
    x = resize_bilinear(np.ascontiguousarray(x), spec.height, spec.width)
    x /= np.float32(pp.scale)
    x -= np.asarray(pp.mean, dtype=np.float32)[:, None, None]
    x /= np.asarray(pp.std, dtype=np.float32)[:, None, None]
    return np.ascontiguousarray(x[None], dtype=np.float32)
