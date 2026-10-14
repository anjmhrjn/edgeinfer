"""Logits -> class map. Upsampling lives here, outside the model graph (D-003)."""

from __future__ import annotations

import numpy as np

from edgeinfer.resize import resize_bilinear


def postprocess(logits: np.ndarray, height: int, width: int) -> np.ndarray:
    """1xCxhxw float32 logits -> HxW uint8 trainIds at (height, width).

    Bilinear-upsample the logits (not the argmax) to the target size, then argmax.
    """
    if logits.ndim != 4 or logits.shape[0] != 1:
        raise ValueError(f"expected 1xCxhxw logits, got {logits.shape}")
    up = resize_bilinear(logits[0].astype(np.float32, copy=False), height, width)
    return up.argmax(axis=0).astype(np.uint8)
