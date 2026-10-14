"""Bilinear resize with an exact, documented definition.

Half-pixel sampling, identical to PyTorch `F.interpolate(mode="bilinear",
align_corners=False, antialias=False)` and to OpenCV `INTER_LINEAR` on float input:

    src = (dst + 0.5) * in_size / out_size - 0.5,  clamped below at 0
    i0 = floor(src), i1 = min(i0 + 1, in_size - 1), w1 = src - i0, w0 = 1 - w1

No antialiasing, so a C++/CUDA runtime can reproduce it exactly. For an exact 2x
downscale this reduces to averaging each 2x2 block.
"""

from __future__ import annotations

import numpy as np


def _axis_weights(in_size: int, out_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dst = np.arange(out_size, dtype=np.float64)
    src = np.maximum((dst + 0.5) * (in_size / out_size) - 0.5, 0.0)
    i0 = np.minimum(np.floor(src).astype(np.int64), in_size - 1)
    i1 = np.minimum(i0 + 1, in_size - 1)
    w1 = (src - i0).astype(np.float32)
    return i0, i1, w1


def resize_bilinear(x: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize the last two axes of a float32 array (..., H, W) to (..., height, width)."""
    if x.dtype != np.float32:
        raise TypeError(f"expected float32, got {x.dtype}")
    in_h, in_w = x.shape[-2:]
    if (in_h, in_w) == (height, width):
        return x.copy()
    # Separable: interpolate rows, then columns.
    i0, i1, w1 = _axis_weights(in_h, height)
    w1 = w1[:, None]
    x = x[..., i0, :] * (1.0 - w1) + x[..., i1, :] * w1
    j0, j1, v1 = _axis_weights(in_w, width)
    x = x[..., j0] * (1.0 - v1) + x[..., j1] * v1
    return x.astype(np.float32, copy=False)
