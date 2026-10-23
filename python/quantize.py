"""Symmetric per-group int8 weight quantization along the input dimension.

Each group of GROUP consecutive input-dimension values shares one float32 scale,
scale = max(abs(w)) / 127, and q = round(w / scale) clipped to [-127, 127]. An
all-zero group gets scale 0 and q = 0. Dequantization is q * scale, so every
element's error is at most scale / 2.
"""

import numpy as np

GROUP = 32
QMAX = 127


def quantize(weights: np.ndarray, axis: int, group: int = GROUP):
    """Quantize a 2-D float32 matrix in groups along `axis` (the input dimension).

    Returns (q, scales): q has the shape of `weights`; scales has the size of
    `weights` with `axis` divided by `group`.
    """
    if weights.ndim != 2 or axis not in (0, 1):
        raise ValueError("quantization expects a matrix and axis 0 or 1")
    if weights.shape[axis] % group:
        raise ValueError("the input dimension must be a multiple of the group size")
    w = np.asarray(weights, dtype=np.float32)
    moved = np.moveaxis(w, axis, 0)  # [input, other]
    blocks = moved.reshape(moved.shape[0] // group, group, moved.shape[1])
    scales = np.abs(blocks).max(axis=1) / QMAX  # [groups, other]
    safe = np.where(scales > 0, scales, 1.0)
    q = np.clip(np.rint(blocks / safe[:, None, :]), -QMAX, QMAX).astype(np.int8)
    q = np.where(scales[:, None, :] > 0, q, 0).astype(np.int8)
    q = np.moveaxis(q.reshape(moved.shape), 0, axis)
    scales = np.moveaxis(scales.astype(np.float32), 0, axis)
    return np.ascontiguousarray(q), np.ascontiguousarray(scales)


def dequantize(q: np.ndarray, scales: np.ndarray, axis: int, group: int = GROUP):
    """Inverse of quantize: q * scale, broadcasting each scale over its group."""
    expanded = np.repeat(scales.astype(np.float32), group, axis=axis)
    return q.astype(np.float32) * expanded
