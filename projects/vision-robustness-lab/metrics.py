from __future__ import annotations

import numpy as np
from PIL import Image


def mean_absolute_pixel_error(reference: Image.Image, candidate: Image.Image) -> float:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32)
    cand = np.asarray(candidate.convert("RGB"), dtype=np.float32)

    if ref.shape != cand.shape:
        raise ValueError("images must have the same dimensions")

    return float(np.abs(ref - cand).mean())


def normalized_l2_distance(reference: Image.Image, candidate: Image.Image) -> float:
    ref = np.asarray(reference.convert("RGB"), dtype=np.float32) / 255.0
    cand = np.asarray(candidate.convert("RGB"), dtype=np.float32) / 255.0

    if ref.shape != cand.shape:
        raise ValueError("images must have the same dimensions")

    return float(np.linalg.norm(ref - cand) / np.sqrt(ref.size))
