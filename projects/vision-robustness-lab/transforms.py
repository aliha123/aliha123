from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def gaussian_noise(image: Image.Image, sigma: float, seed: int = 42) -> Image.Image:
    if sigma < 0:
        raise ValueError("sigma must be non-negative")

    rng = np.random.default_rng(seed)
    array = np.asarray(image.convert("RGB"), dtype=np.float32)
    noise = rng.normal(0.0, sigma, array.shape)
    perturbed = np.clip(array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(perturbed)


def gaussian_blur(image: Image.Image, radius: float) -> Image.Image:
    if radius < 0:
        raise ValueError("radius must be non-negative")
    return image.convert("RGB").filter(ImageFilter.GaussianBlur(radius=radius))


def adjust_brightness(image: Image.Image, factor: float) -> Image.Image:
    if factor < 0:
        raise ValueError("factor must be non-negative")
    return ImageEnhance.Brightness(image.convert("RGB")).enhance(factor)


def center_crop_square(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    width, height = image.size
    side = min(width, height)
    left = (width - side) // 2
    top = (height - side) // 2
    return image.crop((left, top, left + side, top + side))
