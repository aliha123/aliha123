import importlib.util
from pathlib import Path

from PIL import Image
import pytest


def load_module(name: str):
    path = Path(__file__).parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


transforms = load_module("transforms")
metrics = load_module("metrics")


def test_noise_is_reproducible():
    image = Image.new("RGB", (16, 16), color=(120, 120, 120))
    first = transforms.gaussian_noise(image, sigma=10, seed=3)
    second = transforms.gaussian_noise(image, sigma=10, seed=3)
    assert list(first.getdata()) == list(second.getdata())


def test_center_crop_is_square():
    image = Image.new("RGB", (30, 20), color="white")
    cropped = transforms.center_crop_square(image)
    assert cropped.size == (20, 20)


def test_metric_detects_change():
    a = Image.new("RGB", (8, 8), color=(0, 0, 0))
    b = Image.new("RGB", (8, 8), color=(10, 10, 10))
    assert metrics.mean_absolute_pixel_error(a, b) == pytest.approx(10.0)
