import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).parent / "features.py"
spec = importlib.util.spec_from_file_location("retention_features", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)


def test_dataset_is_reproducible():
    first = module.make_retention_dataset(25, seed=7)
    second = module.make_retention_dataset(25, seed=7)
    assert first.equals(second)


def test_dataset_has_binary_target():
    frame = module.make_retention_dataset(200)
    assert set(frame["dropped_out"].unique()).issubset({0, 1})
