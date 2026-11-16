import importlib.util
from pathlib import Path

import pytest


path = Path(__file__).parent / "evaluator.py"
spec = importlib.util.spec_from_file_location("evaluator", path)
evaluator = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evaluator)


def test_exact_match_ignores_case_and_spacing():
    assert evaluator.exact_match("Hello World", " hello   world ")


def test_token_f1_rewards_overlap():
    score = evaluator.token_f1("machine learning model", "learning model")
    assert score > 0.7


def test_keyword_coverage():
    score = evaluator.keyword_coverage(
        "Built with PyTorch and Python",
        ["Python", "PyTorch", "Docker"],
    )
    assert score == pytest.approx(2 / 3)


def test_unsupported_numbers():
    assert evaluator.unsupported_number_count(
        "Accuracy was 91%.",
        "Accuracy was 91%, across 500 samples.",
    ) == 1
