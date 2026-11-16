from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import re
from statistics import mean


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9']+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def exact_match(reference: str, prediction: str) -> bool:
    normalize = lambda value: " ".join(tokenize(value))
    return normalize(reference) == normalize(prediction)


def token_f1(reference: str, prediction: str) -> float:
    reference_tokens = Counter(tokenize(reference))
    prediction_tokens = Counter(tokenize(prediction))

    if not reference_tokens or not prediction_tokens:
        return 0.0

    overlap = sum((reference_tokens & prediction_tokens).values())
    precision = overlap / sum(prediction_tokens.values())
    recall = overlap / sum(reference_tokens.values())

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def keyword_coverage(prediction: str, required_keywords: list[str]) -> float:
    if not required_keywords:
        return 1.0

    normalized = prediction.lower()
    matched = sum(keyword.lower() in normalized for keyword in required_keywords)
    return matched / len(required_keywords)


def unsupported_number_count(reference: str, prediction: str) -> int:
    number_pattern = re.compile(r"\b\d+(?:\.\d+)?%?\b")
    reference_numbers = set(number_pattern.findall(reference))
    prediction_numbers = number_pattern.findall(prediction)
    return sum(number not in reference_numbers for number in prediction_numbers)


@dataclass(frozen=True)
class BatchMetrics:
    exact_match_rate: float
    mean_token_f1: float
    mean_response_tokens: float
    unsupported_numbers: int


def evaluate_batch(references: list[str], predictions: list[str]) -> BatchMetrics:
    if len(references) != len(predictions):
        raise ValueError("references and predictions must have the same length")
    if not references:
        raise ValueError("at least one example is required")

    return BatchMetrics(
        exact_match_rate=mean(exact_match(r, p) for r, p in zip(references, predictions)),
        mean_token_f1=mean(token_f1(r, p) for r, p in zip(references, predictions)),
        mean_response_tokens=mean(len(tokenize(p)) for p in predictions),
        unsupported_numbers=sum(
            unsupported_number_count(r, p) for r, p in zip(references, predictions)
        ),
    )
