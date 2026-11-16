# LLM Evaluation Toolkit

A small evaluation toolkit for comparing generated answers against simple quality checks.

The project focuses on **evaluation infrastructure**, not on claiming that one model is universally better than another.

## Checks included

- exact-match rate
- token-overlap score
- response-length statistics
- required-keyword coverage
- simple hallucination-risk flags based on unsupported numeric claims

## Example

```python
from evaluator import evaluate_batch

references = ["Paris is the capital of France."]
predictions = ["The capital of France is Paris."]

print(evaluate_batch(references, predictions))
```

## Why this project exists

LLM applications need repeatable evaluation. Even simple deterministic checks can catch regressions before adding model-based judges or human review.

## Next directions

- semantic similarity using embeddings
- rubric-based model judging
- retrieval-grounded faithfulness checks
- experiment tracking across prompts and models
- human review exports
