"""Task 2 multi-label metrics: macro-F1, micro-F1, and per-label scores."""

from typing import Any, Iterable, Mapping


def evaluate_task2(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Describe the future Task 2 evaluator."""
    # TODO: Report rare-label behavior and threshold-sensitive metrics.
    raise NotImplementedError("Task 2 evaluation is not implemented in the scaffold.")
