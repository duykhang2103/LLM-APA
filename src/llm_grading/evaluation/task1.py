"""Task 1 rubric metrics: QWK, MAE, and component exact-match.

Evaluation order should be: validate components, derive predicted/reference
totals, compute QWK and MAE on totals, then compute exact match by component.
Report slices for single/multi-problem, score range, and problem type.
"""

from typing import Any, Iterable, Mapping


def evaluate_task1(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Return Task 1 metrics from aligned prediction/reference records."""
    # TODO: Compute deterministic totals from six validated components.
    raise NotImplementedError("Task 1 evaluation is not implemented in the scaffold.")
