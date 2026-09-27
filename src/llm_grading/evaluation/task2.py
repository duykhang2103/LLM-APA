"""Task 2 multi-label metrics: macro-F1, micro-F1, and per-label scores.

Always report support and the per-label table. Macro-F1 can hide a broken
label if the team does not inspect the individual rows, especially when a
label is rare or nearly absent.
"""

from typing import Any, Iterable, Mapping


def evaluate_task2(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Return macro/micro and per-label metrics from aligned records."""
    # TODO: Report rare-label behavior and threshold-sensitive metrics.
    raise NotImplementedError("Task 2 evaluation is not implemented in the scaffold.")
