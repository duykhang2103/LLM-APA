"""Task 3 diagnosis and feedback-level compliance diagnostics.

Keep two axes visible in reports: whether the feedback identifies the right
problem, and whether it obeys the requested level. A fluent answer that leaks
a Level 1 solution should fail compliance even when its diagnosis is correct.
"""

from typing import Any, Iterable, Mapping


def evaluate_task3(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Return diagnosis and level-compliance diagnostics."""
    # TODO: Combine diagnosis correctness with Level 1/2 restriction checks.
    raise NotImplementedError("Task 3 evaluation is not implemented in the scaffold.")
