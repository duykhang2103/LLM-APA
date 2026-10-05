"""Common evaluation helpers and result containers.

Every evaluator should return machine-readable metrics plus enough counts to
interpret them. A single score without support, split, and experiment ID is
not sufficient evidence for a report.
"""

from typing import Any, Iterable, Mapping


def summarize_results(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Combine task metrics with experiment and split metadata."""
    combined: dict[str, Any] = {}
    for item in metrics:
        combined.update(dict(item))
    return combined
