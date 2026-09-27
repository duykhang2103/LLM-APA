"""Validate task-specific predictions before leaderboard submission.

Use the synthetic files under ``examples/predictions/`` as the first schema
fixtures. Validation should check JSON shape, sample IDs, task output fields,
score ranges, taxonomy membership, and required feedback-level diagnostics.
"""

from pathlib import Path
from typing import Any


def validate_predictions(path: str | Path, task: str) -> list[str]:
    """Return validation messages; an empty list means the file is valid."""
    # TODO: Enforce sample IDs, task output schema, ranges, labels, and JSON shape.
    raise NotImplementedError("Prediction validation is not implemented in the scaffold.")
