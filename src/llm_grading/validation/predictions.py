"""Validate task-specific predictions before leaderboard submission."""

from pathlib import Path
from typing import Any


def validate_predictions(path: str | Path, task: str) -> list[str]:
    """Describe the future prediction validator; return validation messages."""
    # TODO: Enforce sample IDs, task output schema, ranges, labels, and JSON shape.
    raise NotImplementedError("Prediction validation is not implemented in the scaffold.")
