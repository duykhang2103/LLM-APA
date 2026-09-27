"""Task 2 threshold configuration boundary.

Example scores and thresholds::

    scores = {"LABEL_01": 0.80, "LABEL_02": 0.22}
    thresholds = {"LABEL_01": 0.50, "LABEL_02": 0.30}
    output = ["LABEL_01"]

Thresholds are tuned on validation data and must be stored in config or run
metadata. Never tune them on the private test set.
"""

from typing import Mapping


def apply_thresholds(
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[str]:
    """Convert per-label scores into validated labels."""
    # TODO: Validate labels and compare global versus tuned threshold policies.
    raise NotImplementedError("Task 2 thresholding is not implemented.")
