"""Task 2 threshold configuration boundary.

Example scores and thresholds::

    scores = {"LABEL_01": 0.80, "LABEL_02": 0.22}
    thresholds = {"LABEL_01": 0.50, "LABEL_02": 0.30}
    output = ["LABEL_01"]

Thresholds are tuned on validation data and must be stored in config or run
metadata. Never tune them on the private test set.
"""

from typing import Mapping

from llm_grading.data.taxonomy import ERROR_LABELS


def apply_thresholds(
    scores: Mapping[str, float],
    thresholds: Mapping[str, float],
) -> list[str]:
    """Convert per-label scores into validated labels."""
    unknown = sorted(set(scores).difference(ERROR_LABELS))
    if unknown:
        raise ValueError(f"Unknown Task 2 labels: {unknown}")
    result = []
    for label in ERROR_LABELS:
        if label not in scores:
            continue
        score = scores[label]
        if not isinstance(score, (int, float)) or not 0 <= score <= 1:
            raise ValueError(f"Score for {label} must be between 0 and 1")
        threshold = thresholds.get(label, 0.5)
        if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 1:
            raise ValueError(f"Threshold for {label} must be between 0 and 1")
        if score >= threshold:
            result.append(label)
    return result
