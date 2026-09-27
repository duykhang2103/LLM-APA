"""Task 3 diagnosis and feedback-level compliance diagnostics.

Keep two axes visible in reports: whether the feedback identifies the right
problem, and whether it obeys the requested level. A fluent answer that leaks
a Level 1 solution should fail compliance even when its diagnosis is correct.
"""

from typing import Any, Iterable, Mapping


def _output(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("output", record)
    if not isinstance(value, Mapping):
        raise ValueError("Task 3 output must be an object")
    return value


def evaluate_task3(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return diagnosis and level-compliance diagnostics."""
    prediction_list = list(predictions)
    reference_list = list(references)
    if len(prediction_list) != len(reference_list):
        raise ValueError("Task 3 predictions and references must have the same length")
    compliance_passes = []
    overlaps = []
    for prediction, reference in zip(prediction_list, reference_list):
        output = _output(prediction)
        compliance_passes.append(bool(output.get("compliance", {}).get("pass", False)))
        predicted_labels = set(output.get("diagnosed_labels", output.get("error_labels", [])))
        reference_labels = set(reference.get("error_labels", []))
        overlaps.append(bool(predicted_labels.intersection(reference_labels)) if reference_labels else not predicted_labels)
    return {
        "count": len(prediction_list),
        "diagnosis_overlap_rate": sum(overlaps) / len(overlaps) if overlaps else 0.0,
        "compliance_pass_rate": sum(compliance_passes) / len(compliance_passes) if compliance_passes else 0.0,
    }
