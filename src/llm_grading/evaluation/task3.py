"""Task 3 diagnosis and feedback-level compliance diagnostics.

Keep two axes visible in reports: whether the feedback identifies the right
problem, and whether it obeys the requested level. A fluent answer that leaks
a Level 1 solution should fail compliance even when its diagnosis is correct.
"""

import hashlib
from typing import Any, Iterable, Mapping

from tasks.task3_feedback.compliance import check_compliance


def _output(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("output", record)
    if not isinstance(value, Mapping):
        raise ValueError("Task 3 output must be an object")
    return value


def evaluate_task3(
    predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Return diagnosis and level-compliance diagnostics."""
    prediction_list = list(predictions)
    reference_list = list(references)
    if len(prediction_list) != len(reference_list):
        raise ValueError("Task 3 predictions and references must have the same length")
    compliance_passes = []
    judgments = []
    for prediction, reference in zip(prediction_list, reference_list):
        output = _output(prediction)
        level = reference.get("feedback_level", output.get("feedback_level"))
        compliance_passes.append(
            check_compliance(output.get("feedback", ""), level)["pass"]
        )
        judgment = output.get("judgment")
        if judgment is not None:
            if (
                not isinstance(judgment, Mapping)
                or not isinstance(judgment.get("judge"), str)
                or not judgment["judge"].strip()
                or any(
                    not isinstance(judgment.get(k), bool)
                    for k in ("diagnosis_correct", "level_compliant")
                )
            ):
                raise ValueError(
                    "Task 3 judgment needs judge provenance and boolean diagnosis_correct/level_compliant"
                )
            if (
                judgment.get("feedback_sha256")
                != hashlib.sha256(
                    output.get("feedback", "").encode("utf-8")
                ).hexdigest()
                or type(judgment.get("feedback_level")) is not int
                or judgment["feedback_level"] != level
            ):
                raise ValueError(
                    "Task 3 judgment does not match feedback text and requested level"
                )
            judgments.append(judgment)
    return {
        "count": len(prediction_list),
        "semantic_review_count": len(judgments),
        "diagnosis_correct_rate": sum(j["diagnosis_correct"] for j in judgments)
        / len(judgments)
        if judgments
        else None,
        "semantic_level_compliance_rate": sum(j["level_compliant"] for j in judgments)
        / len(judgments)
        if judgments
        else None,
        "compliance_pass_rate": sum(compliance_passes) / len(compliance_passes)
        if compliance_passes
        else 0.0,
    }
