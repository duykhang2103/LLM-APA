"""Shared contracts for legacy feedback and release taxonomy-plus-feedback."""

from typing import Any, Mapping

from llm_grading.data.taxonomy import ERROR_LABELS
from tasks.task3_feedback.compliance import check_compliance


def task3_feedback_policy(config: Mapping[str, Any] | None) -> str:
    """Distinguish optional policy warnings from mandatory response schemas."""
    policy = (config or {}).get("task3", {}).get("feedback_policy", "strict")
    if policy not in ("strict", "warn"):
        raise ValueError("task3.feedback_policy must be strict or warn")
    return policy


def task3_feedback_warnings(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Recompute warnings from validated internal records, not model claims."""
    warnings = []
    for row in records:
        output = row["output"]
        diagnostic = check_compliance(output["feedback"], output["feedback_level"])
        if not diagnostic["pass"]:
            warnings.append({
                "sample_id": row["sample_id"],
                "feedback_level": output["feedback_level"],
                "violations": diagnostic["violations"],
                "scope": diagnostic.get("scope", "heuristic"),
            })
    return warnings


def uses_joint_task3_output(config: Mapping[str, Any] | None) -> bool:
    """Select an explicit output contract without changing recorded old runs."""
    output_format = (config or {}).get("task3", {}).get("output_format", "feedback")
    if output_format not in ("feedback", "taxonomy_feedback"):
        raise ValueError("task3.output_format must be feedback or taxonomy_feedback")
    return output_format == "taxonomy_feedback"


def validate_taxonomy_error(value: object) -> list[str]:
    """Require a predicted list of unique official labels; never invent []."""
    if not isinstance(value, list) or any(not isinstance(label, str) for label in value):
        raise ValueError("taxonomy_error must be a list of unique official labels")
    if len(value) != len(set(value)) or set(value) - set(ERROR_LABELS):
        raise ValueError("taxonomy_error must contain unique official labels")
    return list(value)
