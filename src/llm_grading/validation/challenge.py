"""Lecturer sample output contract, separate from internal diagnostics."""

from typing import Any

from llm_grading.data.schema import RUBRIC_RANGES, validate_rubric
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.validation.predictions import validate_prediction_records
from tasks.task3_feedback.compliance import check_compliance


def validate_challenge_predictions(records: Any, task: str) -> list[str]:
    if task not in {"task1", "task2", "task3"} or not isinstance(records, list):
        return ["Unknown task or predictions root is not a list"]
    errors, seen = [], set()
    expected_keys = {
        "task1": {"rubric", "total_score"},
        "task2": {"taxonomy_error"},
        "task3": {"feedback"},
    }[task]
    for i, row in enumerate(records):
        try:
            if not isinstance(row, dict) or set(row) != {"sample_id", "output"}:
                raise ValueError("Expected exactly sample_id and output")
            sid = row["sample_id"]
            if not isinstance(sid, str) or not sid or sid in seen:
                raise ValueError("Missing or duplicate sample_id")
            seen.add(sid)
            output = row["output"]
            if not isinstance(output, dict) or set(output) != expected_keys:
                raise ValueError("Unexpected challenge output fields")
            if task == "task1":
                if not isinstance(output["rubric"], dict) or set(
                    output["rubric"]
                ) != set(RUBRIC_RANGES):
                    raise ValueError("Expected exactly six rubric dimensions")
                total = validate_rubric(output["rubric"])["total"]
                if (
                    type(output["total_score"]) is not int
                    or output["total_score"] != total
                ):
                    raise ValueError("total_score must equal rubric sum")
            elif task == "task2":
                labels = output["taxonomy_error"]
                if (
                    not isinstance(labels, list)
                    or any(not isinstance(v, str) for v in labels)
                    or len(labels) != len(set(labels))
                    or set(labels) - set(ERROR_LABELS)
                ):
                    raise ValueError(
                        "taxonomy_error must contain unique official labels"
                    )
            elif (
                not isinstance(output["feedback"], str)
                or not output["feedback"].strip()
            ):
                raise ValueError("feedback must be nonempty text")
        except ValueError as error:
            errors.append(f"record {i}: {error}")
    return errors


def export_challenge_predictions(
    records: list[dict[str, Any]], task: str
) -> list[dict[str, Any]]:
    errors = validate_prediction_records(records, task)
    if errors:
        raise ValueError(str(errors))
    exported = []
    for row in records:
        output = row["output"]
        if task == "task1":
            rubric = validate_rubric(output)
            target = {
                "rubric": {k: rubric[k] for k in RUBRIC_RANGES},
                "total_score": rubric["total"],
            }
        elif task == "task2":
            target = {"taxonomy_error": list(output["error_labels"])}
        else:
            if not check_compliance(output["feedback"], output["feedback_level"])[
                "pass"
            ]:
                raise ValueError(
                    f"Feedback policy violation for {row['sample_id']}; regenerate before export"
                )
            target = {"feedback": output["feedback"]}
        exported.append({"sample_id": row["sample_id"], "output": target})
    errors = validate_challenge_predictions(exported, task)
    if errors:
        raise ValueError(str(errors))
    return exported
