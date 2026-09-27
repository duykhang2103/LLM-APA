"""Validate task-specific predictions before leaderboard submission.

Use the synthetic files under ``examples/predictions/`` as the first schema
fixtures. Validation should check JSON shape, sample IDs, task output fields,
score ranges, taxonomy membership, and required feedback-level diagnostics.
"""

from pathlib import Path
import json
from typing import Any

from llm_grading.data.taxonomy import ERROR_LABELS
from tasks.task1_grading.postprocess import RUBRIC_RANGES, postprocess_rubric


def validate_predictions(path: str | Path, task: str) -> list[str]:
    """Return validation messages; an empty list means the file is valid."""
    prediction_path = Path(path)
    if not prediction_path.exists():
        return [f"Prediction file does not exist: {prediction_path}"]
    try:
        payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"Could not read prediction JSON: {error}"]
    return validate_prediction_records(payload, task)


def validate_prediction_records(records: Any, task: str) -> list[str]:
    """Validate already-loaded prediction records and return all messages."""
    messages: list[str] = []
    if not isinstance(records, list):
        return ["Prediction root must be a JSON list"]
    seen: set[str] = set()
    for index, record in enumerate(records):
        prefix = f"record {index}"
        if not isinstance(record, dict):
            messages.append(f"{prefix}: must be an object")
            continue
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            messages.append(f"{prefix}: missing sample_id")
        elif sample_id in seen:
            messages.append(f"{prefix}: duplicate sample_id {sample_id}")
        else:
            seen.add(sample_id)
        output = record.get("output")
        if not isinstance(output, dict):
            messages.append(f"{prefix}: output must be an object")
            continue
        if task == "task1":
            try:
                postprocess_rubric(output)
            except ValueError as error:
                messages.append(f"{prefix}: {error}")
        elif task == "task2":
            labels = output.get("error_labels")
            if not isinstance(labels, list):
                messages.append(f"{prefix}: error_labels must be a list")
            else:
                unknown = sorted(set(labels).difference(ERROR_LABELS))
                if unknown:
                    messages.append(f"{prefix}: unknown Task 2 labels {unknown}")
        elif task == "task3":
            feedback = output.get("feedback")
            if not isinstance(feedback, str) or not feedback.strip():
                messages.append(f"{prefix}: missing Task 3 feedback")
            level = output.get("feedback_level")
            if not isinstance(level, int) or level not in {1, 2, 3, 4}:
                messages.append(f"{prefix}: feedback_level must be 1, 2, 3, or 4")
            compliance = output.get("compliance")
            if not isinstance(compliance, dict) or not isinstance(compliance.get("pass"), bool) or not isinstance(compliance.get("violations"), list):
                messages.append(f"{prefix}: missing compliance information")
        else:
            messages.append(f"Unknown task: {task}")
    return messages
