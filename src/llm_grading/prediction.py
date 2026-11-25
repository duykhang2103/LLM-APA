"""Persist batch inference results while keeping schema failures identifiable."""

from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Iterable, Mapping

from llm_grading.runtime import write_json
from tasks.task3_feedback.compliance import SOFT_VIOLATIONS
from llm_grading.validation.predictions import validate_prediction_records
from llm_grading.validation.task3 import (
    task3_feedback_policy,
    task3_feedback_warnings,
    uses_joint_task3_output,
)


def validate_prediction_output_path(
    config: Mapping[str, Any], output_path: str | Path, extra_sources: Iterable[Any] = ()
) -> None:
    """Protect run metadata, configured inputs and CLI input files before any prediction writes."""
    directory = Path(config["saving"]["output_dir"])
    reserved = {
        (directory / name).resolve() for name in (
            "prediction_manifest.json", "resolved_config.json", "responses.jsonl",
            "warnings.json", "failures.json", "source_manifest.json", "config.json",
            "data_audit.json", "development_preview.json", "submission.zip",
            "submission.warnings.json",
        )
    }
    sources = [config.get("_config_path")]
    sources += [config.get("data", {}).get(key) for key in (
        "input_path", "train_path", "validation_path", "test_path",
    )]
    sources += list(extra_sources)
    reserved.update(Path(source).resolve() for source in sources if source)
    if Path(output_path).resolve() in reserved:
        raise ValueError("Prediction output aliases a reserved artifact or configured input")


def archive_previous_submission(directory: str | Path) -> str | None:
    """Retain a previous submission outside the current run's final ZIP path."""
    directory = Path(directory).resolve()
    artifacts = [directory / name for name in ("submission.zip", "submission.warnings.json")]
    artifacts = [path for path in artifacts if path.is_file()]
    if not artifacts:
        return None
    previous = Path(mkdtemp(prefix="previous-submission-", dir=directory))
    for path in artifacts:
        path.replace(previous / path.name)
    return str(previous)


def predict_samples(
    pipeline: Any,
    samples: list[dict[str, Any]],
    config: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Save each validated answer; warning-mode failures do not end the batch."""
    task = config["task"]
    keep_going = task == "task3" and task3_feedback_policy(config) == "warn"
    require_taxonomy = task == "task3" and uses_joint_task3_output(config)
    ids = [s.get("sample_id") for s in samples]
    if not ids or any(not isinstance(sid, str) or not sid for sid in ids):
        raise ValueError("Prediction inputs require nonempty sample IDs")
    if len(ids) != len(set(ids)):
        raise ValueError("Prediction inputs contain duplicate sample IDs")
    directory = Path(config["saving"]["output_dir"])
    output_path = Path(output_path)
    warning_path, failure_path = directory / "warnings.json", directory / "failures.json"
    validate_prediction_output_path(config, output_path)
    archive_previous_submission(directory)
    predictions, warnings, failures = [], [], []
    write_json(output_path, predictions)
    write_json(warning_path, warnings)
    write_json(failure_path, failures)
    for index, sample in enumerate(samples, 1):
        try:
            record = {
                "sample_id": sample["sample_id"],
                "output": pipeline.postprocess(pipeline.predict(sample)),
            }
            errors = validate_prediction_records(
                [record], task, require_task3_taxonomy=require_taxonomy
            )
            if errors:
                raise ValueError(str(errors))
            if task == "task3" and record["output"]["feedback_level"] != sample["feedback_level"]:
                raise ValueError("Prediction changed the requested feedback level")
            record_warnings = task3_feedback_warnings([record]) if task == "task3" else []
            # Soft violations were already accepted by the pipeline on its last retry
            # (Level 2 advice is cut later by scripts/relabel_task3.py).
            hard = [w for w in record_warnings if not set(w.get("violations") or []) <= SOFT_VIOLATIONS]
            if hard and not keep_going:
                raise ValueError("Feedback policy violation: " + str(record_warnings))
        except Exception as exc:
            failures.append({
                "sample_id": sample["sample_id"],
                "error": str(exc),
                "error_type": type(exc).__name__,
            })
            write_json(failure_path, failures)
            print(f"[{index}/{len(samples)}] failed {sample['sample_id']}; saved to {failure_path}", flush=True)
            if not keep_going:
                raise
            continue
        predictions.append(record)
        write_json(output_path, predictions)
        if record_warnings:
            warnings.extend(record_warnings)
            write_json(warning_path, warnings)
        suffix = " (feedback warning)" if record_warnings else ""
        print(f"[{index}/{len(samples)}] saved {sample['sample_id']}{suffix}", flush=True)
    return {
        "status": "incomplete" if failures else "completed",
        "prediction_count": len(predictions),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "predictions": str(output_path),
        "warnings": str(warning_path),
        "failures": str(failure_path),
    }
