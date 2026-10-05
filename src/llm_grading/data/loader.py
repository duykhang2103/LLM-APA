"""Normalize teacher task files or existing flat records without rewriting data."""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .schema import RUBRIC_RANGES, NormalizedSample, validate_rubric
from .taxonomy import ERROR_LABELS

TASK_FILENAMES = {
    "task1": "task1_grading.json",
    "task2": "task2_error_taxonomy.json",
    "task3": "task3_feedback.json",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _source_files(source: Path, task: str | None) -> list[Path]:
    if not source.is_dir():
        return [source]
    task_files = [source / name for name in TASK_FILENAMES.values() if (source / name).is_file()]
    if task_files:
        if task is not None:
            selected = source / TASK_FILENAMES[task]
            if not selected.is_file():
                raise FileNotFoundError(f"No {task} dataset file: {selected}")
            return [selected]
        if len(task_files) != 1:
            raise ValueError("Dataset directory contains multiple tasks; pass task='task1', 'task2', or 'task3'")
        return task_files
    return sorted(path for path in source.glob("*.json") if path.name not in {"exams.json", "label_space.json"})


def _load_exams(root: Path) -> dict[str, dict[str, Any]]:
    path = root / "exams.json"
    payload = _read_json(path)
    exams = payload.get("exams") if isinstance(payload, dict) else None
    if not isinstance(exams, list):
        raise ValueError(f"Expected an exams list in {path}")
    result = {}
    for exam in exams:
        if not isinstance(exam, dict) or not isinstance(exam.get("exam_id"), str) or not exam["exam_id"]:
            raise ValueError(f"Every exam needs a nonempty exam_id: {path}")
        exam_id = exam["exam_id"]
        if exam_id in result:
            raise ValueError(f"Duplicate exam_id {exam_id}: {path}")
        if not isinstance(exam.get("statement"), str) or not exam["statement"].strip():
            raise ValueError(f"Missing statement for exam {exam_id}: {path}")
        result[exam_id] = exam
    return result


def _read_code(code_file: object, root: Path, record_directory: Path, sample_id: str) -> str:
    if not isinstance(code_file, str) or not code_file:
        raise ValueError(f"Missing or invalid code_file for {sample_id}")
    reference = Path(code_file)
    if reference.anchor or reference.drive:
        raise ValueError(f"code_file must be relative to the dataset root for {sample_id}")
    candidates = [record_directory / reference, root / reference]
    for candidate in candidates:
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"code_file escapes the dataset root for {sample_id}: {code_file}")
        if resolved.is_file():
            return resolved.read_text(encoding="utf-8-sig")
    raise FileNotFoundError(f"Could not resolve code_file for {sample_id}: {code_file}")


def _error_labels(value: object, sample_id: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(label, str) for label in value):
        raise ValueError(f"taxonomy_error must be a list of strings for {sample_id}")
    unknown = set(value).difference(ERROR_LABELS)
    if unknown:
        raise ValueError(f"Unknown taxonomy_error for {sample_id}: {sorted(unknown)}")
    return list(value)


def _feedback_level(value: object, sample_id: str) -> int:
    if isinstance(value, str):
        match = re.fullmatch(r"Level\s+([1-4])(?:\s*-\s*.+)?", value.strip())
        value = int(match.group(1)) if match else None
    if isinstance(value, bool) or not isinstance(value, int) or value not in {1, 2, 3, 4}:
        raise ValueError(f"Invalid target_feedback_level for {sample_id}: expected Level 1 through Level 4")
    return value


def _raw_task(payload: dict[str, Any], path: Path, configured_task: str | None) -> str:
    declared = payload.get("task")
    if declared is not None and not isinstance(declared, str):
        raise ValueError(f"Invalid task declaration in {path}")
    inferred = declared.split("_", 1)[0] if declared else path.stem.split("_", 1)[0]
    if declared and inferred not in TASK_FILENAMES:
        raise ValueError(f"Unknown dataset task {declared!r} in {path}")
    if configured_task and inferred in TASK_FILENAMES and inferred != configured_task:
        raise ValueError(f"Configured task {configured_task} does not match dataset task {inferred}: {path}")
    selected = configured_task or inferred
    if selected not in TASK_FILENAMES:
        raise ValueError(f"Cannot identify dataset task in {path}; pass task='task1', 'task2', or 'task3'")
    return selected


def _normalize_raw(
    record: dict[str, Any], exams: dict[str, dict[str, Any]], root: Path, task: str,
) -> NormalizedSample:
    sample_id = record["sample_id"]
    inputs = record.get("input")
    output = record.get("output", {})
    if not isinstance(inputs, dict) or not isinstance(output, dict):
        raise ValueError(f"input and output must be objects for {sample_id}")
    if task == "task3":
        if "target_feedback_level" not in inputs:
            raise ValueError(f"Missing target_feedback_level for {sample_id}")
        if "taxonomy_error" not in inputs:
            raise ValueError(f"Missing input taxonomy_error for {sample_id}")
    target_field = {"task1": "rubric", "task2": "taxonomy_error", "task3": "feedback"}[task]
    if output and target_field not in output:
        raise ValueError(f"Missing {task} output {target_field} for {sample_id}")
    exam_id = inputs.get("exam_id")
    if not isinstance(exam_id, str) or exam_id not in exams:
        raise ValueError(f"Unknown exam_id {exam_id!r} for {sample_id}")
    exam = exams[exam_id]
    problem_type = inputs.get("exam_type", exam.get("exam_type"))
    if problem_type not in {"single_problem", "multi_problem"} or problem_type != exam.get("exam_type"):
        raise ValueError(f"Invalid or conflicting exam_type for {sample_id}")
    sample: NormalizedSample = {
        "sample_id": sample_id,
        "_has_reference": target_field in output,
        "problem_id": exam_id,
        "problem_type": problem_type,
        "problem_statement": exam["statement"],
        "language": inputs.get("language", exam.get("language", "cpp11")),
        "problems": exam.get("problems", []),
        "grading_policy": exam.get("grading_policy", ""),
        "code_file": inputs.get("code_file"),
        "code": _read_code(inputs.get("code_file"), root, root, sample_id),
        "compile_log": inputs.get("compile_log", ""),
        "test_report": inputs.get("test_report"),
        "rubric": {},
        "error_labels": [],
        "feedback": "",
        "feedback_level": 1,
    }
    if "rubric" in output:
        rubric = output["rubric"]
        if not isinstance(rubric, dict) or set(rubric) != set(RUBRIC_RANGES):
            raise ValueError(f"rubric must contain exactly the six dimensions for {sample_id}")
        validated = validate_rubric(rubric)
        total = output.get("total_score", validated["total"])
        if isinstance(total, bool) or not isinstance(total, int) or total != validated["total"]:
            raise ValueError(f"total_score disagrees with the rubric sum for {sample_id}")
        sample["rubric"] = dict(rubric)
        sample["total_score"] = total
    elif "total_score" in output:
        raise ValueError(f"total_score requires a rubric for {sample_id}")
    if "taxonomy_error" in output:
        sample["error_labels"] = _error_labels(output["taxonomy_error"], sample_id)
    if "target_feedback_level" in inputs:
        sample["feedback_level"] = _feedback_level(inputs["target_feedback_level"], sample_id)
        sample["error_labels"] = _error_labels(inputs.get("taxonomy_error", []), sample_id)
    if "feedback" in output:
        if not isinstance(output["feedback"], str):
            raise ValueError(f"feedback must be a string for {sample_id}")
        sample["feedback"] = output["feedback"]
    return sample


def load_samples(
    source: str | Path, max_samples: int | None = None, *, task: str | None = None,
) -> list[NormalizedSample]:
    """Load a task JSON or directory, validating all records before limiting.

    Teacher files contain ``samples`` with nested ``input``/``output`` and
    sibling ``exams.json``. A multi-task directory requires ``task`` because
    the files share sample IDs. Flat normalized object/list files still work.
    No source data is modified, and compiler/test evidence is preserved.
    """
    if task is not None and task not in TASK_FILENAMES:
        raise ValueError(f"Unknown task: {task!r}")
    if max_samples is not None and (isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 0):
        raise ValueError("max_samples must be a nonnegative integer or null")
    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Sample source does not exist: {source_path}")
    root = source_path if source_path.is_dir() else source_path.parent
    paths = _source_files(source_path, task)
    if not paths:
        raise FileNotFoundError(f"No JSON files found under: {source_path}")
    records: list[NormalizedSample] = []
    for path in paths:
        payload = _read_json(path)
        is_raw = isinstance(payload, dict) and "samples" in payload
        raw_task = _raw_task(payload, path, task) if is_raw else None
        exams = _load_exams(root) if is_raw else {}
        items = payload["samples"] if is_raw else ([payload] if isinstance(payload, dict) else payload)
        if not isinstance(items, list):
            raise ValueError(f"Expected an object or list in {path}")
        if is_raw and "n_samples" in payload and payload["n_samples"] != len(items):
            raise ValueError(f"n_samples does not match the number of records in {path}")
        for record in items:
            if not isinstance(record, dict) or not isinstance(record.get("sample_id"), str) or not record["sample_id"]:
                raise ValueError(f"Every sample must be an object with sample_id: {path}")
            if is_raw:
                sample = _normalize_raw(record, exams, root, raw_task)
            else:
                sample = dict(record)
                if not sample.get("code") and sample.get("code_file"):
                    sample["code"] = _read_code(sample["code_file"], root, path.parent, sample["sample_id"])
                sample.setdefault("code", "")
            records.append(sample)
    duplicates = sorted(sample_id for sample_id, count in Counter(record["sample_id"] for record in records).items() if count > 1)
    if duplicates:
        raise ValueError(f"Duplicate sample_id values: {duplicates}")
    return records if max_samples is None else records[:max_samples]
