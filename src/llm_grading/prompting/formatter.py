"""Render only whitelisted task inputs; strictly parse model responses."""

import hashlib
import json
from pathlib import Path

from llm_grading.data.preprocess import (
    build_task1_input,
    build_task2_input,
    build_task3_input,
)
from llm_grading.data.schema import RUBRIC_RANGES, validate_rubric
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.validation.task3 import (
    uses_joint_task3_output,
    validate_taxonomy_error,
)
from tasks.task3_feedback.compliance import check_compliance

ROOT = Path(__file__).resolve().parents[3]
TASK_DIRS = {
    "task1": "task1_grading",
    "task2": "task2_errors",
    "task3": "task3_feedback",
}
BUILDERS = {
    "task1": build_task1_input,
    "task2": build_task2_input,
    "task3": build_task3_input,
}


def format_prompt(template, values):
    try:
        return template.format_map(dict(values))
    except KeyError as e:
        raise ValueError(f"Prompt requested unavailable field: {e.args[0]}") from e


def task_template(task, config):
    version = config.get("prompt", {}).get("version", "v001")
    path = (
        config.get("prompt", {}).get("path")
        or ROOT / "tasks" / TASK_DIRS[task] / "prompts" / f"{version}.txt"
    )
    return Path(path).read_text(encoding="utf-8")


def prompt_hash(task, config):
    return hashlib.sha256(task_template(task, config).encode()).hexdigest()


def render_task_prompt(sample, task, config, examples=None):
    template = task_template(task, config)
    view = BUILDERS[task](sample)
    if task == "task3" and uses_joint_task3_output(config):
        # Legacy examples provide labels as inputs; joint prediction must infer them.
        view.pop("error_labels", None)
        examples = [
            {
                **example,
                "input": {
                    key: value for key, value in example["input"].items()
                    if key != "error_labels"
                },
                "target": json.loads(example["target"])
                if isinstance(example["target"], str) else example["target"],
            }
            for example in (examples or [])
        ]
    settings = config.get(task, {})
    for field in ("compile_log", "test_report"):
        if settings.get(f"use_{field}", True) is False:
            view.pop(field, None)
    # Only serialized whitelisted records are substituted; braces in C++ stay literal.
    return format_prompt(
        template,
        {
            "input_json": json.dumps(view, ensure_ascii=False),
            "taxonomy": json.dumps(list(ERROR_LABELS), ensure_ascii=False),
            "examples": json.dumps(examples or [], ensure_ascii=False),
        },
    )


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def _clean_json_text(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("{"):
        # Fences inside a JSON feedback string are data, not an outer wrapper.
        return cleaned
    if "```" in cleaned:
        start = cleaned.find("```")
        end = cleaned.rfind("```")
        if start != -1 and end != -1 and end > start:
            inner = cleaned[start:end]
            if inner.startswith("```json"):
                inner = inner[7:]
            elif inner.startswith("```"):
                inner = inner[3:]
            cleaned = inner.strip()
    if not cleaned.startswith("{") and "{" in cleaned and "}" in cleaned:
        s = cleaned.find("{")
        e = cleaned.rfind("}")
        if s != -1 and e != -1 and e > s:
            cleaned = cleaned[s : e + 1]
    return cleaned


def parse_response(text, task, sample, config=None):
    if task == "task3" and not uses_joint_task3_output(config):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Empty feedback")
        level = sample["feedback_level"]
        return {
            "feedback": text.strip(),
            "feedback_level": level,
            "compliance": check_compliance(text, level),
        }
    cleaned_text = _clean_json_text(str(text))
    try:
        obj = json.loads(cleaned_text, object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    if not isinstance(obj, dict):
        raise ValueError("Response must be a JSON object")
    if task == "task3":
        if set(obj) != {"taxonomy_error", "feedback"}:
            raise ValueError("Return exactly taxonomy_error and feedback")
        labels = validate_taxonomy_error(obj["taxonomy_error"])
        feedback = obj["feedback"]
        if not isinstance(feedback, str) or not feedback.strip():
            raise ValueError("feedback must be nonempty text")
        return {
            "taxonomy_error": labels,
            "feedback": feedback.strip(),
            "feedback_level": sample["feedback_level"],
            "compliance": check_compliance(feedback, sample["feedback_level"]),
        }
    if task == "task1":
        if set(obj) != set(RUBRIC_RANGES):
            raise ValueError(
                "Return exactly the six rubric fields; no total or extra fields"
            )
        return validate_rubric(obj)
    if task != "task2" or set(obj) != {"error_labels"}:
        raise ValueError("Return exactly error_labels")
    labels = obj["error_labels"]
    if not isinstance(labels, list) or any(not isinstance(x, str) for x in labels):
        raise ValueError("error_labels must be a list of strings")
    if len(labels) != len(set(labels)) or set(labels) - set(ERROR_LABELS):
        raise ValueError("Labels must be unique members of the official taxonomy")
    return obj
