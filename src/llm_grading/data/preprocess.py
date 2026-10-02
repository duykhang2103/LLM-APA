"""Build task-specific views while enforcing input-boundary rules.

The normalized record contains references and labels for every task. A task
builder is the whitelist that decides what the model is allowed to see.
Never implement this by copying the entire sample and deleting fields later;
construct an explicit dictionary for each task.
"""

from typing import Any

from .schema import NormalizedSample


def build_task1_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create a Task 1 input from problem, code, and permitted evidence.

    Example shape::

        {"problem_statement": "...", "code": "...",
         "compile_log": "...", "test_report": "..."}

    ``rubric``, ``error_labels``, and ``feedback`` must not be included.
    """
    result = {
        "problem_statement": sample.get("problem_statement", ""),
        "code": sample.get("code", ""),
        "compile_log": sample.get("compile_log", ""),
        "test_report": sample.get("test_report", ""),
    }
    for key in ("problem_type", "language", "problems", "grading_policy"):
        if key in sample:
            result[key] = sample[key]
    assert "feedback" not in result
    return result


def build_task2_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create a Task 2 input using the same evidence boundary as Task 1.

    The target labels are reference data for training/evaluation, not input
    fields. A future implementation should make this boundary easy to audit.
    """
    result = {
        "problem_statement": sample.get("problem_statement", ""),
        "code": sample.get("code", ""),
        "compile_log": sample.get("compile_log", ""),
        "test_report": sample.get("test_report", ""),
    }
    for key in ("problem_type", "language", "problems", "grading_policy"):
        if key in sample:
            result[key] = sample[key]
    assert "feedback" not in result
    return result


def build_task3_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create a Task 3 input with known error labels and requested level.

    Example shape::

        {"problem_statement": "...", "code": "...",
         "error_labels": ["Lỗi logic"], "feedback_level": 1}

    The reference ``feedback`` remains a training target or evaluation
    reference and is not copied into the inference prompt.
    """
    result = {
        "problem_statement": sample.get("problem_statement", ""),
        "code": sample.get("code", ""),
        "error_labels": list(sample.get("error_labels", [])),
        "feedback_level": int(sample.get("feedback_level", 1)),
    }
    for key in ("problem_type", "language", "problems", "grading_policy"):
        if key in sample:
            result[key] = sample[key]
    return result
