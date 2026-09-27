"""Task 2 dataset view; feedback must never be exposed as model input.

Input example::

    {"problem_statement": "...", "code": "...", "compile_log": "..."}

Target example::

    {"error_labels": ["LABEL_02"]}

The target can also be ``{"error_labels": []}`` for a clean submission.
"""

from typing import Any, Mapping


def build_dataset_view(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Task 2 input/target view from one normalized sample."""
    from llm_grading.data.preprocess import build_task2_input

    return {"input": build_task2_input(sample), "target": {"error_labels": list(sample.get("error_labels", []))}}
