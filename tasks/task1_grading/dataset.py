"""Task 1 dataset view; feedback must never be exposed as model input.

Input example::

    {"problem_statement": "...", "code": "...",
     "compile_log": "...", "test_report": "..."}

Target example::

    {"compilable": 1, "io_format": 1, "logic": 3,
     "edge_case": 1, "complexity": 1, "code_quality": 1}
"""

from typing import Any, Mapping


def build_dataset_view(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Task 1 input/target view from one normalized sample."""
    from llm_grading.data.preprocess import build_task1_input

    return {"input": build_task1_input(sample), "target": dict(sample.get("rubric", {}))}
