"""Task 3 dataset view for labels, requested level, and feedback targets.

Input example::

    {"problem_statement": "...", "code": "...",
     "error_labels": ["LABEL_02"], "feedback_level": 1}

Target example::

    {"feedback": "Hãy kiểm tra lại phép toán..."}
"""

from typing import Any, Mapping


def build_dataset_view(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Build the Task 3 input/target view from one normalized sample."""
    from llm_grading.data.preprocess import build_task3_input

    return {"input": build_task3_input(sample), "target": {"feedback": sample.get("feedback", "")}}
