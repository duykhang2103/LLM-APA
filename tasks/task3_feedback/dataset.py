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
    # TODO: Include labels and level policy while keeping feedback as a target.
    raise NotImplementedError("Task 3 dataset preparation is not implemented.")
