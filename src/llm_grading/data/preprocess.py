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
    # TODO: Select problem/code/auxiliary signals and assert feedback is absent.
    raise NotImplementedError("Task 1 preprocessing is not implemented in the scaffold.")


def build_task2_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create a Task 2 input using the same evidence boundary as Task 1.

    The target labels are reference data for training/evaluation, not input
    fields. A future implementation should make this boundary easy to audit.
    """
    # TODO: Select problem/code/auxiliary signals and assert feedback is absent.
    raise NotImplementedError("Task 2 preprocessing is not implemented in the scaffold.")


def build_task3_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create a Task 3 input with known error labels and requested level.

    Example shape::

        {"problem_statement": "...", "code": "...",
         "error_labels": ["LABEL_02"], "feedback_level": 1}

    The reference ``feedback`` remains a training target or evaluation
    reference and is not copied into the inference prompt.
    """
    # TODO: Include the label context and level policy required for generation.
    raise NotImplementedError("Task 3 preprocessing is not implemented in the scaffold.")
