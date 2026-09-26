"""Build task-specific views while enforcing input-boundary rules."""

from typing import Any

from .schema import NormalizedSample


def build_task1_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create the future Task 1 model input without feedback."""
    # TODO: Select problem/code/auxiliary signals and assert feedback is absent.
    raise NotImplementedError("Task 1 preprocessing is not implemented in the scaffold.")


def build_task2_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create the future Task 2 model input without feedback."""
    # TODO: Select problem/code/auxiliary signals and assert feedback is absent.
    raise NotImplementedError("Task 2 preprocessing is not implemented in the scaffold.")


def build_task3_input(sample: NormalizedSample) -> dict[str, Any]:
    """Create the future Task 3 input with labels and requested level."""
    # TODO: Include the label context and level policy required for generation.
    raise NotImplementedError("Task 3 preprocessing is not implemented in the scaffold.")
