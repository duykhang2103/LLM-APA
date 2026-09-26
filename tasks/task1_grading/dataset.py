"""Task 1 dataset view; feedback must never be exposed as model input."""

from typing import Any, Mapping


def build_dataset_view(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the future Task 1 training/evaluation input view."""
    # TODO: Select code, problem, and permitted auxiliary signals only.
    raise NotImplementedError("Task 1 dataset preparation is not implemented.")
