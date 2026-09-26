"""Task 2 dataset view; feedback must never be exposed as model input."""

from typing import Any, Mapping


def build_dataset_view(sample: Mapping[str, Any]) -> dict[str, Any]:
    """Describe the future Task 2 training/evaluation input view."""
    # TODO: Select code, problem, and permitted auxiliary signals only.
    raise NotImplementedError("Task 2 dataset preparation is not implemented.")
