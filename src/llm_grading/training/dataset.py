"""Convert normalized samples into the format expected by a trainer."""

from typing import Any, Iterable, Mapping


def build_training_dataset(samples: Iterable[Mapping[str, Any]], task: str) -> object:
    """Describe the future task-aware training-dataset conversion."""
    # TODO: Apply task-specific targets without duplicating shared loading logic.
    raise NotImplementedError("Training dataset conversion is not implemented in the scaffold.")
