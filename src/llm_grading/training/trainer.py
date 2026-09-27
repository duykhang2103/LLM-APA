"""Shared trainer construction for Task 1, Task 2, and Task 3 configs.

Keep optimizer, scheduler, checkpoint, evaluation, and logging setup here.
Task modules should supply data/target conversion and task-specific output
post-processing rather than creating unrelated training stacks.
"""

from typing import Any, Mapping


def build_trainer(config: Mapping[str, Any]) -> object:
    """Build the shared trainer described by a validated config."""
    # TODO: Connect the selected open-weight model, dataset, callbacks, and config.
    raise NotImplementedError("Training is not implemented in the scaffold.")
