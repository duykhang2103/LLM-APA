"""Convert normalized samples into the format expected by a trainer.

The converter makes one explicit decision per task:

- Task 1 target: six rubric components, with total derived later.
- Task 2 target: multi-label taxonomy representation.
- Task 3 target: feedback text conditioned on known labels and requested level.

Keep source sample IDs in every training row for debugging and split audits.
"""

from typing import Any, Iterable, Mapping


def build_training_dataset(samples: Iterable[Mapping[str, Any]], task: str) -> object:
    """Convert normalized records into task-specific training examples."""
    # TODO: Apply task-specific targets without duplicating shared loading logic.
    raise NotImplementedError("Training dataset conversion is not implemented in the scaffold.")
