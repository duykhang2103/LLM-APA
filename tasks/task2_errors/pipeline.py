"""Task 2 multi-label classification pipeline boundary.

Implementation sequence for the owner:

1. Build a feedback-free input.
2. Generate one score per canonical taxonomy label.
3. Apply config-driven global or per-label thresholds.
4. Validate labels and allow an empty set.
5. Evaluate macro-F1, micro-F1, and per-label behavior.
"""

from typing import Any


class TaskPipeline:
    """Future Task 2 implementation surface used by shared CLI scripts."""

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        """Expected output: problem/code/evidence fields, never feedback."""
        # TODO: Delegate to the shared leakage-safe Task 2 preprocessing.
        raise NotImplementedError

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        """Expected output: label scores or a structured candidate."""
        # TODO: Produce label scores or structured labels from the selected model.
        raise NotImplementedError

    def postprocess(self, raw_output: object) -> dict[str, object]:
        """Expected output: labels from the central taxonomy only."""
        # TODO: Apply centralized taxonomy and configured thresholds.
        raise NotImplementedError

    def evaluate(
        self,
        predictions: list[dict[str, object]],
        references: list[dict[str, object]],
    ) -> dict[str, float]:
        """Expected output: macro/micro-F1 and per-label metrics."""
        # TODO: Delegate to macro-F1, micro-F1, and per-label metrics.
        raise NotImplementedError
