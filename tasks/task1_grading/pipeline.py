"""Task 1 rubric-grading pipeline boundary.

Implementation sequence for the owner:

1. ``build_input`` calls the shared leakage-safe builder.
2. ``predict`` calls prompting or the shared model runner.
3. ``postprocess`` parses six components and calculates ``total``.
4. ``evaluate`` delegates to the shared Task 1 metric module.

The shared CLI should be able to call this class without knowing Task 1's
internal prompt or policy details.
"""

from typing import Any, Mapping


class TaskPipeline:
    """Future Task 1 implementation surface used by shared CLI scripts."""

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        """Expected output: problem/code/evidence fields, never feedback."""
        # TODO: Delegate to the shared leakage-safe Task 1 preprocessing.
        raise NotImplementedError

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        """Expected output: raw structured candidate before final validation."""
        # TODO: Call the selected prompting or fine-tuned model.
        raise NotImplementedError

    def postprocess(self, raw_output: object) -> dict[str, object]:
        """Expected output: six bounded components plus derived total."""
        # TODO: Parse six components and derive the total deterministically.
        raise NotImplementedError

    def evaluate(
        self,
        predictions: list[dict[str, object]],
        references: list[dict[str, object]],
    ) -> dict[str, float]:
        """Expected output: QWK, MAE, and component exact-match metrics."""
        # TODO: Delegate to QWK, MAE, and per-dimension exact-match metrics.
        raise NotImplementedError
