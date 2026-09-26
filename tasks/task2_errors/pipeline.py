"""Task 2 multi-label classification pipeline boundary."""

from typing import Any


class TaskPipeline:
    """Future Task 2 implementation surface used by shared CLI scripts."""

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        # TODO: Delegate to the shared leakage-safe Task 2 preprocessing.
        raise NotImplementedError

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        # TODO: Produce label scores or structured labels from the selected model.
        raise NotImplementedError

    def postprocess(self, raw_output: object) -> dict[str, object]:
        # TODO: Apply centralized taxonomy and configured thresholds.
        raise NotImplementedError

    def evaluate(
        self,
        predictions: list[dict[str, object]],
        references: list[dict[str, object]],
    ) -> dict[str, float]:
        # TODO: Delegate to macro-F1, micro-F1, and per-label metrics.
        raise NotImplementedError
