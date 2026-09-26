"""Task 1 rubric-grading pipeline boundary."""

from typing import Any, Mapping


class TaskPipeline:
    """Future Task 1 implementation surface used by shared CLI scripts."""

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        # TODO: Delegate to the shared leakage-safe Task 1 preprocessing.
        raise NotImplementedError

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        # TODO: Call the selected prompting or fine-tuned model.
        raise NotImplementedError

    def postprocess(self, raw_output: object) -> dict[str, object]:
        # TODO: Parse six components and derive the total deterministically.
        raise NotImplementedError

    def evaluate(
        self,
        predictions: list[dict[str, object]],
        references: list[dict[str, object]],
    ) -> dict[str, float]:
        # TODO: Delegate to QWK, MAE, and per-dimension exact-match metrics.
        raise NotImplementedError
