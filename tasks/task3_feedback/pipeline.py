"""Task 3 controlled-feedback generation pipeline boundary."""

from typing import Any


class TaskPipeline:
    """Future Task 3 implementation surface used by shared CLI scripts."""

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        # TODO: Include problem, code, labels, and required level.
        raise NotImplementedError

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        # TODO: Generate, check, and optionally regenerate feedback.
        raise NotImplementedError

    def postprocess(self, raw_output: object) -> dict[str, object]:
        # TODO: Return feedback text plus compliance diagnostics.
        raise NotImplementedError

    def evaluate(
        self,
        predictions: list[dict[str, object]],
        references: list[dict[str, object]],
    ) -> dict[str, float]:
        # TODO: Evaluate diagnosis correctness and level compliance.
        raise NotImplementedError
