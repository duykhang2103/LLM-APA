"""Task 3 controlled-feedback generation pipeline boundary.

Implementation sequence for the owner:

1. Build input from problem, code, known labels, and requested level.
2. Generate a candidate in Vietnamese.
3. Run the centralized compliance checker.
4. Regenerate or revise when the level policy is violated.
5. Return feedback plus compliance diagnostics for evaluation.
"""

from typing import Any


class TaskPipeline:
    """Future Task 3 implementation surface used by shared CLI scripts."""

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        """Expected output: problem/code/labels/level, not reference feedback."""
        # TODO: Include problem, code, labels, and required level.
        raise NotImplementedError

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        """Expected output: candidate feedback and checker result."""
        # TODO: Generate, check, and optionally regenerate feedback.
        raise NotImplementedError

    def postprocess(self, raw_output: object) -> dict[str, object]:
        """Expected output: feedback text, level, pass flag, and violations."""
        # TODO: Return feedback text plus compliance diagnostics.
        raise NotImplementedError

    def evaluate(
        self,
        predictions: list[dict[str, object]],
        references: list[dict[str, object]],
    ) -> dict[str, float]:
        """Expected output: diagnosis and compliance metrics separately."""
        # TODO: Evaluate diagnosis correctness and level compliance.
        raise NotImplementedError
