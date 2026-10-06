"""Runnable Task 3 feedback pipeline and policy-checking boundary."""

from typing import Any

from llm_grading.data.preprocess import build_task3_input
from llm_grading.evaluation.task3 import evaluate_task3

from .compliance import check_compliance


class TaskPipeline:
    """Deterministic local baseline for controlled Vietnamese feedback."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        return build_task3_input(sample)

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        view = self.build_input(sample)
        labels = list(view.get("error_labels", []))
        level = int(view.get("feedback_level", 1))
        feedback = "Em hãy kiểm tra lại bài làm."
        if labels and level > 1:
            feedback = (
                "Em hay kiem tra lai phan xu ly duoc danh dau boi "
                + ", ".join(labels)
                + "."
            )
        return {
            "feedback": feedback,
            "feedback_level": level,
            "compliance": check_compliance(feedback, level),
        }

    def postprocess(self, raw_output: object) -> dict[str, object]:
        if not isinstance(raw_output, dict):
            raise ValueError("Task 3 output must be an object")
        feedback = str(raw_output.get("feedback", ""))
        level = int(raw_output.get("feedback_level", 1))
        result = dict(raw_output)
        result["feedback"] = feedback
        result["feedback_level"] = level
        result["compliance"] = check_compliance(feedback, level)
        return result

    def evaluate(
        self, predictions: list[dict[str, object]], references: list[dict[str, object]]
    ) -> dict[str, Any]:
        return evaluate_task3(predictions, references)
