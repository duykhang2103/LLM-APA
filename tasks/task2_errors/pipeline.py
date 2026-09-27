"""Runnable Task 2 pipeline and extension boundary for model adapters."""

from typing import Any

from llm_grading.data.preprocess import build_task2_input
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.evaluation.task2 import evaluate_task2
from .thresholds import apply_thresholds


class TaskPipeline:
    """Deterministic local baseline for multi-label error classification."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        return build_task2_input(sample)

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        view = self.build_input(sample)
        code = str(view.get("code", ""))
        statement = str(view.get("problem_statement", "")).lower()
        report = str(view.get("test_report", "")).lower()
        suspicious = ("sum" in statement and "-" in code) or any(word in report for word in ("failed", "fail", "error"))
        return {"scores": {"LABEL_02": 1.0 if suspicious else 0.0}}

    def postprocess(self, raw_output: object) -> dict[str, object]:
        if not isinstance(raw_output, dict):
            raise ValueError("Task 2 output must be an object")
        if "error_labels" in raw_output:
            labels = raw_output["error_labels"]
            if not isinstance(labels, list) or not set(labels).issubset(ERROR_LABELS):
                raise ValueError("Task 2 output contains invalid labels")
            return {"error_labels": sorted(labels, key=ERROR_LABELS.index)}
        thresholds = self.config.get("task2", {}).get("thresholds", {})
        return {"error_labels": apply_thresholds(raw_output.get("scores", {}), thresholds)}

    def evaluate(self, predictions: list[dict[str, object]], references: list[dict[str, object]]) -> dict[str, Any]:
        return evaluate_task2(predictions, references)
