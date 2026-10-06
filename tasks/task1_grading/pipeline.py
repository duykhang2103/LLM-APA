"""Runnable Task 1 pipeline and extension boundary for model adapters."""

from typing import Any, Mapping

from llm_grading.data.preprocess import build_task1_input
from llm_grading.evaluation.task1 import evaluate_task1

from .postprocess import postprocess_rubric


class TaskPipeline:
    """Deterministic local baseline for rubric grading."""

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        self.config = dict(config or {})

    def build_input(self, sample: dict[str, object]) -> dict[str, object]:
        return build_task1_input(sample)

    def predict(self, sample: dict[str, object]) -> dict[str, object]:
        view = self.build_input(sample)
        code = str(view.get("code", ""))
        compile_log = str(view.get("compile_log", "")).lower()
        test_report = str(view.get("test_report", "")).lower()
        statement = str(view.get("problem_statement", "")).lower()
        compile_ok = not any(word in compile_log for word in ("fail", "error", "not compile"))
        io_ok = int("cin" in code and ("cout" in code or "printf" in code))
        logic_ok = int("+" in code or ("sum" not in statement and "main" in code))
        edge_ok = 0 if any(word in test_report for word in ("failed", "fail", "0/")) else 1
        quality_ok = int("main" in code and ("return" in code or "}" in code))
        return {
            "compilable": int(compile_ok),
            "io_format": io_ok,
            "logic": min(4, logic_ok),
            "edge_case": min(2, edge_ok),
            "complexity": 1,
            "code_quality": quality_ok,
        }

    def postprocess(self, raw_output: object) -> dict[str, object]:
        return postprocess_rubric(raw_output)

    def evaluate(self, predictions: list[dict[str, object]], references: list[dict[str, object]]) -> dict[str, Any]:
        return evaluate_task1(predictions, references)
