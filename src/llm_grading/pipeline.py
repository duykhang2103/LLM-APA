"""Factory for task pipelines used by all local commands."""

from typing import Any, Mapping


def build_task_pipeline(config: Mapping[str, Any]) -> Any:
    """Build a configured pipeline and reject unimplemented model adapters."""
    task = str(config.get("task", ""))
    method = str(config.get("method", ""))
    if method in {"zero_shot", "rag", "lora", "qlora"}:
        from llm_grading.prompting.pipeline import ModelPipeline
        return ModelPipeline(dict(config))
    if method != "heuristic":
        raise ValueError(f"Unknown method: {method}")
    if task == "task1":
        from tasks.task1_grading.pipeline import TaskPipeline
    elif task == "task2":
        from tasks.task2_errors.pipeline import TaskPipeline
    elif task == "task3":
        from tasks.task3_feedback.pipeline import TaskPipeline
    else:
        raise ValueError(f"Unknown task: {task!r}; expected task1, task2, or task3")
    return TaskPipeline(dict(config))
