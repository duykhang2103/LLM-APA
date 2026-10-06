"""Generate held-out completions and expose task metrics for model selection."""

from llm_grading.evaluation.task1 import evaluate_task1
from llm_grading.evaluation.task2 import evaluate_task2
from llm_grading.evaluation.task3 import evaluate_task3
from llm_grading.prompting.formatter import parse_response, render_task_prompt


def evaluate_generated(bundle, samples, config, *, generate=None):
    if generate is None:
        from llm_grading.models.generation import generate_text

        generate = generate_text
    task = config["task"]
    predictions, valid, references = [], [], []
    for sample in samples:
        raw = generate(
            bundle,
            render_task_prompt(sample, task, config),
            config.get("generation", {}),
        )
        record = {"sample_id": sample["sample_id"], "raw_response": raw}
        try:
            output = parse_response(raw, task, sample)
            record["output"] = output
            valid.append({"output": output})
            references.append(sample)
        except ValueError as error:
            record["parse_error"] = str(error)
        predictions.append(record)
    evaluator = {
        "task1": evaluate_task1,
        "task2": evaluate_task2,
        "task3": evaluate_task3,
    }[task]
    metrics = evaluator(valid, references)
    valid_rate = len(valid) / len(samples) if samples else 0.0
    primary = (
        (metrics["qwk_total"] + 1) / 2
        if task == "task1"
        else (
            metrics["macro_f1"] if task == "task2" else metrics["compliance_pass_rate"]
        )
    )
    metrics.update(
        valid_count=len(valid),
        invalid_count=len(samples) - len(valid),
        valid_rate=valid_rate,
        selection_score=primary * valid_rate if task != "task3" else None,
    )
    if task == "task3":
        metrics["heuristic_compliance_score"] = primary * valid_rate
    return {
        "metrics": metrics,
        "predictions": predictions,
        "semantics": "First-pass generation; task metrics cover parsed outputs. Selection score penalizes invalid outputs for Task1/2. Task3 compliance is diagnostic only; no automatic checkpoint selection without independent semantic review.",
    }


def make_task_trainer(enabled, bundle, samples, config, directory):
    from transformers import Trainer

    if not enabled:
        return Trainer

    class TaskTrainer(Trainer):
        def evaluate(
            self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"
        ):
            if self.args.world_size != 1:
                raise ValueError(
                    "Generated checkpoint evaluation currently supports single-process training"
                )
            metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
            model = self.model
            was_training = model.training
            model.eval()
            try:
                with self.compute_loss_context_manager():
                    report = evaluate_generated(
                        {**bundle, "model": model}, samples, config
                    )
            finally:
                model.train(was_training)
            from llm_grading.runtime import write_json

            write_json(
                directory
                / "generated_validation"
                / f"step-{self.state.global_step}.json",
                report,
            )
            metrics.update(
                {
                    f"{metric_key_prefix}_{key}": value
                    for key, value in report["metrics"].items()
                    if isinstance(value, (int, float))
                }
            )
            key = (
                "heuristic_compliance_score"
                if config["task"] == "task3"
                else "selection_score"
            )
            self.log({f"{metric_key_prefix}_{key}": report["metrics"][key]})
            return metrics

    return TaskTrainer
