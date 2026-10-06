"""Evaluate saved predictions without loading a model or requiring CUDA."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.evaluation.task1 import evaluate_task1
from llm_grading.evaluation.task2 import evaluate_task2
from llm_grading.evaluation.task3 import evaluate_task3
from llm_grading.runtime import load_samples_for_config, read_json, write_json
from llm_grading.utils.config import load_config
from llm_grading.utils.logging import sample_metadata
from llm_grading.validation.predictions import validate_prediction_records

EVALUATORS = {"task1": evaluate_task1, "task2": evaluate_task2, "task3": evaluate_task3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "validation", "test"]
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--judgments", help="Private Task 3 review JSON keyed by sample_id"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    task = config["task"]
    predictions = read_json(args.predictions)
    errors = validate_prediction_records(predictions, task)
    if errors:
        raise ValueError(str(errors))
    samples = load_samples_for_config(config, args.split)
    by_id = {s["sample_id"]: s for s in samples}
    if {p["sample_id"] for p in predictions} != set(by_id):
        raise ValueError("Prediction IDs must exactly match configured reference split")
    references = [by_id[p["sample_id"]] for p in predictions]
    if args.judgments:
        judgments = read_json(args.judgments)
        if (
            task != "task3"
            or not isinstance(judgments, dict)
            or set(judgments) - set(by_id)
        ):
            raise ValueError("Judgments must reference only Task 3 prediction IDs")
        for prediction in predictions:
            if prediction["sample_id"] in judgments:
                prediction["output"]["judgment"] = judgments[prediction["sample_id"]]
    target_key = {"task1": "rubric", "task2": "error_labels", "task3": "feedback"}[task]
    if any(s.get("_has_reference") is False or target_key not in s for s in references):
        raise ValueError(
            "This split has no teacher references; validate/package predictions, do not evaluate or invent targets"
        )
    evaluator = EVALUATORS[task]
    metrics = evaluator(predictions, references)
    slices = {}
    for field in [
        "exam_id",
        "problem_type",
        "feedback_level",
        "evidence_conflict_flag",
    ]:
        groups = {}
        for i, s in enumerate(references):
            value = str(sample_metadata(s).get(field))
            groups.setdefault(value, []).append(i)
        for value, indices in groups.items():
            slices[f"{field}={value}"] = {
                "count": len(indices),
                "metrics": evaluator(
                    [predictions[i] for i in indices], [references[i] for i in indices]
                ),
            }
    output = args.output or Path(config["saving"]["output_dir"]) / "metrics.json"
    write_json(
        output,
        {
            "experiment_id": config["experiment"]["id"],
            "task": task,
            "metrics": metrics,
            "slices": slices,
            "metric_semantics": "Task2: all ten labels, zero_division=0; Task3 heuristic compliance plus optional independent semantic judgments",
        },
    )
    print(f"Evaluated {len(predictions)} predictions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
