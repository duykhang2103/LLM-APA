"""Evaluate a prediction JSON file against normalized sample references."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_grading.pipeline import build_task_pipeline
from llm_grading.runtime import load_samples_for_config, read_json, write_json
from llm_grading.utils.config import load_config
from llm_grading.validation.predictions import validate_prediction_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "validation", "test"))
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    predictions = read_json(args.predictions)
    messages = validate_prediction_records(predictions, config["task"])
    if messages:
        raise ValueError("Cannot evaluate invalid predictions:\n- " + "\n- ".join(messages))
    samples = load_samples_for_config(config, args.split)
    reference_by_id = {sample["sample_id"]: sample for sample in samples}
    prediction_ids = {record["sample_id"] for record in predictions}
    if prediction_ids != set(reference_by_id):
        raise ValueError("Prediction IDs do not exactly match the configured reference samples")
    prediction_records = [record for record in predictions if record["sample_id"] in reference_by_id]
    references = [reference_by_id[record["sample_id"]] for record in prediction_records]
    pipeline = build_task_pipeline(config)
    metrics = pipeline.evaluate(prediction_records, references)
    output = Path(args.output) if args.output else ROOT / config["saving"]["output_dir"] / "metrics.json"
    write_json(output, {"experiment_id": config["experiment"]["id"], "task": config["task"], "metrics": metrics})
    print(f"Evaluated {len(prediction_records)} predictions; metrics written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
