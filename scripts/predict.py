"""Generate and validate predictions with one configured task pipeline.

Example:
    uv run python scripts/predict.py --config configs/task1/heuristic.yaml
"""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_grading.pipeline import build_task_pipeline
from llm_grading.runtime import load_samples_for_config, write_json
from llm_grading.utils.config import load_config
from llm_grading.validation.predictions import validate_prediction_records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test", choices=("train", "val", "validation", "test"))
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config)
    pipeline = build_task_pipeline(config)
    samples = load_samples_for_config(config, args.split)
    predictions = []
    for sample in samples:
        raw_output = pipeline.predict(sample)
        output = pipeline.postprocess(raw_output)
        predictions.append({"sample_id": sample["sample_id"], "output": output})

    messages = validate_prediction_records(predictions, config["task"])
    if messages:
        raise ValueError("Generated predictions are invalid:\n- " + "\n- ".join(messages))
    default_output = Path(config["saving"]["output_dir"]) / "predictions.json"
    output_path = Path(args.output) if args.output else ROOT / default_output
    write_json(output_path, predictions)
    print(f"Wrote {len(predictions)} {config['task']} predictions to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
