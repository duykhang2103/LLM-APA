"""Validate the JSON shape before sharing or submitting predictions."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_grading.runtime import read_json
from llm_grading.validation.challenge import validate_challenge_predictions
from llm_grading.validation.predictions import validate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    parser.add_argument("--input", required=True)
    parser.add_argument(
        "--format", choices=("internal", "challenge"), default="internal"
    )
    args = parser.parse_args()
    if args.format == "challenge":
        try:
            messages = validate_challenge_predictions(read_json(args.input), args.task)
        except (OSError, ValueError) as error:
            messages = [str(error)]
    else:
        messages = validate_predictions(args.input, args.task)
    if messages:
        print("Prediction validation failed:")
        for message in messages:
            print(f"- {message}")
        return 1
    print(f"Prediction file is valid for {args.task}: {args.input}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
