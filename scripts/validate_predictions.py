"""Validate the JSON shape before sharing or submitting predictions."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_grading.validation.predictions import validate_predictions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=("task1", "task2", "task3"))
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
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
