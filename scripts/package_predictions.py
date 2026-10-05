"""Validate and package the existing contract without inspecting private targets."""

import argparse
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.validation.predictions import validate_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    errors = validate_predictions(args.input, args.task)
    if errors:
        raise ValueError(str(errors))
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(args.input, "predictions.json")
    print(f"Packaged validated predictions: {path}")


if __name__ == "__main__":
    main()
