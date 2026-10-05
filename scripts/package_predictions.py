"""Validate and package the existing contract without inspecting private targets."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.challenge import export_challenge_predictions
from llm_grading.validation.predictions import validate_predictions


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["challenge", "internal"], default="challenge")
    args = p.parse_args()
    errors = validate_predictions(args.input, args.task)
    if errors:
        raise ValueError(str(errors))
    path = Path(args.output)
    if path.resolve() == Path(args.input).resolve():
        raise ValueError("Export to a separate path to preserve internal predictions")
    records = read_json(args.input)
    if args.format == "challenge":
        records = export_challenge_predictions(records, args.task)
    if path.suffix == ".json":
        write_json(path, records)
        print(f"Exported validated predictions: {path}")
        return 0
    if path.suffix != ".zip":
        raise ValueError("Output must end in .json or .zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "predictions.json", json.dumps(records, ensure_ascii=False, indent=2)
        )
    print(f"Packaged validated predictions: {path}")


if __name__ == "__main__":
    main()
