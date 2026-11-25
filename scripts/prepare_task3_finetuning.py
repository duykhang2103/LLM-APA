"""Prepare a labeled Task 3 pilot and requested-level test inputs without a GPU."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.task3_finetuning import prepare_task3_finetuning
from llm_grading.utils.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--labeled-input", required=True)
    parser.add_argument("--test-input", required=True)
    parser.add_argument("--requested-levels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    prepare_task3_finetuning(
        load_config(args.config), args.labeled_input, args.test_input,
        args.requested_levels, args.output,
    )
    directory = Path(args.output)
    manifest = json.loads((directory / "data_manifest.json").read_text(encoding="utf-8"))
    print(json.dumps(manifest["counts"], indent=2))
    print(f"Prepared configuration: {directory / 'config.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
