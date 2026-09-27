"""Load normalized JSON and write a small data-audit report."""

from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from llm_grading.data.loader import load_samples
from llm_grading.runtime import write_json
from llm_grading.utils.config import load_config
from llm_grading.utils.paths import resolve_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    args = parser.parse_args()
    config = load_config(args.config)
    source = args.input or config["data"].get("input_path")
    if not source:
        raise ValueError("Provide --input or configure data.input_path")
    samples = load_samples(resolve_path(source), config["data"].get("max_samples"))
    report = {
        "experiment_id": config["experiment"]["id"],
        "dataset_version": config["data"].get("dataset_version"),
        "sample_count": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "fields": sorted({key for sample in samples for key in sample}),
    }
    output = Path(args.output) if args.output else ROOT / config["saving"]["output_dir"] / "prepare_report.json"
    write_json(output, report)
    print(f"Loaded {len(samples)} samples; report written to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
