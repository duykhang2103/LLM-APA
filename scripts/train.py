"""Create a reproducible run manifest for the local baseline.

The heuristic method intentionally does not pretend to train. The Qwen3.5-4B
LoRA config is recorded, but its model adapter is an explicit future extension.
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    build_task_pipeline(config)
    samples = load_samples_for_config(config, "train")
    output = ROOT / config["saving"]["output_dir"] / "run_manifest.json"
    write_json(output, {
        "experiment": config["experiment"],
        "method": config["method"],
        "model": config["model"],
        "sample_count": len(samples),
        "status": "heuristic_baseline_ready",
        "note": "No model weights were trained by the deterministic baseline.",
    })
    print(f"Prepared heuristic run manifest for {config['task']} at {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
