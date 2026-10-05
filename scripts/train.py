"""Train QLoRA or resume a Trainer checkpoint; heuristic runs only write a manifest."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.runtime import load_samples_for_config, write_json
from llm_grading.utils.config import load_config
from llm_grading.utils.logging import run_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume-from-checkpoint")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Tokenize and check split/budget without optimization; loads the configured model",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    try:
        if config["method"] == "heuristic":
            samples = load_samples_for_config(config, "train")
            write_json(
                Path(config["saving"]["output_dir"]) / "run_manifest.json",
                {
                    "status": "heuristic_baseline_ready",
                    "sample_count": len(samples),
                    "note": "No weights trained",
                },
            )
        elif config["method"] in {"lora", "qlora"}:
            from llm_grading.training.trainer import train

            train(
                config,
                load_samples_for_config(config, "train"),
                load_samples_for_config(config, "val"),
                resume=args.resume_from_checkpoint,
                smoke=args.smoke,
                dry_run=args.dry_run,
            )
        else:
            raise ValueError(
                "train.py requires method: lora/qlora (or heuristic manifest)"
            )
    except Exception as exc:
        failed = {**run_metadata(config), "status": "failed", "error": str(exc)}
        directory = Path(config["saving"]["output_dir"])
        write_json(directory / "last_training_error.json", failed)
        if not (directory / "run_manifest.json").exists():
            write_json(directory / "run_manifest.json", failed)
        raise
    print(f"Run artifacts: {config['saving']['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
