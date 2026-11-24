"""Persist P0/P1/F0 predictions and report per-sample warnings and failures."""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.pipeline import build_task_pipeline
from llm_grading.prediction import predict_samples, validate_prediction_output_path
from llm_grading.runtime import load_samples_for_config, write_json
from llm_grading.utils.config import load_config
from llm_grading.utils.logging import run_metadata, save_run_config
from llm_grading.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--split", default="val", choices=["train", "val", "validation", "test"]
    )
    parser.add_argument(
        "--input",
        help="Explicit inference/lecturer file; no targets are read into prompts",
    )
    parser.add_argument("--output")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--adapter-path")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.adapter_path:
        config["model"]["adapter_path"] = args.adapter_path
    directory = Path(config["saving"]["output_dir"])
    output = Path(args.output) if args.output else directory / "predictions.json"
    validate_prediction_output_path(config, output)
    directory = save_run_config(config)
    set_seed(config.get("seed", 42))
    samples = (
        load_samples(args.input, task=config["task"])
        if args.input
        else load_samples_for_config(config, args.split)
    )
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        samples = samples[: args.max_samples]
    metadata = {
        **run_metadata(config),
        "status": "running",
        "sample_count": len(samples),
        "split": args.split,
        "smoke_subset": bool(args.max_samples),
    }
    start = time.perf_counter()
    write_json(directory / "prediction_manifest.json", metadata)
    try:
        pipeline = build_task_pipeline(config)
        metadata.update(predict_samples(pipeline, samples, config, output))
        if hasattr(pipeline, "runner"):
            metadata["model"] = pipeline.runner.metadata
    except KeyboardInterrupt:
        metadata.update(status="interrupted")
        raise
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        metadata["elapsed_seconds"] = time.perf_counter() - start
        write_json(directory / "prediction_manifest.json", metadata)
    print(f"Saved {metadata['prediction_count']}/{len(samples)} predictions: {output}")
    print(f"Warnings: {metadata['warning_count']} | Failed samples: {metadata['failure_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
