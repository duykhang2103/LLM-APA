"""One command checks data, inference, optional adapter training/reload, and metrics."""

import argparse
import copy
import gc
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from evaluate import EVALUATORS

from llm_grading.pipeline import build_task_pipeline
from llm_grading.prompting.formatter import render_task_prompt
from llm_grading.runtime import load_samples_for_config, write_json
from llm_grading.training.trainer import train
from llm_grading.utils.config import load_config
from llm_grading.utils.logging import save_run_config
from llm_grading.utils.seed import set_seed
from llm_grading.validation.predictions import validate_prediction_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--training", action="store_true")
    args = p.parse_args()
    cfg = copy.deepcopy(load_config(args.config))
    set_seed(cfg.get("seed", 42))
    cfg["saving"]["output_dir"] = str(Path(cfg["saving"]["output_dir"]) / "smoke")
    cfg["experiment"]["id"] += "-smoke"
    directory = save_run_config(cfg)
    write_json(directory / "smoke_status.json", {"status": "running"})
    try:
        samples = load_samples_for_config(cfg, "val")
        if not samples:
            raise ValueError("Validation split is empty")
        render_task_prompt(samples[0], cfg["task"], cfg)
        if args.training:
            cfg["training"].update(gradient_accumulation_steps=1, max_steps=2)
            cfg["model"]["adapter_path"] = None
            manifest = train(
                cfg, load_samples_for_config(cfg, "train"), samples, smoke=True
            )
            cfg["model"]["revision"] = manifest["model_revision"]
            cfg["model"]["adapter_path"] = manifest["adapter_path"]
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        pipeline = build_task_pipeline(cfg)
        prediction = {
            "sample_id": samples[0]["sample_id"],
            "output": pipeline.postprocess(pipeline.predict(samples[0])),
        }
        errors = validate_prediction_records([prediction], cfg["task"])
        if errors:
            raise ValueError(str(errors))
        metrics = EVALUATORS[cfg["task"]]([prediction], [samples[0]])
        write_json(directory / "predictions.json", [prediction])
        write_json(directory / "metrics.json", metrics)
        write_json(
            directory / "smoke_status.json",
            {
                "status": "passed",
                "training_executed": args.training,
                "checks": ["dataset", "prompt", "inference", "schema", "evaluator"]
                + (
                    [
                        "tokenizer",
                        "model",
                        "finite loss",
                        "LoRA update",
                        "checkpoint save",
                        "adapter reload",
                    ]
                    if args.training
                    else []
                ),
            },
        )
        print(f"Smoke passed: {directory}")
    except Exception as exc:
        write_json(
            directory / "smoke_status.json", {"status": "failed", "error": str(exc)}
        )
        raise


if __name__ == "__main__":
    main()
