"""P0/P1/F0 or heuristic predictions; write only a complete validated file."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.pipeline import build_task_pipeline
from llm_grading.runtime import (
    attach_saved_evidence,
    attach_task3_error_labels,
    load_samples_for_config,
    write_json,
)
from llm_grading.utils.config import load_config
from llm_grading.utils.logging import run_metadata, save_run_config
from llm_grading.utils.seed import set_seed
from llm_grading.validation.predictions import validate_prediction_records


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
    parser.add_argument("--resume", action="store_true", help="Save each completed sample and resume only matching config/data/prompts")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--adapter-path")
    parser.add_argument(
        "--shard",
        help="K/N: predict only every N-th sample starting at K (1-based), e.g. 2/3; merge with scripts/merge_predictions.py",
    )
    parser.add_argument(
        "--task3-labels",
        help="Task 2 prediction file supplying labels when Task 3 input omits taxonomy_error",
    )
    parser.add_argument(
        "--evidence", help="scripts/compile_evidence.py output for these samples (fills compile_log)"
    )
    parser.add_argument(
        "--tests", help="scripts/run_tests.py output for these samples (fills test_report)"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if args.adapter_path:
        config["model"]["adapter_path"] = args.adapter_path
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
    if args.shard:
        index, count = (int(x) for x in args.shard.split("/"))
        if not 1 <= index <= count:
            raise ValueError("--shard must be K/N with 1 <= K <= N")
        samples = samples[index - 1 :: count]
    attach_saved_evidence(samples, args.evidence, args.tests)
    task3_labels = None
    if config["task"] == "task3":
        task3_labels = args.task3_labels or config.get("task3", {}).get(
            "error_labels_path"
        )
        if not config.get("task3", {}).get("two_stage", False):
            attach_task3_error_labels(samples, task3_labels)
    metadata = {
        **run_metadata(config),
        "status": "running",
        "sample_count": len(samples),
        "split": args.split,
        "smoke_subset": bool(args.max_samples),
        "shard": args.shard,
        "task3_labels": str(task3_labels) if task3_labels else None,
        "evidence": args.evidence,
        "tests": args.tests,
    }
    start = time.perf_counter()
    output = Path(args.output) if args.output else directory / "predictions.json"
    checkpoint = output.with_suffix(".partial.json")
    signature = hashlib.sha256(json.dumps(
        {"config": config, "samples": samples,
         "prompts": [p.read_text(encoding="utf-8") for p in sorted((ROOT / "tasks").glob("*/prompts/*.txt"))]},
        sort_keys=True, ensure_ascii=False,
    ).encode()).hexdigest()
    saved = []
    if args.resume and checkpoint.exists():
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if state.get("signature") != signature:
            raise ValueError("Resume config/data/prompts changed; use a new --output path or archive the old partial checkpoint")
        saved = state["predictions"]
        if ([r["sample_id"] for r in saved] != [s["sample_id"] for s in samples[:len(saved)]]
                or validate_prediction_records(saved, config["task"])):
            raise ValueError("Invalid resume checkpoint")
        print(f"Resume: {len(saved)}/{len(samples)} completed samples", flush=True)
    write_json(directory / "prediction_manifest.json", metadata)
    try:
        pipeline = build_task_pipeline(config) if len(saved) < len(samples) else None
        predictions = list(saved)
        resumed_count = len(saved)
        inference_start = time.perf_counter()
        for index, s in enumerate(samples, 1):
            if index <= resumed_count:
                continue
            print(f"[{index}/{len(samples)}] Processing {s['sample_id']} ...", flush=True)
            predictions.append({
                "sample_id": s["sample_id"],
                "output": pipeline.postprocess(pipeline.predict(s)),
            })
            errors = validate_prediction_records([predictions[-1]], config["task"])
            if errors:
                raise ValueError(str(errors))
            if args.resume:
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                temporary = checkpoint.with_suffix(".tmp")
                write_json(temporary, {"signature": signature, "predictions": predictions})
                temporary.replace(checkpoint)
            elapsed = time.perf_counter() - inference_start
            remaining = elapsed / (index - resumed_count) * (len(samples) - index)
            marker = " | FALLBACK (review diagnosis quality)" if predictions[-1]["output"].get("fallback") else ""
            print(
                f"[{index}/{len(samples)}] Done | elapsed {elapsed / 60:.1f} min"
                f" | ETA {remaining / 60:.1f} min{marker}", flush=True,
            )
        errors = validate_prediction_records(predictions, config["task"])
        if errors:
            raise ValueError(str(errors))
        write_json(output, predictions)
        metadata.update(
            status="completed",
            elapsed_seconds=time.perf_counter() - start,
            predictions=str(output),
        )
        if hasattr(pipeline, "runner"):
            metadata["model"] = pipeline.runner.metadata
    except Exception as exc:
        metadata.update(status="failed", error=str(exc))
        raise
    finally:
        write_json(directory / "prediction_manifest.json", metadata)
    print(f"Wrote {len(predictions)} predictions: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
