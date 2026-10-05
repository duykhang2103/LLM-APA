"""Audit data or persist duplicate-grouped training/validation splits."""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.data.quality import audit_dataset
from llm_grading.data.split import check_split_overlap, create_splits
from llm_grading.runtime import write_json
from llm_grading.utils.config import load_config
from llm_grading.utils.logging import sample_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    data = config["data"]
    task = config["task"]
    source = args.input or data.get("input_path")
    if not args.training:
        if not source:
            raise ValueError("Set data.input_path or --input")
        samples = load_samples(source, task=task)
        write_json(
            args.output or Path(config["saving"]["output_dir"]) / "prepare_report.json",
            {
                "task": task,
                "sample_count": len(samples),
                "quality": audit_dataset(samples, task),
                "samples": [sample_metadata(s) for s in samples],
            },
        )
        return 0
    directory = Path(
        args.output or data.get("prepared_dir", f"data/processed/{task}")
    ).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    official = bool(data.get("train_path") or data.get("validation_path"))
    if official:
        if not data.get("train_path") or not data.get("validation_path"):
            raise ValueError(
                "Official train and validation paths must both be provided; no resplitting official data"
            )
        train = load_samples(data["train_path"], task=task)
        val = load_samples(data["validation_path"], task=task)
        test = (
            load_samples(data["test_path"], task=task) if data.get("test_path") else []
        )
        check_split_overlap(train, val)
        if test:
            check_split_overlap(train + val, test)
        samples = train + val + test
        ids = {
            "train": [s["sample_id"] for s in train],
            "val": [s["sample_id"] for s in val],
            "test": [s["sample_id"] for s in test],
        }
    else:
        if not source:
            raise ValueError("Set data.input_path or --input")
        samples = load_samples(source, task=task)
        ids = create_splits(
            samples,
            config.get("seed", 42),
            data.get("validation_fraction", 0.2),
            data.get("test_fraction", 0),
        )
    if not ids["train"] or not ids["val"]:
        raise ValueError(
            "Need at least two distinct code groups for nonempty train/validation data"
        )
    by_id = {s["sample_id"]: s for s in samples}
    if len(by_id) != len(samples):
        raise ValueError("Duplicate IDs across official splits")
    manifest = {
        "task": task,
        "seed": config.get("seed", 42),
        "dataset_version": data.get("dataset_version"),
        "split_version": data.get("split_version"),
        "official_paths_used": official,
        "split_ids": ids,
        "validation_fraction": data.get("validation_fraction", 0.2),
        "test_fraction": data.get("test_fraction", 0),
        "samples_sha256": hashlib.sha256(
            json.dumps(samples, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest(),
    }
    manifest_path = directory / "split_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError(
            "Saved split/source differs. Choose a new prepared_dir and split_version; do not silently regenerate"
        )
    for split, key in [
        ("train", "train_path"),
        ("val", "validation_path"),
        ("test", "test_path"),
    ]:
        path = directory / f"{split}.json"
        write_json(path, [by_id[sid] for sid in ids[split]])
        data[key] = str(path)
    write_json(manifest_path, manifest)
    config_file = directory / (
        "config-"
        + re.sub(r"[^A-Za-z0-9_.-]", "_", config["experiment"]["id"])
        + ".json"
    )
    write_json(config_file, config)
    write_json(
        directory / "audit.json",
        {
            "samples": [sample_metadata(s) for s in samples],
            "counts": {k: len(v) for k, v in ids.items()},
            "quality_by_split": {
                split: audit_dataset([by_id[sid] for sid in split_ids], task)
                for split, split_ids in ids.items()
            },
        },
    )
    print(f"Prepared splits: {directory}; use --config {config_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
