"""Prepare labeled Task 3 splits and apply inference-only requested levels."""

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from llm_grading.data.loader import load_samples
from llm_grading.data.quality import select_training_samples
from llm_grading.data.split import check_split_overlap, create_splits
from llm_grading.runtime import write_json
from llm_grading.training.dataset import build_training_dataset


def apply_requested_levels(
    samples: list[dict[str, Any]], levels_path: str | Path,
) -> list[dict[str, Any]]:
    """Join by ID, requiring exact coverage; never change teacher training levels."""
    rows = json.loads(Path(levels_path).read_text(encoding="utf-8-sig"))
    if not isinstance(rows, list) or not rows:
        raise ValueError("Requested levels must be a nonempty list")
    levels = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Each requested-level record must be an object")
        sid, level = row.get("sample_id"), row.get("requested_level")
        if not isinstance(sid, str) or not sid.strip() or sid in levels:
            raise ValueError("Requested levels need unique, nonempty sample IDs")
        if type(level) is not int or level not in {1, 2, 3, 4}:
            raise ValueError(f"Invalid requested_level for {sid}: expected integer 1..4")
        levels[sid] = level
    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Test sample IDs must be unique")
    missing, extra = set(ids) - levels.keys(), levels.keys() - set(ids)
    if missing or extra:
        raise ValueError(
            f"Requested-level ID mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    result = copy.deepcopy(samples)
    for sample in result:
        sample["feedback_level"] = levels[sample["sample_id"]]
    return result


def prepare_task3_finetuning(
    config: dict[str, Any], labeled_source: str | Path, test_source: str | Path,
    levels_path: str | Path, output_dir: str | Path,
) -> dict[str, Any]:
    """Persist duplicate-grouped splits, training audit, test levels, and config.

    All target and overlap checks happen before writing artifacts or loading a
    model. Quality exclusions apply only in training; teacher text is preserved.
    """
    cfg = copy.deepcopy(config)
    if cfg.get("task") != "task3" or cfg.get("task3", {}).get("output_format") != "taxonomy_feedback":
        raise ValueError("Task 3 fine-tuning requires joint taxonomy_feedback output")
    directory = Path(output_dir).resolve()
    sources = [Path(p).resolve() for p in (labeled_source, test_source, levels_path)]
    outputs = [directory / name for name in (
        "train.json", "val.json", "test.json", "data_manifest.json", "quality_audit.json", "config.json",
    )]
    for source in sources:
        if (source.is_dir() and directory.is_relative_to(source)) or source in outputs:
            raise ValueError("Write prepared artifacts outside the source datasets")

    labeled = load_samples(labeled_source, task="task3")
    original_test = load_samples(test_source, task="task3")
    # This rejects unlabeled inputs, missing feedback, and unknown taxonomy targets.
    build_training_dataset(labeled, "task3", cfg, training=False)
    test = apply_requested_levels(original_test, levels_path)
    check_split_overlap(labeled, test)
    ids = create_splits(labeled, cfg.get("seed", 42), cfg["data"].get("validation_fraction", 0.2), 0)
    by_id = {s["sample_id"]: s for s in labeled}
    train = [by_id[sid] for sid in ids["train"]]
    val = [by_id[sid] for sid in ids["val"]]
    if not train or not val:
        raise ValueError("Need enough distinct code groups for nonempty train and validation splits")
    check_split_overlap(train, val)
    kept, audit = select_training_samples(train, "task3", cfg)
    if not kept:
        raise ValueError("No labeled training examples remain after quality exclusions")

    manifest = {
        "task": "task3", "seed": cfg.get("seed", 42),
        "dataset_version": cfg["data"].get("dataset_version"),
        "split_version": cfg["data"].get("split_version"),
        "counts": {"labeled": len(labeled), "train": len(train), "retained_train": len(kept), "val": len(val), "test": len(test)},
        "split_ids": {"train": ids["train"], "val": ids["val"], "test": [s["sample_id"] for s in test]},
        "requested_levels_changed": sum(a["feedback_level"] != b["feedback_level"] for a, b in zip(original_test, test)),
        "requested_level_support": dict(Counter(str(s["feedback_level"]) for s in test)),
        "requested_levels_sha256": hashlib.sha256(Path(levels_path).read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(json.dumps([labeled, test], sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
        "quality_policy": cfg["data"].get("quality", {}),
        "note": "Sample examples support a pilot; held-out examples are development data, not independent acceptance data.",
    }
    manifest_path = directory / "data_manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
        raise ValueError("Prepared data changed; choose a new output directory and split version")
    for name, key, rows in (
        ("train", "train_path", train), ("val", "validation_path", val), ("test", "test_path", test),
    ):
        path = directory / f"{name}.json"
        write_json(path, rows)
        cfg["data"][key] = str(path)
    cfg["data"].update(input_path=None, max_samples=None, prepared_dir=str(directory))
    write_json(manifest_path, manifest)
    write_json(directory / "quality_audit.json", audit)
    write_json(directory / "config.json", cfg)
    return cfg
