"""Private local JSON artifacts; no tracking service."""

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from pathlib import Path

from llm_grading.data.split import code_hash
from llm_grading.prompting.formatter import prompt_hash
from llm_grading.runtime import write_json


def run_metadata(config):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    packages = {}
    for name in [
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
        "sentence-transformers",
    ]:
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    hardware = {}
    try:
        import torch

        hardware = {
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "vram_bytes": torch.cuda.get_device_properties(0).total_memory
            if torch.cuda.is_available()
            else None,
        }
    except ImportError:
        pass
    return {
        "hardware": hardware,
        "git_commit": commit,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": packages,
        "experiment": config["experiment"],
        "dataset_version": config["data"].get("dataset_version"),
        "split_version": config["data"].get("split_version"),
        "seed": config.get("seed", 42),
        "prompt_version": config.get("prompt", {}).get("version", "v001"),
        "prompt_sha256": prompt_hash(config["task"], config),
        "notes": config["experiment"].get("notes", ""),
    }


def sample_metadata(sample):
    log = str(sample.get("compile_log") or "").lower()
    # A flag for later review, not a correction to teacher annotations.
    conflict = sample.get("rubric", {}).get("compilable") == 1 and any(
        x in log for x in ["error", "fail"]
    )
    return {
        "sample_id": sample["sample_id"],
        "exam_id": sample.get("problem_id"),
        "problem_type": sample.get("problem_type"),
        "feedback_level": sample.get("feedback_level"),
        "code_hash": code_hash(sample.get("code", "")),
        "evidence_conflict_flag": conflict,
        "review_flags": sample.get("review_flags", []),
    }


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")


def save_run_config(config):
    directory = Path(config["saving"]["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    # API credentials must be environment variables, never config literals.
    if any(k in config.get("model", {}) for k in ["api_key", "token"]):
        raise ValueError("Use model.api_key_env; do not put API credentials in YAML")
    write_json(directory / "resolved_config.json", config)
    return directory


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
