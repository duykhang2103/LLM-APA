"""Actual script contracts on private temporary synthetic fixtures."""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from test_experiment_contracts import sample

from llm_grading.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]


def run(*args):
    result = subprocess.run(
        [sys.executable, *map(str, args)], cwd=ROOT, text=True, capture_output=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_prepare_predict_evaluate_package(tmp_path):
    rows = [sample(str(i), f"int f{i}() {{ return {i}; }}") for i in range(8)]
    source = tmp_path / "data.json"
    source.write_text(json.dumps(rows))
    cfg = yaml.safe_load((ROOT / "configs/task1/zero_shot.yaml").read_text())
    cfg["data"].update(input_path=str(source), prepared_dir=str(tmp_path / "prepared"))
    cfg["saving"]["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    run("scripts/prepare_data.py", "--config", path, "--training")
    prepared = tmp_path / "prepared" / "config-task1-P0-v1.json"
    first = json.loads((tmp_path / "prepared" / "split_manifest.json").read_text())
    run("scripts/prepare_data.py", "--config", path, "--training")
    assert (
        json.loads((tmp_path / "prepared" / "split_manifest.json").read_text()) == first
    )
    # Use B0 for CPU script end-to-end; real runner/Trainer exercised separately.
    materialized = json.loads(prepared.read_text())
    materialized["method"] = "heuristic"
    prepared.write_text(json.dumps(materialized))
    output = tmp_path / "predictions.json"
    run(
        "scripts/predict.py", "--config", prepared, "--split", "val", "--output", output
    )
    run(
        "scripts/evaluate.py",
        "--config",
        prepared,
        "--split",
        "val",
        "--predictions",
        output,
    )
    run("scripts/validate_predictions.py", "--task", "task1", "--input", output)
    run(
        "scripts/package_predictions.py",
        "--task",
        "task1",
        "--input",
        output,
        "--output",
        tmp_path / "submission.zip",
    )
    assert (tmp_path / "submission.zip").exists()
    import zipfile

    with zipfile.ZipFile(tmp_path / "submission.zip") as archive:
        challenge = json.loads(archive.read("predictions.json"))
    assert set(challenge[0]["output"]) == {"rubric", "total_score"}
    exported = tmp_path / "challenge.json"
    run(
        "scripts/package_predictions.py",
        "--task",
        "task1",
        "--input",
        output,
        "--output",
        exported,
    )
    run(
        "scripts/validate_predictions.py",
        "--task",
        "task1",
        "--input",
        exported,
        "--format",
        "challenge",
    )


def test_config_validation_and_notebooks(tmp_path):
    for p in (ROOT / "configs").glob("task*/*.yaml"):
        load_config(p)
    cfg = yaml.safe_load((ROOT / "configs/task1/qwen_lora.yaml").read_text())
    cfg["training"]["max_sequence_length"] = 0
    file = tmp_path / "bad.json"
    file.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_config(file)
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        for cell in json.loads(path.read_text())["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), str(path), "exec")
