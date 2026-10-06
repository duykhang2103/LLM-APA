"""Exercise notebook source checks and retries without Kaggle or model downloads."""

import contextlib
import io
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def notebook_cell(index):
    notebook = json.loads(
        (ROOT / "notebooks/train_qwen_qlora_kaggle_v3.ipynb").read_text()
    )
    return "".join(notebook["cells"][index]["source"])


@pytest.mark.parametrize(
    "mount", ["", "datasets/thanhbnhh", "datasets/thanhbnhh/versions/1"]
)
def test_repo_discovery_selects_dataset_across_mount_layouts(tmp_path, mount):
    inputs = tmp_path / "input"
    for slug in ("llm-apa", "llm-apa-source"):
        repo = inputs / mount / slug / "LLM-APA-main"
        (repo / "scripts").mkdir(parents=True)
        (repo / "src/llm_grading").mkdir(parents=True)
        (repo / "data/sample_dataset").mkdir(parents=True)
        (repo / "data/sample_dataset/task1_grading.json").write_text("{}")
        for relative in (
            "pyproject.toml",
            "uv.lock",
            "scripts/prepare_data.py",
            "scripts/smoke.py",
            "scripts/train.py",
            "scripts/predict.py",
            "scripts/evaluate.py",
            "scripts/validate_predictions.py",
            "scripts/package_predictions.py",
        ):
            (repo / relative).write_text(slug)
    source = notebook_cell(2)
    source = source.replace('Path("/kaggle/input")', f"Path({str(inputs)!r})")
    source = source.replace(
        'Path("/kaggle/working")', f"Path({str(tmp_path / 'working')!r})"
    )
    source = source.replace(
        "REPO_DATASET_SLUG = None", 'REPO_DATASET_SLUG = "llm-apa-source"'
    )
    namespace = {}
    exec(source, namespace)
    assert namespace["REPO_INPUT"] == inputs / mount / "llm-apa-source/LLM-APA-main"
    assert (namespace["REPO"] / "pyproject.toml").read_text() == "llm-apa-source"
    data_files = namespace["find_attached_files"]("task1_grading.json", "llm-apa")
    assert data_files == [
        inputs / mount / "llm-apa/LLM-APA-main/data/sample_dataset/task1_grading.json"
    ]


def test_source_preflight_rejects_old_trainer(monkeypatch):
    import llm_grading.training.trainer as trainer_module

    def py_eval(source):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exec(source, {})
        return output.getvalue().strip()

    source = notebook_cell(6).split("# Verify the attached source", 1)[1]
    source = "# Verify the attached source" + source
    namespace = {"json": json, "py_eval": py_eval}
    exec(source, namespace)
    assert namespace["source_checks"]["completion_only_logits"]
    monkeypatch.delattr(trainer_module, "SMOKE_TRAINING_OVERRIDES")
    with pytest.raises(RuntimeError, match="Update the Kaggle repo Dataset"):
        exec(source, namespace)


def test_runtime_subprocesses_use_one_gpu(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setattr(
        subprocess, "run", lambda command, **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda *args, **kwargs: json.dumps(
            {
                "completion_only_logits": True,
                "smoke_overrides": {
                    "warmup_steps": 0,
                    "load_best_model_at_end": False,
                    "metric_for_best_model": None,
                },
            }
        ),
    )
    namespace = {
        "WORKING": tmp_path,
        "REPO": Path("/kaggle/working/LLM-APA-main"),
        "os": os,
        "sys": sys,
        "subprocess": subprocess,
        "json": json,
    }
    exec(notebook_cell(6), namespace)
    namespace["run"]("train.py", "--config", "prepared.json")
    assert calls[-1]["env"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert namespace["RUN_ENV"]["CUDA_VISIBLE_DEVICES"] == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


def test_explicit_split_paths_need_no_repo_dataset(tmp_path):
    source = notebook_cell(10)
    source = source.replace("TRAIN_FILE=None", "TRAIN_FILE='/attached/train.json'")
    source = source.replace("VAL_FILE=None", "VAL_FILE='/attached/val.json'")
    namespace = {"REPO": tmp_path / "repo", "WORKING": tmp_path, "Path": Path}
    exec(source, namespace)
    assert not namespace["USING_SAMPLE"]
    assert namespace["DATA_ROOT"] == Path("/attached")


@pytest.mark.parametrize("sample_path", ["sample_dataset", "data/sample_dataset"])
def test_sample_discovery_matches_documented_layout(tmp_path, sample_path):
    sample = tmp_path / "repo" / sample_path
    sample.mkdir(parents=True)
    (sample / "task1_grading.json").write_text("{}")
    namespace = {"REPO": tmp_path / "repo", "WORKING": tmp_path, "Path": Path}
    exec(notebook_cell(10), namespace)
    assert namespace["DATA_ROOT"] == sample
    assert namespace["USING_SAMPLE"]


@pytest.mark.parametrize("change", ["config", "task", "version"])
def test_smoke_gate_rejects_changed_run(tmp_path, change):
    prepared = tmp_path / "prepared.json"
    config = {"saving": {"output_dir": str(tmp_path / "run")}, "training": {}}
    prepared.write_text(json.dumps(config))

    def run(script, *args):
        request = json.loads(Path(args[1]).read_text())
        directory = Path(request["saving"]["output_dir"]) / "smoke"
        directory.mkdir()
        (directory / "smoke_status.json").write_text(json.dumps({"status": "passed"}))

    namespace = {
        "RUNS_ROOT": tmp_path,
        "TASK": "task1",
        "VERSION": "test",
        "PREPARED": prepared,
        "json": json,
        "subprocess": subprocess,
        "run": run,
    }
    exec(notebook_cell(18), namespace)
    if change == "config":
        prepared.write_text(json.dumps({**config, "seed": 999}))
    elif change == "task":
        namespace["TASK"] = "task2"
    else:
        namespace["VERSION"] = "next"
    with pytest.raises(RuntimeError, match="passed smoke"):
        exec(notebook_cell(20), namespace)


def test_smoke_retry_preserves_failed_attempt_and_prepared_config(tmp_path, capsys):
    prepared = tmp_path / "prepared.json"
    config = {
        "saving": {"output_dir": str(tmp_path / "full-training")},
        "training": {
            "resume_from_checkpoint": "old-checkpoint",
            "warm_start_adapter": "old-adapter",
        },
    }
    prepared.write_text(json.dumps(config))
    attempts = []
    training_runs = []

    def run(script, *args):
        if script == "train.py":
            training_runs.append(script)
            (tmp_path / "full-training").mkdir()
            (tmp_path / "full-training" / "run_manifest.json").write_text(
                json.dumps({"status": "completed"})
            )
            return
        assert script == "smoke.py"
        request = json.loads(Path(args[1]).read_text())
        assert request["training"]["resume_from_checkpoint"] is None
        assert request["training"]["warm_start_adapter"] is None
        directory = Path(request["saving"]["output_dir"]) / "smoke"
        directory.mkdir()
        attempts.append(directory)
        if len(attempts) == 1:
            (directory / "smoke_status.json").write_text(
                json.dumps({"status": "failed", "error": "test failure"})
            )
            raise subprocess.CalledProcessError(1, [script])
        (directory / "smoke_status.json").write_text(json.dumps({"status": "passed"}))

    namespace = {
        "RUNS_ROOT": tmp_path,
        "TASK": "task1",
        "VERSION": "test",
        "PREPARED": prepared,
        "json": json,
        "subprocess": subprocess,
        "run": run,
        "resume_path": None,
        "RUN_DIR": tmp_path / "full-training",
    }
    with pytest.raises(subprocess.CalledProcessError):
        exec(notebook_cell(18), namespace)
    assert "test failure" in capsys.readouterr().out
    with pytest.raises(RuntimeError, match="passed smoke"):
        exec(notebook_cell(20), namespace)
    exec(notebook_cell(18), namespace)
    assert attempts[0] != attempts[1]
    assert (
        json.loads((attempts[0] / "smoke_status.json").read_text())["status"]
        == "failed"
    )
    assert namespace["status"]["status"] == "passed"
    assert json.loads(prepared.read_text()) == config
    exec(notebook_cell(20), namespace)
    assert training_runs == ["train.py"]
