"""Synthetic regressions for the Kaggle diagnosis and level controls."""

import ast
import copy
import json
from pathlib import Path

import pytest
from test_experiment_contracts import sample
from test_prompting_runtime import Runner, config

from llm_grading.prompting.pipeline import ModelPipeline
from llm_grading.utils.config import load_config
from tasks.task3_feedback.compliance import check_compliance, has_correction_direction

ROOT = Path(__file__).resolve().parents[1]


def diagnosis(labels, text):
    return json.dumps({"error_labels": labels, "diagnosis": text})


def test_review_replaces_false_labels_and_sees_full_source(tmp_path):
    cfg = config(tmp_path, "task3")
    cfg["task3"] = {
        "two_stage": True, "compact_diagnosis_prompt": True,
        "diagnosis_prompt_version": "v003", "verify_diagnosis": True,
        "compact_feedback_input": True,
    }
    cfg["prompt"]["version"] = "v007"
    prompts = []

    class Capture(Runner):
        def generate(self, prompt):
            prompts.append(prompt)
            return super().generate(prompt)

    row = sample()
    row.update(code="ORIGINAL_SOURCE", problem_statement="VALID_INPUT_CONSTRAINTS")
    runner = Capture([
        diagnosis(["Lỗi logic", "Lỗi edge case"], "DRAFT_FINDING"),
        diagnosis(["Lỗi logic"], "REVIEWED_FINDING"),
        "Trong hàm tính tổng, giá trị tích khiến kết quả không đúng yêu cầu.",
    ])
    out = ModelPipeline(cfg, runner).predict(row)
    assert out["error_labels"] == ["Lỗi logic"]
    assert "ORIGINAL_SOURCE" in prompts[1] and "VALID_INPUT_CONSTRAINTS" in prompts[1]
    assert "DRAFT_FINDING" in prompts[1]
    assert "REVIEWED_FINDING" in prompts[2] and "DRAFT_FINDING" not in prompts[2]
    assert "SECRET TEACHER TARGET" not in "\n".join(prompts)
    logs = [json.loads(x) for x in (tmp_path / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [x.get("stage") for x in logs[:2]] == ["diagnosis", "diagnosis_review"]


@pytest.mark.parametrize("labels", [[], ["Lỗi logic"]])
def test_level_one_projects_reviewed_labels_without_specific_diagnosis(tmp_path, labels):
    cfg = config(tmp_path, "task3")
    cfg["task3"] = {"two_stage": True, "verify_diagnosis": True, "level1_summary": True}
    row = sample()
    row["feedback_level"] = 1
    # Only two generation calls; no third chance to leak locations or fixes.
    runner = Runner([
        diagnosis(["Lỗi logic"], "SECRET_LOCATION"),
        diagnosis(labels, "SECRET_LOCATION"),
    ])
    out = ModelPipeline(cfg, runner).predict(row)
    assert out["error_labels"] == labels
    assert "SECRET_LOCATION" not in out["feedback"]
    assert "đã đáp ứng" not in out["feedback"]
    assert out["compliance"]["pass"]
    assert not out.get("fallback")


def test_failed_review_does_not_silently_use_unverified_draft(tmp_path):
    cfg = config(tmp_path, "task3")
    cfg["task3"] = {"two_stage": True, "verify_diagnosis": True, "fallback_on_failure": False}
    with pytest.raises(ValueError, match="Invalid diagnosis"):
        ModelPipeline(cfg, Runner([
            diagnosis(["Lỗi logic"], "A draft"), "bad JSON", "bad JSON",
        ])).predict(sample())


def test_level_three_requires_fix_direction_and_retries(tmp_path):
    cfg = config(tmp_path, "task3")
    cfg["task3"] = {"require_correction_direction": True}
    row = sample()
    row.update(feedback_level=3, error_labels=["Lỗi logic"])
    out = ModelPipeline(cfg, Runner([
        "Trong hàm tính tổng, biểu thức đang tính tích nên kết quả sai.",
        "Trong hàm tính tổng, biểu thức đang tính tích nên kết quả sai. Em nên dùng phép cộng để gộp hai giá trị.",
    ])).predict(row)
    assert "Em nên" in out["feedback"]
    assert not has_correction_direction("Biến chưa khởi tạo nên kết quả sai.")
    assert not has_correction_direction("Em chưa khởi tạo biến khiến kết quả sai.")


@pytest.mark.parametrize("text", [
    "Bài làm có một số vấn đề về logic trong phần tạo login.",
    "Bài làm sai ở trường hợp biên.",
    "Trong hàm compute, điều kiện dừng không đúng.",
])
def test_level_one_catches_specific_causes(text):
    assert not check_compliance(text, 1)["pass"]


def test_notebook_quality_path_and_clean_cells():
    notebook = json.loads((ROOT / "notebooks/run_task3_kaggle_v3.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "recover_task3_diagnosis.py" not in source
    assert "validation_report.json" in source
    assert "submission_task3_qwen3_4b_review1.json" in source
    assert "exclude_same_exam'] = True" in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["outputs"] == [] and cell["execution_count"] is None
    cfg = load_config(ROOT / "configs/task3/kaggle_v3.yaml")
    assert cfg["task3"]["verify_diagnosis"]
    assert not cfg["task3"]["fallback_on_failure"]
    assert cfg["prompt"]["version"] == "v007"


def test_validation_queries_remove_targets_and_keep_code(tmp_path):
    notebook = json.loads((ROOT / "notebooks/run_task3_kaggle_v3.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][9]["source"])
    setup = source.split("subprocess.run(", 1)[0]
    rows = []
    for i in range(12):
        row = sample(f"train-{i}", f"SOURCE_{i}")
        row.update(feedback_level=i % 3 + 1, _has_reference=True)
        rows.append(row)
    cfg = load_config(ROOT / "configs/task3/kaggle_v3.yaml")
    namespace = {"train": rows, "OUT": tmp_path, "cfg": cfg, "copy": copy, "json": json}
    exec(compile(setup, "validation-cell", "exec"), namespace)
    queries = json.loads((tmp_path / "validation-input.json").read_text(encoding="utf-8"))
    assert len(queries) == 8
    assert {q["feedback_level"] for q in queries} == {1, 2, 3}
    for query in queries:
        assert query["code"].startswith("SOURCE_") and query["error_labels_missing"]
        assert not set(query) & {"feedback", "error_labels", "rubric", "total_score", "_has_reference"}
    val_cfg = json.loads((tmp_path / "validation-config.json").read_text(encoding="utf-8"))
    assert val_cfg["retrieval"]["exclude_same_exam"] is True
