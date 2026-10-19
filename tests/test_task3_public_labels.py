"""Public-test Task 3 records omit input labels; Task 2 predictions fill them."""

import json

import pytest

from llm_grading.data.loader import load_samples
from llm_grading.data.preprocess import build_task3_input
from llm_grading.runtime import attach_task3_error_labels


def write_dataset(root, with_labels):
    (root / "submissions").mkdir()
    (root / "submissions" / "S1.cpp").write_text("int main() {}", encoding="utf-8")
    exam = {"exam_id": "E1", "exam_type": "single_problem", "statement": "Đề bài"}
    (root / "exams.json").write_text(json.dumps({"exams": [exam]}), encoding="utf-8")
    inputs = {
        "exam_id": "E1",
        "exam_type": "single_problem",
        "code_file": "submissions/S1.cpp",
        "target_feedback_level": "Level 2 - Chẩn đoán",
    }
    if with_labels:
        inputs["taxonomy_error"] = ["Lỗi logic"]
    payload = {"task": "task3_feedback", "samples": [{"sample_id": "S1", "input": inputs}]}
    (root / "task3_feedback.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def test_missing_input_labels_are_flagged_not_empty(tmp_path):
    write_dataset(tmp_path, with_labels=False)
    sample = load_samples(tmp_path, task="task3")[0]
    assert sample["error_labels_missing"] is True
    with pytest.raises(ValueError, match="attach Task 2 predictions"):
        build_task3_input(sample)
    with pytest.raises(ValueError, match="--task3-labels"):
        attach_task3_error_labels([sample], None)


@pytest.mark.parametrize("key", ["error_labels", "taxonomy_error"])
def test_task2_predictions_fill_missing_labels(tmp_path, key):
    write_dataset(tmp_path, with_labels=False)
    samples = load_samples(tmp_path, task="task3")
    predictions = tmp_path / "task2.json"
    predictions.write_text(
        json.dumps([{"sample_id": "S1", "output": {key: ["Lỗi vòng lặp"]}}], ensure_ascii=False),
        encoding="utf-8",
    )
    attach_task3_error_labels(samples, predictions)
    assert build_task3_input(samples[0])["error_labels"] == ["Lỗi vòng lặp"]
    assert "error_labels_missing" not in samples[0]


def test_task2_predictions_must_cover_and_use_official_labels(tmp_path):
    write_dataset(tmp_path, with_labels=False)
    predictions = tmp_path / "task2.json"
    for rows in (
        [{"sample_id": "other", "output": {"error_labels": []}}],
        [{"sample_id": "S1", "output": {"error_labels": ["not official"]}}],
    ):
        predictions.write_text(json.dumps(rows), encoding="utf-8")
        with pytest.raises(ValueError):
            attach_task3_error_labels(load_samples(tmp_path, task="task3"), predictions)


def test_given_input_labels_are_kept(tmp_path):
    write_dataset(tmp_path, with_labels=True)
    samples = load_samples(tmp_path, task="task3")
    attach_task3_error_labels(samples, None)
    assert build_task3_input(samples[0])["error_labels"] == ["Lỗi logic"]
