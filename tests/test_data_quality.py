"""Annotation policies preserve source targets and validation membership."""

import json
from copy import deepcopy

import pytest
from test_experiment_contracts import RUBRIC, sample

from llm_grading.data.quality import (
    audit_dataset,
    audit_sample,
    select_training_samples,
)


def test_audit_detects_compiler_test_and_weight_conflicts():
    row = sample()
    row.update(
        compile_log="undefined reference to main",
        test_report=[{"passed": False}],
        problem_statement="Problem 1 (1 điểm).",
        problems=[{"pid": "P1", "max_score": 2.5}],
    )
    assert set(audit_sample(row, "task1")) == {
        "compiler_rubric_conflict",
        "test_rubric_conflict",
        "exam_weight_conflict",
    }


def test_default_excludes_bad_feedback_but_preserves_originals():
    good = sample("good")
    good["feedback"] = "Vòng lặp thiếu cập nhật con trỏ."
    bad = sample("bad", "int different() {}")
    bad["feedback"] = "Chỉ cần thêm cur = cur->next; vào vòng lặp."
    original = deepcopy([good, bad])
    kept, audit = select_training_samples([good, bad], "task3", {})
    assert [r["sample_id"] for r in kept] == ["good"]
    assert audit["excluded"][0]["sample_id"] == "bad"
    assert [good, bad] == original


def test_conflicting_teacher_score_requires_explicit_exclusion():
    row = sample()
    row["compile_log"] = "error: bad code"
    kept, audit = select_training_samples([row], "task1", {})
    assert kept[0]["rubric"] == RUBRIC
    assert audit["samples"][0]["flags"] == ["compiler_rubric_conflict"]
    cfg = {"data": {"quality": {"exclude_flags": ["compiler_rubric_conflict"]}}}
    kept, _ = select_training_samples([row], "task1", cfg)
    assert kept == []


def test_review_decision_needs_reason_and_cannot_change_labels(tmp_path):
    review = tmp_path / "review.json"
    review.write_text(
        json.dumps({"s1": {"action": "exclude", "reason": "source mismatch"}})
    )
    cfg = {"data": {"quality": {"review_file": str(review)}}}
    kept, audit = select_training_samples([sample()], "task1", cfg)
    assert kept == [] and audit["excluded"][0]["reason"] == "source mismatch"
    review.write_text(json.dumps({"s1": {"action": "keep", "reason": ""}}))
    with pytest.raises(ValueError, match="reason"):
        select_training_samples([sample()], "task1", cfg)
    review.write_text(
        json.dumps({"typo": {"action": "exclude", "reason": "source mismatch"}})
    )
    with pytest.raises(ValueError, match="unknown training sample IDs"):
        select_training_samples([sample()], "task1", cfg)


def test_missing_support_is_reported_and_can_block_training():
    report = audit_dataset([sample()], "task2")
    assert len(report["missing_labels"]) == 10
    cfg = {"data": {"quality": {"require_coverage": True}}}
    with pytest.raises(ValueError, match="coverage"):
        select_training_samples([sample()], "task2", cfg)


def test_validation_target_is_not_filtered():
    from llm_grading.training.dataset import build_training_dataset

    row = sample()
    row["feedback"] = "Chỉ cần lấy abs(năm mất - năm sinh)."
    assert build_training_dataset([row], "task3") == []
    assert (
        build_training_dataset([row], "task3", training=False)[0]["target"]
        == row["feedback"]
    )
