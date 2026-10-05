"""Feedback evaluation must not reward labels copied from its own input."""

import hashlib

import pytest

from llm_grading.evaluation.task3 import evaluate_task3


def test_copied_labels_are_not_a_diagnosis_score():
    output = {
        "feedback": "Tốt.",
        "feedback_level": 2,
        "diagnosed_labels": ["Lỗi logic"],
        "compliance": {"pass": True},
    }
    result = evaluate_task3(
        [{"output": output}], [{"error_labels": ["Lỗi logic"], "feedback_level": 2}]
    )
    assert result["diagnosis_correct_rate"] is None
    assert result["semantic_review_count"] == 0


def test_compliance_is_recomputed_and_judgment_is_independent():
    feedback = "Chỉ cần thay điều kiện bằng i < n."
    output = {
        "feedback": feedback,
        "feedback_level": 2,
        "compliance": {"pass": True},
        "judgment": {
            "judge": "synthetic-review-v1",
            "feedback_sha256": hashlib.sha256(feedback.encode("utf-8")).hexdigest(),
            "feedback_level": 2,
            "diagnosis_correct": False,
            "level_compliant": False,
        },
    }
    result = evaluate_task3([{"output": output}], [{"feedback_level": 2}])
    assert result["compliance_pass_rate"] == 0
    assert result["diagnosis_correct_rate"] == 0
    assert result["semantic_level_compliance_rate"] == 0
    output["feedback"] = "Changed response."
    with pytest.raises(ValueError, match="does not match"):
        evaluate_task3([{"output": output}], [{"feedback_level": 2}])
