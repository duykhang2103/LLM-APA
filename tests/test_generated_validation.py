"""Checkpoint scores come from generated completions, never teacher-forced logits."""

import json

from test_experiment_contracts import RUBRIC, sample

from llm_grading.data.quality import balance_training_samples
from llm_grading.training.validation import evaluate_generated


def test_generated_metrics_penalize_invalid_outputs():
    rows = [sample("a"), sample("b")]
    responses = iter([json.dumps(RUBRIC), "not JSON"])
    result = evaluate_generated(
        {}, rows, {"task": "task1"}, generate=lambda *_: next(responses)
    )
    assert result["metrics"]["valid_count"] == 1
    assert result["metrics"]["invalid_count"] == 1
    assert result["metrics"]["selection_score"] == 0.5
    assert result["predictions"][1]["parse_error"]


def test_task_two_uses_macro_f1_with_fixed_label_space():
    row = sample()
    row["error_labels"] = ["Lỗi logic"]
    result = evaluate_generated(
        {},
        [row],
        {"task": "task2"},
        generate=lambda *_: '{"error_labels": ["Lỗi logic"]}',
    )
    assert result["metrics"]["macro_f1"] == 0.1
    assert result["metrics"]["selection_score"] == 0.1


def test_task_three_scores_generated_level_compliance():
    result = evaluate_generated(
        {},
        [sample()],
        {"task": "task3"},
        generate=lambda *_: "Chỉ cần thêm cur = cur->next;",
    )
    assert result["metrics"]["selection_score"] is None
    assert result["metrics"]["heuristic_compliance_score"] == 0
    assert result["metrics"]["diagnosis_correct_rate"] is None


def test_vague_feedback_cannot_be_a_checkpoint_selection_metric():
    result = evaluate_generated(
        {}, [sample()], {"task": "task3"}, generate=lambda *_: "junk"
    )
    assert result["metrics"]["selection_score"] is None


def test_balancing_repeats_only_existing_training_examples():
    rows = [sample(str(i), str(i)) for i in range(4)]
    for row in rows:
        row["error_labels"] = ["Lỗi logic"]
    rows[-1]["error_labels"] = ["Lỗi hàm"]
    balanced, repeats = balance_training_samples(
        rows, "task2", {"training": {"balance_by": "labels", "max_repeat": 3}}
    )
    assert repeats["3"] == 3
    assert len(balanced) == 6
    assert all(row in rows for row in balanced)
