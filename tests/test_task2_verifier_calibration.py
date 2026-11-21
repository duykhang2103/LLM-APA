import pytest

from llm_grading.task2_verifier import FOCUS_LABELS
from scripts.calibrate_task2_verifier import apply, fit_label, labeled_cells


def test_negative_margins_can_separate_classes_with_a_validated_threshold():
    train = [(-5.0, False), (-4.0, False), (-3.0, True)]
    valid = [(-5.0, False), (-3.1, True)]
    assert fit_label(train, valid, recovery=False)["threshold"] == -3.5
    assert fit_label(train, [(-5.0, False)], recovery=True)["threshold"] is None
    assert fit_label(train, [*valid, (-4.9, True)], recovery=False)["threshold"] is None


def test_unreviewed_cells_do_not_become_negatives():
    logic, array, function, _ = FOCUS_LABELS
    row = {
        "sample_id": "x",
        "verifier_judgments": {
            logic: "present",
            array: "uncertain",
            function: "absent",
        },
    }
    cells = labeled_cells([row], {"x": {"margins": dict.fromkeys(FOCUS_LABELS, -4.0)}})
    assert cells[logic] == [(-4.0, True)]
    assert cells[array] == []
    assert cells[function] == [(-4.0, False)]
    with pytest.raises(ValueError, match="reference"):
        labeled_cells(
            [{"sample_id": "x"}], {"x": {"margins": dict.fromkeys(FOCUS_LABELS, -4.0)}}
        )


def test_abstention_and_skipped_inputs_preserve_baseline():
    logic, array, function, algorithm = FOCUS_LABELS
    rows = [
        {
            "sample_id": "x",
            "margins": {logic: -5, array: -3, function: -1},
            "skipped": {algorithm: "input_budget"},
        }
    ]
    baseline = [{"sample_id": "x", "output": {"taxonomy_error": [logic, algorithm]}}]
    rule = {
        "rules": {k: {"threshold": (-4 if k == array else None)} for k in FOCUS_LABELS}
    }
    (row,) = apply(rows, baseline, rule)
    assert row["full"] == [logic, array, algorithm]
    assert row["broad"] == [logic, algorithm]
    rows[0]["margins"][function] = float("nan")
    with pytest.raises(ValueError, match="margins"):
        apply(rows, baseline, rule)
