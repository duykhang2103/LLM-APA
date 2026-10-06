"""Export the shapes in the lecturer examples without internal diagnostics."""

import pytest
from test_experiment_contracts import RUBRIC

from llm_grading.validation.challenge import (
    export_challenge_predictions,
    validate_challenge_predictions,
)


@pytest.mark.parametrize(
    "task,output,expected",
    [
        ("task1", {**RUBRIC, "total": 10}, {"rubric": RUBRIC, "total_score": 10}),
        ("task2", {"error_labels": []}, {"taxonomy_error": []}),
        (
            "task3",
            {
                "feedback": "Tốt.",
                "feedback_level": 1,
                "compliance": {"pass": True, "violations": []},
            },
            {"feedback": "Tốt."},
        ),
    ],
)
def test_export_round_trip(task, output, expected):
    exported = export_challenge_predictions(
        [{"sample_id": "synthetic", "output": output}], task
    )
    assert exported == [{"sample_id": "synthetic", "output": expected}]
    assert validate_challenge_predictions(exported, task) == []


def test_export_rejects_bad_totals_duplicates_and_extra_fields():
    bad = [{"sample_id": "synthetic", "output": {**RUBRIC, "total": 9}}]
    with pytest.raises(ValueError):
        export_challenge_predictions(bad, "task1")
    row = {"sample_id": "synthetic", "output": {"taxonomy_error": []}}
    assert validate_challenge_predictions([row, row], "task2")
    row["output"]["feedback"] = "extra"
    assert validate_challenge_predictions([row], "task2")
