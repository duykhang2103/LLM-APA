import pytest

from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.evaluation.defined_decisions import evaluate_defined


def test_uncertain_cells_and_environment_not_made_into_negatives():
    rows = [
        {
            "sample_id": "a",
            "functional_defect_expected": True,
            "policy_only": False,
            "severity": 3,
        },
        {
            "sample_id": "b",
            "functional_defect_expected": False,
            "policy_only": False,
            "severity": 0,
        },
        {
            "sample_id": "c",
            "functional_defect_expected": None,
            "policy_only": False,
            "severity": 0,
        },
    ]
    labels = {
        "a": ["present"] + ["absent"] * 9,
        "b": ["absent"] * 10,
        "c": ["uncertain"] * 10,
    }
    predictions = {
        "A": {"a": [ERROR_LABELS[0]], "b": [], "c": [ERROR_LABELS[0]]},
        "B": {"a": [], "b": [], "c": []},
    }
    result = evaluate_defined(rows, predictions, labels)
    assert result["metrics"]["A"]["per_class"][ERROR_LABELS[0]]["fp"] == 0
    assert result["metrics"]["A"]["per_class"][ERROR_LABELS[0]]["known"] == 2
    assert result["metrics"]["A"]["behavior"]["fp"] == 0
    assert result["metrics"]["B"]["behavior"]["fn"] == 1
    assert not result["criteria"]["no_additional_severity3_miss"]
    assert result["contingencies"]["unknown"]["A1_B0"] == 1


def test_partial_prediction_coverage_is_rejected():
    with pytest.raises(ValueError, match="coverage"):
        evaluate_defined(
            [{"sample_id": "a"}], {"A": {}, "B": {}}, {"a": ["absent"] * 10}
        )


def test_no_taxonomy_adjudication_is_inconclusive_not_failure():
    rows = [
        {
            "sample_id": "a",
            "functional_defect_expected": True,
            "policy_only": False,
            "severity": 1,
        }
    ]
    predictions = {"A": {"a": []}, "B": {"a": []}}
    result = evaluate_defined(rows, predictions, {"a": ["uncertain"] * 10})
    assert result["metrics"]["A"]["macro_f1"] is None
    assert result["criteria"]["macro_f1_gain_at_least_003"] is None
    assert result["decision"] == "INCONCLUSIVE"
