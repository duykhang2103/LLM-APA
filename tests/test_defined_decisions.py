from llm_grading.decision_diagnostic import labels_from_probabilities, questions
from llm_grading.defined_decisions import defined_questions


def test_definition_intervention_only_appends_question_text():
    original = questions("noul")
    revised = defined_questions()
    assert list(revised) == list(original)
    for key in original:
        assert revised[key]["type"] == original[key]["type"] == "noul"
        assert set(revised[key]) == set(original[key])
        assert revised[key]["instructions"].startswith(original[key]["instructions"])
        for marker in ("Positive condition:", "Negative condition:", "Overlap rule:"):
            assert marker in revised[key]["instructions"]


def test_threshold_remains_independent_and_strict():
    probabilities = [0.5, 0.50001, 0.49, 0.9, 0.1, 0.8, 0.0, 1.0, 0.51, 0.5]
    assert len(labels_from_probabilities(probabilities)) == 5
