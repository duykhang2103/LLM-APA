import hashlib

import pytest

from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.defined_decisions import defined_questions
from llm_grading.release_ledger import fingerprint
from scripts.replay_task2_defined_decisions import replay_decisions


def decision(sample, index, repeat=False):
    probabilities = [0.5] * len(ERROR_LABELS)
    probabilities[index] = 0.51
    return {
        "sample_id": sample["sample_id"],
        "source_sha256": hashlib.sha256(sample["code"].encode()).hexdigest(),
        "question_sha256": fingerprint(defined_questions()),
        "probabilities_true": probabilities,
        "labels": [ERROR_LABELS[index]],
        "answers": {
            f"label_{i}": {"noul": value} for i, value in enumerate(probabilities)
        },
        "repeat": repeat,
    }


def test_replay_uses_release_order_and_strict_threshold():
    first = {"sample_id": "a", "code": "int main() {}"}
    second = {"sample_id": "b", "code": "int main() { return 0; }"}
    rows = [decision(second, 1), decision(first, 0), decision(second, 1, repeat=True)]
    result = replay_decisions([first, second], rows)
    assert [row["sample_id"] for row in result] == ["a", "b"]
    assert [row["output"]["taxonomy_error"] for row in result] == [
        [ERROR_LABELS[0]],
        [ERROR_LABELS[1]],
    ]
    with pytest.raises(ValueError, match="instability"):
        replay_decisions([first, second], [*rows[:2], decision(second, 2, repeat=True)])
    with pytest.raises(ValueError, match="source or prompt"):
        replay_decisions([{**first, "code": "changed"}, second], rows)
