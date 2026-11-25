from pathlib import Path

import pytest

from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.task2_verifier import (
    FOCUS_LABELS,
    merge_labels,
    routed_labels,
    verifier_choices,
)
from llm_grading.training.dataset import build_training_dataset


def sample():
    return {
        "sample_id": "case",
        "problem_statement": "Return the length.",
        "code": "int length() { return 0; }",
        "error_labels": [ERROR_LABELS[5]],
        "feedback": "Private reference feedback.",
    }


def test_completion_records_use_only_source_and_problem():
    config = {"prompt": {"version": "verifier_v001"}}
    rows = build_training_dataset([sample()], "task2", config, training=False)
    assert len(rows) == 4
    assert [row["target"] for row in rows] == ["No", "No", "Yes", "No"]
    for label, row in zip(FOCUS_LABELS, rows, strict=True):
        assert label in row["prompt"]
        assert sample()["code"] in row["prompt"]
        assert "Private reference feedback" not in row["prompt"]


def test_routes_and_partial_verification_preserve_other_labels():
    logic, array, function, alg = FOCUS_LABELS
    io = ERROR_LABELS[1]
    baseline = [alg, io, logic]
    assert routed_labels(baseline, recover=False) == (logic, alg)
    assert routed_labels(baseline, recover=True) == FOCUS_LABELS
    assert merge_labels(baseline, {logic: False, array: True}, recover=True) == [
        io,
        array,
        alg,
    ]
    assert merge_labels([function], {}, recover=True) == [function]
    with pytest.raises(ValueError, match="outside"):
        merge_labels(baseline, {io: False}, recover=True)


def test_explicit_verifier_judgments_mask_uncertain_and_missing_categories():
    logic, array, function, _ = FOCUS_LABELS
    reviewed = {
        **sample(),
        "verifier_judgments": {
            logic: "absent",
            array: "uncertain",
            function: "present",
        },
    }
    config = {"prompt": {"version": "verifier_v001"}}
    rows = build_training_dataset([reviewed], "task2", config, training=False)
    assert [row["target"] for row in rows] == ["No", "Yes"]
    assert [row["sample_id"] for row in rows] == ["case:2", "case:5"]

    for training in (False, True):
        rebuilt = build_training_dataset([reviewed], "task2", config, training=training)
        assert rebuilt == rows


@pytest.mark.parametrize(
    "judgments", [None, {ERROR_LABELS[0]: "absent"}, {FOCUS_LABELS[0]: True}]
)
def test_explicit_verifier_judgments_reject_invalid_values(judgments):
    reviewed = {**sample(), "verifier_judgments": judgments}
    with pytest.raises(ValueError, match="verifier judgments"):
        build_training_dataset(
            [reviewed],
            "task2",
            {"prompt": {"version": "verifier_v001"}},
            training=False,
        )


def test_pinned_tokenizer_binary_answers_share_identical_prefix():
    snapshot = Path(
        "outputs/local-qwen-diagnostic-v1/huggingface/hub/models--Qwen--Qwen3.5-4B"
        "/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    )
    if not snapshot.is_dir():
        pytest.skip("Pinned Qwen tokenizer is not cached")
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        snapshot, local_files_only=True
    )
    row = build_training_dataset(
        [sample()], "task2", {"prompt": {"version": "verifier_v001"}}, training=False
    )[0]
    prefix, no, yes = verifier_choices(tokenizer, row["prompt"])
    assert prefix and no != yes
    assert tokenizer.decode([no]) == "No"
    assert tokenizer.decode([yes]) == "Yes"
