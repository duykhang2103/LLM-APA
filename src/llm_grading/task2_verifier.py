"""Binary completion records and merges for the Task 2 verifier."""

import json

from llm_grading.data.taxonomy import ERROR_LABELS

PROMPT_VERSION = "verifier_v001"
DEFINITIONS = {
    ERROR_LABELS[
        2
    ]: "Identify a branch or operation that produces the wrong required result.",
    ERROR_LABELS[
        4
    ]: "Check array or string indexing, length, bounds and termination against the contract.",
    ERROR_LABELS[
        5
    ]: "Check required arguments, return values, references and caller-visible behavior.",
    ERROR_LABELS[
        7
    ]: "Identify a strategy that fails a stated algorithmic or complexity requirement.",
}
FOCUS_LABELS = tuple(DEFINITIONS)
RECOVERY_LABELS = frozenset((ERROR_LABELS[4], ERROR_LABELS[5]))
RESPONSES = ("No", "Yes")


def has_source(sample: dict) -> bool:
    return all(
        isinstance(sample.get(key), str) and bool(sample[key].strip())
        for key in ("code", "problem_statement")
    )


def render_verifier_prompt(template: str, sample: dict, label: str) -> str:
    if not has_source(sample):
        raise ValueError("Verifier requires the problem statement and source")
    return template.format(
        category=label,
        definition=DEFINITIONS[label],
        state=json.dumps(
            {key: sample[key] for key in ("problem_statement", "code")},
            ensure_ascii=False,
        ),
    )


def verifier_records(sample: dict, template: str, labels: list[str]) -> list[dict]:
    """Serialize reviewed cells or the closed-set labels of a legacy reference."""
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate Task 2 reference labels")
    if "verifier_judgments" in sample:
        judgments = sample["verifier_judgments"]
        if not isinstance(judgments, dict) or any(
            label not in FOCUS_LABELS
            or status not in ("present", "absent", "uncertain")
            for label, status in judgments.items()
        ):
            raise ValueError("Invalid Task 2 verifier judgments")
    else:
        judgments = {
            label: "present" if label in labels else "absent" for label in FOCUS_LABELS
        }
    return [
        {
            "sample_id": f"{sample['sample_id']}:{ERROR_LABELS.index(label)}",
            "prompt": render_verifier_prompt(template, sample, label),
            "target": RESPONSES[judgments[label] == "present"],
        }
        for label in FOCUS_LABELS
        if judgments.get(label) in ("present", "absent")
    ]


def routed_labels(baseline: list[str], *, recover: bool) -> tuple[str, ...]:
    members = set(baseline)
    if members - set(ERROR_LABELS) or len(members) != len(baseline):
        raise ValueError("Invalid baseline taxonomy labels")
    return tuple(
        label
        for label in FOCUS_LABELS
        if (label in RECOVERY_LABELS and recover and label not in members)
        or (label not in RECOVERY_LABELS and label in members)
    )


def merge_labels(
    baseline: list[str], answers: dict[str, bool], *, recover: bool
) -> list[str]:
    routed = routed_labels(baseline, recover=recover)
    if set(answers) - set(routed):
        raise ValueError("Verifier answer outside the routed categories")
    if any(type(answer) is not bool for answer in answers.values()):
        raise ValueError("Verifier answers must be boolean")
    return [label for label in ERROR_LABELS if answers.get(label, label in baseline)]


def verifier_choices(tokenizer, prompt: str) -> tuple[list[int], int, int]:
    from llm_grading.training.dataset import chat_text

    def encode(text):
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    text = chat_text(tokenizer, prompt)
    prefix = encode(text)
    choices = [encode(answer) for answer in RESPONSES]
    if any(len(ids) != 1 for ids in choices):
        raise ValueError("Verifier requires single-token binary answers")
    if choices[0] == choices[1] or any(
        encode(text + answer) != prefix + ids
        for answer, ids in zip(RESPONSES, choices, strict=True)
    ):
        raise ValueError(
            "Verifier options do not share the training prefix tokenization"
        )
    return prefix, choices[0][0], choices[1][0]
