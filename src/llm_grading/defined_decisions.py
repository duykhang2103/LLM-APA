"""Versioned question-only intervention; model readout and thresholds unchanged."""

import json
from pathlib import Path

from llm_grading.decision_diagnostic import questions

POLICY = (
    Path(__file__).resolve().parents[2]
    / "tasks/task2_errors/prompts/decision_definitions_v001.json"
)


def defined_questions() -> dict:
    base = questions("noul")
    policies = json.loads(POLICY.read_text())
    if len(policies) != len(base):
        raise ValueError("Ten independent category policies required")
    for index, rule in enumerate(policies):
        if set(rule) != {"positive", "negative", "overlap"}:
            raise ValueError("Unexpected category policy fields")
        base[f"label_{index}"]["instructions"] += (
            " Positive condition: "
            + rule["positive"]
            + " Negative condition: "
            + rule["negative"]
            + " Overlap rule: "
            + rule["overlap"]
        )
    return base
