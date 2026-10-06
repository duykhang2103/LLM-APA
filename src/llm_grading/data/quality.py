"""Audit annotations and select training references without rewriting targets."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from llm_grading.data.taxonomy import ERROR_LABELS
from tasks.task3_feedback.compliance import check_compliance

QUALITY_FLAGS = {
    "compiler_rubric_conflict",
    "test_rubric_conflict",
    "exam_weight_conflict",
    "feedback_level_violation",
}


class TrainingQualityError(ValueError):
    def __init__(self, message, audit):
        super().__init__(message)
        self.audit = audit


def audit_sample(sample: dict[str, Any], task: str) -> list[str]:
    flags = list(sample.get("review_flags", []))
    rubric = sample.get("rubric", {})
    if task == "task1":
        if rubric.get("compilable") == 1 and re.search(
            r"(?:fatal )?error:|undefined reference|failed",
            str(sample.get("compile_log") or ""),
            re.I,
        ):
            flags.append("compiler_rubric_conflict")
        tests = sample.get("test_report")
        if (
            isinstance(tests, list)
            and tests
            and rubric.get("logic") == 4
            and any(t.get("passed") is False for t in tests)
        ):
            flags.append("test_rubric_conflict")
    weights = dict(
        re.findall(
            r"Problem\s+(\d+)\s*\((\d+(?:\.\d+)?)\s+điểm\)",
            sample.get("problem_statement", ""),
        )
    )
    if any(
        str(p.get("pid", "")).removeprefix("P") in weights
        and float(weights[str(p["pid"]).removeprefix("P")]) != p.get("max_score")
        for p in sample.get("problems", [])
    ):
        flags.append("exam_weight_conflict")
    if (
        task == "task3"
        and sample.get("feedback")
        and not check_compliance(sample["feedback"], sample["feedback_level"])["pass"]
    ):
        flags.append("feedback_level_violation")
    return sorted(set(flags))


def audit_dataset(samples: list[dict[str, Any]], task: str) -> dict[str, Any]:
    report: dict[str, Any] = {
        "count": len(samples),
        "exam_counts": dict(Counter(s.get("problem_id", "unknown") for s in samples)),
        "samples": [
            {"sample_id": s["sample_id"], "flags": audit_sample(s, task)}
            for s in samples
        ],
    }
    if task == "task1":
        report["score_support"] = dict(
            Counter(sum(s.get("rubric", {}).values()) for s in samples)
        )
        report["rubric_support"] = {
            name: dict(Counter(s.get("rubric", {}).get(name) for s in samples))
            for name in (
                "compilable",
                "io_format",
                "logic",
                "edge_case",
                "complexity",
                "code_quality",
            )
        }
    if task == "task2":
        support = Counter(
            label for s in samples for label in set(s.get("error_labels", []))
        )
        report.update(
            label_support={label: support[label] for label in ERROR_LABELS},
            missing_labels=[label for label in ERROR_LABELS if not support[label]],
            zero_label_count=sum(not s.get("error_labels") for s in samples),
            perfect_macro_f1_ceiling=sum(bool(support[label]) for label in ERROR_LABELS)
            / len(ERROR_LABELS),
        )
    if task == "task3":
        support = Counter(s["feedback_level"] for s in samples)
        report.update(
            level_support={str(level): support[level] for level in range(1, 5)},
            missing_levels=[level for level in range(1, 5) if not support[level]],
        )
    return report


def select_training_samples(samples, task, config):
    """Apply auditable training-only exclusions; validation remains untouched.

    Compiler/test conflicts are auxiliary evidence, so default to review-only.
    Feedback that demonstrably exceeds its requested level is excluded by default.
    A private review file may explicitly keep/exclude records with a reason.
    """
    policy = config.get("data", {}).get("quality", {})
    excluded_flags = policy.get("exclude_flags", ["feedback_level_violation"])
    if not isinstance(excluded_flags, list) or any(
        f not in QUALITY_FLAGS for f in excluded_flags
    ):
        raise ValueError("Unknown quality.exclude_flags")
    decisions = {}
    review_digest = None
    if policy.get("review_file"):
        raw = Path(policy["review_file"]).read_bytes()
        review_digest = hashlib.sha256(raw).hexdigest()
        decisions = json.loads(raw)
        if not isinstance(decisions, dict):
            raise ValueError("Review file must map sample IDs to decisions")
        for decision in decisions.values():
            if (
                not isinstance(decision, dict)
                or set(decision) != {"action", "reason"}
                or decision.get("action") not in {"keep", "exclude"}
                or not isinstance(decision.get("reason"), str)
                or not decision["reason"].strip()
            ):
                raise ValueError(
                    "Every review decision needs keep/exclude action and a nonempty reason"
                )
        unknown = set(decisions) - {sample["sample_id"] for sample in samples}
        if unknown:
            raise ValueError(
                f"Review decisions reference unknown training sample IDs: {sorted(unknown)}"
            )
    report = audit_dataset(samples, task)
    kept, excluded = [], []
    for sample, evidence in zip(samples, report["samples"]):
        decision = decisions.get(sample["sample_id"])
        exclude = bool(set(evidence["flags"]) & set(excluded_flags))
        reason = "quality flags: " + ", ".join(evidence["flags"])
        if decision:
            exclude = decision["action"] == "exclude"
            reason = decision["reason"]
        evidence["decision"] = "exclude" if exclude else "keep"
        evidence["reason"] = reason if exclude or decision else "reference preserved"
        if exclude:
            excluded.append(dict(evidence))
        else:
            kept.append(sample)
    retained = audit_dataset(kept, task)
    report.update(
        excluded=excluded,
        retained_count=len(kept),
        retained_coverage=retained,
        policy=policy,
        review_sha256=review_digest,
    )
    if policy.get("require_coverage", False) and (
        retained.get("missing_labels") or retained.get("missing_levels")
    ):
        raise TrainingQualityError(
            "Training coverage is incomplete; inspect label/level support or explicitly disable require_coverage for smoke runs",
            report,
        )
    return kept, report


def balance_training_samples(samples, task, config):
    """Bounded repetition of existing training references; never invent labels."""
    settings = config.get("training", {})
    mode = settings.get("balance_by", "none")
    if mode not in {"none", "labels", "feedback_level"}:
        raise ValueError("training.balance_by must be none, labels, or feedback_level")
    if (
        mode == "labels"
        and task != "task2"
        or mode == "feedback_level"
        and task != "task3"
    ):
        raise ValueError("Balancing mode does not match the task")
    cap = settings.get("max_repeat", 3)
    if type(cap) is not int or not 1 <= cap <= 10:
        raise ValueError("training.max_repeat must be an integer between 1 and 10")
    groups = [
        (set(s["error_labels"]) or {"no_error"})
        if mode == "labels"
        else {str(s["feedback_level"])}
        if mode == "feedback_level"
        else {"all"}
        for s in samples
    ]
    support = Counter(key for group in groups for key in group)
    largest = max(support.values(), default=1)
    repeated, counts = [], {}
    for s, group in zip(samples, groups):
        count = min(cap, max(1, round(largest / min(support[key] for key in group))))
        repeated.extend([s] * count)
        counts[s["sample_id"]] = count
    return repeated, counts
