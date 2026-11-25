"""Fit or apply label thresholds to saved Kaggle verifier margins."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from llm_grading.data.loader import load_samples  # noqa: E402
from llm_grading.data.split import check_split_overlap  # noqa: E402
from llm_grading.data.taxonomy import ERROR_LABELS  # noqa: E402
from llm_grading.runtime import read_json, write_json  # noqa: E402
from llm_grading.task2_verifier import (  # noqa: E402
    FOCUS_LABELS,
    RECOVERY_LABELS,
    merge_labels,
    routed_labels,
)


def indexed(rows):
    result = {row["sample_id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate sample IDs")
    return result


def valid_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def labeled_cells(rows, scores):
    cells = {label: [] for label in FOCUS_LABELS}
    for row in rows:
        sid = row["sample_id"]
        margins = scores[sid]["margins"]
        if set(margins) != set(FOCUS_LABELS) or not all(
            map(valid_number, margins.values())
        ):
            raise ValueError(
                "Calibration requires complete finite scores for all four categories"
            )
        if "verifier_judgments" in row:
            judgments = row["verifier_judgments"]
            if not isinstance(judgments, dict) or any(
                k not in FOCUS_LABELS or v not in ("absent", "present", "uncertain")
                for k, v in judgments.items()
            ):
                raise ValueError("Invalid reviewed judgments")
        else:
            labels = row.get("error_labels")
            if (
                row.get("_has_reference") is False
                or not isinstance(labels, list)
                or set(labels) - set(ERROR_LABELS)
            ):
                raise ValueError("Calibration requires reference labels")
            judgments = {
                k: "present" if k in labels else "absent" for k in FOCUS_LABELS
            }
        for label, status in judgments.items():
            if status != "uncertain":
                cells[label].append((margins[label], status == "present"))
    return cells


def counts(cells, threshold):
    positive = sum(y for _, y in cells)
    tp = sum(y and x > threshold for x, y in cells)
    tn = sum(not y and x <= threshold for x, y in cells)
    return dict(positive=positive, negative=len(cells) - positive, tp=tp, tn=tn)


def fit_label(train, validation, *, recovery):
    """Fit balanced accuracy on train. Require validation support before editing labels."""
    values = sorted({x for x, _ in train})
    positives = sum(y for _, y in train)
    negatives = len(train) - positives
    if not positives or not negatives or len(values) < 2:
        return dict(threshold=None, reason="insufficient_training_support")

    def rank(threshold):
        c = counts(train, threshold)
        # Integer equivalent of balanced accuracy for a fixed sample.
        return (
            c["tp"] * negatives + c["tn"] * positives,
            c["tn"] if recovery else c["tp"],
        )

    cutoffs = [(lo + hi) / 2 for lo, hi in zip(values, values[1:])]
    selected = max(cutoffs, key=rank)
    c = counts(validation, selected)
    # Require a validated positive and negative. For vetoes protect all known
    # positives. For additions require a majority of negatives to stay negative.
    sensitivity = c["tp"] / c["positive"] if c["positive"] else 0
    specificity = c["tn"] / c["negative"] if c["negative"] else 0
    passes = (
        rank(selected)[0] > positives * negatives
        and sensitivity + specificity > 1
        and (specificity > 0.5 if recovery else sensitivity == 1)
    )
    return dict(
        threshold=selected if passes else None,
        fitted_threshold=selected,
        reason="enabled" if passes else "abstain",
        training=counts(train, selected),
        validation=c,
    )


def apply(rows, baseline_rows, policy):
    rules = policy["rules"]
    if set(rules) != set(FOCUS_LABELS):
        raise ValueError("Calibration requires one rule per focus label")
    thresholds = {label: rules[label]["threshold"] for label in FOCUS_LABELS}
    if any(t is not None and not valid_number(t) for t in thresholds.values()):
        raise ValueError("Invalid threshold")
    baseline = indexed(baseline_rows)
    if indexed(rows).keys() != baseline.keys():
        raise ValueError("Prediction and baseline coverage differ")
    result = []
    for row in rows:
        labels = baseline[row["sample_id"]]["output"]["taxonomy_error"]
        margins, skipped = row["margins"], row["skipped"]
        routes = set(routed_labels(labels, recover=True))
        if (
            (set(margins) | set(skipped)) != routes
            or set(margins) & set(skipped)
            or not all(map(valid_number, margins.values()))
        ):
            raise ValueError("Invalid routed margins")
        answers = {
            k: v > thresholds[k]
            for k, v in margins.items()
            if thresholds[k] is not None
        }
        broad = {k: v for k, v in answers.items() if k not in RECOVERY_LABELS}
        result.append(
            {
                **row,
                "broad": merge_labels(labels, broad, recover=False),
                "full": merge_labels(labels, answers, recover=True),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("fit")
    fit.add_argument("--train", type=Path, required=True)
    fit.add_argument("--val", type=Path, required=True)
    replay = commands.add_parser("apply")
    replay.add_argument("--policy", type=Path, required=True)
    replay.add_argument("--baseline", type=Path, required=True)
    for p in (fit, replay):
        p.add_argument(
            "--run",
            type=Path,
            required=True,
            help="Directory with tuned.json and runtime.json",
        )
        p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a fresh output path")
    scores = read_json(args.run / "tuned.json")
    runtime = read_json(args.run / "runtime.json")
    identity = {
        k: runtime["checks"][k]
        for k in ("adapter_sha256", "model_revision", "model_dtype")
    }
    if args.command == "fit":
        train = load_samples(args.train, task="task2")
        val = load_samples(args.val, task="task2")
        check_split_overlap(train, val)
        by_id = indexed(scores)
        if by_id.keys() != indexed(train + val).keys():
            raise ValueError(
                "Scores must cover exactly the train and validation splits"
            )
        a, b = labeled_cells(train, by_id), labeled_cells(val, by_id)
        result = {
            "model": identity,
            "rules": {
                k: fit_label(a[k], b[k], recovery=k in RECOVERY_LABELS)
                for k in FOCUS_LABELS
            },
        }
    else:
        policy = read_json(args.policy)
        if policy["model"] != identity:
            raise ValueError("Calibration and inference model identities differ")
        result = apply(scores, read_json(args.baseline), policy)
    write_json(args.output, result)


if __name__ == "__main__":
    main()
