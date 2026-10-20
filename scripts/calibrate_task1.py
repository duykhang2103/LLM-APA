"""Post-process Task 1 predictions: empty-submission rule + snap to teacher templates.

Templates come only from --train labels. When a prediction's sample is itself
in --train (evaluating on training data), its own label is excluded (LOO).
If --input carries reference rubrics, before/after metrics are printed,
plus "oracle_snap": the cost of snapping when the model were perfect.
With --evidence (scripts/compile_evidence.py output), a prediction whose
`compilable` contradicts the pipeline's own compile result is replaced by the
most frequent training template of that exam type with the evidence value.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.evaluation.task1 import evaluate_task1
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.predictions import validate_prediction_records
from tasks.task1_grading.calibration import FIELDS, RubricSnapper, is_empty_submission


def summary(metrics):
    keys = ["qwk_total", "mae_total"] + [f"exact_match_{f}" for f in FIELDS]
    return {k: round(metrics[k], 4) for k in keys}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Internal Task 1 predictions")
    parser.add_argument("--input", required=True, help="Dataset the predictions were made on")
    parser.add_argument("--train", required=True, help="Labeled training dataset (templates)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-count", type=int, default=1, help="Ignore templates seen fewer times")
    parser.add_argument("--no-group", action="store_true", help="Share templates across exam types")
    parser.add_argument("--no-snap", action="store_true")
    parser.add_argument("--no-empty-rule", action="store_true")
    parser.add_argument("--evidence", help="Compile evidence JSON; overrides contradicted compilable")
    args = parser.parse_args()

    samples = {s["sample_id"]: s for s in load_samples(args.input, task="task1")}
    train = load_samples(args.train, task="task1")
    train_ids = {s["sample_id"] for s in train}
    snapper = RubricSnapper(train, args.min_count, None if args.no_group else "problem_type")
    evidence = read_json(args.evidence)["evidence"] if args.evidence else {}
    predictions = read_json(args.predictions)
    if {r["sample_id"] for r in predictions} != set(samples):
        raise ValueError("Prediction IDs must match the --input dataset exactly")

    result, audit, reasons = [], [], Counter()
    for row in predictions:
        sid, sample = row["sample_id"], samples[row["sample_id"]]
        before = {f: row["output"][f] for f in FIELDS}
        after, reason = before, "unchanged"
        if not args.no_empty_rule and is_empty_submission(sample["code"]):
            after, reason = dict.fromkeys(FIELDS, 0), "empty_submission"
        elif sid in evidence and int(evidence[sid]["relaxed"]) != before["compilable"]:
            exclude = sid if sid in train_ids else None
            after = snapper.most_frequent(sample, int(evidence[sid]["relaxed"]), exclude) or {
                **before, "compilable": int(evidence[sid]["relaxed"])
            }
            reason = "evidence_flip"
        elif not args.no_snap:
            exclude = sid if sid in train_ids else None
            after, template = snapper.snap(before, sample, exclude)
            reason = "snapped" if after != before else "already_template"
        reasons[reason] += 1
        result.append({"sample_id": sid, "output": {**after, "total": sum(after.values())}})
        audit.append({"sample_id": sid, "reason": reason, "before": before, "after": after})

    errors = validate_prediction_records(result, "task1")
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, result)
    report = {
        "settings": vars(args),
        "reasons": dict(reasons),
        "total_before": dict(sorted(Counter(sum(a["before"].values()) for a in audit).items())),
        "total_after": dict(sorted(Counter(sum(a["after"].values()) for a in audit).items())),
    }
    references = [samples[r["sample_id"]] for r in predictions]
    if all(s.get("rubric") for s in references):
        oracle = [
            {"output": snapper.snap(s["rubric"], s, s["sample_id"] if s["sample_id"] in train_ids else None)[0]}
            for s in references
        ]
        report["metrics_before"] = summary(evaluate_task1(predictions, references))
        report["metrics_after"] = summary(evaluate_task1(result, references))
        report["oracle_snap"] = summary(evaluate_task1(oracle, references))
    audit_path = Path(args.output).with_suffix(".audit.json")
    write_json(audit_path, {**report, "samples": audit})
    print(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    print(f"Wrote {args.output} and {audit_path}")


if __name__ == "__main__":
    raise SystemExit(main())
