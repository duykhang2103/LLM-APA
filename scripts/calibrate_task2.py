"""Post-process Task 2 predictions with the pipeline's own compile evidence.

"Lỗi biên dịch" is dropped when the code compiles (relaxed check, which matched
the teacher's `compilable` on every training sample) and added when it does
not. On a compile failure the teacher kept one other label in 3 of 6 training
samples, so at most one other label is kept (--compile-only drops them all).
The label count is capped by --max-labels. If --input carries reference labels, macro-F1 before/after
is printed.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.predictions import validate_prediction_records

COMPILE = "Lỗi biên dịch"


def calibrate(labels, compiles, max_labels=3, compile_only=False):
    labels = [label for label in labels if label != COMPILE]
    if not compiles:
        return [COMPILE] if compile_only else [COMPILE, *labels][: min(2, max_labels)]
    return labels[:max_labels]


def macro_f1(rows, references):
    space = sorted({x for r in references.values() for x in r} | {x for r in rows for x in r["labels"]})
    scores = []
    for label in space:
        tp = sum(label in r["labels"] and label in references[r["sample_id"]] for r in rows)
        fp = sum(label in r["labels"] and label not in references[r["sample_id"]] for r in rows)
        fn = sum(label not in r["labels"] and label in references[r["sample_id"]] for r in rows)
        scores.append(2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 1.0)
    return round(sum(scores) / len(scores), 4)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Internal Task 2 predictions")
    parser.add_argument("--evidence", required=True, help="scripts/compile_evidence.py output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", help="Dataset with reference labels (optional, for metrics)")
    parser.add_argument("--max-labels", type=int, default=3)
    parser.add_argument("--compile-only", action="store_true", help="Compile failures get only that label")
    args = parser.parse_args()

    evidence = read_json(args.evidence)["evidence"]
    predictions = read_json(args.predictions)
    missing = [r["sample_id"] for r in predictions if r["sample_id"] not in evidence]
    if missing:
        raise ValueError(f"No compile evidence for {missing[:5]}")
    result, changed = [], Counter()
    for row in predictions:
        before = row["output"]["error_labels"]
        after = calibrate(before, evidence[row["sample_id"]]["relaxed"], args.max_labels, args.compile_only)
        changed["changed" if after != before else "unchanged"] += 1
        result.append({"sample_id": row["sample_id"], "output": {**row["output"], "error_labels": after}})
    errors = validate_prediction_records(result, "task2")
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, result)

    def distribution(rows):
        return dict(Counter(x for r in rows for x in r["output"]["error_labels"]).most_common())

    report = {
        "settings": vars(args),
        **changed,
        "labels_per_sample_after": round(sum(len(r["output"]["error_labels"]) for r in result) / len(result), 2),
        "labels_before": distribution(predictions),
        "labels_after": distribution(result),
    }
    if args.input:
        refs = {s["sample_id"]: s["error_labels"] for s in load_samples(args.input, task="task2")}
        if all(r["sample_id"] in refs for r in predictions):
            def view(rows):
                return [{"sample_id": r["sample_id"], "labels": r["output"]["error_labels"]} for r in rows]
            report["macro_f1_before"] = macro_f1(view(predictions), refs)
            report["macro_f1_after"] = macro_f1(view(result), refs)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
