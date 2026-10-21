"""Post-process Task 2 predictions with the pipeline's own evidence.

Rules follow the label definitions, not tuned cut-offs:
- "Lỗi biên dịch" follows the compile result (relaxed check, which matched
  the teacher's `compilable` on every training sample): dropped when the code
  compiles, added when it does not. Other labels are kept.
- With --tests: a compiling submission that passes every generated test keeps
  only labels a test cannot see ("Lỗi style", "Lỗi hard-code", "Lỗi nhập/xuất").
--max-labels caps the list (0 = no cap). A cap keeps the model's first labels,
which come in taxonomy order rather than by confidence; on the training data
caps of 1-3 moved macro-F1 erratically, so the default is no cap.
If --input carries reference labels, macro-F1 before/after is printed.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.evaluation.task2 import evaluate_task2
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.predictions import validate_prediction_records

COMPILE = "Lỗi biên dịch"
UNSEEN_BY_TESTS = ("Lỗi style", "Lỗi hard-code", "Lỗi nhập/xuất")


def calibrate(labels, compiles, max_labels=0, compile_only=False, all_pass=False):
    labels = [label for label in labels if label != COMPILE]
    if not compiles:
        labels = [COMPILE] + ([] if compile_only else labels)
    elif all_pass:
        labels = [label for label in labels if label in UNSEEN_BY_TESTS]
    return labels[:max_labels] if max_labels else labels


def all_tests_pass(report):
    return bool(report and report["total"] and report["pass_rate"] == 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", required=True, help="Internal Task 2 predictions")
    parser.add_argument("--evidence", required=True, help="scripts/compile_evidence.py output")
    parser.add_argument("--tests", help="scripts/run_tests.py output (all-pass rule)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", help="Dataset with reference labels (optional, for metrics)")
    parser.add_argument("--max-labels", type=int, default=0)
    parser.add_argument("--compile-only", action="store_true", help="Compile failures get only that label")
    args = parser.parse_args()

    evidence = read_json(args.evidence)["evidence"]
    reports = read_json(args.tests)["reports"] if args.tests else {}
    predictions = read_json(args.predictions)
    missing = [r["sample_id"] for r in predictions if r["sample_id"] not in evidence]
    if missing:
        raise ValueError(f"No compile evidence for {missing[:5]}")
    result, changed = [], Counter()
    for row in predictions:
        sid, before = row["sample_id"], row["output"]["error_labels"]
        after = calibrate(before, evidence[sid]["relaxed"], args.max_labels, args.compile_only,
                          all_tests_pass(reports.get(sid)))
        changed["changed" if after != before else "unchanged"] += 1
        result.append({"sample_id": sid, "output": {**row["output"], "error_labels": after}})
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
        refs = {s["sample_id"]: s for s in load_samples(args.input, task="task2")}
        if all(r["sample_id"] in refs for r in predictions):
            references = [refs[r["sample_id"]] for r in predictions]
            report["macro_f1_before"] = round(evaluate_task2(predictions, references)["macro_f1"], 4)
            report["macro_f1_after"] = round(evaluate_task2(result, references)["macro_f1"], 4)
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
