"""Task 2 labels from the list prompt plus chosen labels from the per-label run.

The list prompt (one answer with all labels) almost never predicts the rare
labels; the per-label run (one Yes/No per label) finds them but adds many
false positives on the common ones. This keeps every list-prompt label and
adds only --labels from the per-label run. Both inputs should already be
calibrated (pred_task2_calib.json). On labeled data the macro-F1 of the two
inputs and of the result is printed.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.evaluation.task2 import evaluate_task2
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.predictions import validate_prediction_records

RARE = ["Lỗi thuật toán", "Lỗi hard-code", "Lỗi style"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True, help="List-prompt predictions (calibrated)")
    parser.add_argument("--per-label", required=True, help="Per-label predictions (calibrated)")
    parser.add_argument("--labels", nargs="+", default=RARE, help="Labels taken from the per-label run")
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", help="Labeled dataset folder: print macro-F1")
    args = parser.parse_args()
    assert set(args.labels) <= set(ERROR_LABELS), "unknown label"
    base = {r["sample_id"]: r["output"]["error_labels"] for r in read_json(args.list)}
    extra = {r["sample_id"]: r["output"]["error_labels"] for r in read_json(args.per_label)}
    if set(base) != set(extra):
        raise ValueError("The two prediction files cover different samples")
    result = []
    for sid, labels in base.items():
        merged = set(labels) | (set(extra[sid]) & set(args.labels))
        result.append({"sample_id": sid, "output": {"error_labels": [x for x in ERROR_LABELS if x in merged]}})
    errors = validate_prediction_records(result, "task2")
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, result)
    added = sum(len(r["output"]["error_labels"]) - len(base[r["sample_id"]]) for r in result)
    report = {"labels_added": added, "taken_from_per_label": args.labels}
    if args.input:
        refs = {s["sample_id"]: s for s in load_samples(args.input, task="task2")}
        order = [refs[r["sample_id"]] for r in result]
        as_rows = lambda d: [{"sample_id": s, "output": {"error_labels": d[s]}} for s in base]  # noqa: E731
        report["macro_f1"] = {name: round(evaluate_task2(rows, order)["macro_f1"], 4)
                              for name, rows in (("list", as_rows(base)), ("per_label", as_rows(extra)),
                                                 ("combined", result))}
    print(json.dumps(report, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
