"""Offline checks for Task 3 predictions on the labeled training data.

The official Task 3 score (an LLM judge of the feedback plus the labels) is not
available locally. This reports what can be measured without a judge:
  - taxonomy_error macro-F1 against the teacher's labels (all ten labels),
  - the repository's level-compliance check,
  - feedback length per level next to the teacher's,
and prints a few feedbacks per level beside the teacher's for reading.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.runtime import read_json


def macro_f1(pred, gold):
    scores = []
    for label in ERROR_LABELS:
        tp = sum(label in pred[i] and label in gold[i] for i in gold)
        fp = sum(label in pred[i] and label not in gold[i] for i in gold)
        fn = sum(label not in pred[i] and label in gold[i] for i in gold)
        scores.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Labeled dataset folder (sample_dataset)")
    parser.add_argument("--predictions", required=True, help="pred_task3.json")
    parser.add_argument("--show", type=int, default=2, help="Feedbacks to print per level")
    args = parser.parse_args()
    samples = {s["sample_id"]: s for s in load_samples(args.input, task="task3")}
    rows = {r["sample_id"]: r["output"] for r in read_json(args.predictions)}
    ids = sorted(set(rows) & set(samples))
    gold = {i: set(samples[i].get("error_labels") or []) for i in ids}
    pred = {i: set(rows[i].get("error_labels") or rows[i].get("taxonomy_error") or []) for i in ids}
    by_level = defaultdict(list)
    for i in ids:
        by_level[samples[i]["feedback_level"]].append(i)
    report = {
        "samples": len(ids),
        "label_macro_f1": round(macro_f1(pred, gold), 4),
        "compliance_pass_rate": round(sum(bool((rows[i].get("compliance") or {}).get("pass")) for i in ids) / len(ids), 4),
        "fallbacks": sum(bool(rows[i].get("fallback")) for i in ids),
        "chars_per_level": {
            f"L{level}": {"model": sum(len(rows[i]["feedback"]) for i in group) // len(group),
                          "teacher": sum(len(samples[i].get("feedback") or "") for i in group) // len(group)}
            for level, group in sorted(by_level.items())
        },
    }
    print(json.dumps(report, ensure_ascii=False))
    for level, group in sorted(by_level.items()):
        for i in group[: args.show]:
            print(f"\n[{i} L{level}] labels model={sorted(pred[i])} teacher={sorted(gold[i])}")
            print("  model  :", rows[i]["feedback"].replace("\n", " ")[:400])
            print("  teacher:", (samples[i].get("feedback") or "").replace("\n", " ")[:400])


if __name__ == "__main__":
    raise SystemExit(main())
