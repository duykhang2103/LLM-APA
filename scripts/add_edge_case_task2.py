"""Add "Lỗi edge case" to a Task 2 challenge submission.

The list prompt almost never predicts this label. A submission gets it when
either source fires:
  - the per-label run said Yes (calibrated predictions, error_labels), or
  - the generated tests show the definition: every normal/example test passes
    but at least one edge test fails.
Training data has no "Lỗi edge case" label, so this cannot be scored offline;
with macro-F1 over the ten labels, a label absent from the reference keeps F1 0
whatever is predicted, so the change can only help that label's term.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.runtime import read_json, write_json

EDGE = "Lỗi edge case"


def edge_rule(report):
    """True when all normal/example tests pass and some edge test fails."""
    cats = report.get("by_category") or {}
    normal = [cats.get(c, [0, 0]) for c in ("normal", "example")]
    edge = cats.get("edge", [0, 0])
    return sum(t for _, t in normal) > 0 and all(p == t for p, t in normal) and edge[1] > edge[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submission", required=True, help="Task 2 challenge submission (taxonomy_error)")
    parser.add_argument("--per-label", required=True, help="Calibrated per-label predictions (error_labels)")
    parser.add_argument("--tests", required=True, help="tests.json written by run_tests.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    per_label = {r["sample_id"]: set(r["output"].get("error_labels") or []) for r in read_json(args.per_label)}
    reports = read_json(args.tests)["reports"]
    rows, added = [], {"per_label": 0, "rule": 0}
    for row in read_json(args.submission):
        sid, labels = row["sample_id"], set(row["output"]["taxonomy_error"])
        if EDGE not in labels:
            if EDGE in per_label.get(sid, ()):
                labels.add(EDGE)
                added["per_label"] += 1
            elif sid in reports and edge_rule(reports[sid]):
                labels.add(EDGE)
                added["rule"] += 1
        rows.append({"sample_id": sid, "output": {"taxonomy_error": [l for l in ERROR_LABELS if l in labels]}})
    write_json(args.output, rows)
    total = sum(EDGE in r["output"]["taxonomy_error"] for r in rows)
    print(json.dumps({"added": added, "edge_case_total": total, "samples": len(rows)}, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
