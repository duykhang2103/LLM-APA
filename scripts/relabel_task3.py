"""Refresh a Task 3 challenge submission without calling a model.

1. taxonomy_error is replaced by the labels of a (better) Task 2 submission.
2. Level 1 feedback is chosen by code instead of the LLM, because Level 1 may
   only praise or say the work still has problems (label_space.json); the LLM
   sometimes named causes or functions there. No confirmed label -> "Tốt.",
   otherwise -> "Bài còn lỗi, em kiểm tra lại nhé."
3. Level 2 feedback loses the clauses that advise a fix ("em cần thêm ...",
   "em hãy xem lại ..."): Level 2 states location and cause only. The
   diagnosis before those clauses is kept.
Level 3-4 feedback is kept as it is.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.runtime import read_json, write_json
from tasks.task3_feedback.compliance import strip_fix_advice

PRAISE, PROBLEM = "Tốt.", "Bài còn lỗi, em kiểm tra lại nhé."


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task3", required=True, help="Task 3 challenge submission (taxonomy_error + feedback)")
    parser.add_argument("--levels", required=True, help="Internal pred_task3.json that carries feedback_level")
    parser.add_argument("--task2", required=True, help="Task 2 challenge submission whose labels to use")
    parser.add_argument("--output", required=True)
    parser.add_argument("--show", action="store_true", help="Print every Level 2 feedback that was cut")
    args = parser.parse_args()
    levels = {r["sample_id"]: r["output"]["feedback_level"] for r in read_json(args.levels)}
    labels = {r["sample_id"]: r["output"]["taxonomy_error"] for r in read_json(args.task2)}
    rows, stats = [], {"labels_changed": 0, "level1_rewritten": 0, "level2_advice_cut": 0}
    for row in read_json(args.task3):
        sid, old = row["sample_id"], row["output"]
        new = labels[sid]
        stats["labels_changed"] += sorted(new) != sorted(old["taxonomy_error"])
        out = {"taxonomy_error": new, "feedback": old["feedback"]}  # key order of the accepted upload
        if levels[sid] == 1:
            text = PROBLEM if new else PRAISE
            stats["level1_rewritten"] += out["feedback"].strip() != text
            out["feedback"] = text
        elif levels[sid] == 2:
            cut = strip_fix_advice(out["feedback"])
            if cut != out["feedback"]:
                stats["level2_advice_cut"] += 1
                if args.show:
                    print(f"[{sid}]\n  before: {out['feedback']}\n  after : {cut}")
                out["feedback"] = cut
        rows.append({"sample_id": sid, "output": out})
    write_json(args.output, rows)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    raise SystemExit(main())
