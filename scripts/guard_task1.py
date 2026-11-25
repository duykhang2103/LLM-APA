"""Make an LLM Task 1 prediction agree with the real compile result.

The lecturer confirmed that the pipeline compile check is trusted for
"compilable". Two corrections, using only the compile result and the generated
tests (never teacher labels):
  1. compilable follows the compile result (relaxed C++11 build).
  2. When the model scored a file that compiles as not compiling, the model
     misread the evidence, so that sample takes the rule-based rubric
     (compile result + test pass rate, tasks/task1_grading/evidence_rules.py).
Samples without compile evidence keep the model's rubric. On labeled data the
metrics of the input and the output are printed.
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
from tasks.task1_grading.evidence_rules import FIELDS, rubric_from_evidence


def guard(rubric, sample, evidence, report):
    """(new rubric, reason) for one model rubric; reason None when unchanged."""
    if evidence is None:
        return rubric, None
    compiles = bool(evidence["relaxed"])
    if compiles and rubric["compilable"] == 0:
        rule = rubric_from_evidence(sample, evidence, report)
        if rule is not None:
            return rule, "compiling_file_scored_broken"
    if rubric["compilable"] != int(compiles):
        return {**rubric, "compilable": int(compiles)}, "compilable_from_compile_result"
    return rubric, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Dataset folder")
    parser.add_argument("--predictions", required=True, help="LLM Task 1 predictions (pred_task1.json)")
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    samples = {s["sample_id"]: s for s in load_samples(args.input, task="task1")}
    evidence = read_json(args.evidence)["evidence"]
    reports = read_json(args.tests)["reports"]
    predictions = read_json(args.predictions)
    result, reasons = [], Counter()
    for row in predictions:
        sid = row["sample_id"]
        rubric = {f: int(row["output"][f]) for f in FIELDS}
        rubric, reason = guard(rubric, samples[sid], evidence.get(sid), reports.get(sid))
        reasons[reason or "unchanged"] += 1
        result.append({"sample_id": sid, "output": {**rubric, "total": sum(rubric.values())}})
    errors = validate_prediction_records(result, "task1")
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, result)
    report = {"changes": dict(reasons)}
    ordered = [samples[r["sample_id"]] for r in result]
    if len(ordered) > 1 and all(s.get("rubric") for s in ordered):  # smoke runs have too few samples
        try:
            before, after = evaluate_task1(predictions, ordered), evaluate_task1(result, ordered)
            report["qwk_total"] = {"model": round(before["qwk_total"], 4), "guarded": round(after["qwk_total"], 4)}
        except (ValueError, ZeroDivisionError):
            pass
    print(json.dumps(report))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
