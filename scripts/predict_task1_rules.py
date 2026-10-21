"""Task 1 predictions without an LLM: compile evidence + generated-test pass rate.

Samples without a test report keep the rubric from --fallback (e.g. an LLM
prediction file), else get the default 1,1,2,1,1,1 (counted in the output). On labeled data the
metrics are printed.
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

DEFAULT = (1, 1, 2, 1, 1, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--evidence", required=True)
    parser.add_argument("--tests", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fallback", help="Internal Task 1 predictions used where no rule applies")
    args = parser.parse_args()
    samples = load_samples(args.input, task="task1")
    evidence = read_json(args.evidence)["evidence"]
    reports = read_json(args.tests)["reports"]
    fallback = {r["sample_id"]: r["output"] for r in read_json(args.fallback)} if args.fallback else {}
    result, sources = [], Counter()
    for sample in samples:
        sid = sample["sample_id"]
        rubric = rubric_from_evidence(sample, evidence.get(sid), reports.get(sid))
        if rubric is None and sid in fallback:
            rubric, source = {f: fallback[sid][f] for f in FIELDS}, "fallback"
        elif rubric is None:
            # No suite for this exam and no LLM prediction: the most common
            # teacher pattern for compiling code (compiles, logic bug).
            rubric, source = dict(zip(FIELDS, DEFAULT)), "default_no_tests"
        else:
            source = "rule"
        sources[source] += 1
        result.append({"sample_id": sid, "output": {**rubric, "total": sum(rubric.values())}})
    errors = validate_prediction_records(result, "task1")
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, result)
    report = {"sources": dict(sources)}
    if all(s.get("rubric") for s in samples):
        metrics = evaluate_task1(result, samples)
        report["metrics"] = {k: round(metrics[k], 4) for k in ("qwk_total", "mae_total")}
    print(json.dumps(report))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
