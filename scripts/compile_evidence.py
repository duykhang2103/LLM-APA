"""Compile every submission of a dataset and save compile evidence JSON.

If the dataset has teacher rubrics (training data), print how well the strict
and relaxed compile results agree with the teacher's `compilable`.
"""

import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.evidence import compile_evidence, default_compiler
from llm_grading.data.loader import load_samples
from llm_grading.runtime import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    samples = load_samples(args.input, task="task1")
    compiler = default_compiler()
    with ThreadPoolExecutor(args.jobs) as pool:
        results = list(pool.map(lambda s: compile_evidence(s["code"], compiler), samples))
    evidence = {s["sample_id"]: r for s, r in zip(samples, results)}
    write_json(args.output, {"compiler": compiler, "evidence": evidence})
    summary = Counter((r["strict"], r["relaxed"]) for r in results)
    print(json.dumps({"compiler": compiler, "(strict, relaxed) counts": {str(k): v for k, v in summary.items()}}))
    if all(s.get("rubric") for s in samples):
        for mode in ("strict", "relaxed"):
            pairs = Counter((s["rubric"]["compilable"], int(evidence[s["sample_id"]][mode])) for s in samples)
            agree = sum(n for (t, p), n in pairs.items() if t == p) / len(samples)
            print(f"{mode}: agreement with teacher compilable {agree:.3f}; (teacher, evidence) counts {dict(pairs)}")
            for s in samples:
                if s["rubric"]["compilable"] != int(evidence[s["sample_id"]][mode]):
                    print("   disagree", s["sample_id"], "teacher", s["rubric"]["compilable"], evidence[s["sample_id"]]["errors"][:1])
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
