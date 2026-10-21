"""Run generated test suites (scripts/generate_tests.py) on every submission.

Writes {"suites": {...}, "reports": {sample_id: report}}. Samples whose exam
has no suite get no report. On a labeled dataset it also prints how the pass
rate relates to the teacher's scores, which is how a suite is judged useful.
"""

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.evidence import default_compiler
from llm_grading.data.loader import load_samples
from llm_grading.data.testing import run_suite
from llm_grading.runtime import read_json, write_json


def spearman(xs, ys):
    def ranks(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                result[order[k]] = (i + j) / 2
            i = j + 1
        return result

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    var = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return round(cov / var, 3) if var else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Dataset folder")
    parser.add_argument("--suites", required=True, help="Folder of <exam_id>.json suites")
    parser.add_argument("--output", required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=2.0, help="Seconds per test run")
    args = parser.parse_args()

    samples = load_samples(args.input, task="task1")
    suites = {}
    for sample in samples:
        exam = sample["problem_id"]
        path = Path(args.suites, f"{exam}.json")
        if exam not in suites:
            suites[exam] = read_json(path) if path.exists() else None
    missing = sorted(k for k, v in suites.items() if v is None)
    if missing:
        print(f"No suite for {missing}; those samples get no test report")
    compiler = default_compiler(native=True)
    data_dir = Path(args.input, "data_files")
    todo = [s for s in samples if suites[s["problem_id"]]]

    def run(sample):
        return run_suite(sample["code"], suites[sample["problem_id"]], compiler, data_dir, args.timeout)

    with ThreadPoolExecutor(args.jobs) as pool:
        reports = dict(zip((s["sample_id"] for s in todo), pool.map(run, todo)))
    write_json(args.output, {
        "compiler": compiler,
        "suites": {k: len(v["tests"]) for k, v in suites.items() if v},
        "reports": reports,
    })
    for exam in sorted(k for k, v in suites.items() if v):
        rates = [reports[s["sample_id"]]["pass_rate"] for s in todo if s["problem_id"] == exam]
        print(f"{exam}: {len(rates)} submissions, mean pass rate {sum(rates) / len(rates):.2f}, "
              f"all-pass {sum(r == 1 for r in rates)}, zero-pass {sum(r == 0 for r in rates)}")
    labeled = [s for s in todo if s.get("rubric")]
    if labeled:
        print("\nAgreement with teacher scores (only meaningful on labeled data):")
        for exam in sorted({s["problem_id"] for s in labeled}):
            group = [s for s in labeled if s["problem_id"] == exam]
            rates = [reports[s["sample_id"]]["pass_rate"] for s in group]
            totals = [sum(s["rubric"].values()) for s in group]
            logic = [s["rubric"]["logic"] for s in group]
            print(f"  {exam}: spearman(pass_rate, total) = {spearman(rates, totals)}, "
                  f"spearman(pass_rate, logic) = {spearman(rates, logic)}")
            for s, rate in sorted(zip(group, rates), key=lambda pair: -pair[1]):
                rubric = s["rubric"]
                print(f"    {s['sample_id']}: pass {rate:.2f} | teacher total {sum(rubric.values())} "
                      f"logic {rubric['logic']} edge {rubric['edge_case']} io {rubric['io_format']}")
    print(f"Wrote {args.output}")
    print(json.dumps({"samples_with_report": len(reports)}))


if __name__ == "__main__":
    raise SystemExit(main())
