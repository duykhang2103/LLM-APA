"""Rebuild the Task 1 compile-only candidate from private, source-bound rechecks."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_grading.data.loader import load_samples  # noqa: E402
from llm_grading.data.schema import validate_rubric  # noqa: E402
from llm_grading.runtime import read_json, write_json  # noqa: E402

EXAMS = {"Class03-Midterm-2": 9, "Class03-Final-1": 5}
REASONS = {"source_diagnostic", "undefined_reference"}


def apply_rechecks(samples: list[dict], baseline: list[dict], checks: list[dict]) -> list[dict]:
    by_id = {sample["sample_id"]: sample for sample in samples}
    baseline_ids = [row["sample_id"] for row in baseline]
    if len(by_id) != len(samples) or len(baseline_ids) != len(by_id) or set(baseline_ids) != by_id.keys():
        raise ValueError("Baseline must cover the release exactly once")
    edits = {}
    for check in checks:
        sample_id = check["sample_id"]
        sample = by_id.get(sample_id)
        if sample is None or sample_id in edits:
            raise ValueError("Unknown or duplicate recheck sample")
        build = check["current"]
        if (
            build["phase"] != "link"
            or build["status"] != "compile_failure"
            or build["profile"] != "full_program"
            or build["standard"] != "c++11"
            or build["reproducible"] is not True
            or build["reason"] not in REASONS
            or build["reason"] != check["recorded_reason"]
            or build["source_sha256"] != hashlib.sha256(sample["code"].encode("utf-8")).hexdigest()
            or build["statement_sha256"]
            != hashlib.sha256(sample["problem_statement"].encode("utf-8")).hexdigest()
        ):
            raise ValueError("Recheck does not confirm a source-bound standalone build failure")
        edits[sample_id] = build

    candidate = []
    for row in baseline:
        output = row["output"]
        rubric = output["rubric"]
        if "total" in rubric:
            raise ValueError("Task 1 rubric contains an unexpected total field")
        validate_rubric({**rubric, "total": output["total_score"]})
        if row["sample_id"] in edits:
            if rubric["compilable"] != 1:
                raise ValueError("Rechecks apply to baseline compilable=1 only")
            rubric = {**rubric, "compilable": 0}
            row = {**row, "output": {**output, "rubric": rubric, "total_score": sum(rubric.values())}}
        candidate.append(row)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--build-receipt", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path for the candidate")
    samples = load_samples(args.release, task="task1", preserve_source_bytes=True)
    baseline = read_json(args.baseline)
    checks = [check for path in args.build_receipt for check in read_json(path)["checks"]]
    candidate = apply_rechecks(samples, baseline, checks)
    reviewed = {check["sample_id"] for check in checks}
    exams = {}
    for sample in samples:
        if sample["sample_id"] in reviewed:
            exam_id = sample["problem_id"]
            exams[exam_id] = exams.get(exam_id, 0) + 1
    if exams != EXAMS:
        raise ValueError("Expected exactly the two reviewed exam groups")
    write_json(args.output, candidate)
    print(f"Wrote {len(candidate)} rows with {len(checks)} compile-only changes")


if __name__ == "__main__":
    main()
