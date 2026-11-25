"""Validate and package the existing contract without inspecting private targets."""

import argparse
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.challenge import export_challenge_predictions
from llm_grading.validation.predictions import validate_predictions
from llm_grading.validation.task3 import task3_feedback_warnings


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--format", choices=["challenge", "internal"], default="challenge")
    p.add_argument("--require-task3-taxonomy", action="store_true",
                   help="Require the release Task 3 taxonomy_error plus feedback contract")
    p.add_argument("--allow-feedback-warnings", action="store_true",
                   help="Preserve Task 3 feedback-policy warnings without blocking export")
    p.add_argument("--expected-input", help="Require exact sample-ID coverage of this task input")
    p.add_argument("--allow-partial-submission", action="store_true",
                   help="Export valid saved records and report missing expected sample IDs")
    args = p.parse_args()
    if args.task == "task3" and args.format == "challenge" and args.allow_feedback_warnings and not args.expected_input:
        p.error("--expected-input is required for warning-aware Task 3 challenge submissions")
    if args.allow_partial_submission and not args.expected_input:
        p.error("--expected-input is required with --allow-partial-submission")
    errors = validate_predictions(
        args.input, args.task, require_task3_taxonomy=args.require_task3_taxonomy
    )
    if errors:
        raise ValueError(str(errors))
    path = Path(args.output)
    if path.resolve() == Path(args.input).resolve():
        raise ValueError("Export to a separate path to preserve internal predictions")
    records = read_json(args.input)
    coverage = None
    if args.expected_input:
        expected = load_samples(args.expected_input, task=args.task)
        expected_ids = [r["sample_id"] for r in expected]
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("Expected input contains duplicate sample IDs")
        submitted_ids = {r["sample_id"] for r in records}
        if submitted_ids - set(expected_ids):
            raise ValueError("Predictions contain unexpected sample IDs")
        missing = [sid for sid in expected_ids if sid not in submitted_ids]
        if missing and not args.allow_partial_submission:
            raise ValueError("Prediction ID coverage does not match the complete expected input")
        coverage = {
            "expected_count": len(expected_ids), "submitted_count": len(records),
            "missing_count": len(missing), "missing_sample_ids": missing,
            "complete": not missing,
        }
    warnings = task3_feedback_warnings(records) if args.task == "task3" else []
    warning_path = path.with_suffix(".warnings.json")
    coverage_path = path.with_suffix(".coverage.json")
    protected_inputs = {Path(args.input).resolve()}
    if args.expected_input:
        protected_inputs.add(Path(args.expected_input).resolve())
    report_paths = ([warning_path] if warnings else []) + ([coverage_path] if coverage is not None else [])
    if path.resolve() in protected_inputs or any(report.resolve() in protected_inputs for report in report_paths):
        raise ValueError("Export and warning reports must not overwrite prediction or expected-input files")
    if args.expected_input:
        expected_root = Path(args.expected_input).resolve()
        if expected_root.is_dir() and any(candidate.resolve().is_relative_to(expected_root) for candidate in [path, *report_paths]):
            raise ValueError("Export and report paths must be outside the expected-input directory")
    if args.format == "challenge":
        records = export_challenge_predictions(
            records, args.task, require_task3_taxonomy=args.require_task3_taxonomy,
            allow_feedback_warnings=args.allow_feedback_warnings,
        )
    if warnings:
        write_json(warning_path, warnings)
        print(f"Feedback warnings: {len(warnings)}; saved to {warning_path}")
    if coverage is not None:
        write_json(coverage_path, coverage)
        print(f"Coverage: {coverage['submitted_count']}/{coverage['expected_count']}; missing IDs: {coverage['missing_count']}; see {coverage_path}")
    if path.suffix == ".json":
        write_json(path, records)
        print(f"Exported validated predictions: {path}")
        return 0
    if path.suffix != ".zip":
        raise ValueError("Output must end in .json or .zip")
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "predictions.json", json.dumps(records, ensure_ascii=False, indent=2)
        )
    print(f"Packaged validated predictions: {path}")


if __name__ == "__main__":
    main()
