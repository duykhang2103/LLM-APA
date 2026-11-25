"""Score frozen Task 2 verifier arms against a reviewed diagnostic panel."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from llm_grading.evaluation.defined_decisions import evaluate_defined  # noqa: E402
from llm_grading.runtime import read_json, write_json  # noqa: E402


def prediction_map(rows, extract):
    predictions = {row["sample_id"]: extract(row) for row in rows}
    if len(predictions) != len(rows):
        raise ValueError("Duplicate prediction IDs")
    return predictions


def summarize(metrics):
    tp = sum(row["tp"] for row in metrics["per_class"].values())
    fp = sum(row["fp"] for row in metrics["per_class"].values())
    fn = sum(row["fn"] for row in metrics["per_class"].values())
    denominator = 2 * tp + fp + fn
    return {**metrics, "micro_f1": 2 * tp / denominator if denominator else 0.0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--untuned", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh evaluation output path")
    cohort = read_json(args.cohort)
    judgments = read_json(args.judgments)["judgments"]
    baseline = prediction_map(
        read_json(args.baseline), lambda row: row["output"]["taxonomy_error"]
    )
    reports = {}
    for arm in ("untuned", "tuned"):
        rows = read_json(getattr(args, arm))
        for variant in ("broad", "full"):
            predictions = prediction_map(rows, lambda row: row[variant])
            paired = evaluate_defined(
                cohort, {"A": baseline, "B": predictions}, judgments
            )
            metrics = {
                key: summarize(value) for key, value in paired["metrics"].items()
            }
            omitted = {}
            for family in sorted({row["family"] for row in cohort}):
                subset = [row for row in cohort if row["family"] != family]
                ids = [row["sample_id"] for row in subset]
                family_metrics = evaluate_defined(
                    subset,
                    {
                        key: {sid: mapping[sid] for sid in ids}
                        for key, mapping in (("A", baseline), ("B", predictions))
                    },
                    {sid: judgments[sid] for sid in ids},
                )["metrics"]
                a, b = (family_metrics[key]["macro_f1"] for key in ("A", "B"))
                omitted[family] = b - a if a is not None and b is not None else None
            reports[f"{arm}_{variant}"] = {
                "baseline": metrics["A"],
                "candidate": metrics["B"],
                "family_omission_macro_deltas": omitted,
                "paired": paired["paired"],
            }
    write_json(args.output, reports)
    for name, report in reports.items():
        print(name, "macro-F1", report["candidate"]["macro_f1"])


if __name__ == "__main__":
    main()
