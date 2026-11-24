"""Rebuild the defined-d3 submission from its private, source-bound decision rows."""

import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from llm_grading.data.loader import load_samples  # noqa: E402
from llm_grading.decision_diagnostic import labels_from_probabilities  # noqa: E402
from llm_grading.defined_decisions import defined_questions  # noqa: E402
from llm_grading.release_ledger import fingerprint  # noqa: E402
from llm_grading.runtime import read_json, write_json  # noqa: E402


def replay_decisions(samples: list[dict], rows: list[dict]) -> list[dict]:
    source_hashes = {
        sample["sample_id"]: hashlib.sha256(sample["code"].encode("utf-8")).hexdigest()
        for sample in samples
    }
    if len(source_hashes) != len(samples):
        raise ValueError("Duplicate release sample IDs")
    question_hash = fingerprint(defined_questions())
    canonical = {}
    repeats = {}
    for row in rows:
        sample_id = row["sample_id"]
        if not isinstance(row["repeat"], bool):
            raise ValueError("Malformed repeat flag")
        target = repeats if row["repeat"] else canonical
        if sample_id not in source_hashes or sample_id in target:
            raise ValueError("Unknown or duplicate decision sample")
        if (
            row["source_sha256"] != source_hashes[sample_id]
            or row["question_sha256"] != question_hash
        ):
            raise ValueError("Decision source or prompt differs from the release")
        probabilities = row["probabilities_true"]
        labels = labels_from_probabilities(probabilities)
        native = [
            row["answers"][f"label_{index}"]["noul"]
            for index in range(len(probabilities))
        ]
        native_labels = labels_from_probabilities(native)
        if (
            probabilities != native
            or labels != native_labels
            or labels != row["labels"]
        ):
            raise ValueError(
                "Decision labels or probabilities differ from the native answers"
            )
        target[sample_id] = labels
    if canonical.keys() != source_hashes.keys():
        raise ValueError("Decision rows do not cover the whole release")
    if any(labels != canonical[sample_id] for sample_id, labels in repeats.items()):
        raise ValueError("Repeat label instability")
    return [
        {"sample_id": sample_id, "output": {"taxonomy_error": canonical[sample_id]}}
        for sample_id in source_hashes
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a new output path for the candidate")
    samples = load_samples(args.release, task="task2", preserve_source_bytes=True)
    rows = read_json(args.decisions)
    if (
        len(samples) != 328
        or len(rows) != 332
        or sum(row["repeat"] is True for row in rows) != 4
    ):
        raise ValueError("Expected 328 official samples and four repeated decisions")
    write_json(args.output, replay_decisions(samples, rows))
    print(f"Wrote {len(samples)} Task 2 predictions")


if __name__ == "__main__":
    main()
