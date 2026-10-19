"""Merge shard prediction files (from predict.py --shard) into one validated file.

With --input, the merged file must cover exactly that dataset's sample IDs
and is written in its order.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.runtime import read_json, write_json
from llm_grading.validation.predictions import validate_prediction_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, choices=["task1", "task2", "task3"])
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--input", help="Dataset the shards were predicted from")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    merged = {}
    for path in args.shards:
        for row in read_json(path):
            if row.get("sample_id") in merged:
                raise ValueError(f"{row.get('sample_id')} appears in more than one shard")
            merged[row.get("sample_id")] = row
    records = list(merged.values())
    if args.input:
        ids = [s["sample_id"] for s in load_samples(args.input, task=args.task)]
        missing, extra = set(ids) - set(merged), set(merged) - set(ids)
        if missing or extra:
            raise ValueError(f"Shards do not match the dataset: missing {sorted(missing)[:5]}..., extra {sorted(extra)[:5]}")
        records = [merged[i] for i in ids]
    errors = validate_prediction_records(records, args.task)
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, records)
    print(f"Merged {len(records)} predictions from {len(args.shards)} shards: {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
