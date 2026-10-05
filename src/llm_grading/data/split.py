"""Seeded code-group splits; original annotations are never rewritten."""

import hashlib
import random
from collections import defaultdict


def normalized_code(code: str) -> str:
    return " ".join(code.replace("\r\n", "\n").replace("\r", "\n").split())


def code_hash(code: str) -> str:
    return hashlib.sha256(normalized_code(code).encode("utf-8")).hexdigest()


def check_split_overlap(train, validation):
    ids = {s["sample_id"] for s in train}
    hashes = {code_hash(s.get("code", "")) for s in train}
    conflicts = [
        s["sample_id"]
        for s in validation
        if s["sample_id"] in ids or code_hash(s.get("code", "")) in hashes
    ]
    if conflicts:
        raise ValueError(
            f"Train/validation overlap (ID or normalized code): {conflicts}"
        )


def create_splits(samples, seed, validation_fraction=0.2, test_fraction=0.0):
    if (
        not 0 <= validation_fraction < 1
        or not 0 <= test_fraction < 1
        or validation_fraction + test_fraction >= 1
    ):
        raise ValueError("Split fractions must be nonnegative and sum to less than one")
    ids = [s["sample_id"] for s in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be unique before splitting")
    groups = defaultdict(list)
    for s in samples:
        if not s.get("code", "").strip():
            raise ValueError(f"Missing code for {s['sample_id']}")
        groups[code_hash(s["code"])].append(s)
    # Group before stratifying so copies across exams cannot leak.
    strata = defaultdict(list)
    for digest in sorted(groups):
        group = groups[digest]
        key = tuple(sorted({s.get("problem_type", "unknown") for s in group}))
        strata[key].append(group)
    rng = random.Random(seed)
    result = {"train": [], "val": [], "test": []}
    for key in sorted(strata):
        bucket = strata[key]
        rng.shuffle(bucket)
        n = len(bucket)
        val_n = (
            min(n - 1, max(1, round(n * validation_fraction)))
            if validation_fraction and n > 1
            else 0
        )
        test_n = (
            min(n - val_n - 1, max(1, round(n * test_fraction)))
            if test_fraction and n - val_n > 1
            else 0
        )
        for i, group in enumerate(bucket):
            split = "val" if i < val_n else ("test" if i < val_n + test_n else "train")
            result[split].extend(s["sample_id"] for s in group)
    return result
