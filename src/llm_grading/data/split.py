"""Create and persist reproducible train/validation/test split identifiers.

New contributors must decide what is grouped before running experiments:
submission, student, problem, exam, or another unit. A random split can leak
near-duplicate code or problem templates, so the strategy belongs in the
dataset manifest and experiment metadata.
"""

from typing import Any
import random

from .schema import NormalizedSample


def create_splits(samples: list[NormalizedSample], seed: int) -> dict[str, list[str]]:
    """Return split-name to sample-ID mappings using one documented strategy.

    Example return shape::

        {"train": ["synthetic-001"], "val": [], "test": []}
    """
    ids = [str(sample["sample_id"]) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("Sample IDs must be unique before splitting")
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    if len(shuffled) < 3:
        return {"train": shuffled, "val": [], "test": []}
    val_count = max(1, round(len(shuffled) * 0.1))
    test_count = max(1, round(len(shuffled) * 0.1))
    return {
        "train": shuffled[: len(shuffled) - val_count - test_count],
        "val": shuffled[len(shuffled) - val_count - test_count : len(shuffled) - test_count],
        "test": shuffled[len(shuffled) - test_count :],
    }
