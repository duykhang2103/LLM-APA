"""Create and persist reproducible train/validation/test split identifiers.

New contributors must decide what is grouped before running experiments:
submission, student, problem, exam, or another unit. A random split can leak
near-duplicate code or problem templates, so the strategy belongs in the
dataset manifest and experiment metadata.
"""

from typing import Any

from .schema import NormalizedSample


def create_splits(samples: list[NormalizedSample], seed: int) -> dict[str, list[str]]:
    """Return split-name to sample-ID mappings using one documented strategy.

    Example return shape::

        {"train": ["synthetic-001"], "val": [], "test": []}
    """
    # TODO: Choose and document submission/problem grouping and stratification rules.
    raise NotImplementedError("Dataset splitting is not implemented in the scaffold.")
