"""Create and persist reproducible train/validation/test split identifiers."""

from typing import Any

from .schema import NormalizedSample


def create_splits(samples: list[NormalizedSample], seed: int) -> dict[str, list[str]]:
    """Describe the future fixed-seed split entry point."""
    # TODO: Choose and document submission/problem grouping and stratification rules.
    raise NotImplementedError("Dataset splitting is not implemented in the scaffold.")
