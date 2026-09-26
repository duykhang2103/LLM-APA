"""Load raw challenge records and resolve source-code file references."""

from pathlib import Path

from .schema import NormalizedSample


def load_samples(source: str | Path) -> list[NormalizedSample]:
    """Describe the future normalized dataset-loading entry point."""
    # TODO: Read the official JSON files and resolve submissions/<problem>/<file>.
    raise NotImplementedError("Dataset loading is not implemented in the scaffold.")
