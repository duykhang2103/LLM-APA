"""Build and persist a retrieval index over permitted training examples."""

from typing import Any, Iterable, Mapping


def build_index(records: Iterable[Mapping[str, Any]]) -> object:
    """Describe the future retrieval-index builder."""
    # TODO: Define embeddings, storage, versioning, and leakage-safe records.
    raise NotImplementedError("Retrieval indexing is not implemented in the scaffold.")
