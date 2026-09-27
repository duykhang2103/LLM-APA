"""Build and persist a retrieval index over permitted training examples.

The index must be built from an approved split. Store its dataset version,
split version, embedding model, and index configuration so another teammate
can rebuild it. Task 1/2 retrieval records must not expose reference feedback.
"""

from typing import Any, Iterable, Mapping


def build_index(records: Iterable[Mapping[str, Any]]) -> object:
    """Create an index from leakage-safe records selected by the caller."""
    # TODO: Define embeddings, storage, versioning, and leakage-safe records.
    raise NotImplementedError("Retrieval indexing is not implemented in the scaffold.")
