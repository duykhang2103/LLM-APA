"""Select comparable examples for retrieval-augmented prompting."""

from typing import Any, Mapping


def retrieve(index: object, query: Mapping[str, Any], limit: int) -> list[Mapping[str, Any]]:
    """Describe the future retrieval query entry point."""
    # TODO: Return only examples allowed by the selected split and policy.
    raise NotImplementedError("Retrieval is not implemented in the scaffold.")
