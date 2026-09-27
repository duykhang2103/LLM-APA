"""Select comparable examples for retrieval-augmented prompting.

Start with a simple, inspectable retrieval strategy such as problem type plus
text/code similarity. Always return selected-example IDs and metadata so the
prompt and experiment log can be audited.
"""

from typing import Any, Mapping


def retrieve(index: object, query: Mapping[str, Any], limit: int) -> list[Mapping[str, Any]]:
    """Return at most ``limit`` permitted training examples for one query."""
    # TODO: Return only examples allowed by the selected split and policy.
    raise NotImplementedError("Retrieval is not implemented in the scaffold.")
