"""Common evaluation helpers and result containers."""

from typing import Any, Iterable, Mapping


def summarize_results(metrics: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Describe the future common evaluation summary."""
    # TODO: Normalize metric output and attach experiment metadata.
    raise NotImplementedError("Common evaluation is not implemented in the scaffold.")
