"""Resolve repository, data, output, checkpoint, and experiment paths."""

from pathlib import Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Describe the future repository-root resolver."""
    # TODO: Resolve paths without depending on the caller's current directory.
    raise NotImplementedError("Path resolution is not implemented in the scaffold.")
