"""Resolve repository, data, output, checkpoint, and experiment paths.

Paths should be derived from the repository/config root rather than from a
developer's personal current directory. Private data and large checkpoints
must remain outside tracked source paths.
"""

from pathlib import Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root used by shared CLI scripts."""
    # TODO: Resolve paths without depending on the caller's current directory.
    raise NotImplementedError("Path resolution is not implemented in the scaffold.")
