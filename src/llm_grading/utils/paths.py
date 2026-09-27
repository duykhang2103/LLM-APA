"""Resolve repository, data, output, checkpoint, and experiment paths.

Paths should be derived from the repository/config root rather than from a
developer's personal current directory. Private data and large checkpoints
must remain outside tracked source paths.
"""

from pathlib import Path


def resolve_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root used by shared CLI scripts."""
    candidate = Path(start or __file__).resolve()
    if candidate.is_file():
        candidate = candidate.parent
    for directory in (candidate, *candidate.parents):
        if (directory / "pyproject.toml").exists():
            return directory
    return Path.cwd().resolve()


def resolve_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    """Resolve a relative path against the repository root."""
    value = Path(path)
    return value if value.is_absolute() else resolve_project_root(root) / value
