"""Load and merge base/task YAML configuration files."""

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Describe the future config-loader entry point."""
    # TODO: Parse YAML, validate required sections, and preserve config provenance.
    raise NotImplementedError("Config loading is not implemented in the scaffold.")
