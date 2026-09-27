"""Load and merge base/task YAML configuration files.

Config should answer: which task, data version, split, model revision, prompt
version, seed, generation settings, training settings, and output directory
were used? Avoid placing those values directly in task Python files.
"""

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML config and return a validated dictionary."""
    # TODO: Parse YAML, validate required sections, and preserve config provenance.
    raise NotImplementedError("Config loading is not implemented in the scaffold.")
