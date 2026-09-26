"""Load immutable prompt templates by task and version."""

from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Describe the future prompt-file loader."""
    # TODO: Read a versioned prompt file and record its identity in experiments.
    raise NotImplementedError("Prompt loading is not implemented in the scaffold.")
