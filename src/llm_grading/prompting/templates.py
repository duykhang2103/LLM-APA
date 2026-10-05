"""Load immutable prompt templates by task and version.

Prompts are source code for experiments. Keep them in versioned files, review
the input whitelist, and create a new version rather than changing a prompt
after an official run.

Minimal template example::

    SYSTEM: Return only the requested JSON fields.
    PROBLEM: {problem_statement}
    CODE: {code}
    OUTPUT_SCHEMA: {output_schema}
"""

from pathlib import Path


def load_prompt(path: str | Path) -> str:
    """Read one immutable prompt file and return its text."""
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    return prompt_path.read_text(encoding="utf-8")
