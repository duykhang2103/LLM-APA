"""Format normalized task inputs and parse structured model responses."""

from typing import Any, Mapping


def format_prompt(template: str, values: Mapping[str, Any]) -> str:
    """Describe the future safe prompt formatter."""
    # TODO: Make field inclusion explicit so Task 1/2 feedback cannot leak.
    raise NotImplementedError("Prompt formatting is not implemented in the scaffold.")
