"""Format normalized task inputs and parse structured model responses.

Formatting has two jobs: make the prompt readable to the model and make the
allowed fields auditable. A Task 1 formatter should have no way to silently
insert the reference feedback field.
"""

from typing import Any, Mapping


def format_prompt(template: str, values: Mapping[str, Any]) -> str:
    """Fill a versioned template from an explicit task input dictionary."""
    # TODO: Make field inclusion explicit so Task 1/2 feedback cannot leak.
    raise NotImplementedError("Prompt formatting is not implemented in the scaffold.")
