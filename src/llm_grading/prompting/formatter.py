"""Format normalized task inputs and parse structured model responses.

Formatting has two jobs: make the prompt readable to the model and make the
allowed fields auditable. A Task 1 formatter should have no way to silently
insert the reference feedback field.
"""

from typing import Any, Mapping


def format_prompt(template: str, values: Mapping[str, Any]) -> str:
    """Fill a versioned template from an explicit task input dictionary."""
    if not isinstance(template, str):
        raise TypeError("Prompt template must be a string")
    try:
        return template.format_map(dict(values))
    except KeyError as error:
        raise ValueError(f"Prompt template requested unavailable field: {error.args[0]}") from error
