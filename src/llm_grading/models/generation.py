"""Shared text-generation utilities for prompting and feedback tasks.

Generation settings such as temperature, max tokens, stop sequences, and
sampling must come from config and be recorded in experiment metadata.
"""

from typing import Any, Mapping


def generate_text(model: object, prompt: str, settings: Mapping[str, Any]) -> str:
    """Generate raw text while preserving the exact prompt/settings record."""
    # TODO: Apply the configured decoding parameters and return raw text.
    raise NotImplementedError("Text generation is not implemented in the scaffold.")
