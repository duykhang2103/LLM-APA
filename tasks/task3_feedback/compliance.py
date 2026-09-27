"""Centralized Task 3 feedback-level compliance checker boundary.

Example result for an acceptable Level 1 response::

    {"pass": True, "violations": []}

Example result for a response that contains a complete fix::

    {"pass": False, "violations": ["complete_solution_at_level_1"]}

Start with deterministic checks for code fences, long code-like lines,
solution markers, and level-specific forbidden content. Keep false positives
and false negatives in a review file for later improvement.
"""

from typing import Any


def check_compliance(feedback: str, level: int) -> dict[str, Any]:
    """Return ``pass`` and a list of policy violations for one response."""
    # TODO: Detect prohibited complete solutions/code at restricted levels.
    raise NotImplementedError("Task 3 compliance checking is not implemented.")
