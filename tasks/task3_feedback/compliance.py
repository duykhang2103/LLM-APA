"""Deterministic checks for the Task 3 feedback-level policy."""

from typing import Any


def check_compliance(feedback: str, level: int) -> dict[str, Any]:
    """Return ``pass`` and policy violation names for one response."""
    if not isinstance(feedback, str) or not feedback.strip():
        return {"pass": False, "violations": ["empty_feedback"]}
    if level not in {1, 2, 3, 4}:
        raise ValueError("Feedback level must be 1, 2, 3, or 4")
    violations: list[str] = []
    lowered = feedback.lower()
    restricted = level in {1, 2}
    if restricted and "```" in feedback:
        violations.append("code_fence")
    solution_markers = (
        "#include",
        "int main(",
        "int main()",
        "complete solution",
        "here is the corrected code",
        "đáp án hoàn chỉnh",
    )
    if restricted and any(marker in lowered for marker in solution_markers):
        violations.append("complete_solution_at_restricted_level")
    if restricted and any(line.strip().endswith(";") for line in feedback.splitlines()):
        violations.append("code_like_statement_at_restricted_level")
    return {"pass": not violations, "violations": violations}
