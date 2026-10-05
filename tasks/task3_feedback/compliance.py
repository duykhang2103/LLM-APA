"""Deterministic checks for the Task 3 feedback-level policy."""

import re
import unicodedata
from typing import Any


def check_compliance(feedback: str, level: int) -> dict[str, Any]:
    """Return ``pass`` and policy violation names for one response."""
    if not isinstance(feedback, str) or not feedback.strip():
        return {"pass": False, "violations": ["empty_feedback"]}
    if isinstance(level, bool) or level not in {1, 2, 3, 4}:
        raise ValueError("Feedback level must be 1, 2, 3, or 4")
    violations: list[str] = []
    lowered = "".join(
        c
        for c in unicodedata.normalize("NFD", feedback.lower())
        if unicodedata.category(c) != "Mn"
    ).replace("đ", "d")
    restricted = level in {1, 2}
    if restricted and "```" in feedback:
        violations.append("code_fence")
    solution_markers = (
        "#include",
        "int main(",
        "int main()",
        "complete solution",
        "here is the corrected code",
        "dap an hoan chinh",
    )
    complete = any(marker in lowered for marker in solution_markers) or bool(
        re.search(r"\b(?:int|bool|void|long|double)\s+\w+\s*\([^)]*\)\s*\{", lowered)
    )
    if level < 4 and complete:
        violations.append("complete_solution_at_restricted_level")
    # A semicolon in Vietnamese prose is not evidence of a code statement.
    if restricted and re.search(
        r"(?:\b\w+(?:->\w+|\.\w+)*\s*(?:=(?!=)|\+=|-=)|\b(?:return|cout\s*<<|cin\s*>>))[^\n;]*;",
        lowered,
    ):
        violations.append("code_like_statement_at_restricted_level")
    if restricted and re.search(
        r"\b(?:chi can|thay .{0,60} bang|bo chu|them .{0,60} vao|sua .{0,40} thanh)\b",
        lowered,
    ):
        violations.append("concrete_fix_at_restricted_level")
    if level == 1 and re.search(
        r"\b(?:dong (?:so )?\d+|line \d+|p[1-4]|loi (?:logic|bien dich|vong lap|ham|nhap/xuat)|"
        r"thieu|chua khoi tao|khong co ham|sai cong thuc|comment|vi .{1,80} nen)\b",
        lowered,
    ):
        violations.append("specific_diagnosis_at_level_1")
    return {
        "pass": not violations,
        "violations": violations,
        "scope": "heuristic",
        "requires_semantic_review": True,
    }
