"""Deterministic checks for the Task 3 feedback-level policy."""

import re
import unicodedata
from typing import Any


def _level_one_diagnosis_text(text: str) -> str:
    """Mask explicit no-error praise, retaining uncertain negation scopes."""

    def mask(match: re.Match[str]) -> str:
        prefix = re.split(r"[.!?;]", text[: match.start()])[-1]
        # Conservatively retain phrases under an enclosing double negation.
        if re.search(r"\b(?:khong|chang)\s+phai\b", prefix):
            return match.group(0)
        return " "

    return re.sub(
        r"\bkhong\s+co\s+loi\s+(?:logic|bien dich|vong lap|ham|nhap/xuat)\b",
        mask,
        text,
    )


def _plain(text: str) -> str:
    """Lower case without Vietnamese diacritics, for keyword rules."""
    return "".join(
        c for c in unicodedata.normalize("NFD", text.lower()) if unicodedata.category(c) != "Mn"
    ).replace("đ", "d")


# Advice addressed to the student ("em cần thêm ...", "em hãy xem lại ...",
# "Cần sửa ..."): a fix direction, which Level 2 must not give.
_ADVICE = re.compile(
    r"\bem\s+(?:can|nen|hay|phai|thu|kiem tra lai|xem lai)\b"
    r"|(?:^|[.;!?]\s*)(?:can|nen|hay)\s+(?:sua|them|bo sung|kiem tra|xem lai|dieu chinh|hoan thien|viet lai|doi)\b"
)
# Any correction direction in prose, which Level 3 must give.
_DIRECTION = re.compile(
    r"\b(?:can|nen|hay|phai)\b|\b(?:sua|them|bo sung|kiem tra lai|xem lai|dieu chinh|thay doi|doi lai|viet lai|hoan thien)\b"
)
# Violations a retry should try to fix, but that are better kept than replaced
# by generic fallback feedback once the retries are used up.
SOFT_VIOLATIONS = frozenset({"fix_advice_at_level_2", "missing_fix_direction_at_level_3"})


def strip_fix_advice(feedback: str) -> str:
    """Drop the clauses of a Level 2 feedback that advise the student how to fix.

    The diagnosis before them is kept; if every clause is advice the text is
    returned unchanged.
    """
    clauses = re.split(r"(?<=[.;!?])\s+|,\s+(?=[Ee]m\s)", feedback.strip())
    kept = [c for c in clauses if not _ADVICE.search(_plain(c))]
    if not kept or len(kept) == len(clauses):
        return feedback
    text = " ".join(kept).strip().rstrip(",;")
    return text if text.endswith((".", "!", "?")) else text + "."


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
    # "Không có lỗi biên dịch" is praise, not a diagnosis. Exempt only the
    # negated phrase so other diagnoses in the same feedback are still checked.
    diagnosis_text = _level_one_diagnosis_text(lowered) if level == 1 else lowered
    if level == 1 and re.search(
        r"\b(?:dong (?:so )?\d+|line \d+|p[1-4]|loi (?:logic|bien dich|vong lap|ham|nhap/xuat)|"
        r"thieu|chua khoi tao|khong co ham|sai cong thuc|comment|vi .{1,80} nen)\b",
        diagnosis_text,
    ):
        violations.append("specific_diagnosis_at_level_1")
    if level == 2 and _ADVICE.search(lowered):
        violations.append("fix_advice_at_level_2")
    # A Level 3 feedback that reports a problem must say how to fix it; praise needs no fix.
    reports_problem = re.search(r"\b(?:sai|loi|thieu|chua|khong dung|crash)\b", lowered)
    if level == 3 and reports_problem and not _DIRECTION.search(lowered):
        violations.append("missing_fix_direction_at_level_3")
    return {
        "pass": not violations,
        "violations": violations,
        "scope": "heuristic",
        "requires_semantic_review": True,
    }
