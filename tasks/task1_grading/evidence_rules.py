"""Rule-based Task 1 rubric from compile evidence and test pass rate.

A baseline the LLM must beat, and a fallback. Thresholds were fixed before
looking at the training scores (pass rate -> logic 0-4 by rounding, edge_case
by 30%/90%); training data is only used to measure them.
"""

FIELDS = ("compilable", "io_format", "logic", "edge_case", "complexity", "code_quality")


def rubric_from_evidence(sample, evidence, report):
    """Return a rubric dict, or None when there is no test report to use."""
    if evidence is not None and not evidence["relaxed"]:
        quality = 1 if sample.get("problem_type") == "single_problem" else 0
        return dict(zip(FIELDS, (0, 0, 0, 0, 0, quality)))
    if report is None or not report["total"]:
        return None
    rate = report["pass_rate"]
    statuses = report["status_counts"]
    io_format = 0 if statuses.get("format_mismatch", 0) > report["total"] / 2 else 1
    edge = 2 if rate >= 0.9 else 1 if rate >= 0.3 else 0
    return dict(zip(FIELDS, (1, io_format, round(4 * rate), edge, 1, 1)))
