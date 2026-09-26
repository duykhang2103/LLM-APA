"""Normalized sample schema shared by all three task pipelines."""

from typing import Any, TypeAlias

NormalizedSample: TypeAlias = dict[str, Any]

# TODO: Replace this documentation tuple with a validated schema implementation.
EXPECTED_SAMPLE_FIELDS = (
    "sample_id",
    "problem_id",
    "problem_type",
    "problem_statement",
    "code_file",
    "code",
    "compile_log",
    "test_report",
    "rubric",
    "error_labels",
    "feedback",
    "feedback_level",
)
