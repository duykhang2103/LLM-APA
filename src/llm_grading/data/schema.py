"""Normalized sample schema shared by all three task pipelines.

Read ``examples/normalized_sample.json`` before changing this file. The
loader may read more fields internally, but every downstream task should
receive a predictable record with the same identifiers and evidence fields.
"""

from collections.abc import Mapping
from typing import Any, TypeAlias

NormalizedSample: TypeAlias = dict[str, Any]

RUBRIC_RANGES = {
    "compilable": (0, 1),
    "io_format": (0, 1),
    "logic": (0, 4),
    "edge_case": (0, 2),
    "complexity": (0, 1),
    "code_quality": (0, 1),
}


def validate_rubric(raw_output: object) -> dict[str, int]:
    """Validate bounded integer scores and derive the shared internal total."""
    if not isinstance(raw_output, Mapping):
        raise ValueError("Task 1 output must be an object")
    missing = [name for name in RUBRIC_RANGES if name not in raw_output]
    if missing:
        raise ValueError(f"Missing Task 1 components: {missing}")
    result: dict[str, int] = {}
    for name, (minimum, maximum) in RUBRIC_RANGES.items():
        value = raw_output[name]
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Task 1 component {name} must be an integer")
        if not minimum <= value <= maximum:
            raise ValueError(f"Task 1 component {name} must be between {minimum} and {maximum}")
        result[name] = value
    expected_total = sum(result.values())
    if "total" in raw_output and raw_output["total"] != expected_total:
        raise ValueError(f"Task 1 total must equal the component sum ({expected_total})")
    result["total"] = expected_total
    return result


# TODO: Replace this documentation tuple with a validated schema implementation.
EXPECTED_SAMPLE_FIELDS = (
    "sample_id",
    "problem_id",
    "problem_type",
    "problem_statement",
    "language",
    "problems",
    "grading_policy",
    "code_file",
    "code",
    "compile_log",
    "test_report",
    "rubric",
    "total_score",
    "error_labels",
    "feedback",
    "feedback_level",
)

# Field ownership guide for new contributors.
FIELD_DESCRIPTIONS = {
    "sample_id": "Stable submission identifier used in predictions and splits.",
    "problem_id": "Problem/exam identifier used for grouping and analysis.",
    "problem_type": "single_problem or multi_problem.",
    "problem_statement": "The statement shown to the student.",
    "language": "Language/standard supplied by the exam, normally cpp11.",
    "problems": "Subproblem metadata, including weights and prerequisite links, preserved as supplied.",
    "grading_policy": "Exam policy preserved without inventing rubric-to-subproblem mappings.",
    "code_file": "Relative path reference to the source file in private data.",
    "code": "Resolved C++ source text.",
    "compile_log": "Optional environment/toolchain evidence, not automatic truth.",
    "test_report": "Optional test evidence when supplied by the challenge.",
    "rubric": "Reference Task 1 components; never an inference-time input.",
    "total_score": "Optional Task 1 reference total, checked against the rubric sum by the raw loader.",
    "error_labels": "Reference Task 2 labels; supplied to Task 3 as context.",
    "feedback": "Reference Task 3 text; forbidden as Task 1/2 input.",
    "feedback_level": "Requested Task 3 feedback level, normally 1 through 4.",
}

# Safe teaching fixture; use the JSON file as the canonical readable example.
EXAMPLE_NORMALIZED_SAMPLE: NormalizedSample = {
    "sample_id": "synthetic-001",
    "problem_id": "SYNTHETIC_SUM",
    "problem_type": "single_problem",
    "problem_statement": "Read two integers and print their sum.",
    "code_file": "submissions/SYNTHETIC_SUM/SYNTHETIC_SUM-student.cpp",
    "code": "<resolved C++ source>",
    "compile_log": "compile succeeded",
    "test_report": "1/2 tests passed",
    "rubric": {"logic": 1},
    "error_labels": ["Lỗi logic"],
    "feedback": "<reference feedback>",
    "feedback_level": 1,
}
