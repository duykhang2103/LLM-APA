"""Normalized sample schema shared by all three task pipelines.

Read ``examples/normalized_sample.json`` before changing this file. The
loader may read more fields internally, but every downstream task should
receive a predictable record with the same identifiers and evidence fields.
"""

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

# Field ownership guide for new contributors.
FIELD_DESCRIPTIONS = {
    "sample_id": "Stable submission identifier used in predictions and splits.",
    "problem_id": "Problem/exam identifier used for grouping and analysis.",
    "problem_type": "single_problem or multi_problem.",
    "problem_statement": "The statement shown to the student.",
    "code_file": "Relative path reference to the source file in private data.",
    "code": "Resolved C++ source text.",
    "compile_log": "Optional environment/toolchain evidence, not automatic truth.",
    "test_report": "Optional test evidence when supplied by the challenge.",
    "rubric": "Reference Task 1 components; never an inference-time input.",
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
    "error_labels": ["LABEL_02"],
    "feedback": "<reference feedback>",
    "feedback_level": 1,
}
