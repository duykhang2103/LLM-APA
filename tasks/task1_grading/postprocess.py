"""Task 1 structured-output and rubric-range postprocessing boundary.

Example raw model output::

    {"compilable": 1, "io_format": 1, "logic": 3,
     "edge_case": 1, "complexity": 1, "code_quality": 1}

Example postprocessed output adds ``total: 8``. The parser should reject
unknown fields, out-of-range values, and totals that disagree with components.
"""

from collections.abc import Mapping


RUBRIC_RANGES = {
    "compilable": (0, 1),
    "io_format": (0, 1),
    "logic": (0, 4),
    "edge_case": (0, 2),
    "complexity": (0, 1),
    "code_quality": (0, 1),
}


def postprocess_rubric(raw_output: object) -> dict[str, int]:
    """Parse six bounded components and calculate the deterministic total."""
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
