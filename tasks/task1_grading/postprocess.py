"""Task 1 structured-output and rubric-range postprocessing boundary.

Example raw model output::

    {"compilable": 1, "io_format": 1, "logic": 3,
     "edge_case": 1, "complexity": 1, "code_quality": 1}

Example postprocessed output adds ``total: 8``. Shared validation rejects
out-of-range values and totals that disagree with components. The raw dataset
loader also enforces exactly six reference rubric dimensions.
"""

from llm_grading.data.schema import RUBRIC_RANGES as RUBRIC_RANGES
from llm_grading.data.schema import validate_rubric


def postprocess_rubric(raw_output: object) -> dict[str, int]:
    """Parse six bounded components and calculate the deterministic total."""
    return validate_rubric(raw_output)
