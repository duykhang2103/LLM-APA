"""Task 1 structured-output and rubric-range postprocessing boundary.

Example raw model output::

    {"compilable": 1, "io_format": 1, "logic": 3,
     "edge_case": 1, "complexity": 1, "code_quality": 1}

Example postprocessed output adds ``total: 8``. The parser should reject
unknown fields, out-of-range values, and totals that disagree with components.
"""


def postprocess_rubric(raw_output: object) -> dict[str, int]:
    """Parse six bounded components and calculate the deterministic total."""
    # TODO: Validate component ranges and calculate total from components.
    raise NotImplementedError("Task 1 postprocessing is not implemented.")
