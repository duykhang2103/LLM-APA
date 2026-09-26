# Task 1 — Rubric Grading

Owner boundary: M3, coordinated with shared infrastructure.

## Contract

Input is the problem statement, student C++ code, and optional `compile_log` / `test_report`. Output is six rubric components:

```text
compilable 0–1
io_format  0–1
logic      0–4
edge_case  0–2
complexity 0–1
code_quality 0–1
```

The total must be derived from the components. Multi-problem submissions must preserve prerequisite and dependency rules from the problem. The future model input must not contain `feedback`.

## Evaluation target

Primary: QWK on total score. Secondary: MAE and exact match per component.

## Files

- `dataset.py`: task-specific input/target view.
- `pipeline.py`: task interface and orchestration boundary.
- `postprocess.py`: future structured-output and score-range handling.
- `prompts/`: immutable prompt versions.
