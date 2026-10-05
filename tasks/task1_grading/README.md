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

The total must be derived from the components. Multi-problem inputs preserve prerequisite policies. EX01 weight metadata conflicts and has no authoritative per-question-to-rubric mapping; do not invent one. The input whitelist excludes `feedback`, labels, and reference scores.

## Evaluation target

Primary: QWK on total score. Secondary: MAE and exact match per component.

## Files

- `dataset.py`: task-specific input/target view.
- `pipeline.py`: task interface and orchestration boundary.
- `postprocess.py`: bounded rubric validation and deterministic totals.
- `prompts/`: immutable prompt versions.

P0/P1/F0 use the shared model pipeline, this task’s `prompts/v001.txt`, and task-specific training targets. Start with the synthetic fixtures, then follow [RUN_EXPERIMENTS.md](../../docs/project/RUN_EXPERIMENTS.md).
