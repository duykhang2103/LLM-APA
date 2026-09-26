# Task 3 — Controlled Feedback Generation

Owner boundary: M5, coordinated with shared prompting and model infrastructure.

## Contract

Input is the problem, student code, known error labels, and required feedback level. Output is Vietnamese feedback that diagnoses the issue while obeying the centralized Level 1–4 policy.

Level 1/2 feedback must not contain a complete solution or corrected code. A candidate generator should pass through a compliance checker and may be regenerated when it violates the requested level.

## Evaluation target

Evaluate diagnosis correctness and level compliance separately. Record violations and regeneration behavior for error analysis.

## Files

- `dataset.py`: task-specific input/target view.
- `pipeline.py`: generation, checking, and regeneration boundary.
- `compliance.py`: centralized checker interface.
- `prompts/level_policy.md`: one policy for all experiments.
- `prompts/`: immutable prompt versions.
