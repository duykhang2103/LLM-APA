# Task 3 — Controlled Feedback Generation

Owner boundary: M5, coordinated with shared prompting and model infrastructure.

## Contract

Input is the problem, student code, available compile/test evidence, and required feedback level. Legacy examples also supply known error labels; public test inputs may omit them. The loader represents omitted Task 3 labels as `error_labels: null`, distinct from an explicit empty list meaning no errors. Task 3 input views omit unknown labels and preserve supplied lists, including empty ones. Reference feedback remains excluded from model inputs.

The v003 prompt and `task3.output_format: taxonomy_feedback` predict both `taxonomy_error` and Vietnamese `feedback`. Current-submission labels are removed from joint-mode prompts, including legacy examples that supply them. Labels are validated against the official taxonomy; compliance is checked on the feedback text. New labeled files retain output taxonomy separately as `reference_error_labels`, while legacy supplied labels can serve as training targets without being exposed in the joint prompt. Missing supervision is rejected rather than converted to an empty target.

Earlier prompts and feedback-only runs remain supported. For a release submission, pass `--require-task3-taxonomy` to validation and packaging; exported records contain only `sample_id` and `output: {taxonomy_error, feedback}`. See [the Kaggle start guide](../../docs/project/TASK3_KAGGLE.md) for the three-example zero-shot trial.

Level 1/2 feedback must not contain a complete solution or corrected code. A candidate generator should pass through a compliance checker and may be regenerated when it violates the requested level.

## Evaluation target

Evaluate diagnosis correctness and level compliance separately. Record violations and regeneration behavior for error analysis.

## Files

- `dataset.py`: task-specific input/target view.
- `pipeline.py`: deterministic baseline; shared model execution adds generated feedback and compliance diagnostics.
- `compliance.py`: centralized checker interface.
- `prompts/level_policy.md`: one policy for all experiments.
- `prompts/`: immutable prompt versions.

The v002 prompt follows the supplied level definitions. Training and RAG exclude heuristic feedback-level violations by default, with auditable private review overrides. Validation references are preserved. Strict generation configurations allow one bounded compliance retry, then reject persistent violations. The v003 release configuration uses `task3.feedback_policy: warn`: valid answers are saved with warnings, malformed or failed cases are recorded separately, and later cases continue. The Kaggle notebook exports saved valid records by default and reports missing IDs separately; terminal packaging enables this with `--allow-partial-submission`. Missing cases are omitted without invented labels or feedback, and external grader handling of missing IDs is unverified. Heuristic compliance cannot establish semantic correctness; independent judgments are required to report diagnosis accuracy. Follow [RUN_EXPERIMENTS.md](../../docs/project/RUN_EXPERIMENTS.md).
