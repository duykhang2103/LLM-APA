# Task 1 redesign implementation checklist

This is a proposed implementation checklist for [plan.md](plan.md), dated 2026-10-06. No implementation item is complete. Paths marked new and their tests are planned artifacts. Commands below are acceptance checks to run after the relevant task; they were not executed as part of this planning work. Private run configurations must resolve actual dataset paths, immutable model revisions, and T4 FP16 settings.

## Task 1 Review grading authority and score coverage

Produce a private aggregate audit of score bands by problem type and split, and record the handling of conflicting weights/logs without changing teacher labels.

Acceptance criteria:

- [ ] Report sample/group counts, means, score-band support, and existing conflict flags; reproduce the pilot's 3.25 versus 9.33 multi-problem means.
- [ ] Distinguish official scoring rules from proposed prompt interpretations; preserve unresolved multi-problem mapping and original references.
- [ ] Auditing cannot change records, expose reference fields to Task 1 inputs, or silently exclude held-out cases.

Verification: `uv run pytest tests/test_data_quality.py`; run preparation audit on the private pilot configuration and compare aggregate counts with the historical audit. Put individual review decisions only in ignored private files.

Dependencies: none. Scope: medium, three files.

Files likely touched:

- `src/llm_grading/data/quality.py`
- `tests/test_data_quality.py`
- `docs/project/DATA_FINDINGS.md`

## Task 2 Establish canonical generation parity

Make generated training evaluation and final inference resolve the same generation runtime. Keep loss computation separate from that runtime and verify a fresh adapter reload on Kaggle.

Acceptance criteria:

- [ ] Record and align generation precision, adapter dtype, cache, template, thinking, decoding, attention, and token settings; restore training state after evaluation.
- [ ] A saved fresh pinned adapter produces exactly the same first-pass six-field predictions in-memory and after reload on the same recorded GPU environment, or the task stays unresolved with a private divergence report; retry outcomes are reported separately.
- [ ] Weight integrity and runtime settings are attached to the parity evidence; the old adapter's ambiguous revision metadata is preserved.

Verification: `uv run pytest tests/test_generated_validation.py tests/test_training_runtime.py tests/test_model_revision.py`; then a real Kaggle save/reload repeat using the current source snapshot and private pinned configuration. CPU tests alone do not satisfy GPU parity.

Dependencies: none; uses the already implemented revision-pinning fix. Scope: medium, five files.

Files likely touched:

- `src/llm_grading/models/generation.py`
- `src/llm_grading/training/validation.py`
- `src/llm_grading/training/trainer.py`
- `tests/test_generated_validation.py`
- `tests/test_training_runtime.py`

## Checkpoint after Tasks 1 and 2

- [ ] Aggregate findings reproduce the audited pilot.
- [ ] GPU parity evidence exists, or expensive candidate training is deferred.
- [ ] Shared contract tests pass; no private artifacts have entered versioned files.

## Task 3 Freeze the prospective development split

Add score-band support to locally generated grouped splits and their audit. Preserve official assignments and historical split directories. Publish fixed seeds/bands before candidate metrics are generated.

Acceptance criteria:

- [ ] Duplicate groups stay intact, official paths bypass local resplitting, and train/dev/test overlap checks remain active.
- [ ] Fixed score bands within problem type guide allocation where feasible; sparse or conflicting groups produce explicit coverage diagnostics without changing labels or searching seeds by metrics.
- [ ] A new split version and private manifest contain group/support counts; the reviewed six-record split remains historical development data.

Verification: `uv run pytest tests/test_data_loader.py tests/test_cli_workflow.py`; prepare a new private directory twice and confirm identical IDs/manifests for identical inputs, then verify changed source/config requires a new version. Test sparse bands and duplicate groups spanning strata with synthetic fixtures.

Dependencies: Task 1. Scope: medium, four files.

Files likely touched:

- `src/llm_grading/data/split.py`
- `scripts/prepare_data.py`
- `tests/test_data_loader.py`
- `tests/test_cli_workflow.py`

## Task 4 Compare the explicit rubric prompt with untuned Qwen

Create v002 and compare it with v001 using the same untuned base, split, runtime, and permitted inputs. Named-function inspection stays inside the grading prompt; output stays six integers.

Acceptance criteria:

- [ ] Every scoring anchor is sourced to the supplied rubric or identified as a reviewed development hypothesis; no invented per-problem conversion or absolute log override.
- [ ] v001 remains unchanged, and the two P0 configurations differ only in prompt identity and associated output/run identifiers.
- [ ] Both comparisons produce complete validated predictions and paired component/total metrics; changes to reference fields do not alter current-submission prompts.

Verification: `uv run pytest tests/test_prompting_runtime.py tests/test_prediction_schema.py`; on Kaggle run each configuration through `scripts/predict.py`, `scripts/validate_predictions.py`, and `scripts/evaluate.py` using the frozen private prepared configuration. Review largest development errors privately.

Dependencies: Tasks 1, 2, 3. Scope: medium, four files.

Files likely touched:

- `tasks/task1_grading/prompts/v002.txt` (new)
- `configs/task1/zero_shot_v2.yaml` (new)
- `configs/task1/zero_shot_control_v2.yaml` (new)
- `tests/test_prompting_runtime.py`

## Checkpoint after Tasks 3 and 4

- [ ] Split identity and score support are frozen prospectively.
- [ ] Untuned Qwen baselines exist and are compared under identical runtime conditions.
- [ ] Prompt changes preserve the external schema and leakage protections.

## Task 5 Test one training-only demonstration

Use existing retrieval to evaluate whether a reviewed training example improves grading over P0 v002. Start with k=1 and the same measured query budget.

Acceptance criteria:

- [ ] Retrieval uses explicit training data, excludes query identity/duplicates, pins the embedding model revision, and logs the selected example privately.
- [ ] Current-submission reference targets and teacher feedback never enter the query or demonstration context; an oversized example is skipped/logged while the complete query is retained.
- [ ] Compare P1 against P0 v002 on identical IDs, including problem-type support, bias, invalid outputs, and runtime/context cost.

Verification: `uv run pytest tests/test_prompting_runtime.py tests/test_experiment_contracts.py`; run prediction/validation/evaluation on Kaggle and inspect private retrieval provenance and budget logs. Synthetic cases verify held-out exclusion and example-budget overflow.

Dependencies: Task 4. Scope: medium, four files.

Files likely touched:

- `configs/task1/rag_v2.yaml` (new)
- `src/llm_grading/prompting/pipeline.py`
- `src/llm_grading/retrieval/retriever.py`
- `tests/test_prompting_runtime.py`

## Task 6 Run the paired pinned fine-tunes

Train fresh F0 controls for v001 and v002 with unchanged LoRA/training settings and the same prospective data protocol. Keep all run directories separate from the historical adapter.

Acceptance criteria:

- [ ] Both runs record immutable base/source/data/prompt/runtime identities, show finite adapter updates, and pass the Task 2 reload parity gate.
- [ ] Prospective checkpoint ranking requires complete valid predictions, uses QWK with MAE tie-breaker, and relies on canonical generated predictions rather than loss alone.
- [ ] Compare each adapter against its matching untuned prompt control; a training follow-up changes one hypothesis and stays within the declared run budget.

Verification: `uv run pytest tests/test_generated_validation.py tests/test_training_runtime.py tests/test_kaggle_notebook.py`; complete Kaggle smoke, train, predict, validate, and evaluate for each private resolved configuration. Independently recompute saved metrics and adapter/checkpoint equality before accepting the comparison.

Dependencies: Tasks 2, 3, 4 and the checkpoint above; Task 5 provides a cheaper alternative comparison but does not change F0 targets. Scope: medium, five files.

Files likely touched:

- `configs/task1/qwen_lora_control_v3.yaml` (new)
- `configs/task1/qwen_lora_rubric_v3.yaml` (new)
- `src/llm_grading/training/validation.py`
- `src/llm_grading/training/trainer.py`
- `tests/test_generated_validation.py`

## Checkpoint after Tasks 5 and 6

- [ ] Every quality claim has its paired P0/P1/F0 control and matching split/runtime identities.
- [ ] No silent input truncation, held-out retrieval, post-hoc label repair, or historical metadata rewrite occurred.
- [ ] Additional tuning proceeds only when the error analysis identifies a specific hypothesis and GPU budget remains.

## Task 7 Report paired errors and slice support

Extend the existing evaluator to expose signed bias and large errors alongside current component/total metrics. Keep detailed error cases private and publish only non-identifying aggregates.

Acceptance criteria:

- [ ] Report total signed error, count/rate of errors greater than two points, exact totals, component MAE, first-pass validity, and existing exam/type/conflict slice counts.
- [ ] Reordering predictions cannot change paired comparisons; missing/duplicate coverage or incompatible run identities cannot silently produce a valid comparison.
- [ ] Tiny or degenerate slices are labeled with support and metric limitations; no statistical superiority claim is made from the six-record pilot.

Verification: `uv run pytest tests/test_task1_metrics.py tests/test_cli_workflow.py`; recompute the historical total MAE 2.3333 and signed bias -2.3333, then compare new candidate saved artifacts without loading a model. Use synthetic signed-error and reordered-ID cases.

Dependencies: Task 3; final comparative report consumes Tasks 4–6. Scope: medium, five files.

Files likely touched:

- `src/llm_grading/evaluation/task1.py`
- `scripts/evaluate.py`
- `tests/test_task1_metrics.py`
- `tests/test_cli_workflow.py`
- `reports/task1_redesign_summary.md` (new, aggregates only)

## Task 8 Freeze the candidate and perform independent acceptance

Select and freeze one candidate from development results, then use the existing acceptance workflow on new lecturer-provided data. If new independent data is unavailable, close with an explicitly exploratory pilot report.

Acceptance criteria:

- [ ] Frozen configuration/adapter/prompt hashes and grading tolerance are recorded before independent test outputs are inspected; no tuning uses that test.
- [ ] Predictions pass external contract validation and packaging; if references are supplied, compare the frozen candidate with matching untuned Qwen and report support and limitations.
- [ ] Shared tests and lint pass for changed files; private code, predictions, labels, checkpoints, and credentials remain excluded from commits and source uploads.

Verification: `uv run pytest`; `uv run ruff check` on changed Python paths. Run existing predict, validate, and package scripts with the frozen private configuration. Use `scripts/evaluate.py --split test` only when legitimate test references exist; do not invent them. Final report links private artifact manifests and records actual hardware/runtime cost.

Dependencies: Tasks 6, 7, or Tasks 4/5 and 7 if development selects a prompt-only candidate. Scope: small, two tracked files plus ignored private artifacts.

Files likely touched:

- `docs/project/RUN_EXPERIMENTS.md`
- `reports/task1_redesign_summary.md`

## Final checkpoint

- [ ] Candidate and evidence are reviewable; aggregate claims match the private artifacts.
- [ ] Independent acceptance is complete, or absence of independent data remains an explicit limitation.
- [ ] No claim that additional epochs, automatic weight correction, or larger LoRA rank caused improvement without a controlled comparison.
