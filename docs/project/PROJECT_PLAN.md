# Project Plan — Four-week Research Course

The effective model-development window is about **two weeks**. Prioritize runnable P0/P1/F0 experiments now; use the existing modules and scripts rather than adding infrastructure. This plan supersedes broader recommendations in the older meeting brief. Detailed CLI steps are in [RUN_EXPERIMENTS.md](RUN_EXPERIMENTS.md).

## Operating sequence

1. Implement and smoke-test the shared data/model/training paths.
2. Freeze a working baseline on the actual execution platform.
3. Run P0 zero-shot and F0 Qwen3.5-4B QLoRA versions.
4. Add P1 training-only RAG and change one training hypothesis at a time.
5. Resume the same experiment from Trainer checkpoints; use new version/output paths for deliberate retraining changes.
6. Freeze candidates and verify reproduction.
7. Run lecturer-provided acceptance/test data: load → predict → validate → package. Do not tune or manually label private test data.

No week-by-week schedule, workflow engine, service layer, vector database, or tracking platform is needed. JSON/YAML/CSV and conventional Transformers/PEFT suffice.

## Current implementation and acceptance boundary

**Implemented:** normalized teacher/flat loader, input whitelists, six-field rubric/total checks, taxonomy, shared evaluation/validation, grouped split preparation, P0 local/API runner, strict parsing and bounded retry, P1 dense retrieval, QLoRA training, adapter reload and Trainer resume, artifact manifests, CLI and execution notebooks, essential CPU tests.

**Hardware acceptance still required:** load the actual Qwen3.5-4B revision on the intended CUDA GPU, smoke finite-loss training and save/reload, verify parseable validation generation, and measure the longest configured token budget. CPU tiny-model tests are not evidence of successful GPU QLoRA or useful model quality. Record actual GPU/VRAM/environment/time/cost. Move to a larger GPU if local hardware requires unreasonable compromises; choose a documented fallback model only if Qwen is actually incompatible.

**Optional later:** F1 SemIf-style typed decisions, BM25/hybrid retrieval, focused compliance regeneration or probability-based threshold tuning. None blocks the first baseline; label-list generation does not supply calibrated probabilities.

## Research methods

| Family | Run example | Required behavior |
|---|---|---|
| B0 | Existing `T1-001` heuristic | Cheap plumbing/data baseline, not a learned-model claim |
| P0 | `task1-P0-v1` | Separate zero-shot prompts; no demonstrations |
| P1 | `task1-P1-v1` | Same base prompt/model, top-k training examples; k = 1/3/5 |
| F0 smoke | `task1-F0-v0-smoke` | Short Qwen QLoRA training, parameter update, checkpoint save/reload |
| F0 | `task1-F0-v1` | First full QLoRA baseline; subsequent versions answer one question |
| F1 | Later, optional | Typed discrete rubric/label decisions compared against plain F0 |

Keep legacy experiment IDs valid. Never overwrite a previous version's scientific identity. Save resolved config, Git revision (or explicitly unavailable), dataset/split versions, model revision, prompt version, seed, decoding/LoRA settings, losses/metrics, checkpoint path, hardware, elapsed time, and notes. Use private output folders and the existing CSV convention.

## Shared input and comparison contract

- Same dataset, train/validation IDs, permitted inputs, evaluator, and `{sample_id, output}` contract across approaches.
- Task 1/2 never receive teacher feedback, reference rubric, totals, or taxonomy targets as current-submission input. Retrieved Task 1/2 examples expose only their appropriate targets, never feedback.
- Task 1 emits six rubric dimensions; code computes total. Task 2 emits official label subsets, including an empty set. Task 3 receives given taxonomy and requested level; teacher feedback is only its training/retrieval target.
- Treat compile/test logs as auxiliary evidence. Preserve original annotations and flag conflicts; no invented EX01 per-question-to-rubric conversion.
- Official train/dev/test paths override local splitting. Otherwise persist seeded normalized-code groups with exam-type balance where possible. Check ID/code overlap; changed data requires a new split version/directory.
- Retrieve only explicit training data. Exclude query ID; log exact duplicates. Compare normal vs duplicate-restricted RAG. Task 3 filters to the requested level before similarity ranking.

## Evaluation and failure analysis

Task 1: total QWK/MAE, component MAE/exact match. Task 2: macro/micro-F1 and per-label precision/recall/F1/support, with all ten labels and sklearn-style `zero_division=0`; confirm lecturer semantics when supplied. Task 3: separate diagnosis correctness and level compliance; current overlap/marker checks are diagnostics requiring content review.

Use [sample findings](DATA_FINDINGS.md) as analysis dimensions: EX01/EX02, evidence conflicts, common/rare labels, feedback levels, and duplicate/non-duplicate retrieval. Report slice sizes and invalid-output rates. Sample demonstration metrics are not official held-out performance. Do not add unrelated method searches before understanding these failures.

## Team ownership and immediate actions

| Owner | Responsibility | Next action |
|---|---|---|
| M1 | Integration, scripts, reproducibility | Run tests and actual platform smoke; fix concrete failures |
| M2 | Data audit, split IDs, privacy | Load official files, verify sample findings and duplicate/conflict flags |
| M3 | Task 1 prompts/targets/metrics | Run P0 and F0 rubric predictions; inspect EX01 issues |
| M4 | Task 2 taxonomy/metrics | Check zero-support semantics and rare-label results |
| M5 | Task 3 levels/feedback | Review correctness/compliance and noisy supervision |
| M6 | QLoRA/hardware/checkpoints | Select CUDA platform; run smoke, full tune, and resume |

For five members, combine lead/fine-tuning ownership; an extra member can assist retrieval/evaluation. Prompting and fine-tuning proceed in parallel after agreeing the shared contract. Review small PRs with the task owner; include commands/results, experiment version, and any changed input/output contract.

## Final acceptance

The selected configurations and adapters must generate valid predictions without manual repair. Freeze settings before lecturer test material arrives. Use the same predict/validate/package scripts with configured input paths. Confirm the lecturer's final external schema if it differs from the repository contract. Keep private student code, data, API keys, logs, and checkpoints out of Git.

Retain the [Task 1 QLoRA design](../superpowers/specs/2026-10-03-task1-qwen-qlora-design.md) as historical detailed reference; current scripts/configs determine actual execution. Finish report, error analysis, model card, slides, and contribution table. Do not claim GPU training success until it has actually run.
