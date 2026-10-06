# Run P0 / P1 / F0

Four-week course project, about two weeks for model experiments. Use the same code locally or through [training](../../notebooks/train_qwen_qlora.ipynb) / [prompting](../../notebooks/run_prompting.ipynb) notebooks. Start with tests and smoke, then run versions. No tracking service is needed.

## Environment and private data

Python 3.11 is selected by `.python-version`. Notebooks create a separate `.notebook-venv` with uv, so a Colab/Kaggle kernel on another Python version still calls the same supported repository scripts. Install uv, then from the repo root:

```bash
uv sync --group dev --group ml
uv run pytest
```

`uv.lock` records packages. QLoRA requires a CUDA GPU and bitsandbytes; Apple/CPU tiny-model tests do not establish GPU acceptance. Default configs target Qwen3.5-4B, 4-bit NF4, rank 8, batch 1, accumulation 4, 4096 tokens. Measure fit; use a larger GPU or explicitly adjust the budget. Evidence is never silently truncated. Optional fast Qwen kernels are not required but reference kernels can be slower.

Place teacher files and `submissions/` under `data/raw/`. Configure `data.input_path`; if official train/dev/test files exist, set `train_path`, `validation_path`, and `test_path` explicitly. Official splits override local generation. Unlabeled test files work for inference; they cannot be training/evaluation targets.

## P0: first zero-shot run

Edit `configs/task1/zero_shot.yaml` for paths/platform, experiment version, and output directory. Substitute task2/task3 for other tasks.

```bash
uv run python scripts/prepare_data.py --config configs/task1/zero_shot.yaml --training
uv run python scripts/smoke.py --config data/processed/task1-v1/config-task1-P0-v1.json
uv run python scripts/predict.py --config data/processed/task1-v1/config-task1-P0-v1.json --split val
uv run python scripts/evaluate.py --config data/processed/task1-v1/config-task1-P0-v1.json --split val --predictions outputs/task1/P0-v1/predictions.json
```

Preparation prints the exact materialized config path. Split IDs/source identity are persisted; changed data/settings need a new split version/prepared directory. Method-specific configs share these IDs without overwriting each other's materialized configs.

For an API, use `zero_shot_api.yaml`: set provider model and base URL, export `LLM_API_KEY`, and enable `model.allow_private_data` only if course data may be transmitted. A single OpenAI-compatible `/chat/completions` adapter is supported. Never place a key in YAML. Enter per-million token prices only if known; otherwise cost remains unavailable.

Task 1/2 parse JSON, validate, and allow one format-repair retry. Still-invalid output stops clearly; old valid prediction files remain intact. `responses.jsonl` stores raw/parsed responses, errors, retries, latency/usage and retrieval audits privately. Freeze `P0-v1` after inspecting failures.

## P1: training-only RAG

Prepare `configs/task1/rag.yaml` with the same data/split version. It loads the explicit training corpus and a dense encoder in memory. Set `retrieval.k` to 1/3/5, then use the same smoke/predict/evaluate commands with `config-task1-P1-v1.json` and `outputs/task1/P1-v1/`.

Task 1 examples expose rubric targets; Task 2 exposes taxonomy; neither exposes feedback. Task 3 first filters to requested level and may expose feedback targets. Query IDs are always excluded. `exclude_duplicates` and optional similarity cutoff support normal vs restricted RAG; log matches/exclusions/shortfalls. The first embedding/model download needs network access; no vector database or BM25 prerequisite exists.

## F0: smoke, train, reload, resume

The current F0 configs use version 2 and new output/preparation directories. Task 3 P0/P1 also use the stricter `v002` prompt; `v001` is preserved for recorded runs. The 32 supplied samples are format fixtures. Inspect them without loading a model:

```bash
uv run python scripts/prepare_data.py --config configs/task3/qwen_lora.yaml --input data/sample_dataset --output outputs/sample-quality/task3.json
```

Preparation reports coverage separately for train/validation/test. Training writes `training_data_audit.json` with flags, inclusion reasons, review-file hash, retained support, unchanged validation membership, and repetition counts. `data.quality.exclude_flags` defaults to feedback-level violations; compiler/test and question-weight conflicts remain review-only because logs are not lecturer ground truth. Private `data.quality.review_file` decisions can explicitly keep or exclude training records, with nonempty reasons. Every ID must belong to the training split; never review/relabel the private test set. Example review file:

```json
{"synthetic-001": {"action": "exclude", "reason": "Confirmed source-version mismatch"}}
```

Default Task 2/3 fine-tuning configs require all ten labels/four levels in the retained training data. Missing support stops training and saves the audit before model loading. For a deliberately small smoke/ablation, explicitly set `data.quality.require_coverage: false` in a new experiment config; do not invent missing labels or claim official quality from this sample. Task 2 balances observed labels and Task 3 balances observed feedback levels by repeating existing training references at most three times. Validation is never balanced or filtered; coverage and duplicates remain visible.

Generated validation runs at each evaluation interval and saves first-pass raw/parsed completions under `generated_validation/step-*.json`. Task 1 selection uses `(QWK + 1) / 2 * valid_rate`; Task 2 uses `macro_F1 * valid_rate`, retaining the fixed ten-label definition. Invalid responses are counted and penalized, not replaced with teacher answers. The selected checkpoint is reloaded for the final adapter. Task 3 logs heuristic compliance only; automatic best-checkpoint selection is disabled because that proxy cannot measure diagnosis quality. Review generated feedback independently before choosing the Task 3 adapter. Generated checkpoint evaluation currently supports single-process training, and adds generation time to validation.

Edit `qwen_lora.yaml`, then prepare the same data contract:

```bash
uv run python scripts/prepare_data.py --config configs/task1/qwen_lora.yaml --training
uv run python scripts/smoke.py --config data/processed/task1-v2/config-task1-F0-v2.json --training
uv run python scripts/train.py --config data/processed/task1-v2/config-task1-F0-v2.json
```

Smoke checks tokenizer/model, finite loss, LoRA updates, checkpoint save, adapter reload, one validation prediction/schema/evaluator, in a separate `smoke/` folder. Full training saves adapter/tokenizer, train/validation losses, base revision, metadata, and optimizer-bearing checkpoints. Check the longest input budget, not just a short sample, before spending GPU hours. No GPU run has been demonstrated by the CPU test suite.

Two-step smoke training overrides warmup to zero and disables best-checkpoint reload. Otherwise the first step can use zero learning rate and tied generated metrics can restore its unchanged weights, falsely failing the LoRA update check. Full training keeps the requested warmup and checkpoint selection. Smoke contracts identify these overrides, so a smoke checkpoint cannot be resumed as a full-training job. The manifest records `effective_training_settings` and smoke `lora_update_diagnostics`; a real zero-update failure still stops training.

The [Kaggle v3 notebook](../../notebooks/train_qwen_qlora_kaggle_v3.ipynb) copies an attached repository into writable storage and checks the shared completion-only collator and smoke policy before loading Qwen. Update the attached repo Dataset as well as the notebook when taking these fixes; old snapshots stop at preflight. Each smoke attempt writes under `LLM-APA-runs/smoke-checks/<task>/<version>/<attempt>/smoke/`, preserving failed checkpoints for inspection and allowing a fresh retry. The training cell requires a passed smoke when `RUN_SMOKE` is enabled. Completion-only logits preserve the full prompt and supervised answer; loss/gradient equivalence is covered by a tiny Qwen CPU test, while actual CUDA memory fit still requires a Kaggle run.

Repository subprocesses in Kaggle set `CUDA_VISIBLE_DEVICES=0` before importing PyTorch, enforcing the single-GPU training path even when the notebook exposes multiple GPUs ([PyTorch environment reference](https://docs.pytorch.org/docs/stable/cuda_environment_variables.html)). A passed smoke is bound to the task, version and SHA-256 of the prepared config; changing any of these requires a new smoke attempt. Explicit `TRAIN_FILE`/`VAL_FILE` paths work without a dataset inside the repository. Automatic sample discovery checks both the documented `sample_dataset/` layout and `data/sample_dataset/`.

To update Kaggle, import the corrected notebook and replace the attached repository Dataset with a source snapshot containing the current working files, including `dataset.py` and `trainer.py`. Keep private data separate: a source-only snapshot has no lecturer/sample submissions, so attach an authorized private data Dataset and set `DATA_ROOT_OVERRIDE` or both official split paths. Enable GPU and Internet, restart from the first cell, and require source preflight and smoke to pass before training. Save Notebook Output to retain checkpoints.

Kaggle mount prefixes can differ between notebook sessions and saved runs. When multiple repository snapshots are attached, set `REPO_DATASET_SLUG` (for example, `llm-apa-source`) to select the current source by its Dataset directory name. Use `find_attached_files("task1_grading.json", "llm-apa")` to discover a separately attached data file. Avoid hardcoding `/kaggle/input/datasets/<owner>/...` in a deployed notebook; both flat and owner-qualified mounts are covered by notebook regression tests.

Use the **exact revision recorded in `run_manifest.json`** for inference; set it in the materialized config, and provide the adapter:

Remote model loads resolve the cached `config.json` snapshot to its full commit before loading the config, tokenizer or weights. This also works with Transformers versions that no longer retain `_commit_hash` on config objects ([Hub cache and revision documentation](https://huggingface.co/docs/huggingface_hub/guides/download)). Local model directories remain supported. Older runs recorded as `main` need their original cache/snapshot evidence recovered before reusing adapters with this loader; today's Hub `main` is not proof of the training revision. Preserve the original artifacts and record any recovery separately.

```bash
uv run python scripts/predict.py --config data/processed/task1-v2/config-task1-F0-v2.json --split val --adapter-path outputs/task1/F0-v2/adapter
uv run python scripts/evaluate.py --config data/processed/task1-v2/config-task1-F0-v2.json --split val --predictions outputs/task1/F0-v2/predictions.json
uv run python scripts/train.py --config data/processed/task1-v2/config-task1-F0-v2.json --resume-from-checkpoint outputs/task1/F0-v2/checkpoint-100
```

Replace `checkpoint-100` with an existing checkpoint. Resume restores optimizer/scheduler state and rejects changed data/model/template/LoRA settings. A completed run needs additional epochs/steps to continue; it does not repeat completed steps automatically. For a new hypothesis, use F0-v2/new output path; optionally set `training.warm_start_adapter` to start from old adapter weights with a fresh optimizer. An `adapter/` folder alone cannot resume Trainer state.

Cloud notebooks use the same scripts and path-based persistent storage (Drive/mounted volume/Kaggle output). They pin the saved base revision before validation inference. Save/export artifacts before ending a rented runtime.

## Fair comparison and final acceptance

Task 3 generation makes at most one compliance retry by default (`task3.max_compliance_retries`, bounded 0–3). A persistent detected violation stops the prediction command. The checker catches common code/fix/location patterns, not arbitrary semantic disclosure. It is labeled heuristic in artifacts and cannot certify correctness. Evaluation recomputes compliance rather than trusting stored `pass` values; input labels are never counted as predicted diagnoses.

Use `scripts/evaluate.py --judgments path/to/private-review.json` for independent Task 3 semantic judgments on a held-out subset. The file maps prediction IDs to `judge` (reviewer/model and version), `feedback_sha256` (SHA-256 of the exact UTF-8 feedback), `feedback_level`, `diagnosis_correct`, and `level_compliant`. Boolean ratings are required; mismatched text/level is rejected. Without judgments, diagnosis accuracy is reported as unavailable, not inferred from the supplied labels. Review-count and semantic metrics are reported separately from heuristic compliance.

Use identical inputs, splits, evaluator and prediction contract. Task 2 uses fixed ten-label sklearn-style zero_division=0; confirm lecturer semantics when supplied. Task 3 heuristic compliance is a diagnostic; diagnosis accuracy requires independent feedback judgments. Inspect label support and exam/level/conflict slices, plus RAG duplicate audits. Preserve sample annotations; see [DATA_FINDINGS.md](DATA_FINDINGS.md).

Freeze the selected inference config/adapter before final test material. Only configure paths:

```bash
uv run python scripts/predict.py --config path/to/frozen-inference-config.json --input data/raw/lecturer-test --output outputs/final/predictions.json
uv run python scripts/validate_predictions.py --task task1 --input outputs/final/predictions.json
uv run python scripts/package_predictions.py --task task1 --input outputs/final/predictions.json --output outputs/final/submission.zip
```

No retraining or target inspection is involved. Packaging defaults to the lecturer sample contract: Task 1 nested `rubric` and `total_score`, Task 2 `taxonomy_error`, Task 3 `feedback` only. Use `--format internal` solely for internal artifacts. Export to `.json` or `.zip`; validate exported JSON with `scripts/validate_predictions.py --format challenge`. Confirm this contract against the official validator when supplied. Keep code/data/checkpoints/response logs private. SemIf/F1 remains a later optional extension, not a training dependency.

Implementation follows [Qwen text-only loading](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization), and [Trainer resume](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train). GPU memory/quality must be measured on the team's platform.
