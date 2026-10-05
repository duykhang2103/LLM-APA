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

Edit `qwen_lora.yaml`, then prepare the same data contract:

```bash
uv run python scripts/prepare_data.py --config configs/task1/qwen_lora.yaml --training
uv run python scripts/smoke.py --config data/processed/task1-v1/config-task1-F0-v1.json --training
uv run python scripts/train.py --config data/processed/task1-v1/config-task1-F0-v1.json
```

Smoke checks tokenizer/model, finite loss, LoRA updates, checkpoint save, adapter reload, one validation prediction/schema/evaluator, in a separate `smoke/` folder. Full training saves adapter/tokenizer, train/validation losses, base revision, metadata, and optimizer-bearing checkpoints. Check the longest input budget, not just a short sample, before spending GPU hours. No GPU run has been demonstrated by the CPU test suite.

Use the **exact revision recorded in `run_manifest.json`** for inference; set it in the materialized config, and provide the adapter:

```bash
uv run python scripts/predict.py --config data/processed/task1-v1/config-task1-F0-v1.json --split val --adapter-path outputs/task1/F0-v1/adapter
uv run python scripts/evaluate.py --config data/processed/task1-v1/config-task1-F0-v1.json --split val --predictions outputs/task1/F0-v1/predictions.json
uv run python scripts/train.py --config data/processed/task1-v1/config-task1-F0-v1.json --resume-from-checkpoint outputs/task1/F0-v1/checkpoint-100
```

Replace `checkpoint-100` with an existing checkpoint. Resume restores optimizer/scheduler state and rejects changed data/model/template/LoRA settings. A completed run needs additional epochs/steps to continue; it does not repeat completed steps automatically. For a new hypothesis, use F0-v2/new output path; optionally set `training.warm_start_adapter` to start from old adapter weights with a fresh optimizer. An `adapter/` folder alone cannot resume Trainer state.

Cloud notebooks use the same scripts and path-based persistent storage (Drive/mounted volume/Kaggle output). They pin the saved base revision before validation inference. Save/export artifacts before ending a rented runtime.

## Fair comparison and final acceptance

Use identical inputs, splits, evaluator and prediction contract. Task 2 uses fixed ten-label sklearn-style zero_division=0; confirm lecturer semantics when supplied. Task 3 overlap/compliance are diagnostics requiring feedback review. Inspect label support and exam/level/conflict slices, plus RAG duplicate audits. Preserve sample annotations; see [DATA_FINDINGS.md](DATA_FINDINGS.md).

Freeze the selected inference config/adapter before final test material. Only configure paths:

```bash
uv run python scripts/predict.py --config path/to/frozen-inference-config.json --input data/raw/lecturer-test --output outputs/final/predictions.json
uv run python scripts/validate_predictions.py --task task1 --input outputs/final/predictions.json
uv run python scripts/package_predictions.py --task task1 --input outputs/final/predictions.json --output outputs/final/submission.zip
```

No retraining or target inspection is involved. Packaging retains `{sample_id, output}`; confirm any different lecturer submission schema. Keep code/data/checkpoints/response logs private. SemIf/F1 remains a later optional extension, not a training dependency.

Implementation follows [Qwen text-only loading](https://huggingface.co/docs/transformers/model_doc/qwen3_5), [PEFT quantized training](https://huggingface.co/docs/peft/developer_guides/quantization), and [Trainer resume](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Trainer.train). GPU memory/quality must be measured on the team's platform.
