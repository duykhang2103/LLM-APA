# Repository Guidelines

## Project Structure & Module Organization

This Python project assesses C++ submissions through rubric grading, error classification, and Vietnamese feedback. Shared infrastructure lives in `src/llm_grading/` (data, models, training, evaluation, validation, and utilities). Task pipelines and versioned prompts live in `tasks/task1_grading/`, `tasks/task2_errors/`, and `tasks/task3_feedback/`. Keep reusable logic in the shared package.

Use `configs/` for YAML experiment settings, `scripts/` for CLI entry points, `examples/` for synthetic fixtures, and `tests/` for automated checks. `experiments/results.csv` tracks results; `reports/` holds analysis and figures. Contributor workflow details are in `docs/project/SOURCE_SETUP.md`.

## Build, Test, and Development Commands

Use Python 3.10 or 3.11 and run commands from the repository root:

- `uv sync --all-groups`: install runtime, development, and ML dependencies.
- `uv run pytest`: discover and run tests.
- `uv run ruff check .`: lint Python code.
- `uv run python scripts/prepare_data.py --config configs/base.yaml --input examples/normalized_sample.json`: validate and prepare synthetic data.
- `uv run python scripts/predict.py --config configs/task1/heuristic.yaml --output outputs/task1_predictions.json`: run the deterministic baseline without downloading a model.
- `uv run python scripts/evaluate.py --config configs/task1/heuristic.yaml --predictions outputs/task1_predictions.json`: compute metrics.
- `uv run python scripts/validate_predictions.py --task task1 --input outputs/task1_predictions.json`: check prediction contracts.

## Coding Style & Naming Conventions

Use four-space Python indentation, type hints, `snake_case` functions/modules, and `UPPER_CASE` constants. Ruff is included; no custom lint or formatter configuration is defined. Keep experiment parameters in YAML. Version prompts as `v001.txt`, `v002.txt`, rather than changing prompts used by recorded runs.

## Testing Guidelines

Use pytest with `tests/test_*.py` files and `test_*` functions. Existing test files are placeholders; no coverage threshold is configured. Add synthetic tests for loader safety, prediction schemas, score totals, taxonomy handling, and feedback compliance when changing those behaviors. Run prediction validation alongside baseline smoke checks.

## Commit & Pull Request Guidelines

Git history is unavailable in this checkout. Follow the documented style: `feat(data): add normalized loader` or `fix(task2): correct macro F1`. Use focused branches such as `feat/data-loader`. PRs should explain changes, link relevant issues, include test commands/results, and disclose interface or dataset changes. For experiments, include IDs, configs, seeds, and before/after metrics.

## Data Safety & Reproducibility

Never commit student code, private datasets, secrets, checkpoints, or generated predictions. Exclude feedback from Task 1/2 inputs and retrieve examples only from training data. Record model, split, prompt, hardware, seed, and Git revision for experiments.
