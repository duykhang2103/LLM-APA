# Automated Programming Assessment

Scaffold repository for the LLM challenge that grades anonymized C++ submissions, classifies errors, and generates controlled Vietnamese feedback.

## Current status

This repository currently contains structure and ownership boundaries only. The Python modules, CLI scripts, task pipelines, metrics, model loading, and data processing are intentionally incomplete. Future contributors should replace the marked placeholders incrementally while preserving the shared interfaces.

## Project tasks

- Task 1: rubric grading with six component scores and a deterministic total.
- Task 2: multi-label error classification over the official ten-label taxonomy.
- Task 3: Vietnamese feedback generation controlled by Level 1–4 policy.

## Planned repository flow

```text
config -> shared CLI -> task pipeline -> model/prompt -> postprocess -> evaluation/validation
```

## Data security

Student submissions are private course data. Keep raw data under `data/raw/`, do not commit it, do not publish it, and do not use it outside the challenge. Never commit API keys, checkpoints, predictions, or private test annotations.

## Future setup

1. Create Python 3.10 or 3.11 environment.
2. Install only the verified dependencies recorded in `requirements.txt`.
3. Place challenge data according to `data/README.md`.
4. Implement and review the normalized schema before adding task logic.
5. Reproduce experiments through the shared CLI and record metadata in `experiments/`.

## Future commands

```text
python scripts/prepare_data.py --config configs/base.yaml
python scripts/train.py --config configs/task1/qwen_lora.yaml
python scripts/evaluate.py --config configs/task1/qwen_lora.yaml --split val
python scripts/predict.py --config configs/task1/qwen_lora.yaml --split test --output outputs/task1_predictions.json
python scripts/validate_predictions.py --task task1 --input outputs/task1_predictions.json
```

These commands are documented targets for later implementation; they are not expected to run in the scaffold stage.

## References

- `PROJECT_PLAN.md`: ownership, milestones, experiments, and completion criteria.
- `SOURCE_SETUP.md`: repository contracts and contributor workflow.
- `DoAn-LLM-Challenge.docx`: official challenge description.
