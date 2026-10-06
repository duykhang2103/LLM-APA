# SOURCE_SETUP.md

> **Current scope (2026-10-04):** four-week course project, about two weeks for experiments. Follow the runnable CLI/notebook guide in `docs/project/RUN_EXPERIMENTS.md` and current `PROJECT_PLAN.md`. This document retains historical setup/design/brainstorming; its old placeholder status, hardware/test observations, broad method ladders, and schedules do not establish current execution. P0/P1/F0 are implemented; actual Qwen GPU smoke is still required. SemIf/F1, diagnostic stages, self-consistency and threshold/calibration experiments are not prerequisites. Do not invent EX01 per-question-to-rubric scoring.

# Shared Source / Repository Setup

Tài liệu này định nghĩa cách chuẩn bị source code để toàn bộ nhóm có thể làm chung mà không phá pipeline của nhau.

Mục tiêu:

> Một repository chung, module rõ ràng, experiment tái lập được, task code tách biệt nhưng dùng chung infrastructure.

---

# 1. Repository Structure

Recommended structure:

```text
llm-grading-challenge/
│
├── README.md
├── PROJECT_PLAN.md
├── SOURCE_SETUP.md
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── .env.example
│
├── configs/
│   ├── base.yaml
│   ├── task1/
│   ├── task2/
│   └── task3/
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   ├── splits/
│   └── cache/
│
├── src/
│   └── llm_grading/
│       ├── __init__.py
│       │
│       ├── data/
│       │   ├── loader.py
│       │   ├── schema.py
│       │   ├── preprocess.py
│       │   └── split.py
│       │
│       ├── models/
│       │   ├── loader.py
│       │   ├── inference.py
│       │   └── generation.py
│       │
│       ├── prompting/
│       │   ├── templates.py
│       │   └── formatter.py
│       │
│       ├── retrieval/
│       │   ├── index.py
│       │   └── retriever.py
│       │
│       ├── training/
│       │   ├── trainer.py
│       │   ├── dataset.py
│       │   └── lora.py
│       │
│       ├── evaluation/
│       │   ├── task1.py
│       │   ├── task2.py
│       │   ├── task3.py
│       │   └── common.py
│       │
│       ├── utils/
│       │   ├── seed.py
│       │   ├── logging.py
│       │   ├── config.py
│       │   └── paths.py
│       │
│       └── validation/
│           └── predictions.py
│
├── tasks/
│   ├── task1_grading/
│   │   ├── README.md
│   │   ├── prompts/
│   │   ├── dataset.py
│   │   ├── pipeline.py
│   │   └── postprocess.py
│   │
│   ├── task2_errors/
│   │   ├── README.md
│   │   ├── prompts/
│   │   ├── dataset.py
│   │   ├── pipeline.py
│   │   └── thresholds.py
│   │
│   └── task3_feedback/
│       ├── README.md
│       ├── prompts/
│       ├── dataset.py
│       ├── pipeline.py
│       └── compliance.py
│
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── validate_predictions.py
│
├── experiments/
│   ├── README.md
│   ├── results.csv
│   └── runs/
│
├── tests/
│   ├── test_data_loader.py
│   ├── test_task1_metrics.py
│   ├── test_task2_metrics.py
│   └── test_prediction_schema.py
│
└── reports/
    ├── eda/
    ├── ablations/
    ├── error_analysis/
    └── figures/
```

---

# 2. What Must NOT Be Committed

Raw student data must not be committed to a public repository.

Recommended `.gitignore`:

```gitignore
# Python
__pycache__/
*.py[cod]
.pytest_cache/

# Environments
.venv/
venv/
env/

# Environment variables
.env

# Dataset
data/raw/*
data/processed/*
data/cache/*
!data/README.md
!data/splits/.gitkeep

# Models/checkpoints
checkpoints/
models/
*.pt
*.pth
*.ckpt
*.safetensors

# Outputs
outputs/
predictions*.json

# Logs
logs/
wandb/

# OS/editor
.vscode/
.idea/
.DS_Store
Thumbs.db
```

If split files contain only safe sample IDs and are allowed to be shared internally, they may be committed.

---

# 3. Environment Setup

Recommended Python:

```text
Python 3.10 or 3.11
```

Use one common `uv` environment definition:

```powershell
uv sync --all-groups
```

Run project commands through the same environment:

```powershell
uv run python scripts/predict.py --config configs/task1/heuristic.yaml
uv run pytest
```

---

# 4. Dependency Policy

Do not allow every member to freely upgrade core ML libraries.

Keep the dependency source of truth in `pyproject.toml`, and commit the
generated `uv.lock` after the first successful shared setup. Runtime packages
are in `[project.dependencies]`; development and model-training packages are
in `[dependency-groups]`.

When changing a core dependency:
1. explain why in PR;
2. test all shared scripts;
3. update `pyproject.toml` and `uv.lock`;
4. record compatibility issue if any.

---

# 5. Configuration First

Do not hardcode experiment parameters inside Python files.

Every config is standalone and uses the same readable sections. Example
`configs/task1/heuristic.yaml`:

```yaml
experiment:
  id: "T1-001"
  name: "task1_heuristic_baseline"
  description: "Runnable deterministic baseline"
  tags: [task1, baseline, heuristic]

task: task1
method: heuristic

model:
  name_or_path: null
  revision: null

data:
  input_path: examples/normalized_sample.json
  dataset_version: synthetic-v1
  split_version: synthetic-v1
  max_samples: null

training:
  num_train_epochs: 1
  per_device_train_batch_size: 1
  per_device_eval_batch_size: 1
  gradient_accumulation_steps: 1
  learning_rate: 5.0e-5
  lr_scheduler_type: linear
  warmup_ratio: 0.0

logging:
  logging_steps: 10
  report_to: []

saving:
  output_dir: outputs/T1-001
  save_strategy: no
  save_total_limit: 1

evaluation:
  evaluation_strategy: epoch
  metric_for_best_model: qwk_total
  greater_is_better: true
  load_best_model_at_end: false

generation:
  temperature: 0.0
  max_length: 512
  max_new_tokens: 512
  num_beams: 1

task1:
  use_compile_log: true
  use_test_report: true
  use_retrieval: false
```

The fine-tuning configs use `method: lora` and target
`Qwen/Qwen3.5-4B`; they are an explicit adapter handoff, not the local
heuristic baseline.

---

# 6. Dataset Interface

All tasks should consume one normalized sample schema.

Example conceptual structure:

```python
{
    "sample_id": "...",
    "problem_id": "...",
    "problem_type": "single_problem",
    "problem_statement": "...",
    "code_file": "...",
    "code": "...",
    "compile_log": "...",
    "test_report": "...",
    "rubric": {...},
    "error_labels": [...],
    "feedback": "...",
    "feedback_level": 1
}
```

Important:

Raw loader may contain all fields.

Task-specific dataset classes decide which fields are exposed.

### Supplied dataset format (October 2026)

The loader now supports the teacher's `samples` wrapper and nested `input`/`output` records, alongside existing flat normalized records. Keep `sample_dataset/` unchanged and ignored; place future full data under `data/raw/` with the same layout. Raw task JSON files require sibling `exams.json` and `submissions/` files.

`load_samples(source, max_samples=None, *, task=None)` selects `task1_grading.json`, `task2_error_taxonomy.json`, or `task3_feedback.json` when `source` is a teacher-format directory. Shared CLIs pass the config's task. Multiple task files are not concatenated because their sample IDs overlap.

Normalize `input.exam_id` to `problem_id`, `input.exam_type` to `problem_type`, and the joined exam statement to `problem_statement`. Preserve subproblem metadata and `grading_policy`. Resolve `input.code_file` below the dataset root. Map Task 1 references to `rubric` and a checked `total_score`, Task 2 output `taxonomy_error` to reference `error_labels`, and Task 3 input labels/`target_feedback_level` to known labels/integer `feedback_level`. Reference feedback remains a target. The ten supplied labels are centralized in `src/llm_grading/data/taxonomy.py`.

`validation_path` is canonical; both `--split val` and `--split validation` select it. Sample smoke configs under `configs/task*/sample_heuristic.yaml` use all 32 examples for format checks, not an official split. Existing prediction contracts stay unchanged; see the main README for commands and raw-to-normalized field mappings.

EX01's statement weights (1/4/2/3) conflict with metadata weights (2.5 each). Preserve the source values and clarify rubric aggregation before enforcing prerequisite scoring. Normalization does not imply deterministic policy enforcement by the heuristic.

---

# 7. Leakage Guard

This is mandatory.

For Task 1 and Task 2:

```python
assert "feedback" not in model_input
```

Recommended preprocessing design:

```text
RawSample
   │
   ├── build_task1_input()
   │      └── NO feedback
   │
   ├── build_task2_input()
   │      └── NO feedback
   │
   └── build_task3_input()
          └── feedback only as target during training
```

Add an automated unit test for this.

---

# 8. Dataset Versioning

Each processed dataset should have a version.

Example:

```text
dataset-v1
dataset-v2
```

Record:
- source date
- preprocessing commit
- filtering rules
- split rules

Create:

```text
data/processed/dataset_manifest.json
```

Example:

```json
{
  "version": "v1",
  "seed": 42,
  "preprocessing_commit": "abc123",
  "notes": "Initial normalized dataset"
}
```

---

# 9. Internal Split

Do not optimize only against public leaderboard.

Create an internal validation split.

Rules:
- fixed seed;
- same split for all members;
- committed split IDs if allowed;
- do not silently regenerate split.

Recommended files:

```text
data/splits/train_ids.json
data/splits/val_ids.json
```

If problem leakage is possible, evaluate whether split should be:
- random by submission;
- grouped by problem;
- stratified by score/label.

The selected strategy must be documented.

---

# 10. Common CLI Contract

All shared scripts should use a predictable interface.

## Prepare data

```bash
python scripts/prepare_data.py \
  --config configs/base.yaml
```

## Train

```bash
python scripts/train.py \
  --config configs/task1/qwen_lora.yaml
```

## Evaluate

```bash
python scripts/evaluate.py \
  --config configs/task1/qwen_lora.yaml \
  --split val
```

## Predict

```bash
python scripts/predict.py \
  --config configs/task1/qwen_lora.yaml \
  --split test \
  --output outputs/task1_predictions.json
```

## Validate prediction file

```bash
python scripts/validate_predictions.py \
  --task task1 \
  --input outputs/task1_predictions.json
```

---

# 11. Task Interface

Each task should implement the same conceptual interface:

```python
class TaskPipeline:
    def build_input(self, sample):
        ...

    def predict(self, sample):
        ...

    def postprocess(self, raw_output):
        ...

    def evaluate(self, predictions, references):
        ...
```

This keeps shared scripts generic.

---

# 12. Task 1 Implementation Contract

Task 1 output should always be structured.

Recommended internal object:

```python
{
    "compilable": 0,
    "io_format": 0,
    "logic": 0,
    "edge_case": 0,
    "complexity": 0,
    "code_quality": 0
}
```

Validation must enforce allowed ranges.

Total:

```python
total = (
    compilable
    + io_format
    + logic
    + edge_case
    + complexity
    + code_quality
)
```

Never trust an independently generated total if component scores are available.

---

# 13. Task 2 Implementation Contract

Store taxonomy in one central file.

Example:

```text
src/llm_grading/data/taxonomy.py
```

Do not duplicate label order across scripts.

Use:

```python
ERROR_LABELS = [
    ...
]
```

Thresholds should be config-driven:

```yaml
task2:
  thresholds:
    label_a: 0.52
    label_b: 0.31
```

---

# 14. Task 3 Implementation Contract

Create one centralized level policy.

Example:

```text
tasks/task3_feedback/prompts/level_policy.md
```

Do not let every experiment redefine Level 1–4 informally.

Compliance checker interface:

```python
result = check_compliance(
    feedback=text,
    level=level
)
```

Suggested output:

```python
{
    "pass": True,
    "violations": []
}
```

---

# 15. Prompt Versioning

Prompts are source code.

Store prompts in files instead of notebooks.

Example:

```text
tasks/task1_grading/prompts/v001.txt
tasks/task1_grading/prompts/v002.txt
```

Each experiment records exact prompt version.

Avoid editing a prompt file in-place after it has produced an official experiment.

---

# 16. Model Loader

One shared entry point:

```python
load_model(config)
```

It should handle:
- tokenizer
- base model
- adapters
- dtype
- device
- quantization
- revision

Task files should not manually duplicate loading logic.

---

# 17. Fine-tuning Directory

Recommended:

```text
src/llm_grading/training/
├── dataset.py
├── trainer.py
├── lora.py
└── callbacks.py
```

Task-specific dataset conversion can live under each task, but shared Trainer logic should stay centralized.

---

# 18. Experiment Directory

For every official run:

```text
experiments/runs/T1-003/
├── config.yaml
├── metrics.json
├── metadata.json
└── notes.md
```

Example `metadata.json`:

```json
{
  "experiment_id": "T1-003",
  "owner": "M3",
  "git_commit": "abc123",
  "seed": 42,
  "model": "...",
  "model_revision": "...",
  "dataset_version": "v1",
  "split_version": "v1"
}
```

Large checkpoints should be stored elsewhere and referenced by path.

---

# 19. Experiment Result Registry

Create:

```text
experiments/results.csv
```

Suggested columns:

```text
experiment_id
date
owner
task
method
model
seed
dataset_version
split_version
config_path
primary_metric
primary_score
secondary_metrics
git_commit
checkpoint
notes
```

---

# 20. Logging

Minimum logs for every run:

```text
experiment_id
task
model
seed
device
GPU name
dataset version
number of train samples
number of validation samples
config
start time
final metrics
```

Recommended output:

```text
logs/<experiment_id>.log
```

---

# 21. Seed Utility

Use one common helper.

Conceptual:

```python
def set_seed(seed):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Also configure deterministic behavior where practical.

---

# 22. Evaluation Modules

## Task 1
Implement:
- QWK
- MAE
- per-dimension exact match

## Task 2
Implement:
- macro-F1
- micro-F1
- per-label precision/recall/F1

## Task 3
Implement official/available evaluation logic and internal diagnostics.

Never copy metric implementations into individual notebooks.

---

# 23. Unit Tests

Minimum required tests:

```text
test_data_loader
test_missing_code_file
test_no_feedback_leakage_task1
test_no_feedback_leakage_task2
test_task1_score_range
test_task1_total
test_task2_label_schema
test_prediction_schema
test_seed
```

Run:

```bash
pytest
```

before merging core changes.

---

# 24. Git Workflow

## Branch naming

```text
feat/data-loader
feat/task1-zero-shot
feat/task2-thresholds
feat/task3-compliance
feat/lora-training
fix/task1-qwk
refactor/model-loader
```

## Rules
- no direct coding on `main`;
- one feature/fix per branch where possible;
- PR before merge;
- keep branches short-lived;
- pull/rebase frequently.

---

# 25. Commit Style

Recommended:

```text
feat(task1): add structured rubric parser
feat(data): add normalized dataset loader
fix(task2): correct macro F1 evaluation
exp(task3): add compliance checker v2
refactor(model): centralize tokenizer loading
docs: update reproduction guide
```

---

# 26. Pull Request Template

Create:

`.github/pull_request_template.md`

Suggested:

```markdown
## Summary

## Task
- [ ] Shared
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

## Experiment ID

## What changed?

## Why?

## Tests

## Metrics before

## Metrics after

## Interface changes?

## Dataset/schema changes?

## Checklist
- [ ] No secret committed
- [ ] No private dataset committed
- [ ] No feedback leakage into Task 1/2
- [ ] Tests pass
- [ ] Config added/updated
```

---

# 27. Code Ownership

Recommended soft ownership:

```text
src/llm_grading/data/          → M2 + M1
src/llm_grading/training/      → M6 + M1
tasks/task1_grading/           → M3
tasks/task2_errors/            → M4
tasks/task3_feedback/          → M5
src/llm_grading/evaluation/    → task owner + M1
scripts/                       → M1
```

Owners review important changes in their area.

---

# 28. API Keys / Secrets

Use `.env`:

```text
OPENAI_API_KEY=
OTHER_API_KEY=
```

Commit only:

```text
.env.example
```

Never commit actual keys.

---

# 29. Checkpoint Storage

Do not use Git for model checkpoints.

Use:
- local disk
- private Drive
- Hugging Face private repo if permitted
- internal storage

Maintain a mapping:

```text
experiment → checkpoint location
```

---

# 30. Data Storage

Recommended:

```text
data/raw/
```

contains the original challenge data and is ignored by Git.

`data/README.md` explains how each team member should place it.

Example:

```text
data/raw/
├── submissions/
├── train.json
├── dev.json
└── ...
```

Exact filenames should be updated after inspecting the distributed dataset.

---

# 31. README Minimum Content

Root README should eventually include:

1. Project overview.
2. Setup.
3. Dataset placement.
4. Environment installation.
5. Train commands.
6. Evaluation commands.
7. Prediction commands.
8. Reproduction of best run.
9. Hardware requirements.
10. Team contribution.

---

# 32. Definition of Done for a New Method

A method is not "done" just because code runs.

It is done only when:

- [ ] code merged
- [ ] config saved
- [ ] experiment ID assigned
- [ ] seed recorded
- [ ] metric recorded
- [ ] git commit recorded
- [ ] output saved
- [ ] result added to registry
- [ ] notes explain result
- [ ] reproducible command exists

---

# 33. First-Day Setup Checklist

M1:
- [ ] create repository
- [ ] create folder structure
- [ ] add README
- [ ] add `.gitignore`
- [ ] add `.env.example`
- [ ] create base config
- [ ] create PR template

M2:
- [ ] inspect dataset
- [ ] document raw directory layout
- [ ] implement initial loader
- [ ] create schema

M3/M4/M5:
- [ ] read task definition
- [ ] define task output schema
- [ ] prepare first baseline prompt

M6:
- [ ] verify GPU environment
- [ ] shortlist 2–3 open-weight models
- [ ] build model loader prototype

---

# 34. End-of-Week-1 Source Checklist

The repository should support:

```bash
python scripts/prepare_data.py --config configs/base.yaml
pytest
```

And preferably:

```bash
python scripts/evaluate.py \
  --task task1 \
  --predictions path/to/dummy.json
```

At this stage, model quality is not important.

Infrastructure correctness is.

---

# 35. Integration Rule

Before adding an advanced technique:

> Can it plug into the common pipeline without task-specific manual steps?

If no, redesign the interface first.

The final goal is:

```text
config
  ↓
shared CLI
  ↓
task pipeline
  ↓
model
  ↓
postprocess
  ↓
evaluation/prediction
```

not:

```text
member-specific notebook
  ↓
manual edits
  ↓
copy-paste result
```

---

# 36. Recommended Initial GitHub Issues

Create these issues immediately:

```text
[DATA] Inspect raw dataset structure
[DATA] Implement normalized loader
[DATA] Build initial EDA
[SHARED] Define config schema
[SHARED] Implement experiment registry
[SHARED] Implement prediction validator
[T1] Implement QWK/MAE evaluation
[T1] Build zero-shot baseline
[T2] Implement macro/micro F1
[T2] Build zero-shot baseline
[T3] Define Level 1–4 policy
[T3] Build feedback baseline
[MODEL] Build shared model loader
[MODEL] Implement LoRA/QLoRA training
[TEST] Add leakage tests
[DOC] Create reproduction guide
```

Assign each issue to one primary owner.

---

# 37. Final Principle

The repository should make it difficult to produce an unreproducible experiment.

Every official result should answer:

```text
Which code?
Which data?
Which split?
Which model?
Which prompt?
Which config?
Which seed?
Which hardware?
Which metric?
Which checkpoint?
```

If those questions can be answered from the repository, the group is in a strong position for both leaderboard performance and the reproducibility/report components of the course.
