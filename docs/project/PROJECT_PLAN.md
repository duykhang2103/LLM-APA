# PROJECT_PLAN.md

# LLM Challenge — Automated Grading and Feedback for Programming Assignments

## 1. Project Overview

### Course
Large Language Model

### Challenge
Xây dựng hệ thống dựa trên LLM để:

1. Chấm điểm bài lập trình theo rubric.
2. Phân loại lỗi đa nhãn.
3. Sinh phản hồi tiếng Việt có kiểm soát mức độ hỗ trợ.

### Team size
5–7 thành viên.

### Duration
8 tuần.

### Main constraints
- Pipeline phải chạy tự động, không can thiệp thủ công vào từng bài khi inference.
- Phải cố định seed và ghi rõ model/version.
- Phải triển khai và so sánh ít nhất:
  - một hướng prompting trên LLM;
  - một open-weight model được fine-tune và chạy local.
- Không được dùng `feedback` làm input cho Task 1 và Task 2.
- Kết quả tốt nhất phải tái lập được.
- Private leaderboard là leaderboard chính thức.

---

# 2. Problem Definition

## Task 1 — Rubric Grading

### Input
- Problem statement
- Student C++ code
- Optional:
  - `compile_log`
  - `test_report`

### Output
6 rubric dimensions:

| Dimension | Range |
|---|---:|
| compilable | 0–1 |
| io_format | 0–1 |
| logic | 0–4 |
| edge_case | 0–2 |
| complexity | 0–1 |
| code_quality | 0–1 |

Total score: 0–10.

### Metrics
Primary:
- QWK on total score

Secondary:
- MAE
- Exact-match per rubric dimension

### Important constraint
Đối với `multi_problem`, phải tôn trọng dependency/rule của đề. Ví dụ câu tiên quyết sai thì câu sau có thể không được tính điểm.

---

## Task 2 — Multi-label Error Classification

### Input
Giống Task 1.

### Output
Tập nhãn lỗi trong taxonomy 10 nhãn.

Empty label set là hợp lệ.

### Metrics
Primary:
- macro-F1

Secondary:
- micro-F1

### Main challenge
- Label imbalance mạnh.
- Có thể tồn tại nhãn rất hiếm hoặc gần suy biến.
- Cần threshold tuning/calibration theo từng label.

---

## Task 3 — Controlled Feedback Generation

### Input
- Problem statement
- Student code
- Error labels
- Required feedback level

### Output
Vietnamese feedback.

### Main evaluation dimensions
1. Diagnosis correctness.
2. Feedback-level compliance.

### Core constraint
Ở Level 1/2:
- không được cung cấp lời giải đầy đủ;
- không được đưa code sửa hoàn chỉnh.

Task 3 phải coi feedback level là hard constraint, không chỉ là style prompt.

---

# 3. Recommended Team Structure

## Team of 6

| Member | Main ownership | Secondary ownership |
|---|---|---|
| M1 | Technical Lead + Integration | CI, final submission, experiment tracking |
| M2 | Data + EDA + Dataset Audit | Dataset/metric issue investigation |
| M3 | Task 1 Rubric Grading | Multi-problem dependency |
| M4 | Task 2 Error Classification | Imbalance + threshold tuning |
| M5 | Task 3 Feedback Generation | Compliance checker |
| M6 | Fine-tuning + LLM Infrastructure | Model serving/inference |

Ownership không đồng nghĩa làm độc lập. Shared infrastructure phải dùng chung.

---

## If team has 5 members

Merge:
- M1 + M6

Suggested:
- M1: Lead + Fine-tuning + Integration
- M2: Data
- M3: Task 1
- M4: Task 2
- M5: Task 3

---

## If team has 7 members

Add:

### M7 — Retrieval / Evaluation / Research
Responsibilities:
- RAG/retrieval experiments
- LLM-as-judge utilities where appropriate
- error-analysis tooling
- literature comparison
- metric audit
- dataset issue investigation
- leaderboard experiment support

---

# 4. Shared Architecture

```text
                       Problem
                         │
                         ▼
                 Problem Analyzer
                         │
            ┌────────────┴────────────┐
            │                         │
        Student Code            Auxiliary Signals
                                compile_log/test_report
            │                         │
            └────────────┬────────────┘
                         ▼
                Submission Analysis
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
   Rubric Grading   Error Taxonomy   Feedback Generator
      Task 1           Task 2            Task 3
          │              │              │
          ▼              ▼              ▼
  Rule/Constraint   Thresholding    Level Compliance
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                     Outputs
```

Principle:

> LLM xử lý semantic reasoning; deterministic code xử lý constraint có thể biểu diễn chính xác.

---

# 5. Shared Components

Tất cả task phải dùng chung nếu có thể:

- dataset loader
- problem loader
- source-code loader
- model loader
- prompt templates
- config system
- random seed utility
- logging
- experiment registry
- prediction schema
- evaluation utilities
- checkpoint convention
- CLI
- output validation

Không cho phép mỗi task tự xây một pipeline hoàn toàn khác.

---

# 6. Workstream Responsibilities

## M1 — Technical Lead + Integration

### Responsibilities
- Define repository architecture.
- Define interfaces giữa shared modules và task modules.
- Unified CLI.
- Config system.
- Reproducibility.
- Output validation.
- Integration.
- Final prediction generation.
- CI/basic tests.
- Merge strategy.

### Key deliverables
- `scripts/train.py`
- `scripts/evaluate.py`
- `scripts/predict.py`
- config loader
- seed utility
- output validator
- experiment registry
- final end-to-end pipeline

### Definition of Done
Một task phải chạy được kiểu:

```bash
uv run python scripts/train.py --config configs/task1/heuristic.yaml
uv run python scripts/predict.py \
  --config configs/task1/heuristic.yaml \
  --split test \
  --output outputs/task1_predictions.json
uv run python scripts/evaluate.py \
  --config configs/task1/heuristic.yaml \
  --predictions outputs/task1_predictions.json
```

---

## M2 — Data + EDA + Dataset Audit

### Responsibilities

#### Dataset profiling
- number of samples
- number of problems
- single vs multi problem
- code length
- score distribution
- rubric distribution
- label distribution
- label co-occurrence
- feedback-level distribution
- compile success rate
- test-report availability

#### Validation
Check:
- missing code files
- broken `code_file`
- duplicate samples
- duplicate/near-duplicate code
- rubric sum mismatch
- invalid score
- invalid taxonomy label
- empty/malformed fields
- compile log vs compilable inconsistency
- test report vs label/score inconsistency

#### Leakage prevention
Task 1/2 preprocessing must explicitly drop feedback.

#### Bonus investigation
Search for:
- near-degenerate labels
- inconsistent annotations
- skewed score distributions
- metric weaknesses
- duplicate leakage
- problem-specific shortcuts
- annotation-policy conflicts

### Outputs
- `reports/eda.md`
- `reports/data_audit.md`
- `data/processed/`
- `data/splits/`
- plots/tables for report

---

## M3 — Task 1 Rubric Grading

### Baseline ladder

1. Simple heuristic baseline.
2. Zero-shot LLM.
3. Structured zero-shot.
4. Few-shot.
5. Retrieval-based few-shot.
6. Fine-tuned open-weight model.
7. Fine-tuned + deterministic constraints.

### Output format
Prefer structured JSON:

```json
{
  "compilable": 1,
  "io_format": 1,
  "logic": 3,
  "edge_case": 1,
  "complexity": 1,
  "code_quality": 1
}
```

Total score must be computed deterministically.

### Special work
Multi-problem:

```text
Problem
  ↓
Subproblem extraction
  ↓
Per-subproblem evaluation
  ↓
Dependency/policy rules
  ↓
Final rubric
```

### Evaluation
Track:
- QWK
- MAE
- exact match per dimension
- performance by problem type
- performance by score range

---

## M4 — Task 2 Error Classification

### Baseline ladder
1. Zero-shot classification.
2. Structured prompting.
3. Few-shot.
4. Fine-tuned classification/generation.
5. Class weighting/resampling.
6. Per-label threshold tuning.
7. Optional ensemble/rules.

### Important experiment
Compare:

```text
global threshold = 0.5
vs
per-label tuned thresholds
```

### Analysis
Report:
- per-label precision
- per-label recall
- per-label F1
- macro-F1
- micro-F1
- support count
- rare-label behavior
- co-occurring label errors

---

## M5 — Task 3 Feedback Generation

### Feedback policy

#### Level 1
Allowed:
- nhẹ nhàng chỉ ra vùng cần xem lại;
- concept hint.

Forbidden:
- exact fix;
- corrected code;
- full solution.

#### Level 2
Allowed:
- identify mistake more clearly;
- suggest direction/concept.

Forbidden:
- full implementation;
- complete answer.

#### Level 3
Allowed:
- detailed correction strategy;
- pseudocode if policy allows.

Avoid:
- unnecessary full solution.

#### Level 4
Allowed:
- explicit solution;
- detailed fix;
- code example where appropriate.

### Suggested architecture

```text
Generator
   ↓
Candidate Feedback
   ↓
Compliance Checker
   ├── PASS → output
   └── FAIL → regenerate/revise
```

### Ablation
Compare:
- generator only
- stronger system prompt
- generator + checker
- generator + checker + regeneration

---

## M6 — Fine-tuning / Model Infrastructure

### Responsibilities
- shortlist open-weight models <= recommended size
- tokenizer
- dataset formatting
- LoRA/QLoRA
- trainer
- checkpoints
- eval hooks
- inference
- VRAM usage
- training time
- experiment configs
- hardware report

### Shared trainer principle

```text
shared trainer
    ├── task1 config
    ├── task2 config
    └── task3 config
```

Avoid separate unrelated training stacks for each task.

---

# 7. Experimental Ladder

Use a common progression:

```text
B0 — simple heuristic / majority baseline
B1 — zero-shot LLM
B2 — structured zero-shot
B3 — few-shot
B4 — retrieval-based few-shot
B5 — fine-tuned open-weight model
B6 — fine-tuned + task-specific constraints
B7 — best combined system
```

Every improvement must be compared against a previous baseline.

---

# 8. Experiment Naming Convention

Use:

```text
T{task}-{number}
```

Examples:

```text
T1-001
T1-002
T2-001
T3-005
```

Each experiment stores:

- owner
- date
- git commit
- dataset version
- split version
- model
- model revision
- seed
- config
- prompt version
- hardware
- metric
- checkpoint
- notes

---

# 9. Experiment Registry

Recommended:

`experiments/results.csv`

Columns:

```text
experiment_id
task
owner
method
model
model_revision
seed
dataset_version
split_version
config
primary_metric
primary_score
secondary_metrics
checkpoint
git_commit
notes
```

Never use names such as:

```text
final_v2_best_REAL_final_new
```

---

# 10. Parallel Workstreams and Delivery Gates

The project still has an eight-week course constraint, but the work should be organized by responsibilities and deliverables instead of forcing prompting to happen before fine-tuning. After the shared data and evaluation contracts are agreed, both technical groups start in parallel.

## 10.1 Shared foundation

The shared owners support both groups. They should provide:

- normalized sample schema and loader;
- safe resolution of `code_file` references;
- fixed train/validation split and split version;
- explicit Task 1/2 input builders that exclude `feedback`;
- QWK, MAE, exact-match, macro-F1, micro-F1, and Task 3 compliance diagnostics;
- prediction schema and validator;
- config loading, seed setup, logging, and experiment registry;
- common CLI contract for prepare, train, evaluate, predict, and validate.

### Beginner output example

Before modeling, one sample should be understandable in a normalized shape:

```python
{
    "sample_id": "example-001",
    "problem_id": "EX01",
    "problem_type": "single_problem",
    "problem_statement": "...",
    "code": "...",
    "compile_log": "...",
    "test_report": "...",
    "rubric": {"logic": 3},
    "error_labels": [],
    "feedback": "...",
    "feedback_level": 2,
}
```

Task 1 and Task 2 builders must remove `feedback` before creating model input. Task 3 may use feedback as a training target, not as an input feature.

## 10.2 Workstream A: Prompting and Retrieval

### Goal

Build a strong baseline without changing model weights, then improve it with structured prompts, few-shot examples, retrieval, and deterministic post-processing.

### Recommended order

1. Zero-shot prompt for all three tasks.
2. Structured JSON output.
3. Output parser and validator.
4. Few-shot examples from the training split only.
5. Similarity-based RAG from the training split only.
6. Task-specific constraints: prerequisite rules, per-label thresholds, and feedback compliance.
7. Versioned prompts and experiment records.

### Task outputs

- Task 1: six component scores; compute total in code.
- Task 2: a valid subset of the central ten-label taxonomy; empty set allowed.
- Task 3: Vietnamese feedback plus compliance-check result.

### Definition of done

- zero-shot baseline exists for all three tasks;
- prompts are versioned and reproducible;
- few-shot/RAG examples are training-only;
- structured output invalid-rate is measured;
- Task 3 Level 1/2 violations are detected and recorded;
- one prompting candidate can generate a valid prediction file.

## 10.3 Workstream B: Fine-tuning and Model Infrastructure

### Goal

Select an open-weight model that fits the team's hardware, train a local LoRA/QLoRA adapter, and compare it fairly against Workstream A.

### Recommended order

1. Inventory GPU, VRAM, RAM, operating system, and available run time.
2. Benchmark two or three candidate models with the same smoke input.
3. Select a model under the course size recommendation and record its license/revision.
4. Convert normalized samples into task-specific instruction/response records.
5. Run a short LoRA/QLoRA pilot.
6. Confirm the adapter loads for local inference.
7. Train separate task adapters first; consider multi-task training only after the separate baselines are understood.
8. Compare against the best prompting candidate using the same split, metrics, and output rules.

### Definition of done

- at least one open-weight model runs locally;
- at least one fine-tuning pilot completes;
- checkpoint and model revision are recorded outside Git;
- training data and filtering rules are documented;
- hardware, VRAM, time, cost, and latency are recorded;
- fine-tuned predictions pass the same validator as prompting predictions.

## 10.4 Cross-workstream integration gates

These gates are deliverables, not calendar weeks.

### Gate A: shared contract

Both groups use the same normalized fields, split IDs, metrics, prediction schema, seed policy, and config convention.

### Gate B: comparable baselines

Both groups can create a prediction file for at least one task and evaluate it with the same script.

### Gate C: parallel candidates

Workstream A has a versioned prompting/RAG candidate. Workstream B has a locally running fine-tuned candidate. Each has an experiment ID and metadata.

### Gate D: fair comparison

Compare quality, latency, VRAM, cost, and reproducibility under the same input signals, split, output constraints, and evaluator.

### Gate E: final selection

Choose the final system using leaderboard results when available, local metrics, ablations, error slices, implementation risk, and reproducibility. Do not select solely by one metric.

## 10.5 Shared ablation and error-analysis checklist

The team should compare:

- code only vs code plus compile log vs code plus test report;
- without retrieval vs fixed few-shot vs similarity RAG;
- free output vs structured output plus validator;
- without multi-problem dependency rules vs deterministic rules;
- global Task 2 threshold vs per-label thresholds;
- Task 3 generator only vs generator plus checker vs checker plus regeneration;
- best prompting candidate vs best fine-tuned candidate.

Analyze slices for compile failure, logic failure, edge-case failure, single/multi-problem type, rare labels, low/high scores, and short/long code.

## 10.6 Delivery checklist

The work is ready for finalization when:

- both technical approaches run end-to-end;
- the best run for each approach is reproducible from config and seed;
- prediction files pass validation;
- ablation results explain which components help;
- error analysis explains where the system fails and why;
- the report, model card, slides, predictions, and contribution table are ready;
- no private data, secrets, or checkpoints are committed.

# 11. Weekly Team Meeting Format

Recommended 2 meetings/week.

## Short sync — 20–30 minutes

Each member answers:

1. What was completed?
2. What is the current metric?
3. What changed compared with baseline?
4. What is blocked?
5. What will be completed next?

No long debugging during sync.

---

## Experiment review — 45–60 minutes

Review:
- leaderboard/local metrics
- failed experiments
- ablation results
- data issues
- integration
- next experiments

---

# 12. Pull Request Policy

Every meaningful change uses PR.

PR must state:

```text
What changed?
Why?
Which task?
Experiment ID?
Expected behavior?
How was it tested?
Does it change prediction format?
Does it change dataset format?
```

For core interfaces:
- minimum one reviewer;
- M1 approval recommended.

---

# 13. Branch Strategy

Recommended lightweight strategy:

```text
main
│
├── feat/data-eda
├── feat/task1-rubric
├── feat/task2-errors
├── feat/task3-feedback
├── feat/training
└── fix/...
```

Rules:
- `main` should stay runnable.
- No direct work on `main`.
- Small PRs preferred.
- Rebase/merge frequently to prevent long-lived divergence.

---

# 14. Reproducibility Checklist

Every official experiment must record:

- seed
- model name
- exact model revision if possible
- tokenizer version
- package lock
- hardware
- dataset version
- split version
- prompt version
- config
- git commit
- inference parameters

---

# 15. Data Security Rules

Do not:
- publish student submissions
- commit raw private dataset to public repo
- identify students
- manually label private test
- share predictions across groups

Use:
- `.gitignore`
- local/private storage
- environment variables for API keys

---

# 16. Report Ownership

## M1
- architecture
- system integration
- reproducibility
- deployment

## M2
- dataset
- EDA
- dataset issues

## M3
- Task 1
- Task 1 ablation/error analysis

## M4
- Task 2
- Task 2 ablation/error analysis

## M5
- Task 3
- pedagogical control analysis

## M6
- fine-tuning
- hardware/cost
- model card

One final editor merges style and terminology.

---

# 17. Final Success Criteria

Project is considered ready when:

- [ ] 3 tasks run end-to-end.
- [ ] Prompting baseline exists.
- [ ] Fine-tuned open-weight model exists.
- [ ] All main metrics implemented.
- [ ] Best model is reproducible.
- [ ] Ablation table completed.
- [ ] Error analysis completed.
- [ ] Dataset audit completed.
- [ ] Prediction files pass validator.
- [ ] Task 1/2 contain no feedback leakage.
- [ ] Model card completed.
- [ ] Report and slides completed.
- [ ] Contribution table agreed by team.
