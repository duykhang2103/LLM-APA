# Task 1 Qwen QLoRA Baseline Design

Date: 2026-10-03

## Intent and approval

Implement the course project's Task 1 rubric-grading baseline with
`Qwen/Qwen3.5-4B`, using supervised fine-tuning and local inference. The
main project contract is `README.md`; `docs/project/` supplies supplementary
requirements. The user approved the proposed design and selected this
computer as the training machine.

Success means a reproducible prepare/train/predict/evaluate/validate workflow
that trains an adapter, reloads it, and produces the existing Task 1 prediction
format. A working training implementation and a successful GPU training run
are separate deliverables and must be reported separately.

This document captures the approved design for written review. Implementation
planning follows that review; implementation and dependency installation
follow review of the implementation plan.

## Observed starting point

- The teacher-format loader resolves exam metadata and C++ files safely.
- Task 1 already has an input whitelist, bounded integer score validation,
  deterministic totals, QWK, total MAE, and component exact match.
- The model loader, inference utilities, training dataset converter, LoRA
  configuration, and trainer are placeholders. Non-heuristic pipelines fail
  explicitly. The training CLI currently writes a heuristic manifest only.
- There are 32 demonstration submissions: 15 EX01 and 17 EX02. All references
  have six rubric components and consistent totals. No exact duplicate source
  files were found. Four submissions have compiler diagnostics despite a
  teacher-assigned `compilable = 1`.
- Hardware: NVIDIA GeForce RTX 4050 Laptop GPU, 6141 MiB VRAM, Windows x86-64,
  Python 3.11.9. Installed PyTorch is `2.14.1+cpu`; CUDA is unavailable to it.
  `bitsandbytes` is absent. The NVIDIA driver reports version 616.92.
- Existing verification: 35 tests and 5 subtests passed before this design.
  This is evidence for the current infrastructure, not Qwen training.
- The workspace contains pre-existing modified and untracked files. Preserve
  those changes and isolate subsequent implementation according to the
  repository workflow.

## Approach and scope

Use instruction-to-JSON supervised fine-tuning with one Task 1 PEFT adapter.
The base model is frozen and loaded with 4-bit NF4 quantization. Train LoRA
parameters and calculate the language-model loss only on the assistant answer.
Use the same versioned prompt and chat formatting for training and inference.

This approach preserves the current output contract and supports a later
comparison against the unadapted model using identical inputs. Six separate
classification heads would introduce another training and inference interface.
Unquantized LoRA requires more base-weight memory and is not the local default.

Include shared model loading, adapter training and reload, Task 1 integration,
pilot data preparation, evaluation, run metadata, tests, and reproduction
documentation. Tasks 2 and 3 retain their current behavior and unsupported
model methods remain explicit errors. Retrieval, multi-task training, synthetic
labels, full-weight fine-tuning, and leaderboard submission are outside this
baseline's scope.

Component responsibilities follow the existing repository boundaries:

| Boundary | Responsibility |
|---|---|
| `tasks/task1_grading/dataset.py` and `prompts/` | Allowed input, rubric target, versioned grading instructions |
| `src/llm_grading/training/dataset.py` | Chat tokenization, answer masks, padding, token coverage |
| `src/llm_grading/training/lora.py` and `trainer.py` | Adapter settings, supervised optimization, checkpoints |
| `src/llm_grading/models/` | Base model/tokenizer loading, adapter reload, generation |
| `tasks/task1_grading/pipeline.py` and `postprocess.py` | Method selection, response parsing, retries, deterministic totals |
| `src/llm_grading/data/split.py` and runtime utilities | Persisted pilot splits, overlap checks, manifests |
| `src/llm_grading/evaluation/task1.py` | Shared grading metrics and exam-type slices |
| `scripts/` | Prepare, train, predict, evaluate, validate orchestration |

## Data contract and pilot splits

Read `sample_dataset/` through the existing loader without changing the supplied
files. Keep raw data, processed records, predictions, logs, and checkpoints in
ignored local directories. The sample set is a format-demonstration dataset;
pilot results cannot establish official validation or leaderboard performance.

Extend `scripts/prepare_data.py` with an explicit `--training` mode for the
pilot split and training-record preparation. Keep its existing format-audit
mode. Extend `scripts/train.py` with `--dry-run` to inspect records, split
integrity, and package metadata without downloading weights or optimizing.
Full training performs the CUDA preflight; dry-run success does not establish
GPU readiness. The existing `--config`, `--split val`, `--output`, and
`--predictions` CLI arguments keep their meanings.

Training preparation writes normalized `train.json`, `validation.json`, split
IDs, and a data manifest below `data/processed/task1-sample-pilot-v1/`. Use seed
42 and a validation fraction of 0.20 within each exam. Shuffle groups with a
local seeded random generator; group identical source after normalizing line
endings and trailing line whitespace. Round the validation target to the
nearest integer, keep at least one group in each split when possible, and
record actual counts when a duplicate group prevents an exact target. For
unique records this produces 26 train and 6 validation submissions. There is
no manufactured test split.

The manifest records source checksums, normalized code hashes, seed, grouping
strategy, dataset and split versions, sample IDs, counts, and the sample-only
limitation. Persist and reuse these splits. A later prepare command verifies
the source and settings against the manifest and fails on a mismatch instead
of silently regenerating the split. Detect overlap and duplicate source groups
across explicitly supplied train/validation paths too. Near-duplicate code
detection is a documented limitation of this baseline.

Learned-model training requires explicit, nonempty train and validation paths;
it cannot fall back to the entire input dataset. Prediction requires the
requested split path, except that an explicitly requested input-file mode may
use `input_path`. Keep the existing heuristic fallback behavior compatible.
Future official train/validation/test paths replace the pilot paths without
changing the training or prediction interfaces.

## Prompt, targets, and tokenization

Add an immutable Task 1 prompt version that instructs the model to grade C++
according to the six rubric dimensions and return one JSON object. State that
compiler logs are auxiliary evidence and that function-only submissions can
be valid. Include the exam policy and prerequisite metadata, preserving their
source values. Treat submitted code as evidence rather than instructions.

Construct inputs through `build_task1_input`, with configurable inclusion of
compile logs and test reports. The prompt can contain only the problem
statement, code, language, problem type, subproblem metadata, grading policy,
and allowed evidence. Reference rubric, total, error labels, feedback, sample
IDs, and source-file paths are excluded. Do not join other task files to obtain
extra input features.

Validate targets before conversion and serialize exactly these six fields in
their shared rubric order: `compilable`, `io_format`, `logic`, `edge_case`,
`complexity`, `code_quality`. Do not train the model to generate `total`.
Retain the sample ID alongside each training record for audit purposes.

Use the Qwen tokenizer's chat template with `enable_thinking=False`. Build the
inference prompt with the assistant generation prefix; append the JSON answer
and end-of-turn token for training. Verify the token boundary rather than
assuming separately tokenized strings concatenate identically. Mask system,
user, assistant prefix, and padding labels with `-100`; supervise the complete
JSON answer and its end-of-turn token. A row with no supervised answer tokens
is invalid. Padding must not mask real end-of-turn tokens when token IDs overlap.

Default source budget: 4096 tokens, including instructions and chat framing.
Default target budget: 128 tokens. Reject over-budget source or target with the
sample ID and measured lengths. Do not silently truncate the C++ program,
grading policy, or JSON target. Report token-length coverage before training.
Changing budgets creates a recorded configuration change, and sufficient
VRAM for these budgets must be established by the pilot.

## Model loading and local environment

Keep model-library imports behind the shared model boundary so heuristic
commands, data preparation, and CPU contract tests do not require a GPU or
download weights. Return an explicit model/tokenizer bundle with resolved
revision, device, dtype, and quantization metadata.

Use the text-only `Qwen3_5ForCausalLM` path and the checkpoint's text
configuration. Verify the installed Transformers version can map the original
checkpoint's language-model weights correctly; do not train randomly
initialized parameters or substitute a different Qwen model. Exclude the
vision encoder from this text task. Unexpected missing backbone weights are
fatal. Preserve tied input/output embeddings.

Preflight requires CUDA-enabled PyTorch, GPU availability, compatible Qwen3.5
Transformers support, PEFT, Accelerate, and `bitsandbytes`. Document and verify
the CUDA-wheel installation through the repository's `uv` workflow, then
update dependency declarations and `uv.lock` together. Keep the tested exact
package versions in the lockfile and run manifest. Prefer native PyTorch
fallbacks over requiring FlashAttention or platform-specific DeltaNet builds
for the initial Windows implementation; record which attention path is used.

Use BF16 on hardware that supports it, otherwise FP16 with appropriate mixed
precision handling. For local quantized training, place the model on one CUDA
device explicitly; inference-style automatic CPU/disk dispatch is not an
unreported training workaround. Enable gradient checkpointing and disable the
KV cache during training. Verify dtype and memory after k-bit preparation;
avoid allocating full-sequence vocabulary logits unnecessarily when only the
short assistant answer contributes to loss. A memory optimization must preserve
the same causal answer loss and receive a numerical equivalence check.

Resolve a model revision to an immutable commit before loading the model and
tokenizer, save it, and use that same revision for adapter reload. A saved
adapter must carry the tokenizer and prompt identity needed for inference.
Downloading public model artifacts is allowed within the local workflow;
student source and labels are never sent to remote inference or logging.

## Training configuration and artifacts

Retain `method: lora`; quantization settings make the experiment QLoRA. Update
`configs/task1/qwen_lora.yaml` for experiment `T1-101`, and add a separate short
pilot configuration `T1-100`. Both target `Qwen/Qwen3.5-4B` and the same prepared
split. All choices are configuration-driven.

Local starting values are 4-bit NF4 with double quantization, LoRA rank 8,
alpha 16, dropout 0.05, all linear layers in the text backbone, batch size 1,
gradient accumulation 4, learning rate `1.0e-4`, linear scheduler, warmup ratio
0.06, seed 42, and three epochs for `T1-101`. Do not attach adapters to the
embedding or vocabulary output head. The pilot runs two optimizer steps and
uses the shortest available training example from each exam type, recording
that selection. Validation remains outside optimization. Both configurations
initially use the token budgets above.

Use the shared trainer boundary with a standard Transformers Trainer, a
completion-masked dataset, and a dynamic-padding collator. Verify that only
adapter parameters train. Save adapters and tokenizers rather than duplicating
base weights. Save epoch checkpoints with a limit of two. Select the checkpoint
by validation loss for this first baseline, then evaluate its generated grades
with the existing Task 1 evaluator. Do not present token argmax metrics as QWK.

Write artifacts below the configured output directory: resolved configuration,
run manifest, training metrics, adapter/tokenizer, generated validation
predictions, grading metrics, and generation diagnostics. The manifest includes
experiment ID, seed, Git commit and dirty state, environment versions, resolved
model/tokenizer revisions, prompt identity, dataset/split manifest checksums,
training and validation IDs, actual token budgets, trainable parameter count,
GPU/VRAM, dtype, quantization, elapsed time, and peak allocated/reserved memory.
Record run states as running, completed, or failed; a failed run cannot receive
a completed status. Result registry entries contain aggregate metrics and
artifact paths, without student code.

## Inference, parsing, and evaluation

The Task 1 pipeline chooses the existing heuristic or the configured model
method. Load the model once for prediction. Training uses the shared trainer
directly and does not require a previously saved adapter. Model inference
requires an existing compatible adapter path and its recorded base revision.
Metrics-only evaluation must not load model weights or require CUDA.

Use greedy decoding, one beam, sampling disabled, and at most 192 new tokens.
Decode only newly generated tokens. Parse one JSON object containing exactly
the six rubric fields. Reject boolean, fractional, missing, additional,
out-of-range, malformed, or conflicting outputs. The postprocessor calculates
`total` from the validated components. Preserve support for valid dictionary
outputs from the heuristic and preserve the shared `{sample_id, output}`
prediction format.

Allow one automatic correction attempt after an invalid response, using the
same allowed input and a fixed schema-correction instruction. Save the raw
responses, parse errors, and retry counts locally. If correction still fails,
stop with an actionable sample-specific error; do not fabricate scores or
silently substitute the heuristic. Track first-attempt invalid rate and final
failure count separately from grading quality. A failed command must not
replace a previously valid prediction file with incomplete predictions.

Use the shared validator and evaluator. Report QWK on deterministic totals,
total MAE, component MAE, and component exact match, with counts and exam-type
slices. Mark pilot results clearly and explain degenerate or tiny slices.

## EX01 policy limitation

EX01's statement assigns question weights 1/4/2/3 while metadata assigns 2.5
to each question. Its policy requires P1 before P2-P4, but the labels contain
only whole-submission rubric scores. There is no authoritative mapping from
per-question status to the six rubric dimensions.

For this approved baseline, pass the statement, original metadata, and policy
to the model and train on the teacher's whole-submission rubric. Record
`policy_enforcement: prompt_only` and the unresolved weight/rubric mapping in
configuration and run metadata. Do not invent a deduction, silently select
one weighting scheme, or claim deterministic prerequisite enforcement. A
configuration requesting deterministic policy scoring fails clearly until an
authoritative mapping is available. This baseline remains incomplete for
final course submission until that policy requirement is implemented.

## Verification and acceptance

CPU tests verify input leakage guards, exact target fields, preserved policy
and function-only inputs, split persistence and overlap rejection, token
boundary masking and padding, over-budget rejection, strict JSON parsing,
retry/failure behavior, component metrics, dependency preflight errors, and
CLI compatibility. Use small fixtures and injected model/tokenizer doubles;
normal tests cannot download the Qwen checkpoint. If answer-logit selection is
implemented, compare its loss and gradients with full causal loss on a tiny
real local model.

The actual GPU acceptance run checks CUDA availability, quantized loading,
finite training loss, adapter parameter updates, peak VRAM, adapter save and
reload, automatic prediction, schema validation, and Task 1 evaluation. Run
the two-step pilot before attempting the three-epoch sample baseline. A
short-example pilot alone does not establish memory fit for the longest
training example; probe that budget before the longer run. On OOM, record the
failure and measured memory, then make an explicit recorded configuration
adjustment. Do not claim a 6 GB fit without this evidence.

Completion requires fresh CPU test results, a tested reproduction guide, and
an honest report of GPU acceptance status. If installation, model download, or
GPU execution is blocked, deliver the implemented and verified unaffected
parts with the exact remaining blocker rather than claiming training success.

## Sources

- Project contracts: `README.md`, `docs/project/PROJECT_PLAN.md`,
  `docs/project/SOURCE_SETUP.md`, `docs/project/DoAn-LLM-Challenge.docx`,
  and `sample_dataset/README.md`.
- [Qwen3.5 Transformers documentation](https://huggingface.co/docs/transformers/model_doc/qwen3_5):
  text-only loading and hybrid attention implementation.
- [PEFT quantization guide](https://huggingface.co/docs/peft/main/en/developer_guides/quantization):
  k-bit preparation, NF4, and LoRA integration.
- [uv PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/):
  selecting and recording the appropriate PyTorch backend.
- [bitsandbytes installation guide](https://huggingface.co/docs/bitsandbytes/main/en/installation):
  CUDA and Windows platform support.
