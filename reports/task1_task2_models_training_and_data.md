# Task 1 and Task 2 models, training, and data

| Result | Model used for the submission | Team training | Official public score |
| --- | --- | --- | --- |
| Task 1 rubric grading, attempt 18 | Qwen3.5-4B with the `F0-v2-kaggle-fix` adapter, followed by a compilation-score correction | Supervised QLoRA on the course sample | **0.8212** total quadratic weighted kappa (QWK), **1.5884** total mean absolute error (MAE) |
| Task 2 error taxonomy, defined-d3 v001 | `vllm-sr/d3-mini` with ten category-specific questions | Inference only with the released checkpoint | **0.2760** macro-F1, **0.3398** micro-F1 |

Both submissions used the official test release of 328 unlabeled C++ submissions per task. The course sample, public test, and authored diagnostic sets served different roles, as described below. Exact file hashes and replay commands are in the [reproducibility report](task1_task2_reproducibility_2026-11-21.md).

## Task 1 rubric grading

### Model and training scheme

The historical `F0-v2-kaggle-fix` adapter was trained on a Kaggle Tesla T4 with `Qwen/Qwen3.5-4B`. Supervised fine-tuning used a completion-only loss on the six-component rubric JSON. The trained adapter contains 16,232,448 parameters.

| Setting | Recorded training value |
| --- | --- |
| Quantization and compute | 4-bit NF4 with double quantization, BF16 compute |
| LoRA | Rank 8, alpha 16, dropout 0.05, all linear modules |
| Optimizer and schedule | AdamW, learning rate `1e-4`, linear schedule, 2 effective warmup steps |
| Batch and sequence | Batch size 1, gradient accumulation 4, maximum 4,096 tokens |
| Duration and selection | 3 epochs, 21 optimizer steps, validation after each epoch, best checkpoint reloaded |
| Seed and prompt | 42, Task 1 `v001` |

The historical training manifest records the base revision as `main`, and the exact original base commit and chat template could not be certified. The later public-test inference used revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` on a Tesla T4. The adapter weight hash matched the historical export. Inference used FP16 compute, 4-bit NF4 double quantization, and greedy decoding with thinking disabled. [Task 1 prompt](../tasks/task1_grading/prompts/v001.txt)

### Data and scoring

- **Training and validation.** The supplied course sample (`sample-v1`, split `grouped-v1`) contains 32 C++ submissions with instructor rubric scores. The training split includes 14 single-problem and 12 multi-problem submissions. Validation includes three of each. Both problem families appear in both splits. The sample has zero independent test rows and zero synthetic training rows for this adapter.
- **Public inference.** The 328 official examples supplied the problem statement and C++ source. The release had empty or absent compiler logs and test reports. First-pass inference produced the `0.8180` QWK baseline.
- **Attempt 18.** The correction used C++11 build and link rechecks performed with GCC 14.2.0 on Linux ARM64. It changed `compilable` from 1 to 0 on nine Remove Duplicates submissions and five Earliest Login submissions, then recomputed each total. The other rubric fields carried through. The correction involved no additional optimizer updates or model inference. QWK rose from 0.8180 to 0.8212, while exact compilability agreement fell from 0.7470 to 0.7348. [Replay script](../scripts/replay_task1_build_patch.py)

## Task 2 error taxonomy

### Scored model and input data

The team ran the released `vllm-sr/d3-mini` checkpoint at revision `97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e` on a Kaggle Tesla T4. The runner used a BF16 backbone, FP32 readout, batch size 1 and seed 42. It scores one yes-or-no question per taxonomy category and emits each label whose native yes probability is strictly greater than 0.5. The team trained zero additional weights for this result. [Release configuration](../configs/releases/task2_defined_decisions_v001.yaml) · [Category definitions](../tasks/task2_errors/prompts/decision_definitions_v001.json)

The scored inputs were the official problem statements, C++ sources, and source-bound local compiler evidence when available. The compiler input included 254 recorded build successes, 62 recorded build failures, and 12 unavailable observations. No instructor feedback or reference taxonomy labels were supplied to the model. The run produced 3,280 canonical probabilities across 328 submissions. Four repeated inputs produced identical probabilities and label sets. [Replay script](../scripts/replay_task2_defined_decisions.py)

### Experimental Qwen verifier

A separate adapter was trained from the pinned Qwen3.5-4B revision on a Kaggle Tesla T4. The verifier learned a one-token `Yes` or `No` completion for a reviewed category. The four categories were logic, array/string, function, and algorithm. Each authored program supplied supervision for its intended category. Unreviewed categories contributed no training target.

| Setting | Recorded training value |
| --- | --- |
| Quantization and compute | 4-bit NF4 with double quantization, BF16 compute |
| LoRA | Rank 8, alpha 16, dropout 0.05, all linear modules |
| Schedule | Learning rate `1e-5`, constant schedule, zero warmup |
| Batch and sequence | Batch size 1, gradient accumulation 2, maximum 1,024 tokens |
| Duration | 2 epochs, 32 optimizer steps |
| Seed and prompt | 42, `verifier_v001` |

The matched-pair dataset contains 64 authored C++ programs across 32 exercises. Each exercise pairs a correct implementation with a localized defect. C++11 compiler and sanitizer checks verified each pair on authored tests. The family-disjoint split used 32 programs for training, 16 for development validation, and 16 for reserved confirmation. Per category, training contained four positive and four negative examples. Validation and confirmation each contained two positives and two negatives per category. [Training configuration](../configs/task2/verifier_pairs_v002.yaml) · [Dataset generator](../examples/task2_verifier_pairs/build.py)

The verifier improved macro-F1 from 0.5605 to 0.6152 on a 32-program development panel. On a separate 40-program reviewed panel it reduced macro-F1 from 0.5650 to 0.4904. All 19 additional recovery labels on that panel were unsupported. A further 24-program array diagnostic held array F1 at 0.5455 with the recovery gate. The official 0.2760 submission used defined-d3 alone. [Array diagnostic generator](../examples/task2_array_gate/build.py)

## Reproducibility and data access

For future adapter training, use the [Kaggle QLoRA notebook](../notebooks/train_qwen_qlora_kaggle_v3.ipynb). The repository contains the authored fixture generators and aggregate results. Course submissions, private datasets, checkpoints, predictions and row-level receipts remain in authorized private storage. Task 3 is outside this report because the active 0.3704 score lacks a verified link to our archived artifact.
