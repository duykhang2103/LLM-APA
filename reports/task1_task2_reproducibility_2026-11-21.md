# Task 1 and Task 2 reproducibility package

| Task | Scored artifact | Public result | Verification |
| --- | --- | --- | --- |
| Task 1 | Attempt 18, November 19, 2026 | 0.8212 total QWK, 1.5884 total MAE | Reproduced all 328 rows byte for byte |
| Task 2 | Defined-d3, October 16, 2026, replayed November 19 | 0.2760 macro-F1, 0.3398 micro-F1 | Reproduced all 328 rows byte for byte |

The package starts from team GitHub main at `f60845d6d6f1f50bf3527bd44bf62cb42109d8ef`. The team pipeline uses its own Task 2 configuration. The scripts here are explicit entry points for the scored experiments. Task 3's November 20 public score was 0.3704. The archived 0.2171 v008 artifact is the only local Task 3 submission with verified provenance, so this package covers Tasks 1 and 2.

The training settings and data splits are documented in the [models, training, and data report](task1_task2_models_training_and_data.md).

Keep the release files, student sources, decision rows, build receipts, weights and predictions outside Git. Supply those files from authorized private storage. The C++ references and generated fixtures under `examples/` are authored.

## Task 1 compile-only replay

Attempt 18 used the original 0.8180 QWK predictions as its baseline. The Kaggle public-test inference used Qwen/Qwen3.5-4B revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`, the retained F0 adapter and seed 42. The historical adapter training records the base as `main`, with its exact revision unresolved. The correction was limited to nine Remove Duplicates rows and five Earliest Login rows. Each had an independently reproduced full-program C++11 build or link failure under GCC 14.2.0 on Linux ARM64. The replay changes `compilable` from 1 to 0 and recomputes the total. The other 314 rows carry through unchanged. The QWK gain was 0.0032. Compilability exact match decreased from 0.7470 to 0.7348.

Keep the release JSON next to its original `exams.json` and source tree.

```sh
python scripts/replay_task1_build_patch.py \
  --release /path/to/private/release/task1_grading.json \
  --baseline /path/to/private/task1-original-0.8180.json \
  --build-receipt /path/to/private/remove-duplicates/build-recheck-private.json \
  --build-receipt /path/to/private/earliest-login/build-recheck-private.json \
  --output outputs/task1-attempt18.json
```

| File | SHA-256 |
| --- | --- |
| Task 1 release JSON | `20f2cbcd9b6c33b1d735fa44a3ddfd4ef4bee38e522985a09f3728b5c7d18b75` |
| Original baseline | `25d5e3a31a3f58cdc819ac79e4c2161e267dcd388db078776bdd6cbbf9e951d6` |
| Remove Duplicates rechecks | `643673dbfe6abe0609e76134a7fbce570188ab85dc44553288ec5222b5f34e86` |
| Earliest Login rechecks | `bcf2a9dbf77210464f37de2421374ed1e7bfd3296d866004a5e15934f4eb6147` |
| Attempt 18 output | `2a5db870587425c6a653e91aa4fbec510e7a0ea507071741e3e851faa0e491f5` |

## Task 2 defined-d3 replay

The run used `vllm-sr/d3-mini` revision `97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e` on a Kaggle Tesla T4 with seed 42. The backbone used BF16 and the decision readout used FP32. The private job ran with PyTorch 2.8.0+cu126 and Transformers 5.17.0. Each taxonomy label has its own versioned question and is emitted when its probability is strictly greater than 0.5. The replay checks exact release coverage, native answers, original source hashes, question wording and stable repeat labels.

```sh
python scripts/replay_task2_defined_decisions.py \
  --release /path/to/private/release/task2_error_taxonomy.json \
  --decisions /path/to/private/decision-rows.json \
  --output outputs/task2-defined-d3.json
```

| File | SHA-256 |
| --- | --- |
| Task 2 release JSON | `e248df35a2d3e29ec3f544a115503b545bffd769a2b9a544f4a7470690cd6ff0` |
| Decision rows (328 canonical and 4 repeats) | `a1f4d279925e479c969a1ae53bb6a9a7d63d0d7e3a2f665b64629496f74559b1` |
| Execution freeze | `b86e65aff84a9d9d9ec458d20777d01c59a3362e422b4cf66c95516ae9020fa6` |
| Submitted output | `6b7b12a95c0c5d4cc67e3f5cd3bead2f189dde6adb4467835927f2a20a6014b8` |

The settings are recorded in [the release configuration](../configs/releases/task2_defined_decisions_v001.yaml). For fresh inference, attach the original private bundle to Kaggle and run `python scripts/run_defined_decisions_release.py --bundle <bundle-directory> --output <working-directory>`. The bundle contains `execution-freeze.json` and `decision-inputs.json`. The freeze pins input bytes, token IDs, the six initial stress jobs, the four repeats and the runtime. Keep the output directory for provenance if the job stops. Use the replay command on its `decision-rows.json`.

## Task 2 verifier experiments

The v002 matched-pair adapter improved macro-F1 from 0.5605 to 0.6152 on the 32-program historical panel. On the separate 40-program panel it decreased macro-F1 from 0.5650 to 0.4904. Its 19 added recovery labels matched zero reviewed positives. Logic and algorithm checks alone newly missed two defects, above the allowance of one. The separate 24-case array recovery gate held the baseline at 3/8 detected array bugs and 0.5455 array F1. The defined-d3 result remains the scored release.

```sh
python examples/task2_verifier_pairs/build.py --output outputs/task2-verifier-pairs-v002
python examples/task2_array_gate/build.py --output outputs/task2-array-gate
```

The generators use `clang++` in C++11 mode with the undefined-behavior sanitizer and execute only their authored cases. The matched-pair generator produces 32 training, 16 validation and 16 confirmation programs. Each program has supervision for its reviewed category. The array gate checks 24 programs over 81 inputs each.

Run training, training smoke and resume through a Kaggle copy of `notebooks/train_qwen_qlora_kaggle_v3.ipynb`. Attach the source snapshot and authored train and validation splits. In the versioned run config cell, use `configs/task2/verifier_pairs_v002.yaml` as the base and set the explicit training and validation paths to the attached files. Keep confirmation separate. Use the smoke settings before training and preserve the configured BF16 compute and NF4 double quantization. The original pilot completed 32 optimizer steps on a Tesla T4.

The `predict_task2_verifier.py`, `calibrate_task2_verifier.py` and `evaluate_task2_verifier.py` scripts accept explicit inputs and outputs. The prediction script takes the archived first-pass predictions and optionally the private adapter. Use the diagnostic outputs for paired evaluation against reviewed references.
