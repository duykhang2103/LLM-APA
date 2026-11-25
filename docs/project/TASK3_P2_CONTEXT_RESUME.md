# P2 context overflow recovery

The 8192-token **admission limit** reserves 512 output tokens, allowing at most
7680 input tokens. All eight reported counts exceed 7680. The largest reported
input is 9246, requiring 9758 total tokens. `export_results()` correctly refuses
320/328 results; its validation is unchanged.

## Verified model and backend

Read on 2026-11-25 at the pinned revision:

- Model: Qwen/Qwen3-4B, full SHA `1cfa9a7208912126459214e8b04321603b3df60c`.
- [Pinned config](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/config.json):
  `max_position_embeddings=40960`, `rope_scaling=null`,
  `use_sliding_window=false`, `sliding_window=null`, `rope_theta=1000000`.
- [Pinned tokenizer config](https://huggingface.co/Qwen/Qwen3-4B/blob/1cfa9a7208912126459214e8b04321603b3df60c/tokenizer_config.json):
  `model_max_length=131072`. This does **not** override the model's 40960 limit.
- Runtime uses Transformers 4.51.3 `AutoModelForCausalLM`, SDPA, FP16,
  dynamic generation cache, `max_new_tokens=512`, greedy decoding.
  No vLLM engine, static 8192 cache, or other configured 8192 backend cap exists.
  The worker already checks the loaded model's native positions. Tokenization
  explicitly sets `truncation=False`; no tokenizer setting needs changing.
- [Transformers Qwen3 API at 4.51.3](https://huggingface.co/docs/transformers/v4.51.3/en/model_doc/qwen3).

P2 now uses 12288 total tokens. This covers 9758 and leaves 2530 tokens beyond
that reported maximum for existing retry instructions. Each retry still passes
the same validator; no input is trimmed. This does not guarantee arbitrary future
inputs fit. Model weights/revision, chat templates, source/problem, RAG retrieval
and top-k, schema/taxonomy, retry policy and decoding stay unchanged. P2 is one
feedback call with demonstrations; diagnosis/cache reuse applies to P3 only.

## Safe reuse of 320 results

Changing context modifies the contract and fingerprint, even though context only
gates whether generation may begin. The fingerprint also includes full runtime
source. We deliberately keep `src/llm_grading/task3_controlled.py` unchanged.
`scripts/resume_task3_p2_context.py` verifies:

1. The old runtime SHA and reconstructed old fingerprint, including target data.
2. P2/dev, 328 unique targets, original context=8192, output=512; retain the
   original positive batch size, including batch=2. Worker and contract must agree.
3. Actual pinned snapshot/full SHA, native full attention, and GPU availability.
4. Original worker config, package versions, PyTorch/CUDA versions, GPU type and
   effective greedy/FP16/no-thinking configuration.
5. All completed raw generations parse to the exact saved outputs, have EOS and
   valid token accounting, match retrieved IDs, and their original full prompts
   fit 8192. Every unresolved item must have a feedback context-overflow failure.
6. Re-prepared target objects, ordered examples, corpus/retrieval audit and full
   contract match exactly, with **only context_limit** changed in the contract.

The destination and comparison directories must be new. The entire original run
is copied to `original_checkpoint_backup/` and verified with SHA-256 per file.
Only valid sample caches are copied byte-for-byte into the new active run.
Failure logs remain in the backup. Provenance, old/new fingerprints, file hashes,
prompt hashes/token counts, reused/pending IDs, model config and memory estimate
are saved in `context_migration.private.json`. The original run is never written.
Normal fingerprint validation remains active afterward. A rejected migration
must be investigated; do not edit fingerprints or copy caches manually.

Thus reuse is safe **if these checks pass on the actual Kaggle checkpoint**.
It has been demonstrated with synthetic 320+8 tests locally, not with private
Kaggle artifacts. Fresh generation across sessions is not claimed bitwise
identical; no existing generation is regenerated or changed.

## Exact Kaggle cells

Stop the previous workers before starting. Keep/restore the complete private
`/kaggle/working/task3_P2_dev/` directory, original data mounts, environment,
and model cache. Preserve the original shared comparison directory as well.
Open the updated **Task3_P2_RAG.ipynb**. Run sections 1–3, retaining the original
package versions. In section 4, change the first assignment to:

```python
RESUME_P2_CONTEXT_FROM = '/kaggle/working/task3_P2_dev'
```

Run that section's full embedded helper cell. It derives configuration from the
old manifest, preserving actual dev/data/sidecar/retry settings; notebook smoke
defaults are not used for recovery. It must print `Migrated 320/328 ... pending 8`.
Then run section 5 unchanged:

```python
submission_path = runtime.run_inference(prepared, runtime_path)
```

Destination (must not exist before the first migration):

```text
/kaggle/working/p2_context12288/task3_P2_dev/
/kaggle/working/p2_context12288/original_checkpoint_backup/
/kaggle/working/p2_context12288/predictions_P2.json
```

Only the eight missing IDs are scheduled. Already valid samples are skipped on
both workers. Existing bounded retries can issue additional calls for one of
those eight if its generated output fails validation. No old overflow prompt
is inserted into the new initial prompt; overflow produced no model output.
The original runtime's retry behavior within each resumed sample is unchanged.

For a later interruption, run notebook sections 1–3, then **replace section 4's
cell** with the following, rather than migrating again:

```python
migrated_run = Path('/kaggle/working/p2_context12288/task3_P2_dev')
prepared = runtime.prepare(runtime.read_json(migrated_run / 'manifest.json')['config'])
submission_path = runtime.run_inference(prepared, runtime_path)
```

Optional audit after the first migration, before inference:

```python
audit = runtime.read_json(Path(prepared['run_dir']) / 'context_migration.private.json')
assert len(audit['reused_ids']) == 320 and len(audit['pending_ids']) == 8
print(audit['pending_ids'])
print(audit['snapshot_preflight'])
```

Do not rerun migration against its destination. If checks fail after creation of
a new destination, retain it for review and use another fresh destination for a
corrected attempt. Strict export requires 328 unique, schema-valid, nonempty,
unflagged results; no dummy, fallback or skipped item is introduced.

## GPU memory and experimental comparisons

Each T4 has one independent full FP16 model replica. Fresh notebooks default to
batch size 1; migration preserves the original batch size (including 2).
Existing CUDA OOM splitting can reduce a failed batch to single samples without
changing prompts or decoding. At 12288 tokens, per-sample theoretical FP16 KV cache is
`2 * 36 * 8 * 128 * 12288 * 2 = 1811939328` bytes (1.6875 GiB).
At 9758 total tokens it is about 1.34 GiB. This excludes approximately 8 GB of
weights, activations, logits, allocator and attention workspace. T4 SDPA may
select math attention with quadratic temporary memory; **local inspection cannot
guarantee no CUDA OOM**. The exact backend/precision is retained. Existing OOM
batch splitting remains available for larger batches but cannot split a single
sample. OOM still blocks export and preserves completed caches. Review
`worker_*_startup.json`, `worker_*_stats.json`, `worker_*.log` after running; remove
unrelated GPU processes before retrying. No quantization, backend or offload
change is silently applied to rescue results.

Controlled P1/P3 retain 8192. Legacy CLI P0 (`zero_shot.yaml`) uses
`max_input_tokens=4096`; its RAG P1 (`rag.yaml`) and Kaggle/v3 use
`max_input_tokens=8192`. Those are **input-only** budgets in the separate shared
generation module, not this runtime's prompt-plus-output `context_limit`.
Legacy Kaggle/v3 and other configs are unchanged. The new P2
comparison directory explicitly separates the amended contract; do not claim
the original identical-context comparison contract still holds. Since 8192 only
rejects requests and never changes accepted model computations, old accepted
results remain compatible under the checks above. For a formally common context
protocol, approve and record a new 12288 contract for **all** methods, preflight
all their prompts and P3 diagnosis/render stages, then assess compatible caches
under a separate audited migration. This patch does not authorize or perform
that cross-method protocol change.

Local tests use synthetic data and CPU inference doubles. Actual 320 private
outputs, exact eight prompt counts, available Kaggle VRAM, SDPA kernel selection,
and final 328-output quality/completion require the Kaggle run.

## Local validation (2026-11-25)

```powershell
.venv/Scripts/python.exe -m pytest tests -q --basetemp tmp/p2-context-full-tests-03 -p no:cacheprovider
# 256 passed, 1 skipped, 15 subtests passed; 21 existing PEFT Conv1D warnings.
.venv/Scripts/python.exe -m ruff check scripts/resume_task3_p2_context.py scripts/build_task3_controlled_notebooks.py tests/test_task3_context_resume.py
# All checks passed.
```

Regression coverage includes all eight reported input lengths, native model
limit versus tokenizer metadata, context-only compatibility and rejection of
corrupt provenance/results, verified backup hashes, byte-identical reuse of 320
caches, exactly eight resumed feedback calls, idempotent subsequent resume,
and incomplete-versus-complete strict export. Notebook syntax and embedded
runtime consistency are also checked. The embedded migration helper was compared
to its source after regeneration. Root-level pytest discovery was interrupted
while traversing accumulated temporary artifacts; the complete `tests/` suite
above was run to completion.
