"""Build three standalone Task 3 notebooks without student data."""
import json
import pprint
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "P1": "Task3_P1_ZeroShot_LevelAware",
    "P2": "Task3_P2_RAG",
    "P3": "Task3_P3_PlanThenRender",
}


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "source": text.splitlines(True),
            "execution_count": None, "outputs": []}


def build() -> None:
    runtime = (ROOT / "src/llm_grading/task3_controlled.py").read_text(encoding="utf-8")
    defaults = yaml.safe_load((ROOT / "configs/task3/notebooks/controlled_feedback.yaml").read_text(encoding="utf-8"))
    p2_context_limit = defaults.pop("p2_context_limit")
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    defaults["source_git_revision"] = revision.stdout.strip() if revision.returncode == 0 else None
    for method, name in NAMES.items():
        config = {"method": method, **defaults}
        if method == "P2":
            config["context_limit"] = p2_context_limit
            config["mode"] = "dev"
            config["output_root"] = "/kaggle/working/p2_context12288_fresh"
            config["shared_dir"] += "_p2_context12288"
        if method == "P2" and config["train_path"] is None:
            config["train_path"] = "auto_sample_dataset"
        cells = [
            markdown(f"# Task 3 {method}: Controlled Feedback Generation\n\n"
                     "P1: zero-shot. P2: two TF-IDF training demonstrations. P3: structured diagnosis then rendering, without RAG. "
                     "This notebook executes only its configured method. Target labels are never inputs. "
                     "Standalone embedded runtime; no source archive required. Qwen/Qwen3-4B, FP16, no fine-tuning, no external LLM API. "
                     "Keep student datasets, runs, checkpoints, and logs private."),
            markdown("## 1. Environment\n\nSelect GPU T4 x2; one GPU also works. Run each notebook separately. "
                     "One full model replica per GPU in independent subprocesses; no sharding. Internet downloads public packages/weights only. "
                     "For offline runs attach the pinned HF cache and wheels, install wheels first, set INSTALL_DEPENDENCIES=False, "
                     "HF_HOME to the cache, and HF_HUB_OFFLINE=1. Start from a fresh kernel."),
            code('''import os
import subprocess
import sys

INSTALL_DEPENDENCIES = True
if INSTALL_DEPENDENCIES:
    # Preserve Kaggle's CUDA PyTorch; same dependency versions for every method.
    subprocess.run([
        sys.executable, "-m", "pip", "install", "--quiet",
        "transformers==4.51.3", "accelerate==1.6.0", "huggingface-hub==0.30.2",
        "tokenizers==0.21.1", "safetensors==0.5.3", "scikit-learn==1.6.1",
    ], check=True)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
'''),
            markdown("## 2. Parameters\n\nAttach public_test.zip or its extracted directory. Set dev_path explicitly if multiple releases are attached. "
                     "levels_path optionally points to the supplied requested-level sidecar; it overrides embedded levels and must cover all 328 IDs. "
                     "First run mode='smoke' (10 samples), then mode='dev' (328). Defaults: seed 42, batch size 1 per GPU, 512 new tokens, greedy decoding. "
                     "context_limit includes prompt plus output; overflow fails without truncation. gpu_ids=['0'] forces single GPU.\n\n"
                     "The user has authorized sample_dataset for P2 demonstrations. Attach its 32 annotated examples "
                     "(levels 1/2/3: 4/18/10; no Level 4). P2 defaults to train_path='auto_sample_dataset', "
                     "which selects the unique 32-row annotated dataset under input_root; set an explicit directory/ZIP if ambiguous. "
                     "confirm_training_partition=True records this existing authorization. P1/P3 need no training data. "
                     "P2 uses all 32 samples as its initial corpus without a validation split (smoke/dev modes). "
                     "Never use public outputs for training, retrieval demonstrations, or prompt optimization."),
            code("CONFIG = " + pprint.pformat(config, sort_dicts=False, width=100) + "\n"),
            markdown("## 3. Shared runtime and prompts\n\nRuntime controlled_runtime_v005, prompt controlled_feedback_v003 includes all ten exact taxonomy labels and definitions. "
                     "Level 1: indicate a problem/praise without location/cause. Level 2: location/cause without fixes/code. "
                     "Level 3: correction suggestions without a complete solution. Level 4: corrected code/model solution permitted. "
                     "Teacher exceptions for MSVC, missing main in function-only submissions, and input prompts are explicit. "
                     "Uses the [official Qwen3 chat template](https://huggingface.co/Qwen/Qwen3-4B) with enable_thinking=False. "
                     "Model revision is pinned and the full resolved SHA is recorded. Greedy decoding follows the requested experiment settings."),
            code("import importlib.util\nfrom pathlib import Path\n\n"
                 "RUNTIME_SOURCE = " + repr(runtime) + "\n"
                 "runtime_path = Path(CONFIG['output_root']) / 'task3_controlled_runtime.py'\n"
                 "runtime_path.parent.mkdir(parents=True, exist_ok=True)\n"
                 "runtime_path.write_text(RUNTIME_SOURCE, encoding='utf-8')\n"
                 "spec = importlib.util.spec_from_file_location('task3_controlled_runtime', runtime_path)\n"
                 "runtime = importlib.util.module_from_spec(spec)\n"
                 "sys.modules[spec.name] = runtime\n"
                 "spec.loader.exec_module(runtime)\n"),
            markdown("## 4. Prepare before loading weights\n\nResolve code_file relative to the data root. Preserve full statements, multi_problem subproblems, "
                     "grading policy, exam type and supplied compile/test evidence. No execution of student code. Discard target labels/reference feedback. "
                     "Validate counts, IDs, levels, taxonomy and paths.\n\n"
                     "P2 normalizes old-schema references in memory: input has the joined problem/source/level, and output has taxonomy_error plus feedback. "
                     "The original dataset is unchanged. Fit character TF-IDF (3-5 grams) only on eligible corpus code. "
                     "Retrieve up to 2 examples at the requested level, prioritizing exam ID then cosine code similarity with deterministic ID tie-breaking. "
                     "Target ground-truth labels never participate. Exclude public IDs/code copies, duplicates, and feedback with detected level violations "
                     "(including inline fixes/code). Use one example if only one qualifies; none uses exactly P1 zero-shot prompting, especially for Level 4. "
                     "All 32 samples form the initial corpus with no split. Private normalized_corpus and retrieval_audit files record normalization, "
                     "quality exclusions, label/level coverage and fallback IDs."),
            code("prepared = runtime.prepare(CONFIG)\n"),
            markdown("## 5. Inference and resume\n\nTwo independent workers use left-padded FP16 batches. Progress prints every 10 seconds. "
                     "Atomic per-sample checkpoints allow resume with one or two GPUs. P3 validates/caches structured evidence before rendering and locks predicted labels. "
                     "Base inference calls: one for P1/P2, two for P3. Malformed JSON is mechanically repaired without inventing content. "
                     "Malformed JSON or invalid required/internal fields get at most json_retries additional local calls (default 1). "
                     "Lexical level checks are non-fatal warnings, never retries. Exhausted sample failures are saved privately while other samples continue. "
                     "After the first pass, failure_retry_passes (default 1) retries unresolved samples before export. "
                     "No fabricated fallback answers or silent skips. GPU assignment/memory and verified effective greedy config are saved in worker startup logs.\n\n"
                     "To resume restore private run and shared-comparison directories to /kaggle/working with identical config, runtime, data and packages. "
                     "Fingerprints prevent mixing experiments. Inspect worker logs for OOM/context errors; explicitly adjust batch/context/token settings in a fresh "
                     "comparison/output directory. No important input is silently truncated."),
            code("submission_path = runtime.run_inference(prepared, runtime_path)\n"),
            markdown("## 6. Outputs and limitations\n\nOnly complete dev runs export "
                     f"/kaggle/working/predictions_{method}.json, exactly 328 records: sample_id and output containing taxonomy_error and feedback, "
                     "matching the official sample. Normalize only confirmed taxonomy aliases plus Unicode/case/whitespace, deduplicate in order, then strictly validate official labels. Unknown labels still fail. Normalization events are logged separately. Strictly validate missing/duplicate IDs, non-empty feedback and internal content. "
                     "Lexical level flags are recorded in private warning logs without blocking valid predictions. Unresolved failures block final export. "
                     "Smoke/validation write smoke_P*.json/validation_P*.json, which are not leaderboard submissions. Diagnostics/metadata never enter final output. "
                     "runtime_stats.json separately records wall/GPU time, tokens, sample-generation calls/retries, package versions and peak GPU memory. "
                     "Completed cached work is included; interrupted uncheckpointed calls are excluded. Lexical checks cannot prove semantic consistency, "
                     "level compliance or diagnostic accuracy. Evaluate independently on authorized validation data. No experimental scores are fabricated."),
            code("import json\n\n"
                 "stats = runtime.read_json(Path(prepared['run_dir']) / 'runtime_stats.json')\n"
                 "print(json.dumps(stats, ensure_ascii=False, indent=2))\n"
                 "print('Output:', submission_path)\n"
                 "if CONFIG['mode'] != 'dev':\n"
                 "    print('SMOKE/VALIDATION ONLY - not a leaderboard submission.')\n"),
        ]
        if method == "P2":
            cells[3]["source"] += markdown(
                "\n\nP2 context amendment: 12288 prompt+output tokens (512 output unchanged). "
                "Pinned Qwen3-4B has native 40960 positions; SDPA has no configured 8192 cap. "
                "No RoPE scaling or tokenizer/prompt changes. P1/P3 retain 8192 and use their original comparison contract. "
                "Report this context amendment with comparisons; do not pool amended and original contracts silently."
            )["source"]
            cells[1] = markdown(
                "## 1. Environment\n\nStart a new Kaggle GPU T4 x2 session, attach public_test and "
                "the authorized sample_dataset, then Run All. This notebook runs all 328 dev samples. "
                "One FP16 replica per GPU; batch size 1 per GPU. No migration or old checkpoint is required."
            )
            cells[3] = markdown(
                "## 2. Parameters\n\nReady for a full dev run: 328 samples, seed 42, "
                "context_limit=12288 (input plus output), max_new_tokens=512, batch size 1 per GPU. "
                "The output directory is /kaggle/working/p2_context12288_fresh, separate from old runs. "
                "Dev and the authorized 32-row sample_dataset are discovered under /kaggle/input. "
                "Set dev_path/train_path explicitly only if multiple candidate datasets are attached. "
                "Target labels/reference feedback are never model inputs. Overflow fails without truncation. "
                "Pinned Qwen3-4B supports 40960 native positions; no RoPE or tokenizer changes. "
                "P1/P3 retain their original 8192 contracts; report P2's context amendment in comparisons."
            )
            cells[9] = markdown(
                "## 5. Inference\n\nRun this cell after preparation. Workers process all 328 samples "
                "in the new output directory, with bounded retries and per-sample checkpoints. "
                "P2 uses one feedback stage with up to two retrieved demonstrations, without a diagnosis stage. "
                "Prompts, retrieval, model revision and greedy decoding are unchanged. CUDA OOM for a single "
                "sample still fails; valid checkpoints are preserved. Submission export requires all 328 "
                "valid results. No dummy output or skipped failure is allowed."
            )
            cells[11] = markdown(
                "## 6. Outputs and limitations\n\nThe validated submission is "
                "/kaggle/working/p2_context12288_fresh/predictions_P2.json. "
                "Private logs, per-sample checkpoints and runtime_stats.json are in task3_P2_dev under "
                "the same output directory. Strict validation blocks incomplete, flagged or invalid outputs. "
                "Batch size 1 limits memory pressure but actual T4 VRAM must be verified during execution. "
                "Lexical checks do not prove semantic correctness; no experimental scores are claimed."
            )
        notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        }, "cells": [{"id": f"{method.lower()}-{i:02d}", **cell} for i, cell in enumerate(cells)]}
        output = ROOT / "notebooks" / f"{name}.ipynb"
        output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        print(output.relative_to(ROOT))


if __name__ == "__main__":
    build()
