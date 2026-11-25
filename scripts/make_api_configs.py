"""Write RAG configs: API for Task 1/2, local models for Task 1/2/3.
or, with --provider local, the team's local Hugging Face runner (Qwen on GPU).

Configs start from configs/<task>/rag.yaml and only change the model, prompt,
paths and retrieval settings, so the team's model code is used unchanged.

--eval writes the leave-one-exam-out variant used to score a setup on the
labeled training data (retrieval never shows an example of the same exam,
as for the unseen public/private exams).
"""

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
TASK_DIRS = {"task1": "task1_grading", "task2": "task2_errors", "task3": "task3_feedback"}
# USD per million tokens (Groq list prices, mid-2026); unknown models get no cost estimate.
PRICES = {
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
}


def prompt_for(task, wanted):
    """The requested prompt version, or the newest older one this task has."""
    folder = ROOT / "tasks" / TASK_DIRS[task] / "prompts"
    versions = sorted(p.stem for p in folder.glob("v*.txt") if p.stem <= wanted)
    if not versions:
        raise ValueError(f"No prompt <= {wanted} for {task}")
    return versions[-1]


def make_configs(out_dir, train_dir, model, prompt, train_compile=None, train_tests=None,
                 eval_mode=False, base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY",
                 max_cost=20.0, cache_path=None, provider="api", dtype=None, extra_body=None,
                 method="rag", evidence=True, revision=None, price=None, attn=None,
                 tasks=None, task3_prompt="v005", task3_labels=False, diagnosis_prompt="v005",
                 timeout=120):
    """method rag|zero_shot; evidence=False hides compile/test results from the prompt
    (statement + code only, like the fine-tuned Task 1 model's public inference).
    tasks: which configs to write (default: task1/task2 for api, all three for local).
    Task 3 follows the team's T4 notebook (prompt v005, two-stage diagnosis then feedback,
    1024 new tokens); task3_labels=True instead takes labels from a Task 2 prediction file."""
    out_dir = Path(out_dir)
    paths = {}
    for task in TASK_DIRS:
        if tasks is None and task == "task3" and provider == "api":
            continue
        if tasks is not None and task not in tasks:
            continue
        cfg = yaml.safe_load((ROOT / "configs" / task / "rag.yaml").read_text(encoding="utf-8"))
        cfg["method"] = method
        version = prompt_for(task, task3_prompt if task == "task3" else prompt)
        tag = f"{provider}-{method}-{version}{'' if evidence else '-noevid'}{'-loeo' if eval_mode else ''}"
        cfg["prompt"]["version"] = version
        cfg["experiment"].update(
            id=f"{task}-{tag}", name=f"{task}_{tag}",
            notes=f"{model}; prompt {version}; "
            + ("code-keyed BM25 k=1" if method == "rag" else "zero-shot")
            + ("; compile+test evidence" if evidence else "; no compile/test evidence")
            + ("; leave-one-exam-out evaluation" if eval_mode else ""),
        )
        cache = str(Path(cache_path or ROOT / "outputs" / f"{provider}_cache.jsonl").resolve())
        if provider == "local":
            # Keep the team's local model block (quantization, adapter).
            cfg["model"].update(name_or_path=model, cache_path=cache)
            if revision:
                cfg["model"]["revision"] = revision
            if dtype:
                cfg["model"]["dtype"] = dtype
            if attn:
                cfg["model"]["attn_implementation"] = attn
        else:
            cfg["model"] = {
                "provider": "api",
                "name_or_path": model,
                "revision": revision,  # recorded only; the API serves its own build
                "base_url": base_url,
                "api_key_env": key_env,
                "allow_private_data": True,  # lecturer allowed sending student code to an API
                "timeout_seconds": timeout,
                "cost_per_million_tokens": (dict(zip(("prompt", "completion"), price)) if price
                                            else PRICES.get(model, {})),
                "max_cost_usd": max_cost,
                "cache_path": cache,
            }
            if extra_body:
                # Provider request fields, e.g. {"reasoning": {"enabled": false}} so a
                # reasoning model answers within max_new_tokens instead of thinking.
                cfg["model"]["extra_body"] = extra_body
        train = str(Path(train_dir).resolve())
        cfg["data"].update(input_path=train, train_path=train, validation_path=None, test_path=None,
                           prepared_dir=str((out_dir / task / "prepared").resolve()))
        cfg["saving"]["output_dir"] = str((out_dir / task).resolve())
        cfg.setdefault(task, {})["use_grading_policy"] = False
        if not evidence:
            cfg[task].update(use_compile_log=False, use_test_report=False)
            train_compile = train_tests = None
        # One unparseable sample must not stop a 328-sample run; fallbacks are logged.
        cfg[task]["fallback_on_failure"] = True
        if task == "task2":
            cfg["task2"].update(salvage_labels=True, max_labels=3)
        if task == "task3":
            cfg["task3"].update(max_compliance_retries=2, two_stage=not task3_labels,
                                diagnosis_prompt=diagnosis_prompt)
            cfg["task2"] = {"use_grading_policy": False}  # the diagnosis stage
            cfg["generation"]["max_new_tokens"] = 1024
        cfg["retrieval"].update(k=1, key="code", index="bm25", same_exam_first=not eval_mode,
                                exclude_same_exam=eval_mode, compact_examples=True,
                                exclude_duplicates=True, near_duplicate_threshold=0.95)
        cfg["evidence"] = {
            "train_compile": str(Path(train_compile).resolve()) if train_compile else None,
            "train_tests": str(Path(train_tests).resolve()) if train_tests else None,
        }
        path = out_dir / "configs" / f"config_{task}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
        paths[task] = path
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="Run folder: configs/ and per-task logs go here")
    parser.add_argument("--train", required=True, help="Labeled training dataset folder")
    parser.add_argument("--provider", choices=["api", "local"], default="api")
    parser.add_argument("--model", default="llama-3.1-8b-instant")
    parser.add_argument("--dtype", help="local only, e.g. float16 on a T4")
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1")
    parser.add_argument("--api-key-env", default="GROQ_API_KEY")
    parser.add_argument("--prompt", default="v004")
    parser.add_argument("--train-compile")
    parser.add_argument("--train-tests")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    for task, path in make_configs(args.out, args.train, args.model, args.prompt,
                                   args.train_compile, args.train_tests, args.eval,
                                   provider=args.provider, dtype=args.dtype,
                                   base_url=args.base_url, key_env=args.api_key_env).items():
        print(task, "->", path)


if __name__ == "__main__":
    raise SystemExit(main())
