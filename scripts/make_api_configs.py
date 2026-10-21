"""Write RAG configs for task1/2/3: an OpenAI-compatible API (default Groq)
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
                 max_cost=20.0, cache_path=None, provider="api", dtype=None):
    out_dir = Path(out_dir)
    paths = {}
    for task in TASK_DIRS:
        cfg = yaml.safe_load((ROOT / "configs" / task / "rag.yaml").read_text(encoding="utf-8"))
        version = prompt_for(task, prompt)
        tag = f"{provider}-{version}{'-loeo' if eval_mode else ''}"
        cfg["prompt"]["version"] = version
        cfg["experiment"].update(
            id=f"{task}-{tag}", name=f"{task}_{tag}",
            notes=f"{model}; prompt {version}; code-keyed BM25 k=1; compile+test evidence"
            + ("; leave-one-exam-out evaluation" if eval_mode else ""),
        )
        cache = str(Path(cache_path or ROOT / "outputs" / f"{provider}_cache.jsonl").resolve())
        if provider == "local":
            # Keep the team's local model block (quantization, revision, adapter).
            cfg["model"].update(name_or_path=model, cache_path=cache)
            if dtype:
                cfg["model"]["dtype"] = dtype
        else:
            cfg["model"] = {
                "provider": "api",
                "name_or_path": model,
                "revision": None,
                "base_url": base_url,
                "api_key_env": key_env,
                "allow_private_data": True,  # lecturer allowed sending student code to an API
                "timeout_seconds": 120,
                "cost_per_million_tokens": PRICES.get(model, {}),
                "max_cost_usd": max_cost,
                "cache_path": cache,
            }
        train = str(Path(train_dir).resolve())
        cfg["data"].update(input_path=train, train_path=train, validation_path=None, test_path=None,
                           prepared_dir=str((out_dir / task / "prepared").resolve()))
        cfg["saving"]["output_dir"] = str((out_dir / task).resolve())
        cfg.setdefault(task, {})["use_grading_policy"] = False
        # One unparseable sample must not stop a 328-sample run; fallbacks are logged.
        cfg[task]["fallback_on_failure"] = True
        if task == "task2":
            cfg["task2"].update(salvage_labels=True, max_labels=3)
        if task == "task3":
            cfg["task3"].update(max_compliance_retries=2)
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
