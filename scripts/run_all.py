"""One command for the whole prompting/RAG pipeline on one dataset.

Steps (each skipped when its output already exists, unless --force):
  1. compile evidence for the dataset and the training data
  2. LLM test suites per exam (once per exam, shared in outputs/testsuites/)
  3. run the suites on every submission
  4. API configs, then Task 1 -> Task 2 -> Task 3 predictions
  5. challenge-format submission files (+ a rule-based Task 1 file)

Public/private test:  python scripts/run_all.py --data <dataset folder>
Score a setup first:  python scripts/run_all.py --data <train folder> --eval
(--eval: leave-one-exam-out retrieval, then metrics against teacher labels.)
--no-llm stops after the rule-based Task 1 file (no API calls for grading).

Prompting variants (compare with the fine-tuned model on the same Qwen3.5-4B):
  P0 zero-shot, statement + code only:  --method zero_shot --no-evidence
  P1 RAG k=1, statement + code only:    --no-evidence
  P2 RAG k=1 + compile/test evidence:   (default)
"""

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from make_api_configs import make_configs  # noqa: E402

# Same Qwen3.5-4B build as the team's fine-tuned Task 1 model, so prompting and
# fine-tuning differ only in the method (reports/task1_task2_models_training_and_data.md).
PINNED_REVISIONS = {"Qwen/Qwen3.5-4B": "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}
DEFAULT_TRAIN = ROOT.parent / "Data" / "public-test-cham-diem-phan-hoi-files" / "sample_dataset" / "sample_dataset"
SCRIPTS = ROOT / "scripts"


def tag_of(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", Path(path).resolve().name).strip("-") or "data"


def check_models(models, base_url="https://api.groq.com/openai/v1", key_env="GROQ_API_KEY"):
    """Fail before any work when the key is missing or a model is unavailable."""
    key = os.environ.get(key_env)
    if not key:
        raise SystemExit(f"Set {key_env} first (needed for {models} at {base_url})")
    request = urllib.request.Request(base_url + "/models", headers={
        "Authorization": f"Bearer {key}", "User-Agent": "Mozilla/5.0 (compatible; llm-apa/1.0)"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            available = sorted(m["id"] for m in json.load(response)["data"])
    except urllib.error.HTTPError as error:
        raise SystemExit(f"API key check failed: {error.code} {error.read().decode(errors='replace')[:300]}")
    # Hugging Face router ids may name a provider: "Qwen/Qwen3.5-4B:featherless-ai".
    missing = [m for m in models if m not in available and m.split(":")[0] not in available]
    if missing:
        raise SystemExit(f"Model(s) not available for this key: {missing}\nAvailable: {available}\n"
                         "Choose with --testgen-model (test suites) and --model (grading).")
    print("Models OK:", models)


def stale_tests(dataset, tests_path, suites):
    """True when a suite was added or rewritten after tests_path was made."""
    if not tests_path.exists():
        return True
    used = set(json.loads(tests_path.read_text(encoding="utf-8"))["suites"])
    exams = {e["exam_id"] for e in json.loads((dataset / "exams.json").read_text(encoding="utf-8"))["exams"]}
    for exam in exams:
        suite = suites / f"{exam}.json"
        if suite.exists() and (exam not in used or suite.stat().st_mtime > tests_path.stat().st_mtime):
            print(f"Test results are older than the {exam} suite; rerunning the tests")
            return True
    return False


class Runner:
    def __init__(self, dry_run):
        self.dry_run = dry_run

    def __call__(self, script, *args):
        command = [sys.executable, str(SCRIPTS / script), *map(str, args)]
        print("\n$ " + " ".join(command), flush=True)
        if not self.dry_run:
            subprocess.run(command, check=True, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Dataset folder to predict (exams.json, submissions/)")
    parser.add_argument("--train", default=str(DEFAULT_TRAIN), help="Labeled training dataset (RAG examples)")
    parser.add_argument("--prompt", default="v004")
    parser.add_argument("--provider", choices=["api", "local"], default="api",
                        help="api = OpenAI-compatible endpoint; local = Hugging Face model on this GPU")
    parser.add_argument("--model", help="Grading model (<=10B open weights); default Qwen/Qwen3.5-4B for local, "
                        "required for api")
    parser.add_argument("--base-url", default="https://api.groq.com/openai/v1",
                        help="Grading API (OpenAI-compatible), e.g. https://openrouter.ai/api/v1")
    parser.add_argument("--api-key-env", default="GROQ_API_KEY", help="Env variable holding the grading API key")
    parser.add_argument("--no-reasoning", dest="extra_body", action="store_const",
                        const={"reasoning": {"enabled": False}},
                        help="OpenRouter: turn off hidden reasoning so a reasoning model (e.g. DeepSeek V4) "
                             "answers within the token budget")
    parser.add_argument("--no-thinking", action="store_true",
                        help="Qwen3/3.5 behind vLLM-style APIs (e.g. Hugging Face router): send "
                             "chat_template_kwargs.enable_thinking=false (local runs already do this)")
    parser.add_argument("--extra-body", dest="extra_body_json", type=json.loads, default=None, metavar="JSON",
                        help='Extra API request fields as JSON, e.g. {"reasoning_effort": "none"}')
    parser.add_argument("--reasoning-effort", metavar="EFFORT",
                        help="OpenAI-style reasoning_effort field, e.g. none (Requesty); no JSON quoting needed")
    parser.add_argument("--revision", help="Model revision (local: loaded; api: recorded). "
                        "Default for Qwen/Qwen3.5-4B: the fine-tuned model's revision")
    parser.add_argument("--method", choices=["rag", "zero_shot"], default="rag",
                        help="rag = one BM25 training example in the prompt (P1/P2); zero_shot = none (P0)")
    parser.add_argument("--no-evidence", action="store_true",
                        help="Prompt with statement + code only, no compile/test results (P0/P1); "
                             "implies --no-tests and skips the evidence-based Task 2 calibration")
    parser.add_argument("--price", nargs=2, type=float, metavar=("IN", "OUT"),
                        help="API price in USD per million prompt/completion tokens, e.g. 0.0525 0.0735; "
                             "makes the cost estimate and --max-cost work for this model")
    parser.add_argument("--max-cost", type=float, default=20.0,
                        help="Stop a Task 1 or Task 2 step once its estimated API cost reaches this many USD")
    parser.add_argument("--attn", help="local only: attention implementation, e.g. sdpa (default from rag.yaml)")
    parser.add_argument("--cache-path", help="Response cache file (default outputs/<provider>_cache.jsonl); "
                        "give parallel runs separate files")
    parser.add_argument("--dtype", help="local only: bfloat16 (default) or float16 for T4/P100")
    parser.add_argument("--testgen-config", default=str(ROOT / "configs" / "testgen" / "groq.yaml"))
    parser.add_argument("--testgen-model", help="Model that writes test suites (default: from --testgen-config)")
    parser.add_argument("--suites", default=str(ROOT / "outputs" / "testsuites"), help="Shared test-suite folder")
    parser.add_argument("--tasks", nargs="+", choices=["task1", "task2", "task3"])
    parser.add_argument("--task3-prompt", default="v005", help="Task 3 prompt version (team notebook: v005)")
    parser.add_argument("--task3-diagnosis-prompt", default="v004",
                        help="Task 2 prompt for Task 3's diagnosis stage (team notebook: v005; v004 scored higher on train)")
    parser.add_argument("--task3-labels", metavar="task2|PATH",
                        help="Task 3 writes feedback for given labels instead of diagnosing them itself: "
                             "'task2' = this run's Task 2 output, or a Task 2 prediction file. "
                             "Default: two-stage (the model diagnoses labels, then writes feedback)")
    parser.add_argument("--name", help="Run folder name under outputs/runs/ (default <data>-<prompt>)")
    parser.add_argument("--eval", action="store_true", help="Leave-one-exam-out run + metrics (labeled data)")
    parser.add_argument("--no-llm", action="store_true", help="Only evidence, tests and rule-based Task 1")
    parser.add_argument("--no-tests", action="store_true", help="Skip test generation/running")
    parser.add_argument("--force", action="store_true", help="Recompute compile evidence and test results")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--max-samples", type=int, help="Smoke run on the first N samples")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands only")
    args = parser.parse_args()
    if args.no_evidence:
        args.no_tests = True
    if args.extra_body_json:
        args.extra_body = {**(args.extra_body or {}), **args.extra_body_json}
    if args.reasoning_effort:
        args.extra_body = {**(args.extra_body or {}), "reasoning_effort": args.reasoning_effort}
    if args.no_thinking:
        args.extra_body = {**(args.extra_body or {}), "chat_template_kwargs": {"enable_thinking": False}}
    if args.tasks is None:
        args.tasks = ["task1", "task2"] if args.provider == "api" else ["task1", "task2", "task3"]
    if args.task3_labels and "task3" not in args.tasks:
        parser.error("--task3-labels needs --tasks ... task3")

    os.environ.setdefault("ZIG_GLOBAL_CACHE_DIR", str(ROOT / "outputs" / "tmp" / "zigcache"))
    os.environ.setdefault("ZIG_LOCAL_CACHE_DIR", os.environ["ZIG_GLOBAL_CACHE_DIR"])
    run = Runner(args.dry_run)
    data, train = Path(args.data).resolve(), Path(args.train).resolve()
    if not (data / "exams.json").exists() or not (train / "exams.json").exists():
        raise SystemExit("--data and --train must be dataset folders containing exams.json")
    suites = Path(args.suites).resolve()
    model = args.model or ("Qwen/Qwen3.5-4B" if args.provider == "local" else None)
    revision = args.revision or PINNED_REVISIONS.get((model or "").split(":")[0])
    if args.provider == "api" and not args.no_llm and not model:
        raise SystemExit("--provider api needs --model (a <=10B open-weight model the API offers)")
    if not args.dry_run:
        # Check keys and model names before any long step.
        exams = {e["exam_id"] for path in {data, train}
                 for e in json.loads((path / "exams.json").read_text(encoding="utf-8"))["exams"]}
        if not args.no_tests and any(not (suites / f"{exam}.json").exists() for exam in exams):
            testgen = yaml.safe_load(Path(args.testgen_config).read_text(encoding="utf-8"))["model"]
            check_models([args.testgen_model or testgen["name_or_path"]], testgen["base_url"], testgen["api_key_env"])
        if args.provider == "api" and not args.no_llm:
            check_models([model], args.base_url, args.api_key_env)
    variant = ("-zeroshot" if args.method == "zero_shot" else "") + ("-noevid" if args.no_evidence else "")
    name = args.name or f"{tag_of(data)}-{args.prompt}{variant}{'-eval' if args.eval else ''}"
    out = ROOT / "outputs" / "runs" / name
    out.mkdir(parents=True, exist_ok=True)
    print(f"Run folder: {out}")

    def dataset_files(path):
        folder = ROOT / "outputs" / "datasets" / tag_of(path)
        return folder / "compile.json", folder / "tests.json"

    datasets = [data] if data == train else [data, train]
    for path in datasets:
        compile_path, tests_path = dataset_files(path)
        if args.force or not compile_path.exists():
            run("compile_evidence.py", "--input", path, "--output", compile_path, "--jobs", args.jobs)
        if args.no_tests:
            continue
        run("generate_tests.py", "--dataset", path, "--config", args.testgen_config,
            "--output-dir", suites, "--skip-existing",
            *(["--model", args.testgen_model] if args.testgen_model else []))
        if args.force or not tests_path.exists() or stale_tests(path, tests_path, suites):
            # Without --force, keep results of unchanged suites and run only new/rewritten ones.
            run("run_tests.py", "--input", path, "--suites", suites, "--output", tests_path, "--jobs", args.jobs,
                *([] if args.force else ["--incremental"]))

    data_compile, data_tests = dataset_files(data)
    train_compile, train_tests = dataset_files(train)
    tests_flag = [] if args.no_tests else ["--tests", data_tests]
    # --no-evidence: the model sees no compile or test results at all.
    evidence_flag = [] if args.no_evidence else ["--evidence", data_compile]

    def rules_task1():
        """Rule-based Task 1; exams without a suite fall back to the LLM prediction."""
        if args.no_tests:
            return
        fallback = out / "pred_task1.json"
        run("predict_task1_rules.py", "--input", data, "--evidence", data_compile, "--tests", data_tests,
            "--output", out / "rules_task1.json", *(["--fallback", fallback] if fallback.exists() else []))
        if not args.eval:
            run("package_predictions.py", "--task", "task1", "--input", out / "rules_task1.json",
                "--output", out / "submission_task1_rules.json")

    if args.no_llm:
        rules_task1()
        print(f"\nDone (no LLM). Files in {out}")
        return 0

    configs = make_configs(out, train, model, args.prompt, train_compile,
                           None if args.no_tests else train_tests, eval_mode=args.eval,
                           provider=args.provider, dtype=args.dtype,
                           base_url=args.base_url, key_env=args.api_key_env,
                           extra_body=args.extra_body, method=args.method,
                           evidence=not args.no_evidence, revision=revision, max_cost=args.max_cost,
                           price=args.price, cache_path=args.cache_path, attn=args.attn,
                           tasks=args.tasks, task3_prompt=args.task3_prompt,
                           task3_labels=bool(args.task3_labels), diagnosis_prompt=args.task3_diagnosis_prompt)
    limit = ["--max-samples", args.max_samples] if args.max_samples else []
    task2_labels = out / "pred_task2.json"
    for task in args.tasks:
        pred = out / f"pred_{task}.json"
        extra = []
        if task == "task3" and args.task3_labels:
            labels = Path(args.task3_labels) if args.task3_labels != "task2" else task2_labels
            if labels == out / "pred_task2.json" and (out / "pred_task2_calib.json").exists():
                labels = out / "pred_task2_calib.json"
            if not labels.exists() and not args.dry_run:
                raise SystemExit(f"Task 3 labels not found: {labels} (run task2 first or give a file)")
            extra = ["--task3-labels", labels]
        run("predict.py", "--config", configs[task], "--input", data, "--output", pred,
            *evidence_flag, *tests_flag, *extra, *limit)
        if task == "task2" and not args.no_evidence:
            # Evidence rules (scripts/calibrate_task2.py): +0.04 macro-F1 on train.
            raw, pred = pred, out / "pred_task2_calib.json"
            run("calibrate_task2.py", "--predictions", raw, "--evidence", data_compile, *tests_flag,
                "--output", pred, *(["--input", data] if args.eval else []))
            task2_labels = pred
            if not args.eval:
                run("package_predictions.py", "--task", task, "--input", raw,
                    "--output", out / "submission_task2_raw.json")
        log = out / task / "responses.jsonl"
        if log.exists():
            rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]
            fallbacks = {r["sample_id"] for r in rows if r.get("fallback")}
            salvaged = {r["sample_id"] for r in rows if (r.get("parsed_output") or {}).get("salvaged")}
            print(f"{task}: {len(fallbacks)} samples used the fallback, {len(salvaged)} had labels salvaged "
                  f"(see {log})")
        if args.eval and not args.max_samples:  # metrics need the whole labeled set
            run("evaluate.py", "--config", configs[task], "--predictions", pred, "--input", data,
                "--output", out / f"metrics_{task}.json")
            if task == "task3":
                run("score_task3_train.py", "--input", data, "--predictions", pred)
        else:
            run("package_predictions.py", "--task", task, "--input", pred, "--output", out / f"submission_{task}.json")

    if "task1" in args.tasks and not args.max_samples:
        rules_task1()
    if args.eval and not args.dry_run:
        print("\n=== Metrics (leave-one-exam-out on labeled data) ===")
        for task in args.tasks:
            path = out / f"metrics_{task}.json"
            if path.exists():
                metrics = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                keys = {"task1": ["qwk_total", "mae_total"], "task2": ["macro_f1", "micro_f1"]}.get(task, list(metrics)[:4])
                print(task, {k: metrics[k] for k in keys if k in metrics})
    print(f"\nDone. Files in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
