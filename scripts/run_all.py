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
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from make_api_configs import make_configs  # noqa: E402

DEFAULT_TRAIN = ROOT.parent / "Data" / "public-test-cham-diem-phan-hoi-files" / "sample_dataset" / "sample_dataset"
SCRIPTS = ROOT / "scripts"


def tag_of(path):
    return re.sub(r"[^A-Za-z0-9_-]+", "-", Path(path).resolve().name).strip("-") or "data"


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
    parser.add_argument("--model", default="llama-3.1-8b-instant", help="Grading model (<=10B)")
    parser.add_argument("--testgen-config", default=str(ROOT / "configs" / "testgen" / "groq.yaml"))
    parser.add_argument("--suites", default=str(ROOT / "outputs" / "testsuites"), help="Shared test-suite folder")
    parser.add_argument("--tasks", nargs="+", default=["task1", "task2", "task3"], choices=["task1", "task2", "task3"])
    parser.add_argument("--name", help="Run folder name under outputs/runs/ (default <data>-<prompt>)")
    parser.add_argument("--eval", action="store_true", help="Leave-one-exam-out run + metrics (labeled data)")
    parser.add_argument("--no-llm", action="store_true", help="Only evidence, tests and rule-based Task 1")
    parser.add_argument("--no-tests", action="store_true", help="Skip test generation/running")
    parser.add_argument("--force", action="store_true", help="Recompute compile evidence and test results")
    parser.add_argument("--jobs", type=int, default=6)
    parser.add_argument("--max-samples", type=int, help="Smoke run on the first N samples")
    parser.add_argument("--dry-run", action="store_true", help="Print the commands only")
    args = parser.parse_args()

    os.environ.setdefault("ZIG_GLOBAL_CACHE_DIR", str(ROOT / "outputs" / "tmp" / "zigcache"))
    os.environ.setdefault("ZIG_LOCAL_CACHE_DIR", os.environ["ZIG_GLOBAL_CACHE_DIR"])
    run = Runner(args.dry_run)
    data, train = Path(args.data).resolve(), Path(args.train).resolve()
    if not (data / "exams.json").exists() or not (train / "exams.json").exists():
        raise SystemExit("--data and --train must be dataset folders containing exams.json")
    suites = Path(args.suites).resolve()
    name = args.name or f"{tag_of(data)}-{args.prompt}{'-eval' if args.eval else ''}"
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
            "--output-dir", suites, "--skip-existing")
        if args.force or not tests_path.exists():
            run("run_tests.py", "--input", path, "--suites", suites, "--output", tests_path, "--jobs", args.jobs)

    data_compile, data_tests = dataset_files(data)
    train_compile, train_tests = dataset_files(train)
    tests_flag = [] if args.no_tests else ["--tests", data_tests]

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

    configs = make_configs(out, train, args.model, args.prompt, train_compile,
                           None if args.no_tests else train_tests, eval_mode=args.eval)
    limit = ["--max-samples", args.max_samples] if args.max_samples else []
    for task in args.tasks:
        pred = out / f"pred_{task}.json"
        extra = ["--task3-labels", out / "pred_task2.json"] if task == "task3" else []
        run("predict.py", "--config", configs[task], "--input", data, "--output", pred,
            "--evidence", data_compile, *tests_flag, *extra, *limit)
        if args.eval:
            run("evaluate.py", "--config", configs[task], "--predictions", pred, "--input", data,
                "--output", out / f"metrics_{task}.json")
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
