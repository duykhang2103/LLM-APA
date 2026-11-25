"""Chạy lại toàn bộ hướng Prompting (Task 1, 2, 3) bằng một lệnh.

Đọc cấu hình đã chốt trong prompting/model/prompting_model.yaml rồi gọi các
script có sẵn trong scripts/ theo đúng thứ tự đã dùng cho bản nộp tốt nhất:

  1. run_all.py: biên dịch, sinh và chạy test, Task 1 + Task 2 (prompt liệt kê nhãn)
  2. run_all.py --task2-method per_label (định nghĩa v001): nguồn nhãn "Lỗi edge case"
  3. run_all.py --task2-method per_label (định nghĩa v002 + ví dụ train): nguồn 3 nhãn hiếm
  4. combine_task2.py: nhãn liệt kê + 3 nhãn hiếm từ bước 3
  5. package_predictions.py + add_edge_case_task2.py: file nộp Task 2
  6. run_all.py --tasks task3: feedback, nhãn lấy từ file nộp Task 2
  7. relabel_task3.py: hậu xử lý Task 3 (Level 1 chuẩn, Level 2 không lời khuyên, nhãn khớp Task 2)
  8. kiểm tra định dạng 3 file trong outputs/runs/<name>/final/

Tập test:   python prompting/run_prompting.py --data <thư mục dataset>
Đo trên train (leave-one-exam-out, in điểm): thêm --eval
Xem lệnh mà không chạy:                      thêm --dry-run
"""

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
MODEL_FILE = ROOT / "prompting" / "model" / "prompting_model.yaml"
RUNS = ROOT / "outputs" / "runs"


def script(name):
    return [sys.executable, str(ROOT / "scripts" / name)]


def tests_file(data):
    """tests.json that run_all.py writes for this dataset."""
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(data).resolve().name).strip("-") or "data"
    return ROOT / "outputs" / "datasets" / tag / "tests.json"


def build_steps(spec, data, name, train=None, evaluate=False):
    """(description, argv) for every step; paths are those run_all.py writes."""
    llm, t1, t2, t3 = spec["llm"], spec["task1"], spec["task2"], spec["task3"]
    common = ["--data", str(data), *(["--train", str(train)] if train else []),
              "--base-url", llm["base_url"], "--api-key-env", llm["api_key_env"], "--model", llm["model"],
              "--price", *map(str, llm["price_usd_per_million"]), "--reasoning-effort", llm["reasoning_effort"],
              "--testgen-config", str(ROOT / spec["test_generation"]["config"]),
              "--k", str(spec["retrieval"]["k"]), *(["--eval"] if evaluate else [])]
    p2, edge_run, rare_run = RUNS / f"{name}-P2", RUNS / f"{name}-t2pl", RUNS / f"{name}-t2pl-v002"
    t3_run, final = RUNS / f"{name}-t3", RUNS / name / "final"

    # Replay the Task 1 cache only on the test data; on train the default caches are used.
    def cache(path):
        return [] if evaluate else ["--cache-path", str(ROOT / path)]

    def per_label(run, definitions, examples=0):
        return [*script("run_all.py"), *common, "--tasks", "task2", "--task2-method", "per_label",
                "--task2-definitions", str(ROOT / definitions),
                *(["--task2-examples", str(examples)] if examples else []),
                "--workers", str(t2["per_label_workers"]), "--max-cost", str(t2["per_label_max_cost_usd"]),
                "--name", run.name]

    combined = rare_run / "pred_task2_combined.json"
    task2_file = final / "submission_task2.json"
    # On train, Task 3 takes the combined internal labels (no edge case: train never has it).
    task3_labels = combined if evaluate else task2_file
    steps = [
        ("Task 1 + Task 2 (liệt kê nhãn)",
         [*script("run_all.py"), *common, "--tasks", "task1", "task2", "--max-cost", str(t1["max_cost_usd"]),
          *cache(t1["cache"]), "--name", p2.name]),
        ("Task 2 (hỏi từng nhãn, nguồn nhãn edge case)", per_label(edge_run, t2["edge_case_prompt"])),
        ("Task 2 (hỏi từng nhãn có ví dụ, nguồn 3 nhãn hiếm)",
         per_label(rare_run, t2["rare_labels_prompt"], t2["rare_label_examples"])),
        ("Task 2 (ghép nhãn liệt kê + 3 nhãn hiếm)",
         [*script("combine_task2.py"), "--list", str(p2 / "pred_task2_calib.json"),
          "--per-label", str(rare_run / "pred_task2_calib.json"), "--labels", *t2["combine_rare_labels"],
          "--output", str(combined), *(["--input", str(data)] if evaluate else [])]),
    ]
    if not evaluate:
        steps += [
            ("Đóng gói Task 2",
             [*script("package_predictions.py"), "--task", "task2", "--input", str(combined),
              "--output", str(rare_run / "submission_task2_combined.json")]),
            ("Task 2 (thêm nhãn edge case)",
             [*script("add_edge_case_task2.py"), "--submission", str(rare_run / "submission_task2_combined.json"),
              "--per-label", str(edge_run / "pred_task2_calib.json"), "--tests", str(tests_file(data)),
              "--output", str(task2_file)]),
        ]
    steps.append(("Task 3 (feedback)",
                  [*script("run_all.py"), *common, "--tasks", "task3", "--task3-prompt", t3["prompt"],
                   "--task3-labels", str(task3_labels), "--shards", str(t3["shards"]),
                   "--max-cost", str(t3["max_cost_usd"]), "--name", t3_run.name]))
    if evaluate:
        return steps, {}
    steps.append(("Task 3 (hậu xử lý)",
                  [*script("relabel_task3.py"), "--task3", str(t3_run / "submission_task3.json"),
                   "--levels", str(t3_run / "pred_task3.json"), "--task2", str(task2_file),
                   "--output", str(final / "submission_task3.json")]))
    return steps, {final / "submission_task1.json": p2 / t1["submit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Thư mục dataset (có exams.json, submissions/)")
    parser.add_argument("--train", help="Thư mục train có nhãn cho RAG (mặc định như run_all.py)")
    parser.add_argument("--name", default="prompting", help="Tiền tố tên thư mục trong outputs/runs/")
    parser.add_argument("--model-file", default=str(MODEL_FILE))
    parser.add_argument("--eval", action="store_true", help="Dữ liệu có nhãn: đo điểm thay vì tạo file nộp")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in lệnh")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")  # Windows consoles default to cp1252

    spec = yaml.safe_load(Path(args.model_file).read_text(encoding="utf-8"))
    data = Path(args.data).resolve()
    steps, copies = build_steps(spec, data, args.name, args.train, args.eval)
    print(f"Model: {spec['name']} ({spec['llm']['model']})")
    for i, (title, argv) in enumerate(steps, 1):
        print(f"\n=== Bước {i}/{len(steps)}: {title} ===\n" + " ".join(argv), flush=True)
        if not args.dry_run:
            if "--output" in argv:
                Path(argv[argv.index("--output") + 1]).parent.mkdir(parents=True, exist_ok=True)
            if subprocess.run(argv, cwd=ROOT).returncode:
                raise SystemExit(f"Bước {i} lỗi. Chạy lại đúng lệnh này: câu trả lời API đã có nằm trong cache.")
    if args.dry_run or args.eval:
        return 0
    for target, source in copies.items():
        shutil.copyfile(source, target)
    for task in ("task1", "task2", "task3"):
        path = RUNS / args.name / "final" / f"submission_{task}.json"
        check = subprocess.run([*script("validate_predictions.py"), "--task", task, "--format", "challenge",
                                *(["--require-task3-taxonomy"] if task == "task3" else []),
                                "--input", str(path)], cwd=ROOT)
        if check.returncode:
            raise SystemExit(f"{path} sai định dạng")
    print(f"\nXong. 3 file nộp nằm trong {RUNS / args.name / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
