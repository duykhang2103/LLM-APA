"""Chạy lại toàn bộ hướng Prompting (Task 1, 2, 3) bằng một lệnh.

Đọc cấu hình đã chốt trong prompting/model/prompting_model.yaml rồi gọi các
script có sẵn trong scripts/ theo đúng thứ tự đã dùng cho bản nộp:

  1. run_all.py: biên dịch, sinh và chạy test, Task 1 + Task 2 (prompt liệt kê nhãn)
  2. run_all.py --task2-method per_label: Task 2 hỏi Yes/No từng nhãn
  3. combine_task2.py: ghép 2 cách hỏi (lấy nhãn hiếm từ cách 2)
  4. run_all.py --tasks task3: feedback, nhãn lấy từ bước 1
  5. chép 3 file nộp vào outputs/runs/<name>/final/ và kiểm tra định dạng

Tập test:   python prompting/run_prompting.py --data <thư mục dataset>
Đo trên train (leave-one-exam-out, in điểm): thêm --eval
Xem lệnh mà không chạy:                      thêm --dry-run
"""

import argparse
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


def build_steps(spec, data, name, train=None, evaluate=False):
    """(description, argv) for every step; paths are those run_all.py writes."""
    llm, t1, t2, t3 = spec["llm"], spec["task1"], spec["task2"], spec["task3"]
    common = ["--data", str(data), *(["--train", str(train)] if train else []),
              "--base-url", llm["base_url"], "--api-key-env", llm["api_key_env"], "--model", llm["model"],
              "--price", *map(str, llm["price_usd_per_million"]), "--reasoning-effort", llm["reasoning_effort"],
              "--testgen-config", str(ROOT / spec["test_generation"]["config"]),
              "--k", str(spec["retrieval"]["k"]), *(["--eval"] if evaluate else [])]
    p2, pl, t3_run = RUNS / f"{name}-P2", RUNS / f"{name}-t2pl", RUNS / f"{name}-t3"
    # Replay caches only on the test data; on train the default caches are used.
    def cache(path):
        return [] if evaluate else ["--cache-path", str(ROOT / path)]

    combined = pl / "pred_task2_combined.json"
    steps = [
        ("Task 1 + Task 2 (liệt kê nhãn)",
         [*script("run_all.py"), *common, "--tasks", "task1", "task2", "--max-cost", str(t1["max_cost_usd"]),
          *cache(t1["cache"]), "--name", p2.name]),
        ("Task 2 (hỏi Yes/No từng nhãn)",
         [*script("run_all.py"), *common, "--tasks", "task2", "--task2-method", "per_label",
          "--workers", str(t2["per_label_workers"]), "--max-cost", str(t2["per_label_max_cost_usd"]),
          "--name", pl.name]),
        ("Task 2 (ghép 2 cách hỏi)",
         [*script("combine_task2.py"), "--list", str(p2 / "pred_task2_calib.json"),
          "--per-label", str(pl / "pred_task2_calib.json"), "--labels", *t2["combine_rare_labels"],
          "--output", str(combined), *(["--input", str(data)] if evaluate else [])]),
        ("Task 3 (feedback)",
         [*script("run_all.py"), *common, "--tasks", "task3", "--task3-prompt", t3["prompt"],
          "--task3-diagnosis-prompt", t3["diagnosis_prompt"], "--task3-diagnosis-tokens", str(t3["diagnosis_tokens"]),
          "--task3-labels", str(p2 / "pred_task2_calib.json"), "--shards", str(t3["shards"]),
          "--timeout", str(t3["timeout_seconds"]), "--max-cost", str(t3["max_cost_usd"]),
          *cache(t3["cache"]), "--name", t3_run.name]),
    ]
    if evaluate:
        return steps, {}
    final = RUNS / name / "final"
    steps.append(("Đóng gói Task 2",
                  [*script("package_predictions.py"), "--task", "task2", "--input", str(combined),
                   "--output", str(final / "submission_task2.json")]))
    copies = {final / "submission_task1.json": p2 / t1["submit"],
              final / "submission_task3.json": t3_run / "submission_task3.json"}
    return steps, copies


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
                                "--input", str(path)], cwd=ROOT)
        if check.returncode:
            raise SystemExit(f"{path} sai định dạng")
    print(f"\nXong. 3 file nộp nằm trong {RUNS / args.name / 'final'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
