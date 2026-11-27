"""Đóng gói thư mục nộp cho giảng viên, chỉ gồm các file pipeline đang dùng.

    python prompting/make_submission.py --name <MSHV>_submission

Tạo (mặc định cạnh repo, ngoài git):
    <name>/code/          mã nguồn chạy được + đúng các prompt trong prompting_model.yaml
    <name>/submission/    task1.json, task2.json, task3.json (lấy từ submissions/)
    <name>/README.md      hướng dẫn (prompting/SUBMISSION_README.md)
    <name>/approach.pdf   nếu truyền --approach

Không chép: .venv, outputs/ (trừ bộ test đã sinh), dữ liệu sinh viên, cache API, tests/,
notebooks, các prompt cũ không dùng (vẫn còn trên GitHub).
"""

import argparse
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
CODE_DIRS = ["src", "scripts", "configs", "prompting", "tasks"]
CODE_FILES = ["pyproject.toml", "requirements.txt", "uv.lock"]
SKIP = shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", "make_submission.py", "SUBMISSION_README.md",
                              "plan.md", "todo.md")


def used_prompts(spec):
    """Paths (relative to ROOT) of every prompt file the submitted pipeline reads."""
    t1, t2, t3 = spec["task1"], spec["task2"], spec["task3"]
    paths = {t1["prompt"], t2["list_prompt"], t2["rare_labels_prompt"], t2["edge_case_prompt"],
             f"tasks/task3_feedback/prompts/{t3['prompt']}.txt"}
    testgen = yaml.safe_load((ROOT / spec["test_generation"]["config"]).read_text(encoding="utf-8"))
    version = testgen["prompt_version"]
    paths |= {f"tasks/testgen/prompts/{version}{suffix}.txt" for suffix in ("", "_function", "_stdio")}
    return sorted(paths)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--name", default="MSHV_submission")
    parser.add_argument("--dest", default=str(ROOT.parent), help="Thư mục chứa gói nộp (mặc định: cạnh repo)")
    parser.add_argument("--approach", help="File approach.pdf để chép vào gói")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    spec = yaml.safe_load((ROOT / "prompting" / "model" / "prompting_model.yaml").read_text(encoding="utf-8"))
    out = Path(args.dest, args.name)
    if out.exists():
        shutil.rmtree(out)
    code = out / "code"
    for name in CODE_DIRS:
        shutil.copytree(ROOT / name, code / name, ignore=SKIP)
    for name in CODE_FILES:
        shutil.copy2(ROOT / name, code / name)

    # Prompt: chỉ giữ đúng các file pipeline đọc.
    keep = used_prompts(spec)
    for missing in [p for p in keep if not (ROOT / p).exists()]:
        sys.exit(f"Thiếu prompt {missing}")
    for prompt_dir in (code / "tasks").glob("*/prompts"):
        for f in prompt_dir.iterdir():
            if f.relative_to(code).as_posix() not in keep:
                f.unlink()

    # Bộ test đã sinh (chỉ đọc đề, không chứa code sinh viên): chạy lại khỏi cần GROQ key.
    shutil.copytree(ROOT / spec["test_generation"]["suites_dir"], code / "outputs" / "testsuites")

    sub = out / "submission"
    sub.mkdir(parents=True)
    for task in ("task1", "task2", "task3"):
        shutil.copy2(ROOT / "submissions" / f"submission_{task}.json", sub / f"{task}.json")
    shutil.copy2(ROOT / "prompting" / "SUBMISSION_README.md", out / "README.md")
    if args.approach:
        shutil.copy2(args.approach, out / "approach.pdf")

    print(f"Đã tạo {out}")
    print(f"Task 3 dùng prompt {spec['task3']['prompt']}; điểm ghi trong model: {spec['public_scores']}")
    print("Prompt đã chép:")
    for p in keep:
        print("  ", p)
    if not args.approach:
        print("Chưa có approach.pdf (thêm bằng --approach <file>)")


if __name__ == "__main__":
    raise SystemExit(main())
