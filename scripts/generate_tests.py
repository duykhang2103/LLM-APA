"""Generate and validate one test suite per exam with an LLM (lecturer-approved).

For each exam the LLM writes a reference solution and test inputs (stdio) or
driver programs (function-only exams). The reference is compiled and run on
every test: expected outputs come from it, tests it cannot run are dropped,
and for stdio exams it must reproduce every example of the statement, else
the exam is retried with the error fed back. Only the exam statement and data
file headers are sent; never student code or teacher labels.

--response-dir reuses saved raw responses (<exam_id>.txt) instead of calling
the API, so a suite can be rebuilt or audited offline.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.evidence import build, default_compiler
from llm_grading.data.testing import PASSING, compare, program_source, run_binary
from llm_grading.models.inference import ModelRunner
from llm_grading.runtime import write_json

PROMPTS = ROOT / "tasks" / "testgen" / "prompts"


def mode_of(exam):
    return "function" if exam["exam_type"] == "multi_problem" else "stdio"


def data_preview(exam, data_dir):
    names = [exam["data_file"]] if exam.get("data_file") else []
    previews = []
    for name in names:
        lines = Path(data_dir, name).read_text(encoding="utf-8", errors="replace").splitlines()
        previews.append({"name": name, "first_lines": lines[:4], "line_count": len(lines)})
    return names, previews


def render_prompt(exam, previews, version):
    mode = mode_of(exam)
    template = (PROMPTS / f"{version}.txt").read_text(encoding="utf-8")
    return template.format_map({
        "exam_type": exam["exam_type"],
        "problems": json.dumps(exam.get("problems", []), ensure_ascii=False),
        "data_files": json.dumps(previews, ensure_ascii=False),
        "statement": exam["statement"],
        "mode_instructions": (PROMPTS / f"{version}_{mode}.txt").read_text(encoding="utf-8").format_map({}),
    })


def parse(raw):
    text = raw.strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response has no JSON object")
    value = json.loads(text[start : end + 1])
    if not isinstance(value.get("reference_solution"), str) or not isinstance(value.get("tests"), list):
        raise ValueError("JSON must contain reference_solution and tests")
    return value


def validate(exam, response, data_names, data_dir, compiler, min_tests):
    """Run the reference; return (suite, report) or raise ValueError with the reason."""
    mode = mode_of(exam)
    reference = response["reference_solution"]
    kept, dropped = [], []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        for name in data_names:
            Path(directory, name).write_bytes(Path(data_dir, name).read_bytes())
        if mode == "stdio":
            binary = Path(directory, "reference.exe")
            ok, errors = build(program_source(reference), binary, compiler, link=True)
            if not ok:
                raise ValueError(f"reference solution does not compile: {errors[:2]}")
            for example in response.get("examples", []):
                state, actual = run_binary(binary, example["input"], directory, 5)
                if state != "ok" or compare(example["output"], actual) not in PASSING:
                    raise ValueError(
                        f"reference output {actual.strip()[:200]!r} differs from the statement example "
                        f"{example['output'].strip()[:200]!r} for input {example['input'][:200]!r}"
                    )
            if not response.get("examples"):
                raise ValueError("no statement examples were copied into 'examples'")
            cases = [{**t, "input": t["input"]} for t in response["tests"]]
            cases += [
                {"name": f"example{i + 1}", "category": "example", "input": e["input"]}
                for i, e in enumerate(response["examples"])
            ]
        else:
            cases = response["tests"]
        for index, test in enumerate(cases):
            if mode == "function":
                binary = Path(directory, f"driver{index}.exe")
                ok, errors = build(program_source(reference, test, mode), binary, compiler, 180, link=True)
                if not ok:
                    dropped.append({"name": test.get("name"), "reason": f"build: {errors[:1]}"})
                    continue
            state, actual = run_binary(binary, test.get("input", ""), directory, 5)
            if state != "ok" or not actual.strip():
                dropped.append({"name": test.get("name"), "reason": state if state != "ok" else "empty output"})
                continue
            kept.append({
                "name": str(test.get("name") or f"t{index + 1}"),
                "category": test.get("category", "normal"),
                "problem": test.get("problem"),
                **({"input": test["input"]} if mode == "stdio" else {"driver": test["driver"]}),
                "expected": actual,
            })
    if len(kept) < min_tests:
        raise ValueError(f"only {len(kept)} usable tests (need {min_tests}); dropped: {dropped[:5]}")
    if mode == "function":
        missing = {p["pid"] for p in exam.get("problems", [])} - {t["problem"] for t in kept}
        if missing:
            raise ValueError(f"no usable driver for problems {sorted(missing)}; dropped: {dropped[:5]}")
    suite = {"exam_id": exam["exam_id"], "mode": mode, "data_files": data_names, "tests": kept}
    return suite, {"kept": len(kept), "dropped": dropped}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Dataset folder with exams.json (and data_files/)")
    parser.add_argument("--config", default=str(ROOT / "configs" / "testgen" / "groq.yaml"))
    parser.add_argument("--model", help="Override the config's model name")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--exam", action="append", help="Only these exam_ids (repeatable)")
    parser.add_argument("--response-dir", help="Use saved raw responses instead of the API")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--min-tests", type=int, default=6)
    parser.add_argument("--skip-existing", action="store_true", help="Keep suites already in --output-dir")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    exams = json.loads((dataset / "exams.json").read_text(encoding="utf-8"))["exams"]
    exams = [e for e in exams if not args.exam or e["exam_id"] in args.exam]
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if args.model and args.model != config["model"]["name_or_path"]:
        config["model"].update(name_or_path=args.model, cost_per_million_tokens={})
    runner = None if args.response_dir else ModelRunner(config)
    compiler = default_compiler(native=True)
    output = Path(args.output_dir)
    (output / "raw").mkdir(parents=True, exist_ok=True)
    summary = {}
    for exam in exams:
        if args.skip_existing and (output / f"{exam['exam_id']}.json").exists():
            print(f"{exam['exam_id']}: suite exists, skipped")
            continue
        names, previews = data_preview(exam, dataset / "data_files")
        prompt = render_prompt(exam, previews, config.get("prompt_version", "v001"))
        error, log = None, []
        for attempt in range(args.attempts):
            if args.response_dir:
                if attempt:
                    break
                raw = Path(args.response_dir, f"{exam['exam_id']}.txt").read_text(encoding="utf-8")
            else:
                retry = "" if error is None else (
                    f"\n\nYour previous answer was rejected: {error}\nFix it and return the full JSON again."
                )
                raw = runner.generate(prompt + retry)
            (output / "raw" / f"{exam['exam_id']}.attempt{attempt + 1}.txt").write_text(raw, encoding="utf-8")
            try:
                suite, report = validate(exam, parse(raw), names, dataset / "data_files", compiler, args.min_tests)
            except (ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                error = str(exc)
                log.append({"attempt": attempt + 1, "error": error[:500]})
                print(f"{exam['exam_id']}: attempt {attempt + 1} rejected: {error[:300]}")
                continue
            suite["generator"] = {
                "model": config["model"]["name_or_path"],
                "prompt_version": config.get("prompt_version", "v001"),
                "attempt": attempt + 1,
                "compiler": compiler,
            }
            write_json(output / f"{exam['exam_id']}.json", suite)
            summary[exam["exam_id"]] = {**report, "attempts": log}
            print(f"{exam['exam_id']}: {report['kept']} tests kept, {len(report['dropped'])} dropped")
            break
        if exam["exam_id"] not in summary:
            summary[exam["exam_id"]] = {"failed": True, "attempts": log}
    summary_path = output / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    write_json(summary_path, {**previous, **summary})
    if runner is not None:
        print(f"API cost this run: ${runner.total_cost:.4f}")
    failed = [k for k, v in summary.items() if v.get("failed")]
    if failed:
        print(f"No suite for {failed}; their samples simply get no test_report.")


if __name__ == "__main__":
    raise SystemExit(main())
