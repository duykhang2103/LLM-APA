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
import re
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
    """The last JSON object with reference_solution and tests. Reasoning
    models may put text (and C++ braces) before it, so scan every '{'."""
    decoder, found = json.JSONDecoder(), None
    for index, char in enumerate(raw):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(raw, index)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "reference_solution" in value:
            found = value
    if found is None:
        raise ValueError("response has no JSON object with reference_solution")
    if not isinstance(found.get("reference_solution"), str) or not isinstance(found.get("tests"), list):
        raise ValueError("JSON must contain reference_solution (string) and tests (list)")
    return found


# Models ignore prompt rules, so the main checks are enforced here.
STRUCT_ARRAY = re.compile(r"\b([A-Z]\w*)\s+(\w+)\s*\[[^\]]*\]\s*[;={]")
NULL_LINE = re.compile(r"^\s*(null|nullptr)\s*$", re.I | re.M)


def file_problems(exam, data_names):
    """Problems that read a data file which is not supplied: untestable."""
    if data_names:
        return set()
    return {p["pid"] for p in exam.get("problems", []) if "file" in p.get("prototype", "").lower()}


def screen_function_tests(exam, tests, data_names, strict=True):
    """Drop tests of untestable problems. Drivers that declare struct arrays
    on the stack (students may delete[] what the real program allocated with
    new) raise when strict, so the model rewrites them; otherwise (last
    attempt) only those tests are dropped."""
    skip = file_problems(exam, data_names)
    kept = [t for t in tests if t.get("problem") not in skip]
    dropped = [{"name": t.get("name"), "reason": "needs an unavailable data file"} for t in tests if t not in kept]
    stack = {t.get("name"): m.group(0).strip() for t in kept for m in STRUCT_ARRAY.finditer(t.get("driver", ""))}
    if stack and strict:
        raise ValueError(
            "drivers declare struct arrays on the stack; allocate every array and node passed to a required "
            "function with new and set every field: " + "; ".join(f"{k}: {v}" for k, v in list(stack.items())[:6])
        )
    dropped += [{"name": name, "reason": "struct array on the stack"} for name in stack]
    return [t for t in kept if t.get("name") not in stack], dropped


def validate(exam, response, data_names, data_dir, compiler, min_tests, strict=True):
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
            cases = []
            for index, test in enumerate(response["tests"]):
                if isinstance(test.get("input"), str):
                    cases.append(test)
                    continue
                # Large inputs come from a small C++ program that prints them.
                program = test.get("generator")
                built = Path(directory, f"generator{index}.exe")
                ok, errors = build(program, built, compiler, 180, link=True) if isinstance(program, str) else (False, [])
                state, text = run_binary(built, "", directory, 10) if ok else ("build", "")
                if state != "ok" or not text.strip():
                    dropped.append({"name": test.get("name"), "reason": f"generator: {state} {errors[:1]}"})
                    continue
                cases.append({**test, "input": text})
            cases += [
                {"name": f"example{i + 1}", "category": "example", "input": e["input"]}
                for i, e in enumerate(response["examples"])
            ]
        else:
            cases, dropped = screen_function_tests(exam, response["tests"], data_names, strict)
        null_allowed = bool(re.search(r"\bnull\b|nullptr", exam.get("statement", ""), re.I))
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
            if mode == "function" and not null_allowed and NULL_LINE.search(actual):
                dropped.append({"name": test.get("name"), "reason": "checks a NULL result the statement does not specify"})
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
        problems = exam.get("problems", [])
        # A file-reading problem cannot be tested when the data file is not supplied.
        untestable = file_problems(exam, data_names)
        missing = {p["pid"] for p in problems} - {t["problem"] for t in kept}
        if missing - untestable:
            raise ValueError(f"no usable driver for problems {sorted(missing - untestable)}; dropped: {dropped[:5]}")
    suite = {"exam_id": exam["exam_id"], "mode": mode, "data_files": data_names, "tests": kept}
    if mode == "function" and missing:
        suite["untested_problems"] = sorted(missing)
    return suite, {"kept": len(kept), "dropped": dropped}


def request_config(config, prompt):
    """Cap max_tokens so prompt + completion fit the provider's per-request
    token limit (Groq counts max_tokens against tokens-per-minute)."""
    limit = config["generation"].get("request_token_limit")
    if not limit:
        return config
    estimate = len(prompt) // 3 + 200  # conservative tokens for code + Vietnamese text
    budget = min(config["generation"]["max_new_tokens"], limit - estimate)
    if budget < 2000:
        raise RuntimeError(f"prompt too long for request_token_limit={limit} (about {estimate} tokens)")
    return {**config, "generation": {**config["generation"], "max_new_tokens": budget}}


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
                # The attempt number keeps identical retries from hitting the response cache.
                retry = "" if error is None else (
                    f"\n\nAttempt {attempt + 1}. Your previous answer was rejected: {error}\n"
                    "Fix it and return the full JSON again."
                )
                try:
                    raw = runner.generate(prompt + retry, request_config(config, prompt + retry))
                except RuntimeError as exc:  # API/quota error: skip this exam, keep the others
                    error = str(exc)
                    log.append({"attempt": attempt + 1, "error": error[:500]})
                    print(f"{exam['exam_id']}: API error, exam skipped: {error[:300]}")
                    break
            (output / "raw" / f"{exam['exam_id']}.attempt{attempt + 1}.txt").write_text(raw, encoding="utf-8")
            try:
                last = args.response_dir or attempt == args.attempts - 1
                suite, report = validate(exam, parse(raw), names, dataset / "data_files", compiler, args.min_tests,
                                         strict=not last)
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
