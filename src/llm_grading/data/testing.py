"""Run LLM-generated test suites against submissions (allowed by the lecturer).

A suite is generated once per exam by scripts/generate_tests.py and validated
against an LLM-written reference solution, which also produces every expected
output. Nothing here is hand-written per exam and no teacher label is read.

Suite JSON:
    {"exam_id", "mode": "stdio" | "function", "data_files": [...],
     "tests": [{"name", "category", "problem", "input" | "driver", "expected"}]}

stdio tests feed `input` to the student's program. function tests (exams
where students submit only functions) compile the student file together with
a `driver` main() that calls one required function and prints its results;
the student's own main(), if any, is renamed so it cannot clash.
"""

import re
import shutil
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from llm_grading.data.evidence import PRELUDE, VOID_MAIN, build

STUDENT_MAIN = "#define main llm_apa_student_main\n"
STUDENT_MAIN_END = "\n#undef main\n"
OUTPUT_LIMIT = 1 << 20


def program_source(code, test=None, mode="stdio"):
    """Source to build: the student's code, plus the driver in function mode."""
    code = PRELUDE + VOID_MAIN.sub("int main(", code)
    if mode == "stdio":
        return code
    return STUDENT_MAIN + code + STUDENT_MAIN_END + test["driver"]


def normalize(text):
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


PASSING = {"pass", "pass_extra_text"}


def compare(expected, actual):
    """pass; pass_extra_text (the answer follows an input prompt such as
    "Input n: ", which the teacher did not penalise in the training data);
    format_mismatch (same content, different spacing/case/punctuation); wrong."""
    want, got = expected.split(), actual.split()
    if normalize(expected) == normalize(actual) or want == got:
        return "pass"
    if want and got[-len(want):] == want and not any(ch.isdigit() for ch in " ".join(got[: -len(want)])):
        return "pass_extra_text"
    loose = re.compile(r"[^0-9a-z.\-]+")
    if loose.sub(" ", expected.lower()).split() == loose.sub(" ", actual.lower()).split():
        return "format_mismatch"
    return "wrong"


def run_binary(binary, stdin, cwd, timeout):
    try:
        done = subprocess.run(
            [str(binary)], input=stdin.encode("utf-8"), capture_output=True, cwd=cwd, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return "timeout", ""
    output = done.stdout[:OUTPUT_LIMIT].decode("utf-8", errors="replace").replace("\r\n", "\n")
    return ("runtime_error" if done.returncode else "ok"), output


def run_suite(code, suite, compiler, data_dir=None, timeout=2.0, build_timeout=60):
    """Run every test; return a JSON-serialisable report."""
    results, build_errors = [], {}
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as directory:
        for name in suite.get("data_files", []):
            shutil.copy(Path(data_dir, name), Path(directory, name))
        binaries = {}
        for index, test in enumerate(suite["tests"]):
            # stdio builds once; each function driver is its own program.
            key = "program" if suite["mode"] == "stdio" else index
            if key not in binaries:
                binary = Path(directory, f"t{index}.exe")
                ok, errors = build(program_source(code, test, suite["mode"]), binary, compiler, build_timeout, link=True)
                binaries[key] = binary if ok else None
                if not ok:
                    build_errors[test.get("problem") or "program"] = errors[:2]
            binary = binaries[key]
            if binary is None:
                status, actual = "build_error", ""
            else:
                state, actual = run_binary(binary, test.get("input", ""), directory, timeout)
                status = state if state != "ok" else compare(test["expected"], actual)
            results.append({**{k: test.get(k) for k in ("name", "category", "problem")}, "status": status,
                            "input": test.get("input", ""), "expected": test["expected"], "actual": actual})
    return summarize(results, build_errors)


def summarize(results, build_errors):
    def tally(rows):
        return [sum(r["status"] in PASSING for r in rows), len(rows)]

    by = {}
    for field in ("category", "problem"):
        groups = {}
        for row in results:
            if row[field]:
                groups.setdefault(row[field], []).append(row)
        by[field] = {key: tally(rows) for key, rows in sorted(groups.items())}
    passed, total = tally(results)
    failures = [
        {k: (v[:160] if isinstance(v, str) else v) for k, v in row.items()}
        for row in results
        if row["status"] not in PASSING
    ]
    return {
        "passed": passed,
        "total": total,
        "pass_rate": round(passed / total, 4) if total else 0.0,
        "status_counts": dict(Counter(r["status"] for r in results)),
        "by_category": by["category"],
        "by_problem": by["problem"],
        "build_errors": build_errors,
        "failures": failures[:4],
    }


def test_report_text(report):
    """Compact English summary placed in the model input as test_report."""
    if not report["total"]:
        return ""
    parts = [f"Pipeline-generated tests: passed {report['passed']}/{report['total']}."]
    for title, key in (("By problem", "by_problem"), ("By category", "by_category")):
        if report[key]:
            parts.append(title + ": " + ", ".join(f"{k} {p}/{t}" for k, (p, t) in report[key].items()) + ".")
    counts = report["status_counts"]
    if counts.get("pass_extra_text"):
        parts.append(
            f"{counts['pass_extra_text']} of the passed tests printed the right answer after extra prompt text "
            "(not penalised)."
        )
    problems = {k: v for k, v in counts.items() if k not in PASSING}
    if problems:
        parts.append("Failures: " + ", ".join(f"{k} {v}" for k, v in problems.items()) + ".")
    if report["build_errors"]:
        parts.append("Could not build with the test driver: " + ", ".join(report["build_errors"]) + ".")
    for row in report["failures"][:2]:
        if row["status"] in {"wrong", "format_mismatch"}:
            shown = f"input {row['input']!r}; " if row["input"] else ""
            parts.append(
                f"Example failing test {row['name']} ({row['category']}): {shown}"
                f"expected {row['expected']!r}, got {row['actual']!r}."
            )
    return " ".join(parts)


def attach_test_reports(samples, reports):
    """Fill test_report from saved reports; never overwrite a supplied one."""
    for sample in samples:
        report = reports.get(sample["sample_id"])
        if report is None:
            continue
        sample["test_results"] = report
        if not sample.get("test_report"):
            sample["test_report"] = test_report_text(report)
    return samples
