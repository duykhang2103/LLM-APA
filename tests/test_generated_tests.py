"""Generated-test harness: comparison, reports, suite validation, rule baseline."""

import json
import sys
from pathlib import Path

import pytest

from llm_grading.data import testing
from llm_grading.data.testing import (
    attach_test_reports,
    compare,
    program_source,
    summarize,
)
from llm_grading.data.testing import test_report_text as report_text
from tasks.task1_grading.evidence_rules import rubric_from_evidence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_tests  # noqa: E402


@pytest.mark.parametrize(
    "expected,actual,status",
    [
        ("True\n", "True\r\n", "pass"),
        ("6\n1 5 3\n", "6 \n1 5 3", "pass"),
        ("True", "Input number: True\n", "pass_extra_text"),
        ("6\n1 5 3", "Count: 6\n1 5 3", "pass_extra_text"),
        ("6\n1 5 3", "7 6\n1 5 3", "wrong"),  # extra numbers are not a prompt
        ("True", "TRUE", "format_mismatch"),
        ("1, 2, 3", "1 2 3", "format_mismatch"),
        ("True", "False", "wrong"),
        ("True", "", "wrong"),
    ],
)
def test_compare(expected, actual, status):
    assert compare(expected, actual) == status


def row(status, category="normal", problem=None):
    return {"name": "t", "category": category, "problem": problem, "status": status,
            "input": "5", "expected": "True", "actual": "False"}


def test_summary_counts_extra_text_as_pass_and_explains_failures():
    report = summarize([row("pass"), row("pass_extra_text"), row("wrong", "edge"), row("timeout", "large")], {})
    assert (report["passed"], report["total"], report["pass_rate"]) == (2, 4, 0.5)
    assert report["by_category"] == {"edge": [0, 1], "large": [0, 1], "normal": [2, 2]}
    text = report_text(report)
    assert text.startswith("Pipeline-generated tests: passed 2/4.")
    assert "timeout 1" in text and "expected 'True', got 'False'" in text
    sample = {"sample_id": "S", "test_report": None}
    attach_test_reports([sample, {"sample_id": "other"}], {"S": report})
    assert sample["test_report"] == text and sample["test_results"] is report
    kept = {"sample_id": "S", "test_report": "teacher report"}
    assert attach_test_reports([kept], {"S": report})[0]["test_report"] == "teacher report"


def test_function_mode_renames_student_main_and_appends_driver():
    source = program_source("void main() {}\nint prob1() { return 1; }", {"driver": "int main(){ return prob1(); }"}, "function")
    assert source.index("#define main") < source.index("int main() {}") < source.index("#undef main")
    assert source.rstrip().endswith("int main(){ return prob1(); }")


def test_validate_uses_reference_outputs_and_checks_statement_examples(monkeypatch, tmp_path):
    outputs = {"1634\n": "True\n", "123\n": "False\n", "153\n": "True\n", "": ""}
    monkeypatch.setattr(generate_tests, "build", lambda *a, **k: (True, []))
    monkeypatch.setattr(generate_tests, "run_binary", lambda b, stdin, cwd, t: ("ok", outputs.get(stdin, "False\n")))
    exam = {"exam_id": "E", "exam_type": "single_problem"}
    response = {
        "reference_solution": "int main(){}",
        "examples": [{"input": "1634\n", "output": "True"}, {"input": "123\n", "output": "False"}],
        "tests": [{"name": "a", "category": "normal", "input": "153\n"}, {"name": "b", "category": "edge", "input": "1\n"}],
    }
    suite, report = generate_tests.validate(exam, response, [], tmp_path, ["cc"], min_tests=3)
    assert [t["expected"] for t in suite["tests"]] == ["True\n", "False\n", "True\n", "False\n"]
    assert {t["category"] for t in suite["tests"]} == {"normal", "edge", "example"}
    response["examples"][0]["output"] = "False"
    with pytest.raises(ValueError, match="differs from the statement example"):
        generate_tests.validate(exam, response, [], tmp_path, ["cc"], min_tests=3)


def test_function_suites_need_every_problem(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_tests, "build", lambda *a, **k: (True, []))
    monkeypatch.setattr(generate_tests, "run_binary", lambda *a: ("ok", "1\n"))
    exam = {"exam_id": "M", "exam_type": "multi_problem", "problems": [{"pid": "P1"}, {"pid": "P2"}]}
    response = {"reference_solution": "", "tests": [{"name": "x", "problem": "P1", "driver": "int main(){}"}]}
    with pytest.raises(ValueError, match="P2"):
        generate_tests.validate(exam, response, [], tmp_path, ["cc"], min_tests=1)


def test_parse_accepts_fenced_json_and_rejects_missing_fields():
    assert generate_tests.parse('```json\n{"reference_solution": "x", "tests": []}\n```')["tests"] == []
    with pytest.raises(ValueError):
        generate_tests.parse(json.dumps({"tests": []}))
    noisy = 'Thinking: int f() { return 1; } then {"a": 1}\n' + json.dumps({"reference_solution": "int main(){}", "tests": [1]})
    assert generate_tests.parse(noisy)["tests"] == [1]


def test_rule_baseline():
    single = {"problem_type": "single_problem"}
    assert rubric_from_evidence(single, {"relaxed": False}, None)["code_quality"] == 1
    assert sum(rubric_from_evidence({"problem_type": "multi_problem"}, {"relaxed": False}, None).values()) == 0
    full = {"total": 10, "pass_rate": 1.0, "status_counts": {"pass": 10}}
    assert sum(rubric_from_evidence(single, {"relaxed": True}, full).values()) == 10
    half = {"total": 10, "pass_rate": 0.5, "status_counts": {"pass": 5, "wrong": 5}}
    assert rubric_from_evidence(single, {"relaxed": True}, half)["logic"] == 2
    assert rubric_from_evidence(single, {"relaxed": True}, None) is None


def test_run_suite_reports_build_error_without_compiler(monkeypatch, tmp_path):
    monkeypatch.setattr(testing, "build", lambda *a, **k: (False, ["error: x"]))
    suite = {"mode": "stdio", "tests": [{"name": "a", "category": "normal", "input": "1", "expected": "1"}]}
    report = testing.run_suite("int main(){}", suite, ["cc"], tmp_path)
    assert report["status_counts"] == {"build_error": 1} and report["build_errors"] == {"program": ["error: x"]}


def test_file_problem_without_data_file_may_be_untested(monkeypatch, tmp_path):
    monkeypatch.setattr(generate_tests, "build", lambda *a, **k: (True, []))
    monkeypatch.setattr(generate_tests, "run_binary", lambda *a: ("ok", "1\n"))
    exam = {"exam_id": "M", "exam_type": "multi_problem", "problems": [
        {"pid": "P1", "prototype": "General* prob1(string filename, int& n);"},
        {"pid": "P2", "prototype": "Node* prob2(General* g, int n);"}]}
    response = {"reference_solution": "", "tests": [{"name": "x", "problem": "P2", "driver": "int main(){}"}]}
    suite, _ = generate_tests.validate(exam, response, [], tmp_path, ["cc"], min_tests=1)
    assert suite["untested_problems"] == ["P1"]
    (tmp_path / "data.csv").write_text("h\n", encoding="utf-8")
    with pytest.raises(ValueError, match="P1"):  # the data file exists, so P1 must be tested
        generate_tests.validate(exam, response, ["data.csv"], tmp_path, ["cc"], min_tests=1)
    report = {**summarize([row("pass", problem="P2")], {}), "untested_problems": ["P1"]}
    assert "No tests for P1" in report_text(report)
