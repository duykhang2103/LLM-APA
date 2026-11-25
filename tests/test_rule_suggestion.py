import json

from llm_grading.prompting.formatter import rule_suggestion


def report(passed, total):
    return {"passed": passed, "total": total, "pass_rate": passed / total, "status_counts": {"pass": passed}}


def test_rule_suggestion_follows_compile_and_pass_rate():
    ok = {"strict": True, "relaxed": True, "errors": []}
    half = rule_suggestion({"problem_type": "single_problem", "compile_evidence": ok, "test_results": report(8, 16)})
    assert json.loads(half.split(" (total")[0]) == {
        "compilable": 1, "io_format": 1, "logic": 2, "edge_case": 1, "complexity": 1, "code_quality": 1}
    assert half.endswith("(total 7)")

    broken = {"strict": False, "relaxed": False, "errors": ["x"]}
    single = rule_suggestion({"problem_type": "single_problem", "compile_evidence": broken})
    multi = rule_suggestion({"problem_type": "multi_problem", "compile_evidence": broken})
    assert single.endswith("(total 1)") and multi.endswith("(total 0)")

    assert rule_suggestion({"problem_type": "single_problem"}).startswith("none")
