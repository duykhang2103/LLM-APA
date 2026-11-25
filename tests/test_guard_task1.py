import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from guard_task1 import guard  # noqa: E402

OK = {"strict": True, "relaxed": True, "errors": []}
BROKEN = {"strict": False, "relaxed": False, "errors": ["x"]}
TESTS = {"passed": 10, "total": 15, "pass_rate": 10 / 15, "status_counts": {"pass": 10, "wrong": 5}}
ZERO = dict.fromkeys(["compilable", "io_format", "logic", "edge_case", "complexity", "code_quality"], 0)


def test_compiling_file_scored_broken_takes_the_rule_rubric():
    rubric, reason = guard(ZERO, {"problem_type": "single_problem"}, OK, TESTS)
    assert reason == "compiling_file_scored_broken" and rubric["compilable"] == 1 and rubric["logic"] == 3


def test_compilable_follows_the_compile_result_otherwise():
    model = {**ZERO, "compilable": 1, "logic": 4}
    rubric, reason = guard(model, {"problem_type": "single_problem"}, BROKEN, None)
    assert reason == "compilable_from_compile_result" and rubric == {**model, "compilable": 0}
    assert guard(model, {"problem_type": "single_problem"}, None, None) == (model, None)
    assert guard(model, {"problem_type": "single_problem"}, OK, TESTS) == (model, None)
