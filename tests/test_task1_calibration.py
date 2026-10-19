"""Template snapping keeps compilable, respects exam type, and supports LOO."""

import pytest

from tasks.task1_grading.calibration import RubricSnapper, is_empty_submission

R = ("compilable", "io_format", "logic", "edge_case", "complexity", "code_quality")


def row(sid, values, problem_type="single_problem"):
    return {"sample_id": sid, "problem_type": problem_type, "rubric": dict(zip(R, values))}


TRAIN = [
    row("a", (1, 1, 4, 2, 1, 1)),
    row("b", (1, 1, 2, 1, 1, 1)),
    row("c", (1, 1, 2, 1, 1, 1)),
    row("d", (0, 0, 0, 0, 0, 1)),
    row("m", (1, 1, 0, 0, 0, 0), "multi_problem"),
]


def test_snap_keeps_compilable_and_uses_exam_type():
    snapper = RubricSnapper(TRAIN)
    single = {"problem_type": "single_problem"}
    rubric, template = snapper.snap(dict(zip(R, (1, 1, 0, 0, 0, 0))), single)
    assert template == (1, 1, 2, 1, 1, 1)  # not the closer compile-error template
    multi = {"problem_type": "multi_problem"}
    assert snapper.snap(dict(zip(R, (1, 1, 0, 0, 0, 0))), multi)[1] == (1, 1, 0, 0, 0, 0)
    assert snapper.snap(dict(zip(R, (0, 1, 3, 2, 1, 1))), single)[0]["compilable"] == 0


def test_leave_one_out_and_min_count():
    snapper = RubricSnapper(TRAIN, min_count=2)
    single = {"problem_type": "single_problem"}
    assert snapper.snap(dict(zip(R, (1, 1, 4, 2, 1, 1))), single)[1] == (1, 1, 2, 1, 1, 1)
    loo = RubricSnapper(TRAIN, min_count=2)
    assert loo.templates("single_problem", exclude_id="b") == {}
    assert RubricSnapper(TRAIN).templates("single_problem", exclude_id="a").get((1, 1, 4, 2, 1, 1)) is None


def test_no_template_leaves_prediction_unchanged():
    rubric, template = RubricSnapper(TRAIN).snap(dict(zip(R, (0, 0, 0, 0, 0, 0))), {"problem_type": "other"})
    assert template is None and sum(rubric.values()) == 0


@pytest.mark.parametrize(
    "code,empty",
    [
        ("", True),
        ("gnjhy", True),
        ("#include <iostream>\n", True),
        ("#include <string>\nvoid f(string s)\n{\n}\n", True),
        ("#include <iostream>\nusing namespace std;\nint main()\n{\n    return 0;\n}", True),
        ("//int main() {\n//  cin >> n;\n//}", False),  # commented-out work was graded 10/10
        ("int main(){ int n; cin >> n; }", False),
    ],
)
def test_empty_submission_rule(code, empty):
    assert is_empty_submission(code) is empty
