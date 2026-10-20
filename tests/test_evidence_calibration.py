"""Compile evidence: log attachment, Task 1 flip template, Task 2 label rule."""

import sys
from pathlib import Path

from llm_grading.data import evidence as ev
from tasks.task1_grading.calibration import RubricSnapper

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from calibrate_task2 import COMPILE, calibrate  # noqa: E402

R = ("compilable", "io_format", "logic", "edge_case", "complexity", "code_quality")


def test_relaxed_compile_only_when_strict_fails(monkeypatch):
    calls = []

    def fake(code, compiler, timeout):
        calls.append(code)
        return "int main(" in code and "#include <cmath>" in code, ["error: x"]

    monkeypatch.setattr(ev, "_compile", fake)
    result = ev.compile_evidence("void main() {}", ["cc"])
    assert result == {"strict": False, "relaxed": True, "errors": ["error: x"]} and len(calls) == 2
    sample = {"sample_id": "S", "compile_log": ""}
    ev.attach_evidence([sample], {"S": result})
    assert "teacher treats this as compilable" in sample["compile_log"]


def test_most_frequent_template_for_evidence_flip():
    train = [
        {"sample_id": i, "problem_type": "single_problem", "rubric": dict(zip(R, v))}
        for i, v in enumerate([(1, 1, 2, 1, 1, 1), (1, 1, 2, 1, 1, 1), (1, 1, 4, 2, 1, 1), (0, 0, 0, 0, 0, 1)])
    ]
    snapper = RubricSnapper(train)
    single = {"problem_type": "single_problem"}
    assert tuple(snapper.most_frequent(single, 1).values()) == (1, 1, 2, 1, 1, 1)
    assert tuple(snapper.most_frequent(single, 0).values()) == (0, 0, 0, 0, 0, 1)
    assert snapper.most_frequent(single, 0, exclude_id=3) is None


def test_task2_compile_label_follows_evidence():
    assert calibrate([COMPILE, "Lỗi logic"], True) == ["Lỗi logic"]
    assert calibrate(["Lỗi logic", "Lỗi vòng lặp"], False) == [COMPILE, "Lỗi logic"]
    assert calibrate(["Lỗi logic"], False, compile_only=True) == [COMPILE]
    assert calibrate(["a", "b", "c", "d"], True, max_labels=3) == ["a", "b", "c"]
