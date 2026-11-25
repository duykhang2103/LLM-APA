import hashlib
import json

import pytest

from llm_grading.data.loader import load_samples
from scripts.replay_task1_build_patch import apply_rechecks


def test_source_loader_retains_bytes_only_when_requested(tmp_path):
    (tmp_path / "main.cpp").write_bytes(b"\xef\xbb\xbfint main() {}\r\n")
    dataset = tmp_path / "samples.json"
    dataset.write_text(json.dumps([{"sample_id": "demo", "code_file": "main.cpp"}]))
    assert load_samples(dataset)[0]["code"] == "int main() {}\n"
    assert load_samples(dataset, preserve_source_bytes=True)[0]["code"] == "\ufeffint main() {}\r\n"


def test_compile_only_replay_preserves_other_scores_and_input():
    code = "int main() {"
    statement = "Read an integer."
    sample = {"sample_id": "demo", "code": code, "problem_statement": statement}
    rubric = dict(compilable=1, io_format=1, logic=2, edge_case=1, complexity=1, code_quality=0)
    baseline = [{"sample_id": "demo", "output": {"rubric": rubric, "total_score": 6}}]
    build = {
        "phase": "link", "status": "compile_failure", "profile": "full_program",
        "standard": "c++11", "reproducible": True, "reason": "source_diagnostic",
        "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
        "statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
    }
    check = {"sample_id": "demo", "recorded_reason": "source_diagnostic", "current": build}
    candidate = apply_rechecks([sample], baseline, [check])
    assert candidate[0]["output"] == {"rubric": {**rubric, "compilable": 0}, "total_score": 5}
    assert baseline[0]["output"]["rubric"]["compilable"] == 1
    with pytest.raises(ValueError, match="source-bound"):
        apply_rechecks([sample], baseline, [{**check, "current": {**build, "source_sha256": "wrong"}}])
    with pytest.raises(ValueError, match="duplicate"):
        apply_rechecks([sample], baseline, [check, check])
