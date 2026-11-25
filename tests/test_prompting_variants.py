import json

import yaml

from llm_grading.prompting.formatter import render_task_prompt
from scripts.make_api_configs import make_configs


def test_variants_set_method_evidence_and_revision(tmp_path):
    p2 = make_configs(tmp_path / "p2", tmp_path / "train", "Qwen/Qwen3.5-4B", "v004",
                      train_compile=tmp_path / "c.json", train_tests=tmp_path / "t.json",
                      provider="local", revision="abc")
    cfg = yaml.safe_load(p2["task1"].read_text(encoding="utf-8"))
    assert cfg["method"] == "rag" and cfg["model"]["revision"] == "abc"
    assert cfg["evidence"]["train_compile"] and cfg["task1"].get("use_compile_log", True)

    p0 = make_configs(tmp_path / "p0", tmp_path / "train", "Qwen/Qwen3.5-4B", "v004",
                      train_compile=tmp_path / "c.json", train_tests=tmp_path / "t.json",
                      method="zero_shot", evidence=False)
    for task in ("task1", "task2"):
        cfg = yaml.safe_load(p0[task].read_text(encoding="utf-8"))
        assert cfg["method"] == "zero_shot"
        assert cfg[task]["use_compile_log"] is False and cfg[task]["use_test_report"] is False
        assert cfg["evidence"] == {"train_compile": None, "train_tests": None}


def test_hidden_evidence_is_hidden_in_rag_examples_too():
    sample = {"sample_id": "s", "problem_type": "single_problem", "problem_statement": "Sum",
              "code": "int main(){}", "compile_log": "QUERY_LOG", "test_report": "QUERY_TESTS"}
    example = {"input": {"code": "x", "compile_log": "EXAMPLE_LOG", "test_report": "EXAMPLE_TESTS"},
               "target": {"compilable": 1}}
    config = {"prompt": {"version": "v004"}, "task1": {"use_compile_log": False, "use_test_report": False}}
    prompt = render_task_prompt(sample, "task1", config, [example])
    assert not any(word in prompt for word in ("QUERY_LOG", "QUERY_TESTS", "EXAMPLE_LOG", "EXAMPLE_TESTS"))
    shown = render_task_prompt(sample, "task1", {"prompt": {"version": "v004"}}, [example])
    assert "EXAMPLE_LOG" in shown and json.dumps("QUERY_LOG") in shown
