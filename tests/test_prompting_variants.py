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


def test_price_sets_the_cost_cap_rates(tmp_path):
    paths = make_configs(tmp_path, tmp_path / "train", "runware/qwen3.5-4b", "v004",
                         max_cost=1.0, price=(0.0525, 0.0735), extra_body={"reasoning_effort": "none"})
    model = yaml.safe_load(paths["task1"].read_text(encoding="utf-8"))["model"]
    assert model["cost_per_million_tokens"] == {"prompt": 0.0525, "completion": 0.0735}
    assert model["max_cost_usd"] == 1.0 and model["extra_body"] == {"reasoning_effort": "none"}


def test_task3_api_config_follows_the_team_notebook(tmp_path):
    paths = make_configs(tmp_path / "a", tmp_path / "train", "m", "v004", tasks=["task1", "task3"])
    assert set(paths) == {"task1", "task3"}
    cfg = yaml.safe_load(paths["task3"].read_text(encoding="utf-8"))
    assert cfg["prompt"]["version"] == "v005" and cfg["model"]["provider"] == "api"
    assert cfg["task3"]["two_stage"] is True and cfg["generation"]["max_new_tokens"] == 1024
    assert yaml.safe_load(paths["task1"].read_text(encoding="utf-8"))["prompt"]["version"] == "v004"
    labelled = make_configs(tmp_path / "b", tmp_path / "train", "m", "v004", tasks=["task3"], task3_labels=True)
    assert yaml.safe_load(labelled["task3"].read_text(encoding="utf-8"))["task3"]["two_stage"] is False


def test_task3_diagnosis_prompt_and_v006(tmp_path):
    paths = make_configs(tmp_path, tmp_path / "train", "m", "v004", tasks=["task3"], task3_prompt="v006",
                         diagnosis_prompt="v004")
    cfg = yaml.safe_load(paths["task3"].read_text(encoding="utf-8"))
    assert cfg["prompt"]["version"] == "v006" and cfg["task3"]["diagnosis_prompt"] == "v004"
    sample = {"sample_id": "s", "problem_type": "multi_problem", "problem_statement": "P", "code": "int f(){}",
              "feedback_level": 2, "error_labels": ["Lỗi logic"]}
    prompt = render_task_prompt(sample, "task3", cfg, [])
    assert "P1 đúng; P2 sai" in prompt and "Lỗi logic" in prompt


def test_v007_renders_and_k_reaches_the_config(tmp_path):
    sample = {"sample_id": "s", "problem_type": "single_problem", "problem_statement": "Sum", "code": "int main(){}"}
    prompt = render_task_prompt(sample, "task1", {"prompt": {"version": "v007"}})
    assert "Suspicion is not a bug" in prompt and '"compilable": 0-1' in prompt
    paths = make_configs(tmp_path, tmp_path / "train", "m", "v007", k=3)
    assert yaml.safe_load(paths["task1"].read_text(encoding="utf-8"))["retrieval"]["k"] == 3
