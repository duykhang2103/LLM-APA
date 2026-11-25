import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("run_prompting", ROOT / "prompting" / "run_prompting.py")
run_prompting = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_prompting)


def _model():
    return yaml.safe_load((ROOT / "prompting" / "model" / "prompting_model.yaml").read_text(encoding="utf-8"))


def test_model_file_points_to_existing_prompts():
    model = _model()
    for path in (model["task1"]["prompt"], model["task2"]["list_prompt"], model["task2"]["per_label_prompt"],
                 model["test_generation"]["config"], f"tasks/task3_feedback/prompts/{model['task3']['prompt']}.txt"):
        assert (ROOT / path).exists(), path


def test_test_run_chains_the_submitted_pipeline(tmp_path):
    steps, copies = run_prompting.build_steps(_model(), tmp_path, "x")
    argv = [" ".join(a) for _, a in steps]
    assert "--tasks task1 task2" in argv[0] and "di9b-cache-2.jsonl" in argv[0]
    assert "--task2-method per_label" in argv[1]
    assert "combine_task2.py" in argv[2] and "Lỗi hard-code" in argv[2]
    assert "--task3-prompt v007" in argv[3] and "x-P2" in argv[3] and "--shards 6" in argv[3]
    assert "package_predictions.py" in argv[4]
    assert {p.name for p in copies} == {"submission_task1.json", "submission_task3.json"}


def test_eval_scores_instead_of_packaging(tmp_path):
    steps, copies = run_prompting.build_steps(_model(), tmp_path, "x", evaluate=True)
    assert len(steps) == 4 and not copies
    assert all("--eval" in a for _, a in steps if "run_all.py" in a[1])
    assert not any("--cache-path" in a for _, a in steps)
