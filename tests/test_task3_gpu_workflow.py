import ast
import json
from pathlib import Path

from scripts.make_api_configs import make_configs

ROOT = Path(__file__).resolve().parents[1]


def test_api_configs_exclude_task3_but_local_keeps_it(tmp_path):
    assert set(make_configs(tmp_path / "api", tmp_path / "train", "model", "v005")) == {"task1", "task2"}
    assert "task3" in make_configs(tmp_path / "local", tmp_path / "train", "model", "v005", provider="local")


def test_gpu_notebook_is_local_and_resumable():
    notebook = json.loads((ROOT / "notebooks/run_task3_kaggle_t4x2.ipynb").read_text(encoding="utf-8"))
    source = "\n".join("".join(c["source"]) for c in notebook["cells"])
    assert "'provider':'local'" in source
    assert "'--resume'" in source
    assert "fallback_on_failure=True" in source
    assert "OPENROUTER_API_KEY" not in source
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == []
            ast.parse("".join(cell["source"]))
