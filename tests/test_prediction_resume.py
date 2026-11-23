import json
import sys
from pathlib import Path

import pytest
import yaml
from test_experiment_contracts import sample

from scripts import predict


def test_resume_retains_completed_samples_and_rejects_config_change(tmp_path, monkeypatch):
    source = tmp_path / "samples.json"
    source.write_text(json.dumps([sample("a"), sample("b")]), encoding="utf-8")
    cfg = yaml.safe_load((Path(__file__).parents[1] / "configs/task1/heuristic.yaml").read_text())
    cfg["saving"]["output_dir"] = str(tmp_path / "run")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    output = tmp_path / "predictions.json"
    monkeypatch.setattr(sys, "argv", ["predict", "--config", str(path), "--input", str(source),
                                     "--output", str(output), "--resume"])
    real_builder = predict.build_task_pipeline
    calls = []

    class InterruptedPipeline:
        def predict(self, row):
            calls.append(row["sample_id"])
            if row["sample_id"] == "b":
                raise RuntimeError("interrupted")
            return real_builder(cfg).predict(row)

        def postprocess(self, value):
            return value

    monkeypatch.setattr(predict, "build_task_pipeline", lambda _: InterruptedPipeline())
    with pytest.raises(RuntimeError, match="interrupted"):
        predict.main()
    assert not output.exists()
    state = json.loads(output.with_suffix(".partial.json").read_text())
    assert [r["sample_id"] for r in state["predictions"]] == ["a"]

    pipeline = real_builder(cfg)
    original_predict = pipeline.predict
    resumed = []

    def tracked(row):
        resumed.append(row["sample_id"])
        return original_predict(row)

    monkeypatch.setattr(pipeline, "predict", tracked)
    monkeypatch.setattr(predict, "build_task_pipeline", lambda _: pipeline)
    assert predict.main() == 0
    assert resumed == ["b"]
    assert len(json.loads(output.read_text())) == 2
    cfg["seed"] = 123
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="Resume config/data/prompts changed"):
        predict.main()
