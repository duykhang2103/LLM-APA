import json
from unittest.mock import patch

import numpy as np
import pytest
from test_experiment_contracts import RUBRIC, sample

from llm_grading.prompting.pipeline import ModelPipeline
from llm_grading.retrieval.retriever import Retriever


class Runner:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.metadata = {"latency_seconds": 0}

    def generate(self, prompt):
        return next(self.responses)


class Encoder:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, 0.0] for _ in texts])


def config(tmp_path, task="task1"):
    return {
        "task": task,
        "method": "zero_shot",
        "prompt": {"version": "v001"},
        "saving": {"output_dir": str(tmp_path)},
    }


def test_one_retry_and_logged_failure(tmp_path):
    cfg = config(tmp_path)
    pipeline = ModelPipeline(cfg, Runner(["bad", json.dumps(RUBRIC)]))
    assert pipeline.predict(sample())["total"] == 10
    logs = [
        json.loads(x) for x in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]
    assert logs[0]["parse_error"] and logs[1]["retry_count"] == 1
    pipeline = ModelPipeline(cfg, Runner(["bad", "bad"]))
    with pytest.raises(ValueError, match="after one retry"):
        pipeline.predict(sample())


def test_rag_excludes_self_and_duplicates_and_feedback(tmp_path):
    rows = [sample("self"), sample("duplicate"), sample("other", "int f() {return 2;}")]
    cfg = config(tmp_path)
    cfg["retrieval"] = {"k": 3, "exclude_duplicates": True}
    retriever = Retriever(rows, "task1", cfg, Encoder())
    examples, audit = retriever.retrieve(sample("self"))
    assert len(examples) == 1 and audit["shortfall"] == 2
    assert audit["exact_matches"] == ["duplicate"]
    assert "SECRET" not in json.dumps(examples)
    cfg["retrieval"]["exclude_duplicates"] = False
    examples, _ = retriever.retrieve(sample("self"))
    assert len(examples) == 2


def test_task3_same_level_filter(tmp_path):
    rows = [sample("same", "one"), sample("wrong", "two")]
    rows[1]["feedback_level"] = 1
    cfg = config(tmp_path, "task3")
    cfg["retrieval"] = {"k": 1, "exclude_duplicates": False}
    examples, audit = Retriever(rows, "task3", cfg, Encoder()).retrieve(
        sample("q", "three")
    )
    assert audit["retrieved"][0]["sample_id"] == "same"
    assert examples[0]["target"] == "SECRET TEACHER TARGET"


def test_api_request_and_usage_without_network(tmp_path, monkeypatch):
    from llm_grading.models.inference import ModelRunner

    cfg = config(tmp_path)
    cfg.update(
        model={
            "provider": "api",
            "allow_private_data": True,
            "name_or_path": "fake",
            "base_url": "https://example.invalid/v1",
        },
        generation={"temperature": 0, "max_new_tokens": 20},
    )
    monkeypatch.setenv("LLM_API_KEY", "test-only")
    from io import BytesIO

    with patch(
        "urllib.request.urlopen",
        return_value=BytesIO(
            json.dumps(
                {
                    "choices": [{"message": {"content": "ok"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                }
            ).encode()
        ),
    ) as request:
        runner = ModelRunner(cfg)
        assert runner.generate("prompt") == "ok"
        assert runner.metadata["usage"]["prompt_tokens"] == 4
        assert b"prompt" in request.call_args.args[0].data


def test_real_dense_encoder_without_download(tmp_path):
    from sentence_transformers import SentenceTransformer
    from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
    from test_training_runtime import tiny_model

    tiny_model(tmp_path / "embedding-model")
    transformer = Transformer(str(tmp_path / "embedding-model"), max_seq_length=64)
    encoder = SentenceTransformer(
        modules=[
            transformer,
            Pooling(transformer.get_embedding_dimension()),
        ]
    )
    saved = tmp_path / "encoder"
    encoder.save(str(saved))
    cfg = config(tmp_path)
    cfg["retrieval"] = {
        "k": 1,
        "embedding_model": str(saved),
        "exclude_duplicates": True,
    }
    retriever = Retriever([sample("a", "code"), sample("b", "hello")], "task1", cfg)
    examples, audit = retriever.retrieve(sample("query", "code hello"))
    assert len(examples) == 1 and len(audit["retrieved"]) == 1


def test_f0_needs_adapter(tmp_path):
    cfg = config(tmp_path)
    cfg.update(method="lora", model={})
    with pytest.raises(ValueError, match="requires model.adapter_path"):
        ModelPipeline(cfg, Runner([]))


def test_feedback_violation_is_retried_and_persistent_failure_stops(tmp_path):
    cfg = config(tmp_path, "task3")
    row = sample()
    pipeline = ModelPipeline(
        cfg, Runner(["Chỉ cần thêm cur = cur->next;", "Vòng lặp thiếu cập nhật."])
    )
    assert pipeline.predict(row)["compliance"]["pass"]
    logs = [
        json.loads(line)
        for line in (tmp_path / "responses.jsonl").read_text().splitlines()
    ]
    assert logs[0]["parse_error"] and logs[1]["retry_count"] == 1
    pipeline = ModelPipeline(cfg, Runner(["Chỉ cần thêm cur = cur->next;"] * 2))
    with pytest.raises(ValueError, match="retry"):
        pipeline.predict(row)


def test_feedback_does_not_claim_input_labels_as_a_diagnosis(tmp_path):
    row = sample()
    row["error_labels"] = ["Lỗi logic"]
    output = ModelPipeline(config(tmp_path, "task3"), Runner(["Tốt."])).predict(row)
    assert "diagnosed_labels" not in output
