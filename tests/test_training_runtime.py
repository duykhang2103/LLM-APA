"""Actual tiny CPU adapter training/reload/resume; no network or GPU."""

import copy
import json
from pathlib import Path

import pytest
from test_experiment_contracts import sample

from llm_grading.models.generation import generate_text
from llm_grading.models.loader import load_model
from llm_grading.training.dataset import tokenize_record
from llm_grading.training.trainer import resume_path, train


def tiny_model(path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tok = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "hello": 3, "code": 4, "Answer": 5},
            unk_token="[UNK]",
        )
    )
    tok.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tok, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]"
    )
    tokenizer.save_pretrained(path)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=6,
            n_positions=2048,
            n_embd=16,
            n_layer=1,
            n_head=2,
            eos_token_id=1,
            pad_token_id=0,
            bos_token_id=1,
        )
    )
    model.save_pretrained(path)
    return tokenizer


def test_mask_budget_and_padding(tmp_path):
    from llm_grading.training.dataset import CompletionCollator

    tokenizer = tiny_model(tmp_path / "model")
    row = tokenize_record(
        {"sample_id": "s", "prompt": "code", "target": "hello"}, tokenizer, 20
    )
    assert -100 in row["labels"] and 3 in row["labels"]
    with pytest.raises(ValueError, match="exceed"):
        tokenize_record(
            {"sample_id": "s", "prompt": "code " * 50, "target": "hello"}, tokenizer, 2
        )
    collated = CompletionCollator(tokenizer)([row, {k: v[:-1] for k, v in row.items()}])
    assert collated["labels"][1, -1].item() == -100
    assert collated["attention_mask"][1, -1].item() == 0


@pytest.mark.parametrize(
    "task,generated_metrics,warmup_steps",
    [
        ("task1", False, None),
        ("task1", True, None),
        ("task1", True, 100),
        ("task2", True, None),
        ("task3", True, None),
    ],
)
def test_train_save_reload_resume(tmp_path, task, generated_metrics, warmup_steps):
    path = tmp_path / "model"
    tiny_model(path)
    cfg = {
        "experiment": {"id": "cpu-smoke"},
        "task": task,
        "method": "lora",
        "seed": 42,
        "model": {
            "provider": "local",
            "name_or_path": str(path),
            "dtype": "float32",
            "revision": "main",
        },
        "data": {"dataset_version": "synthetic", "split_version": "cpu"},
        "prompt": {"version": "v002" if task == "task3" else "v001"},
        "quantization": {"load_in_4bit": False},
        "lora": {"rank": 2, "alpha": 4, "dropout": 0.0, "target_modules": "all-linear"},
        "training": {
            "max_sequence_length": 1024,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
            "warmup_ratio": 0.06,
            "gradient_checkpointing": False,
            "max_steps": 2,
        },
        "logging": {"logging_steps": 1},
        "saving": {
            "output_dir": str(tmp_path / "run"),
            "save_strategy": "steps",
            "save_steps": 1,
            "save_total_limit": 3,
        },
        "evaluation": {
            "strategy": "steps",
            "eval_steps": 1,
            "generated_metrics": generated_metrics,
        },
        "generation": {"max_input_tokens": 1024, "max_new_tokens": 2},
    }
    if warmup_steps is not None:
        cfg["training"]["warmup_steps"] = warmup_steps
    original_config = copy.deepcopy(cfg)
    tr = [sample("train", "int main() {}")]
    val = [sample("val", "int f() {return 2;}")]
    result = train(cfg, tr, val, smoke=True)
    assert result["status"] == "completed" and result["lora_parameters_updated"]
    assert result["best_checkpoint"] is None and result["best_metric"] is None
    assert result["effective_training_settings"] == {
        "warmup_steps": 0,
        "max_steps": 2,
        "load_best_model_at_end": False,
    }
    assert result["lora_update_diagnostics"]["changed_tensors"] > 0
    assert result["lora_update_diagnostics"]["max_abs_delta"] > 0
    assert cfg == original_config
    state = json.loads((tmp_path / "run" / "trainer_state.json").read_text())
    assert next(r["learning_rate"] for r in state["log_history"] if "loss" in r) > 0
    assert (tmp_path / "run" / "training_data_audit.json").exists()
    if generated_metrics:
        if task == "task3":
            assert result["best_checkpoint"] is None and result["best_metric"] is None
            assert "eval_heuristic_compliance_score" in result["training_metrics"]
            assert "eval_selection_score" not in result["training_metrics"]
        else:
            assert "eval_selection_score" in result["training_metrics"]
        assert (tmp_path / "run" / "generated_validation" / "step-1.json").exists()
    cp = resume_path(str(tmp_path / "run" / "checkpoint-1"), tmp_path / "run")
    assert Path(cp, "optimizer.pt").exists()
    assert resume_path(True, tmp_path / "run").endswith("checkpoint-2")
    cfg["model"]["adapter_path"] = result["adapter_path"]
    bundle = load_model(cfg)
    assert isinstance(generate_text(bundle, "hello", cfg["generation"]), str)
    cfg["model"].pop("adapter_path")
    with pytest.raises(ValueError, match="contract differs"):
        train(cfg, tr, val, resume=cp)
    resumed = train(cfg, tr, val, resume=cp, smoke=True)
    assert resumed["status"] == "completed" and resumed["resumed_from"] == cp
    assert (
        resumed["effective_training_settings"] == result["effective_training_settings"]
    )
    altered = [sample("new", "int different() {}")]
    with pytest.raises(ValueError, match="contract differs"):
        train(cfg, altered, val, resume=cp, smoke=True)
    cfg["saving"]["output_dir"] = str(tmp_path / "full-run")
    full = train(cfg, tr, val)
    assert full["effective_training_settings"]["warmup_steps"] == (
        warmup_steps if warmup_steps is not None else 1
    )
    assert full["effective_training_settings"]["load_best_model_at_end"] == (
        generated_metrics and task != "task3"
    )
    if generated_metrics and task != "task3":
        assert full["best_checkpoint"]
    full_cp = str(tmp_path / "full-run" / "checkpoint-1")
    full_resumed = train(cfg, tr, val, resume=full_cp)
    assert full_resumed["status"] == "completed"
    assert full_resumed["resumed_from"] == str(Path(full_cp).resolve())
    with pytest.raises(ValueError, match="contract differs"):
        train(cfg, altered, val, resume=full_cp)


def test_invalid_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Not a resumable"):
        resume_path(str(tmp_path), tmp_path)


def test_smoke_still_rejects_no_optimizer_update(tmp_path):
    from llm_grading.utils.config import load_config

    path = tmp_path / "model"
    tiny_model(path)
    cfg = load_config(
        Path(__file__).resolve().parents[1] / "configs/task1/qwen_smoke.yaml"
    )
    cfg["model"].update(name_or_path=str(path), dtype="float32")
    cfg["quantization"]["load_in_4bit"] = False
    cfg["training"].update(learning_rate=0.0, gradient_checkpointing=False)
    cfg["saving"]["output_dir"] = str(tmp_path / "run")
    with pytest.raises(RuntimeError, match="did not update LoRA parameters"):
        train(
            cfg,
            [sample("train", "int main() {}")],
            [sample("val", "int f() {return 2;}")],
            smoke=True,
        )
    manifest = json.loads((tmp_path / "run" / "run_manifest.json").read_text())
    assert manifest["status"] == "failed"
    assert manifest["lora_update_diagnostics"] == {
        "changed_tensors": 0,
        "max_abs_delta": 0.0,
        "global_step": 2,
    }


def test_qwen_completion_logits_preserve_loss_and_gradients():
    from types import SimpleNamespace

    import torch
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    from llm_grading.training.dataset import CompletionCollator

    torch.manual_seed(42)
    model = Qwen3_5ForCausalLM(
        Qwen3_5TextConfig(
            vocab_size=6,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=16,
            layer_types=["full_attention"],
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        )
    )
    model.eval()
    batch = CompletionCollator(SimpleNamespace(pad_token_id=0))(
        [
            {
                "input_ids": [3, 4, 3, 1],
                "attention_mask": [1, 1, 1, 1],
                "labels": [-100, -100, 3, 1],
            },
            {
                "input_ids": [4, 3, 4, 3, 1],
                "attention_mask": [1, 1, 1, 1, 1],
                "labels": [-100, -100, -100, 3, 1],
            },
        ]
    )
    assert batch["logits_to_keep"].tolist() == [1, 2, 3]
    assert batch["shift_labels"].tolist() == [[3, 1, -100], [-100, 3, 1]]
    full = model(**{k: batch[k] for k in ("input_ids", "attention_mask", "labels")})
    full.loss.backward()
    gradients = {
        name: p.grad.clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    model.zero_grad(set_to_none=True)
    reduced = model(**batch)
    reduced.loss.backward()
    torch.testing.assert_close(reduced.loss, full.loss)
    torch.testing.assert_close(
        reduced.logits, full.logits[:, batch["logits_to_keep"], :]
    )
    assert reduced.logits.shape[1] < full.logits.shape[1]
    for name, parameter in model.named_parameters():
        if name in gradients:
            torch.testing.assert_close(parameter.grad, gradients[name])
    with pytest.raises(ValueError, match="no supervised completion tokens"):
        CompletionCollator(SimpleNamespace(pad_token_id=0))(
            [{"input_ids": [3, 4], "attention_mask": [1, 1], "labels": [-100, -100]}]
        )


def test_coverage_failure_writes_audit_before_model_load(tmp_path, monkeypatch):
    import json

    from llm_grading.data.quality import TrainingQualityError

    def unexpected_load(*args, **kwargs):
        raise AssertionError("Model must not load before coverage is checked")

    monkeypatch.setattr("llm_grading.training.trainer.load_model", unexpected_load)
    cfg = {
        "task": "task2",
        "data": {"quality": {"require_coverage": True}},
        "saving": {"output_dir": str(tmp_path)},
    }
    with pytest.raises(TrainingQualityError):
        train(cfg, [sample("train", "int a(){}")], [sample("val", "int b(){}")])
    report = json.loads((tmp_path / "training_data_audit.json").read_text())
    assert len(report["retained_coverage"]["missing_labels"]) == 10


def test_qwen_text_loader_preserves_full_checkpoint_weights(tmp_path):
    """Exercise actual Qwen full-checkpoint → text-only mapping, not a mock."""
    import torch
    from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration

    path = tmp_path / "qwen"
    tiny_model(path)  # reuse a local tokenizer, then replace only model artifacts
    full = Qwen3_5Config(
        text_config={
            "vocab_size": 6,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 1,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 16,
            "layer_types": ["full_attention"],
            "max_position_embeddings": 512,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 2],
            },
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "patch_size": 2,
            "temporal_patch_size": 1,
            "spatial_merge_size": 1,
            "out_hidden_size": 32,
            "num_position_embeddings": 16,
        },
    )
    model = Qwen3_5ForConditionalGeneration(full)
    weights = model.model.language_model.embed_tokens.weight.detach().clone()
    model.save_pretrained(path)
    cfg = {
        "task": "task1",
        "model": {"name_or_path": str(path), "dtype": "float32", "device_map": "cpu"},
    }
    bundle = load_model(cfg)
    assert torch.equal(weights, bundle["model"].model.embed_tokens.weight)
