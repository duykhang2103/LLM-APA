"""Actual tiny CPU adapter training/reload/resume; no network or GPU."""

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


def test_train_save_reload_resume(tmp_path):
    path = tmp_path / "model"
    tiny_model(path)
    cfg = {
        "experiment": {"id": "cpu-smoke"},
        "task": "task1",
        "method": "lora",
        "seed": 42,
        "model": {
            "provider": "local",
            "name_or_path": str(path),
            "dtype": "float32",
            "revision": "main",
        },
        "data": {"dataset_version": "synthetic", "split_version": "cpu"},
        "prompt": {"version": "v001"},
        "quantization": {"load_in_4bit": False},
        "lora": {"rank": 2, "alpha": 4, "dropout": 0.0, "target_modules": "all-linear"},
        "training": {
            "max_sequence_length": 1024,
            "num_train_epochs": 1,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 0.001,
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
        "evaluation": {"strategy": "steps", "eval_steps": 1},
        "generation": {"max_input_tokens": 1024, "max_new_tokens": 2},
    }
    tr = [sample("train", "int main() {}")]
    val = [sample("val", "int f() {return 2;}")]
    result = train(cfg, tr, val, smoke=True)
    assert result["status"] == "completed" and result["lora_parameters_updated"]
    cp = resume_path(str(tmp_path / "run" / "checkpoint-1"), tmp_path / "run")
    assert Path(cp, "optimizer.pt").exists()
    assert resume_path(True, tmp_path / "run").endswith("checkpoint-2")
    cfg["model"]["adapter_path"] = result["adapter_path"]
    bundle = load_model(cfg)
    assert isinstance(generate_text(bundle, "hello", cfg["generation"]), str)
    cfg["model"].pop("adapter_path")
    resumed = train(cfg, tr, val, resume=cp)
    assert resumed["status"] == "completed" and resumed["resumed_from"] == cp
    altered = [sample("new", "int different() {}")]
    with pytest.raises(ValueError, match="contract differs"):
        train(cfg, altered, val, resume=cp)


def test_invalid_checkpoint(tmp_path):
    with pytest.raises(ValueError, match="Not a resumable"):
        resume_path(str(tmp_path), tmp_path)


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
