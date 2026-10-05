"""One Transformers Trainer path for local/cloud QLoRA and checkpoint resume."""

import hashlib
import json
import math
import time
from pathlib import Path

from llm_grading.data.split import check_split_overlap
from llm_grading.models.loader import load_model
from llm_grading.prompting.formatter import prompt_hash
from llm_grading.runtime import write_json
from llm_grading.training.callbacks import make_callbacks
from llm_grading.training.dataset import (
    CompletionCollator,
    build_training_dataset,
    tokenize_record,
)
from llm_grading.training.lora import configure_lora
from llm_grading.utils.logging import run_metadata, save_run_config
from llm_grading.utils.seed import set_seed


def resume_path(value, output_dir):
    if not value:
        return None
    if value is True:
        checkpoints = list(Path(output_dir).glob("checkpoint-*"))
        checkpoints = [p for p in checkpoints if (p / "trainer_state.json").exists()]
        if not checkpoints:
            raise ValueError("No Trainer checkpoint available to resume")
        value = max(checkpoints, key=lambda p: int(p.name.rsplit("-", 1)[1]))
    path = Path(value)
    if (
        not (path / "trainer_state.json").is_file()
        or not (path / "adapter_config.json").is_file()
        or not (path / "optimizer.pt").is_file()
        or not (path / "scheduler.pt").is_file()
    ):
        raise ValueError(
            f"Not a resumable Trainer adapter checkpoint: {path}; final adapter-only folders cannot restore optimizer state"
        )
    return str(path.resolve())


def train(
    config,
    train_samples,
    validation_samples,
    *,
    resume=None,
    smoke=False,
    dry_run=False,
):
    import torch
    from transformers import Trainer, TrainingArguments

    check_split_overlap(train_samples, validation_samples)
    if not train_samples or not validation_samples:
        raise ValueError(
            "Training requires nonempty explicit train and validation splits"
        )
    set_seed(config.get("seed", 42))
    directory = Path(config["saving"]["output_dir"])
    directory.mkdir(parents=True, exist_ok=True)
    if config["model"].get("provider", "local") != "local":
        raise ValueError("Fine-tuning requires a local model")
    checkpoint = resume_path(
        resume or config["training"].get("resume_from_checkpoint"), directory
    )
    if checkpoint and config["training"].get("warm_start_adapter"):
        raise ValueError("Choose resume or warm_start_adapter, not both")
    if (
        not checkpoint
        and not dry_run
        and (any(directory.glob("checkpoint-*")) or (directory / "adapter").exists())
    ):
        raise ValueError(
            "Checkpoints already exist; choose --resume-from-checkpoint or a new version/output directory"
        )
    rows = build_training_dataset(train_samples, config["task"], config)
    val_rows = build_training_dataset(validation_samples, config["task"], config)
    bundle = load_model(config, training=True)
    tokenizer = bundle["tokenizer"]
    max_length = int(config["training"]["max_sequence_length"])
    encoded = [tokenize_record(row, tokenizer, max_length) for row in rows]
    val_encoded = [tokenize_record(row, tokenizer, max_length) for row in val_rows]
    identity = dict(
        model_name=bundle["name"],
        model_revision=bundle["revision"],
        task=config["task"],
        prompt_version=config.get("prompt", {}).get("version", "v001"),
        prompt_sha256=prompt_hash(config["task"], config),
    )
    contract = {
        **identity,
        "lora": config["lora"],
        "quantization": config.get("quantization", {}),
        "seed": config.get("seed", 42),
        "max_sequence_length": max_length,
        "optimizer_settings": {
            key: config["training"].get(key)
            for key in [
                "learning_rate",
                "per_device_train_batch_size",
                "gradient_accumulation_steps",
                "warmup_ratio",
                "warmup_steps",
                "lr_scheduler_type",
                "optim",
            ]
        },
        "records_sha256": hashlib.sha256(
            json.dumps(rows + val_rows, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest(),
    }
    if checkpoint:
        saved = Path(checkpoint) / "training_contract.json"
        if not saved.exists() or json.loads(saved.read_text()) != contract:
            raise ValueError(
                "Resume data/model/template/LoRA contract differs; use a fresh version (or warm-start weights) instead"
            )
    metadata = {
        **run_metadata(config),
        **identity,
        "status": "running",
        "train_count": len(rows),
        "validation_count": len(val_rows),
        "max_tokens_observed": max(len(r["input_ids"]) for r in encoded + val_encoded),
    }
    save_run_config(config)
    if dry_run:
        metadata["status"] = "dry_run"
        write_json(directory / "run_manifest.json", metadata)
        return metadata
    warm_start = config["training"].get("warm_start_adapter")
    if warm_start:
        old_identity = json.loads(
            (Path(warm_start) / "adapter_metadata.json").read_text()
        )
        old_contract = json.loads(
            (Path(warm_start) / "training_contract.json").read_text()
        )
        if (
            any(
                old_identity[k] != identity[k]
                for k in ["model_name", "model_revision", "task"]
            )
            or old_contract["lora"] != config["lora"]
        ):
            raise ValueError(
                "Warm-start adapter has incompatible base/task/LoRA settings"
            )
    model = configure_lora(bundle["model"], config)
    model.config.use_cache = False
    trainable = (
        {
            name: p.detach().cpu().clone()
            for name, p in model.named_parameters()
            if p.requires_grad
        }
        if smoke
        else None
    )
    if not any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("No trainable LoRA parameters")
    if any(
        p.requires_grad and "lora_" not in name for name, p in model.named_parameters()
    ):
        raise RuntimeError("Unexpected trainable base parameters")
    training = config["training"]
    saving = config["saving"]
    evaluation = config["evaluation"]
    strategy = evaluation.get(
        "strategy", evaluation.get("evaluation_strategy", "epoch")
    )
    save_strategy = saving.get("save_strategy", "epoch")
    steps_per_epoch = math.ceil(
        math.ceil(len(encoded) / training["per_device_train_batch_size"])
        / training["gradient_accumulation_steps"]
    )
    total_steps = (
        2
        if smoke
        else (
            training.get("max_steps", -1)
            if training.get("max_steps", -1) > 0
            else math.ceil(steps_per_epoch * training["num_train_epochs"])
        )
    )
    warmup_steps = training.get(
        "warmup_steps", math.ceil(total_steps * training.get("warmup_ratio", 0))
    )
    kwargs = dict(
        output_dir=str(directory),
        num_train_epochs=training["num_train_epochs"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training.get("per_device_eval_batch_size", 1),
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        warmup_steps=warmup_steps,
        lr_scheduler_type=training.get("lr_scheduler_type", "linear"),
        max_steps=2 if smoke else training.get("max_steps", -1),
        seed=config.get("seed", 42),
        data_seed=config.get("seed", 42),
        gradient_checkpointing=training.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_nan_inf_filter=False,
        logging_steps=1 if smoke else config["logging"].get("logging_steps", 10),
        report_to=[],
        eval_strategy="steps" if smoke else strategy,
        eval_steps=1 if smoke else evaluation.get("eval_steps", 50),
        save_strategy="steps" if smoke else save_strategy,
        save_steps=1 if smoke else saving.get("save_steps", 50),
        save_total_limit=saving.get("save_total_limit", 2),
        save_only_model=False,
        bf16=torch.cuda.is_available() and config["model"].get("dtype") == "bfloat16",
        fp16=torch.cuda.is_available() and config["model"].get("dtype") == "float16",
        remove_unused_columns=False,
        dataloader_pin_memory=torch.cuda.is_available(),
        optim=training.get("optim", "adamw_torch"),
        load_best_model_at_end=False,
    )
    args = TrainingArguments(**kwargs)
    if smoke:
        encoded = encoded[: min(2, len(encoded))]
        val_encoded = val_encoded[:1]
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=encoded,
        eval_dataset=val_encoded,
        data_collator=CompletionCollator(tokenizer),
        processing_class=tokenizer,
        callbacks=make_callbacks(directory, identity, contract, tokenizer),
    )
    start = time.perf_counter()
    metadata["trainable_parameters"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    metadata["total_parameters"] = sum(p.numel() for p in model.parameters())
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        metadata["gpu"] = torch.cuda.get_device_name()
        metadata["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
    else:
        metadata["gpu"] = None
    write_json(directory / "run_manifest.json", metadata)
    try:
        result = trainer.train(resume_from_checkpoint=checkpoint)
        if not math.isfinite(result.training_loss):
            raise RuntimeError("Training loss is not finite")
        if smoke:
            updated = any(
                not torch.equal(
                    before, dict(model.named_parameters())[name].detach().cpu()
                )
                for name, before in trainable.items()
            )
            if not updated:
                raise RuntimeError("Smoke training did not update LoRA parameters")
            metadata["lora_parameters_updated"] = True
        metrics = {**result.metrics, **trainer.evaluate()}
        adapter = directory / "adapter"
        trainer.save_model(str(adapter))
        tokenizer.save_pretrained(adapter)
        trainer.save_state()
        write_json(adapter / "adapter_metadata.json", identity)
        write_json(adapter / "training_contract.json", contract)
        metadata.update(
            status="completed",
            adapter_path=str(adapter),
            checkpoint_paths=[str(p) for p in sorted(directory.glob("checkpoint-*"))],
            elapsed_seconds=time.perf_counter() - start,
            model_revision=bundle["revision"],
            training_metrics=metrics,
            resumed_from=checkpoint,
        )
        if torch.cuda.is_available():
            metadata["peak_vram_bytes"] = torch.cuda.max_memory_allocated()
        write_json(directory / "training_metrics.json", metrics)
    except Exception as exc:
        metadata.update(
            status="failed", error=str(exc), elapsed_seconds=time.perf_counter() - start
        )
        raise
    finally:
        write_json(directory / "run_manifest.json", metadata)
    return metadata
