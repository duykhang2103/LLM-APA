"""Conventional PEFT adapters; frozen base weights, configurable LoRA."""


def configure_lora(model, config):
    from peft import (
        LoraConfig,
        PeftModel,
        get_peft_model,
        prepare_model_for_kbit_training,
    )

    training = config["training"]
    checkpointing = training.get("gradient_checkpointing", True)
    if config.get("quantization", {}).get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )
    if training.get("warm_start_adapter"):
        return PeftModel.from_pretrained(
            model, training["warm_start_adapter"], is_trainable=True
        )
    settings = config["lora"]
    adapter = LoraConfig(
        r=settings["rank"],
        lora_alpha=settings["alpha"],
        lora_dropout=settings["dropout"],
        target_modules=settings["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, adapter)
