"""One local Hugging Face loader, shared by prompting and adapter training."""

import json
from pathlib import Path

from llm_grading.prompting.formatter import prompt_hash


def load_model(config, *, training=False):
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
    )
    from transformers.utils import CONFIG_NAME, cached_file, extract_commit_hash

    settings = config["model"]
    name = settings["name_or_path"]
    revision = settings.get("revision") or "main"
    quant = config.get("quantization", {})
    use_4bit = quant.get("load_in_4bit", False)
    if use_4bit and not torch.cuda.is_available():
        raise RuntimeError(
            "4-bit QLoRA requires a CUDA GPU and bitsandbytes. Use a CUDA notebook, or disable quantization only for a small CPU smoke model."
        )
    adapter = settings.get("adapter_path") if not training else None
    adapter_meta = None
    if adapter:
        meta_path = Path(adapter) / "adapter_metadata.json"
        if not meta_path.exists():
            raise ValueError(f"Adapter metadata missing: {meta_path}")
        adapter_meta = json.loads(meta_path.read_text())
        if settings.get("revision") in {None, "main"}:
            revision = adapter_meta["model_revision"]
    if not Path(name).is_dir():
        # New Transformers configs need not retain _commit_hash. Resolve the
        # cached snapshot first, then pin config, tokenizer and weights together.
        config_file = cached_file(name, CONFIG_NAME, revision=revision)
        resolved_revision = extract_commit_hash(config_file, None)
        if resolved_revision is None:
            raise ValueError(f"Cannot resolve an immutable model revision for {name}")
        revision = resolved_revision
    base_config = AutoConfig.from_pretrained(
        name, revision=revision, trust_remote_code=False
    )
    tokenizer = AutoTokenizer.from_pretrained(
        name, revision=revision, trust_remote_code=False
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer has no pad or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    dtype_name = settings.get(
        "dtype", "bfloat16" if torch.cuda.is_available() else "float32"
    )
    if dtype_name not in {"float32", "float16", "bfloat16"}:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    dtype = getattr(torch, dtype_name)
    if (
        dtype == torch.bfloat16
        and torch.cuda.is_available()
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError(
            "GPU does not support bfloat16; explicitly configure float16"
        )
    kwargs = dict(
        revision=revision,
        dtype=dtype,
        trust_remote_code=False,
        attn_implementation=settings.get("attn_implementation", "eager"),
    )
    if use_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant.get("quant_type", "nf4"),
            bnb_4bit_use_double_quant=quant.get("double_quant", True),
            bnb_4bit_compute_dtype=dtype,
        )
        kwargs["device_map"] = {"": torch.cuda.current_device()} if training else "auto"
    elif not training:
        kwargs["device_map"] = settings.get("device_map", "auto")
    # Load only Qwen's text backbone; the library maps full checkpoint prefixes.
    if base_config.model_type == "qwen3_5":
        from transformers import Qwen3_5ForCausalLM

        kwargs["config"] = base_config.text_config
        model = Qwen3_5ForCausalLM.from_pretrained(name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    if training:
        if not use_4bit:
            model.to("cuda" if torch.cuda.is_available() else "cpu")
        return {
            "model": model,
            "tokenizer": tokenizer,
            "revision": revision,
            "name": name,
        }
    if adapter:
        from peft import PeftModel

        meta = adapter_meta
        if (
            meta["model_name"] != name
            or meta["model_revision"] != revision
            or meta["task"] != config["task"]
        ):
            raise ValueError("Adapter base revision/model/task does not match config")
        if meta["prompt_version"] != config.get("prompt", {}).get("version", "v001"):
            raise ValueError("Adapter prompt version does not match config")
        if meta["prompt_sha256"] != prompt_hash(config["task"], config):
            raise ValueError(
                "Adapter prompt content has changed under the same version"
            )
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return {"model": model, "tokenizer": tokenizer, "revision": revision, "name": name}
