"""Greedy generation without silently truncating student evidence."""

from llm_grading.training.dataset import chat_text


def generate_text(bundle, prompt, settings):
    import torch

    tokenizer, model = bundle["tokenizer"], bundle["model"]
    inputs = tokenizer(
        chat_text(tokenizer, prompt), return_tensors="pt", add_special_tokens=False
    )
    length = inputs["input_ids"].shape[1]
    budget = int(settings.get("max_input_tokens", 4096))
    if length > budget:
        raise ValueError(
            f"Prompt has {length} tokens, exceeds max_input_tokens={budget}; increase budget or reduce retrieval k"
        )
    inputs = {
        key: value.to(model.get_input_embeddings().weight.device)
        for key, value in inputs.items()
    }
    sample = bool(settings.get("do_sample", False))
    kwargs = {
        "max_new_tokens": int(settings.get("max_new_tokens", 256)),
        "do_sample": sample,
        "num_beams": int(settings.get("num_beams", 1)),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if not sample:
        # Model repositories may ship sampling defaults even for greedy runs.
        from copy import deepcopy

        generation_config = deepcopy(model.generation_config)
        generation_config.do_sample = False
        generation_config.temperature = 1.0
        generation_config.top_p = 1.0
        generation_config.top_k = 50
        kwargs["generation_config"] = generation_config
    if sample:
        kwargs["temperature"] = float(settings.get("temperature", 0.7))
        kwargs["top_p"] = float(settings.get("top_p", 1.0))
    try:
        with torch.inference_mode():
            tokens = model.generate(**inputs, **kwargs)[0, length:]
    except torch.OutOfMemoryError as error:
        # A long prompt (T4, ~7k tokens) can exhaust VRAM; free it and report it
        # like an oversized prompt so the pipeline can shorten it or fall back.
        del inputs
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        raise ValueError(f"CUDA out of memory with a {length}-token prompt") from error
    bundle["last_usage"] = {"prompt_tokens": length, "completion_tokens": len(tokens)}
    return tokenizer.decode(tokens, skip_special_tokens=True)
