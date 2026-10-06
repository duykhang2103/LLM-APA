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
    if sample:
        kwargs["temperature"] = float(settings.get("temperature", 0.7))
        kwargs["top_p"] = float(settings.get("top_p", 1.0))
    with torch.inference_mode():
        tokens = model.generate(**inputs, **kwargs)[0, length:]
    bundle["last_usage"] = {"prompt_tokens": length, "completion_tokens": len(tokens)}
    return tokenizer.decode(tokens, skip_special_tokens=True)
