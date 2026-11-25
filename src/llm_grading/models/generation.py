"""Greedy generation without silently truncating student evidence."""

from contextlib import contextmanager, nullcontext

from llm_grading.training.dataset import chat_text


def _expand_gqa_sdpa(original):
    """Adapt GQA to equal-head SDPA kernels without changing attention math."""
    def attention(query, key, value, *args, **kwargs):
        if query.shape[-3] != key.shape[-3]:
            heads, kv_heads = query.shape[-3], key.shape[-3]
            if kv_heads != value.shape[-3] or heads % kv_heads:
                raise ValueError("Incompatible GQA head counts")
            key = key.repeat_interleave(heads // kv_heads, dim=-3)
            value = value.repeat_interleave(heads // kv_heads, dim=-3)
        kwargs.pop("enable_gqa", None)
        return original(query, key, value, *args, **kwargs)
    return attention


def _chunked_sdpa(original, chunk_size):
    """Bound the attention score matrix by chunking query rows, not evidence."""
    if chunk_size < 1:
        raise ValueError("attention_query_chunk_size must be positive")

    def attention(query, key, value, attn_mask=None, dropout_p=0.0,
                  is_causal=False, *, scale=None, enable_gqa=False):
        import torch

        if dropout_p:
            raise ValueError("Chunked attention is inference-only (dropout must be zero)")
        heads, kv_heads = query.shape[-3], key.shape[-3]
        if heads != kv_heads:
            if kv_heads != value.shape[-3] or heads % kv_heads:
                raise ValueError("Incompatible GQA head counts")
            key = key.repeat_interleave(heads // kv_heads, dim=-3)
            value = value.repeat_interleave(heads // kv_heads, dim=-3)
        outputs = []
        for start in range(0, query.shape[-2], chunk_size):
            stop = min(start + chunk_size, query.shape[-2])
            mask = attn_mask
            if mask is not None and mask.ndim >= 2 and mask.shape[-2] > 1:
                mask = mask[..., start:stop, :]
            if is_causal:
                causal = (torch.arange(key.shape[-2], device=query.device)[None, :]
                          <= torch.arange(start, stop, device=query.device)[:, None])
                if mask is None:
                    mask = causal
                elif mask.dtype == torch.bool:
                    mask = mask & causal
                else:
                    mask = mask + torch.zeros_like(causal, dtype=query.dtype).masked_fill(~causal, float("-inf"))
            outputs.append(original(query[..., start:stop, :], key, value,
                                    attn_mask=mask, dropout_p=0.0,
                                    is_causal=False, scale=scale))
        return torch.cat(outputs, dim=-2)
    return attention


@contextmanager
def _equal_head_sdpa(chunk_size=0):
    # Each notebook worker generates synchronously in its own process.
    import torch.nn.functional as functional

    original = functional.scaled_dot_product_attention
    functional.scaled_dot_product_attention = (
        _chunked_sdpa(original, chunk_size) if chunk_size else _expand_gqa_sdpa(original)
    )
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original


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
    attention_context = nullcontext()
    gqa_context = nullcontext()
    device = model.get_input_embeddings().weight.device
    chunk_size = int(settings.get("attention_query_chunk_size", 0))
    if chunk_size < 0:
        raise ValueError("attention_query_chunk_size cannot be negative")
    if device.type == "cuda" and chunk_size:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        attention_context = sdpa_kernel(SDPBackend.MATH)
        gqa_context = _equal_head_sdpa(chunk_size)
    elif (device.type == "cuda"
            and torch.cuda.get_device_capability(device)[0] == 7
            and getattr(model.config, "_attn_implementation", None) == "sdpa"):
        from torch.nn.attention import SDPBackend, sdpa_kernel

        # T4 cannot run FlashAttention-2. Avoid the SDPA math fallback's
        # quadratic attention matrix for long prompts. Fail explicitly if
        # this runtime cannot run memory-efficient attention.
        attention_context = sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION)
        gqa_context = _equal_head_sdpa()
    with torch.inference_mode(), gqa_context, attention_context:
        tokens = model.generate(**inputs, **kwargs)[0, length:]
    bundle["last_usage"] = {"prompt_tokens": length, "completion_tokens": len(tokens)}
    return tokenizer.decode(tokens, skip_special_tokens=True)
