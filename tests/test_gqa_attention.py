import pytest
import torch
import torch.nn.functional as functional

from llm_grading.models.generation import (
    _chunked_sdpa,
    _equal_head_sdpa,
    _expand_gqa_sdpa,
)


@pytest.mark.parametrize("query_length,key_length,causal", [(9, 9, True), (1, 9, False)])
def test_gqa_expansion_matches_reference(query_length, key_length, causal):
    torch.manual_seed(42)
    query = torch.randn(1, 8, query_length, 16)
    key = torch.randn(1, 2, key_length, 16)
    value = torch.randn(1, 2, key_length, 16)
    expected = functional.scaled_dot_product_attention(
        query, key, value, is_causal=causal, enable_gqa=True
    )
    actual = _expand_gqa_sdpa(functional.scaled_dot_product_attention)(
        query, key, value, is_causal=causal, enable_gqa=True
    )
    torch.testing.assert_close(actual, expected)


def test_attention_restored_on_exception():
    original = functional.scaled_dot_product_attention
    with pytest.raises(RuntimeError):
        with _equal_head_sdpa():
            assert functional.scaled_dot_product_attention is not original
            raise RuntimeError("generation failed")
    assert functional.scaled_dot_product_attention is original


@pytest.mark.parametrize("causal,mask_kind,length", [
    (True, None, 9), (False, None, 1), (False, "bool", 9),
    (False, "additive", 9), (True, "bool", 9), (True, "additive", 9),
])
def test_chunked_attention_matches_full_and_bounds_query(causal, mask_kind, length):
    query = torch.randn(1, 8, length, 16)
    key = torch.randn(1, 2, 9, 16)
    value = torch.randn_like(key)
    mask = None
    if mask_kind:
        mask = torch.ones(length, 9, dtype=torch.bool)
        mask[:, -1] = False
        if mask_kind == "additive":
            mask = torch.zeros(length, 9).masked_fill(~mask, float("-inf"))
    reference = _expand_gqa_sdpa(functional.scaled_dot_product_attention)
    if causal and mask is not None:
        causal_mask = torch.arange(9)[None, :] <= torch.arange(length)[:, None]
        combined = (mask & causal_mask if mask.dtype == torch.bool
                    else mask + torch.zeros(length, 9).masked_fill(~causal_mask, float("-inf")))
        expected = reference(query, key, value, attn_mask=combined)
    else:
        expected = reference(query, key, value, attn_mask=mask, is_causal=causal)
    seen = []

    def checked(q, k, v, **kwargs):
        seen.append(q.shape[-2])
        return functional.scaled_dot_product_attention(q, k, v, **kwargs)

    actual = _chunked_sdpa(checked, 3)(query, key, value, attn_mask=mask, is_causal=causal)
    assert max(seen) <= 3
    torch.testing.assert_close(actual, expected)
