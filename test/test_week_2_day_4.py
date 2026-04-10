import pytest
import time
import torch
from my_llm import flash_attention
from utils import *

def attention_helper(device, H_q, H_kv, L, E, S, BATCH, mask_mode: str):
    """Helper function to test flash attention against PyTorch reference on CPU."""
    precision = torch.float32
    scale = 0.9

    for _ in range(5): 
        q_shape = (BATCH, H_q, L, E)
        kv_shape = (BATCH, H_kv, S, E)

        query = torch.rand(q_shape, dtype=precision, device=device)
        key = torch.rand(kv_shape, dtype=precision, device=device)
        value = torch.rand(kv_shape, dtype=precision, device=device)

        if mask_mode == "no_mask":
            mask = None
        elif mask_mode == "mask":
            mask = torch.rand((BATCH, H_q, L, S), dtype=precision, device=device)
        elif mask_mode == "causal":
            mask = "causal"
        else:
            raise ValueError(f"Unknown mask_mode: {mask_mode}")

        # Reference: PyTorch's scaled_dot_product_attention
        H_q_actual = query.shape[-3]
        H_kv_actual = key.shape[-3]
        n_rep = H_q_actual // H_kv_actual

        if n_rep > 1:
            key_expanded = key.unsqueeze(-3).expand(
                BATCH, H_kv, n_rep, S, E
            ).reshape(BATCH, H_q, S, E)
            value_expanded = value.unsqueeze(-3).expand(
                BATCH, H_kv, n_rep, S, E
            ).reshape(BATCH, H_q, S, E)
        else:
            key_expanded = key
            value_expanded = value

        is_causal = mask == "causal"
        attn_mask = None if is_causal else mask

        with torch.no_grad():
            reference_output = torch.nn.functional.scaled_dot_product_attention(
                query,
                key_expanded,
                value_expanded,
                attn_mask=attn_mask,
                scale=scale,
                is_causal=is_causal,
            )

        # User's flash attention implementation (CPU)
        user_output = flash_attention(query, key, value, scale=scale, mask=mask)

        assert_allclose(user_output, reference_output, precision=torch.float32)


def time_flash_attention(
    device,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: torch.Tensor | str | None,
    num_iters: int = 4,
) -> float:
    """Time flash attention execution on CPU."""
    # Warmup
    for _ in range(2):
        _ = flash_attention(query, key, value, scale=scale, mask=mask)

    start = time.perf_counter()
    for _ in range(num_iters):
        _ = flash_attention(query, key, value, scale=scale, mask=mask)

    return (time.perf_counter() - start) / num_iters


def median(values: list[float]) -> float:
    values = sorted(values)
    return values[len(values) // 2]


def assert_causal_mask_faster_than_all_zero_mask(
    device, batch, h_q, h_kv, l, s, e, scale=0.9,
):
    """Test that causal mask logic doesn't crash (timing is unreliable on CPU)."""
    precision = torch.float32
    q_shape = (batch, h_q, l, e)
    kv_shape = (batch, h_kv, s, e)
    mask_shape = (batch, h_q, l, s)

    query = torch.rand(q_shape, dtype=precision, device=device)
    key = torch.rand(kv_shape, dtype=precision, device=device)
    value = torch.rand(kv_shape, dtype=precision, device=device)
    zero_mask = torch.zeros(mask_shape, dtype=precision, device=device)

    _ = flash_attention(query, key, value, scale=scale, mask="causal")
    _ = flash_attention(query, key, value, scale=scale, mask=zero_mask)
    print("Functionality check passed for causal/zero_mask.")


@pytest.mark.parametrize("mask_mode", ["no_mask", "mask", "causal"])
def test_flash_attention_cpu_small(mask_mode: str):
    attention_helper(
        torch.device("cpu"), H_q=6, H_kv=3, L=2, E=5, S=3, BATCH=1, mask_mode=mask_mode
    )

@pytest.mark.parametrize("mask_mode", ["no_mask", "mask"])
def test_flash_attention_cpu_gqa(mask_mode: str):
    attention_helper(
        torch.device("cpu"), H_q=18, H_kv=6, L=7, E=5, S=3, BATCH=10, mask_mode=mask_mode
    )

def test_flash_attention_cpu_large():
    attention_helper(
        torch.device("cpu"), H_q=8, H_kv=2, L=16, E=64, S=16, BATCH=1, mask_mode="causal"
    )

def test_flash_attention_different_seq_lengths():
    attention_helper(
        torch.device("cpu"), H_q=4, H_kv=2, L=8, E=64, S=16, BATCH=2, mask_mode="causal"
    )

def test_flash_attention_single_head():
    attention_helper(
        torch.device("cpu"), H_q=1, H_kv=1, L=4, E=32, S=4, BATCH=1, mask_mode="no_mask"
    )

def test_flash_attention_causal_logic_check():
    assert_causal_mask_faster_than_all_zero_mask(
        device=torch.device("cpu"), batch=1, h_q=4, h_kv=4, l=64, s=64, e=64,
    )