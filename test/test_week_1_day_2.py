import torch
import numpy as np
from my_llm import *
from utils import *


def test_task_1_rope_torch_traditional():
    """
    Test traditional RoPE implementation (B, H, L, D)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.float32

    # Test parameters
    batch_size = 1
    num_heads = 8
    head_dim = 4
    max_seq_len = 20
    seq_len = 10
    base = 10000

    test_cases = [
        (False, "without offset"),
        (True, "with offset"),
    ]

    for with_offset, desc in test_cases:
        for _ in range(100):
            user_layer = RoPE(head_dim, max_seq_len, base, traditional=True).to(device)

            x = torch.rand(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=precision,
                device=device,
            )

            if with_offset:
                input_pos = np.random.randint(0, max_seq_len - seq_len)
                input_pos_user = slice(input_pos, input_pos + seq_len)
                offset = input_pos
            else:
                input_pos_user = None
                offset = 0

            pos = torch.arange(seq_len, device=device) + offset
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
            )
            freqs = torch.outer(pos, inv_freq)

            cos = torch.cos(freqs).view(1, 1, seq_len, head_dim // 2)
            sin = torch.sin(freqs).view(1, 1, seq_len, head_dim // 2)

            mid = head_dim // 2
            x_half1 = x[..., :mid]
            x_half2 = x[..., mid:]

            ref_half1 = x_half1 * cos - x_half2 * sin
            ref_half2 = x_half2 * cos + x_half1 * sin
            ref = torch.cat([ref_half1, ref_half2], dim=-1)

            user_output = user_layer(x, input_pos_user)

            atol = 5e-6 if precision == torch.float32 else 1e-3
            assert_allclose(
                user_output.to(precision),
                ref.to(precision),
                precision,
                atol=atol,
            )


def test_task_2_rope_torch_non_traditional():
    """
    Test non-traditional RoPE implementation (B, H, L, D)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    precision = torch.float32

    batch_size = 1
    num_heads = 8
    head_dim = 4
    max_seq_len = 20
    seq_len = 10
    base = 10000

    test_cases = [
        (False, "without offset"),
        (True, "with offset"),
    ]

    for with_offset, desc in test_cases:
        for _ in range(100):
            user_layer = RoPE(head_dim, max_seq_len, base, traditional=False).to(device)

            x = torch.rand(
                (batch_size, num_heads, seq_len, head_dim),
                dtype=precision,
                device=device,
            )

            if with_offset:
                input_pos = np.random.randint(0, max_seq_len - seq_len)
                input_pos_user = slice(input_pos, input_pos + seq_len)
                offset = input_pos
            else:
                input_pos_user = None
                offset = 0

            pos = torch.arange(seq_len, device=device) + offset
            inv_freq = 1.0 / (
                base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
            )
            freqs = torch.outer(pos, inv_freq)

            mid = head_dim // 2
            cos = torch.cos(freqs).view(1, 1, seq_len, mid)
            sin = torch.sin(freqs).view(1, 1, seq_len, mid)

            x_half1 = x[..., :mid]
            x_half2 = x[..., mid:]

            ref_half1 = x_half1 * cos - x_half2 * sin
            ref_half2 = x_half2 * cos + x_half1 * sin

            ref = torch.cat([ref_half1, ref_half2], dim=-1)

            user_output = user_layer(x, input_pos_user)

            atol = 5e-6 if precision == torch.float32 else 1e-3
            assert_allclose(
                user_output.to(precision),
                ref.to(precision),
                precision,
                atol=atol,
            )