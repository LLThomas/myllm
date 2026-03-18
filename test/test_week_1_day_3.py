import pytest
import torch
import numpy as np
import torch.nn.functional as F

from my_llm import *
from utils import *


def grouped_attention_helper(
    device: str,
    precision: torch.dtype,
    batch_dimension: int,
    scale: float | None,
    is_causal_mask: bool,
):
    H_q = 18
    H = 6
    L = 3
    D = 5
    S = 7
    BATCH = 10
    BATCH_2 = 2

    if precision == torch.float16 and device == "cpu":
        pytest.skip("float16 not supported on CPU")

    for _ in range(100):
        if batch_dimension == 0:
            q_shape = (H_q, L, D)
            kv_shape = (H, S, D)
            mask_shape = (H_q, L, S)
            batch_prefix = ()
        elif batch_dimension == 1:
            q_shape = (BATCH, H_q, L, D)
            kv_shape = (BATCH, H, S, D)
            mask_shape = (BATCH, H_q, L, S)
            batch_prefix = (BATCH,)
        elif batch_dimension == 2:
            q_shape = (BATCH_2, BATCH, H_q, L, D)
            kv_shape = (BATCH_2, BATCH, H, S, D)
            mask_shape = (BATCH_2, BATCH, H_q, L, S)
            batch_prefix = (BATCH_2, BATCH)
        else:
            raise ValueError("invalid batch_dimension")

        q = torch.rand(q_shape, dtype=precision, device=device)
        k = torch.rand(kv_shape, dtype=precision, device=device)
        v = torch.rand(kv_shape, dtype=precision, device=device)
        mask = torch.rand(mask_shape, dtype=precision, device=device)

        n_repeat = H_q // H

        q = q.reshape(*batch_prefix, H, n_repeat, L, D)
        out = torch.empty_like(q)
        for h in range(H):
            q_h = q[..., h, :, :, :]
            k_h = k[..., h : h + 1, :, :]
            v_h = v[..., h : h + 1, :, :]

            q_h = q_h.reshape(-1, n_repeat, L, D)
            k_h = k_h.reshape(-1, 1, S, D)
            v_h = v_h.reshape(-1, 1, S, D)

            if is_causal_mask:
                out_h = F.scaled_dot_product_attention(
                    q_h,
                    k_h,
                    v_h,
                    is_causal=True,
                    scale=scale if scale is not None else (1.0 / D**0.5),
                )
            else:
                mask_h = mask.reshape(*batch_prefix, H, n_repeat, L, S)[..., h, :, :, :]
                mask_h = mask_h.reshape(-1, n_repeat, L, S)
                out_h = F.scaled_dot_product_attention(
                    q_h,
                    k_h,
                    v_h,
                    attn_mask=mask_h,
                    is_causal=False,
                    scale=scale if scale is not None else (1.0 / D**0.5),
                )

            out[..., h, :, :, :] = out_h.reshape(*batch_prefix, n_repeat, L, D)

        reference_out = out.reshape(*batch_prefix, H_q, L, D)

        user_out = scaled_dot_product_attention_grouped(
            q.reshape(*batch_prefix, H_q, L, D),
            k,
            v,
            scale=scale,
            mask=mask if not is_causal_mask else "causal",
        )

        user_out = user_out.reshape(*batch_prefix, H_q, L, D)

        assert_allclose(user_out, reference_out, precision=precision)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8], ids=["no_scale", "scale_0.8"])
def test_task_1_grouped_attention(
    device: str, precision: torch.dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(device, precision, batch_dimension, scale, False)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_2_mask_only_same_dim(device: str):
    if device == "cpu" and not torch.cuda.is_available():
        # nothing special, we run on CPU
        pass

    L = 3
    S = 3
    # causal_mask is expected to return a tensor with 0 and -inf positions
    user_output = causal_mask(L, S, torch.float32)
    expected = torch.tensor(
        [
            [0.0, -float("inf"), -float("inf")],
            [0.0, 0.0, -float("inf")],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert_allclose(user_output, expected, precision=torch.float32)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
def test_task_2_mask_only_different_dim(device: str):
    L = 3
    S = 5
    user_output = causal_mask(L, S, torch.float32)
    expected = torch.tensor(
        [
            [0.0, -float("inf"), -float("inf"), -float("inf"), -float("inf")],
            [0.0, 0.0, -float("inf"), -float("inf"), -float("inf")],
            [0.0, 0.0, 0.0, -float("inf"), -float("inf")],
        ],
        dtype=torch.float32,
    )
    assert_allclose(user_output, expected, precision=torch.float32)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
@pytest.mark.parametrize("scale", [None, 0.8], ids=["no_scale", "scale_0.8"])
def test_task_2_grouped_attention_causal_mask(
    device: str, precision: torch.dtype, batch_dimension: int, scale: float | None
):
    grouped_attention_helper(device, precision, batch_dimension, scale, True)


@pytest.mark.parametrize(
    "device", ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
)
@pytest.mark.parametrize(
    "precision", [torch.float32, torch.float16], ids=["fp32", "fp16"]
)
@pytest.mark.parametrize("mask", [None, "causal"], ids=["no_mask", "causal_mask"])
def test_task_3_qwen2_grouped_query_attention_torch(
    device: str,
    precision: torch.dtype,
    mask: str | None,
):
    if precision == torch.float16 and device == "cpu":
        pytest.skip("float16 not supported on CPU")

    torch.manual_seed(42)

    batch_size = 1  # batch size
    seq_len = 4  # seq len
    hidden_size = 32  # E
    num_heads = 4  # query head
    num_kv_heads = 2  # kv head
    max_seq_len = 64
    theta = 10000

    from transformers.models.qwen2.modeling_qwen2 import Qwen2Attention, Qwen2Config, Qwen2RotaryEmbedding

    config = Qwen2Config(
        hidden_size=hidden_size,
        num_attention_heads=num_heads,
        num_key_value_heads=num_kv_heads,
        intermediate_size=hidden_size * 4,
        num_hidden_layers=2,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        rope_theta=theta,
        max_position_embeddings=max_seq_len,
    )

    torch_attention = Qwen2Attention(config, layer_idx=0).to(device=device, dtype=precision)
    torch_attention.eval()

    wq = torch_attention.q_proj.weight
    wk = torch_attention.k_proj.weight
    wv = torch_attention.v_proj.weight
    wo = torch_attention.o_proj.weight
    bq = torch_attention.q_proj.bias
    bk = torch_attention.k_proj.bias
    bv = torch_attention.v_proj.bias

    x = torch.empty(
        batch_size,
        seq_len,
        hidden_size,
        device=device,
        dtype=precision,
    ).uniform_(-1.0, 1.0)

    user_attention = qwen2_week1.Qwen2MultiHeadAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        wq=wq,
        wk=wk,
        wv=wv,
        wo=wo,
        bq=bq,
        bk=bk,
        bv=bv,
        max_seq_len=max_seq_len,
        theta=theta,
    ).to(device=device, dtype=precision)

    user_attention.eval()

    with torch.no_grad():
        user_output = user_attention(x, mask=mask)

        rotary_emb = Qwen2RotaryEmbedding(config).to(device=device, dtype=precision)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        position_embeddings = rotary_emb(x, position_ids)
        causal_mask_4d = None
        if mask == "causal":
            causal_mask_4d = torch.full((seq_len, seq_len), torch.finfo(precision).min, device=device)
            causal_mask_4d = torch.triu(causal_mask_4d, diagonal=1)
            causal_mask_4d = causal_mask_4d.view(1, 1, seq_len, seq_len)

        outputs = torch_attention(
            hidden_states=x,
            position_embeddings=position_embeddings,
            attention_mask=causal_mask_4d,
            position_ids=position_ids,
            use_cache=False,
        )
        torch_output = outputs[0]

    assert_allclose(user_output, torch_output, precision=precision)