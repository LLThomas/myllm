import pytest
import torch
from my_llm import *
from utils import *


def get_device(stream):
    """Get appropriate device based on availability."""
    return torch.device(
        "cuda" if stream == "cuda" and torch.cuda.is_available() else "cpu"
    )


@pytest.mark.parametrize("stream", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_1_softmax(stream, precision):
    """Test softmax implementation with standard parameters."""
    device = get_device(stream)

    batch_size = 10
    dim = 10
    for _ in range(10):
        x = torch.rand((batch_size, dim), dtype=precision, device=device)
        user_output = softmax(x, dim=-1)
        reference_output = torch.softmax(x, dim=-1)
        assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention(stream, precision, batch_dimension):
    """Test simple attention implementation with different batch configurations."""
    device = get_device(stream)

    if batch_dimension == 0:
        batch_size = ()
    elif batch_dimension == 1:
        batch_size = (2, 3)
    elif batch_dimension == 2:
        batch_size = (2, 3, 3)

    dim_l = 4
    dim_d = 5
    for _ in range(10):
        query = torch.rand((*batch_size, dim_l, dim_d), dtype=precision, device=device)
        key = torch.rand((*batch_size, dim_l, dim_d), dtype=precision, device=device)
        value = torch.rand((*batch_size, dim_l, dim_d), dtype=precision, device=device)

        # Reference using torch.nn.functional.scaled_dot_product_attention (PyTorch >= 2.0)
        reference_output = torch.nn.functional.scaled_dot_product_attention(
            query.reshape(1, -1, dim_l, dim_d),
            key.reshape(1, -1, dim_l, dim_d),
            value.reshape(1, -1, dim_l, dim_d),
            scale=1.0 / (dim_d**0.5),
            is_causal=False,
        ).reshape(*batch_size, dim_l, dim_d)

        user_output = scaled_dot_product_attention_simple(query, key, value)
        assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
@pytest.mark.parametrize(
    "batch_dimension", [0, 1, 2], ids=["batch_0", "batch_1", "batch_2"]
)
def test_task_1_simple_attention_scale_mask(stream, precision, batch_dimension):
    """Test attention with scaling and masking for different batch configurations."""
    device = get_device(stream)

    if batch_dimension == 0:
        batch_size = ()
    elif batch_dimension == 1:
        batch_size = (2, 3)
    elif batch_dimension == 2:
        batch_size = (2, 3, 3)

    dim_l = 4
    dim_d = 5
    for _ in range(10):
        query = torch.rand((*batch_size, dim_l, dim_d), dtype=precision, device=device)
        key = torch.rand((*batch_size, dim_l, dim_d), dtype=precision, device=device)
        value = torch.rand((*batch_size, dim_l, dim_d), dtype=precision, device=device)
        mask = torch.randn((*batch_size, dim_l, dim_l), dtype=precision, device=device)
        scale = 0.5

        reference_output = torch.nn.functional.scaled_dot_product_attention(
            query.reshape(1, -1, dim_l, dim_d),
            key.reshape(1, -1, dim_l, dim_d),
            value.reshape(1, -1, dim_l, dim_d),
            attn_mask=mask.reshape(1, -1, dim_l, dim_l),
            scale=scale,
            is_causal=False,
        ).reshape(*batch_size, dim_l, dim_d)

        user_output = scaled_dot_product_attention_simple(
            query, key, value, scale=scale, mask=mask
        )
        assert_allclose(user_output, reference_output, precision)


@pytest.mark.parametrize("stream", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_linear(stream, precision):
    """Test linear layer implementation with different precision types."""
    device = get_device(stream)

    batch_size = 10
    dim_y = 10
    dim_x = 12

    for _ in range(100):
        x = torch.rand((batch_size, dim_x), dtype=precision, device=device)
        w = torch.rand((dim_y, dim_x), dtype=precision, device=device)
        b = torch.rand((dim_y,), dtype=precision, device=device)

        user_output = linear(x, w, b)
        if precision == torch.float16 and device.type == "cpu":
            # unsupported
            break
        reference_output = torch.addmm(b, x, w.T)
        assert_allclose(user_output, reference_output, precision=precision)


@pytest.mark.parametrize("stream", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", PRECISIONS, ids=PRECISION_IDS)
def test_task_2_simple_multi_head_attention(stream, precision):
    """
    Test MultiHeadAttention implementation with standard parameters.
    Assumes Q/K/V are of the same dimensions.
    """
    device = get_device(stream)

    seq_len = 11
    head_dim = 9
    num_heads = 3
    batch_size = 10

    for _ in range(100):
        query = torch.rand(
            (batch_size, seq_len, num_heads * head_dim), dtype=precision, device=device
        )
        key = torch.rand(
            (batch_size, seq_len, num_heads * head_dim), dtype=precision, device=device
        )
        value = torch.rand(
            (batch_size, seq_len, num_heads * head_dim), dtype=precision, device=device
        )

        q_proj_weight = torch.rand(
            (num_heads * head_dim, num_heads * head_dim), dtype=precision, device=device
        )
        k_proj_weight = torch.rand(
            (num_heads * head_dim, num_heads * head_dim), dtype=precision, device=device
        )
        v_proj_weight = torch.rand(
            (num_heads * head_dim, num_heads * head_dim), dtype=precision, device=device
        )
        out_proj_weight = torch.rand(
            (num_heads * head_dim, num_heads * head_dim), dtype=precision, device=device
        )
        mask = torch.rand((seq_len, seq_len), dtype=precision, device=device)

        # PyTorch MultiheadAttention
        reference_mha = torch.nn.MultiheadAttention(
            embed_dim=num_heads * head_dim, num_heads=num_heads
        )
        reference_mha = reference_mha.to(dtype=precision, device=device)
        reference_mha.in_proj_weight.data.copy_(
            torch.cat([q_proj_weight, k_proj_weight, v_proj_weight], dim=0)
        )
        reference_mha.out_proj.weight.data.copy_(out_proj_weight)

        reference_output, _ = reference_mha(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            attn_mask=mask,
        )
        reference_output = reference_output.transpose(0, 1)

        user_output = SimpleMultiHeadAttention(
            num_heads * head_dim,
            num_heads,
            q_proj_weight,
            k_proj_weight,
            v_proj_weight,
            out_proj_weight,
        )(query, key, value, mask=mask)

        assert_allclose(user_output, reference_output, precision=precision)