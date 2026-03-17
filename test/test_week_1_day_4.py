import pytest
import torch
import torch.nn.functional as F

from my_llm import *
from utils import *


def test_task_1_rms_norm():
    """
    Test RMSNorm implementation
    """
    # Test different devices and precisions
    test_devices = AVAILABLE_DEVICES
    test_precisions = PRECISIONS

    for device in test_devices:
        for precision in test_precisions:
            # Test different sizes
            test_sizes = [
                (100, 111),  # Standard size
                (1, 1),  # Minimal size
                (1000, 1000),  # Large size
            ]

            for size, size_y in test_sizes:
                # Generate test data
                data = torch.rand(size, size_y, device=device, dtype=precision)
                weight = torch.rand(size_y, device=device, dtype=precision)
                eps = torch.finfo(precision).eps

                # Reference implementation
                var = data.pow(2).mean(dim=-1, keepdim=True)
                reference_output = data * torch.rsqrt(var + eps) * weight

                # Our implementation
                user_output = RMSNorm(size_y, weight, eps=eps)(data)

                # Compare results
                assert_allclose(user_output, reference_output, precision)


def test_task_1_rms_norm_cast_to_float32():
    """
    Test RMSNorm with float32 casting
    """
    device = "cpu"  # Use CPU for consistency
    precision = torch.float16
    size, size_y = 32, 64

    # Generate test data
    data = torch.empty(size, size_y, device=device, dtype=precision).uniform_(
        -1000, 1000
    )
    weight = torch.empty(size_y, device=device, dtype=precision).uniform_(-1000, 1000)
    eps = torch.finfo(precision).eps

    # Reference implementation
    data_f32 = data.float()
    weight_f32 = weight.float()
    var = data_f32.pow(2).mean(dim=-1, keepdim=True)
    normed_f32 = data_f32 * torch.rsqrt(var + eps)
    output_f32 = normed_f32 * weight_f32
    reference_output = output_f32.to(precision)

    # Our implementation
    user_output = RMSNorm(size_y, weight, eps=eps)(data)

    # Compare results
    assert_allclose(user_output, reference_output, precision)


def test_task_2_silu():
    """
    Test SiLU activation function
    """
    # Test different devices and precisions
    test_devices = AVAILABLE_DEVICES
    test_precisions = PRECISIONS

    for device in test_devices:
        for precision in test_precisions:
            # Skip float16 on CPU as noted in original test
            if device == "cpu" and precision == torch.float16:
                pytest.skip("float16 not supported on CPU")

            # Test different sizes
            test_sizes = [
                (10, 10),  # Standard size
                (1, 1),  # Minimal size
                (100, 100),  # Large size
            ]

            for batch_size, dim in test_sizes:
                # Generate test data
                x = torch.rand(batch_size, dim, device=device, dtype=precision)

                # Reference implementation
                reference_output = F.silu(x)

                # Our implementation
                user_output = silu(x)

                # Compare results
                assert_allclose(user_output, reference_output, precision)


def test_task_2_qwen_mlp():
    """
    Test Qwen MLP implementation
    """
    # Define test parameters
    dim_params = [
        {"batch_size": 1, "seq_len": 5, "dim": 4, "hidden_dim": 8, "id": "small_dims"},
        {
            "batch_size": 2,
            "seq_len": 16,
            "dim": 32,
            "hidden_dim": 64,
            "id": "large_dims",
        },
        {
            "batch_size": 1,
            "seq_len": 1,
            "dim": 128,
            "hidden_dim": 256,
            "id": "single_token",
        },
    ]

    # Test different devices and precisions
    test_devices = AVAILABLE_DEVICES
    test_precisions = PRECISIONS

    for device in test_devices:
        for precision in test_precisions:
            # Skip float16 on CPU as noted in original test
            if device == "cpu" and precision == torch.float16:
                pytest.skip("float16 not supported on CPU")

            for dims in dim_params:
                # Extract parameters
                batch_size, seq_len, dim, hidden_dim = (
                    dims["batch_size"],
                    dims["seq_len"],
                    dims["dim"],
                    dims["hidden_dim"],
                )

                # Generate test data
                x = torch.rand(batch_size, seq_len, dim, dtype=precision, device=device)
                w_gate = torch.rand(hidden_dim, dim, dtype=precision, device=device)
                w_up = torch.rand(hidden_dim, dim, dtype=precision, device=device)
                w_down = torch.rand(dim, hidden_dim, dtype=precision, device=device)

                # Our implementation
                user_mlp = qwen2_week1.Qwen2MLP(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w_gate=w_gate,
                    w_up=w_up,
                    w_down=w_down,
                )
                user_output = user_mlp(x)

                # Reference implementation
                from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP

                config = type(
                    "Config",
                    (),
                    {
                        "hidden_size": dim,
                        "intermediate_size": hidden_dim,
                        "hidden_act": "silu",
                    },
                )()
                reference_mlp = Qwen2MLP(config).to(device=device, dtype=precision)
                reference_mlp.gate_proj.weight.data = w_gate
                reference_mlp.up_proj.weight.data = w_up
                reference_mlp.down_proj.weight.data = w_down
                reference_output = reference_mlp(x)

                # Compare results
                assert_allclose(user_output, reference_output, precision)