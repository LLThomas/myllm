import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.dim
        ), f"Input dim {x.shape[-1]} != expected {self.dim}"

        orig_dtype = x.dtype

        # promote precision
        if orig_dtype in [torch.float16, torch.bfloat16]:
            compute_dtype = torch.float32
        else:
            compute_dtype = orig_dtype
        x_compute = x.to(compute_dtype)

        # y = x · weight / sqrt(mean(x^2) + eps)
        variance = x_compute.pow(2).mean(-1, keepdim=True)
        x_normed = x_compute * torch.rsqrt(variance + self.eps)

        weight_compute = self.weight.to(compute_dtype)
        return (weight_compute * x_normed).to(orig_dtype)