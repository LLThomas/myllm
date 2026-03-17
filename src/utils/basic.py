import torch
import torch.nn.functional as F


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return F.softmax(x, dim=dim)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is not None:
        return torch.matmul(x, w.T) + bias
    else:
        return torch.matmul(x, w.T)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x / (1 + torch.exp(-x))