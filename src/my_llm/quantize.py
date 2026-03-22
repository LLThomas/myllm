import torch
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torchao")
from torchao.quantization import dequantize_affine
from typing import Any


def dequantize_linear(layer: Any) -> torch.Tensor:
    scales = getattr(layer, "scales", None)
    if scales is None:
        return layer.weight

    w = layer.weight
    zp = getattr(layer, "biases", None)
    group_size = getattr(layer, "group_size", 0)
    bits = getattr(layer, "bits", 8)

    if w.dim() == 2:
        actual_group_size = group_size if (group_size > 0 and group_size < w.shape[1]) else w.shape[1]
        block_size = (1, actual_group_size)
    else:
        block_size = (group_size,) if group_size > 0 else w.shape

    quant_min = -(2 ** (bits - 1))
    quant_max = 2 ** (bits - 1) - 1

    return dequantize_affine(
        input=w,
        block_size=block_size,
        scale=scales,
        zero_point=zp,
        input_dtype=w.dtype,
        quant_min=quant_min,
        quant_max=quant_max,
        output_dtype=layer.weight.dtype,
    )


class QuantizedWeights:
    def __init__(
        self,
        scales: torch.Tensor | None,
        biases: torch.Tensor | None,
        group_size: int | None,
        bits: int | None,
        weight: torch.Tensor,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size if group_size is not None else 0
        self.bits = bits if bits is not None else 0
        self.weight = weight

    @staticmethod
    def from_weight(layer: Any) -> "QuantizedWeights":
        return QuantizedWeights(
            scales=getattr(layer, "scales", None),
            biases=getattr(layer, "biases", None),
            group_size=getattr(layer, "group_size", None),
            bits=getattr(layer, "bits", None),
            weight=layer.weight,
        )


def quantized_matmul(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    pass


def quantized_linear(
    x: torch.Tensor,
    w: QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if bias is None:
        return torch.matmul(x, w.weight.T)
    return torch.matmul(x, w.weight.T) + bias