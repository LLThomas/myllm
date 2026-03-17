import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        super().__init__()
        self.dims = dims
        self.traditional = traditional

        # 1. compute θ: 1 / (base ^ (2i / d))
        # (dims/2) -> 1 / (base ^ (2i / d))
        # inv_freq = 1.0 / (base ** (torch.arange(0, dims, 2).float() / dims))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dims, 2, dtype=torch.int64).float() / dims)
        )

        # 2. get pos
        # (seq_len) -> pos
        t = torch.arange(seq_len, dtype=torch.int64)

        # 3. compute outer
        # (seq_len, dims/2) -> pos · θ
        # pos0 -> (θ0, θ1, θ2 ..., θdims/2)
        # pos1 -> ...
        # ...
        # posn-1 -> (θ0, θ1, θ2 ..., θdims/2)
        freqs = torch.outer(t, inv_freq)

        # 4. register Cos and Sin buffer
        # (seq_len, dims/2) -> (seq_len, dim)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False
        )

    def __call__(
        self, x: torch.Tensor, offset: list[slice] | slice | None = None
    ) -> torch.Tensor:
        # (B, H, L, D)
        seq_len = x.shape[-2]

        # precision promote
        orig_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if orig_dtype in [torch.float16, torch.bfloat16]
            else orig_dtype
        )
        x_compute = x.to(compute_dtype)

        # 1. get cached sin/cos val according to offset
        if offset is None:
            cos = self.cos_cached[:seq_len].to(compute_dtype)
            sin = self.sin_cached[:seq_len].to(compute_dtype)
        else:
            cos = self.cos_cached[offset].to(compute_dtype)
            sin = self.sin_cached[offset].to(compute_dtype)

        # 2. adjust cos/sin shape for (Broadcasting)
        # cos/sin: (L, D/2)
        # (1, 1, L, D/2) shape match ->  (B, H, L, D)
        cos = cos.view(1, 1, cos.shape[0], -1)
        sin = sin.view(1, 1, sin.shape[0], -1)

        # 3. fillup
        # (1, 1, L, D)
        # cos: c0, c1, c0, c1
        # sin: s0, s1, s0, s1
        # cos = torch.cat([cos, cos], dim=-1)
        # sin = torch.cat([sin, sin], dim=-1)

        # 4. half rotate
        # first half is x, last half is y
        # x: -x2, -x3, x0, x1
        def rotate_half(x):
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # 5. apply rotrary matrix
        # x' = x·cosθ0 - y·sinθ0
        # y' = x·cosθ0 + y·sinθ0
        #
        # x0 · c0 - x2 * s0
        # ...
        res = (x_compute * cos) + (rotate_half(x_compute) * sin)
        return res.to(orig_dtype)