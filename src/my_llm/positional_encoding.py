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

        # Compute θ = 1 / (base ^ (2i / d))
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dims, 2, dtype=torch.int64).float() / dims)
        )

        # Compute pos · θ for all positions
        t = torch.arange(seq_len, dtype=torch.int64)
        freqs = torch.outer(t, inv_freq)

        # Register cos and sin buffers: (seq_len, dims)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos().to(torch.get_default_dtype()), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin().to(torch.get_default_dtype()), persistent=False
        )

    def __call__(
        self, x: torch.Tensor, offset: list[slice] | slice | int | None = None
    ) -> torch.Tensor:
        """
        Apply Rotary Position Embedding.

        Args:
            x: Input tensor of shape (B, H, L, D)
            offset: Position offset(s)
                - None: positions [0, 1, ..., L-1]
                - int: positions [offset, offset+1, ..., offset+L-1]
                - slice: positions [slice.start, ..., slice.stop-1]
                - list[slice]: batched, each element has different positions

        Returns:
            Tensor with same shape as x, with RoPE applied.
        """
        # x shape: (B, H, L, D) where B is batch size
        B = x.shape[0]
        seq_len = x.shape[-2]
        orig_dtype = x.dtype
        compute_dtype = (
            torch.float32
            if orig_dtype in [torch.float16, torch.bfloat16]
            else orig_dtype
        )
        x_compute = x.to(compute_dtype)

        def rotate_half(x):
            """Rotate half the hidden dims of the input."""
            x1 = x[..., : x.shape[-1] // 2]
            x2 = x[..., x.shape[-1] // 2 :]
            return torch.cat((-x2, x1), dim=-1)

        # Get cos/sin values based on offset type
        if isinstance(offset, list):
            # Batched mode: each batch has different position offsets
            assert len(offset) == B, (
                f"offsets length {len(offset)} must equal batch size {B}"
            )
            for o in offset:
                slice_len = (o.stop or 0) - (o.start or 0)
                assert slice_len == seq_len, (
                    f"slice length {slice_len} must equal seq_len {seq_len}"
                )

            # Generate position indices: (B, L)
            positions = torch.stack([
                torch.arange(s.start, s.stop, device=x.device, dtype=torch.long)
                for s in offset
            ])

            # Index cos/sin_cached: (B, L, D)
            cos = self.cos_cached[positions].to(compute_dtype)
            sin = self.sin_cached[positions].to(compute_dtype)

            # Reshape for broadcasting: (B, 1, L, D) -> (B, H, L, D)
            cos = cos.view(B, 1, seq_len, -1)
            sin = sin.view(B, 1, seq_len, -1)
        else:
            # Single offset mode
            if offset is None:
                start_pos = 0
            elif isinstance(offset, int):
                start_pos = offset
            elif isinstance(offset, slice):
                start_pos = offset.start if offset.start is not None else 0
            else:
                start_pos = 0

            cos = self.cos_cached[start_pos : start_pos + seq_len].to(compute_dtype)
            sin = self.sin_cached[start_pos : start_pos + seq_len].to(compute_dtype)

            # Reshape for broadcasting: (1, 1, L, D) -> (B, H, L, D)
            cos = cos.view(1, 1, cos.shape[0], -1)
            sin = sin.view(1, 1, sin.shape[0], -1)

        # Apply rotary embedding: x' = x·cos - y·sin, y' = x·sin + y·cos
        res = (x_compute * cos) + (rotate_half(x_compute) * sin)
        return res.to(orig_dtype)