import math
import torch


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    # 0. get shape
    # L, D from query
    L = query.size(-2)
    D = query.size(-1)
    S = key.size(-2)

    # 1. Q · Kt
    # Q: (B, H, L, D), K: (B, H, S, D) -> scores: (B, H, L, S)
    scores = torch.matmul(query, key.transpose(-2, -1))

    # 2. Q · Kt / sqrt(d_k)
    # scale by sqrt(d_k)
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    scores = scores * scale

    # 3. Q · Kt / sqrt(d_k) + mask
    # apply mask
    if mask is not None:
        if isinstance(mask, str):
            assert mask == "causal"
            # causal mask shape: (L, S)
            causal = causal_mask(L, S, query.dtype, device=query.device)
            scores = scores + causal
        else:
            scores = scores + mask

    # 4. softmax(Q · Kt / sqrt(d_k) + mask)
    # softmax over last dimension
    attn_probs = torch.softmax(scores.to(torch.float32), dim=-1).to(query.dtype)

    # 5. softmax(Q · Kt / sqrt(d_k) + mask) · V
    # weighted sum
    # attn_probs: (B, H, L, S) · value: (B, H, S, D) -> res: (B, H, L, D)
    return torch.matmul(attn_probs, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # get shape
        B, L, _ = query.shape

        # 1. get Q, K, V
        # (BATCH_SIZE, L, H * D) · (H * D, H * D) -> (BATCH_SIZE, L, H * D)
        Q = query @ self.wq.T
        K = key @ self.wk.T
        V = value @ self.wv.T

        # 2. split to multi-head
        # (BATCH_SIZE, L, H * D) -> (BATCH_SIZE, H, L, D)
        Q = Q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. scaled dot-product attention
        # ... -> (BATCH_SIZE, H, L, D)
        attention_res = scaled_dot_product_attention_simple(Q, K, V, None, mask)

        # 4. concat multi-head res
        # (BATCH_SIZE, H, L, D) -> (BATCH_SIZE, L, H * D)
        attention_res = (
            attention_res.transpose(1, 2).contiguous().view(B, L, self.hidden_size)
        )

        # 5. linear layer output
        # (BATCH_SIZE, L, H * D) · (H * D, H * D) -> (BATCH_SIZE, L, H * D)
        return attention_res @ self.wo.T


def causal_mask(
    L: int,
    S: int,
    dtype: torch.dtype,
    device: torch.device = None,
) -> torch.Tensor:
    mask = torch.triu(torch.ones(L, S, device=device), diagonal=1)
    res = torch.zeros((L, S), dtype=dtype, device=device)
    res = res.masked_fill(mask == 1, float("-inf"))
    return res


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,  # (..., H_q, L, D)
    key: torch.Tensor,  # (..., H_kv, S, D)
    value: torch.Tensor,  # (..., H_kv, S, D)
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,  # (..., H_q, L, S)
) -> torch.Tensor:
    # shapes
    H_q = query.shape[-3]
    H_kv, _, _ = key.shape[-3:]
    if H_q % H_kv != 0:
        raise ValueError(f"H_q ({H_q}) must be a multiple of H_kv ({H_kv})")
    n_rep = H_q // H_kv

    if n_rep > 1:
        batch_prefix = key.shape[:-3]
        kv_head, seq_len, head_dim = key.shape[-3:]

        # key: (..., H_kv, S, D) -> (..., H_kv, 1, S, D) -> expand(..., H_kv, n_rep, S, D)
        key = key.unsqueeze(-3).expand(*batch_prefix, kv_head, n_rep, seq_len, head_dim)
        key = key.reshape(*batch_prefix, kv_head * n_rep, seq_len, head_dim)

        value = value.unsqueeze(-3).expand(
            *batch_prefix, kv_head, n_rep, seq_len, head_dim
        )
        value = value.reshape(*batch_prefix, kv_head * n_rep, seq_len, head_dim)

    return scaled_dot_product_attention_simple(
        query, key, value, scale=scale, mask=mask
    )


def flash_attention(
    query: torch.Tensor,    # B, H_q, L, D
    key: torch.Tensor,      # B, H_kv, S, D
    value: torch.Tensor,    # B, H_kv, S, D
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:

    from extensions import flash_attention_forward

    *B, H_q, L, D = query.shape
    _, H_kv, S, _ = key.shape

    if scale is None:
        scale = 1.0 / math.sqrt(D)

    if H_q % H_kv != 0:
        raise ValueError(f"H_q ({H_q}) must be a multiple of H_kv ({H_kv})")

    query_3d = query.reshape(-1, L, D).contiguous()
    key_3d = key.reshape(-1, S, D).contiguous()
    value_3d = value.reshape(-1, S, D).contiguous()

    N = query_3d.shape[0]
    is_causal = mask == "causal"
    if is_causal:
        mask_tensor = causal_mask(L, S, torch.float32, query.device)
        mask_tensor = mask_tensor.unsqueeze(0).expand(N, L, S).contiguous()
    elif mask is None:
        mask_tensor = torch.zeros(N, L, S, dtype=torch.float32, device=query.device)
    else:
        # mask shape: (B, H_q, L, S) -> reshape to (N, L, S) where N = B * H_q
        mask_tensor = mask.reshape(N, L, S).contiguous().to(torch.float32)

    output = flash_attention_forward(
        query_3d,
        key_3d,
        value_3d,
        mask_tensor,
        scale,
        is_causal,
        H_q,
        H_kv,
    )

    return output.reshape(*B, H_q, L, D)