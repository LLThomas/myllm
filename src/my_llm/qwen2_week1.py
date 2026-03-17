import torch
import torch.nn as nn
from utils import linear, silu
from .attention import scaled_dot_product_attention_grouped
from .embedding import Embedding
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any


class Qwen2MultiHeadAttention(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        max_seq_len: int = 32768,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.bq = bq
        self.bk = bk
        self.bv = bv
        self.wo = wo

        self.hidden_size = hidden_size
        # query heads
        self.num_heads = num_heads
        # kv heads
        self.num_kv_heads = num_kv_heads
        self.max_seq_len = max_seq_len

        self.rope = RoPE(hidden_size // num_heads, max_seq_len, theta)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        # x: (B, L, E) -> (1, 4, 32)
        # wq: (num_heads * head_dim, hidden_size) -> (4 * 8, 32)
        # bq: (num_heads * head_dim, 1) -> (4 * 8, 1)
        # q: (B, L, Eq) -> (1, 4, 32)
        q = linear(x, self.wq, self.bq)
        # x: (B, L, E) -> (1, 4, 32)
        # wk: (key_value_heads * head_dim, hidden_size) -> (2 * 8, 32)
        # bk: (key_value_heads * head_dim, 1) -> (2 * 8, 1)
        # k: (B, L, Ek) -> (1, 4, 16)
        k = linear(x, self.wk, self.bk)
        v = linear(x, self.wv, self.bv)

        # get seq_len and dims
        batch = x.shape[0]
        seq_len = x.shape[-2]
        dims = self.hidden_size // self.num_heads

        # chage shape
        # (B, L, E) -> (B, L, H, D) -> (B, H, L, D)
        q = q.view(batch, seq_len, self.num_heads, dims).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, dims).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, dims).transpose(1, 2)

        q = self.rope(q)
        k = self.rope(k)

        # (B, H_kv, n_rep, L, D)
        attn_output = scaled_dot_product_attention_grouped(q, k, v, None, mask)
        if attn_output.dim() == 5:
            attn_output = attn_output.permute(0, 3, 1, 2, 4).reshape(batch, seq_len, -1)
        else:
            attn_output = attn_output.transpose(1, 2).reshape(
                batch, seq_len, self.hidden_size
            )

        # linear layer
        # (B, L, E) · (hidden_size, num_heads * head_dim)
        # wo: (hidden_size, num_heads * head_dim) -> (32, 32)
        return linear(attn_output, self.wo)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim

        assert w_gate.shape == (
            hidden_dim,
            dim,
        ), f"w_gate shape {w_gate.shape} != {(hidden_dim, dim)}"
        assert w_up.shape == (
            hidden_dim,
            dim,
        ), f"w_up shape {w_up.shape} != {(hidden_dim, dim)}"
        assert w_down.shape == (
            dim,
            hidden_dim,
        ), f"w_down shape {w_down.shape} != {(dim, hidden_dim)}"

        self.w_gate = torch.nn.Parameter(w_gate)
        self.w_up = torch.nn.Parameter(w_up)
        self.w_down = torch.nn.Parameter(w_down)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert (
            x.shape[-1] == self.dim
        ), f"Input last dim {x.shape[-1]} != expected {self.dim}"

        # (SiLU(Wgate(x)) ⊙ Wup(x)) ·  Wdown
        # 1. gate linear proj
        # (N.. x L x E) · (I x E)^T -> (N.. x L x I)
        w_gate_proj = linear(x, self.w_gate)
        # 2. silu activation
        # (N.. x L x I)
        silu_w_gate_proj = silu(w_gate_proj)
        # 3. up linear proj
        # (N.. x L x E) · (I x E)^T -> (N.. x L x I)
        up_proj = linear(x, self.w_up)
        # 4. element-wise multiplcation of silu_w_gate_proj and up_proj
        # (N.. x L x I) ⊙ (N.. x L x I)
        ele_res = silu_w_gate_proj * up_proj
        # 5. apply down linear proj
        # (N.. x L x I) · (E x I)^T -> (N.. x L x E)
        return linear(ele_res, self.w_down)


class Qwen2TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 10000.0,
    ):
        super().__init__()

        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, eps=rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )

        self.attention = Qwen2MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            wq,
            wk,
            wv,
            wo,
            bq,
            bk,
            bv,
            max_seq_len,
            theta,
        )

        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        residual = x
        # pre norm
        x = self.input_layernorm(x)
        # attention
        x = self.attention(x, mask=mask)
        # residual
        x = residual + x

        residual = x
        # post attention layernorm
        x = self.post_attention_layernorm(x)
        # mlp
        x = self.mlp(x)
        # residual
        return residual + x


class Qwen2ModelWeek1(nn.Module):
    def __init__(self, model: Any):
        super().__init__()
        config = model.config
        hf_model = model.model

        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, hf_model.embed_tokens.weight.data
        )

        self.layers = nn.ModuleList(
            [
                Qwen2TransformerBlock(
                    num_attention_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    wq=hf_model.layers[i].self_attn.q_proj.weight.data,
                    wk=hf_model.layers[i].self_attn.k_proj.weight.data,
                    wv=hf_model.layers[i].self_attn.v_proj.weight.data,
                    wo=hf_model.layers[i].self_attn.o_proj.weight.data,
                    bq=hf_model.layers[i].self_attn.q_proj.bias.data,
                    bk=hf_model.layers[i].self_attn.k_proj.bias.data,
                    bv=hf_model.layers[i].self_attn.v_proj.bias.data,
                    w_gate=hf_model.layers[i].mlp.gate_proj.weight.data,
                    w_up=hf_model.layers[i].mlp.up_proj.weight.data,
                    w_down=hf_model.layers[i].mlp.down_proj.weight.data,
                    w_input_layernorm=hf_model.layers[i].input_layernorm.weight.data,
                    w_post_attention_layernorm=hf_model.layers[
                        i
                    ].post_attention_layernorm.weight.data,
                    theta=1000000.0,
                )
                for i in range(config.num_hidden_layers)
            ]
        )

        self.norm = RMSNorm(
            config.hidden_size, hf_model.norm.weight.data, eps=config.rms_norm_eps
        )

        self.tie_word_embeddings = config.tie_word_embeddings
        if not self.tie_word_embeddings:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            self.lm_head.weight.data = model.lm_head.weight.data

    def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(inputs)

        mask = "causal" if inputs.shape[1] > 1 else None

        # many Transformer block
        for layer in self.layers:
            # 1. pre norm
            # 2. multi-head attention
            # 3. add residual
            # 4. post norm
            # 5. mlp
            # 6. add residual
            x = layer(x, mask=mask)

        # norm
        x = self.norm(x)

        # linear proj
        if self.tie_word_embeddings:
            x = self.embed_tokens.as_linear(x)
        else:
            x = self.lm_head(x)

        return x