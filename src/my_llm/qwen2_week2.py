import math
import torch
import torch.nn as nn
from utils import linear, silu
from .attention import scaled_dot_product_attention_grouped, flash_attention
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .embedding import Embedding
from .quantize import dequantize_linear, QuantizedWeights, quantized_linear
from .kv_cache import TinyKvCache, TinyKvFullCache
from typing import Any


class Qwen2MultiHeadAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        max_seq_len: int = 32768,
        theta: float = 1000000.0,
        use_flash_attention: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = hidden_size // num_heads

        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.bq = bq
        self.bk = bk
        self.bv = bv

        self.rope = RoPE(self.head_dim, max_seq_len, theta)
        self.use_flash_attention = use_flash_attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = quantized_linear(x, self.wq, self.bq)
        k = quantized_linear(x, self.wk, self.bk)
        v = quantized_linear(x, self.wv, self.bv)

        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q = self.rope(q, offset=offset)
        k = self.rope(k, offset=offset)

        full_k, full_v, _, mask_out = cache.update_and_fetch(k, v, mask=mask)

        if self.use_flash_attention:
            attn_output = flash_attention(
                q.to(torch.float32),
                full_k.to(torch.float32),
                full_v.to(torch.float32),
                scale=self.scale,
                mask=mask_out,
            ).to(x.dtype)
        else:
            attn_output = scaled_dot_product_attention_grouped(
                q, full_k, full_v, None, mask=mask_out
            )

        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, self.hidden_size)
        return quantized_linear(attn_output, self.wo)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = quantized_linear(x, self.w_gate)
        gate = silu(gate)
        up = quantized_linear(x, self.w_up)
        return quantized_linear(gate * up, self.w_down)


class Qwen2TransformerBlock(nn.Module):
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: float = 1000000.0,
        use_flash_attention: bool = False,
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
            wq, wk, wv, wo,
            bq, bk, bv,
            max_seq_len,
            theta,
            use_flash_attention,
        )

        self.mlp = Qwen2MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)

    def forward(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.attention(x, offset, cache, mask)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        return residual + x


class Qwen2ModelWeek2(nn.Module):
    def __init__(
        self,
        model: Any,
        enable_flash_attn: bool = False,
    ):
        super().__init__()
        config = model.config

        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.tie_word_embeddings = config.tie_word_embeddings
        precision = torch.float16
        self.precision = precision

        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dequantize_linear(model.model.embed_tokens).to(precision)
        )

        theta = (config.rope_parameters.get("rope_theta") 
                 if hasattr(config, "rope_parameters") and config.rope_parameters 
                 else getattr(config, "rope_theta", 1000000.0))
        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            layer = model.model.layers[i]
            
            def safe_to_precision(obj):
                return obj.to(precision) if obj is not None else None

            self.layers.append(
                Qwen2TransformerBlock(
                    num_attention_heads=config.num_attention_heads,
                    num_kv_heads=config.num_key_value_heads,
                    hidden_size=config.hidden_size,
                    intermediate_size=config.intermediate_size,
                    rms_norm_eps=config.rms_norm_eps,
                    wq=QuantizedWeights.from_weight(layer.self_attn.q_proj),
                    wk=QuantizedWeights.from_weight(layer.self_attn.k_proj),
                    wv=QuantizedWeights.from_weight(layer.self_attn.v_proj),
                    wo=QuantizedWeights.from_weight(layer.self_attn.o_proj),
                    bq=safe_to_precision(layer.self_attn.q_proj.bias),
                    bk=safe_to_precision(layer.self_attn.k_proj.bias),
                    bv=safe_to_precision(layer.self_attn.v_proj.bias),
                    
                    w_gate=QuantizedWeights.from_weight(layer.mlp.gate_proj),
                    w_up=QuantizedWeights.from_weight(layer.mlp.up_proj),
                    w_down=QuantizedWeights.from_weight(layer.mlp.down_proj),
                    
                    w_input_layernorm=layer.input_layernorm.weight.to(precision),
                    w_post_attention_layernorm=layer.post_attention_layernorm.weight.to(precision),
                    max_seq_len=config.max_position_embeddings,
                    theta=theta,
                    use_flash_attention=enable_flash_attn,
                )
            )

        self.norm = RMSNorm(
            config.hidden_size,
            model.model.norm.weight.to(precision),
            eps=config.rms_norm_eps
        )

        if not self.tie_word_embeddings:
            self.w_lm_head = QuantizedWeights.from_layer(model.lm_head)
        else:
            self.w_lm_head = None

    def forward(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
    ) -> torch.Tensor:
        x = self.embed_tokens(inputs)

        mask = "causal" if inputs.shape[1] > 1 else None

        for i, layer in enumerate(self.layers):
            x = layer(x, offset, cache[i], mask)

        x = self.norm(x)

        if self.w_lm_head is not None:
            return quantized_linear(x, self.w_lm_head)
        else:
            return self.embed_tokens.as_linear(x)