#pragma once

#include <torch/extension.h>
#include <vector>

namespace my_llm_ext {

// Flash Attention forward pass
// Q: [N, L, E] - N = batch * num_heads
// K: [N_KV, S, E] - N_KV = batch * num_kv_heads
// V: [N_KV, S, E]
// Mask: [N, L, S]
// Output: [N, L, E]
torch::Tensor flash_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    double scale,
    bool is_causal,
    int64_t num_heads,
    int64_t num_kv_heads
);

}  // namespace my_llm_ext