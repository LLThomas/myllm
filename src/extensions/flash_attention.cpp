#include "flash_attention.h"

namespace my_llm_ext {

torch::Tensor flash_attention_cpu(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    double scale,
    bool is_causal,
    int64_t num_heads,
    int64_t num_kv_heads
);

torch::Tensor flash_attention_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& mask,
    double scale,
    bool is_causal,
    int64_t num_heads,
    int64_t num_kv_heads
) {
    TORCH_CHECK(q.dim() == 3, "Q must be 3D: [N, L, E]");
    TORCH_CHECK(k.dim() == 3, "K must be 3D: [N_KV, S, E]");
    TORCH_CHECK(v.dim() == 3, "V must be 3D: [N_KV, S, E]");
    TORCH_CHECK(mask.dim() == 3, "Mask must be 3D: [N, L, S]");

    TORCH_CHECK(q.size(2) == k.size(2), "Q and K must have same embedding dimension");
    TORCH_CHECK(k.size(1) == v.size(1), "K and V must have same sequence length");
    TORCH_CHECK(k.size(0) == v.size(0), "K and V must have same batch size");

    TORCH_CHECK(num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads");
    TORCH_CHECK(q.size(0) % num_heads == 0, "Q batch must be divisible by num_heads");
    TORCH_CHECK(k.size(0) % num_kv_heads == 0, "K batch must be divisible by num_kv_heads");
    TORCH_CHECK(q.size(0) / num_heads == k.size(0) / num_kv_heads, "Batch size mismatch");

    return flash_attention_cpu(q, k, v, mask, scale, is_causal, num_heads, num_kv_heads);
}

}  // namespace my_llm_ext