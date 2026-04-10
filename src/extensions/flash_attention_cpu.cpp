#include "flash_attention.h"
#include <cmath>
#include <limits>
#include <algorithm>

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
) {
    auto q_contiguous = q.contiguous().to(torch::kFloat32);
    auto k_contiguous = k.contiguous().to(torch::kFloat32);
    auto v_contiguous = v.contiguous().to(torch::kFloat32);
    auto mask_contiguous = mask.contiguous().to(torch::kFloat32);

    const int64_t N = q.size(0);        // batch * num_heads
    const int64_t L = q.size(1);        // query sequence length
    const int64_t E = q.size(2);        // head dimension
    const int64_t N_KV = k.size(0);     // batch * num_kv_heads
    const int64_t S = k.size(1);        // key/value sequence length

    const int64_t Br = 32;
    const int64_t Bc = 32;
    const int64_t Tr = (L + Br - 1) / Br;
    const int64_t Tc = (S + Bc - 1) / Bc;

    const int64_t q_kv_ratio = num_heads / num_kv_heads;

    auto output = torch::zeros({N, L, E}, q.options());

    const float* q_ptr = q_contiguous.data_ptr<float>();
    const float* k_ptr = k_contiguous.data_ptr<float>();
    const float* v_ptr = v_contiguous.data_ptr<float>();
    const float* mask_ptr = mask_contiguous.data_ptr<float>();
    float* out_ptr = output.data_ptr<float>();

    const int64_t N_Q_HEAD = L * E;
    const int64_t N_K_HEAD = S * E;

    for (int64_t n = 0; n < N; n++) {
        const float* q_batch = q_ptr + n * N_Q_HEAD;
        const float* k_batch = k_ptr + (n / q_kv_ratio) * N_K_HEAD;
        const float* v_batch = v_ptr + (n / q_kv_ratio) * N_K_HEAD;

        for (int64_t i = 0; i < Tr; i++) {
            std::vector<float> q_i(Br * E, 0.0f);
            int br_upper_bound = std::min(L - i * Br, Br);

            for (int64_t a = 0; a < br_upper_bound; a++) {
                for (int64_t b = 0; b < E; b++) {
                    int q_idx = (i * Br + a) * E + b;
                    q_i[a * E + b] = q_batch[q_idx];
                }
            }

            std::vector<float> o_i(Br * E, 0.0f);
            std::vector<float> l_i(Br, 0.0f);
            std::vector<float> m_i(Br, -std::numeric_limits<float>::infinity());

            const int64_t causal_offset = S - L;

            for (int64_t j = 0; j < Tc; j++) {
                int64_t row_max = i * Br + br_upper_bound - 1;
                int64_t col_min = j * Bc;

                if (is_causal && col_min > row_max + causal_offset) {
                    continue;
                }

                int bc_upper_bound = std::min(S - j * Bc, Bc);

                std::vector<float> k_j(Bc * E, 0.0f);
                std::vector<float> v_j(Bc * E, 0.0f);

                for (int64_t a = 0; a < bc_upper_bound; a++) {
                    int64_t kv_idx_base = j * Bc + a;
                    for (int64_t b = 0; b < E; b++) {
                        int kv_idx = kv_idx_base * E + b;
                        if (kv_idx_base < S) {
                            k_j[a * E + b] = k_batch[kv_idx];
                            v_j[a * E + b] = v_batch[kv_idx];
                        }
                    }
                }

                std::vector<float> s_i(Br * Bc, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t b = 0; b < bc_upper_bound; b++) {
                        for (int64_t c = 0; c < E; c++) {
                            s_i[a * Bc + b] += q_i[a * E + c] * k_j[b * E + c];
                        }
                    }
                }

                int64_t row_min = i * Br;
                int64_t col_max = j * Bc + bc_upper_bound - 1;
                bool block_all_valid = is_causal && (col_max <= row_min + causal_offset);

                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t b = 0; b < bc_upper_bound; b++) {
                        s_i[a * Bc + b] *= scale;

                        if (!block_all_valid) {
                            int m_idx = n * L * S + (i * Br + a) * S + (j * Bc + b);
                            s_i[a * Bc + b] += mask_ptr[m_idx];
                        }
                    }
                }

                std::vector<float> m_i_diff(Br, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    float rowmax = -std::numeric_limits<float>::infinity();
                    for (int64_t b = 0; b < bc_upper_bound; b++) {
                        rowmax = std::max(rowmax, s_i[a * Bc + b]);
                    }
                    float new_max = std::max(m_i[a], rowmax);
                    m_i_diff[a] = m_i[a] - new_max;
                    m_i[a] = new_max;
                }

                std::vector<float> p(Br * Bc, 0.0f);
                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t b = 0; b < bc_upper_bound; b++) {
                        p[a * Bc + b] = std::exp(s_i[a * Bc + b] - m_i[a]);
                    }
                }

                for (int64_t a = 0; a < br_upper_bound; a++) {
                    float rowsum = 0.0f;
                    for (int64_t b = 0; b < bc_upper_bound; b++) {
                        rowsum += p[a * Bc + b];
                    }
                    l_i[a] = std::exp(m_i_diff[a]) * l_i[a] + rowsum;
                }

                for (int64_t a = 0; a < br_upper_bound; a++) {
                    for (int64_t c = 0; c < E; c++) {
                        float res = 0.0f;
                        for (int64_t b = 0; b < bc_upper_bound; b++) {
                            res += p[a * Bc + b] * v_j[b * E + c];
                        }
                        o_i[a * E + c] = std::exp(m_i_diff[a]) * o_i[a * E + c] + res;
                    }
                }
            }

            for (int64_t a = 0; a < br_upper_bound; a++) {
                for (int64_t b = 0; b < E; b++) {
                    o_i[a * E + b] /= l_i[a];
                }
            }

            for (int64_t a = 0; a < br_upper_bound; a++) {
                for (int64_t b = 0; b < E; b++) {
                    int out_idx = i * Br + a;
                    if (out_idx < L) {
                        out_ptr[n * N_Q_HEAD + out_idx * E + b] = o_i[a * E + b];
                    }
                }
            }
        }
    }

    return output;
}

}  // namespace my_llm_ext