#include "flash_attention.h"
#include <pybind11/pybind11.h>

namespace py = pybind11;

PYBIND11_MODULE(my_llm_ext, m) {
    m.doc() = "MyLLM C++ extensions for Flash Attention";

    m.def(
        "flash_attention_forward",
        &my_llm_ext::flash_attention_forward,
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("mask"),
        py::arg("scale"),
        py::arg("is_causal"),
        py::arg("num_heads"),
        py::arg("num_kv_heads"),
        R"doc(
        Flash Attention forward pass.

        Args:
            q: Query tensor of shape [N, L, E] where N = batch * num_heads
            k: Key tensor of shape [N_KV, S, E] where N_KV = batch * num_kv_heads
            v: Value tensor of shape [N_KV, S, E]
            mask: Attention mask of shape [N, L, S]
            scale: Scaling factor for attention scores
            is_causal: Whether to apply causal masking optimizations
            num_heads: Number of query heads
            num_kv_heads: Number of key/value heads

        Returns:
            Output tensor of shape [N, L, E]
        )doc"
    );
}