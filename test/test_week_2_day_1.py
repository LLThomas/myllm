import pytest
import torch
from utils import *
from my_llm import (
    Qwen2ModelWeek2,
    TinyKvFullCache,
)
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_MAP = {
    "0.5B": "Qwen/Qwen2-0.5B-Instruct",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B model not found"
)
def test_utils_qwen_2_05b():
    pass


def helper_test_task_3(model_name: str, iters: int = 10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    
    model = Qwen2ModelWeek2(hf_model).to(device)
    
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        input_ids = torch.randint(
            0, tokenizer.vocab_size, (1, 10), device=device
        )
        
        user_output = model(input_ids, 0, cache)
        user_output = user_output - torch.logsumexp(user_output, dim=-1, keepdim=True)
        
        with torch.no_grad():
            ref_output = hf_model(input_ids).logits
            ref_output = ref_output - torch.logsumexp(ref_output, dim=-1, keepdim=True)
        
        torch.testing.assert_close(user_output, ref_output, rtol=0.1, atol=0.5)


@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B model not found"
)
def test_task_3_qwen_2_05b():
    helper_test_task_3("Qwen/Qwen2-0.5B-Instruct", 5)


def helper_test_task_4(model_name: str, seq_len: int, iters: int = 1):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device
    )
    model = Qwen2ModelWeek2(hf_model).to(device)
    
    for _ in range(iters):
        cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
        inputs = torch.randint(0, tokenizer.vocab_size, (1, seq_len), device=device)
        
        with torch.no_grad():
            ref_outputs = hf_model(inputs).logits
            
        for offset in range(seq_len):
            user_out = model(
                inputs[:, offset : offset + 1], 
                offset, 
                cache
            )
            ref_out = ref_outputs[:, offset : offset + 1, :]
            
            user_out = user_out - torch.logsumexp(user_out, dim=-1, keepdim=True)
            ref_out = ref_out - torch.logsumexp(ref_out, dim=-1, keepdim=True)
            
            torch.testing.assert_close(user_out, ref_out, rtol=1e-1, atol=0.5)

@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B model not found"
)
def test_task_4_qwen_2_05b():
    helper_test_task_4("Qwen/Qwen2-0.5B-Instruct", seq_len=3)