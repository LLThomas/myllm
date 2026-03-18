import inspect
import pytest
import torch
import torch.nn.functional as F
from my_llm import *
from utils import *
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer, Qwen2RotaryEmbedding


def qwen_model_exists(model_id):
    try:
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return True
    except Exception:
        return False


MODEL_MAP = {
    "0.5B": "Qwen/Qwen2-0.5B-Instruct",
}


def test_task_1_transformer_block():
    """
    Test transformer block implementation
    """
    # Test different devices and precisions
    test_devices = AVAILABLE_DEVICES
    test_precisions = PRECISIONS

    for device in test_devices:
        for precision in test_precisions:
            # Test different mask configurations
            masks = [None, "causal"]

            for mask in masks:
                torch.manual_seed(42)

                # Parameters for testing
                batch_size = 1
                seq_len = 10
                num_attention_head = 4
                num_kv_heads = 2
                hidden_size = 32
                intermediate_size = hidden_size * 4

                config = Qwen2Config(
                    model_type="qwen2",
                    hidden_size=hidden_size,
                    num_hidden_layers=1,
                    intermediate_size=intermediate_size,
                    num_attention_heads=num_attention_head,
                    num_key_value_heads=num_kv_heads,
                    rms_norm_eps=1e-6,
                    vocab_size=1000,
                )

                hf_block = Qwen2DecoderLayer(config, layer_idx=0).to(
                    device=device, dtype=precision
                )
                hf_block.eval()

                user_transformer_block = qwen2_week1.Qwen2TransformerBlock(
                    num_attention_heads=num_attention_head,
                    num_kv_heads=num_kv_heads,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    rms_norm_eps=1e-6,
                    wq=hf_block.self_attn.q_proj.weight.clone(),
                    wk=hf_block.self_attn.k_proj.weight.clone(),
                    wv=hf_block.self_attn.v_proj.weight.clone(),
                    wo=hf_block.self_attn.o_proj.weight.clone(),
                    bq=hf_block.self_attn.q_proj.bias.clone(),
                    bk=hf_block.self_attn.k_proj.bias.clone(),
                    bv=hf_block.self_attn.v_proj.bias.clone(),
                    w_gate=hf_block.mlp.gate_proj.weight.clone(),
                    w_up=hf_block.mlp.up_proj.weight.clone(),
                    w_down=hf_block.mlp.down_proj.weight.clone(),
                    w_input_layernorm=hf_block.input_layernorm.weight.clone(),
                    w_post_attention_layernorm=hf_block.post_attention_layernorm.weight.clone(),
                ).to(device=device, dtype=precision)
                user_transformer_block.eval()

                x = torch.rand(
                    (batch_size, seq_len, hidden_size), dtype=precision, device=device
                )
                with torch.no_grad():
                    user_output = user_transformer_block(x, mask=mask)

                    bo_zeros = torch.zeros(
                        (batch_size, seq_len, config.hidden_size),
                        device=device,
                        dtype=precision,
                    )
                    hf_attention_mask = None
                    if mask == "causal":
                        hf_attention_mask = (
                            torch.triu(
                                torch.full(
                                    (seq_len, seq_len),
                                    float("-inf"),
                                    dtype=precision,
                                    device=device,
                                ),
                                diagonal=1,
                            )
                            .unsqueeze(0)
                            .unsqueeze(0)
                        )
                    
                    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
                    rope = Qwen2RotaryEmbedding(config).to(device=device, dtype=precision)
                    position_embeddings = rope(x, position_ids)

                    hf_output = hf_block(
                        x,
                        bo=bo_zeros,
                        attention_mask=hf_attention_mask,
                        position_ids=position_ids,
                        position_embeddings=position_embeddings,
                    )[0]

                assert_allclose(user_output, hf_output, precision)


@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B not found"
)
def test_utils_qwen_2_05b():
    """
    Placeholder test for Qwen2-0.5B model existence check
    """
    pass


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B not found"
)
def test_task_2_embedding_call(device, precision):
    """
    Improved test for embedding call functionality
    """
    model_id = MODEL_MAP["0.5B"]
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=precision).to(
        device
    )

    embedding = Embedding(
        hf_model.config.vocab_size,
        hf_model.config.hidden_size,
        hf_model.model.embed_tokens.weight.data.clone(),
    )
    embedding = embedding.to(device)

    # Test multiple iterations
    for i in range(50):
        input_ids = torch.randint(
            low=0, high=hf_model.config.vocab_size, size=(1, 10), device=device
        )
        with torch.no_grad():
            user_output = embedding(input_ids)
            ref_output = hf_model.model.embed_tokens(input_ids)

        assert_allclose(user_output, ref_output, precision)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.parametrize("precision", [torch.float16], ids=["fp16"])
@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B not found"
)
def test_task_2_embedding_as_linear(device, precision):
    """
    Improved test for embedding as linear functionality
    """
    model_id = MODEL_MAP["0.5B"]
    hf_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=precision).to(
        device
    )

    embedding = Embedding(
        hf_model.config.vocab_size,
        hf_model.config.hidden_size,
        hf_model.model.embed_tokens.weight.data.clone(),
    )
    embedding = embedding.to(device)

    # Test multiple iterations
    for i in range(50):
        x = torch.rand(
            (1, 10, hf_model.config.hidden_size), dtype=precision, device=device
        )

        with torch.no_grad():
            user_output = embedding.as_linear(x)
            ref_output = hf_model.lm_head(x)

        assert_allclose(user_output, ref_output, precision)


def helper_test_task_3(model_name: str, device: str, iters: int = 10):
    """
    Helper function for task 3 testing with improved structure
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    precision = torch.float32
    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=precision).to(device).eval()

    model = Qwen2ModelWeek1(hf_model).to(device).eval()

    sig = inspect.signature(model.forward)
    needs_mask = "mask" in sig.parameters

    # Test multiple iterations
    for i in range(iters):
        input_ids = torch.randint(
            low=0, high=tokenizer.vocab_size, size=(1, 10), device=device
        )

        with torch.no_grad():
            if needs_mask:
                seq_len = input_ids.shape[1]
                causal_mask = torch.triu(
                    torch.full((seq_len, seq_len), float("-inf"), device=device, dtype=precision), 
                    diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                user_output = model(input_ids, mask=causal_mask)
            else:
                user_output = model(input_ids)

            user_output = user_output - torch.logsumexp(
                user_output, dim=-1, keepdim=True
            )

            ref_output = hf_model(input_ids).logits
            ref_output = ref_output - torch.logsumexp(ref_output, dim=-1, keepdim=True)

        assert_allclose(user_output, ref_output, atol=1e-4)


@pytest.mark.parametrize("device", AVAILABLE_DEVICES, ids=AVAILABLE_DEVICES_IDS)
@pytest.mark.skipif(
    not qwen_model_exists(MODEL_MAP["0.5B"]), reason="Qwen2-0.5B not found"
)
def test_task_3_qwen_2_05b(device):
    """
    Test Qwen2 0.5B model
    """
    helper_test_task_3(MODEL_MAP["0.5B"], device, 5)