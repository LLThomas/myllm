import torch
from transformers import AutoTokenizer

AVAILABLE_DEVICES = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
AVAILABLE_DEVICES_IDS = AVAILABLE_DEVICES
PRECISIONS = [torch.float32, torch.float64]
PRECISION_IDS = ["fp32", "fp64"]


def assert_allclose(
    a: torch.Tensor, b: torch.Tensor, precision=torch.float32, rtol=1e-5, atol=1e-6
):
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"Mismatch found.\n{a}\n{b}"


def qwen_model_exists(model_id):
    try:
        AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        return True
    except Exception:
        return False