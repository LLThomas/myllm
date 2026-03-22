import torch
from typing import Callable, Optional, Any

from .qwen2_week1 import Qwen2ModelWeek1
from .qwen2_week2 import Qwen2ModelWeek2
from .kv_cache import TinyKvCache, TinyKvFullCache


def simple_generate(
    model: Qwen2ModelWeek1,
    tokenizer: Any,
    prompt: str,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> str:
    def _step(model: Qwen2ModelWeek1, y: torch.Tensor):
        # (S) -> (1, S)
        input_tensor = y.unsqueeze(0)

        # (1, S, vocab_size)
        logits = model(input_tensor)

        # (1, S, vocab_size) -> (1, vocab_size)
        last_token_logits = logits[:, -1, :]

        # greedy decoding
        next_token = sampler(last_token_logits)
        return next_token.item()

    # 1. tokenize
    tokenized_prompt = tokenizer.encode(prompt)
    tokenized_prompt = torch.tensor(tokenized_prompt, dtype=torch.long)
    generated_tokens = tokenized_prompt.tolist()

    max_length = 100
    eos_token_id = tokenizer.eos_token_id

    # 2. prefill: first call to _step
    # 3. decode: subsequent call to _step
    for _ in range(max_length):
        next_token = _step(model, torch.tensor(generated_tokens))

        if next_token == eos_token_id:
            break

        generated_tokens.append(next_token)

        # return word in real time
        print(tokenizer.decode([next_token]), end="", flush=True)

    # token id -> text (batch return)
    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


def simple_generate_with_kv_cache(
    model: Qwen2ModelWeek2,
    tokenizer: Any,
    prompt: str,
    sampler: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> str:
    device = next(model.parameters()).device

    def _step(
        model: Qwen2ModelWeek2,
        token: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
    ) -> int:
        # token: (1,) -> (1, 1)
        input_tensor = token.unsqueeze(0).unsqueeze(0)

        # (1, 1, vocab_size)
        logits = model(input_tensor, offset, cache)

        # (1, 1, vocab_size) -> (vocab_size,)
        last_token_logits = logits[0, 0, :]

        # sample
        next_token = sampler(last_token_logits)
        return next_token.item()

    # 1. tokenize
    tokenized_prompt = tokenizer.encode(prompt)
    generated_tokens = tokenized_prompt.copy()

    max_length = 100
    eos_token_id = tokenizer.eos_token_id

    # 2. init KV cache for each layer
    cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]

    # 3. prefill: process all prompt tokens
    prompt_tensor = torch.tensor(tokenized_prompt, dtype=torch.long, device=device)
    logits = model(prompt_tensor.unsqueeze(0), 0, cache)

    # Get first generated token
    last_token_logits = logits[:, -1, :]
    next_token = sampler(last_token_logits).item()
    generated_tokens.append(next_token)
    print(tokenizer.decode([next_token]), end="", flush=True)

    # 4. decode: generate tokens one by one using KV cache
    offset = len(tokenized_prompt)
    for _ in range(max_length - len(tokenized_prompt)):
        next_token = _step(
            model,
            torch.tensor(generated_tokens[-1], dtype=torch.long, device=device),
            offset,
            cache,
        )

        if next_token == eos_token_id:
            break

        generated_tokens.append(next_token)
        offset += 1

        print(tokenizer.decode([next_token]), end="", flush=True)

    generated_text = tokenizer.decode(generated_tokens)
    return generated_text


# def speculative_generate(
#     draft_model: "Qwen2ModelWeek2",
#     model: "Qwen2ModelWeek2",
#     draft_tokenizer: Any,
#     tokenizer: Any,
#     prompt: str,
# ) -> str:
#     device = next(model.parameters()).device
#     pass