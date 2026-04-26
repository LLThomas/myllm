import torch
from datetime import datetime
from typing import Any

from .kv_cache import TinyKvFullCache, BatchingKvCache, TinyKvCache
from .qwen2_week2 import Qwen2ModelWeek2


def _step(
    model: Qwen2ModelWeek2,
    tokens: torch.Tensor,
    offsets: list[int],
    kv_cache: list[TinyKvCache],
) -> torch.Tensor:
    """
    Single step of model inference (prefill or decode).

    Args:
        model: The Qwen2 model
        tokens: Input tokens, shape (B, L)
        offsets: Position offsets for each request
        kv_cache: List of KV caches, one per layer (TinyKvFullCache or BatchingKvCache)

    Returns:
        Next token predictions, shape (B,)
    """
    logits = model(tokens, offsets, kv_cache)
    # Take the last position's logits
    logits = logits[:, -1, :]

    # Compute log probabilities
    logprobs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    # Greedy sampling: argmax (same as tiny-llm)
    next_tokens = torch.argmax(logprobs, dim=-1)
    return next_tokens


class Request:
    """
    Represents a single generation request.

    Manages the request's KV cache, prefill state, and generated text.
    """

    def __init__(
        self,
        model: Qwen2ModelWeek2,
        tokenizer: Any,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
    ):
        """
        Initialize a request.

        Args:
            model: The Qwen2 model (needed for num_hidden_layers)
            tokenizer: HuggingFace tokenizer
            prompt: The prompt string
            prefill_max_step: Maximum tokens to prefill per chunk
            prompt_idx: Index of this prompt in the original list
        """
        self.prompt = prompt
        self.kv_cache = [
            TinyKvFullCache() for _ in range(model.num_hidden_layers)
        ]
        self.model = model
        self.tokenizer = tokenizer

        # Encode prompt
        self.prefill_tokens = torch.tensor(
            tokenizer.encode(prompt, add_special_tokens=False),
            dtype=torch.long
        )
        self.prefill_max_step = prefill_max_step
        self.is_done = False
        self.is_prefill_done = False
        self.eos_token_id = tokenizer.eos_token_id
        self.next_token = None
        self.offset = 0
        self.prompt_idx = prompt_idx

        # Use incremental decoding like tiny-llm
        # Store generated tokens for incremental decode
        self.generated_tokens = []
        self._text = ""  # Cache decoded text

    def try_prefill(self, device: torch.device = torch.device("cpu")):
        """
        Prefill this request up to max_step tokens.

        For chunked prefill, this processes a chunk of the prompt tokens
        and updates the internal KV cache.

        Raises:
            ValueError: If called after prefill is already done
        """
        if self.is_prefill_done:
            raise ValueError("prefill called after done")

        # Determine how many tokens to prefill in this chunk
        tokens_to_prefill = min(
            self.prefill_max_step,
            len(self.prefill_tokens) - self.offset
        )

        # Get the chunk of tokens
        chunk = self.prefill_tokens[self.offset : self.offset + tokens_to_prefill]
        chunk = chunk.unsqueeze(0).to(device)  # shape (1, L)

        # Run model step
        next_token = _step(
            self.model,
            chunk,
            [self.offset],
            self.kv_cache,
        )

        self.offset += tokens_to_prefill

        # Check if prefill is complete
        if self.offset == len(self.prefill_tokens):
            self.is_prefill_done = True
            # First generated token from prefill
            self.decode_done(next_token.item(), update_offset=False)

    def decode_done(self, token: int, update_offset: bool = True):
        """
        Process a decoded token.

        Args:
            token: The decoded token ID
            update_offset: Whether to increment offset (False for prefill's last token)

        Raises:
            ValueError: If called after request is done
        """
        if self.is_done:
            raise ValueError("decode called after done")

        if token == self.eos_token_id:
            self.is_done = True
            return

        self.generated_tokens.append(token)
        self.next_token = token

        if update_offset:
            self.offset += 1

    def text(self) -> str:
        """Get the generated text so far."""
        if len(self.generated_tokens) == 0:
            return ""
        return self.tokenizer.decode(
            self.generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )


def _print_progress(
    requests: list[Request | None],
    pending_prefill_request: Request | None,
    queue_size: int,
    progress_cnt: int,
    start_time: datetime,
):
    """Print progress of all active requests."""
    elapsed = datetime.now() - start_time
    animation_frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    animation_frame = animation_frames[progress_cnt % len(animation_frames)]

    print(f"  --- {elapsed}")
    for i, request in enumerate(requests):
        if request is None:
            print(f"  Decode #{i}: idle", flush=True)
        else:
            text_preview = request.text()[-80:].replace("\n", " ")
            print(
                f"{animation_frame} Decode [req {request.prompt_idx}, {request.offset}]: {text_preview}",
                flush=True,
            )

    if pending_prefill_request is not None:
        if pending_prefill_request.is_prefill_done:
            print(
                f"  Prefill [req {pending_prefill_request.prompt_idx}]: done, waiting for slot, {queue_size} requests in queue",
                flush=True,
            )
            return
        percentage = (
            pending_prefill_request.offset / len(pending_prefill_request.prefill_tokens)
        ) * 100
        remaining = len(pending_prefill_request.prefill_tokens) - pending_prefill_request.offset
        print(
            f"{animation_frame} Prefill [req {pending_prefill_request.prompt_idx}]: {percentage:.2f}% ({remaining} remaining tokens)",
            flush=True,
        )
    else:
        print(f"  Prefill: idle, {queue_size} requests in queue", flush=True)


def batch_generate(
    model: Qwen2ModelWeek2,
    tokenizer: Any,
    prompts: list[str],
    max_seq_len: int = 512,
    batch_size: int = 5,
    prefill_steps: int = 128,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
) -> list[tuple[int, str]]:
    """
    Generate text for multiple prompts using continuous batching.

    This implements the continuous batching algorithm:
    1. Prefill requests one chunk at a time
    2. When a prefill completes, add to decode batch if slot available
    3. Decode all active requests together
    4. When a request finishes, remove and free slot

    Args:
        model: The Qwen2 model
        tokenizer: HuggingFace tokenizer
        prompts: List of prompt strings
        max_seq_len: Maximum sequence length per request
        batch_size: Maximum concurrent requests
        prefill_steps: Tokens to prefill per chunk
        device: Device to run on
        verbose: Whether to print progress

    Returns:
        List of (prompt_idx, generated_text) tuples
    """
    decode_requests: list[Request | None] = [None] * batch_size

    kv_cache = [
        BatchingKvCache(max_active_requests=batch_size, max_seq_len=max_seq_len)
        for _ in range(model.num_hidden_layers)
    ]

    result = []
    pending_prefill_request = None
    prompts = list(prompts)  # Make a copy we can pop from
    next_request_idx = 0
    progress_cnt = 0
    start_time = datetime.now()

    while True:
        if len(prompts) == 0 and all(req is None for req in decode_requests):
            break

        # Start prefilling next request if we have capacity
        if len(prompts) > 0 and pending_prefill_request is None:
            prompt = prompts.pop(0)
            pending_prefill_request = Request(
                model, tokenizer, prompt, prefill_steps, next_request_idx
            )
            next_request_idx += 1

        # Prefill step
        if pending_prefill_request is not None:
            made_progress = False

            # Process a chunk of prefill
            if not pending_prefill_request.is_prefill_done:
                pending_prefill_request.try_prefill(device)
                made_progress = True

            # If prefill done, try to add to decode batch
            if pending_prefill_request.is_prefill_done:
                prefill_kv_cache = pending_prefill_request.kv_cache
                found_slot = False

                for i in range(batch_size):
                    if decode_requests[i] is None:
                        # Add this request to the decode batch
                        for prefill_cache, batch_cache in zip(
                            prefill_kv_cache, kv_cache
                        ):
                            batch_cache.add_request(prefill_cache, i)
                        decode_requests[i] = pending_prefill_request
                        found_slot = True
                        made_progress = True
                        break

                if found_slot:
                    pending_prefill_request = None

            if made_progress and verbose:
                _print_progress(
                    decode_requests,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1

        # Decode step for all active requests
        if any(req is not None for req in decode_requests):
            # Gather next tokens and offsets for active requests
            next_tokens = []
            offsets = []
            for req in decode_requests:
                if req is not None:
                    next_tokens.append(req.next_token if req.next_token is not None else 0)
                    offsets.append(req.offset)
                else:
                    next_tokens.append(0)
                    offsets.append(0)

            next_tokens = torch.tensor(next_tokens, dtype=torch.long, device=device)
            next_tokens = next_tokens.reshape(-1, 1)  # (B, 1)

            # Run decode step (simple argmax, same as tiny-llm)
            next_tokens = _step(model, next_tokens, offsets, kv_cache)

            # Process results for each request
            for i in range(batch_size):
                req = decode_requests[i]
                if req is not None:
                    token = next_tokens[i].item()
                    req.decode_done(token)

                    # Debug: Check if token is unusual
                    if token == 0 or token > 100:
                        pass  # Suppress debug

                    # Check for completion
                    remove_reason = None
                    if req.is_done:
                        remove_reason = "EOS"
                    elif req.offset >= max_seq_len:
                        remove_reason = "max seq len"

                    if remove_reason is not None:
                        if verbose:
                            print(f"Removing request {i} due to {remove_reason}", flush=True)
                        # Remove from batch cache
                        for layer_cache in kv_cache:
                            layer_cache.remove_request(i)
                        # Save result
                        result.append((req.prompt_idx, req.text()))
                        decode_requests[i] = None

            if verbose:
                _print_progress(
                    decode_requests,
                    pending_prefill_request,
                    len(prompts),
                    progress_cnt,
                    start_time,
                )
                progress_cnt += 1

    return result