import torch
import copy


def make_sampler(temp: float, top_p: float, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        if temp == 0:
            return torch.argmax(logprobs, dim=-1)

        logits = logprobs.clone()

        if top_k is not None and top_k > 0:
            top_k_logits, top_k_indices = torch.topk(
                logits, min(top_k, logits.size(-1)), dim=-1
            )
            logits_filtered = torch.full_like(logits, float("-inf"))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered

        if top_p is not None and top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 0] = False

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits = logits.masked_fill(indices_to_remove, float("-inf"))

        logits = logits / temp
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).squeeze(-1)

        return next_token

    return sample