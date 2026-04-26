import torch
import torch.nn as nn
import torch.nn.functional as F


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, weight: torch.tensor):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data = weight

    def __call__(self, x: torch.tensor) -> torch.tensor:
        return self.embedding(x)

    def as_linear(self, x: torch.tensor) -> torch.tensor:
        # Ensure x and weight have the same dtype
        if x.dtype != self.embedding.weight.dtype:
            x = x.to(self.embedding.weight.dtype)
        return F.linear(x, self.embedding.weight)