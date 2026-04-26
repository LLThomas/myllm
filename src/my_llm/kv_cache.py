import torch
from abc import ABC, abstractmethod
from typing import Optional, Union


class TinyKvCache(ABC):
    @abstractmethod
    def update_and_fetch(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        mask_length: Optional[int] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        """
        Update the key-value cache and fetch the updated key-value cache.
        """


class BatchingKvCache(TinyKvCache):
    def __init__(self, max_active_requests: int, max_seq_len: int):
        self.max_active_requests = max_active_requests
        self.max_seq_len = max_seq_len
        self.kv_caches: list[Optional[TinyKvCache]] = [None] * max_active_requests
        self.HD = None  # Cache (H, D) dimensions for validation

    def update_and_fetch(
        self,
        keys: torch.Tensor,      # (B, H, S, D)
        values: torch.Tensor,    # (B, H, S, D)
        mask_length: Optional[int] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        B, H, S, D = keys.shape
        assert keys.shape == values.shape
        assert S <= self.max_seq_len
        assert B == self.max_active_requests

        if self.HD is None:
            self.HD = (H, D)
        else:
            assert self.HD == (H, D), f"expect {self.HD} but got {(H, D)}"

        # Step 1: Update each slot's individual cache
        data = []
        for b in range(B):
            if self.kv_caches[b] is None:
                data.append(None)
                continue
            # keys[b:b+1] keeps batch dim: shape (1, H, S, D)
            new_key, new_value, seq_len, _ = self.kv_caches[b].update_and_fetch(
                keys[b : b + 1], values[b : b + 1]
            )
            # Remove batch dim: (H, S_total, D)
            data.append((new_key[0], new_value[0], seq_len))

        # Step 2: Find max sequence length across all active slots
        def get_seq_len(d):
            return d[2] if d is not None else 0

        max_seq = max(map(get_seq_len, data))

        # Step 3: Assemble batched tensors with right-aligned padding
        #   - keys/values: right-align each slot's data, pad left with zeros
        #   - mask: fill -inf everywhere, then set valid positions with causal mask
        batched_keys = torch.zeros(
            B, H, max_seq, D, dtype=keys.dtype, device=keys.device
        )
        batched_values = torch.zeros(
            B, H, max_seq, D, dtype=values.dtype, device=values.device
        )
        batched_mask = torch.full(
            (B, mask_length, max_seq), float("-inf"), dtype=keys.dtype, device=keys.device
        )

        from .attention import causal_mask

        for b in range(B):
            if data[b] is None:
                continue
            k, v, s = data[b]
            # Right-align: place data at [max_seq - s : max_seq]
            batched_keys[b, :, max_seq - s : max_seq, :] = k
            batched_values[b, :, max_seq - s : max_seq, :] = v
            # Generate causal mask for valid positions
            batched_mask[b, :, max_seq - s : max_seq] = causal_mask(
                mask_length, s, dtype=keys.dtype, device=keys.device
            )

        # mask shape: (B, 1, L, S) for broadcasting with attention scores (B, H_q, L, S)
        return batched_keys, batched_values, None, batched_mask.reshape(B, 1, mask_length, max_seq)

    def add_request(self, prefilled: TinyKvCache, id: int):
        """Add a prefilled KV cache to the specified slot."""
        if id >= self.max_active_requests:
            raise ValueError(f"Request id {id} is out of range")
        # Validate HD dimensions if prefilled cache has data
        if getattr(prefilled, "key_values", None) is not None:
            keys, _ = prefilled.key_values
            _, H, _, D = keys.shape
            if self.HD is None:
                self.HD = (H, D)
            else:
                assert self.HD == (H, D), f"expect {self.HD} but got {(H, D)}"
        self.kv_caches[id] = prefilled

    def remove_request(self, id: int):
        """Remove a request from the specified slot."""
        if self.kv_caches[id] is None:
            raise ValueError(f"Request id {id} is not in the cache")
        self.kv_caches[id] = None


class TinyKvFullCache(TinyKvCache):
    def __init__(self):
        self.key_values = None
        self.offset = 0

    def update_and_fetch(
        self,
        key: torch.Tensor,      # B, H, L, D
        value: torch.Tensor,    # B, H, L, D
        mask_length: Optional[int] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        # 1. concat new coming key, val
        if self.key_values is None:
            self.key_values = (key, value)
        else:
            prev_key, prev_value = self.key_values
            # (B, H, L_old, D) -> (B, H, L_old + L_new, D)
            curr_key = torch.cat([prev_key, key], dim=-2)
            curr_value = torch.cat([prev_value, value], dim=-2)
            # update cache
            self.key_values = (curr_key, curr_value)

        # 2. update seq len offset
        self.offset += key.shape[-2]

        # 3. return concated key, val
        full_key, full_value = self.key_values
        return full_key, full_value, self.offset, mask