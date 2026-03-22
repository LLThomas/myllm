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

    def update_and_fetch(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask_length: Optional[int] = None,
        mask: Optional[Union[torch.Tensor, str]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
        pass

    def add_request(self, prefilled: "TinyKvCache", id: int):
        pass

    def remove_request(self, id: int):
        pass


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
        self.offset += key.shape[1]

        # 3. return concated key, val
        full_key, full_value = self.key_values
        return full_key, full_value, self.offset, mask