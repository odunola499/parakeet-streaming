from dataclasses import dataclass

import torch
from torch import Tensor


class ModelCache:
    def __init__(
        self,
        num_layers: int,
        attn_context_size: list | tuple,
        embed_dim: int,
        conv_kernel_size: int,
        batch_size: int = 1,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.left_attn = int(attn_context_size[0])
        self.right_attn = int(attn_context_size[1])
        self.attn_context_size = (self.left_attn, self.right_attn)
        self.attn_cache_len = self.left_attn + self.right_attn + 1
        self.conv_cache_len = int(conv_kernel_size - 1)
        self.attn_k = torch.zeros(
            (batch_size, num_layers, self.attn_cache_len, embed_dim),
            device=device,
            dtype=dtype,
        )
        self.attn_v = torch.zeros_like(self.attn_k)
        self.conv = torch.zeros(
            (batch_size, num_layers, embed_dim, self.conv_cache_len),
            device=device,
            dtype=dtype,
        )

    def update_attn_cache(self, layer_idx: int, key: Tensor, value: Tensor):
        self.attn_k[:, layer_idx, : self.left_attn].copy_(
            self.attn_k[:, layer_idx, -self.left_attn :].clone()
        )
        self.attn_v[:, layer_idx, : self.left_attn].copy_(
            self.attn_v[:, layer_idx, -self.left_attn :].clone()
        )
        self.attn_k[:, layer_idx, self.left_attn :].copy_(key)
        self.attn_v[:, layer_idx, self.left_attn :].copy_(value)
        return self.attn_k[:, layer_idx], self.attn_v[:, layer_idx]

    def get_conv_cache(self, layer_idx: int) -> Tensor:
        return self.conv[:, layer_idx]

    def update_conv_cache(self, layer_idx: int, state: Tensor) -> None:
        if state is None:
            return
        self.conv[:, layer_idx].copy_(state)

    def reset(self) -> None:
        self.attn_k.zero_()
        self.attn_v.zero_()
        self.conv.zero_()

    def reset_slot(self, slot: int) -> None:
        self.attn_k[slot].zero_()
        self.attn_v[slot].zero_()
        self.conv[slot].zero_()

    def view(self, batch_size: int) -> "ModelCache":
        if batch_size == self.attn_k.size(0):
            return self
        view = ModelCache.__new__(ModelCache)
        view.left_attn = self.left_attn
        view.right_attn = self.right_attn
        view.attn_context_size = self.attn_context_size
        view.attn_cache_len = self.attn_cache_len
        view.conv_cache_len = self.conv_cache_len
        view.attn_k = self.attn_k[:batch_size]
        view.attn_v = self.attn_v[:batch_size]
        view.conv = self.conv[:batch_size]
        return view


@dataclass
class StreamingState:
    cache: ModelCache
    processed_frames: Tensor | int = 0
    cache_lengths: Tensor | int = 0

    def attn_cache_len(self) -> int:
        if isinstance(self.cache_lengths, Tensor):
            return int(self.cache_lengths.max().item())
        return int(self.cache_lengths)


__all__ = [
    "ModelCache",
    "StreamingState",
]
