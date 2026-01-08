from dataclasses import dataclass

import torch
from torch import Tensor


class AttnCacheLayer:
    def __init__(self, layer_idx: int, L_attn: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.L_attn = L_attn

        self.k_cache = None
        self.v_cache = None

    def update(self, state: tuple, max_cache_len: int | None = None):
        k, v = state
        if self.k_cache is None:
            self.k_cache = k.new_zeros((k.size(0), self.L_attn, k.size(-1)))
            self.v_cache = v.new_zeros((v.size(0), self.L_attn, v.size(-1)))

        max_cache_len = self.L_attn if max_cache_len is None else int(max_cache_len)
        if max_cache_len > 0:
            cache_k = self.k_cache[:, -max_cache_len:, :]
            cache_v = self.v_cache[:, -max_cache_len:, :]
        else:
            cache_k = self.k_cache[:, :0, :]
            cache_v = self.v_cache[:, :0, :]

        full_k = torch.concat([cache_k, k], dim=1)
        full_v = torch.concat([cache_v, v], dim=1)

        if self.L_attn > 0:
            updated_k = torch.concat([self.k_cache, k], dim=1)
            updated_v = torch.concat([self.v_cache, v], dim=1)
            self.k_cache = updated_k[:, -self.L_attn :, :]
            self.v_cache = updated_v[:, -self.L_attn :, :]
        else:
            self.k_cache = torch.concat([self.k_cache, k], dim=1)
            self.v_cache = torch.concat([self.v_cache, v], dim=1)

        return full_k, full_v


class ConvCacheLayer:
    def __init__(self, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.cache = None

    def get(self):
        return self.cache

    def update(self, state: Tensor):
        self.cache = state


class ModelCache:
    def __init__(self, num_layers: int = 17, L_attn: int = 70):
        super().__init__()
        self.L_attn = L_attn
        self.current_max_len = 0
        self.attn_caches = {
            i: AttnCacheLayer(i, L_attn=L_attn) for i in range(num_layers)
        }
        self.conv_caches = {i: ConvCacheLayer(i) for i in range(num_layers)}

    def get_conv_cache(self, layer_idx):
        return self.conv_caches[layer_idx].get()

    def update_conv_cache(self, layer_idx, state):
        self.conv_caches[layer_idx].update(state)

    def update_attn_cache(self, layer_idx, layer_cache):
        return self.attn_caches[layer_idx].update(
            layer_cache, max_cache_len=self.current_max_len
        )

    def reset(self):
        for layer in self.attn_caches.values():
            layer.k_cache = None
            layer.v_cache = None
        for layer in self.conv_caches.values():
            layer.cache = None
        self.current_max_len = 0


@dataclass
class StreamingState:
    cache: ModelCache
    processed_frames: Tensor | int = 0
    cache_lengths: Tensor | int = 0

    def attn_cache_len(self) -> int:
        if isinstance(self.cache_lengths, Tensor):
            return int(self.cache_lengths.max().item())
        return int(self.cache_lengths)


__all__ = ["AttnCacheLayer", "ConvCacheLayer", "ModelCache", "StreamingState"]
