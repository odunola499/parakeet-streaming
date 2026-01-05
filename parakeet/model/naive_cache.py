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

    def update(self, state: tuple):
        k, v = state
        if self.k_cache is None:
            full_k = k
            full_v = v
        else:
            full_k = torch.concat([self.k_cache, k], dim=1)
            full_v = torch.concat([self.v_cache, v], dim=1)

        if self.L_attn > 0:
            self.k_cache = full_k[:, -self.L_attn :, :]
            self.v_cache = full_v[:, -self.L_attn :, :]
        else:
            self.k_cache = full_k
            self.v_cache = full_v

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
        self.attn_caches = {
            i: AttnCacheLayer(i, L_attn=L_attn) for i in range(num_layers)
        }
        self.conv_caches = {i: ConvCacheLayer(i) for i in range(num_layers)}

    def get_conv_cache(self, layer_idx):
        return self.conv_caches[layer_idx].get()

    def update_conv_cache(self, layer_idx, state):
        self.conv_caches[layer_idx].update(state)

    def update_attn_cache(self, layer_idx, layer_cache):
        return self.attn_caches[layer_idx].update(layer_cache)

    def reset(self):
        for layer in self.attn_caches.values():
            layer.k_cache = None
            layer.v_cache = None
        for layer in self.conv_caches.values():
            layer.cache = None


@dataclass
class StreamingState:
    cache: ModelCache
    processed_frames: int = 0

    def attn_cache_len(self) -> int:
        first = next(iter(self.cache.attn_caches.values()))
        if first.k_cache is None:
            return 0
        return int(first.k_cache.size(1))
