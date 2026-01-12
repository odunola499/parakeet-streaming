from dataclasses import dataclass

from torch import Tensor


class AttnCacheLayer:
    def __init__(self, layer_idx: int, left_attn: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.left_attn = left_attn

        self.k_cache = None
        self.v_cache = None

    def update(self, state: tuple):
        k, v = state
        B, T, D = k.shape

        if self.k_cache is None:
            self.k_cache = k.new_zeros((B, self.left_attn + T, D))
            self.v_cache = v.new_zeros((B, self.left_attn + T, D))

        self.k_cache[:, : self.left_attn].copy_(
            self.k_cache[:, -self.left_attn :].clone()
        )
        self.v_cache[:, : self.left_attn].copy_(
            self.v_cache[:, -self.left_attn :].clone()
        )

        self.k_cache[:, self.left_attn :].copy_(k)
        self.v_cache[:, self.left_attn :].copy_(v)

        full_k = self.k_cache
        full_v = self.v_cache

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
    def __init__(self, num_layers: int = 17, left_attn: int = 70):
        super().__init__()
        self.left_attn = left_attn

        self.attn_caches = {
            i: AttnCacheLayer(i, left_attn=left_attn) for i in range(num_layers)
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
    processed_frames: Tensor | int = 0
    cache_lengths: Tensor | int = 0

    def attn_cache_len(self) -> int:
        if isinstance(self.cache_lengths, Tensor):
            return int(self.cache_lengths.max().item())
        return int(self.cache_lengths)


__all__ = [
    "AttnCacheLayer",
    "ConvCacheLayer",
    "ModelCache",
    "StreamingState",
]
