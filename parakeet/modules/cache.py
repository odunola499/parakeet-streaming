from dataclasses import dataclass
from collections import deque
import math

import torch
from torch import Tensor

from parakeet.kernels.paged_kv import (
    paged_kv_gather_triton,
    paged_kv_scatter_triton,
    paged_kv_write_triton,
)


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


class PagedModelCache:
    def __init__(
        self,
        num_layers: int,
        attn_context_size: list | tuple,
        embed_dim: int,
        conv_kernel_size: int,
        batch_size: int,
        page_size: int,
        max_pages: int | None = None,
        device=None,
        dtype=None,
    ):
        self.left_attn = int(attn_context_size[0])
        self.right_attn = int(attn_context_size[1])
        self.attn_context_size = (self.left_attn, self.right_attn)
        self.attn_cache_len = self.left_attn + self.right_attn + 1
        self.conv_cache_len = int(conv_kernel_size - 1)
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.page_size = max(1, int(page_size))
        self.pages_per_stream = int(math.ceil(self.attn_cache_len / self.page_size))
        if max_pages is None:
            max_pages = batch_size * self.pages_per_stream
        self.max_pages = int(max_pages)
        if self.max_pages < self.pages_per_stream:
            raise ValueError("paged_kv_max_pages too small for a single stream")

        self.attn_k_pages = torch.zeros(
            (self.max_pages, num_layers, self.page_size, embed_dim),
            device=device,
            dtype=dtype,
        )
        self.attn_v_pages = torch.zeros_like(self.attn_k_pages)
        self.attn_start = torch.zeros(
            (batch_size, num_layers), device=device, dtype=torch.int32
        )
        self.slot_pages = torch.full(
            (batch_size, self.pages_per_stream),
            -1,
            device=device,
            dtype=torch.long,
        )
        self._free_pages: deque[int] = deque(range(self.max_pages))

        self.conv = torch.zeros(
            (batch_size, num_layers, embed_dim, self.conv_cache_len),
            device=device,
            dtype=dtype,
        )

    @property
    def is_paged(self) -> bool:
        return True

    def _allocate_pages_for_slot(self, slot: int) -> None:
        if int(self.slot_pages[slot, 0].item()) != -1:
            return
        if len(self._free_pages) < self.pages_per_stream:
            raise RuntimeError("Paged KV cache exhausted.")
        pages = [self._free_pages.popleft() for _ in range(self.pages_per_stream)]
        self.slot_pages[slot].copy_(
            torch.tensor(pages, device=self.slot_pages.device, dtype=self.slot_pages.dtype)
        )

    def reset_slot(self, slot: int) -> None:
        self._allocate_pages_for_slot(slot)
        page_ids = self.slot_pages[slot].detach().cpu().tolist()
        if page_ids:
            self.attn_k_pages[page_ids].zero_()
            self.attn_v_pages[page_ids].zero_()
        self.attn_start[slot].zero_()
        self.conv[slot].zero_()

    def release_slot(self, slot: int) -> None:
        page_ids = self.slot_pages[slot].detach().cpu().tolist()
        if page_ids and page_ids[0] != -1:
            for page_id in page_ids:
                self._free_pages.append(page_id)
        self.slot_pages[slot].fill_(-1)
        self.attn_start[slot].zero_()
        self.conv[slot].zero_()

    def get_conv_cache(self, layer_idx: int) -> Tensor:
        return self.conv[:, layer_idx]

    def update_conv_cache(self, layer_idx: int, state: Tensor) -> None:
        if state is None:
            return
        self.conv[:, layer_idx].copy_(state)

    def gather_kv(self, slot_index: Tensor) -> tuple[Tensor, Tensor]:
        pages = self.slot_pages.index_select(0, slot_index)
        starts = self.attn_start.index_select(0, slot_index)
        batch = pages.size(0)
        out = torch.empty(
            (batch, self.num_layers, self.attn_cache_len, self.embed_dim),
            device=self.attn_k_pages.device,
            dtype=self.attn_k_pages.dtype,
        )
        out_v = torch.empty_like(out)
        stats = paged_kv_gather_triton(self.attn_k_pages, out, pages, starts)
        stats_v = paged_kv_gather_triton(self.attn_v_pages, out_v, pages, starts)
        if not stats.used_triton or not stats_v.used_triton:
            for b in range(batch):
                for l in range(self.num_layers):
                    start = int(starts[b, l].item())
                    attn = torch.empty(
                        (self.attn_cache_len, self.embed_dim),
                        device=self.attn_k_pages.device,
                        dtype=self.attn_k_pages.dtype,
                    )
                    for idx in range(self.attn_cache_len):
                        phys = (start + idx) % self.attn_cache_len
                        page = phys // self.page_size
                        offset = phys % self.page_size
                        page_id = int(pages[b, page].item())
                        attn[idx] = self.attn_k_pages[page_id, l, offset]
                    out[b, l] = attn
            for b in range(batch):
                for l in range(self.num_layers):
                    start = int(starts[b, l].item())
                    attn = torch.empty(
                        (self.attn_cache_len, self.embed_dim),
                        device=self.attn_v_pages.device,
                        dtype=self.attn_v_pages.dtype,
                    )
                    for idx in range(self.attn_cache_len):
                        phys = (start + idx) % self.attn_cache_len
                        page = phys // self.page_size
                        offset = phys % self.page_size
                        page_id = int(pages[b, page].item())
                        attn[idx] = self.attn_v_pages[page_id, l, offset]
                    out_v[b, l] = attn
        return out, out_v

    def scatter_kv(self, slot_index: Tensor, attn_k: Tensor, attn_v: Tensor) -> None:
        pages = self.slot_pages.index_select(0, slot_index)
        starts = self.attn_start.index_select(0, slot_index)
        stats = paged_kv_scatter_triton(self.attn_k_pages, attn_k, pages, starts)
        stats_v = paged_kv_scatter_triton(self.attn_v_pages, attn_v, pages, starts)
        if stats.used_triton and stats_v.used_triton:
            return
        batch = pages.size(0)
        for b in range(batch):
            for l in range(self.num_layers):
                start = int(starts[b, l].item())
                for idx in range(self.attn_cache_len):
                    phys = (start + idx) % self.attn_cache_len
                    page = phys // self.page_size
                    offset = phys % self.page_size
                    page_id = int(pages[b, page].item())
                    self.attn_k_pages[page_id, l, offset] = attn_k[b, l, idx]
                    self.attn_v_pages[page_id, l, offset] = attn_v[b, l, idx]

    def update_attn_cache(
        self,
        layer_idx: int,
        slot_index: Tensor,
        slot_pages: Tensor,
        key: Tensor,
        value: Tensor,
    ) -> Tensor:
        starts_batch = self.attn_start.index_select(0, slot_index)
        starts = starts_batch[:, layer_idx]
        key_len = int(key.size(1))
        new_start = (starts + key_len) % self.attn_cache_len
        self.attn_start[slot_index, layer_idx] = new_start
        starts_batch = starts_batch.clone()
        starts_batch[:, layer_idx] = new_start
        stats_k = paged_kv_write_triton(
            self.attn_k_pages,
            key,
            slot_pages,
            starts_batch,
            layer_idx,
            self.left_attn,
            self.attn_cache_len,
        )
        stats_v = paged_kv_write_triton(
            self.attn_v_pages,
            value,
            slot_pages,
            starts_batch,
            layer_idx,
            self.left_attn,
            self.attn_cache_len,
        )
        if not stats_k.used_triton or not stats_v.used_triton:
            for b in range(slot_pages.size(0)):
                start = int(new_start[b].item())
                for t in range(key_len):
                    phys = (start + self.left_attn + t) % self.attn_cache_len
                    page = phys // self.page_size
                    offset = phys % self.page_size
                    page_id = int(slot_pages[b, page].item())
                    self.attn_k_pages[page_id, layer_idx, offset] = key[b, t]
                    self.attn_v_pages[page_id, layer_idx, offset] = value[b, t]
        return new_start.contiguous()


class PagedBatchCache:
    def __init__(
        self,
        pool: PagedModelCache,
        slot_index: Tensor,
        conv: Tensor,
    ):
        self.pool = pool
        self.slot_index = slot_index
        self.slot_pages = pool.slot_pages.index_select(0, slot_index)
        self.conv = conv
        self.is_paged = True
        self.use_paged_attention = True

    def update_attn_cache(self, layer_idx: int, key: Tensor, value: Tensor) -> Tensor:
        return self.pool.update_attn_cache(
            layer_idx, self.slot_index, self.slot_pages, key, value
        )

    def get_conv_cache(self, layer_idx: int) -> Tensor:
        return self.conv[:, layer_idx]

    def update_conv_cache(self, layer_idx: int, state: Tensor) -> None:
        if state is None:
            return
        self.conv[:, layer_idx].copy_(state)


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
    "PagedModelCache",
    "PagedBatchCache",
    "StreamingState",
]
