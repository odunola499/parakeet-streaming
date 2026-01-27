from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

try:
    import triton
    import triton.language as tl

    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False


@dataclass
class PagedKVStats:
    used_triton: bool
    reason: str | None = None


def _is_triton_ready(*tensors: torch.Tensor) -> bool:
    if not _TRITON_AVAILABLE:
        return False
    if not tensors:
        return False
    device = tensors[0].device
    if device.type != "cuda":
        return False
    for tensor in tensors:
        if tensor.device != device:
            return False
        if not tensor.is_contiguous():
            return False
        if tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.int32,
            torch.int64,
        ):
            return False
    return True


if _TRITON_AVAILABLE:

    @triton.jit
    def _paged_gather_kernel(
        pages_ptr,
        out_ptr,
        slot_pages_ptr,
        start_ptr,
        K,
        D,
        page_size,
        stride_p0,
        stride_p1,
        stride_p2,
        stride_p3,
        stride_o0,
        stride_o1,
        stride_o2,
        stride_o3,
        stride_sp0,
        stride_sp1,
        stride_start0,
        stride_start1,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        D_BLOCKS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_kd = tl.program_id(2)

        k_block = pid_kd // D_BLOCKS
        d_block = pid_kd - k_block * D_BLOCKS

        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_k = offs_k < K
        mask_d = offs_d < D

        start = tl.load(start_ptr + pid_b * stride_start0 + pid_l * stride_start1)
        phys = (start + offs_k) % K
        page = phys // page_size
        offset = phys - page * page_size

        page_ids = tl.load(
            slot_pages_ptr + pid_b * stride_sp0 + page * stride_sp1,
            mask=mask_k,
            other=0,
        )

        head_offset = tl.zeros([1], dtype=tl.int32)

        ptrs = (
            pages_ptr
            + page_ids[:, None] * stride_p0
            + pid_l * stride_p1
            + offset[:, None] * stride_p2
            + (head_offset + offs_d[None]) * stride_p3
        )
        vals = tl.load(
            ptrs,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0,
        )

        out_ptrs = (
            out_ptr
            + pid_b * stride_o0
            + pid_l * stride_o1
            + offs_k[:, None] * stride_o2
            + offs_d[None] * stride_o3
        )
        tl.store(
            out_ptrs,
            vals,
            mask=mask_k[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _paged_scatter_kernel(
        pages_ptr,
        src_ptr,
        slot_pages_ptr,
        start_ptr,
        K,
        D,
        page_size,
        stride_p0,
        stride_p1,
        stride_p2,
        stride_p3,
        stride_s0,
        stride_s1,
        stride_s2,
        stride_s3,
        stride_sp0,
        stride_sp1,
        stride_start0,
        stride_start1,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
        D_BLOCKS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_l = tl.program_id(1)
        pid_kd = tl.program_id(2)

        k_block = pid_kd // D_BLOCKS
        d_block = pid_kd - k_block * D_BLOCKS

        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_k = offs_k < K
        mask_d = offs_d < D

        start = tl.load(start_ptr + pid_b * stride_start0 + pid_l * stride_start1)
        phys = (start + offs_k) % K
        page = phys // page_size
        offset = phys - page * page_size

        page_ids = tl.load(
            slot_pages_ptr + pid_b * stride_sp0 + page * stride_sp1,
            mask=mask_k,
            other=0,
        )

        head_offset = tl.zeros([1], dtype=tl.int32)

        src_ptrs = (
            src_ptr
            + pid_b * stride_s0
            + pid_l * stride_s1
            + offs_k[:, None] * stride_s2
            + offs_d[None] * stride_s3
        )
        vals = tl.load(
            src_ptrs,
            mask=mask_k[:, None] & mask_d[None, :],
            other=0,
        )

        dst_ptrs = (
            pages_ptr
            + page_ids[:, None] * stride_p0
            + pid_l * stride_p1
            + offset[:, None] * stride_p2
            + (head_offset + offs_d[None]) * stride_p3
        )
        tl.store(
            dst_ptrs,
            vals,
            mask=mask_k[:, None] & mask_d[None, :],
        )

    @triton.jit
    def _paged_write_kernel(
        pages_ptr,
        src_ptr,
        slot_pages_ptr,
        start_ptr,
        attn_len,
        T,
        D,
        page_size,
        stride_p0,
        stride_p1,
        stride_p2,
        stride_p3,
        stride_s0,
        stride_s1,
        stride_s2,
        stride_sp0,
        stride_sp1,
        stride_start0,
        stride_start1,
        layer_idx,
        left_attn,
        BLOCK_T: tl.constexpr,
        BLOCK_D: tl.constexpr,
        D_BLOCKS: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_td = tl.program_id(1)

        t_block = pid_td // D_BLOCKS
        d_block = pid_td - t_block * D_BLOCKS

        offs_t = t_block * BLOCK_T + tl.arange(0, BLOCK_T)
        offs_d = d_block * BLOCK_D + tl.arange(0, BLOCK_D)

        mask_t = offs_t < T
        mask_d = offs_d < D

        start = tl.load(start_ptr + pid_b * stride_start0 + layer_idx * stride_start1)
        phys = (start + left_attn + offs_t) % attn_len
        page = phys // page_size
        offset = phys - page * page_size

        page_ids = tl.load(
            slot_pages_ptr + pid_b * stride_sp0 + page * stride_sp1,
            mask=mask_t,
            other=0,
        )

        src_ptrs = (
            src_ptr
            + pid_b * stride_s0
            + offs_t[:, None] * stride_s1
            + offs_d[None] * stride_s2
        )
        vals = tl.load(
            src_ptrs,
            mask=mask_t[:, None] & mask_d[None, :],
            other=0,
        )

        dst_ptrs = (
            pages_ptr
            + page_ids[:, None] * stride_p0
            + layer_idx * stride_p1
            + offset[:, None] * stride_p2
            + offs_d[None] * stride_p3
        )
        tl.store(
            dst_ptrs,
            vals,
            mask=mask_t[:, None] & mask_d[None, :],
        )

else:

    def _paged_gather_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available.")

    def _paged_scatter_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available.")

    def _paged_write_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available.")


def paged_kv_gather_triton(
    pages: torch.Tensor,
    out: torch.Tensor,
    slot_pages: torch.Tensor,
    start: torch.Tensor,
) -> PagedKVStats:
    if not _is_triton_ready(pages, out, slot_pages, start):
        return PagedKVStats(False, "tensor_not_ready")
    if slot_pages.dtype != torch.int32:
        slot_pages = slot_pages.to(dtype=torch.int32)
    if start.dtype != torch.int32:
        start = start.to(dtype=torch.int32)
    B, L, K, D = out.shape
    block_k = 32
    block_d = 128
    d_blocks = triton.cdiv(D, block_d)
    grid = (B, L, triton.cdiv(K, block_k) * d_blocks)
    _paged_gather_kernel[grid](
        pages,
        out,
        slot_pages,
        start,
        K,
        D,
        pages.size(2),
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        pages.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        slot_pages.stride(0),
        slot_pages.stride(1),
        start.stride(0),
        start.stride(1),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        D_BLOCKS=d_blocks,
        num_warps=4,
        num_stages=2,
    )
    return PagedKVStats(True)


def paged_kv_scatter_triton(
    pages: torch.Tensor,
    src: torch.Tensor,
    slot_pages: torch.Tensor,
    start: torch.Tensor,
) -> PagedKVStats:
    if not _is_triton_ready(pages, src, slot_pages, start):
        return PagedKVStats(False, "tensor_not_ready")
    if slot_pages.dtype != torch.int32:
        slot_pages = slot_pages.to(dtype=torch.int32)
    if start.dtype != torch.int32:
        start = start.to(dtype=torch.int32)
    B, L, K, D = src.shape
    block_k = 32
    block_d = 128
    d_blocks = triton.cdiv(D, block_d)
    grid = (B, L, triton.cdiv(K, block_k) * d_blocks)
    _paged_scatter_kernel[grid](
        pages,
        src,
        slot_pages,
        start,
        K,
        D,
        pages.size(2),
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        pages.stride(3),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        slot_pages.stride(0),
        slot_pages.stride(1),
        start.stride(0),
        start.stride(1),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        D_BLOCKS=d_blocks,
        num_warps=4,
        num_stages=2,
    )
    return PagedKVStats(True)


def paged_kv_write_triton(
    pages: torch.Tensor,
    src: torch.Tensor,
    slot_pages: torch.Tensor,
    start: torch.Tensor,
    layer_idx: int,
    left_attn: int,
    attn_len: int,
) -> PagedKVStats:
    if not _is_triton_ready(pages, src, slot_pages, start):
        return PagedKVStats(False, "tensor_not_ready")
    if slot_pages.dtype != torch.int32:
        slot_pages = slot_pages.to(dtype=torch.int32)
    if start.dtype != torch.int32:
        start = start.to(dtype=torch.int32)
    B, T, D = src.shape
    block_t = 16
    block_d = 128
    d_blocks = triton.cdiv(D, block_d)
    grid = (B, triton.cdiv(T, block_t) * d_blocks)
    _paged_write_kernel[grid](
        pages,
        src,
        slot_pages,
        start,
        attn_len,
        T,
        D,
        pages.size(2),
        pages.stride(0),
        pages.stride(1),
        pages.stride(2),
        pages.stride(3),
        src.stride(0),
        src.stride(1),
        src.stride(2),
        slot_pages.stride(0),
        slot_pages.stride(1),
        start.stride(0),
        start.stride(1),
        layer_idx,
        left_attn,
        BLOCK_T=block_t,
        BLOCK_D=block_d,
        D_BLOCKS=d_blocks,
        num_warps=4,
        num_stages=2,
    )
    return PagedKVStats(True)


def paged_kv_status(device: torch.device | str) -> tuple[bool, str]:
    if not _TRITON_AVAILABLE:
        return False, "triton_not_available"
    dev = torch.device(device)
    if dev.type != "cuda":
        return False, "device_not_cuda"
    if not torch.cuda.is_available():
        return False, "cuda_unavailable"
    return True, "triton_ready"
