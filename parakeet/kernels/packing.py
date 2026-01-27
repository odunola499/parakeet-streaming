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
class PackStats:
    used_triton: bool
    reason: str | None = None


def _pick_block_size(num_cols: int) -> int:
    if num_cols >= 1024:
        return 1024
    if num_cols >= 512:
        return 512
    return 256


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
    def _gather_rows_kernel(
        src_ptr,
        dst_ptr,
        idx_ptr,
        n_cols,
        src_stride0,
        dst_stride0,
        BLOCK: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)
        offs = pid_col * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_cols

        row_idx = tl.load(idx_ptr + pid_row).to(tl.int64)
        src = src_ptr + row_idx * src_stride0 + offs
        dst = dst_ptr + pid_row * dst_stride0 + offs

        tl.store(dst, tl.load(src, mask=mask, other=0), mask=mask)

    @triton.jit
    def _scatter_rows_kernel(
        src_ptr,
        dst_ptr,
        idx_ptr,
        n_cols,
        src_stride0,
        dst_stride0,
        BLOCK: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)
        offs = pid_col * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_cols

        row_idx = tl.load(idx_ptr + pid_row).to(tl.int64)
        src = src_ptr + pid_row * src_stride0 + offs
        dst = dst_ptr + row_idx * dst_stride0 + offs

        tl.store(dst, tl.load(src, mask=mask, other=0), mask=mask)

else:

    def _gather_rows_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available.")

    def _scatter_rows_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available.")


def _gather_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    n_rows = dst.shape[0]
    n_cols = dst.shape[1]
    block = _pick_block_size(n_cols)
    grid = (n_rows, triton.cdiv(n_cols, block))
    _gather_rows_kernel[grid](
        src,
        dst,
        idx,
        n_cols,
        src.stride(0),
        dst.stride(0),
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )


def _scatter_rows(
    src: torch.Tensor,
    dst: torch.Tensor,
    idx: torch.Tensor,
) -> None:
    n_rows = src.shape[0]
    n_cols = src.shape[1]
    block = _pick_block_size(n_cols)
    grid = (n_rows, triton.cdiv(n_cols, block))
    _scatter_rows_kernel[grid](
        src,
        dst,
        idx,
        n_cols,
        src.stride(0),
        dst.stride(0),
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )


def _flatten_for_pack(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() <= 1:
        raise ValueError("Expected tensor with rank >= 2.")
    if not tensor.is_contiguous():
        return tensor.contiguous().view(tensor.size(0), -1)
    return tensor.view(tensor.size(0), -1)


def pack_states_triton(
    pool_state: Any, batch_state: Any, slot_index: torch.Tensor
) -> PackStats:
    if not _TRITON_AVAILABLE:
        return PackStats(False, "triton_unavailable")

    attn_k = pool_state.cache.attn_k
    attn_v = pool_state.cache.attn_v
    conv = pool_state.cache.conv
    out_attn_k = batch_state.cache.attn_k
    out_attn_v = batch_state.cache.attn_v
    out_conv = batch_state.cache.conv

    if not _is_triton_ready(
        attn_k, attn_v, conv, out_attn_k, out_attn_v, out_conv, slot_index
    ):
        return PackStats(False, "tensor_not_ready")

    if slot_index.dtype != torch.int32:
        slot_index = slot_index.to(dtype=torch.int32)

    src_k = _flatten_for_pack(attn_k)
    src_v = _flatten_for_pack(attn_v)
    src_conv = _flatten_for_pack(conv)
    dst_k = _flatten_for_pack(out_attn_k)
    dst_v = _flatten_for_pack(out_attn_v)
    dst_conv = _flatten_for_pack(out_conv)

    _gather_rows(src_k, dst_k, slot_index)
    _gather_rows(src_v, dst_v, slot_index)
    _gather_rows(src_conv, dst_conv, slot_index)

    return PackStats(True)


def unpack_states_triton(
    pool_state: Any, batch_state: Any, slot_index: torch.Tensor
) -> PackStats:
    if not _TRITON_AVAILABLE:
        return PackStats(False, "triton_unavailable")

    attn_k = pool_state.cache.attn_k
    attn_v = pool_state.cache.attn_v
    conv = pool_state.cache.conv
    in_attn_k = batch_state.cache.attn_k
    in_attn_v = batch_state.cache.attn_v
    in_conv = batch_state.cache.conv

    if not _is_triton_ready(
        attn_k, attn_v, conv, in_attn_k, in_attn_v, in_conv, slot_index
    ):
        return PackStats(False, "tensor_not_ready")

    if slot_index.dtype != torch.int32:
        slot_index = slot_index.to(dtype=torch.int32)

    dst_k = _flatten_for_pack(attn_k)
    dst_v = _flatten_for_pack(attn_v)
    dst_conv = _flatten_for_pack(conv)
    src_k = _flatten_for_pack(in_attn_k)
    src_v = _flatten_for_pack(in_attn_v)
    src_conv = _flatten_for_pack(in_conv)

    _scatter_rows(src_k, dst_k, slot_index)
    _scatter_rows(src_v, dst_v, slot_index)
    _scatter_rows(src_conv, dst_conv, slot_index)

    return PackStats(True)


def triton_pack_status(device: torch.device | str) -> tuple[bool, str]:
    if not _TRITON_AVAILABLE:
        return False, "triton_not_available"
    dev = torch.device(device)
    if dev.type != "cuda":
        return False, "device_not_cuda"
    if not torch.cuda.is_available():
        return False, "cuda_unavailable"
    return True, "triton_ready"
