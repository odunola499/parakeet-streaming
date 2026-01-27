from __future__ import annotations

from dataclasses import dataclass

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
class PagedAttentionStats:
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
    def _paged_attention_kernel(
        q_ptr,
        k_pages_ptr,
        v_pages_ptr,
        slot_pages_ptr,
        start_ptr,
        bias_ptr,
        out_ptr,
        K,
        D,
        embed_dim,
        attn_len,
        page_size,
        stride_q0,
        stride_q1,
        stride_q2,
        stride_q3,
        stride_k0,
        stride_k1,
        stride_k2,
        stride_k3,
        stride_v0,
        stride_v1,
        stride_v2,
        stride_v3,
        stride_sp0,
        stride_sp1,
        stride_start0,
        stride_b0,
        stride_b1,
        stride_b2,
        stride_b3,
        stride_o0,
        stride_o1,
        stride_o2,
        stride_o3,
        layer_idx,
        scale,
        BLOCK_K: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_h = tl.program_id(1)
        pid_t = tl.program_id(2)

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        q_ptrs = (
            q_ptr
            + pid_b * stride_q0
            + pid_t * stride_q1
            + pid_h * stride_q2
            + offs_d * stride_q3
        )
        q = tl.load(q_ptrs, mask=mask_d, other=0).to(tl.float32)

        head_offset = pid_h * D
        start = tl.load(start_ptr + pid_b * stride_start0)

        m = tl.full([1], -float("inf"), dtype=tl.float32)
        l = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([BLOCK_D], dtype=tl.float32)

        for k_block in range(0, K, BLOCK_K):
            offs_k = k_block + tl.arange(0, BLOCK_K)
            mask_k = offs_k < K

            bias_ptrs = (
                bias_ptr
                + pid_b * stride_b0
                + pid_h * stride_b1
                + pid_t * stride_b2
                + offs_k * stride_b3
            )
            bias = tl.load(bias_ptrs, mask=mask_k, other=-float("inf")).to(tl.float32)

            phys = (start + offs_k) % attn_len
            page = phys // page_size
            offset = phys - page * page_size
            page_ids = tl.load(
                slot_pages_ptr + pid_b * stride_sp0 + page * stride_sp1,
                mask=mask_k,
                other=0,
            )

            k_ptrs = (
                k_pages_ptr
                + page_ids[:, None] * stride_k0
                + layer_idx * stride_k1
                + offset[:, None] * stride_k2
                + (head_offset + offs_d[None]) * stride_k3
            )
            k = tl.load(
                k_ptrs,
                mask=mask_k[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.float32)

            scores = tl.sum(k * q[None, :], axis=1) * scale + bias
            scores = tl.where(mask_k, scores, -float("inf"))
            block_max = tl.max(scores, axis=0)

            m_new = tl.maximum(m, block_max)
            m_new = tl.where(m_new == -float("inf"), 0.0, m_new)
            exp_m = tl.where(m == -float("inf"), 0.0, tl.exp(m - m_new))

            exp_scores = tl.exp(scores - m_new)
            exp_scores = tl.where(mask_k, exp_scores, 0.0)

            l = l * exp_m + tl.sum(exp_scores, axis=0)

            v_ptrs = (
                v_pages_ptr
                + page_ids[:, None] * stride_v0
                + layer_idx * stride_v1
                + offset[:, None] * stride_v2
                + (head_offset + offs_d[None]) * stride_v3
            )
            v = tl.load(
                v_ptrs,
                mask=mask_k[:, None] & mask_d[None, :],
                other=0,
            ).to(tl.float32)
            acc = acc * exp_m + tl.sum(v * exp_scores[:, None], axis=0)
            m = m_new

        l = tl.where(l > 0, l, 1.0)
        out = acc / l

        out_ptrs = (
            out_ptr
            + pid_b * stride_o0
            + pid_t * stride_o1
            + pid_h * stride_o2
            + offs_d * stride_o3
        )
        tl.store(out_ptrs, out, mask=mask_d)

else:

    def _paged_attention_kernel(*args, **kwargs):
        raise RuntimeError("Triton is not available.")


def paged_attention_triton(
    q: torch.Tensor,
    k_pages: torch.Tensor,
    v_pages: torch.Tensor,
    slot_pages: torch.Tensor,
    start: torch.Tensor,
    bias: torch.Tensor,
    layer_idx: int,
    scale: float,
) -> tuple[torch.Tensor, PagedAttentionStats]:
    if not _is_triton_ready(q, k_pages, v_pages, slot_pages, start, bias):
        return q, PagedAttentionStats(False, "tensor_not_ready")
    if slot_pages.dtype != torch.int32:
        slot_pages = slot_pages.to(dtype=torch.int32)
    if start.dtype != torch.int32:
        start = start.to(dtype=torch.int32)
    B, T, H, D = q.shape
    K = bias.size(-1)
    attn_len = k_pages.size(2) * slot_pages.size(1)
    out = torch.empty_like(q)
    block_k = 64
    block_d = 128
    grid = (B, H, T)
    _paged_attention_kernel[grid](
        q,
        k_pages,
        v_pages,
        slot_pages,
        start,
        bias,
        out,
        K,
        D,
        k_pages.size(-1),
        attn_len,
        k_pages.size(2),
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k_pages.stride(0),
        k_pages.stride(1),
        k_pages.stride(2),
        k_pages.stride(3),
        v_pages.stride(0),
        v_pages.stride(1),
        v_pages.stride(2),
        v_pages.stride(3),
        slot_pages.stride(0),
        slot_pages.stride(1),
        start.stride(0),
        bias.stride(0),
        bias.stride(1),
        bias.stride(2),
        bias.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        layer_idx,
        scale,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out, PagedAttentionStats(True)


def paged_attention_status(device: torch.device | str) -> tuple[bool, str]:
    if not _TRITON_AVAILABLE:
        return False, "triton_not_available"
    dev = torch.device(device)
    if dev.type != "cuda":
        return False, "device_not_cuda"
    if not torch.cuda.is_available():
        return False, "cuda_unavailable"
    return True, "triton_ready"
