import math

import torch
from torch import Tensor, nn

from parakeet.modules.cache import ModelCache


def sdpa_attention_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor,
    dropout: float = 0.0,
):
    attn_output = nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attention_mask, dropout_p=dropout, scale=None
    )
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output


class PositionalEncoding(nn.Module):
    def __init__(
        self, hidden_size: int, dropout: float, max_len: int, dropout_rate_emb: float
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.dropout_emb = nn.Dropout(dropout_rate_emb)

    def create_pe(self, positions):
        pos_length = positions.size(0)
        pe = torch.zeros(pos_length, self.hidden_size, device=positions.device)
        div_term = torch.exp(
            (
                torch.arange(
                    0, self.hidden_size, 2, dtype=torch.float32, device=positions.device
                )
                * -(math.log(10000.0) / self.hidden_size)
            )
        )
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0).to(positions.device)
        self.register_buffer("pe", pe, persistent=False)

    def extend_pe(self, length, device):
        positions = torch.arange(
            length - 1, -length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        self.create_pe(positions=positions)

    def forward(self, x: Tensor, cache_len=0):
        input_len = x.size(1) + cache_len
        center_pos = self.pe.size(1) // 2 + 1
        start_pos = center_pos - input_len
        end_pos = center_pos + input_len - 1
        pos_emb = self.pe[:, start_pos:end_pos]
        pos_emb = self.dropout_emb(pos_emb)
        return self.dropout(x), pos_emb


class ConformerAttention(nn.Module):
    def __init__(self, layer_idx, num_heads, hidden_size: int, use_bias: bool = False):
        super().__init__()
        self.layer_idx = layer_idx
        self.d_k = hidden_size // num_heads
        self.s_d_k = math.sqrt(self.d_k)
        self.num_heads = num_heads

        self.linear_out = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.linear_pos = nn.Linear(hidden_size, hidden_size, bias=use_bias)
        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=use_bias)
        self.pos_bias_u = nn.Parameter(torch.FloatTensor(self.num_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.FloatTensor(self.num_heads, self.d_k))

    def rel_shift(self, x: Tensor):
        B, H, qlen, pos_len = x.size()
        x = nn.functional.pad(x, pad=(1, 0))
        x = x.view(B, H, -1, qlen)[:, :, 1:].view(B, H, qlen, pos_len)
        return x

    def forward(
        self, x: Tensor, pos_emb: Tensor, attn_mask: Tensor, cache: ModelCache = None
    ):
        B, T = x.shape[:2]
        pos_emb_B = pos_emb.shape[0]
        attn_mask = attn_mask.bool()
        if attn_mask.ndim == 3:
            attn_mask = attn_mask.unsqueeze(1)

        qkv = self.qkv(x)
        query, key, value = qkv.chunk(3, dim=-1)
        if cache:
            key, value = cache.update_attn_cache(self.layer_idx, (key, value))

        query = query.view(B, -1, self.num_heads, self.d_k)
        key = key.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        pos = (
            self.linear_pos(pos_emb)
            .view(pos_emb_B, -1, self.num_heads, self.d_k)
            .transpose(1, 2)
        )

        q_with_bias_u = (query + self.pos_bias_u).transpose(1, 2)
        q_with_bias_v = (query + self.pos_bias_v).transpose(1, 2)

        matrix_bd = torch.matmul(q_with_bias_v, pos.transpose(-2, -1))
        matrix_bd = self.rel_shift(matrix_bd)
        matrix_bd = matrix_bd[:, :, :, : key.size(-2)] * (1 / self.s_d_k)

        matrix_bd = matrix_bd.masked_fill(attn_mask, -1e5)

        output = sdpa_attention_forward(
            query=q_with_bias_u, key=key, value=value, attention_mask=matrix_bd
        )

        all_masked_rows = torch.all(attn_mask, dim=-1).unsqueeze(-1).unsqueeze(-1)
        if all_masked_rows.ndim == 3:
            all_masked_rows = all_masked_rows.squeeze(1)
        output = output.masked_fill(all_masked_rows, 0.0)

        output = output.reshape(B, -1, self.num_heads * self.d_k)
        output = self.linear_out(output)
        return output


__all__ = [
    "ConformerAttention",
    "PositionalEncoding",
    "sdpa_attention_forward",
]
