import torch
from torch import nn, Tensor
from parakeet.model.attention import ConformerAttention, PositionalEncoding
from parakeet.model.convolution import ConformerConvolution
from parakeet.model.subsample import ConvSubsampling
from parakeet.model.feedforward import ConformerFeedForward
from parakeet.model.naive_cache import ModelCache, StreamingState


class ConformerLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        hidden_size: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
        conv_kernel_size: int,
        use_bias: bool,
        stream: bool,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size

        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = ConformerFeedForward(
            hidden_size=hidden_size, d_ff=d_ff, dropout=dropout, use_bias=use_bias
        )
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ConformerConvolution(
            hidden_size=hidden_size, kernel_size=conv_kernel_size, stream=stream
        )
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = ConformerAttention(
            layer_idx=self.layer_idx,
            hidden_size=hidden_size,
            num_heads=num_heads,
            use_bias=use_bias,
        )
        self.norm_feed_forward2 = nn.LayerNorm(hidden_size)
        self.feed_forward2 = ConformerFeedForward(
            hidden_size=hidden_size,
            d_ff=d_ff,
            dropout=dropout,
            use_bias=use_bias,
        )
        self.norm_out = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: Tensor,
        pos_emb: Tensor,
        pad_mask: Tensor,
        attn_mask: Tensor,
        cache: ModelCache = None,
    ):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual += x * 0.5

        x = self.norm_self_att(residual)
        if cache:
            x = self.self_attn(x, pos_emb, attn_mask=attn_mask, cache=cache)
        else:
            x = self.self_attn(x, pos_emb, attn_mask=attn_mask)
        residual += x

        x = self.norm_conv(residual)
        if cache:
            conv_cache = cache.get_conv_cache(self.layer_idx)
            x, updated_conv_cache = self.conv(x, pad_mask=pad_mask, cache=conv_cache)
            cache.update_conv_cache(self.layer_idx, updated_conv_cache)
        else:
            x = self.conv(x, pad_mask=pad_mask)
        residual += x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual += x * 0.5

        x = self.norm_out(residual)
        return x


class ConformerEncoder(nn.Module):
    def __init__(
        self,
        feat_in: int,
        hidden_size,
        ff_expansion_factor: int,
        num_layers: int,
        num_heads: int,
        subsampling_factor: int,
        pos_emb_max_len: int,
        conv_kernel_size: int,
        att_context_size,
        subsampling_conv_channels: int,
        use_bias: bool,
        stream: bool = True,
        chunk_size: list | tuple = (9, 16),
        shift_size: list | tuple = (9, 16),
        cache_drop_size: int = 0,
        last_channel_cache_size: int = 70,
        valid_out_len: int = 2,
        pre_encode_cache_size: list | tuple = (0, 9),
        drop_extra_pre_encoded: int = 2,
        last_channel_num: int = 0,
        last_time_num: int = 0,
    ):
        super().__init__()

        d_ff = hidden_size * ff_expansion_factor
        self.pre_encode = ConvSubsampling(
            hidden_size=hidden_size,
            subsampling_factor=subsampling_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            stride=2,
            kernel_size=3,
            feat_in=feat_in,
        )

        self.pos_enc = PositionalEncoding(
            hidden_size=hidden_size,
            dropout_rate_emb=0.0,
            max_len=pos_emb_max_len,
            dropout=0.0,
        )
        self.pos_enc.extend_pe(pos_emb_max_len, device=next(self.parameters()).device)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ConformerLayer(
                layer_idx=i,
                hidden_size=hidden_size,
                d_ff=d_ff,
                num_heads=num_heads,
                dropout=0.0,
                conv_kernel_size=conv_kernel_size,
                use_bias=use_bias,
                stream=stream,
            )
            self.layers.append(layer)

        self.att_context_size = att_context_size
        self.stream = stream
        self.chunk_size = chunk_size
        self.shift_size = shift_size
        self.cache_drop_size = cache_drop_size
        self.last_channel_cache_size = last_channel_cache_size
        self.valid_out_len = valid_out_len
        self.pre_encode_cache_size = pre_encode_cache_size
        self.drop_extra_pre_encoded = drop_extra_pre_encoded
        self.last_channel_num = last_channel_num
        self.last_time_num = last_time_num

    def init_streaming_state(
        self, batch_size: int = 1, device: torch.device | None = None
    ):
        if self.att_context_size[1] < 0:
            raise ValueError(
                "Right context must be non-negative for chunked_limited streaming."
            )
        if device is None:
            device = next(self.parameters()).device
        cache_len = self.att_context_size[0]
        if cache_len < 0:
            raise ValueError(
                "Unlimited left context is not supported for streaming caches."
            )
        cache = ModelCache(num_layers=len(self.layers), L_attn=cache_len)
        return StreamingState(cache=cache, processed_frames=0)

    def _create_streaming_masks(
        self,
        current_len: int,
        cache_len: int,
        processed_frames: int,
        device: torch.device,
        lengths: Tensor | None = None,
    ):
        chunk_size = self.att_context_size[1] + 1
        if chunk_size <= 0:
            raise ValueError(
                "Right context must be non-negative for chunked_limited streaming."
            )

        if self.att_context_size[0] >= 0:
            left_chunks_num = self.att_context_size[0] // chunk_size
        else:
            left_chunks_num = 10000

        key_start = max(processed_frames - cache_len, 0)
        key_len = cache_len + current_len
        key_pos = torch.arange(
            key_start, key_start + key_len, device=device, dtype=torch.int
        )
        query_pos = key_pos[-current_len:]

        chunk_idx_k = torch.div(key_pos, chunk_size, rounding_mode="trunc")
        chunk_idx_q = torch.div(query_pos, chunk_size, rounding_mode="trunc")
        diff_chunks = chunk_idx_q.unsqueeze(1) - chunk_idx_k.unsqueeze(0)
        allowed = torch.logical_and(
            torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
        )

        if lengths is None:
            batch_size = 1
            pad_mask = torch.zeros(
                batch_size, current_len, device=device, dtype=torch.bool
            )
            att_mask = ~allowed.unsqueeze(0)
            return pad_mask, att_mask

        batch_size = lengths.size(0)
        valid_current = torch.arange(current_len, device=device).expand(
            batch_size, -1
        ) < lengths.unsqueeze(-1)
        pad_mask = ~valid_current

        valid_total = torch.cat(
            [
                torch.ones(batch_size, cache_len, device=device, dtype=torch.bool),
                valid_current,
            ],
            dim=1,
        )
        valid_for_att = valid_current.unsqueeze(2) & valid_total.unsqueeze(1)

        allowed = allowed.unsqueeze(0).expand(batch_size, -1, -1)
        allowed = torch.logical_and(valid_for_att, allowed)
        att_mask = ~allowed
        return pad_mask, att_mask

    def forward_streaming(
        self,
        x: Tensor,
        state: StreamingState,
        length: Tensor | None = None,
        bypass_pre_encode: bool = False,
    ):
        if not self.stream:
            raise ValueError("forward_streaming requires encoder.stream=True.")
        if not bypass_pre_encode:
            if length is None:
                length = x.new_full(
                    (x.size(0),), x.size(-1), dtype=torch.int64, device=x.device
                )
            x = x.transpose(1, 2)
            x, length = self.pre_encode(x, length)
        else:
            if length is None:
                length = x.new_full(
                    (x.size(0),), x.size(1), dtype=torch.int64, device=x.device
                )

        cache_len = state.attn_cache_len()
        cache_len = min(cache_len, state.processed_frames)

        x, pos_emb = self.pos_enc(x, cache_len=cache_len)
        pad_mask, att_mask = self._create_streaming_masks(
            current_len=x.size(1),
            cache_len=cache_len,
            processed_frames=state.processed_frames,
            device=x.device,
            lengths=length,
        )

        for layer in self.layers:
            x = layer(
                x,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                attn_mask=att_mask,
                cache=state.cache,
            )

        state.processed_frames += x.size(1)
        x = x.transpose(1, 2)
        return x, state

    def forward(self, x: Tensor, length: Tensor = None, cache: ModelCache = None):
        raise RuntimeError(
            "Offline encoder forward is removed. Use forward_streaming instead."
        )


if __name__ == "__main__":
    encoder = ConformerEncoder(
        feat_in=128,
        hidden_size=512,
        ff_expansion_factor=4,
        num_layers=17,
        num_heads=8,
        subsampling_factor=8,
        pos_emb_max_len=5000,
        conv_kernel_size=9,
        att_context_size=[70, 1],
        subsampling_conv_channels=256,
        use_bias=False,
    ).eval()
    x = torch.randn(1, 128, 201)
    lengths = torch.LongTensor([201])
    output = encoder(x, lengths)
    print(output)
