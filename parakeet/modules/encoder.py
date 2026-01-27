import torch
from torch import Tensor, nn

from parakeet.modules.attention import ConformerAttention, PositionalEncoding
from parakeet.modules.convolution import ConformerConvolution
from parakeet.modules.feedforward import ConformerFeedForward
from parakeet.modules.subsample import ConvSubsampling
from parakeet.modules.cache import ModelCache, PagedModelCache, StreamingState


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
            hidden_size=hidden_size, kernel_size=conv_kernel_size
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
        cache: ModelCache,
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
        conv_cache = cache.get_conv_cache(self.layer_idx)
        x, updated_conv_cache = self.conv(x, pad_mask=pad_mask, cache=conv_cache)
        cache.update_conv_cache(self.layer_idx, updated_conv_cache)
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
        pre_encode_cache_size: list | tuple = (0, 9),
        drop_extra_pre_encoded: int = 2,
    ):
        super().__init__()

        d_ff = hidden_size * ff_expansion_factor
        self.embed_dim = hidden_size
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
            )
            self.layers.append(layer)

        self.att_context_size: list = att_context_size
        self.conv_kernel_size = conv_kernel_size
        self.pre_encode_cache_size = pre_encode_cache_size
        self.drop_extra_pre_encoded = drop_extra_pre_encoded

    def init_streaming_state(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        paged_kv_cache: bool = False,
        paged_kv_page_size: int = 16,
        paged_kv_max_pages: int | None = None,
    ):
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        if paged_kv_cache:
            cache = PagedModelCache(
                num_layers=len(self.layers),
                attn_context_size=self.att_context_size,
                embed_dim=self.embed_dim,
                conv_kernel_size=self.conv_kernel_size,
                batch_size=batch_size,
                page_size=paged_kv_page_size,
                max_pages=paged_kv_max_pages,
                device=device,
                dtype=dtype,
            )
        else:
            cache = ModelCache(
                num_layers=len(self.layers),
                attn_context_size=self.att_context_size,
                embed_dim=self.embed_dim,
                conv_kernel_size=self.conv_kernel_size,
                batch_size=batch_size,
                device=device,
                dtype=dtype,
            )
        processed_frames = torch.zeros(batch_size, dtype=torch.int64, device=device)
        cache_lengths = torch.zeros(batch_size, dtype=torch.int64, device=device)
        return StreamingState(
            cache=cache,
            processed_frames=processed_frames,
            cache_lengths=cache_lengths,
        )

    def _create_streaming_masks(
        self,
        current_len: int,
        cache_lengths: Tensor,
        processed_frames: Tensor,
        device: torch.device,
        lengths: Tensor,
    ):
        chunk_size = self.att_context_size[1] + 1
        max_cache_len = self.att_context_size[0]
        left_chunks_num = self.att_context_size[0] // chunk_size
        batch_size = cache_lengths.shape[0]

        key_len_max = max_cache_len + current_len
        pad_mask = torch.zeros(batch_size, current_len, device=device, dtype=torch.bool)
        att_mask = torch.ones(
            batch_size, current_len, key_len_max, device=device, dtype=torch.bool
        )

        for idx in range(batch_size):
            cache_len = int(cache_lengths[idx].item())
            proc = int(processed_frames[idx].item())
            key_start = max(proc - cache_len, 0)
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

            valid_current = torch.arange(current_len, device=device) < lengths[idx]
            pad_mask[idx] = ~valid_current

            valid_total = torch.zeros(
                cache_len + current_len, device=device, dtype=torch.bool
            )
            if cache_len:
                valid_total[:cache_len] = True
            valid_total[cache_len : cache_len + current_len] = valid_current

            offset = max_cache_len - cache_len
            allowed_full = torch.zeros(
                current_len, key_len_max, device=device, dtype=torch.bool
            )
            allowed_full[:, offset : offset + key_len] = allowed

            valid_total_full = torch.zeros(key_len_max, device=device, dtype=torch.bool)
            valid_total_full[offset : offset + key_len] = valid_total

            valid_for_att = valid_current.unsqueeze(1) & valid_total_full.unsqueeze(0)
            allowed_full = torch.logical_and(allowed_full, valid_for_att)
            att_mask[idx] = ~allowed_full

        return pad_mask, att_mask

    def forward(
        self,
        x: Tensor,
        state: StreamingState,
        length: Tensor | None = None,
        bypass_pre_encode: bool = False,
    ):

        cache_lengths = state.cache_lengths

        x, pos_emb = self.pos_enc(x, cache_len=self.att_context_size[0])
        pad_mask, att_mask = self._create_streaming_masks(
            current_len=x.size(1),
            cache_lengths=cache_lengths,
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

        if isinstance(length, Tensor):
            state.processed_frames.add_(length)
            state.cache_lengths.add_(length)
            state.cache_lengths.clamp_(max=self.att_context_size[0])
        else:
            state.processed_frames += x.size(1)
            state.cache_lengths = min(
                int(state.cache_lengths) + x.size(1), self.att_context_size[0]
            )
        x = x.transpose(1, 2)
        return x, state

    def forward_with_masks(
        self,
        x: Tensor,
        state: StreamingState,
        pad_mask: Tensor,
        attn_mask: Tensor,
        length: Tensor | None = None,
    ):
        x, pos_emb = self.pos_enc(x, cache_len=self.att_context_size[0])

        for layer in self.layers:
            x = layer(
                x,
                pos_emb=pos_emb,
                pad_mask=pad_mask,
                attn_mask=attn_mask,
                cache=state.cache,
            )

        if isinstance(length, Tensor):
            state.processed_frames.add_(length)
            state.cache_lengths.add_(length)
            state.cache_lengths.clamp_(max=self.att_context_size[0])
        else:
            state.processed_frames += x.size(1)
            state.cache_lengths = min(
                int(state.cache_lengths) + x.size(1), self.att_context_size[0]
            )
        x = x.transpose(1, 2)
        return x, state
