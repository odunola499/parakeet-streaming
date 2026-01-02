import torch
from torch import nn, Tensor
from parakeet.model.attention import ConformerAttention, PositionalEncoding
from parakeet.model.config import ModelConfig
from parakeet.model.convolution import ConformerConvolution
from parakeet.model.subsample import ConvSubsampling
from parakeet.model.feedforward import ConformerFeedForward


class ConformerLayer(nn.Module):
    def __init__(
            self,
            hidden_size:int,
            d_ff:int,
            num_heads:int,
            dropout:float,
            conv_kernel_size:int,
            use_bias:bool,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.norm_feed_forward1 = nn.LayerNorm(hidden_size)
        self.feed_forward1 = ConformerFeedForward(
            hidden_size = hidden_size,
            d_ff = d_ff,
            dropout=dropout,
            use_bias=use_bias
        )
        self.norm_conv = nn.LayerNorm(hidden_size)
        self.conv = ConformerConvolution(
            hidden_size = hidden_size,
            kernel_size=conv_kernel_size
        )
        self.norm_self_att = nn.LayerNorm(hidden_size)
        self.self_attn = ConformerAttention(
            hidden_size = hidden_size,
            num_heads = num_heads,
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
            x:Tensor,
            pos_emb:Tensor,
            pad_mask:Tensor,
            attn_mask:Tensor,
            cache = None
    ):
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        #residual += self.dropout(x) * 0.5
        residual += x * 0.5

        x = self.norm_self_att(residual)
        x = self.self_attn(x, pos_emb, attn_mask=attn_mask)
        #residual += self.dropout(x)
        residual += x

        x = self.norm_conv(residual)
        if cache:
            x, _ = self.conv(x, pad_mask=pad_mask, cache=cache)
        else:
            x = self.conv(x, pad_mask=pad_mask)
        #residual += self.dropout(x)
        residual += x

        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        #residual += self.dropout(x) * 0.5
        residual += x * 0.5

        x = self.norm_out(residual)
        return x


class ConformerEncoder(nn.Module):
    def __init__(
            self,
            feat_in:int,
            hidden_size,
            ff_expansion_factor:int,
            num_layers:int,
            num_heads:int,
            subsampling_factor:int,
            pos_emb_max_len:int,
            conv_kernel_size:int,
            att_context_size,
            subsampling_conv_channels:int,
            use_bias:bool,
    ):
        super().__init__()

        d_ff = hidden_size * ff_expansion_factor
        self.pre_encode = ConvSubsampling(
            hidden_size = hidden_size,
            subsampling_factor=subsampling_factor,
            subsampling_conv_channels=subsampling_conv_channels,
            stride = 2,
            kernel_size = 3,
            feat_in = feat_in,
        )

        self.pos_enc = PositionalEncoding(
            hidden_size = hidden_size,
            dropout_rate_emb=0.0,
            max_len=pos_emb_max_len,
            dropout=0.0
        )
        self.pos_enc.extend_pe(
            pos_emb_max_len, device=next(self.parameters()).device
        )

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ConformerLayer(
                hidden_size = hidden_size,
                d_ff = d_ff,
                num_heads = num_heads,
                dropout = 0.0,
                conv_kernel_size = conv_kernel_size,
                use_bias = use_bias,
            )
            self.layers.append(layer)

        self.att_context_size = att_context_size

    def _create_masks(self, padding_length, max_audio_length, device):
        att_context_size = self.att_context_size
        attn_mask = torch.ones(
            1, max_audio_length, max_audio_length, device=device, dtype = torch.bool
        )
        chunk_size = att_context_size[1] + 1
        left_chunks_num = (
            att_context_size[0] // chunk_size
        )
        chunk_idx = torch.arange(0, max_audio_length, device=device, dtype = torch.int)
        diff_chunks = chunk_idx.unsqueeze(1) - chunk_idx.unsqueeze(0)
        chunked_limited_mask = torch.logical_and(
            torch.le(diff_chunks, left_chunks_num), torch.ge(diff_chunks, 0)
        )
        att_mask = torch.logical_and(attn_mask, chunked_limited_mask.unsqueeze(0))

        pad_mask = torch.arange(0, max_audio_length, device=device).expand(
            padding_length.size(0), -1
        ) < padding_length.unsqueeze(-1)

        pad_mask_for_att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        pad_mask_for_att_mask = torch.logical_and(
            pad_mask_for_att_mask, pad_mask_for_att_mask.transpose(1, 2)
        )
        att_mask = torch.logical_and(
            pad_mask_for_att_mask, att_mask.to(pad_mask_for_att_mask.device)
        )
        att_mask = ~att_mask

        pad_mask = ~pad_mask
        return pad_mask, att_mask


    def forward(
            self,
            x:Tensor,
            length:Tensor = None,
    ):
        if length is None:
            length = x.new_full(
                (x.size(0),), x.size(-1), dtype=torch.int64, device=x.device
            )
        x = x.transpose(1,2)
        x, length = self.pre_encode(x, length)
        x, pos_emb = self.pos_enc(x)

        pad_mask, att_mask = self._create_masks(
            padding_length=length,
            max_audio_length=x.shape[1],
            device = x.device
        )

        for layer in self.layers:
            x = layer(x, pos_emb = pos_emb, pad_mask=pad_mask, attn_mask=att_mask)
        x = x.transpose(1, 2)
        return x

if __name__ == "__main__":
    encoder = ConformerEncoder(
        feat_in = 128,
        hidden_size = 512,
        ff_expansion_factor = 4,
        num_layers = 17,
        num_heads = 8,
        subsampling_factor = 8,
        pos_emb_max_len = 5000,
        conv_kernel_size = 9,
        att_context_size = [70,1],
        subsampling_conv_channels = 256,
        use_bias = False,
    ).eval()
    x = torch.randn(1, 128, 201)
    lengths = torch.LongTensor([201])
    output = encoder(x, lengths)
    print(output)

