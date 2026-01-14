from typing import Union

import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
    ):

        self._left_padding = padding[0]
        self._right_padding = padding[1]

        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )

    def update_cache(self, x, cache=None):
        B, C, T = x.shape
        if cache is None:
            cache = torch.zeros(
                B, C, self.kernel_size[0] - 1, device=x.device, dtype=x.dtype
            )
        new_x = F.pad(x, pad=(0, self._right_padding))
        new_x = torch.cat([cache, new_x], dim=-1)
        next_cache = new_x[:, :, -cache.size(-1) :]

        return new_x, next_cache

    def forward(self, x, cache=None):
        x, cache = self.update_cache(x, cache=cache)
        x = super().forward(x)
        return x, cache


class CausalConv2D(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Union[str, int] = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device=None,
        dtype=None,
    ) -> None:
        if padding is not None:
            raise ValueError("Argument padding should be set to None for CausalConv2D.")
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1

        padding = 0
        super(CausalConv2D, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            device,
            dtype,
        )

    def forward(
        self,
        x,
    ):
        x = F.pad(
            x,
            pad=(
                self._left_padding,
                self._right_padding,
                self._left_padding,
                self._right_padding,
            ),
        )
        x = super().forward(x)
        return x


class ConformerConvolution(nn.Module):
    def __init__(
        self,
        hidden_size,
        kernel_size,
        use_bias=False,
    ):
        super().__init__()
        self.d_model = hidden_size
        self.kernel_size = kernel_size
        conv_context_size = [kernel_size - 1, 0]

        self.pointwise_conv1 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

        self.depthwise_conv = CausalConv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=1,
            groups=hidden_size,
            padding=conv_context_size,
            bias=use_bias,
        )

        self.batch_norm = nn.LayerNorm(hidden_size)

        self.activation = nn.SiLU()
        self.pointwise_conv2 = nn.Conv1d(
            in_channels=hidden_size,
            out_channels=hidden_size,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=use_bias,
        )

    def forward(self, x, pad_mask, cache=None):
        x = x.transpose(1, 2)
        x = self.pointwise_conv1(x)
        x = F.glu(x, dim=1)

        x = x.masked_fill(pad_mask.unsqueeze(1), 0.0)
        x, cache = self.depthwise_conv(x, cache=cache)
        x = self.batch_norm(x.transpose(1, 2)).transpose(1, 2)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        x = x.transpose(1, 2)
        return x, cache
