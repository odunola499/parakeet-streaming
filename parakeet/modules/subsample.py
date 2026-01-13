import math

import torch
from torch import Tensor, nn

from parakeet.modules.convolution import CausalConv2D


def calc_length(lengths, all_paddings, kernel_size, stride, repeat_num=1):
    add_pad: float = all_paddings - kernel_size
    one: float = 1.0
    for _i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)


def apply_channel_mask(tensor, mask):
    batch_size, channels, time, features = tensor.shape
    expanded_mask = mask.unsqueeze(1).expand(batch_size, channels, time, features)
    return tensor * expanded_mask


def calculate_conv_output_size(
    input_size: torch.Tensor, kernel_size: int, stride: int, padding: tuple[int, int]
):
    return (input_size + padding[0] + padding[1] - kernel_size) // stride + 1


class MaskedConvSequential(nn.Sequential):
    def forward(self, x, lengths):
        x = x.unsqueeze(1)
        current_lengths = lengths.clone().float()
        mask = self._create_mask(x, current_lengths.long())

        for layer in self:
            x = apply_channel_mask(x, mask)

            x = layer(x)
            if hasattr(layer, "stride") and layer.stride != (1, 1):
                padding = (layer._left_padding, layer._right_padding)
                current_lengths = calculate_conv_output_size(
                    current_lengths, layer.kernel_size[0], layer.stride[0], padding
                )
                mask = self._create_mask(x, current_lengths.long())

        x = apply_channel_mask(x, mask)
        return x, current_lengths.long()

    def _create_mask(self, tensor, lengths):
        batch_size, channels, time, features = tensor.shape
        time_mask = torch.arange(time, device=tensor.device).expand(
            batch_size, time
        ) < lengths.unsqueeze(1)
        return (
            time_mask.unsqueeze(-1).expand(batch_size, time, features).to(tensor.dtype)
        )


class ConvSubsampling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        subsampling_conv_channels: int,
        feat_in: int,
        subsampling_factor: int,
        stride: int = 2,
        kernel_size: int = 3,
    ):
        super().__init__()
        self._conv_channels = subsampling_conv_channels
        self.feat_in = feat_in
        self.feat_out = -1
        conv_channels = subsampling_conv_channels

        self.sampling_num = int(math.log(subsampling_factor, 2))
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_chunking_factor = 1
        activation = nn.ReLU(inplace=True)

        in_channels = 1
        layers = []

        self._stride = stride
        self._kernel_size = kernel_size
        self.ceil_mode = False
        self._left_padding = kernel_size - 1
        self._right_padding = stride - 1
        self._max_cache_len = subsampling_factor + 1

        layers.append(
            CausalConv2D(
                in_channels=in_channels,
                out_channels=conv_channels,
                kernel_size=self._kernel_size,
                stride=self._stride,
                padding=None,
            )
        )

        in_channels = conv_channels
        layers.append(activation)

        for _i in range(self.sampling_num - 1):
            layers.append(
                CausalConv2D(  # depthwise conv
                    in_channels=in_channels,
                    out_channels=in_channels,
                    kernel_size=self._kernel_size,
                    stride=self._stride,
                    groups=in_channels,
                    padding=None,
                )
            )
            layers.append(
                CausalConv2D(  # pointwise conv
                    in_channels=in_channels,
                    out_channels=conv_channels,
                    kernel_size=1,
                    stride=1,
                    groups=1,
                    padding=None,
                )
            )
            layers.append(activation)
            in_channels = conv_channels

        final_freq_dim = calc_length(
            lengths=torch.tensor(self.feat_in, dtype=torch.float),
            all_paddings=self._left_padding + self._right_padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            repeat_num=self.sampling_num,
        ).item()
        self.out = nn.Linear(
            conv_channels * final_freq_dim,
            hidden_size,
            bias=True,
        )
        self.conv = MaskedConvSequential(*layers)

    def forward(self, x: Tensor, lengths: Tensor):
        x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
        if torch.numel(x) > x_ceil:
            x, lengths, success = self.conv_split_by_batch(x, lengths)
            if not success:
                x, lengths = self.conv_forward(x, lengths)
        else:
            x, lengths = self.conv_forward(x, lengths)

        b, c, t, f = x.size()
        x = x.transpose(1, 2).reshape(b, t, -1)
        x = self.out(x)
        return x, lengths

    def conv_split_by_batch(self, x: Tensor, lengths: Tensor):
        batch_size = x.shape[0]
        if batch_size == 1:
            return x, lengths, False
        x_ceil = 2**31 / self._conv_channels * self._stride * self._stride
        p = math.ceil(math.log(torch.numel(x) / x_ceil, 2))
        cf = 2**p

        new_batch_size = batch_size // cf
        outputs = [
            self.conv_forward(chunk, ln)
            for chunk, ln in zip(
                torch.split(x, new_batch_size, dim=0),
                torch.split(lengths, new_batch_size, dim=0),
                strict=False,
            )
        ]
        return (
            torch.cat([a[0] for a in outputs]),
            torch.cat([a[1] for a in outputs]),
            True,
        )

    def conv_forward(self, x: Tensor, lengths: Tensor):
        x, lengths = self.conv(x, lengths)
        return x, lengths


__all__ = [
    "ConvSubsampling",
    "calc_length",
    "apply_channel_mask",
    "calculate_conv_output_size",
]
