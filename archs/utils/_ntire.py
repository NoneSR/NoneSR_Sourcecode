# -------------------------------------------------------------------------------------
# Basic Modules for Image Restoration Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f

__all__ = ['NTIREMeanShift', 'NTIREPixelMixer', 'NTIREShiftConv2d1x1']


class NTIREMeanShift(nn.Module):
    r"""MeanShift for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        rgb_range (int):
        sign (int):
        data_type (str):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, rgb_range: int, sign: int = -1, data_type: str = 'DIV2K') -> None:
        super(NTIREMeanShift, self).__init__()

        self.sign = sign

        self.rgb_range = rgb_range
        self.rgb_std = (1.0, 1.0, 1.0)
        if data_type == 'DIV2K':
            # RGB mean for DIV2K 1-800
            self.rgb_mean = (0.4488, 0.4371, 0.4040)
        elif data_type == 'DF2K':
            # RGB mean for DF2K 1-3450
            self.rgb_mean = (0.4690, 0.4490, 0.4036)
        else:
            raise NotImplementedError(f'Unknown data type for MeanShift: {data_type}.')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        std = torch.Tensor(self.rgb_std)
        weight = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        bias = self.sign * self.rgb_range * torch.Tensor(self.rgb_mean) / std
        return f.conv2d(input=x, weight=weight.type_as(x), bias=bias.type_as(x))


class NTIREPixelMixer(nn.Module):
    r"""Pixel Mixer for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        planes (int):
        mix_margin (int):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(NTIREPixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin

        mask = torch.zeros(self.planes, 1, m * 2 + 1, m * 2 + 1)
        mask[3::5, 0, 0, m] = 1.
        mask[2::5, 0, -1, m] = 1.
        mask[1::5, 0, m, 0] = 1.
        mask[0::5, 0, m, -1] = 1.
        mask[4::5, 0, m, m] = 1.

        return f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                        weight=mask.type_as(x), bias=None, stride=(1, 1), padding=(0, 0),
                        dilation=(1, 1), groups=self.planes)


class NTIREShiftConv2d1x1(nn.Conv2d):
    r"""ShiftConv2d1x1 for NTIRE 2023 Challenge on Efficient Super-Resolution.

    This implementation avoids counting the non-optimized parameters
        into the model parameters.

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    Note:
        May slow down the inference of the model!

    """

    def __init__(self, in_channels: int, out_channels: int, stride: tuple = (1, 1),
                 dilation: tuple = (1, 1), bias: bool = True, shift_mode: str = '+', val: float = 1.,
                 **kwargs) -> None:
        super(NTIREShiftConv2d1x1, self).__init__(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=(1, 1), stride=stride, padding=(0, 0),
                                                  dilation=dilation, groups=1, bias=bias, **kwargs)

        assert in_channels % 5 == 0, f'{in_channels} % 5 != 0.'
        self.in_channels = in_channels
        self.channel_per_group = in_channels // 5
        self.shift_mode = shift_mode
        self.val = val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cgp = self.channel_per_group
        mask = torch.zeros(self.in_channels, 1, 3, 3)
        if self.shift_mode == '+':
            mask[0 * cgp:1 * cgp, 0, 1, 2] = self.val
            mask[1 * cgp:2 * cgp, 0, 1, 0] = self.val
            mask[2 * cgp:3 * cgp, 0, 2, 1] = self.val
            mask[3 * cgp:4 * cgp, 0, 0, 1] = self.val
            mask[4 * cgp:, 0, 1, 1] = self.val
        elif self.shift_mode == 'x':
            mask[0 * cgp:1 * cgp, 0, 0, 0] = self.val
            mask[1 * cgp:2 * cgp, 0, 0, 2] = self.val
            mask[2 * cgp:3 * cgp, 0, 2, 0] = self.val
            mask[3 * cgp:4 * cgp, 0, 2, 2] = self.val
            mask[4 * cgp:, 0, 1, 1] = self.val
        else:
            raise NotImplementedError(f'Unknown shift mode for ShiftConv2d1x1: {self.shift_mode}.')

        x = f.conv2d(input=x, weight=mask.type_as(x), bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.in_channels)
        x = f.conv2d(input=x, weight=self.weight, bias=self.bias,
                     stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        return x
