# -------------------------------------------------------------------------------------
# Basic Modules for Super Resolution Networks
#
# Implemented/Modified by Fried Rice Lab (https://github.com/Fried-Rice-Lab)
# -------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as f


from ._conv import Conv2d1x1

from utils._conv import Conv2d1x1

from archs.utils._conv import Conv2d1x1



class PixelMixer(nn.Module):
    def __init__(self, planes: int, mix_margin: int = 1) -> None:
        super(PixelMixer, self).__init__()

        assert planes % 5 == 0

        self.planes = planes
        self.mix_margin = mix_margin  # 像素的偏移量
        self.mask = nn.Parameter(torch.zeros((self.planes, 1, mix_margin * 2 + 1, mix_margin * 2 + 1)),
                                 requires_grad=False)

        self.mask[3::5, 0, 0, mix_margin] = 1.
        self.mask[2::5, 0, -1, mix_margin] = 1.
        self.mask[1::5, 0, mix_margin, 0] = 1.
        self.mask[0::5, 0, mix_margin, -1] = 1.
        self.mask[4::5, 0, mix_margin, mix_margin] = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.mix_margin
        x = f.conv2d(input=f.pad(x, pad=(m, m, m, m), mode='circular'),
                     weight=self.mask, bias=None, stride=(1, 1), padding=(0, 0),
                     dilation=(1, 1), groups=self.planes)
        return x


class FocalModulation(nn.Module):
    r"""Focal Modulation.

    Modified from https://github.com/microsoft/FocalNet.

    Args:
        dim (int): Number of input channels.
        focal_level (int): Number of focal levels
        focal_window (int): Focal window size at focal level 1
        focal_factor (int): Step to increase the focal window
        act_layer (nn.Module):

    """

    def __init__(self, dim: int, focal_level: int = 2, focal_window: int = 7, focal_factor: int = 2,
                 act_layer: nn.Module = nn.ReLU) -> None:
        super().__init__()

        self.dim = dim
        self.focal_level = focal_level
        self.focal_window = focal_window
        self.focal_factor = focal_factor

        self.f = nn.Sequential(Conv2d1x1(dim, 2 * dim + (self.focal_level + 1)),
                               nn.BatchNorm2d(2 * dim + (self.focal_level + 1)))

        self.focal_layers = nn.ModuleList()
        for k in range(self.focal_level):
            kernel_size = self.focal_factor * k + self.focal_window
            self.focal_layers.append(
                nn.Sequential(PixelMixer(dim, mix_margin=kernel_size // 2),
                              # nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1,
                              #           groups=dim, padding=kernel_size // 2, bias=False),
                              nn.GELU()))

        self.h = Conv2d1x1(dim, dim)

        self.proj = Conv2d1x1(dim, dim)

        self.act = act_layer(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.f(x)  # b c h w -> b (2c+fl+1) h w

        q, ctx, gates = torch.split(x, [self.dim, self.dim, self.focal_level + 1], 1)  # b c/c/fl+1 h w

        ctx_all = 0
        for l in range(self.focal_level):
            ctx = self.focal_layers[l](ctx)
            ctx_all = ctx_all + ctx * gates[:, l:l + 1]
        ctx_global = self.act(ctx.mean(2, keepdim=True).mean(3, keepdim=True))
        ctx_all = ctx_all + ctx_global * gates[:, self.focal_level:]
        x_out = q * self.h(ctx_all)

        return self.proj(x_out)  # norm本来在proj之前

class Shift(nn.Module):
    r"""

    Args:
        in_channels (int):
        out_channels (int):
        stride (tuple):
        dilation (tuple):
        bias (bool):
        shift_mode (str):
        val (float):

    """

    def __init__(self, planes: int,  mix_margin: float = 1.) -> None:
        super(Shift, self).__init__()

        assert planes % 5 == 0, f'{planes} % 5 != 0.'
        self.planes = planes

        channel_per_group = planes // 5
        self.mask = nn.Parameter(torch.zeros((planes, 1, 3, 3)), requires_grad=False)
        self.mask[0 * channel_per_group:1 * channel_per_group, 0, 1, 2] = mix_margin
        self.mask[1 * channel_per_group:2 * channel_per_group, 0, 1, 0] = mix_margin
        self.mask[2 * channel_per_group:3 * channel_per_group, 0, 2, 1] = mix_margin
        self.mask[3 * channel_per_group:4 * channel_per_group, 0, 0, 1] = mix_margin
        self.mask[4 * channel_per_group:, 0, 1, 1] = mix_margin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = f.conv2d(input=x, weight=self.mask, bias=None,
                     stride=(1, 1), padding=(1, 1), dilation=(1, 1), groups=self.planes)
        return x


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    module = FocalModulation(60)
    print(count_parameters(module))
    # print(module)

    data = torch.randn((1, 60, 64, 64))
    print(module(data).size())

    # a = torch.randn(1, 5, 7, 7)
    # print(a[0][0])

    # block = PixelMixer(5, mix_margin=2)
    # print(block(a)[0][0])
