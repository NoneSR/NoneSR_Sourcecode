# ---------------------------------------------------------------------------
# Efficient Long-Range Attention Network for Image Super-resolution
# Official GitHub: https://github.com/xindongzhang/ELAN
#
# Modified by Tianle Liu (tianle.l@outlook.com)
# ---------------------------------------------------------------------------
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as f
from basicsr.utils.registry import ARCH_REGISTRY
from einops import rearrange

from archs.utils import Conv2d1x1, Conv2d3x3, ShiftConv2d1x1, MeanShift, Upsampler
from thop import profile
from thop import clever_format

class LFE(nn.Module):
    r"""Local Feature Extraction.

       Args:
           planes: Number of input channels
           r_expand: Channel expansion ratio
           act_layer:

       """

    def __init__(self, planes: int, r_expand: int = 2,
                 act_layer: nn.Module = nn.ReLU, img_size = (320, 180)) -> None:
        super(LFE, self).__init__()

        self.lfe = nn.Sequential(ShiftConv2d1x1(planes, planes * r_expand),
                                 act_layer(inplace=True),
                                 ShiftConv2d1x1(planes * r_expand, planes))

        self.img_size = img_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lfe(x)

    def flops(self):
        H, W = (320, 180)
        flops = H * W * 60 * 120 * 2
        return flops


class GMSA(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           planes: Number of input channels
           shifts:
           window_sizes: Window size
           pass_attn:

       """

    def __init__(self, planes: int = 60, shifts: int = 0,
                 window_sizes: tuple = (4, 8, 12), pass_attn: int = 0) -> None:

        super(GMSA, self).__init__()
        self.shifts = shifts
        self.window_sizes = window_sizes
        self.dim = planes
        self.prev_atns = pass_attn

        if pass_attn == 0:
            self.split_chns = [planes * 2 // 3, planes * 2 // 3, planes * 2 // 3]
            self.project_inp = nn.Sequential(
                Conv2d1x1(planes, planes * 2),
                nn.BatchNorm2d(planes * 2)
            )
            self.project_out = Conv2d1x1(planes, planes)
        else:
            self.split_chns = [planes // 3, planes // 3, planes // 3]
            self.project_inp = nn.Sequential(
                Conv2d1x1(planes, planes),
                nn.BatchNorm2d(planes)
            )
            self.project_out = Conv2d1x1(planes, planes)

    def forward(self, x: torch.Tensor, prev_atns: list = None):
        b, c, h, w = x.shape
        x = self.project_inp(x)
        # print(x.shape)
        xs = torch.split(x, self.split_chns, dim=1)
        ys = []
        atns = []

        if prev_atns is None:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                q, v = rearrange(
                    x_, 'b (qv c) (h dh) (w dw) -> qv (b h w) (dh dw) c',
                    qv=2, dh=wsize, dw=wsize
                )
                atn = (q @ q.transpose(-2, -1))
                atn = atn.softmax(dim=-1)
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
                atns.append(atn)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, atns
        else:
            for idx, x_ in enumerate(xs):
                wsize = self.window_sizes[idx]
                if self.shifts > 0:
                    x_ = torch.roll(x_, shifts=(-wsize // 2, -wsize // 2), dims=(2, 3))
                atn = prev_atns[idx]
                v = rearrange(
                    x_, 'b (c) (h dh) (w dw) -> (b h w) (dh dw) c',
                    dh=wsize, dw=wsize
                )
                y_ = (atn @ v)
                y_ = rearrange(
                    y_, '(b h w) (dh dw) c-> b (c) (h dh) (w dw)',
                    h=h // wsize, w=w // wsize, dh=wsize, dw=wsize
                )
                if self.shifts > 0:
                    y_ = torch.roll(y_, shifts=(wsize // 2, wsize // 2), dims=(2, 3))
                ys.append(y_)
            y = torch.cat(ys, dim=1)
            y = self.project_out(y)
            return y, prev_atns

    def flops(self):
        # 假设输入的尺寸是 (b, c, h, w)
        b, c, h, w = 1, self.dim, 320, 180  # 这里示例输入大小，根据实际情况修改

        # 初始化FLOPs为0
        flops = 0

        # 第一个分支的FLOPs
        proj_input_flops = c * c * 1 * 1  # Conv2d1x1的计算复杂度为输入通道数的平方乘以1*1卷积
        bn_flops = 2 * c * h * w  # BatchNorm2d的计算复杂度是2*c*h*w
        proj_output_flops = c * c * 1 * 1  # Conv2d1x1的计算复杂度为输出通道数的平方乘以1*1卷积
        first_branch_flops = proj_input_flops + bn_flops + proj_output_flops

        # 第二个分支的FLOPs
        # 假设有三个窗口大小，每个窗口大小对应的FLOPs不同
        window_flops = [flops_size ** 2 * c * (2 * c) for flops_size in self.window_sizes]
        second_branch_flops = sum(window_flops)

        # 分支之间的FLOPs比例
        if self.prev_atns is None:
            # 如果没有输入先前的注意力
            first_branch_ratio = 1.0
            second_branch_ratio = 0.0
        else:
            # 如果有输入先前的注意力
            first_branch_ratio = 0.0
            second_branch_ratio = 1.0

        # 计算总的FLOPs
        total_flops = first_branch_ratio * first_branch_flops + second_branch_ratio * second_branch_flops

        return total_flops


class ELAB(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           planes: Number of input channels
           r_expand: Channel expansion ratio
           shifts:
           window_sizes: Window size
           n_share: Depth of shared attention.

       """

    def __init__(self, planes: int = 60, r_expand: int = 2, shifts: int = 0,
                 window_sizes: tuple = (4, 8, 12), n_share: int = 1, img_size = (320, 180)) -> None:
        super(ELAB, self).__init__()

        self.modules_lfe = nn.ModuleList([LFE(planes=planes, r_expand=r_expand)
                                          for _ in range(n_share + 1)])
        self.modules_gmsa = nn.ModuleList([GMSA(planes=planes, shifts=shifts,
                                                window_sizes=window_sizes, pass_attn=i)
                                           for i in range(n_share + 1)])
        self.img_size = img_size
        self.dim = planes
        self.window_sizes = window_sizes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        atn = None
        for module1, module2 in zip(self.modules_lfe, self.modules_gmsa):
            x = module1(x) + x
            y, atn = module2(x, atn)
            x = y + x
        return x

    def flops(self):
        # 假设输入的尺寸是 (b, c, h, w)
        b, c, h, w = 1, self.dim, 320, 180  # 这里示例输入大小，根据实际情况修改

        # 初始化FLOPs为0
        flops = 0

        # 计算所有LFE子模块的FLOPs并相加
        lfe_flops = sum(lfe_module.flops() for lfe_module in self.modules_lfe)

        # 计算所有GMSA子模块的FLOPs并相加
        gmsa_flops = sum(gmsa_module.flops() for gmsa_module in self.modules_gmsa)

        # 计算总的FLOPs
        total_flops = lfe_flops + gmsa_flops

        return total_flops


@ARCH_REGISTRY.register()
class ELAN(nn.Module):
    r"""Residual Feature Distillation Network.

       Args:
           upscale:
           planes: Number of input channels
           num_blocks: Number of RFDB
           window_sizes: Window size
           n_share: Depth of shared attention
           r_expand: Channel expansion ratio

       """

    def __init__(self, upscale: int, num_in_ch: int, num_out_ch: int, task: str,
                 planes: int = 60, num_blocks: int = 24,
                 window_sizes: tuple = (4, 8, 12), n_share: int = 1, r_expand: int = 2) -> None:
        super(ELAN, self).__init__()

        self.window_sizes = window_sizes
        self.upscale = upscale
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)
        self.dim = planes

        self.head = Conv2d3x3(num_in_ch, planes)

        m_body = [ELAB(planes, r_expand, i % 2, window_sizes, n_share)
                  for i in range(num_blocks // (1 + n_share))]
        self.body = nn.Sequential(*m_body)

        self.tail = Upsampler(upscale=upscale, in_channels=planes,
                              out_channels=num_out_ch, upsample_mode=task)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        wsize = self.window_sizes[0]
        for i in range(1, len(self.window_sizes)):
            wsize = wsize * self.window_sizes[i] // math.gcd(wsize, self.window_sizes[i])
        mod_pad_h = (wsize - h % wsize) % wsize
        mod_pad_w = (wsize - w % wsize) % wsize
        x = f.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.size()
        x = self.check_image_size(x)

        # reduce the mean of pixels
        sub_x = self.sub_mean(x)

        # head
        head_x = self.head(sub_x)

        # body
        t1 = time.time()
        body_x = self.body(head_x)
        body_x = body_x + head_x
        t2 = time.time()
        print(f"time: {t2 - t1}")

        # tail
        tail_x = self.tail(body_x)

        # add the mean of pixels
        add_x = self.add_mean(tail_x)

        return add_x[:, :, 0:h * self.upscale, 0:w * self.upscale]

    # def flops(self):
    #     # 假设输入的尺寸是 (b, c, h, w)
    #     b, c, h, w = 1, 3, 320, 180  # 这里示例输入大小，根据实际情况修改
    #
    #     # 初始化FLOPs为0
    #     flops = 0
    #
    #     # 计算Conv2d3x3模块的FLOPs
    #     # 假设Conv2d3x3的卷积核大小是3x3，输出通道数是planes
    #     conv2d3x3_flops = c * self.dim * 3 * 3 * h * w
    #
    #     # 计算所有ELAB子模块的FLOPs并相加
    #     elab_flops = sum(elab_module.flops() for elab_module in self.body)
    #
    #     # 计算Upsampler模块的FLOPs
    #     # 假设Upsampler的上采样倍数是upscale，输入通道数是planes，输出通道数是num_out_ch
    #     upsampler_flops = b * self.dim * h * w * self.upscale**2 * 3
    #
    #     # 计算总的FLOPs
    #     total_flops = conv2d3x3_flops + elab_flops + upsampler_flops
    #
    #     return total_flops


if __name__ == '__main__':
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


    # 导入所需的库和ELAN模型类
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 创建ELAN模型实例
    upscale = 4
    height = (1280 // upscale)
    width = (720 // upscale)
    net = ELAN(upscale=3, num_in_ch=3, num_out_ch=3, planes=60, window_sizes=(4, 8, 16), num_blocks=24,
               n_share=1, task='lsr')
    net.to(device)

    input_tensor = torch.randn(1, 3, 426, 240) #1280X720(640X360)(426X240)(320X180)
    input_tensor = input_tensor.cuda()
    # input_tensor.to(device)


    def time_text(t):
        if t >= 3600:
            return '{:.1f}h'.format(t / 3600)
        elif t >= 60:
            return '{:.1f}m'.format(t / 60)
        else:
            return '{:.1f}s'.format(t)

    import time

    t1 = time.time()

    net(input_tensor)
    t2 = time.time()
    print('end',t2-t1)

    # flops, params = profile(net, inputs=(input_tensor,))
    # flops, params = clever_format([net.flops(), count_parameters(net)], "%.3f")


    # print(net(input_tensor).shape)

    # 将模型放在设备上（此处假设使用CPU）
    # device = torch.device('cpu')
    # net.to(device)
    #
    # # 计算ELAN模型的FLOPs
    # total_flops = 0
    # for m in net.modules():
    #     if hasattr(m, 'flops'):
    #         total_flops += m.flops()
    # flops_count = total_flops / 1e9  # 将结果转换为十亿次浮点运算次数
    #
    # print("ELAN模型的FLOPs：{:.2f} GFLOPs".format(flops_count))

    # print(count_parameters(net))
    #
    # data = torch.randn(1, 3, 120, 80)
    # print(net(data).size())
    # print(height, width, net.flops() / 1e9)




