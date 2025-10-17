# RepDSFPN_components.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from common import Conv, autopad


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class SPD(nn.Module):
    def __init__(self, block_size=2):
        super().__init__()
        self.block_size = block_size

    def forward(self, x):
        b, c, h, w = x.shape
        if h % self.block_size != 0 or w % self.block_size != 0:
            raise ValueError("Input height and width must be divisible by block_size.")

        return x.view(b, c, h // self.block_size, self.block_size, w // self.block_size, self.block_size).permute(0, 3,
                                                                                                                  5, 1,
                                                                                                                  2,
                                                                                                                  4).reshape(
            b, c * self.block_size ** 2, h // self.block_size, w // self.block_size)


class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups

        if style == 'pl':
            in_channels_offset = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            in_channels_offset = in_channels
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels_offset, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels_offset, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        yy, xx = torch.meshgrid(h, h, indexing='ij')
        return torch.stack([xx, yy], dim=0).repeat(1, self.groups, 1, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)

        coords_h = torch.arange(H, device=x.device, dtype=x.dtype) + 0.5
        coords_w = torch.arange(W, device=x.device, dtype=x.dtype) + 0.5
        coords_yy, coords_xx = torch.meshgrid(coords_h, coords_w, indexing='ij')
        coords = torch.stack([coords_xx, coords_yy], dim=0).unsqueeze(0).unsqueeze(2)

        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1

        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale)
        coords = coords.view(B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)

        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward(self, x):
        if self.style == 'pl':
            x_ = F.pixel_shuffle(x, self.scale)
            offset_input = x_
        else:
            offset_input = x

        if hasattr(self, 'scope'):
            offset = self.offset(offset_input) * self.scope(offset_input).sigmoid() * 0.5
        else:
            offset = self.offset(offset_input) * 0.25

        if self.style == 'pl':
            offset = F.pixel_unshuffle(offset, self.scale)

        return self.sample(x, offset + self.init_pos)


class RepConv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, d=1, act=True, bn=False, deploy=False):
        super().__init__()
        self.g = g
        self.c1 = c1
        self.c2 = c2
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self.deploy = deploy

        if deploy:
            self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=True)
        else:
            self.bn = nn.BatchNorm2d(c1) if bn and c2 == c1 and s == 1 else None
            self.conv1 = Conv(c1, c2, k, s, p=p, g=g, act=False)
            self.conv2 = Conv(c1, c2, 1, s, p=(p - k // 2), g=g, act=False)

    def forward(self, x):
        if self.deploy:
            return self.act(self.conv(x))

        id_out = 0 if self.bn is None else self.bn(x)
        return self.act(self.conv1(x) + self.conv2(x) + id_out)

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        kernelid, biasid = self._fuse_bn_tensor(self.bn)
        return kernel3x3 + F.pad(kernel1x1, [1, 1, 1, 1]) + kernelid, bias3x3 + bias1x1 + biasid

    def _fuse_bn_tensor(self, branch):
        if branch is None: return 0, 0
        if isinstance(branch, Conv):
            kernel, bn = branch.conv.weight, branch.bn
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, "id_tensor"):
                input_dim = self.c1 // self.g
                kernel_value = torch.zeros(self.c1, input_dim, 3, 3)
                for i in range(self.c1):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = kernel_value.to(branch.weight.device)
            kernel, bn = self.id_tensor, branch

        std = (bn.running_var + bn.eps).sqrt()
        t = (bn.weight / std).reshape(-1, 1, 1, 1)
        return kernel * t, bn.bias - bn.running_mean * bn.weight / std

    def fuse(self):
        if self.deploy: return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(self.c1, self.c2, 3, self.conv1.conv.stride, self.conv1.conv.padding, groups=self.g,
                              bias=True)
        self.conv.weight.data = kernel
        self.conv.bias.data = bias
        self.deploy = True
        for module in self.children():
            if module != self.conv:
                del module


class BasicBlock_3x3_Reverse(nn.Module):
    def __init__(self, c_in, c_hidden_ratio, c_out, shortcut=True):
        super().__init__()
        self.conv1 = Conv(int(c_in * c_hidden_ratio), c_out, 3, s=1)
        self.conv2 = RepConv(c_in, int(c_in * c_hidden_ratio), 3, s=1)
        self.shortcut = shortcut and c_in == c_out

    def forward(self, x):
        y = self.conv2(x)
        y = self.conv1(y)
        return x + y if self.shortcut else y


class CSPStage(nn.Module):
    def __init__(self, c_in, c_out, n, block_fn='BasicBlock_3x3_Reverse', c_hidden_ratio=1.0, spp=False):
        super().__init__()
        c_first = c_out // 2
        c_mid = c_out - c_first
        self.conv1 = Conv(c_in, c_first, 1)
        self.conv2 = Conv(c_in, c_mid, 1)

        self.blocks = nn.Sequential(*[
            BasicBlock_3x3_Reverse(c_mid, c_hidden_ratio, c_mid) for _ in range(n)
        ])

        self.conv3 = Conv(c_first + c_mid, c_out, 1)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.blocks(self.conv2(x))
        return self.conv3(torch.cat((y1, y2), dim=1))