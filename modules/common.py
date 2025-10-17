# common.py

import torch
import torch.nn as nn

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

# NOTE: The base classes C3k, C3k2, C2PSA must be imported from your project's module library.
# Example: from ultralytics.nn.modules.block import C3k, C3k2, C2PSA
# For this standalone script to be runnable, placeholder classes are defined below.
class C3k(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
class C3k2(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()
class C2PSA(nn.Module):
    def __init__(self, *args, **kwargs): super().__init__()