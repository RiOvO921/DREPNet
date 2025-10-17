# EMFE.py

import torch.nn as nn
import torch.nn.functional as F

from common import Conv, C3k, C3k2


class EdgeEnhancer(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.out_conv = Conv(in_dim, in_dim, act=nn.Sigmoid())
        self.pool = nn.AvgPool2d(3, stride=1, padding=1)

    def forward(self, x):
        residual = x - self.pool(x)
        attention = self.out_conv(residual)
        return x + residual * attention


class EMFE(nn.Module):
    def __init__(self, inc, bins=(3, 6, 9, 12)):
        super().__init__()
        self.num_bins = len(bins)
        self.split_inc = inc // self.num_bins

        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                Conv(inc, self.split_inc, 1),
                Conv(self.split_inc, self.split_inc, 3, g=self.split_inc)
            ) for bin_size in bins
        ])

        self.ees = nn.ModuleList([EdgeEnhancer(self.split_inc) for _ in bins])

        self.local_conv = Conv(inc, inc, 3)
        self.final_conv = Conv(inc * 2, inc)

    def forward(self, x):
        x_size = x.size()
        out_features = [self.local_conv(x)]

        for i in range(self.num_bins):
            branch_feature = self.features[i](x)
            branch_feature_interp = F.interpolate(branch_feature, x_size[2:], mode='bilinear', align_corners=True)
            out_features.append(self.ees[i](branch_feature_interp))

        return self.final_conv(torch.cat(out_features, 1))


class REE_(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        self.m = nn.Sequential(*(EMFE(int(c2 * e)) for _ in range(n)))


class REE(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            REE_(int(c2 * e), int(c2 * e), 2, shortcut, g) if c3k else EMFE(int(c2 * e)) for _ in range(n))


class SEI_(C3k):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=3):
        super().__init__(c1, c2, n, shortcut, g, e, k)
        self.m = nn.Sequential(*(EMFE(int(c2 * e)) for _ in range(n)))


class SEI(C3k2):
    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, c3k, e, g, shortcut)
        self.m = nn.ModuleList(
            SEI_(int(c2 * e), int(c2 * e), 2, shortcut, g) if c3k else EMFE(int(c2 * e)) for _ in range(n))