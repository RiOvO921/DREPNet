# PGA.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Conv, C2PSA


class PolarityGatedAttention(nn.Module):
    def __init__(self, dim, hw, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., sr_ratio=1, kernel_size=5,
                 alpha=4):
        super().__init__()
        self.h, self.w = hw
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.sr_ratio = sr_ratio
        self.alpha = alpha

        self.qg = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(self.head_dim, self.head_dim, kernel_size, groups=self.head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        self.scale = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x):
        B, N, C = x.shape
        q, g = self.qg(x).view(B, N, 2, C).unbind(2)

        if self.sr_ratio > 1:
            x_sr = x.permute(0, 2, 1).reshape(B, C, self.h, self.w)
            x_sr = self.sr(x_sr).reshape(B, C, -1).permute(0, 2, 1)
            x_sr = self.norm(x_sr)
            kv = self.kv(x_sr).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).view(B, -1, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv.unbind(0)

        # Reshape query for multi-head
        q = q.view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Polarity decomposition with learnable power
        power = 1 + self.alpha * torch.sigmoid(self.power)
        q_pos, q_neg = (F.relu(q) ** power), (F.relu(-q) ** power)
        k_pos, k_neg = (F.relu(k) ** power), (F.relu(-k) ** power)

        # Split value vector for separate streams
        v_s, v_o = v.chunk(2, dim=-1)

        # Same-signed stream
        kv_s = (k_pos.transpose(-2, -1) @ v_s) + (k_neg.transpose(-2, -1) @ v_s)
        x_s = q_pos @ kv_s + q_neg @ kv_s

        # Opposite-signed stream
        kv_o = (k_neg.transpose(-2, -1) @ v_o) + (k_pos.transpose(-2, -1) @ v_o)
        x_o = q_pos @ kv_o + q_neg @ kv_o

        # Combine streams and reshape
        x = torch.cat([x_s, x_o], dim=-1).transpose(1, 2).reshape(B, N, C)

        # Local context from depth-wise conv on value
        v_local = self.dwc(
            v.permute(0, 2, 1, 3).reshape(-1, self.head_dim, self.h // self.sr_ratio, self.w // self.sr_ratio))
        v_local = v_local.reshape(B, self.num_heads, self.head_dim, -1).permute(0, 2, 1, 3).reshape(B, C, -1)
        if self.sr_ratio > 1:
            v_local = F.interpolate(v_local, size=N, mode='linear', align_corners=False)
        v_local = v_local.permute(0, 2, 1)

        # Final projection with gating
        x = self.proj((x + v_local) * g)
        x = self.proj_drop(x)

        return x


class PGA(C2PSA):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__(c1, c2, n, e)
        # Note: hw (height, width) of the feature map needs to be correctly passed during model construction.
        # This is a placeholder for a common feature map size.
        self.m = nn.Sequential(*(PolarityGatedAttention(self.c, hw=(20, 20), num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        a, b = self.cv1(x).chunk(2, dim=1)  # Assumes c2 = 2 * c1
        BS, C, H, W = b.shape
        b_flat = b.flatten(2).permute(0, 2, 1)
        b_pga = self.m(b_flat)
        b_out = b_pga.permute(0, 2, 1).view(BS, C, H, W)
        return self.cv2(torch.cat((a, b_out), 1))