import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from math import inf

class CrossAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, y, mask=None):

        B, N, _ = x.shape
        _, M, _ = y.shape

        Q = self.W_q(x)
        K = self.W_k(y)
        V = self.W_v(y)

        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        # Q: [B, H, N, D], KV: [B, H, M, D]

        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [B, H, N, M]

        # padding mask for Cross-Attention
        if mask is not None:
            # mask: [B, M]
            mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, M]
            # scores = scores + mask  # 1. 数值写法
            scores = scores.masked_fill(mask, float('-inf'))  # 2. bool 写法

        scores = F.softmax(scores, dim=-1)

        out = scores @ V  # [B, H, N, D]
        out = out.transpose(1, 2).contiguous()
        out = out.view(B, N, -1)

        out = self.W_o(out)

        return out


if __name__ == '__main__':
    x = torch.randn((2, 5, 10))
    y = torch.randn((2, 3, 10))
    # 1. 数值写法
    # mask = torch.tensor([[0., 0., -inf],
    #                      [0., -inf, -inf]])  # [B, M] = [2, 3]
    # 2. bool 写法
    mask = torch.tensor([[False, False, True],
                         [False, True, True]])  # [B, M] = [2, 3]
    num_heads = 5
    cross_attn = CrossAttention(d_model=x.shape[-1], num_heads=num_heads)
    out = cross_attn(x, y, mask)
    print(f"x = {x}, \ny = {y}, \nout = {out}")