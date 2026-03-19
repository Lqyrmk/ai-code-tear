import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()

        assert dim % num_heads == 0

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.W_q = nn.Linear(dim, dim)
        self.W_k = nn.Linear(dim, dim)
        self.W_v = nn.Linear(dim, dim)

        self.W_o = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        B, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        Q = Q.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # [B, num_heads, seq_len, head_dim]
        scores = (Q @ K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # causal mask
        scores = scores + torch.triu(
            torch.full((seq_len, seq_len), float('-inf')),
            diagonal=1
        ).unsqueeze(0).unsqueeze(0)

        # padding mask: [B, seq_len]
        # if mask is not None:
        #     padding_mask = mask.unsqueeze(1).unsqueeze(2)
        #     padding_mask = (1.0 - padding_mask) * -1e9
        #     scores = scores + padding_mask

        scores = F.softmax(scores.float(), dim=-1).type_as(Q)

        out = scores @ V  # [B, num_heads, seq_len, head_dim]
        out = out.transpose(1, 2).contiguous()  # [B, seq_len, num_heads, head_dim]
        out = out.view(B, seq_len, -1)  # [B, seq_len, num_heads * head_dim]
        out = self.W_o(out)
        return out

if __name__ == '__main__':
    x = torch.randn((2, 3, 10))
    num_heads = 5
    mha = MultiHeadAttention(dim=x.shape[-1], num_heads=num_heads)
    attn_x = mha(x)
    print(f"x = {x}, \nout = {attn_x}")