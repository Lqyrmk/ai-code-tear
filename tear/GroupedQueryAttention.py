import torch
import torch.nn as nn
import torch.nn.functional as F

from math import inf

def repeat_kv(x, n_rep):
    if n_rep == 1:
        return x
    B, H_kv, L, D = x.shape
    x = x[:, :, None, :, :].expand(B, H_kv, n_rep, L, D)
    return x.reshape(B, H_kv * n_rep, L, D)

class GroupedQueryAttention(nn.Module):

    def __init__(self, d_model, num_heads, num_kv_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0
        assert num_heads % num_kv_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.n_rep = num_heads // num_kv_heads

        self.W_q = nn.Linear(d_model, num_heads * self.head_dim)
        self.W_k = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_v = nn.Linear(d_model, num_kv_heads * self.head_dim)
        self.W_o = nn.Linear(num_heads * self.head_dim, d_model)

        self.scale = self.head_dim ** -0.5
        self.dropout = dropout



    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False):

        batch_size, seq_len, _ = x.shape

        q = self.W_q(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Q:  [B, H, L, dh]
        # KV: [B, H_kv, L, dh]
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)  # [B, H_kv, L + dL, dh]
            v = torch.cat([past_kv[1], v], dim=2)  # [B, H_kv, L + dL, dh]
        new_kv = (k, v) if use_cache else None

        S = k.shape[2]
        k = repeat_kv(k, self.n_rep)  # [B, H_kv * n_rep, L + dL, dh] = [B, H, L + dL, dh]
        v = repeat_kv(v, self.n_rep)  # [B, H_kv * n_rep, L + dL, dh] = [B, H, L + dL, dh]

        scores = (q @ k.transpose(-2, -1)) * self.scale

        if attn_mask is None:
            causal_attn_mask = torch.ones((seq_len, S), dtype=torch.bool).tril(diagonal=S - seq_len)
            scores = scores.masked_fill(~causal_attn_mask, -inf)
        else:
            scores = scores + attn_mask

        scores = F.softmax(scores.float(), dim=-1).to(q.device)
        scores = F.dropout(scores, self.dropout)

        out = scores @ v  # [B, H, L, dh]
        out = out.transpose(1, 2).reshape(batch_size, seq_len, -1)
        out = self.W_o(out)
        return out, new_kv

if __name__ == '__main__':
    batch_size = 2
    seq_len = 5
    d_model = 64
    num_heads = 8
    num_kv_heads = 4

    x = torch.randn((batch_size, seq_len, d_model))
    gqa = GroupedQueryAttention(d_model=d_model, num_heads=num_heads, num_kv_heads=num_kv_heads)
    attn_x = gqa(x)

    print(f"x = {x}, \nout = {attn_x}")