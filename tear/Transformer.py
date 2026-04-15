import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, xq, xk, xv, mask):

        B = xq.size(0)

        Q = self.W_q(xq).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(xk).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(xv).view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn_scores = (Q @ K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask, -1e9)

        attn_probs = F.softmax(attn_scores, dim=-1)
        out = attn_probs @ V  # [B, H, L, dh]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.dropout(self.W_o(out))
        return out


class PositionalEncoding(nn.Module):

    def __init__(self, max_len, d_model, dropout=0.1):
        super().__init__()

        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(-1)  # [max_len, 1]
        even_i = torch.arange(0, d_model, 2, dtype=torch.float)  # [0, 2, 4, ...]
        base = torch.tensor(10000.0)
        div_term = torch.exp(-torch.log(base) * even_i / d_model)

        x = pos * div_term
        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, L, D]
        x = x + self.pe[:, :x.size(1), :].detach()
        return self.dropout(x)


class FeedForward(nn.Module):

    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(F.relu(self.fc1(x))))


class EncoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_x = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_x))

        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))
        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def forward(self, x, enc_out, enc_mask=None, dec_mask=None):

        # 1. self-attention with causal mask (of target)
        attn_x = self.self_attn(x, x, x, dec_mask)
        x = self.norm1(x + self.dropout1(attn_x))

        # 2. cross-attention with padding mask (of source)
        cross_attn_x = self.cross_attn(x, enc_out, enc_out, enc_mask)
        x = self.norm2(x + self.dropout2(cross_attn_x))

        # 3. ffn
        ffn_out = self.ffn(x)
        x = self.norm3(x + self.dropout3(ffn_out))

        return x


class Encoder(nn.Module):

    def __init__(self, src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(src_vocab_size, d_model)
        self.pe = PositionalEncoding(max_len, d_model, dropout)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, mask=None):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(nn.Module):

    def __init__(self, tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pe = PositionalEncoding(max_len, d_model, dropout)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])

    def forward(self, x, enc_out, enc_mask=None, dec_mask=None):
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, enc_out, enc_mask, dec_mask)
        return x


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_len=5000,
        dropout=0.1
    ):
        super().__init__()
        self.encoder = Encoder(src_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.decoder = Decoder(tgt_vocab_size, d_model, n_layers, n_heads, d_ff, max_len, dropout)
        self.fc = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, enc_x, dec_x, enc_mask=None, dec_mask=None):

        enc_out = self.encoder(enc_x, enc_mask)
        dec_out = self.decoder(dec_x, enc_out, enc_mask, dec_mask)

        logits = self.fc(dec_out)

        return logits

def create_pad_mask(seq, pad_idx):
    # seq: [batch_size, seq_len] -> token id
    # output: [batch_size, 1, 1 seq_len] <-> [B, H, Lq, Lkv] for cross-attention
    return (seq != pad_idx).unsqueeze(1).unsqueeze(1)

def create_causal_mask(seq):
    # seq: [batch_size, seq_len] -> token id
    seq_len = seq.size(1)
    mask = torch.tril(torch.ones(seq_len, seq_len))  # [seq_len, seq_len]
    # output: [1, 1, seq_len, seq_len] <-> [B, H, L, L] for self-attention
    return mask.bool().unsqueeze(0).unsqueeze(0)

def create_data(batch_size, max_len, vocab_size, pad_idx):

    # the length of a sequence
    seq_len = torch.randint(3, max_len, (batch_size,))

    # [1, max_len] < [batch_size, 1]  ===>  [batch_size, max_len]
    # token_mask = torch.arange(0, max_len).unsqueeze(0) < seq_len.unsqueeze(-1)
    token_mask = torch.arange(0, max_len)[None, :] < seq_len[:, None]
    print(f"token_mask: {token_mask.shape}")

    seq = torch.randint(1, vocab_size, (batch_size, max_len))
    seq = seq.masked_fill(~token_mask, pad_idx)

    return seq

if __name__ == "__main__":

    src_vocab_size = 1000
    tgt_vocab_size = 1000
    d_model = 512
    n_layers = 2
    n_heads = 8
    d_ff = 2048
    batch_size = 32
    src_len = 20
    tgt_len = 15
    pad_idx = 0

    src = create_data(batch_size, src_len, src_vocab_size, pad_idx)  # [batch_size, src_len]
    tgt = create_data(batch_size, tgt_len, tgt_vocab_size, pad_idx)  # [batch_size, tgt_len]
    print(f"src: {src.shape}")
    print(f"tgt: {tgt.shape}")

    enc_mask = create_pad_mask(src, pad_idx)
    dec_pad_mask = create_pad_mask(tgt, pad_idx)
    dec_causal_mask = create_causal_mask(tgt)
    dec_mask = dec_pad_mask & dec_causal_mask
    print(f"enc_mask = {enc_mask.shape}")
    print(f"dec_mask = {dec_mask.shape}")

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, n_layers, n_heads, d_ff)

    out = model(src, tgt, enc_mask, dec_mask)

    print("output shape:", out.shape)  # torch.Size([32, 15, 1000])