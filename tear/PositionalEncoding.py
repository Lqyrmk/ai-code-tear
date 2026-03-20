import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, seq_len, d_model):
        super().__init__()
        pe = torch.zeros((seq_len, d_model))  # [seq_len, d]
        even = torch.arange(0, d_model, 2)  # [d // 2,]
        pos = torch.arange(0, seq_len).unsqueeze(1)  # [seq_len, 1] 列向量

        # 广播计算
        # x = pos / (torch.pow(10000.0, even / d_model))
        x = pos * torch.exp(-math.log(10000.0) * even / d_model)  # [seq_len, d // 2]

        pe[:, 0::2] = torch.sin(x)
        pe[:, 1::2] = torch.cos(x)

        pe = pe.unsqueeze(0)  # [1, seq_len, d]
        self.register_buffer('pe', pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

if __name__ == '__main__':
    batch_size = 2
    seq_len = 10
    d_model = 6
    x = torch.ones((batch_size, seq_len, d_model))
    PE = PositionalEncoding(seq_len, d_model)
    out = PE(x)
    print(f'x = {x}\nout = {out}')