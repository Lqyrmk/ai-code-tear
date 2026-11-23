import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        mean = x.mean(dim=-1, keepdim=True)

        # 也可以 var = x.var(dim=-1, unbiased=False, keepdim=True)
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)

        # 也可以 out = (x - mean) * torch.rsqrt(var + self.eps)
        out = (x - mean) * torch.rsqrt(var + self.eps)

        return out

    def forward(self, x):
        return self.gamma * self._norm(x) + self.beta

if __name__ == "__main__":
    x = torch.randn((2, 3, 5))
    ln = LayerNorm(dim=x.shape[-1])
    out = ln(x)
    print(f"x = {x}, \nout = {out}")