import torch
import torch.nn as nn

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.gamma * self._norm(x.float()).type_as(x)

if __name__ == '__main__':
    x = torch.randn((2, 3, 5))
    rms = RMSNorm(dim=x.shape[-1])
    out = rms(x)
    print(f"x = {x}, \nout = {out}")