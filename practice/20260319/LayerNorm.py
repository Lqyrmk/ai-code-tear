import torch
import torch.nn as nn

class LayerNorm(nn.Module):

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # x: [B, N, D]
        mean = x.mean(dim=-1, keepdim=True)  # [B, N, 1]
        var = (x - mean).pow(2).mean(dim=-1, keepdim=True)  # [B, N, 1]
        out = (x - mean) * torch.rsqrt(var + self.eps)  # [B, N, D]
        return out

    def forward(self, x):
        return self.gamma * self._norm(x) + self.beta

if __name__ == "__main__":
    nums = [[[-0.2516, -0.1895, -0.3662, -0.3905, -0.8898],
             [-0.2124, 0.4689, -0.4415, -0.0924, -0.2313],
             [-1.8788, 0.2217, -0.5853, -0.9763, 1.2989]],
            [[-0.0598, -0.5630, -0.4373, 1.0963, 0.4904],
             [-0.5804, 0.5629, 0.5194, 0.0577, 0.3512],
             [-0.2438, 0.4798, 1.6694, 0.1164, 0.1718]]]
    x = torch.tensor(nums)
    ln = LayerNorm(dim=x.shape[-1])
    out = ln(x)
    print(f"x = {x}, \nout = {out}")