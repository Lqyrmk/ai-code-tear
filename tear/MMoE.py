import torch
import torch.nn as nn
import torch.nn.functional as F

class MMoE(nn.Module):

    def __init__(
        self,
        in_dim,
        num_experts,
        expert_dim,
        num_tasks,
        task_dim
    ):
        super().__init__()

        self.num_tasks = num_tasks

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, expert_dim)
            ) for _ in range(num_experts)
        ])

        self.gates = nn.ModuleList([
            nn.Linear(in_dim, num_experts) for _ in range(num_tasks)
        ])

        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_dim, task_dim),
                nn.ReLU(),
                nn.Linear(task_dim, 1)
            ) for _ in range(num_tasks)
        ])

    def forward(self, x):

        expert_outs = [expert(x) for expert in self.experts]  # expert(x): [B, d]
        expert_outs = torch.stack(expert_outs, dim=1)  # [B, E, d]

        task_outs = []
        for i in range(self.num_tasks):

            gate_out = self.gates[i](x)
            gate_weight = F.softmax(gate_out, dim=-1)  # [B, E]

            # method 1: matrix multiplication
            # [B, 1, E] @ [B, E, d] -> [B, 1, d] -> [B, d]
            fused = torch.matmul(gate_weight.unsqueeze(1), expert_outs).squeeze(1)  # or torch.bmm
            # method 2: element-wise product and sum
            # [B, E, 1] * [B, E, d] -> [B, E, d] -> [B, d]
            fused = torch.sum(gate_weight.unsqueeze(-1) * expert_outs, dim=1)

            out = self.task_towers[i](fused)  # [B, 1]
            task_outs.append(out)

        return task_outs

if __name__ == '__main__':

    batch_size = 16
    in_dim = 10
    num_experts = 4
    expert_dim = 8
    num_tasks = 3
    task_dim = 8

    x = torch.randn(batch_size, in_dim)

    model = MMoE(in_dim, num_experts, expert_dim, num_tasks, task_dim)

    y = model(x)

    for i, pred in enumerate(y):
        print(f"task_{i}: {pred}")
