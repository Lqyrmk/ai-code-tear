from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LoRALayer(nn.Module):

    def __init__(self,
                 original_linear: nn.Linear,
                 rank: int = 4,
                 alpha: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()

        self.original_linear = original_linear
        # 冻结原始权重，不参与梯度更新
        for param in self.original_linear.parameters():
            param.requires_grad = False

        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.scaling = alpha / rank

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features

        # A: (rank, in_dim)
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

        # B: (out_dim, rank)
        self.B = nn.Parameter(torch.zeros(out_dim, rank))

        self.dropout = nn.Dropout(p=dropout)

        self.merged = False

    def forward(self, x):
        with torch.no_grad():
            original_output = self.original_linear(x)

        if self.merged:
            # original_output = (W_0 + ΔW) @ x
            return original_output

        # original_output = W_0 @ x
        lora_output = self.dropout(x) @ self.A.T @ self.B.T
        lora_output = lora_output * self.scaling

        return original_output + lora_output  # W_0 @ x + ΔW @ x

    def merge_weights(self):

        if self.merged:
            return

        with torch.no_grad():
            # ΔW = B @ A
            delta_weight = self.B @ self.A  # (out_dim, in_dim)
            self.original_linear.weight += delta_weight * self.scaling

        self.merged = True

    def unmerge_weights(self):

        if not self.merged:
            return

        with torch.no_grad():
            # ΔW = B @ A
            delta_weight = self.B @ self.A
            self.original_linear.weight -= delta_weight * self.scaling

        self.merged = False

    def get_lora_params(self):
        return [self.A, self.B]

# 工具函数
def apply_lora_to_model(model: nn.Module,
                        target_modules: List[str] = ["q_proj", "v_proj"],
                        rank: int = 4,
                        alpha: float = 1.0,
                        dropout: float = 0.1):
    """
    冻结 backbone
    并将模型中的 Linear 层替换为 LoRALayer
    """

    for p in model.parameters():
        p.requires_grad = False

    def _replace(module):
        for name, m in module.named_children():
            if name in target_modules and isinstance(m, nn.Linear):
                new_layer = LoRALayer(m, rank, alpha, dropout)
                setattr(module, name, new_layer)
            else:
                _replace(m)

    _replace(model)

    return model

def get_trainable_parameters(model: nn.Module):
    """
    统计可训练参数量
    """
    trainable = all_params = 0
    for p in model.parameters():
        all_params += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    print(f"\nTotal: {all_params:,} | Trainable: {trainable:,} | Ratio: {trainable/all_params:.4%}")
    return trainable, all_params



# 模型merge/unmerge

# 1.手动遍历 merge
# def merge_lora(model: nn.Module):
#     for module in model.modules():
#         if isinstance(module, LoRALayer):
#             module.merge_weights()

# 2.通过 apply 进行 merge
def merge_lora(model: nn.Module):

    def _merge(module: nn.Module):
        if isinstance(module, LoRALayer):
            module.merge_weights()

    model.apply(_merge)
    return model

def unmerge_lora(model: nn.Module):

    def _unmerge(module: nn.Module):
        if isinstance(module, LoRALayer):
            module.unmerge_weights()

    model.apply(_unmerge)
    return model


# 模拟一个简单的预训练模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.q_proj = nn.Linear(256, 256)  # 拟注意力 Q
        self.v_proj = nn.Linear(256, 256)  # 拟注意力 V
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        q = self.q_proj(x)
        v = self.v_proj(x)
        return self.fc2(q + v)

def show_lora(model):
    for name, module in model.named_modules():
        if isinstance(module, LoRALayer):
            # show lora params
            lora_A, lora_B = model.q_proj.get_lora_params()
            print(f"{name} | A: {lora_A.shape} | B: {lora_B.shape} | merged: {module.merged}")

if __name__ == "__main__":

    target_modules = ["q_proj", "v_proj"]
    rank = 16
    alpha = 1.0
    dropout = 0.1
    lr = 1e-4

    model = SimpleModel()

    # print(f"before: \n{model}")

    model = apply_lora_to_model(model, target_modules, rank, alpha, dropout)

    # print(f"after: \n{model}")

    # 检查参数状态
    # 原始 fc1 和 fc2 冻结
    # q_proj / v_proj 变为 LoRALayer
    for name, param in model.named_parameters():
        print(f"{name:20} | requires_grad: {param.requires_grad} | shape: {param.shape}")

    get_trainable_parameters(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params=trainable_params, lr=lr)

    # 模拟多步训练循环
    for step in range(3):
        x = torch.randn(4, 128)
        y = torch.randint(0, 10, (4,))

        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

        print(f"\n=== Step {step} | loss = {loss.item():.4f} ===")
        for name, param in model.named_parameters():
            if not param.requires_grad:
                status = "frozen"
                grad_info = param.grad  # None
            elif param.grad is None:
                status = "trainable"
                grad_info = "grad=None"
            else:
                status = "trainable"
                grad_info = f"grad_mean={param.grad.mean().item():+.6f}"
            print(f"{name:35} | {status:9} | {grad_info}")

    print("=" * 100)

    # test merge
    model.eval()

    test_x = torch.randn(2, 128)

    print("## Merged Model")
    model = merge_lora(model)
    show_lora(model)
    test_merged_out = model(x)
    # print(f"Merged Result: {test_merged_out}")

    print("=" * 100)

    print("## Unmerged Model")
    model = unmerge_lora(model)
    show_lora(model)
    test_unmerged_out = model(x)
    # print(f"Unmerged Result: {test_unmerged_out}")

    is_close = torch.allclose(test_merged_out, test_unmerged_out, atol=1e-5, rtol=1e-4)
    print(f"\nMerged output is equals to unmerged output: {is_close}")
