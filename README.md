# ai-code-tear

算法岗面试「手撕代码」。一个组件一个文件，依赖 PyTorch / NumPy。

```bash
uv sync
uv run python tear/Transformer.py    # 每个文件都是独立可执行脚本
uv run python main.py                # 冒烟测试：打印问候语与 CUDA 是否可用
```

## 环境

| 项 | 值 |
|---|---|
| Python | 3.12 |
| 包管理 | [uv](https://github.com/astral-sh/uv) |
| 依赖 | `torch >= 2.6.0` · `numpy >= 2.3.5` · `scikit-learn >= 1.8.0` |


## 实现清单

| 文件 | 实现 | 要点 | 框架 |
|---|---|---|---|
| `tear/RMSNorm.py` | RMSNorm | 不做均值中心化，`x · rsqrt(mean(x²)+eps)`，fp32 算完 `type_as` 回落 | PyTorch |
| `tear/LayerNorm.py` | LayerNorm | 手写 mean / var，注释给出 `x.var(unbiased=False)` 的等价写法 | PyTorch |
| `tear/MultiHeadAttention.py` | Self-Attention | `[B,L,D] → [B,H,L,dh]`，内置 causal mask，padding mask 留注释 | PyTorch |
| `tear/CrossAttention.py` | Cross-Attention | Q 取自 x、KV 取自 y，mask 给出数值加法与 bool `masked_fill` 两版 | PyTorch |
| `tear/GroupedQueryAttention.py` | GQA + KV-Cache | `repeat_kv` 广播展开，`past_kv` 沿 seq 维拼接，`use_cache` 返回新 KV | PyTorch |
| `tear/PositionalEncoding.py` | 正弦位置编码 | `pos · exp(-log(10000) · 2i/d)` 广播，`register_buffer` 不进梯度 | PyTorch |
| `tear/Transformer.py` | Transformer | Encoder / Decoder 全栈，附 `create_pad_mask`、`create_causal_mask`、`create_data` | PyTorch |
| `tear/CrossEntropyLoss.py` | BCE / CE | sigmoid 与 softmax 两版，`clip` 做数值稳定，`p[arange, y]` 取目标类 | NumPy |
| `tear/AUC.py` | AUC | 三版对照：暴力 O(MN)、排序秩和 O(n log n)、并列分数平均秩 | NumPy |
| `tear/KMeans/KMeans_numpy.py` | K-Means | 随机初始化、空簇兜底、`tol` 早停，`‖x−c‖²` 用展开式一次算完 | NumPy |
| `tear/KMeans/KMeans_pytorch.py` | K-Means | 上述算法的 GPU 版，`clamp(min=0)` 吸收展开式的浮点负数 | PyTorch |
| `tear/KMeans/RQ_KMeans.py` | RQ-KMeans | 残差量化：逐层拟合码本，`encode` 成离散 token、`decode` 重建并报 MSE | PyTorch |
| `tear/MMoE.py` | MMoE | 共享专家 + 每任务 gate + tower，融合给出矩阵乘与逐元素两版 | PyTorch |
| `tear/LoRA.py` | LoRA | 冻结 backbone、A/B 低秩分解、`merge` / `unmerge`，`allclose` 自证等价 | PyTorch |
| `tear/LinearRegression/01_full_pytorch.py` | 线性回归 | `nn.Parameter` + `torch.optim.SGD` + `backward()` | PyTorch |
| `tear/LinearRegression/02_manual_gd_pytorch.py` | 线性回归 | 不用 optimizer：`no_grad` 下手写 `w -= lr * w.grad` 与梯度清零 | PyTorch |
| `tear/MLP/01_manual_gd_pytorch.py` | 两层 MLP | 手写前向 + 手写梯度下降，`torch.maximum` 实现 ReLU | PyTorch |
| `written_test/LogisticRegression.py` | 对数几率回归 | 仅用 NumPy，IRLS（牛顿法）解正规方程 | NumPy |
| `written_test/GRU.py` | 单层 GRU 前向 | 权重按 `[r; z; h]` 切块，`h = (1−z)·h + z·h̃` | NumPy |
| `written_test/OCSVM.py` | One-Class SVM | 标准化 + `gamma × nu` 网格搜索（AUC / F1 双指标）+ 全量重训 | sklearn |
| `written_test/MMR.py` | MMR 重排 | `λ·score − (1−λ)·类别新颖度`，同分按 id 稳定排序 | 纯 Python |

## 笔试

题干在文件头 docstring 里（偏个人娱乐）。

| 文件 | 出处 | 考点 |
|---|---|---|
| `written_test/LogisticRegression.py` | 美团 2026 春招第四场 | 仅用 NumPy 实现 LR |
| `written_test/GRU.py` | 美团 2026 春招第二场 | 单层 GRU 前向 |
| `written_test/OCSVM.py` | 美团 2026 春招第三场 | 网格搜索与指标选择 |
| `written_test/MMR.py` | 京东 2026 秋招 | 重排多样性 |

## 目录结构

```
tear/              面试手撕实现
written_test/      笔试真题复现
main.py            环境冒烟测试
practice/    本地练习草稿（.gitignore 排除，不随仓库分发）
```

喜欢的话点个 Star 谢谢喵~