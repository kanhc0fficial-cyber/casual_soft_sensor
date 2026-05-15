# README — PR2：工艺因果组分支软测量模型（临时说明文档）

> **此文档为临时说明，由 PR2 自动生成，后续由项目维护者整理。**

---

## 本 PR 新增内容

| 文件/目录 | 说明 |
|---|---|
| `src/models/group_branch.py` | 工艺因果组分支软测量模型核心实现 |
| `scripts/train_group_branch.py` | 训练脚本（含数据加载、窗口化、训练、评估、保存） |
| `configs/group_branch.yaml` | 默认配置文件（合成演示数据 + 3 个变量组） |
| `configs/ablation/` | 消融实验配置（不同分支类型、gate 设置） |
| `tests/test_group_branch.py` | 轻量 pytest 测试套件（57 个用例，全部通过） |
| `docs/BUG_REPORT_PR2.md` | 代码审查 + 测试发现的 bug 文档 |

---

## 关于"库里没有数据"

这是正常的。真实工业数据集（`data/modeling_dataset_xin2_final.parquet`）
**不会提交到 Git 仓库**，原因是数据涉及实际工艺参数，不适合公开存储。

训练脚本提供了两种运行模式：

| 模式 | 条件 | 说明 |
|------|------|------|
| **演示模式** | 配置中设置 `allow_synthetic_demo: true`（当前默认） | 若 `data/` 下无 parquet 文件，自动生成 2000 条合成数据运行完整流程 |
| **真实数据模式** | 将 parquet 文件放到 `data/` 目录 | 脚本自动检测并读取真实数据 |

> 当前 `configs/group_branch.yaml` 已设置 `allow_synthetic_demo: true`，
> 克隆仓库后可以直接运行，无需任何额外数据文件。

---

## 快速上手

```bash
# 1. 安装依赖
pip install torch numpy pandas scikit-learn pyyaml

# 2. 用合成演示数据运行完整训练流程
python scripts/train_group_branch.py --config configs/group_branch.yaml

# 3. 查看结果
ls results/group_branch/
#   group_branch_metrics.csv      ← MAE / RMSE / R2
#   group_branch_gates.csv        ← 各组 gate 值
#   group_branch_contributions.csv← 各组平均贡献
#   predictions_test.csv          ← 测试集逐样本预测
#   run_log.txt                   ← 训练日志
```

---

## 模型架构简介

```
输入 X: [batch, window_size, num_features]
         ↓
┌─────────────────────────────────────────────┐
│  Group 0 (indices=[0,1])  → Branch → z_0   │  × gate_0
│  Group 1 (indices=[2,3])  → Branch → z_1   │  × gate_1
│  Group 2 (indices=[4,5,6])→ Branch → z_2   │  × gate_2
└─────────────────────────────────────────────┘
         ↓
  y_hat = bias + Σ gate_k * z_k   [batch, 1]
```

支持的分支类型（通过 `branch_type` 配置）：

| 类型 | 说明 |
|------|------|
| `gru` | GRU + 全连接，适合时序依赖强的变量组 |
| `mlp` | 时间窗口展平后两层 MLP，计算轻量 |
| `tcn` | 两层因果卷积，适合捕捉局部时序模式 |

---

## 配置说明

`configs/group_branch.yaml` 主要配置项：

```yaml
# 数据
data_path: "data/modeling_dataset_xin2_final.parquet"
allow_synthetic_demo: true   # 没有真实数据时使用合成数据

# 训练
window_size: 12
epochs: 50
batch_size: 64
lr: 0.001
patience: 8

# 变量分组（按特征索引分组，索引从 0 开始）
groups:
  feed:
    indices: [0, 1]
    branch_type: "gru"
    hidden_dim: 32
  reagent:
    indices: [2, 3]
    branch_type: "gru"
    hidden_dim: 32
  state:
    indices: [4, 5, 6]
    branch_type: "gru"
    hidden_dim: 32

# 模型
model:
  use_gate: true
  trainable_gate: true
  gate_init: 0.5
  output_bias: true
```

> 特征索引请参考脚本启动时打印的"特征列-索引对照表"，真实数据下需手工填写。

---

## 运行测试

```bash
pytest tests/test_group_branch.py -v
# 预期：57 passed
```

测试覆盖范围：
- 三种分支模块的正向传播形状验证（含边缘 window_size）
- 模型初始化的各类异常校验（空配置、索引越界、特征重叠等）
- 门控机制（可训练 / 固定 / 关闭）
- 梯度流（反向传播 + gate 梯度）
- `make_windows` 正常 + 边界情况（含已知 Bug 行为记录）
- `split_data` 正常切分 + 已知 bug 行为记录
- `compute_metrics` 完美预测 / 有误差情况

---

## 已知 Bug（详见 `docs/BUG_REPORT_PR2.md`）

| 编号 | 描述 | 严重级别 |
|------|------|----------|
| Bug 1 | `test_ratio` 配置项被静默忽略，实际 test 集大小由前两个比例决定 | 中 |
| Bug 2 | `window_size > 分段数据行数` 时静默返回空数组，下游触发 `ZeroDivisionError` | 高 |
| Bug 3 | `train/val/test ratio` 之和无校验，配置错误时静默产出错误切分 | 中 |
| Bug 4 | IQR 离群值裁剪用了 1%/99% 分位数但命名为 q1/q3，裁剪近乎无效 | 低 |
| Bug 5 | TCN 分支用对称 padding + 手动裁剪模拟因果性，注释与 PyTorch 语义不符 | 低 |

---

*生成时间：2026-05-15 | 分支：`copilot/add-process-causal-branch-soft-sensor`*
