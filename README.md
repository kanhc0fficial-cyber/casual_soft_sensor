# casual_soft_sensor

本仓库当前主要包含一个用于验证 **DML 正交残差软测量** 思路的原型实现，以及若干与因果发现/因果分析相关的脚本。

## 当前重点脚本

- `scripts/train_dml_residual_soft_sensor.py`
- `configs/residual_soft_sensor.yaml`

## DML 正交残差软测量简介

普通软测量通常是：

```text
X_seq -> LSTM -> y
```

当前实现改成：

```text
C -> g(C) -> y_base
y_res = y - y_base

C -> q_A(C) -> A_res
C -> q_S(C) -> S_res

[A_res, S_res]_seq -> LSTM -> y_res_hat
y_hat = y_base + y_res_hat
```

其中：

- `y`：目标变量
- `A`：操作变量
- `C`：工况/混杂变量
- `S`：状态变量

核心思想是：先用 `C` 把慢变工况和混杂成分解释掉，再让时序模型只学习残差部分。

## 已实现内容

### 1. 变量角色推断

脚本会给每个变量分配以下角色之一：

- `target`
- `operation_A`
- `confounder_C`
- `state_S`
- `excluded`

角色推断会优先使用 DAG 边表；DAG 不足时会退回启发式规则。

### 2. 三组对比模型

脚本会输出三组模型结果：

1. `baseline`
   - 原始特征序列 -> LSTM -> y
2. `causal_input`
   - A + S -> LSTM -> y
3. `dml_residual_soft_sensor`
   - C -> y_base
   - C -> A/S residuals
   - residual LSTM -> y_res
   - `y_hat = y_base + y_res_hat`

### 3. 输出文件

运行后默认会在：

- `results/residual_soft_sensor/`

生成以下文件：

- `variable_roles.csv`
- `y_baseline_predictions.csv`
- `residual_feature_summary.csv`
- `predictions_test.csv`
- `metrics_compare.csv`
- `run_log.txt`

## 运行方式

在仓库根目录执行：

```bash
python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor.yaml
```

## 配置说明

配置文件：

- `configs/residual_soft_sensor.yaml`

关键字段：

- `data_path`：输入数据路径
- `dag_path`：DAG 边表路径
- `target_col`：目标变量名
- `time_col`：时间列名
- `operation_vars`：操作变量列表
- `exclude_cols`：排除列
- `window_size`：滑动窗口长度
- `train_ratio / val_ratio / test_ratio`：数据切分比例
- `residualization_model`：残差化基模型
- `residual_sequence_model`：残差序列模型
- `random_seed`：随机种子
- `output_dir`：输出目录

## 目录概览

目前仓库内较关键的内容包括：

- `src/target_causal_projection.py`
  - 用于从 DAG 边表抽取目标变量的父节点、祖先、后代和 DML 任务表
- `scripts/train_dml_residual_soft_sensor.py`
  - DML 残差软测量训练主脚本
- `configs/residual_soft_sensor.yaml`
  - 对应配置文件
- `results/residual_soft_sensor/`
  - 当前运行结果目录

## 当前已知问题

详见：

- `BUG_REPORT.md`

当前 review 中发现的主要问题包括：

- 数据文件缺失时会自动切换为合成数据；
- `operation_vars` 留空时，配置注释和真实行为不一致；
- `variable_roles.csv` 对 object 列覆盖不完整；
- 日志中的特征列与最终建模列不完全一致。

## 适用性说明

当前实现更适合作为：

- 原型验证
- 方法闭环演示
- 后续工程化实现的基础版本

如果要用于真实工业实验，建议先处理 `BUG_REPORT.md` 中列出的问题，再补充真实数据校验、配置检查和更严格的变量角色审计逻辑。
