# 约束消融实验说明

## 概述

已成功修改 `train_dml_residual_soft_sensor.py` 脚本，添加了跳过模型的功能，并创建了4个消融实验配置文件，用于测试工艺约束和反事实约束对 Model 3（DML残差软测量）性能的影响。

## 新增命令行参数

```bash
--skip-model0          # 跳过 Model 0（基线软测量）
--skip-model1          # 跳过 Model 1（因果输入软测量）
--skip-model2          # 跳过 Model 2（DML效应权重软测量）
--only-model3          # 只运行 Model 3（等同于同时设置上述三个参数）
```

## 消融实验配置文件

### 1. 基线（无约束）
**配置文件**: `configs/ablation_constraints_baseline.yaml`
- `use_counterfactual_constraint: false`
- `use_process_constraint: false`
- **输出目录**: `results/ablation_constraints/baseline_no_constraints/`

### 2. 仅反事实约束
**配置文件**: `configs/ablation_constraints_counterfactual.yaml`
- `use_counterfactual_constraint: true` (lambda=0.1)
- `use_process_constraint: false`
- **输出目录**: `results/ablation_constraints/counterfactual_only/`
- **约束说明**: 基于DML因果效应的单调性约束，自动从DML效应表中推断

### 3. 仅工艺约束
**配置文件**: `configs/ablation_constraints_process.yaml`
- `use_counterfactual_constraint: false`
- `use_process_constraint: true` (lambda=0.1)
- **输出目录**: `results/ablation_constraints/process_only/`
- **约束规则**:
  - `td_rough_freq_positive`: 捕收剂添加频率增加 → 品位提升
  - `cx1_air_positive`: 粗选1充气量增加 → 品位提升（在30%-90%分位数范围内）
  - `cx2_air_positive`: 粗选2充气量增加 → 品位提升（在30%-90%分位数范围内）

### 4. 两种约束
**配置文件**: `configs/ablation_constraints_both.yaml`
- `use_counterfactual_constraint: true`
- `use_process_constraint: true`
- **输出目录**: `results/ablation_constraints/both_constraints/`

## 运行方式

### 单独运行某个实验

```bash
cd casual_soft_sensor

# 实验1：基线（无约束）
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_baseline.yaml \
    --only-model3

# 实验2：仅反事实约束
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_counterfactual.yaml \
    --only-model3

# 实验3：仅工艺约束
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_process.yaml \
    --only-model3

# 实验4：两种约束
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_both.yaml \
    --only-model3
```

### 批量运行所有4个实验

```bash
cd casual_soft_sensor
bash run_ablation_constraints.sh
```

## 结果分析

每个实验会在各自的输出目录中生成以下文件：

- `run_log.txt`: 完整的运行日志
- `metrics_compare.csv`: 各模型的性能对比（重点关注 Model 3）
- `dml_residual_predictions_test.csv`: Model 3 的测试集预测结果
- `counterfactual_violations_test.csv`: 反事实约束违反情况（如果启用）
- `process_violations_test.csv`: 工艺约束违反情况（如果启用）
- `process_rules_screened.csv`: 筛选后的工艺约束规则（如果启用）

### 关键指标对比

查看各实验的 `metrics_compare.csv` 文件，对比 Model 3 的以下指标：

- **MAE** (Mean Absolute Error): 平均绝对误差，越小越好
- **RMSE** (Root Mean Squared Error): 均方根误差，越小越好
- **R²** (R-squared): 决定系数，越接近1越好

### 约束违反率分析

如果启用了约束，查看违反率文件：

- `violation_rate`: 约束违反的比例（0-1之间，越小越好）
- `mean_violation_magnitude`: 平均违反幅度
- `mean_expected_response`: 平均期望响应（应为正值）

## 约束配置说明

### 反事实约束参数

```yaml
use_counterfactual_constraint: true
counterfactual_lambda: 0.1                    # 约束权重
counterfactual_delta_std: 0.05                # 扰动幅度（标准化后）
counterfactual_apply_to: "operation_A_only"  # 只对操作变量应用
counterfactual_min_abs_effect: 0.01          # 最小效应阈值
counterfactual_require_recommended: true      # 只使用推荐的DML效应
counterfactual_use_last_step_only: true       # 只在最后时间步应用
```

### 工艺约束参数

```yaml
use_process_constraint: true
process_lambda: 0.1                           # 约束权重
process_delta_std: 0.05                       # 扰动幅度
process_require_dml_agree: true               # 要求与DML方向一致
process_use_in_train_default: true            # 训练时启用
process_use_in_eval_default: true             # 评估时启用

process_constraints:
  - name: "rule_name"
    variable: "variable_name"                 # 变量名
    direction: 1                              # 1=正向，-1=负向
    lag: 0                                    # 滞后步数
    delta_std: 0.05                           # 扰动幅度
    lambda: 0.1                               # 规则权重
    require_dml_agree: true                   # 要求与DML一致
    min_abs_dml_effect: 0.01                  # 最小DML效应
    use_in_train: true                        # 训练时使用
    use_in_eval: true                         # 评估时使用
    active_region:                            # 有效区域
      type: "all"                             # all | quantile_range | value_range
```

## 注意事项

1. **训练时间**: 每个实验大约需要10-30分钟（取决于硬件配置）
2. **随机种子**: 所有实验使用相同的随机种子（42）保证可复现性
3. **约束只影响 Model 3**: Model 0-2 不受约束影响（但已跳过以节省时间）
4. **日志编码问题**: 如果遇到 Unicode 编码错误，这是日志输出的问题，不影响模型训练和结果保存

## 预期结果

根据约束的理论基础，预期结果：

1. **基线（无约束）**: 提供性能基准
2. **仅反事实约束**: 应该提升模型的因果一致性，可能略微提升或保持性能
3. **仅工艺约束**: 应该提升模型符合领域知识的程度，可能提升性能
4. **两种约束**: 结合两种约束的优势，预期获得最佳的因果一致性和领域知识符合度

## 故障排除

### 如果遇到 "ValueError: The truth value of a DataFrame is ambiguous"

这个问题已经修复。如果仍然出现，请确保使用最新版本的脚本。

### 如果训练时间过长

可以修改配置文件中的以下参数来加快训练：

```yaml
lstm_epochs: 20          # 减少训练轮数（默认50）
lstm_batch_size: 128     # 增大批次大小（默认64）
lstm_hidden_size: 32     # 减小隐藏层大小（默认64）
lstm_num_layers: 1       # 减少LSTM层数（默认2）
```

## 后续分析

完成所有实验后，可以：

1. 对比4个实验的 `metrics_compare.csv` 文件
2. 分析约束违反率的变化趋势
3. 可视化预测结果对比
4. 撰写消融实验报告

---

**创建日期**: 2026-05-16  
**版本**: 1.0
