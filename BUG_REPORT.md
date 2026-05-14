# Bug Report

本报告基于对以下内容的 review：

- `scripts/train_dml_residual_soft_sensor.py`
- `configs/residual_soft_sensor.yaml`
- `results/residual_soft_sensor/run_log.txt`

## 结论摘要

当前脚本已经能跑通最小闭环，但仍存在几个会影响真实性、可配置性和变量角色报告完整性的缺陷。最严重的问题是：当真实数据路径不存在时，脚本会自动退化到**合成数据**并继续产出结果，这会让使用者误以为实验已经基于真实数据完成。

---

## Bug 1：真实数据缺失时自动使用合成数据，可能产出“伪成功”结果

- **严重级别**：高
- **位置**：
  - `scripts/train_dml_residual_soft_sensor.py:144-158`
  - `results/residual_soft_sensor/run_log.txt:28-29`

### 现象

当 `data_path` 指向的 parquet 文件不存在时，脚本不会报错退出，而是直接调用 `_generate_synthetic_data()` 生成一份演示数据继续训练和输出指标。

### 风险

- 会掩盖真实配置错误；
- 会让 `metrics_compare.csv`、`predictions_test.csv` 等结果看起来“正常产出”，但其实完全不是基于真实工业数据；
- 与任务中“不要伪造结果”的要求相冲突。

### 证据

代码中：

```python
if data_path.exists():
    ...
else:
    logger.warning(f"数据文件不存在: {data_path}，使用合成演示数据")
    return _generate_synthetic_data(cfg, logger)
```

运行日志中也明确出现：

```text
数据文件不存在: data/modeling_dataset_xin2_final.parquet，使用合成演示数据
```

### 建议修复

- 默认行为应改为：**数据文件缺失时直接报错退出**；
- 若确实需要演示模式，应增加显式配置，例如 `allow_synthetic_demo: true`，默认关闭。

---

## Bug 2：配置声称 `operation_vars` 留空时会自动推断，但真实数据路径下并未实现

- **严重级别**：高
- **位置**：
  - `configs/residual_soft_sensor.yaml:15-17`
  - `scripts/train_dml_residual_soft_sensor.py:185-204`
  - `scripts/train_dml_residual_soft_sensor.py:239-275`

### 现象

配置文件写着：

```yaml
# 填入你项目中已知的操作变量名；若留空列表，脚本将尝试从变量名规律自动推断
operation_vars: []
```

但代码里真正“自动补上操作变量”的逻辑，只出现在 `_generate_synthetic_data()` 里。也就是说：

- **只有在使用合成数据时**，`operation_vars` 留空才会被补成 `["op_reagent_flow", "op_air_flow"]`；
- **真实数据路径下** 并没有实现自动识别 A 的逻辑。

### 风险

- 用户会误以为留空也能自动识别真实数据中的 A；
- 真实数据下 `A` 可能为空，导致 `causal_input` 和 `dml_residual_soft_sensor` 退化甚至不可用；
- 变量角色推断会偏离任务要求中“已有人工给出的 A”。

### 建议修复

- 如果当前版本不支持真实数据下自动识别 A，应把配置注释改为“必须手工填写”；
- 或者真正实现一套 `operation_vars` 自动识别规则。

---

## Bug 3：变量角色报告会直接跳过所有 object 列，导致 `variable_roles.csv` 不完整

- **严重级别**：中
- **位置**：
  - `scripts/train_dml_residual_soft_sensor.py:245-248`

### 现象

代码当前使用：

```python
datetime_cols = set(df.select_dtypes(include=["datetime64", "object"]).columns.tolist())
all_cols = [c for c in df.columns if c != time_col and c not in datetime_cols]
```

这意味着：

- 不仅 datetime 列被排除；
- **所有 object 列也会被直接跳过**；
- 它们既不会出现在 `variable_roles.csv` 里，也不会被标记为 `excluded`。

### 风险

- 变量角色表不再覆盖“所有原始列”；
- 如果真实数据中存在字符串型 ID、批次号、设备标签、工况编码等列，这些列会无声消失；
- 用户无法知道这些列是被排除了，还是脚本漏处理了。

### 建议修复

- 只自动跳过 `datetime` 类型列；
- 对 `object` 列应保留在角色表中，并明确标为 `excluded`，理由如“非数值列，未参与建模”。

---

## Bug 4：日志中的“全部特征列”与实际进入模型的列不一致

- **严重级别**：低
- **位置**：
  - `scripts/train_dml_residual_soft_sensor.py:1156-1177`
  - `results/residual_soft_sensor/run_log.txt:39`

### 现象

日志先打印：

```text
全部特征列 (8): ['timestamp', 'env_temperature', ...]
```

但后续真正进入 `df_model` 的只保留了数值列，`timestamp` 实际又被丢掉了。

### 风险

- 运行日志会误导使用者，以为时间列已经进入模型；
- 排查特征问题时会增加混乱。

### 建议修复

- 在打印“全部特征列”之前，先完成最终可建模列筛选；
- 或者分别打印“原始候选特征列”和“最终建模特征列”。

---

## 优先级建议

建议按以下顺序处理：

1. **先修 Bug 1**：禁止默认使用合成数据伪装真实实验；
2. **再修 Bug 2**：修正文档或实现 `operation_vars` 自动推断；
3. **再修 Bug 3**：补齐变量角色表覆盖范围；
4. **最后修 Bug 4**：统一日志与实际建模列。

---

## 总评

当前版本适合作为“原型验证脚本”，但还不适合直接作为真实工业数据实验脚本交付。主要原因不是模型结构本身，而是：

- 它会在关键输入缺失时自动伪造数据继续跑；
- 配置语义和实际行为存在不一致；
- 角色报告和日志还不够严格、可审计。
