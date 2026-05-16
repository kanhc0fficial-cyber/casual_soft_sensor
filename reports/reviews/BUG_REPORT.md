# Bug Report

本报告基于对以下内容的 review：

- `scripts/train_dml_residual_soft_sensor.py`
- `configs/residual_soft_sensor.yaml`
- `results/residual_soft_sensor/run_log.txt`

## 结论摘要

当前脚本已经能跑通最小闭环，但存在几个会影响真实性、可配置性和变量角色报告完整性的缺陷。最严重的问题是：当真实数据路径不存在时，脚本会自动退化到**合成数据**并继续产出结果，这会让使用者误以为实验已经基于真实数据完成。

**所有 bug 均已在本次提交中修复。**

---

## Bug 1：真实数据缺失时自动使用合成数据，可能产出"伪成功"结果

- **严重级别**：高
- **状态**：✅ 已修复
- **位置**：
  - `scripts/train_dml_residual_soft_sensor.py:144-158`

### 现象

当 `data_path` 指向的 parquet 文件不存在时，脚本不会报错退出，而是直接调用 `_generate_synthetic_data()` 生成一份演示数据继续训练和输出指标。

### 风险

- 会掩盖真实配置错误；
- 会让 `metrics_compare.csv`、`predictions_test.csv` 等结果看起来"正常产出"，但其实完全不是基于真实工业数据。

### 修复方案

增加了 `allow_synthetic_demo` 配置项（默认 `false`）。数据文件缺失时：
- 若 `allow_synthetic_demo: false`（默认），直接抛 `FileNotFoundError` 终止运行；
- 若显式设置 `allow_synthetic_demo: true`，才允许退回合成演示数据。

---

## Bug 2：配置声称 `operation_vars` 留空时会自动推断，但真实数据路径下并未实现

- **严重级别**：高
- **状态**：✅ 已修复
- **位置**：
  - `configs/residual_soft_sensor.yaml:15-17`
  - `scripts/train_dml_residual_soft_sensor.py:185-204`

### 现象

配置文件注释声称"若留空列表，脚本将尝试从变量名规律自动推断"，但自动补全逻辑只存在于合成数据生成函数中，真实数据下 `A` 列会为空。

### 风险

- 用户会误以为留空也能自动识别真实数据中的 A；
- 真实数据下 `A` 为空，导致 `causal_input` 和 `dml_residual_soft_sensor` 退化甚至不可用。

### 修复方案

1. 更正了 YAML 注释，明确说明真实数据下必须手工填写；
2. 在 `infer_variable_roles()` 中，当 `operation_vars` 为空时发出 `WARNING` 日志。

---

## Bug 3：变量角色报告会直接跳过所有 object 列，导致 `variable_roles.csv` 不完整

- **严重级别**：中
- **状态**：✅ 已修复
- **位置**：
  - `scripts/train_dml_residual_soft_sensor.py:245-248`

### 现象

`select_dtypes(include=["datetime64", "object"])` 导致所有 object 列也被无声跳过，不出现在 `variable_roles.csv` 中。

### 风险

- 真实数据中存在字符串型 ID、批次号、设备标签等列时，这些列会无声消失；
- 用户无法从角色表知道这些列是被排除了还是脚本漏处理了。

### 修复方案

1. `datetime_cols` 只收集真正的 datetime 类型列；
2. 单独维护 `object_cols`；
3. 在角色分配循环中，对 object 列明确标记为 `excluded`，理由为"非数值列，未参与建模"。

---

## Bug 4：日志中的"全部特征列"与实际进入模型的列不一致

- **严重级别**：低
- **状态**：✅ 已修复
- **位置**：
  - `scripts/train_dml_residual_soft_sensor.py:1156-1177`

### 现象

日志先打印包含 `timestamp` 的特征列列表，但后续数值过滤会把 `timestamp` 丢掉，日志与实际建模列不符。

### 修复方案

将 `logger.info(f"全部特征列 ...")` 的打印移到数值过滤完成、`all_feature_cols` 更新之后，并改为"全部特征列（建模用数值列）"，保证打印的即为最终进入模型的列。同时去掉了冗余的两次 `select_dtypes` 调用（合并为一次）。

---

## 修复汇总

| 编号 | 标题 | 严重级别 | 状态 |
|------|------|----------|------|
| Bug 1 | 数据缺失时自动退化合成数据 | 高 | ✅ 已修复 |
| Bug 2 | `operation_vars` 留空行为与注释不符 | 高 | ✅ 已修复 |
| Bug 3 | object 列在角色表中无声消失 | 中 | ✅ 已修复 |
| Bug 4 | 日志与实际建模列不一致 | 低 | ✅ 已修复 |
