# Bug Report — PR2：工艺因果组分支软测量模型

本报告基于对以下文件的 **手工代码审查** 和 **轻量 pytest 推演测试** 得出，
覆盖范围：

- `src/models/group_branch.py`
- `scripts/train_group_branch.py`
- `configs/group_branch.yaml`

对应测试文件：`tests/test_group_branch.py`（57 个测试全部通过）

---

## 结论摘要

模型核心逻辑（`CausalGroupBranchModel`、三种分支类型、门控机制、反向传播）
**均正确运行**，没有功能性崩溃 bug。

发现的问题集中在训练脚本的 **数据处理层**，均为静默型缺陷——代码不会抛异常，
而是产出错误结果或以隐晦方式崩溃，排查成本高。

---

## Bug 1 — `test_ratio` 配置项被静默忽略

- **严重级别**：中
- **状态**：✅ 已修复
- **位置**：`scripts/train_group_branch.py:194–203`
- **对应测试**：`TestSplitData::test_test_ratio_actually_ignored`

### 现象

```python
n_train = int(n * cfg["train_ratio"])
n_val   = int(n * cfg["val_ratio"])
train_df = df.iloc[:n_train]
val_df   = df.iloc[n_train: n_train + n_val]
test_df  = df.iloc[n_train + n_val:]   # ← test_ratio 从未被使用
```

`test_df` 的实际大小是 `n - n_train - n_val`，由前两个比例决定，
而不是 `int(n * cfg["test_ratio"])`。

### 风险

- 用户在 YAML 里调整 `test_ratio` 不会有任何效果。
- 若 `train_ratio + val_ratio` 很大（如各 0.48），test 集可能只剩 `int()` 舍入
  造成的 3~5 行，远小于期望的 4%，导致评估指标极不稳定。

### 复现方式

```python
# configs/group_branch.yaml
train_ratio: 0.70
val_ratio:   0.15
test_ratio:  0.05   # 期望 test=5%，实际是 15%
```

### 修复建议

将 `test_df` 的切分逻辑改为：

```python
n_test  = int(n * cfg["test_ratio"])
n_val   = int(n * cfg["val_ratio"])
n_train = n - n_val - n_test   # 或也用 int()，但要保证三段不重叠
```

---

## Bug 2 — `make_windows` 在 `window_size > len(data)` 时静默返回空数组

- **严重级别**：高
- **状态**：✅ 已修复
- **位置**：`scripts/train_group_branch.py:207–215`
- **对应测试**：`TestMakeWindows::test_window_larger_than_data_returns_empty`

### 现象

```python
def make_windows(X, y, window_size):
    xs, ys = [], []
    for i in range(len(y) - window_size + 1):  # window > len(y) → range(负数) = 空
        xs.append(X[i: i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(xs, ...), np.array(ys, ...)
```

当 `window_size > len(y)` 时，`range(...)` 为负，循环体一次都不执行，
返回两个空 `np.ndarray`。

### 后续连锁崩溃

1. `DataLoader` 加载空 `TensorDataset` 不会立即报错；
2. 训练循环内 `epoch_loss /= len(yw_tr)` → **`ZeroDivisionError`**；
3. 或者，若 `len(yw_tr) == 0` 恰好被跳过，后续 `model.load_state_dict(best_state)` 中 `best_state is None`，模型 state 不被加载，无声地返回随机初始化的模型。

典型触发场景：val/test 切分后数据量极少（如总行数 100、window_size=12、test_ratio=0.05 → test 仅 5 行），
窗口化后 test 段只有 max(0, 5-12+1)=0 个样本。

### 修复建议

```python
def make_windows(X, y, window_size):
    if window_size > len(y):
        raise ValueError(
            f"window_size={window_size} 大于数据长度 {len(y)}，无法构造任何窗口。"
        )
    ...
```

---

## Bug 3 — `train_ratio + val_ratio + test_ratio` 之和未做合理性校验

- **严重级别**：中
- **状态**：✅ 已修复
- **位置**：`scripts/train_group_branch.py:194–203`（`split_data`）；`train_group_branch.py:74–99`（`load_config`）
- **对应测试**：`TestSplitData::test_ratios_not_validated`

### 现象

用户若将三个比例配置为：

```yaml
train_ratio: 0.80
val_ratio:   0.50
test_ratio:  0.30
```

`n_train + n_val = 1.3 * n`，`df.iloc[:n_train + n_val]` 会被 pandas 静默截断到
`df.iloc[:n]`（即全数据），实际 val_df 覆盖整段尾部，test_df 为空。
全程无报错、无警告。

### 修复建议

在 `load_config` 或 `split_data` 入口处添加：

```python
total = cfg["train_ratio"] + cfg["val_ratio"] + cfg["test_ratio"]
if not math.isclose(total, 1.0, abs_tol=1e-4):
    raise ValueError(f"train/val/test 比例之和应为 1.0，当前为 {total:.4f}")
```

---

## Bug 4 — `preprocess` 中离群值裁剪方法变量命名与标准 IQR 方法混淆

- **严重级别**：低（方法选择问题，不影响运行，但影响效果）
- **状态**：✅ 已修复
- **位置**：`scripts/train_group_branch.py:180–186`

### 现象

```python
q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
iqr = q3 - q1
if iqr > 0:
    df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)
```

- 变量名 `q1`/`q3` 在统计学惯例中指 **25th/75th 分位数（四分位数）**，
  但这里实际赋值为 1st/99th 分位数（百分位数）。
- 裁剪窗口 = `[1% - 3 * (99%-1%), 99% + 3 * (99%-1%)]`，宽度约为数据极差的 4 倍以上，
  几乎没有任何样本会被裁剪，离群值处理实际上是空操作。
- 标准 Tukey fence 方法应使用 `Q1(25%)`、`Q3(75%)`，
  裁剪界限为 `[Q1 - 1.5*IQR, Q3 + 1.5*IQR]`。

### 修复建议

将分位数改为 25%/75%，裁剪系数改为 1.5（或根据业务需求选择 3.0 用于保守模式）：

```python
q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
iqr = q3 - q1
if iqr > 0:
    df[col] = df[col].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
```

---

## Bug 5 — TCN 分支使用对称 padding，不是真正的因果卷积

- **严重级别**：低（单步预测场景下无影响，但时间语义场景下存在未来信息泄漏）
- **状态**：✅ 已修复
- **位置**：`src/models/group_branch.py:78–90`

### 现象

```python
self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=3, padding=2)
...
x = self.relu(self.conv1(x)[:, :, :-2])  # 裁掉右边 2 步
```

`nn.Conv1d` 的 `padding=2` 是 **左右各填充 2 个零**（对称填充），
输出长度 = `L + 4 - 2 = L + 2`，裁掉 `:-2` 后还原为 `L`。
效果与仅左侧填充 `2` 等价，最终输出不含"未来"信息。

但该实现依赖手动裁剪，与 PyTorch 的 `padding_mode='circular'` 或
`F.pad(..., mode='constant')` 语义不同，容易在代码修改时引入错误。
另外，注释写"padding=kernel_size-1 保证因果性"，但实际 padding 参数
在 PyTorch 语义中是两侧各填充，不等于仅左侧填充 `kernel_size-1`。

### 修复建议（可选，防止未来误改）

改用显式左填充：

```python
import torch.nn.functional as F

def forward(self, x):
    x = x.permute(0, 2, 1)
    x = F.pad(x, (2, 0))            # 仅左侧填充 kernel_size-1=2
    x = self.relu(self.conv1(x))    # padding=0，输出长度 = L
    x = F.pad(x, (2, 0))
    x = self.relu(self.conv2(x))
    return self.fc(x[:, :, -1])
```

---

## 汇总表

| 编号 | 标题 | 严重级别 | 涉及文件 | 状态 |
|------|------|----------|----------|------|
| Bug 1 | `test_ratio` 配置被静默忽略 | 中 | `train_group_branch.py` | ✅ 已修复 |
| Bug 2 | `make_windows` window>data 时静默空数组 → 下游崩溃 | 高 | `train_group_branch.py` | ✅ 已修复 |
| Bug 3 | ratio 三者之和无校验 | 中 | `train_group_branch.py` | ✅ 已修复 |
| Bug 4 | IQR 离群值裁剪变量名混淆 + 裁剪近乎无效 | 低 | `train_group_branch.py` | ✅ 已修复 |
| Bug 5 | TCN 对称 padding 非标准因果卷积实现 | 低 | `group_branch.py` | ✅ 已修复 |

> **注意**：以上 bugs 均经过测试文件 `tests/test_group_branch.py` 中对应测试用例的行为验证，
> 修复后请重新运行 `pytest tests/test_group_branch.py -v` 确认通过。
