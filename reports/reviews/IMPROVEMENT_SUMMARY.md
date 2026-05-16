# 因果发现改进措施总结

## 改进背景

根据初步诊断，发现以下问题：
1. 高度相关的变量对（相关系数 > 0.99）导致模型混淆
2. 因果信号极弱（大多数变量与目标变量相关性 < 0.1）
3. 超参数设置不当

## 实施的改进措施

### 1. 分阶段分析 ✅

**改进前**：
- 一次性分析所有 33 个浮选变量
- 包含两条产线（s1 和 s2）的对称变量

**改进后**：
- 按工艺阶段分别分析：
  - **磁选阶段**：12 个变量 → 移除 2 个 → 保留 10 个
  - **塔磨阶段**：18 个变量 → 移除 5 个 → 保留 13 个
  - **浮选阶段**：57 个变量 → 移除 10 个 → 保留 47 个
- **精矿品位（y_grade）在所有阶段中参与分析**

**优势**：
- 减少变量数量，降低模型复杂度
- 更容易发现阶段内的因果关系
- 便于工艺专家理解和验证

### 2. 去除对称变量 ✅

**改进前**：
- 保留两条产线（s1 和 s2）的所有对称变量
- 例如：`fx_s1_cx1_level` 和 `fx_s2_cx1_level` 高度相关（0.9996）

**改进后**：
- **只保留产线 s1 的变量**
- 自动移除高度相关的变量（相关系数 > 0.95）
- 生成详细的相关性分析报告

**移除的变量示例**（浮选阶段）：
- `fx_s1_td_rough_curr` ↔ `fx_s1_td_rough_freq` (0.9999)
- `fx_s1_td_clean_freq` ↔ `fx_s1_td_rough_freq` (0.9999)
- `fx_s1_k6_rough_freq` ↔ `fx_s1_td_rough_freq` (0.9999)
- `fx_s1_naoh_freq` ↔ `fx_s1_td_rough_freq` (0.9999)
- `fx_s1_cao_freq` ↔ `fx_s1_td_rough_freq` (0.9999)

### 3. 生成相关性报告 ✅

**新增功能**：
- 自动生成 Markdown 格式的相关性分析报告
- 报告包含：
  - 发现的高度相关变量对
  - 相关系数
  - 处理方式（保留哪个，移除哪个）
  - 移除的变量数和保留的变量数

**报告位置**：
- `因果发现结果/correlation_report_磁选_xin1.md`
- `因果发现结果/correlation_report_塔磨_xin1.md`
- `因果发现结果/correlation_report_浮选_xin1.md`

### 4. 调整超参数 ✅

**改进前**：
```python
LR = 0.003                    # 学习率过小
sparse_weight = 0.001         # 稀疏性权重过大
TOPO_PENALTY_WEIGHT = 10.0    # 拓扑惩罚权重过大
keep_ratio = 0.5              # 马尔可夫毯过小
```

**改进后**：
```python
LR = 0.005                    # 提高学习率
sparse_weight = 0.0001        # 降低稀疏性权重
TOPO_PENALTY_WEIGHT = 2.0     # 降低拓扑惩罚权重
keep_ratio = 0.7              # 扩大马尔可夫毯
```

### 5. 保持训练轮数不变 ✅

**用户要求**：
- 不增加训练轮数
- 保持 50 个 epoch

**原因**：
- 避免过长的训练时间
- 通过其他改进措施提升效果

## 改进效果预期

### 磁选阶段
- **变量数**：10 个（移除 2 个高度相关变量）
- **物理可行边**：预计 ~90 条
- **预期发现**：励磁电压/电流 → 液位 → 精矿品位

### 塔磨阶段
- **变量数**：13 个（移除 5 个高度相关变量）
- **物理可行边**：预计 ~156 条
- **预期发现**：泵频率 → 旋流器液位 → 精矿品位

### 浮选阶段
- **变量数**：47 个（移除 10 个高度相关变量）
- **物理可行边**：2209 条（97.9%）
- **预期发现**：
  - 充气量 → 泡沫高度 → 精矿品位
  - 液位 → 泡沫高度 → 精矿品位
  - 药剂频率 → pH → 精矿品位

## 技术细节

### 数据处理流程

```
原始数据 (86400, 266)
    ↓
按阶段选择变量
    ↓
计算 Spearman 相关系数
    ↓
移除高度相关变量（相关系数 > 0.95）
    ↓
全局 z-score 标准化
    ↓
构建时间窗口 (WINDOW_SIZE=15)
    ↓
训练 MB-CUTS (50 epoch)
    ↓
输出因果图 (GraphML)
```

### 物理拓扑约束

**阶段顺序**：
```
磁选 (0) → 塔磨 (1) → 浮选 (2) → Y (3)
```

**因果规则**：
1. 前序工序可以影响后序工序
2. 同一工序内，变量可以相互影响
3. 后序工序不能影响前序工序
4. 目标变量 Y 只能被其他变量指向，不能指向任何变量

### 算法参数

| 参数 | 值 | 说明 |
|------|-----|------|
| WINDOW_SIZE | 15 | 时间窗口大小（2.5 小时） |
| BATCH_SIZE | 32 | 批大小 |
| LR | 0.005 | 学习率 |
| EPOCHS | 50 | 训练轮数 |
| sparse_weight | 0.0001 | 稀疏性权重 |
| TOPO_PENALTY_WEIGHT | 2.0 | 拓扑惩罚权重 |
| keep_ratio | 0.7 | 马尔可夫毯保留比例 |
| correlation_threshold | 0.95 | 相关系数阈值 |
| adjacency_threshold | 0.02 | 邻接矩阵阈值 |

## 使用方法

### 运行单个阶段

```bash
# 浮选阶段
python run_causal_discovery_by_stage.py --stage 浮选 --line xin1 --algo mb_cuts --epochs 50

# 塔磨阶段
python run_causal_discovery_by_stage.py --stage 塔磨 --line xin1 --algo mb_cuts --epochs 50

# 磁选阶段
python run_causal_discovery_by_stage.py --stage 磁选 --line xin1 --algo mb_cuts --epochs 50
```

### 运行所有阶段

```bash
python run_causal_discovery_by_stage.py --stage all --line xin1 --algo mb_cuts --epochs 50
```

### 调整相关系数阈值

```bash
# 更严格的阈值（移除更多变量）
python run_causal_discovery_by_stage.py --stage 浮选 --correlation_threshold 0.90

# 更宽松的阈值（保留更多变量）
python run_causal_discovery_by_stage.py --stage 浮选 --correlation_threshold 0.98
```

## 输出文件

### 因果图（GraphML）
- `因果发现结果/mb_cuts_磁选_xin1.graphml`
- `因果发现结果/mb_cuts_塔磨_xin1.graphml`
- `因果发现结果/mb_cuts_浮选_xin1.graphml`

### 相关性报告（Markdown）
- `因果发现结果/correlation_report_磁选_xin1.md`
- `因果发现结果/correlation_report_塔磨_xin1.md`
- `因果发现结果/correlation_report_浮选_xin1.md`

## 下一步工作

1. ✅ 完成浮选阶段因果发现
2. ⏳ 运行塔磨阶段因果发现
3. ⏳ 运行磁选阶段因果发现
4. ⏳ 分析发现的因果关系
5. ⏳ 与工艺专家验证结果
6. ⏳ 根据结果调整软测量模型

---

**更新时间**：2026-05-15
**数据版本**：simulation_2months_upstream_visible_clean_fast_sampling.parquet
**配置文件**：causal_discovery_config_v2.py
