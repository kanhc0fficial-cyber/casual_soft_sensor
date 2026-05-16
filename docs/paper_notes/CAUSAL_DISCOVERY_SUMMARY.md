# 因果发现执行总结

## 任务概述

对 `simulation_2months_upstream_visible_clean_fast_sampling.parquet` 数据文件使用 MB-CUTS 算法进行因果发现。

## 执行结果

### 数据加载 ✅
- **文件**: simulation_2months_upstream_visible_clean_fast_sampling.parquet
- **大小**: 86,400 行 × 266 列
- **产线**: xin1 (Group A + Group C)
- **选中变量**: 33 个浮选工艺相关变量

### 因果发现 ⚠️
- **发现的因果关系**: 0 条边
- **原因**: 数据中的因果信号极弱，变量间高度相关

### 诊断分析 ✅
- **数据质量**: 良好（无缺失值，样本充足）
- **变量相关性**: 发现高度相关的变量对（相关系数 > 0.99）
- **因果信号**: 弱（大多数变量与目标变量相关性 < 0.1）

---

## 关键发现

### 1. 数据特性

**强相关性变量对**:
- `fx_s1_sx3_level` ↔ `fx_s2_sx3_level`: 0.9996
- `fx_s1_sx2_level` ↔ `fx_s2_sx2_level`: 0.9993
- `fx_s1_sx1_level` ↔ `fx_s2_sx1_level`: 0.9986
- `fx_s1_jx_level` ↔ `fx_s2_jx_level`: 0.9974

**问题**: 这些高度相关的变量对模型造成混淆，无法区分因果关系

### 2. 因果信号强度

**与目标变量的相关性**:
- 最强: `fx_s1_jx_air_flow` (0.0817)
- 平均: < 0.05
- 大多数: < 0.1

**问题**: 因果信号极弱，模型难以学习

### 3. 物理拓扑约束

- **可行边数**: 732 / 1,122 (65.2%)
- **工序分布**: 粗选 (9 变量) → 扫选 (18 变量) → 精选 (6 变量)

---

## 改进方案

### 立即可行 (优先级: 高)

1. **移除高度相关变量**
   - 从相关系数 > 0.95 的变量对中保留一个
   - 预期可减少 4-6 个变量

2. **调整超参数**
   - 降低稀疏性权重: 0.00001
   - 降低拓扑惩罚权重: 0.1
   - 增加学习率: 0.05
   - 增加训练轮数: 200

3. **扩大马尔可夫毯**
   - keep_ratio: 0.95 (从 0.7)

### 中期改进 (优先级: 中)

1. 特征工程（一阶差分、滑动平均）
2. 时间窗口优化（尝试 WINDOW_SIZE = 30, 60）
3 尝试其他算法（BiAttn-CUTS, MultiScale-NTS）

### 长期改进 (优先级: 低)

1. 领域知识融合
2. 多源数据融合
3 干预实验验证

---

## 生成的文件

### 报告文件
- `MB_CUTS_CAUSAL_DISCOVERY_REPORT.md` - 详细分析报告
- `CAUSAL_DISCOVERY_SUMMARY.md` - 本文件

### 诊断工具
- `diagnose_mb_cuts.py` - 数据诊断脚本

### 结果文件
- `因果发现结果/mb_cuts_real_dag_xin1.graphml` - 因果图（GraphML 格式）

---

## 建议的后续步骤

1. **立即执行**:
   ```bash
   # 1. 运行改进版本的 MB-CUTS
   python run_innovation_real_data.py --line xin1 --algo mb_cuts --epochs 200
   
   # 2. 尝试其他算法
   python run_innovation_real_data.py --line xin1 --algo multiscale_nts --epochs 200
   python run_innovation_real_data.py --line xin1 --algo biattn_cuts --epochs 200
   ```

2. **数据预处理**:
   - 创建去重版本的数据（移除高度相关变量）
   - 计算一阶差分特征
   - 尝试不同的标准化方法

3. **参数调优**:
   - 网格搜索最优的超参数组合
   - 尝试不同的时间窗口大小

4. **结果验证**:
   - 与工艺专家讨论发现的因果关系
   - 设计干预实验验证

---

## 技术细节

### 数据统计
- 样本数: 86,400
- 变量数: 33
- 缺失值: 0
- 时间窗口: 15 步

### 物理约束
- 粗选 → 扫选 → 精选 → Y
- 同工序同组变量可相互影响
- 任何变量可指向 Y
- Y 不能指向任何变量

### 算法参数
- 学习率: 0.005
- 批大小: 32
- 训练轮数: 50
- 稀疏性权重: 0.0001
- 拓扑惩罚权重: 2.0

---

**执行日期**: 2026-05-15
**数据版本**: simulation_2months_upstream_visible_clean_fast_sampling.parquet
**算法**: MB-CUTS (Markov Blanket-based Causal Discovery with Temporal Smoothness)
