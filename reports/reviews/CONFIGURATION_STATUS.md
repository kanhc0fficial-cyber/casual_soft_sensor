# 数据配置状态报告

## 配置完成情况 ✅

### 1. 数据路径配置 ✅
- **原始数据**: `C:\Users\goldenwhale\Downloads\my_mining_simulation\output\simulation_2months_seq_hybrid_normal_fast_sampling.parquet`
- **数据验证**: 已确认文件存在
- **数据规模**: 86,400 行 × 266 列

### 2. DML残差软测量配置 ✅
**配置文件**: `configs/residual_soft_sensor.yaml`

已配置项目:
- ✅ 数据路径指向真实数据
- ✅ 目标变量设置为 `y_fx_xin1`
- ✅ 时间列设置为 `t`
- ✅ 操作变量 (17个) 已手工指定
- ✅ 排除列 (52个) 已配置，排除实验室数据和系统2目标
- ✅ 模型超参数已配置

### 3. 门控分组软测量配置 ✅
**配置文件**: `configs/group_branch.yaml`

已配置项目:
- ✅ 数据路径指向真实数据
- ✅ 目标变量设置为 `y_fx_xin1`
- ✅ 时间列设置为 `t`
- ✅ 排除列 (52个) 已配置
- ✅ 变量分组 (7组) 已根据工艺流程配置
- ✅ 模型超参数已配置

### 4. 测试配置 ✅
**配置文件**: `configs/group_branch_test.yaml`
- ✅ 创建了快速测试配置（5 epochs，简化分组）
- 用途：快速验证配置正确性

## 配置详情

### 操作变量 (A) - 17个

| 类别 | 变量名 | 说明 |
|------|--------|------|
| 药剂添加 | fx_s1_td_rough_freq | 粗选捕收剂频率 |
| 药剂添加 | fx_s1_naoh_freq | NaOH调整剂频率 |
| 药剂添加 | fx_s1_cao_freq | CaO调整剂频率 |
| 充气量 | fx_s1_cx1_air_sp | 粗选槽1充气设定 |
| 充气量 | fx_s1_cx2_air_sp | 粗选槽2充气设定 |
| 充气量 | fx_s1_cx3_air_sp | 粗选槽3充气设定 |
| 充气量 | fx_s1_jx_air_sp | 精选槽充气设定 |
| 充气量 | fx_s1_sx1_air_sp | 扫选槽1充气设定 |
| 充气量 | fx_s1_sx2_air_sp | 扫选槽2充气设定 |
| 充气量 | fx_s1_sx3_air_sp | 扫选槽3充气设定 |
| 液位控制 | fx_s1_cx1_level_valve_sp | 粗选槽1液位阀设定 |
| 液位控制 | fx_s1_cx2_level_valve_sp | 粗选槽2液位阀设定 |
| 液位控制 | fx_s1_cx3_level_valve_sp | 粗选槽3液位阀设定 |
| 温度控制 | fx_s1_tk1_steam_sp | 加温槽1蒸汽设定 |
| 温度控制 | fx_s1_tk2_steam_sp | 加温槽2蒸汽设定 |
| 温度控制 | fx_s1_tk3_steam_sp | 加温槽3蒸汽设定 |

### 变量分组 (门控分组模型) - 7组

| 组名 | 特征数 | Hidden Dim | 说明 |
|------|--------|------------|------|
| preprocessing | 29 | 48 | 磁选+塔磨+浓密机 |
| reagent | 12 | 32 | 药剂控制+pH |
| roughing | 24 | 40 | 粗选浮选槽 |
| cleaning | 8 | 32 | 精选浮选槽 |
| scavenging | 24 | 40 | 扫选浮选槽 |
| temperature | 9 | 24 | 温度控制 |
| auxiliary | 15 | 32 | 泵池+鼓风机等 |

**总计**: 121个特征被分组（剩余93个特征未使用，会触发警告）

## 潜在问题与建议

### ⚠️ 问题1: DAG文件缺失
**状态**: `data/features/global_edges.csv` 不存在

**影响**: 
- DML残差模型无法使用DAG进行工况变量(C)推断
- 将退回到启发式规则推断

**建议**:
1. 如果有因果发现结果，将边表CSV放到 `data/features/global_edges.csv`
2. 或者运行因果发现脚本生成DAG
3. 或者接受启发式规则推断（可能不够准确）

### ⚠️ 问题2: 训练时间较长
**原因**: 
- 数据量大 (86,400样本)
- 特征维度高 (214维)
- 窗口化后样本数更多

**建议**:
1. 使用GPU加速（如果可用）
2. 先用测试配置验证 (`group_branch_test.yaml`)
3. 考虑减少数据量或增大batch size

### ⚠️ 问题3: 特征覆盖不完整
**状态**: 门控分组模型只使用了121/214个特征

**影响**: 
- 会触发 `warn_unused_features` 警告
- 部分信息未被利用

**建议**:
1. 检查未使用的特征是否重要
2. 考虑添加更多分组或扩展现有分组
3. 或者设置 `warn_unused_features: false` 忽略警告

## 运行测试

### 快速测试（推荐先运行）
```bash
cd casual_soft_sensor
python scripts/train_group_branch.py --config configs/group_branch_test.yaml
```
预计时间: 5-10分钟

### 完整训练

#### 门控分组模型
```bash
cd casual_soft_sensor
python scripts/train_group_branch.py --config configs/group_branch.yaml
```
预计时间: 20-40分钟

#### DML残差模型
```bash
cd casual_soft_sensor
python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor.yaml
```
预计时间: 30-60分钟

## 输出位置

- 门控分组测试: `results/group_branch_test/`
- 门控分组完整: `results/group_branch/`
- DML残差模型: `results/residual_soft_sensor/`

## 下一步行动

1. ✅ **配置已完成** - 两个模型的配置文件都已正确设置
2. ⏭️ **运行快速测试** - 验证配置正确性
3. ⏭️ **检查测试结果** - 确认模型能正常训练
4. ⏭️ **运行完整训练** - 获得最终模型
5. ⏭️ **分析结果** - 对比两种模型性能

## 配置文件清单

| 文件 | 用途 | 状态 |
|------|------|------|
| `configs/residual_soft_sensor.yaml` | DML残差模型配置 | ✅ 已配置 |
| `configs/group_branch.yaml` | 门控分组模型配置 | ✅ 已配置 |
| `configs/group_branch_test.yaml` | 快速测试配置 | ✅ 已创建 |
| `check_features.py` | 特征索引查看工具 | ✅ 已创建 |
| `feature_index_mapping.txt` | 特征索引映射表 | ✅ 已生成 |
| `DATA_CONFIGURATION_SUMMARY.md` | 配置详细说明 | ✅ 已创建 |
| `CONFIGURATION_STATUS.md` | 本文件 | ✅ 当前文档 |

---

**配置完成时间**: 2026-05-15  
**配置人员**: Kiro AI Assistant  
**数据来源**: simulation_2months_seq_hybrid_normal_fast_sampling.parquet
