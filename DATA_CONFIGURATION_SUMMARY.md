# 数据配置总结

## 配置完成时间
2026-05-15

## 数据源
- **原始数据路径**: `C:\Users\goldenwhale\Downloads\my_mining_simulation\output\simulation_2months_seq_hybrid_normal_fast_sampling.parquet`
- **数据规模**: 86,400 行 × 266 列
- **时间跨度**: 2个月的模拟数据（10秒采样间隔）

## 目标变量
- **主目标**: `y_fx_xin1` - 浮选系统1精矿品位
- **备选目标**: `y_fx_xin2` - 浮选系统2精矿品位（当前配置中已排除）

## 特征配置

### 1. DML 残差软测量 (train_dml_residual_soft_sensor.py)

**配置文件**: `configs/residual_soft_sensor.yaml`

#### 操作变量 (A) - 17个
人工指定的可控操作变量：
- **药剂添加** (3个): 
  - `fx_s1_td_rough_freq` - 粗选捕收剂频率
  - `fx_s1_naoh_freq` - NaOH调整剂频率
  - `fx_s1_cao_freq` - CaO调整剂频率

- **充气量设定** (7个):
  - 粗选: `fx_s1_cx1_air_sp`, `fx_s1_cx2_air_sp`, `fx_s1_cx3_air_sp`
  - 精选: `fx_s1_jx_air_sp`
  - 扫选: `fx_s1_sx1_air_sp`, `fx_s1_sx2_air_sp`, `fx_s1_sx3_air_sp`

- **液位控制** (3个):
  - `fx_s1_cx1_level_valve_sp`, `fx_s1_cx2_level_valve_sp`, `fx_s1_cx3_level_valve_sp`

- **温度控制** (3个):
  - `fx_s1_tk1_steam_sp`, `fx_s1_tk2_steam_sp`, `fx_s1_tk3_steam_sp`

#### 工况变量 (C)
由脚本根据DAG自动推断（如果DAG文件存在）或使用启发式规则：
- 前处理阶段变量（磁选、塔磨）
- 给矿条件相关变量
- 外部环境变量

#### 状态变量 (S)
除A、C、目标变量和排除变量外的所有数值变量

#### 排除变量 - 52个
- `y_fx_xin2` - 系统2目标变量
- 所有实验室化验数据 (lab_* 开头，共51个)
  - 原因：离线数据，不适合实时软测量

### 2. 门控分组软测量 (train_group_branch.py)

**配置文件**: `configs/group_branch.yaml`

#### 变量分组 - 7组

1. **preprocessing** (前处理阶段) - 29个特征
   - 磁选阶段 (12个): 励磁、阀门、液位等
   - 塔磨阶段 (13个): 旋流器、泵、液位等
   - 浓密机 (4个): 电流、浓度
   - Hidden dim: 48

2. **reagent** (药剂控制) - 12个特征
   - 捕收剂频率/电流
   - 调整剂频率/电流
   - pH值
   - 药剂槽液位
   - Hidden dim: 32

3. **roughing** (粗选) - 24个特征
   - 3个粗选槽 (cx1, cx2, cx3)
   - 泡沫层高度、液位、充气量、阀门位置
   - 搅拌电机电流
   - Hidden dim: 40

4. **cleaning** (精选) - 8个特征
   - 1个精选槽 (jx)
   - 泡沫层高度、液位、充气量、阀门位置
   - 搅拌电机电流
   - Hidden dim: 32

5. **scavenging** (扫选) - 24个特征
   - 3个扫选槽 (sx1, sx2, sx3)
   - 泡沫层高度、液位、充气量、阀门位置
   - 搅拌电机电流
   - Hidden dim: 40

6. **temperature** (温度控制) - 9个特征
   - 3个加温槽温度
   - 蒸汽设定值和反馈值
   - Hidden dim: 24

7. **auxiliary** (辅助系统) - 15个特征
   - 3个泵池液位和泵频率/电流
   - 鼓风机压力
   - 其他辅助设备
   - Hidden dim: 32

**总计**: 214个特征（排除目标变量、时间列和实验室数据后）

## 模型配置

### 共同参数
- **窗口大小**: 12 (对应2分钟历史数据)
- **数据切分**: 训练70% / 验证15% / 测试15%
- **随机种子**: 42
- **时间列**: `t`

### DML残差模型特定参数
- **残差化模型**: LightGBM
- **序列模型**: LSTM
  - Hidden size: 64
  - Layers: 2
  - Dropout: 0.1
  - Epochs: 50
  - Batch size: 64
  - Learning rate: 0.001
  - Patience: 8

### 门控分组模型特定参数
- **分支类型**: GRU (所有分组)
- **门控**: 可训练，初始值0.5
- **输出偏置**: 启用
- **损失函数**: MSE
- **训练参数**:
  - Epochs: 50
  - Batch size: 64
  - Learning rate: 0.001
  - Patience: 8

## 运行命令

### 1. DML残差软测量
```bash
cd casual_soft_sensor
python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor.yaml
```

### 2. 门控分组软测量
```bash
cd casual_soft_sensor
python scripts/train_group_branch.py --config configs/group_branch.yaml
```

## 输出结果

### DML残差软测量输出 (results/residual_soft_sensor/)
- `variable_roles.csv` - 变量角色分类
- `residual_feature_summary.csv` - 残差化特征统计
- `y_baseline_predictions.csv` - 基线预测
- `predictions_test.csv` - 测试集预测
- `metrics_compare.csv` - 模型对比指标
- `run_log.txt` - 训练日志

### 门控分组软测量输出 (results/group_branch/)
- `group_branch_metrics.csv` - 评估指标
- `group_branch_gates.csv` - 各组门控值
- `group_branch_contributions.csv` - 各组贡献度
- `predictions_test.csv` - 测试集预测
- `run_log.txt` - 训练日志

## 注意事项

1. **DAG文件**: 当前配置中的 `dag_path: "data/features/global_edges.csv"` 可能不存在。如果不存在，DML模型会使用启发式规则推断工况变量C。

2. **特征索引**: 门控分组模型的特征索引已根据实际数据配置。如果数据结构变化，需要重新运行 `check_features.py` 更新索引。

3. **系统2数据**: 当前配置针对浮选系统1 (`y_fx_xin1`)。如需训练系统2模型，需要：
   - 修改 `target_col: "y_fx_xin2"`
   - 修改 `exclude_cols` 排除 `y_fx_xin1`
   - 更新操作变量为系统2的变量 (fx_s2_*)
   - 更新门控分组索引

4. **计算资源**: 
   - 数据量较大 (86,400样本)
   - 特征维度高 (214维)
   - 建议使用GPU加速训练
   - 预计单次训练时间: 10-30分钟（取决于硬件）

## 下一步

配置已完成，可以开始训练。建议按以下顺序：

1. 先运行门控分组模型（不依赖DAG）
2. 检查结果和特征重要性
3. 如需要，准备DAG文件后运行DML残差模型
4. 对比两种模型的性能
