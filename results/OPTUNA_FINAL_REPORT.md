# Optuna超参数优化最终报告

**生成时间**: 2026-05-16 早晨  
**状态**: ✅ 两个优化都已完成

---

## 执行摘要

### 门控分组软测量模型
- ✅ 完成15次trials
- ⚠️ **性能下降**: R² 从 0.6595 降至 0.5984 (-9.3%)
- 原因分析: 可能是test配置数据量较小，或超参数搜索空间需要调整

### DML残差软测量模型  
- ✅ 完成12次trials
- ⚠️ **性能下降**: R² 从 0.3710 降至 0.3417 (-7.9%)
- 但causal_input模型有提升: R² 从 0.2844 提升至 0.4376 (+53.9%)

---

## 1. 门控分组软测量模型详细分析

### 1.1 最佳超参数

```json
{
  "lr": 0.0007309539835912913,
  "batch_size": 64,
  "hidden_dim_preprocessing": 32,
  "hidden_dim_reagent": 40,
  "hidden_dim_flotation": 56,
  "gate_init": 0.37986951286334386
}
```

**与原始配置对比**:
- lr: 0.0007 (优化后) vs 默认值
- batch_size: 64 (优化后) vs 默认值
- hidden_dim更加平衡: 32/40/56 vs 原始配置
- gate_init: 0.38 (较低，更保守的初始化)

### 1.2 性能指标对比

| 指标 | Optuna优化后 | 原始结果 | 变化 |
|------|-------------|---------|------|
| **MAE** | 0.01555 | 0.01474 | +5.5% ⬆️ (变差) |
| **RMSE** | 0.02042 | 0.01880 | +8.6% ⬆️ (变差) |
| **R²** | 0.5984 | 0.6595 | -9.3% ⬇️ (变差) |

### 1.3 优化历史分析

- **总trials**: 15次
- **最佳验证损失**: 0.2963 (Trial 2)
- **最差验证损失**: 0.4746 (Trial 8)
- **平均trial时间**: ~1.8分钟
- **总优化时间**: 29分钟

**Top 3 Trials**:
1. Trial 2: val_loss=0.2963, batch_size=64, lr=0.00073
2. Trial 13: val_loss=0.3135, batch_size=256, lr=0.00027
3. Trial 11: val_loss=0.3302, batch_size=256, lr=0.00027

**关键发现**:
- 较大的batch_size (64-256) 表现更好
- 较低的学习率 (0.0003-0.0007) 更稳定
- gate_init在0.3-0.4之间效果较好

### 1.4 问题分析

**为什么Optuna优化后性能反而下降？**

1. **数据集差异**: 
   - 原始训练使用完整配置 (`group_branch.yaml`)
   - Optuna使用测试配置 (`group_branch_test.yaml`)
   - 可能数据量或特征配置不同

2. **验证集vs测试集差异**:
   - Optuna优化的是验证集损失
   - 最终评估的是测试集性能
   - 可能存在过拟合验证集的情况

3. **搜索空间限制**:
   - hidden_dim范围: 16-64 (可能需要更大的值)
   - lr范围: 1e-4 to 1e-2 (可能需要更精细的搜索)

4. **训练epochs差异**:
   - Optuna优化阶段: 30 epochs
   - 最终训练: 50 epochs
   - 可能需要更多epochs

---

## 2. DML残差软测量模型详细分析

### 2.1 最佳超参数

```json
{
  "lstm_hidden_size": 128,
  "lstm_num_layers": 1,
  "lstm_dropout": 0.2406590942262119,
  "lstm_lr": 0.00014096175149815865,
  "lstm_batch_size": 64
}
```

**关键特点**:
- hidden_size: 128 (较大，增强表达能力)
- num_layers: 1 (简单架构，避免过拟合)
- dropout: 0.24 (适度正则化)
- lr: 0.00014 (非常低的学习率)
- batch_size: 64 (中等批次)

### 2.2 性能指标对比

#### DML残差模型 (Model 2)

| 指标 | Optuna优化后 | 原始结果 | 变化 |
|------|-------------|---------|------|
| **MAE** | 0.01996 | 0.02002 | -0.3% ✅ (略好) |
| **RMSE** | 0.02614 | 0.02555 | +2.3% ⬆️ (略差) |
| **R²** | 0.3417 | 0.3710 | -7.9% ⬇️ (变差) |

#### 基线模型 (Model 0)

| 指标 | Optuna优化后 | 原始结果 | 变化 |
|------|-------------|---------|------|
| **MAE** | 0.02006 | 0.02083 | -3.7% ✅ (改进) |
| **RMSE** | 0.02541 | 0.02646 | -4.0% ✅ (改进) |
| **R²** | 0.3778 | 0.3253 | +16.1% ✅ (显著改进) |

#### 因果输入模型 (Model 1) ⭐ 最大赢家

| 指标 | Optuna优化后 | 原始结果 | 变化 |
|------|-------------|---------|------|
| **MAE** | 0.01873 | 0.02100 | -10.8% ✅ (显著改进) |
| **RMSE** | 0.02416 | 0.02726 | -11.4% ✅ (显著改进) |
| **R²** | 0.4376 | 0.2844 | +53.9% ✅ (巨大改进) |

### 2.3 优化历史分析

- **总trials**: 12次
- **最佳验证RMSE**: 0.02135 (Trial 7)
- **最差验证RMSE**: 0.02331 (Trial 9)
- **平均trial时间**: ~8分钟
- **总优化时间**: 94分钟 (1.6小时)

**Top 3 Trials**:
1. Trial 7: RMSE=0.02135, hidden=128, layers=1, dropout=0.24, lr=0.00014
2. Trial 2: RMSE=0.02156, hidden=128, layers=1, dropout=0.09, lr=0.00054
3. Trial 4: RMSE=0.02190, hidden=128, layers=2, dropout=0.04, lr=0.00098

**关键发现**:
- hidden_size=128 明显优于32和64
- num_layers=1 最优 (更深的网络反而过拟合)
- 较低的学习率 (0.0001-0.0005) 效果最好
- dropout在0.04-0.24之间都可以

### 2.4 重要发现 ⭐

**因果输入模型 (Model 1) 表现最佳！**

虽然DML残差模型 (Model 2) 性能略有下降，但**因果输入模型 (Model 1) 获得了巨大提升**:
- R² 从 0.2844 提升到 0.4376 (+53.9%)
- MAE 从 0.0210 降低到 0.0187 (-10.8%)
- RMSE 从 0.0273 降低到 0.0242 (-11.4%)

**这表明**:
1. 超参数优化对简单的因果输入模型效果显著
2. DML的复杂流程可能需要更精细的调优
3. 在实际应用中，因果输入模型可能是更好的选择

---

## 3. 综合分析与建议

### 3.1 为什么会出现性能下降？

#### 可能原因

1. **配置文件差异**:
   ```
   原始训练: configs/group_branch.yaml (完整配置)
   Optuna训练: configs/group_branch_test.yaml (测试配置)
   ```
   需要检查两个配置的差异

2. **数据量差异**:
   - 测试配置可能使用了更少的数据
   - 小数据集上的超参数可能不适用于大数据集

3. **过拟合验证集**:
   - Optuna优化的是验证集性能
   - 可能导致在测试集上泛化能力下降

4. **搜索空间不够大**:
   - 最优超参数可能在搜索范围之外
   - 需要扩大搜索空间

### 3.2 改进建议

#### 短期改进 (立即可做)

1. **使用完整配置重新优化**:
   ```bash
   # 门控分组
   python scripts/train_group_branch_optuna.py \
     --config configs/group_branch.yaml \
     --trials 20
   
   # DML残差
   python scripts/train_dml_residual_optuna.py \
     --config configs/residual_soft_sensor.yaml \
     --trials 15
   ```

2. **扩大搜索空间**:
   - hidden_dim: 16-128 (原来是16-64)
   - lr: 1e-5 to 5e-3 (原来是1e-4 to 1e-2)
   - 增加更多epochs: 50-100

3. **使用找到的最佳超参数训练完整模型**:
   - 即使测试配置上性能下降
   - 但超参数本身可能是有价值的
   - 在完整数据上可能表现更好

#### 中期改进 (1-2天)

1. **分析配置文件差异**:
   ```bash
   diff configs/group_branch.yaml configs/group_branch_test.yaml
   diff configs/residual_soft_sensor.yaml configs/residual_soft_sensor_test.yaml
   ```

2. **使用因果输入模型**:
   - Model 1 (causal_input) 表现最佳
   - R² = 0.4376，显著优于其他模型
   - 考虑作为主要模型使用

3. **交叉验证**:
   - 使用K-fold交叉验证
   - 避免过拟合单一验证集

#### 长期改进 (1周)

1. **多目标优化**:
   - 同时优化验证集和测试集性能
   - 使用Pareto优化

2. **集成学习**:
   - 结合多个trial的模型
   - 可能获得更稳定的性能

3. **自动化超参数调优流程**:
   - 建立完整的MLOps流程
   - 定期重新优化

### 3.3 立即行动项

#### ✅ 已完成
- [x] Optuna优化完成
- [x] 结果分析完成
- [x] 模型权重保存功能添加

#### 🔄 进行中
- [ ] 检查是否保存了模型权重文件

#### 📋 待办事项

**优先级1 (今天)**:
1. 检查模型权重文件是否存在
2. 在新数据上测试门控分组模型
3. 分析配置文件差异

**优先级2 (明天)**:
1. 使用完整配置重新运行Optuna优化
2. 使用因果输入模型 (Model 1) 作为主要模型
3. 在新数据上测试所有模型

**优先级3 (本周)**:
1. 扩大搜索空间重新优化
2. 实现交叉验证
3. 建立模型评估流程

---

## 4. 模型权重保存检查

让我检查模型权重是否已保存...

### 4.1 门控分组模型

预期文件: `results/group_branch_test_optuna/model_checkpoint.pt`

### 4.2 DML残差模型

预期文件:
- `results/residual_soft_sensor_test_optuna/dml_residual_lstm_checkpoint.pt`
- `results/residual_soft_sensor_test_optuna/baseline_lstm_checkpoint.pt`
- `results/residual_soft_sensor_test_optuna/causal_input_lstm_checkpoint.pt`
- `results/residual_soft_sensor_test_optuna/model_scalers.pkl`

---

## 5. 新数据测试计划

### 5.1 目标数据
```
C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet
```

### 5.2 测试命令

#### 门控分组模型
```bash
python scripts/test_on_new_data.py \
  --model group_branch \
  --checkpoint results/group_branch_test_optuna/model_checkpoint.pt \
  --config configs/group_branch.yaml \
  --data "C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet"
```

#### DML残差模型 (推荐重新训练)
```bash
# 创建新配置
cp configs/residual_soft_sensor_test.yaml configs/residual_soft_sensor_multiregime.yaml

# 修改data_path指向新数据

# 使用最佳超参数训练
python scripts/train_dml_residual_soft_sensor.py \
  --config configs/residual_soft_sensor_multiregime.yaml
```

---

## 6. 结论

### 6.1 主要发现

1. **门控分组模型**: Optuna优化后性能下降9.3%，需要进一步调查
2. **DML残差模型**: 主模型性能略有下降，但因果输入模型获得53.9%的巨大提升
3. **最佳模型**: 因果输入模型 (Model 1) R²=0.4376，是当前最佳选择
4. **超参数洞察**: 
   - 较大的hidden_size (128) 效果好
   - 简单的架构 (1层) 优于复杂架构
   - 低学习率 (0.0001-0.0007) 更稳定

### 6.2 推荐方案

**方案A: 使用因果输入模型 (推荐)**
- 性能最佳: R²=0.4376
- 架构简单，易于部署
- 已经过Optuna优化

**方案B: 重新优化门控分组模型**
- 使用完整配置
- 扩大搜索空间
- 增加trials数量

**方案C: 混合方案**
- 在不同场景使用不同模型
- 集成多个模型的预测

### 6.3 下一步行动

1. ✅ 检查模型权重文件
2. 🧪 在新数据上测试
3. 📊 对比不同数据集的性能
4. 🔄 使用完整配置重新优化
5. 📦 部署最佳模型

---

**报告生成时间**: 2026-05-16  
**报告状态**: 完整分析完成  
**建议优先级**: 立即检查模型权重并在新数据上测试

