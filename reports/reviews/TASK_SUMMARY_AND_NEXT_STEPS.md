# 任务总结与后续步骤

**更新时间**: 2026-05-15 23:55  
**状态**: 等待Optuna优化完成 ⏳

---

## 已完成的工作 ✅

### 1. DAG合并
- ✅ 合并了磁选、塔磨、浮选三个DAG文件
- ✅ 总共225条边
- ✅ 输出: `data/features/global_edges.csv`

### 2. Optuna脚本创建
- ✅ `scripts/train_group_branch_optuna.py` - 门控分组模型优化
- ✅ `scripts/train_dml_residual_optuna.py` - DML残差模型优化
- ✅ 修复了StandardScaler导入错误

### 3. 模型保存功能
- ✅ 修改了`train_group_branch.py`，添加模型权重保存
- ✅ 修改了`train_dml_residual_soft_sensor.py`，添加LSTM权重和scalers保存
- ✅ 保存内容包括:
  - 模型state_dict
  - Y scaler的mean和scale
  - 其他必要的scalers和基模型

### 4. 新数据测试脚本
- ✅ 创建了`scripts/test_on_new_data.py`
- ✅ 支持在新数据上测试已训练的模型
- ✅ 目标数据: `multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet`

### 5. 自动等待和报告脚本
- ✅ 创建了`scripts/wait_and_check_optuna.py`
- ✅ 每5分钟检查一次优化状态
- ✅ 完成后自动生成对比报告

---

## 当前运行状态 🔄

### Terminal 9: DML残差Optuna优化
- 配置: `configs/residual_soft_sensor_test.yaml`
- Trials: 12次
- 状态: 运行中
- 预计完成: ~30-40分钟

### Terminal 10: 门控分组Optuna优化
- 配置: `configs/group_branch_test.yaml`
- Trials: 15次
- 状态: 运行中 (1/15完成)
- 预计完成: ~40分钟

### Terminal 12: 等待和报告脚本
- 每2小时检查一次
- 无最大等待时间限制
- 完成后生成: `results/OPTUNA_FINAL_REPORT.txt`

---

## 模型保存详情

### 门控分组模型
**保存文件**:
```
results/group_branch_test_optuna/
├── model_checkpoint.pt          # 模型权重
│   ├── model_state_dict         # PyTorch模型参数
│   ├── y_scaler_mean            # Y标准化的均值
│   └── y_scaler_scale           # Y标准化的标准差
├── best_params.json             # 最佳超参数
├── optimization_history.csv     # 优化历史
└── [其他输出文件]
```

**模型大小估算**: ~1-5 MB (取决于hidden_dim配置)

### DML残差模型
**保存文件**:
```
results/residual_soft_sensor_test_optuna/
├── dml_residual_lstm_checkpoint.pt  # DML残差LSTM权重
│   ├── model_state_dict             # PyTorch LSTM参数
│   ├── input_size                   # 输入维度
│   ├── hidden_size                  # 隐藏层维度
│   ├── num_layers                   # 层数
│   └── dropout                      # Dropout率
├── baseline_lstm_checkpoint.pt      # 基线LSTM权重
├── causal_input_lstm_checkpoint.pt  # 因果输入LSTM权重
├── model_scalers.pkl                # Scalers和基模型
│   ├── y_res_scaler                 # 残差Y的scaler
│   ├── as_scaler                    # A+S特征的scaler
│   ├── c_scaler                     # C特征的scaler
│   ├── g_model                      # C->Y的基模型
│   └── q_models                     # C->X_j的基模型列表
├── best_params.json                 # 最佳超参数
├── optimization_history.csv         # 优化历史
└── [其他输出文件]
```

**模型大小估算**: ~5-20 MB (包含LightGBM模型)

---

## 在新数据上测试

### 目标数据
```
C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet
```

### 测试方法

#### 方法1: 使用测试脚本（推荐用于门控分组模型）
```bash
# 门控分组模型
python scripts/test_on_new_data.py \
  --model group_branch \
  --checkpoint results/group_branch_test_optuna/model_checkpoint.pt \
  --config configs/group_branch.yaml \
  --data "C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet"
```

**输出**:
```
results/test_on_new_data/group_branch/
├── test_metrics.csv        # 测试指标
├── test_predictions.csv    # 预测结果
├── test_gates.csv          # Gate值
└── test_log.txt            # 日志
```

#### 方法2: 重新训练（推荐用于DML残差模型）
由于DML残差模型依赖于完整的训练流程（g_model和q_models），建议在新数据上重新训练：

```bash
# 创建新的配置文件
cp configs/residual_soft_sensor_test.yaml configs/residual_soft_sensor_multiregime.yaml

# 修改配置文件中的data_path
# data_path: "C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet"

# 使用最佳超参数训练
python scripts/train_dml_residual_optuna.py \
  --config configs/residual_soft_sensor_multiregime.yaml \
  --trials 1  # 只运行1次，使用已知的最佳参数
```

---

## 注意事项 ⚠️

### 1. 特征标准化
- **门控分组模型**: 测试脚本使用新数据的统计量进行标准化
- **更好的做法**: 在checkpoint中保存训练时的feature_scaler
- **影响**: 如果新数据分布差异大，可能影响性能

### 2. DML模型的复杂性
- DML残差模型包含多个组件（g_model, q_models, LSTM）
- 完整复现需要所有组件
- 建议在新数据上重新训练，而不是直接加载

### 3. 数据兼容性
- 确保新数据包含所有必需的特征列
- 确保目标变量名称一致（`y_fx_xin1`）
- 确保数据格式一致（parquet）

### 4. 存储空间
- 门控分组模型: ~1-5 MB
- DML残差模型: ~5-20 MB
- 总计: ~10-30 MB（完全可接受）

---

## 后续步骤（优化完成后）

### 立即执行
1. ✅ 等待脚本会自动生成报告
2. 📊 读取`results/OPTUNA_FINAL_REPORT.txt`
3. 📈 分析性能提升
4. 📝 决定是否需要调整

### 短期任务
1. 🧪 在新数据上测试门控分组模型
   ```bash
   python scripts/test_on_new_data.py --model group_branch --checkpoint results/group_branch_test_optuna/model_checkpoint.pt --config configs/group_branch.yaml --data "C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet"
   ```

2. 🔄 在新数据上重新训练DML残差模型
   - 创建新配置文件
   - 使用Optuna找到的最佳超参数
   - 在新数据上训练

3. 📊 对比不同数据集上的性能
   - 原始数据: `simulation_2months_seq_hybrid_normal_fast_sampling.parquet`
   - 新数据: `multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet`

### 中期任务
1. 🎯 使用完整配置训练最终模型
   - `configs/group_branch.yaml` (完整数据)
   - `configs/residual_soft_sensor.yaml` (完整数据)
   - 使用Optuna找到的最佳超参数

2. 📦 保存生产就绪的模型
   - 包含完整的feature_scaler
   - 包含所有必要的元数据
   - 添加版本信息

3. 📚 更新文档
   - 模型使用说明
   - 超参数说明
   - 性能基准

---

## 预期结果

### 门控分组模型
**当前最佳** (无Optuna):
- R² = 0.6595
- MAE = 0.0147
- RMSE = 0.0188

**Optuna预期**:
- R² > 0.67 (提升 ~2-5%)
- MAE < 0.014 (降低 ~5-10%)

### DML残差模型
**当前最佳** (无Optuna):
- R² = 0.3710
- MAE = 0.0200
- RMSE = 0.0256

**Optuna预期**:
- R² > 0.40 (提升 ~5-10%)
- MAE < 0.019 (降低 ~5-10%)

---

## 监控命令

### 查看进程状态
```python
# 在Kiro中
listProcesses
```

### 查看实时输出
```python
# 门控分组
getProcessOutput(terminalId="10", lines=50)

# DML残差
getProcessOutput(terminalId="9", lines=50)

# 等待脚本
getProcessOutput(terminalId="11", lines=50)
```

### 手动检查完成状态
```bash
# 检查文件是否存在
ls results/group_branch_test_optuna/best_params.json
ls results/residual_soft_sensor_test_optuna/best_params.json

# 查看最终报告
cat results/OPTUNA_FINAL_REPORT.txt
```

---

## 故障排除

### 如果优化时间过长
- 检查进程是否卡住: `listProcesses`
- 查看日志: `cat results/*/run_log.txt`
- 如果需要，可以减少trials数量重新运行

### 如果内存不足
- 减少batch_size
- 减少hidden_dim搜索范围
- 一次只运行一个优化

### 如果测试脚本失败
- 检查checkpoint文件是否存在
- 检查新数据路径是否正确
- 检查特征列是否匹配

---

## 文件清单

### 新创建的文件
1. `scripts/test_on_new_data.py` - 新数据测试脚本
2. `scripts/wait_and_check_optuna.py` - 自动等待和报告脚本
3. `TASK_SUMMARY_AND_NEXT_STEPS.md` - 本文档

### 修改的文件
1. `scripts/train_group_branch.py` - 添加模型保存
2. `scripts/train_dml_residual_soft_sensor.py` - 添加模型保存
3. `scripts/train_group_branch_optuna.py` - 修复导入错误

### 将要生成的文件
1. `results/OPTUNA_FINAL_REPORT.txt` - 优化完成报告
2. `results/group_branch_test_optuna/model_checkpoint.pt` - 模型权重
3. `results/residual_soft_sensor_test_optuna/*.pt` - 模型权重
4. `results/test_on_new_data/` - 新数据测试结果

---

**当前状态**: 等待Optuna优化完成（预计1-1.5小时）  
**下一步**: 自动生成报告，然后在新数据上测试

---

*此文档会在优化完成后更新*
