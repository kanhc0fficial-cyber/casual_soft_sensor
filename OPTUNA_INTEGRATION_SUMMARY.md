# Optuna集成总结

## 完成时间
2026-05-15

## 已完成工作

### 1. DAG更新 ✅
- 添加磁选阶段DAG：`mb_cuts_磁选_real_dag_xin1.graphml`
- 重新合并DAG：磁选(3边) + 塔磨(26边) + 浮选(196边) = **225边**
- 输出：`data/features/global_edges.csv`

### 2. Optuna脚本创建 ✅
创建了两个带Optuna优化的训练脚本：

#### `train_group_branch_optuna.py`
**超参数搜索空间**：
- `lr`: 1e-4 ~ 1e-2 (log scale)
- `batch_size`: [32, 64, 128, 256]
- `hidden_dim_{group}`: 16 ~ 64 (step=8) for each group
- `gate_init`: 0.3 ~ 0.7

**优化设置**：
- 默认trials: 20
- Pruner: MedianPruner (n_startup_trials=5, n_warmup_steps=10)
- Sampler: TPESampler
- 优化目标: 最小化验证集MSE loss

#### `train_dml_residual_optuna.py`
**超参数搜索空间**：
- `lstm_hidden_size`: [32, 64, 128]
- `lstm_num_layers`: 1 ~ 3
- `lstm_dropout`: 0.0 ~ 0.3
- `lstm_lr`: 1e-4 ~ 1e-2 (log scale)
- `lstm_batch_size`: [64, 128, 256]

**优化设置**：
- 默认trials: 15
- Pruner: MedianPruner (n_startup_trials=3, n_warmup_steps=5)
- Sampler: TPESampler
- 优化目标: 最小化验证集RMSE

### 3. 运行状态 🔄

#### 当前运行的进程
1. **Terminal 7**: DML残差模型（原始版本）- 已完成 ✅
2. **Terminal 8**: 门控分组Optuna优化 - 运行中 🔄
3. **Terminal 9**: DML残差Optuna优化 - 运行中 🔄

## 使用方法

### 门控分组模型 + Optuna
```bash
# 快速测试（15 trials）
python scripts/train_group_branch_optuna.py --config configs/group_branch_test.yaml --trials 15

# 完整训练（20 trials）
python scripts/train_group_branch_optuna.py --config configs/group_branch.yaml --trials 20
```

### DML残差模型 + Optuna
```bash
# 快速测试（12 trials）
python scripts/train_dml_residual_optuna.py --config configs/residual_soft_sensor_test.yaml --trials 12

# 完整训练（15 trials）
python scripts/train_dml_residual_optuna.py --config configs/residual_soft_sensor.yaml --trials 15
```

## 输出文件

### Optuna优化输出
```
results/{model_name}_optuna/
├── best_params.json              # 最佳超参数
├── optimization_history.csv      # 优化历史
├── run_log.txt                   # 训练日志
└── [标准模型输出文件]
```

### 最佳参数示例
```json
{
  "lr": 0.0005611516415334506,
  "batch_size": 32,
  "hidden_dim_preprocessing": 24,
  "hidden_dim_reagent": 16,
  "hidden_dim_flotation": 64,
  "gate_init": 0.5404460046972834
}
```

## 优化策略

### 为什么trials数量较少？
1. **数据量大**: 86,400样本，每次trial需要较长时间
2. **特征维度高**: 214维特征
3. **实用性**: 15-20次试验通常足以找到较好的超参数
4. **时间成本**: 平衡优化效果和运行时间

### Pruning策略
- **MedianPruner**: 如果trial的中间结果明显差于中位数，提前终止
- **好处**: 节省时间，专注于有希望的超参数组合
- **设置**: 
  - 门控分组: 5个startup trials, 10个warmup steps
  - DML残差: 3个startup trials, 5个warmup steps

### 优化流程
1. **Optuna优化阶段** (减少epochs)
   - 门控分组: 30 epochs max, patience=5
   - DML残差: 20 epochs max, patience=5
   - 目的: 快速评估超参数质量

2. **最终训练阶段** (使用最佳参数)
   - 门控分组: 50 epochs, patience=8
   - DML残差: 50 epochs, patience=8
   - 目的: 充分训练获得最佳性能

## 预期改进

### 门控分组模型
- **当前最佳**: R²=0.6595, MAE=0.0147
- **Optuna后预期**: R²=0.67-0.70, MAE=0.013-0.014
- **改进来源**: 
  - 更优的学习率
  - 更合适的hidden_dim组合
  - 更好的gate初始化

### DML残差模型
- **当前最佳**: R²=0.3710, MAE=0.0200
- **Optuna后预期**: R²=0.40-0.45, MAE=0.018-0.019
- **改进来源**:
  - 更优的LSTM架构
  - 更合适的dropout率
  - 更好的学习率和batch size

## 技术细节

### 导入修复
修复了两个脚本的导入问题：
- 显式导入 `StandardScaler`
- 显式导入 `torch` 和相关模块
- 显式导入 `numpy`

### 搜索空间设计原则
1. **学习率**: log scale搜索，覆盖1e-4到1e-2
2. **Batch size**: 离散选择，2的幂次
3. **Hidden dim**: 8的倍数，便于GPU优化
4. **Dropout**: 连续搜索，0到0.3之间
5. **Gate init**: 连续搜索，0.3到0.7之间

### 性能考虑
- 使用TPESampler（Tree-structured Parzen Estimator）
- 比随机搜索更高效
- 利用历史trial信息指导搜索
- 适合中等规模的搜索空间

## 下一步

### 立即（等待完成）
1. ⏳ 等待Terminal 8完成（门控分组Optuna）
2. ⏳ 等待Terminal 9完成（DML残差Optuna）
3. 📊 对比优化前后的性能

### 短期（1天内）
1. 📊 分析optimization_history.csv
2. 📈 可视化超参数重要性
3. 📝 撰写优化结果报告
4. 🔧 根据结果调整搜索空间

### 中期（1周内）
1. 🎯 使用最佳参数训练完整模型
2. 📦 保存优化后的模型
3. 📚 更新文档和配置
4. 🚀 准备生产部署

## 检查进度

### 查看运行状态
```bash
# 列出所有后台进程
python -c "import subprocess; subprocess.run(['ps', 'aux'])"

# 或使用Kiro的进程管理
# 在Kiro中运行: listProcesses
```

### 查看输出
```bash
# 门控分组Optuna
cat results/group_branch_test_optuna/run_log.txt
cat results/group_branch_test_optuna/best_params.json

# DML残差Optuna
cat results/residual_soft_sensor_test_optuna/run_log.txt
cat results/residual_soft_sensor_test_optuna/best_params.json
```

### 查看优化历史
```python
import pandas as pd

# 门控分组
history = pd.read_csv('results/group_branch_test_optuna/optimization_history.csv')
print(history[['number', 'value', 'params_lr', 'params_batch_size']].head(10))

# DML残差
history = pd.read_csv('results/residual_soft_sensor_test_optuna/optimization_history.csv')
print(history[['number', 'value', 'params_lstm_hidden_size', 'params_lstm_lr']].head(10))
```

## 总结

### 完成情况
- ✅ DAG更新（添加磁选，225边）
- ✅ Optuna脚本创建（两个模型）
- ✅ 修复导入问题
- 🔄 运行优化（进行中）

### 技术亮点
- 🎯 合理的搜索空间设计
- ⚡ 高效的pruning策略
- 🔧 灵活的trials数量控制
- 📊 完整的优化历史记录

### 预期收益
- 📈 模型性能提升5-10%
- 🎯 自动化超参数调优
- 📚 可复现的优化流程
- 🚀 更快的模型迭代

---

**报告生成时间**: 2026-05-15  
**状态**: Optuna优化运行中  
**预计完成时间**: 10-15分钟
