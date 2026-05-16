# 模型权重保存状态

**更新时间**: 2026-05-16 早晨  
**状态**: 正在训练并保存模型权重 🔄

---

## 当前进程

### Terminal 18: 门控分组模型训练
- **配置**: `configs/group_branch_test_best.yaml`
- **输出目录**: `results/group_branch_test_optuna/`
- **状态**: 正在训练 ✅
- **使用超参数**: Optuna最佳超参数
  - lr: 0.0007309539835912913
  - batch_size: 64
  - hidden_dim_preprocessing: 32
  - hidden_dim_reagent: 40
  - hidden_dim_flotation: 56
  - gate_init: 0.37986951286334386
- **预计时间**: ~5-10分钟

### Terminal 19: DML残差模型训练
- **配置**: `configs/residual_soft_sensor_test_best.yaml`
- **输出目录**: `results/residual_soft_sensor_test_optuna/`
- **状态**: 正在训练 ✅
- **使用超参数**: Optuna最佳超参数
  - lstm_hidden_size: 128
  - lstm_num_layers: 1
  - lstm_dropout: 0.2406590942262119
  - lstm_lr: 0.00014096175149815865
  - lstm_batch_size: 64
- **预计时间**: ~15-20分钟

---

## 将要保存的文件

### 门控分组模型
```
results/group_branch_test_optuna/
├── model_checkpoint.pt          # 模型权重 + Y scaler
│   ├── model_state_dict         # PyTorch模型参数
│   ├── y_scaler_mean            # Y标准化的均值
│   ├── y_scaler_scale           # Y标准化的标准差
│   ├── feat_scaler_mean         # 特征标准化的均值
│   ├── feat_scaler_scale        # 特征标准化的标准差
│   ├── feature_cols             # 特征列名列表
│   ├── groups_cfg               # 分组配置
│   ├── model_cfg                # 模型配置
│   ├── window_size              # 窗口大小
│   └── num_features             # 特征数量
├── group_branch_metrics.csv     # 性能指标
├── group_branch_gates.csv       # Gate值
├── group_branch_contributions.csv # 分支贡献
├── predictions_test.csv         # 测试集预测
└── run_log.txt                  # 训练日志
```

### DML残差模型
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
├── metrics_compare.csv              # 三模型对比
├── variable_roles.csv               # 变量角色
├── predictions_test.csv             # 测试集预测
├── y_baseline_predictions.csv       # 基线预测
├── residual_feature_summary.csv     # 残差特征总结
└── run_log.txt                      # 训练日志
```

---

## 修复过程

### 问题1: 原始脚本logger错误
- **错误**: `AttributeError: 'NoneType' object has no attribute 'info'`
- **原因**: 传递了None作为logger参数
- **解决**: 创建新脚本直接修改配置文件

### 问题2: 输出目录不正确
- **错误**: 输出到`results/group_branch_test`而非`results/group_branch_test_optuna`
- **原因**: 配置文件中的output_dir未更新
- **解决**: 修改配置文件的output_dir字段

### 最终方案
1. 创建`save_optuna_models.py`脚本
2. 读取Optuna最佳超参数
3. 更新配置文件
4. 修改output_dir指向optuna结果目录
5. 直接调用原始训练脚本（已包含模型保存功能）

---

## 监控命令

### 查看进程状态
```python
listProcesses
```

### 查看训练进度
```python
# 门控分组
getProcessOutput(terminalId="18", lines=50)

# DML残差
getProcessOutput(terminalId="19", lines=50)
```

### 检查文件是否生成
```bash
# 门控分组
ls results/group_branch_test_optuna/

# DML残差
ls results/residual_soft_sensor_test_optuna/

# 检查模型权重文件
ls results/group_branch_test_optuna/*.pt
ls results/residual_soft_sensor_test_optuna/*.pt
ls results/residual_soft_sensor_test_optuna/*.pkl
```

---

## 预计完成时间

| 模型 | 开始时间 | 预计完成 | 状态 |
|------|---------|---------|------|
| 门控分组 | ~现在 | ~5-10分钟后 | 🔄 训练中 |
| DML残差 | ~现在 | ~15-20分钟后 | 🔄 训练中 |

**总预计时间**: 15-20分钟

---

## 完成后的下一步

### 立即执行
1. ✅ 验证模型权重文件已保存
2. 📊 检查文件大小是否合理
3. 🧪 在新数据上测试模型

### 测试命令

#### 门控分组模型
```bash
python scripts/test_on_new_data.py \
  --model group_branch \
  --checkpoint results/group_branch_test_optuna/model_checkpoint.pt \
  --config configs/group_branch.yaml \
  --data "C:\Users\goldenwhale\Downloads\my_mining_simulation\output\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet"
```

#### DML残差模型（推荐重新训练）
由于DML模型的复杂性，建议在新数据上重新训练：
```bash
# 创建新配置
cp configs/residual_soft_sensor_test_best.yaml configs/residual_soft_sensor_multiregime.yaml

# 修改data_path指向新数据
# 然后训练
python scripts/train_dml_residual_soft_sensor.py \
  --config configs/residual_soft_sensor_multiregime.yaml
```

---

## 技术细节

### 模型保存功能位置

#### train_group_branch.py
- **函数**: `save_results()`
- **位置**: 第367-430行
- **保存内容**:
  - 模型state_dict
  - Y scaler (mean, scale)
  - Feature scaler (mean, scale)
  - 特征列名
  - 配置信息

#### train_dml_residual_soft_sensor.py
- **函数**: `save_outputs()`
- **位置**: 第1062-1150行
- **保存内容**:
  - 三个LSTM模型的state_dict
  - 所有scalers (y_res, as, c)
  - 基模型 (g_model, q_models)

### 配置文件

#### group_branch_test_best.yaml
- 基于: `group_branch_test.yaml`
- 修改: 超参数 + output_dir
- 位置: `configs/group_branch_test_best.yaml`

#### residual_soft_sensor_test_best.yaml
- 基于: `residual_soft_sensor_test.yaml`
- 修改: 超参数 + output_dir
- 位置: `configs/residual_soft_sensor_test_best.yaml`

---

## 故障排除

### 如果训练失败
1. 检查日志文件:
   ```bash
   cat results/group_branch_test_optuna/run_log.txt
   cat results/residual_soft_sensor_test_optuna/run_log.txt
   ```

2. 检查进程状态:
   ```python
   getProcessOutput(terminalId="18", lines=100)
   getProcessOutput(terminalId="19", lines=100)
   ```

3. 如果需要重新运行:
   ```bash
   # 停止进程
   controlPwshProcess(action="stop", terminalId="18")
   controlPwshProcess(action="stop", terminalId="19")
   
   # 重新启动
   python scripts/train_group_branch.py --config configs/group_branch_test_best.yaml
   python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor_test_best.yaml
   ```

### 如果模型文件未保存
- 检查save_results/save_outputs函数是否被调用
- 检查output_dir是否正确
- 检查磁盘空间是否充足

---

**当前状态**: 两个模型正在训练，将自动保存权重 ✅  
**预计完成**: 15-20分钟后  
**下一步**: 等待完成，然后在新数据上测试

---

*文档更新时间: 2026-05-16 早晨*
