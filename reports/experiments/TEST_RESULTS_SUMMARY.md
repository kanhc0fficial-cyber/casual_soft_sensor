# 模型测试结果总结

## 测试时间
2026-05-15

## 测试状态

### ✅ 1. DAG合并 - 成功
**脚本**: `merge_dags.py`
**输出**: `data/features/global_edges.csv`

**结果**:
- 成功合并塔磨和浮选两个DAG文件
- 总边数: 222条
- 唯一源节点: 44个
- 唯一目标节点: 52个
- 权重范围: [0.0202, 1.1726]
- 指向目标变量的边: 3条
  - fx_s1_cx1_froth_h → y_fx_xin1 (0.0412)
  - fx_s1_sx3_froth_h → y_fx_xin1 (0.0324)
  - fx_s1_jx_froth_h → y_fx_xin1 (0.0271)

### ✅ 2. 门控分组软测量 - 快速测试成功
**配置**: `configs/group_branch_test.yaml`
**输出**: `results/group_branch_test/`

**训练参数**:
- Epochs: 5 (早停@4)
- Batch size: 128
- 分组数: 3 (preprocessing, reagent, flotation)

**测试结果**:
- **MAE**: 0.0173
- **RMSE**: 0.0227
- **R²**: 0.5057

**Gate值**:
- preprocessing: 0.5135
- reagent: 0.5026
- flotation: 0.5398

**分支贡献**:
| 分组 | Gate | 平均分支输出 | 平均绝对贡献 |
|------|------|-------------|-------------|
| preprocessing | 0.514 | -0.248 | 0.266 |
| reagent | 0.503 | 0.306 | 0.403 |
| flotation | 0.540 | 0.324 | 0.525 |

### ✅ 3. 门控分组软测量 - 完整训练成功
**配置**: `configs/group_branch.yaml`
**输出**: `results/group_branch/`

**训练参数**:
- Epochs: 50 (早停@9)
- Batch size: 64
- 分组数: 7 (preprocessing, reagent, roughing, cleaning, scavenging, temperature, auxiliary)
- 模型参数: 43,151

**测试结果** ⭐:
- **MAE**: 0.0147
- **RMSE**: 0.0188
- **R²**: 0.6595

**Gate值**:
| 分组 | Gate值 | 说明 |
|------|--------|------|
| preprocessing | 0.5065 | 前处理阶段 |
| reagent | 0.4997 | 药剂控制 |
| roughing | 0.5198 | 粗选 |
| cleaning | 0.5002 | 精选 |
| scavenging | 0.5265 | 扫选（最高） |
| temperature | 0.4900 | 温度控制 |
| auxiliary | 0.4846 | 辅助系统（最低） |

**分支贡献分析**:
| 分组 | Gate | 平均分支输出 | 平均绝对贡献 | 重要性排名 |
|------|------|-------------|-------------|-----------|
| scavenging | 0.526 | -0.079 | **1.093** | 🥇 1 |
| roughing | 0.520 | -0.648 | **0.633** | 🥈 2 |
| reagent | 0.500 | 0.420 | **0.342** | 🥉 3 |
| preprocessing | 0.507 | -0.089 | 0.209 | 4 |
| cleaning | 0.500 | 0.242 | 0.176 | 5 |
| temperature | 0.490 | 0.204 | 0.100 | 6 |
| auxiliary | 0.485 | 0.145 | 0.076 | 7 |

**关键发现**:
1. **扫选阶段最重要**: 平均绝对贡献1.093，远超其他分组
2. **粗选次之**: 贡献0.633，说明粗选效果对最终品位影响大
3. **药剂控制第三**: 贡献0.342，药剂添加策略很关键
4. **辅助系统贡献最小**: 仅0.076，主要起支持作用

### ⏸️ 4. DML残差软测量 - 运行中/超时
**配置**: `configs/residual_soft_sensor.yaml`
**状态**: 运行超时（>5分钟）

**可能原因**:
1. 数据量大（86,400样本）
2. 特征维度高（214维）
3. 需要训练多个残差化模型（g_model + 多个q_model）
4. LSTM训练时间长（50 epochs）

**已创建快速测试配置**: `configs/residual_soft_sensor_test.yaml`
- 减少操作变量（17→4）
- 减少LSTM参数（hidden_size: 64→32, layers: 2→1）
- 减少训练轮数（50→10）
- 增大batch size（64→128）

## 性能对比

### 门控分组模型性能提升
| 配置 | Epochs | R² | MAE | RMSE | 训练时间 |
|------|--------|-----|-----|------|---------|
| 快速测试 | 5 | 0.5057 | 0.0173 | 0.0227 | ~2分钟 |
| 完整训练 | 50 | **0.6595** | **0.0147** | **0.0188** | ~8分钟 |
| 提升 | - | **+30.4%** | **-15.0%** | **-17.2%** | - |

## 输出文件清单

### 门控分组模型
```
results/group_branch/
├── group_branch_metrics.csv          # 评估指标
├── group_branch_gates.csv            # 各组gate值
├── group_branch_contributions.csv    # 各组贡献度
├── predictions_test.csv              # 测试集预测
├── run_log.txt                       # 训练日志
└── console_output.txt                # 控制台输出
```

### DAG文件
```
data/features/
└── global_edges.csv                  # 合并后的因果图
```

### 辅助文件
```
casual_soft_sensor/
├── merge_dags.py                     # DAG合并脚本
├── check_features.py                 # 特征索引查看工具
├── feature_index_mapping.txt         # 特征索引映射
├── DATA_CONFIGURATION_SUMMARY.md     # 配置详细说明
├── CONFIGURATION_STATUS.md           # 配置状态报告
└── TEST_RESULTS_SUMMARY.md           # 本文件
```

## 下一步建议

### 1. DML残差模型测试
```bash
# 使用快速测试配置
python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor_test.yaml
```

### 2. 完整DML模型训练
如果快速测试成功，再运行完整配置：
```bash
python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor.yaml
```
预计时间: 30-60分钟

### 3. 结果分析
- 对比门控分组模型和DML残差模型的性能
- 分析C变量推断是否合理
- 检查残差化效果

### 4. 模型优化
根据测试结果：
- 调整操作变量列表
- 优化分组策略
- 调整超参数

## 技术亮点

### 门控分组模型
✅ **工艺可解释性强**: 7个分组对应实际工艺流程
✅ **自动学习权重**: Gate值自动学习各阶段重要性
✅ **贡献度可视化**: 清晰展示各工艺段对品位的影响
✅ **训练效率高**: 9个epoch即收敛，总时间<10分钟

### DML残差模型
✅ **因果推断**: 基于DAG识别混杂变量
✅ **去混杂**: 通过残差化消除工况影响
✅ **理论保证**: DML框架提供统计保证

## 结论

1. **数据配置成功**: 所有数据路径、特征、分组均已正确配置
2. **DAG合并成功**: 成功整合塔磨和浮选两阶段因果图
3. **门控分组模型表现优秀**: R²=0.6595，MAE=0.0147
4. **工艺洞察清晰**: 扫选>粗选>药剂，符合浮选工艺特点
5. **DML模型待测试**: 需要更多时间完成训练

---

**测试完成度**: 75% (3/4 完成)
**下一步**: 运行DML残差模型快速测试
