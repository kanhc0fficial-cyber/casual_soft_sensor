#!/bin/bash
# ==============================================================================
# 约束消融实验批处理脚本
# 运行4个实验：基线（无约束）、仅反事实约束、仅工艺约束、两种约束
# 只运行 Model 3，跳过 Model 0-2 以节省时间
# ==============================================================================

echo "========================================================================"
echo "约束消融实验 - 开始（只运行 Model 3）"
echo "========================================================================"
echo ""

# 实验1：基线（无约束）
echo "========================================================================"
echo "实验 1/4: 基线（无约束）"
echo "========================================================================"
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_baseline.yaml \
    --only-model3
echo ""
echo "实验 1/4 完成"
echo ""

# 实验2：仅反事实约束
echo "========================================================================"
echo "实验 2/4: 仅反事实约束"
echo "========================================================================"
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_counterfactual.yaml \
    --only-model3
echo ""
echo "实验 2/4 完成"
echo ""

# 实验3：仅工艺约束
echo "========================================================================"
echo "实验 3/4: 仅工艺约束"
echo "========================================================================"
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_process.yaml \
    --only-model3
echo ""
echo "实验 3/4 完成"
echo ""

# 实验4：两种约束都启用
echo "========================================================================"
echo "实验 4/4: 反事实约束 + 工艺约束"
echo "========================================================================"
python scripts/train_dml_residual_soft_sensor.py \
    --config configs/ablation_constraints_both.yaml \
    --only-model3
echo ""
echo "实验 4/4 完成"
echo ""

# 汇总结果
echo "========================================================================"
echo "所有实验完成！"
echo "========================================================================"
echo ""
echo "结果目录："
echo "  1. 基线（无约束）:        results/ablation_constraints/baseline_no_constraints/"
echo "  2. 仅反事实约束:          results/ablation_constraints/counterfactual_only/"
echo "  3. 仅工艺约束:            results/ablation_constraints/process_only/"
echo "  4. 反事实+工艺约束:       results/ablation_constraints/both_constraints/"
echo ""
echo "查看各实验的 metrics_compare.csv 文件对比 Model 3 的性能差异"
echo ""
