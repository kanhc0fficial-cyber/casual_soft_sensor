# PowerShell脚本：运行剩余的消融实验

Write-Host "========================================================================"
Write-Host "运行剩余的消融实验（实验3和实验4）"
Write-Host "========================================================================"
Write-Host ""

# 等待实验2完成
Write-Host "等待实验2（仅反事实约束）完成..."
while (-not (Test-Path "results/ablation_constraints/counterfactual_only/metrics_compare.csv")) {
    Start-Sleep -Seconds 30
    Write-Host "." -NoNewline
}
Write-Host ""
Write-Host "实验2完成！"
Write-Host ""

# 实验3：仅工艺约束
Write-Host "========================================================================"
Write-Host "实验 3/4: 仅工艺约束"
Write-Host "========================================================================"
python scripts/train_dml_residual_soft_sensor.py `
    --config configs/ablation_constraints_process.yaml `
    --only-model3
Write-Host ""
Write-Host "实验 3/4 完成"
Write-Host ""

# 实验4：两种约束
Write-Host "========================================================================"
Write-Host "实验 4/4: 反事实约束 + 工艺约束"
Write-Host "========================================================================"
python scripts/train_dml_residual_soft_sensor.py `
    --config configs/ablation_constraints_both.yaml `
    --only-model3
Write-Host ""
Write-Host "实验 4/4 完成"
Write-Host ""

# 汇总结果
Write-Host "========================================================================"
Write-Host "所有实验完成！"
Write-Host "========================================================================"
Write-Host ""
Write-Host "结果目录："
Write-Host "  1. 基线（无约束）:        results/ablation_constraints/baseline_no_constraints/"
Write-Host "  2. 仅反事实约束:          results/ablation_constraints/counterfactual_only/"
Write-Host "  3. 仅工艺约束:            results/ablation_constraints/process_only/"
Write-Host "  4. 反事实+工艺约束:       results/ablation_constraints/both_constraints/"
Write-Host ""

# 汇总所有实验的指标
Write-Host "汇总所有实验的Model 3性能指标："
Write-Host "========================================================================"

$experiments = @(
    @{Name="基线（无约束）"; Path="results/ablation_constraints/baseline_no_constraints/metrics_compare.csv"},
    @{Name="仅反事实约束"; Path="results/ablation_constraints/counterfactual_only/metrics_compare.csv"},
    @{Name="仅工艺约束"; Path="results/ablation_constraints/process_only/metrics_compare.csv"},
    @{Name="两种约束"; Path="results/ablation_constraints/both_constraints/metrics_compare.csv"}
)

foreach ($exp in $experiments) {
    if (Test-Path $exp.Path) {
        Write-Host ""
        Write-Host "$($exp.Name):"
        $csv = Import-Csv $exp.Path
        $model3 = $csv | Where-Object { $_.model_name -eq "dml_residual_lstm" }
        if ($model3) {
            Write-Host "  MAE:  $($model3.MAE)"
            Write-Host "  RMSE: $($model3.RMSE)"
            Write-Host "  R2:   $($model3.R2)"
        }
    }
}

Write-Host ""
Write-Host "========================================================================"
