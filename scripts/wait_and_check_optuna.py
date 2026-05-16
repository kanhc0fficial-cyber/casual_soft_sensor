"""
等待Optuna优化完成并生成报告
"""
import time
import json
from pathlib import Path
import pandas as pd

def check_completion():
    """检查两个Optuna优化是否完成"""
    group_branch_done = False
    dml_residual_done = False
    
    # 检查门控分组
    gb_best_params = Path("results/group_branch_test_optuna/best_params.json")
    gb_history = Path("results/group_branch_test_optuna/optimization_history.csv")
    gb_metrics = Path("results/group_branch_test_optuna/group_branch_metrics.csv")
    
    if gb_best_params.exists() and gb_history.exists() and gb_metrics.exists():
        group_branch_done = True
    
    # 检查DML残差
    dml_best_params = Path("results/residual_soft_sensor_test_optuna/best_params.json")
    dml_history = Path("results/residual_soft_sensor_test_optuna/optimization_history.csv")
    dml_metrics = Path("results/residual_soft_sensor_test_optuna/metrics_compare.csv")
    
    if dml_best_params.exists() and dml_history.exists() and dml_metrics.exists():
        dml_residual_done = True
    
    return group_branch_done, dml_residual_done

def generate_report():
    """生成优化结果报告"""
    report = []
    report.append("=" * 80)
    report.append("OPTUNA优化完成报告")
    report.append("=" * 80)
    report.append("")
    
    # 门控分组模型
    report.append("## 1. 门控分组软测量模型")
    report.append("-" * 80)
    
    gb_best_params_path = Path("results/group_branch_test_optuna/best_params.json")
    gb_history_path = Path("results/group_branch_test_optuna/optimization_history.csv")
    gb_metrics_path = Path("results/group_branch_test_optuna/group_branch_metrics.csv")
    
    if gb_best_params_path.exists():
        with open(gb_best_params_path, 'r') as f:
            gb_best_params = json.load(f)
        report.append("### 最佳超参数:")
        for key, value in gb_best_params.items():
            report.append(f"  {key}: {value}")
        report.append("")
    
    if gb_metrics_path.exists():
        gb_metrics = pd.read_csv(gb_metrics_path)
        report.append("### 最终测试指标:")
        report.append(f"  MAE: {gb_metrics['MAE'].values[0]:.6f}")
        report.append(f"  RMSE: {gb_metrics['RMSE'].values[0]:.6f}")
        report.append(f"  R²: {gb_metrics['R2'].values[0]:.6f}")
        report.append("")
    
    if gb_history_path.exists():
        gb_history = pd.read_csv(gb_history_path)
        report.append(f"### 优化历史: {len(gb_history)} trials")
        report.append(f"  最佳验证损失: {gb_history['value'].min():.6f}")
        report.append(f"  最差验证损失: {gb_history['value'].max():.6f}")
        report.append("")
    
    # 对比原始结果
    gb_original_metrics_path = Path("results/group_branch/group_branch_metrics.csv")
    if gb_original_metrics_path.exists() and gb_metrics_path.exists():
        gb_original = pd.read_csv(gb_original_metrics_path)
        gb_optuna = pd.read_csv(gb_metrics_path)
        
        report.append("### 性能对比 (Optuna vs 原始):")
        report.append(f"  MAE: {gb_optuna['MAE'].values[0]:.6f} vs {gb_original['MAE'].values[0]:.6f} (改进: {(gb_original['MAE'].values[0] - gb_optuna['MAE'].values[0]) / gb_original['MAE'].values[0] * 100:.2f}%)")
        report.append(f"  RMSE: {gb_optuna['RMSE'].values[0]:.6f} vs {gb_original['RMSE'].values[0]:.6f} (改进: {(gb_original['RMSE'].values[0] - gb_optuna['RMSE'].values[0]) / gb_original['RMSE'].values[0] * 100:.2f}%)")
        report.append(f"  R²: {gb_optuna['R2'].values[0]:.6f} vs {gb_original['R2'].values[0]:.6f} (改进: {(gb_optuna['R2'].values[0] - gb_original['R2'].values[0]) / gb_original['R2'].values[0] * 100:.2f}%)")
        report.append("")
    
    report.append("")
    
    # DML残差模型
    report.append("## 2. DML残差软测量模型")
    report.append("-" * 80)
    
    dml_best_params_path = Path("results/residual_soft_sensor_test_optuna/best_params.json")
    dml_history_path = Path("results/residual_soft_sensor_test_optuna/optimization_history.csv")
    dml_metrics_path = Path("results/residual_soft_sensor_test_optuna/metrics_compare.csv")
    
    if dml_best_params_path.exists():
        with open(dml_best_params_path, 'r') as f:
            dml_best_params = json.load(f)
        report.append("### 最佳超参数:")
        for key, value in dml_best_params.items():
            report.append(f"  {key}: {value}")
        report.append("")
    
    if dml_metrics_path.exists():
        dml_metrics = pd.read_csv(dml_metrics_path)
        dml_residual_row = dml_metrics[dml_metrics['model_name'] == 'dml_residual_soft_sensor']
        if not dml_residual_row.empty:
            report.append("### 最终测试指标 (DML残差模型):")
            report.append(f"  MAE: {dml_residual_row['MAE'].values[0]:.6f}")
            report.append(f"  RMSE: {dml_residual_row['RMSE'].values[0]:.6f}")
            report.append(f"  R²: {dml_residual_row['R2'].values[0]:.6f}")
            report.append("")
    
    if dml_history_path.exists():
        dml_history = pd.read_csv(dml_history_path)
        report.append(f"### 优化历史: {len(dml_history)} trials")
        report.append(f"  最佳验证RMSE: {dml_history['value'].min():.6f}")
        report.append(f"  最差验证RMSE: {dml_history['value'].max():.6f}")
        report.append("")
    
    # 对比原始结果
    dml_original_metrics_path = Path("results/residual_soft_sensor_test/metrics_compare.csv")
    if dml_original_metrics_path.exists() and dml_metrics_path.exists():
        dml_original = pd.read_csv(dml_original_metrics_path)
        dml_optuna = pd.read_csv(dml_metrics_path)
        
        dml_original_row = dml_original[dml_original['model_name'] == 'dml_residual_soft_sensor']
        dml_optuna_row = dml_optuna[dml_optuna['model_name'] == 'dml_residual_soft_sensor']
        
        if not dml_original_row.empty and not dml_optuna_row.empty:
            report.append("### 性能对比 (Optuna vs 原始):")
            report.append(f"  MAE: {dml_optuna_row['MAE'].values[0]:.6f} vs {dml_original_row['MAE'].values[0]:.6f} (改进: {(dml_original_row['MAE'].values[0] - dml_optuna_row['MAE'].values[0]) / dml_original_row['MAE'].values[0] * 100:.2f}%)")
            report.append(f"  RMSE: {dml_optuna_row['RMSE'].values[0]:.6f} vs {dml_original_row['RMSE'].values[0]:.6f} (改进: {(dml_original_row['RMSE'].values[0] - dml_optuna_row['RMSE'].values[0]) / dml_original_row['RMSE'].values[0] * 100:.2f}%)")
            report.append(f"  R²: {dml_optuna_row['R2'].values[0]:.6f} vs {dml_original_row['R2'].values[0]:.6f} (改进: {(dml_optuna_row['R2'].values[0] - dml_original_row['R2'].values[0]) / dml_original_row['R2'].values[0] * 100:.2f}%)")
            report.append("")
    
    report.append("")
    report.append("=" * 80)
    report.append("报告结束")
    report.append("=" * 80)
    
    return "\n".join(report)

# 主循环
print("开始等待Optuna优化完成...")
print("检查间隔: 2小时")
print("无最大等待时间限制，将持续等待直到完成")

check_interval = 2 * 60 * 60  # 2小时
elapsed_time = 0

while True:
    gb_done, dml_done = check_completion()
    
    hours = elapsed_time // 3600
    minutes = (elapsed_time % 3600) // 60
    print(f"\n[已等待 {hours}小时{minutes}分钟] 检查状态:")
    print(f"  门控分组: {'✓ 完成' if gb_done else '⏳ 运行中'}")
    print(f"  DML残差: {'✓ 完成' if dml_done else '⏳ 运行中'}")
    
    if gb_done and dml_done:
        print("\n两个优化都已完成！生成报告...")
        report = generate_report()
        
        # 保存报告
        report_path = Path("results/OPTUNA_FINAL_REPORT.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n报告已保存: {report_path}")
        print("\n" + report)
        break
    
    print(f"下次检查时间: 2小时后")
    time.sleep(check_interval)
    elapsed_time += check_interval
