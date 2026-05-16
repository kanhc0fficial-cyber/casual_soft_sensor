#!/usr/bin/env python3
"""
自动运行剩余的消融实验
等待实验2完成后，依次运行实验3和4
"""

import subprocess
import time
from pathlib import Path
import sys

def check_experiment_completed(exp_dir):
    """检查实验是否完成"""
    metrics_file = Path(exp_dir) / "metrics_compare.csv"
    return metrics_file.exists()

def run_experiment(config_file, exp_name):
    """运行单个实验"""
    print(f"\n{'='*80}")
    print(f"开始运行: {exp_name}")
    print(f"配置文件: {config_file}")
    print(f"{'='*80}\n")
    
    cmd = [
        "python",
        "scripts/train_dml_residual_soft_sensor.py",
        "--config", config_file,
        "--only-model3"
    ]
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        print(f"\n✓ {exp_name} 完成！\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ {exp_name} 失败: {e}\n")
        return False
    except KeyboardInterrupt:
        print(f"\n⚠ {exp_name} 被用户中断\n")
        return False

def main():
    print("="*80)
    print("自动运行剩余的消融实验")
    print("="*80)
    print()
    
    # 检查实验1是否完成
    exp1_dir = "results/ablation_constraints/baseline_no_constraints"
    if not check_experiment_completed(exp1_dir):
        print("❌ 实验1（基线）尚未完成，请先运行实验1")
        return 1
    print("✓ 实验1（基线）已完成")
    
    # 等待实验2完成
    exp2_dir = "results/ablation_constraints/counterfactual_only"
    if not check_experiment_completed(exp2_dir):
        print("⏳ 等待实验2（仅反事实约束）完成...")
        print("   (每30秒检查一次，按Ctrl+C取消)")
        
        try:
            while not check_experiment_completed(exp2_dir):
                time.sleep(30)
                print(".", end="", flush=True)
        except KeyboardInterrupt:
            print("\n⚠ 用户取消等待")
            return 1
        
        print("\n✓ 实验2（仅反事实约束）已完成")
    else:
        print("✓ 实验2（仅反事实约束）已完成")
    
    # 运行实验3
    exp3_config = "configs/ablation_constraints_process.yaml"
    exp3_dir = "results/ablation_constraints/process_only"
    
    if check_experiment_completed(exp3_dir):
        print("✓ 实验3（仅工艺约束）已完成，跳过")
    else:
        if not run_experiment(exp3_config, "实验3（仅工艺约束）"):
            print("❌ 实验3失败，停止后续实验")
            return 1
    
    # 运行实验4
    exp4_config = "configs/ablation_constraints_both.yaml"
    exp4_dir = "results/ablation_constraints/both_constraints"
    
    if check_experiment_completed(exp4_dir):
        print("✓ 实验4（两种约束）已完成，跳过")
    else:
        if not run_experiment(exp4_config, "实验4（两种约束）"):
            print("❌ 实验4失败")
            return 1
    
    # 汇总结果
    print("\n" + "="*80)
    print("所有实验完成！正在汇总结果...")
    print("="*80 + "\n")
    
    subprocess.run(["python", "summarize_ablation_results.py"])
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
