#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dml_causal_effect_value/run_dml_batch_xin1.py
==============================================
批量 DML 因果效应估计 — 新一线浮选精矿品位

基于 DAG 自动提取所有祖先节点（直接和间接），为每个祖先变量计算因果效应 θ。

工作流程：
    1. 调用 target_causal_projection.py 从 global_edges.csv 提取目标的祖先节点
    2. 读取生成的 dml_jobs.csv（包含所有需要 DML 的处理变量）
    3. 对每个处理变量，调用 run_dml_xin2.py 的核心函数计算 θ
    4. 汇总所有结果到一个 CSV 文件

输出：
    dml_causal_effect_value/结果/<timestamp>/dml_theta_all_xin1.csv  — 所有变量的 θ 汇总
    dml_causal_effect_value/结果/<timestamp>/dml_jobs.csv           — DML 任务表（从投影生成）
    dml_causal_effect_value/结果/<timestamp>/target_ancestors.csv   — 祖先节点列表

运行示例：
    python dml_causal_effect_value/run_dml_batch_xin1.py \\
        --dataset path/to/simulation.parquet \\
        --n-folds 5
"""

import os
import sys
import time
import argparse
import subprocess
from datetime import datetime
from pathlib import Path

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

if sys.platform.startswith("win"):
    sys.stdout.reconfigure(encoding="utf-8")

# 导入单变量 DML 的核心函数
_THIS_DIR = Path(__file__).parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from dml_causal_effect_value.run_dml_xin2 import (
    load_data,
    preprocess_features,
    run_dml,
    TEMPORAL_LAGS,
    TEMPORAL_WINDOWS,
)

# ═══════════════════════════════════════════════════════════════════════════
#  路径配置
# ═══════════════════════════════════════════════════════════════════════════
_RUN_TAG = datetime.now().strftime("%Y%m%d_%H%M%S")
_BASE_RESULT = _THIS_DIR / "结果"
RESULT_DIR = _BASE_RESULT / _RUN_TAG

# DAG 边表路径
EDGE_PATH = _PROJECT_ROOT / "data" / "features" / "global_edges.csv"

# 目标变量
TARGET_VARIABLE = "y_fx_xin1"

# 默认数据集路径
SIMULATION_DATASET = r"C:\Users\goldenwhale\Downloads\my_mining_simulation\output\simulation_2months_seq_hybrid_normal_fast_sampling.parquet"

# 全局超参数
RANDOM_SEED = 42
N_FOLDS = 5


# ═══════════════════════════════════════════════════════════════════════════
#  因果投影：提取祖先节点和 DML 任务
# ═══════════════════════════════════════════════════════════════════════════

def run_causal_projection(edge_path: Path, target: str, output_dir: Path):
    """
    调用 target_causal_projection.py 生成 DML 任务表。
    
    输出文件：
        - dml_jobs.csv：每行一个处理变量及其调整集
        - target_ancestors.csv：所有祖先节点
        - target_parents.csv：直接父节点
    """
    print("\n" + "=" * 70)
    print("  步骤 1：因果投影 — 提取祖先节点和 DML 任务")
    print("=" * 70)
    
    projection_script = _PROJECT_ROOT / "src" / "target_causal_projection.py"
    
    if not projection_script.exists():
        raise FileNotFoundError(f"找不到因果投影脚本：{projection_script}")
    
    if not edge_path.exists():
        raise FileNotFoundError(f"找不到 DAG 边表：{edge_path}")
    
    cmd = [
        sys.executable,
        str(projection_script),
        "--edge_path", str(edge_path),
        "--target", target,
        "--output_dir", str(output_dir),
    ]
    
    print(f"[run_causal_projection] 执行命令：{' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    
    if result.returncode != 0:
        print("[ERROR] 因果投影失败：")
        print(result.stderr)
        raise RuntimeError("因果投影脚本执行失败")
    
    print(result.stdout)
    print("[run_causal_projection] 因果投影完成")


# ═══════════════════════════════════════════════════════════════════════════
#  批量 DML 执行
# ═══════════════════════════════════════════════════════════════════════════

def run_batch_dml(
    dml_jobs_path: Path,
    dataset_path: str,
    n_folds: int,
) -> pd.DataFrame:
    """
    读取 DML 任务表，对每个处理变量执行 DML，返回汇总结果。
    
    返回 DataFrame，列：
        treatment, treatment_variable, theta, se, ci_lo, ci_hi, n_effective, status, error_msg
    """
    print("\n" + "=" * 70)
    print("  步骤 2：批量 DML 因果效应估计")
    print("=" * 70)
    
    if not dml_jobs_path.exists():
        raise FileNotFoundError(f"找不到 DML 任务表：{dml_jobs_path}")
    
    dml_jobs = pd.read_csv(dml_jobs_path)
    print(f"[run_batch_dml] 读取 DML 任务表：{len(dml_jobs)} 个处理变量")
    
    if len(dml_jobs) == 0:
        print("[WARNING] DML 任务表为空，没有需要计算的因果效应")
        return pd.DataFrame()
    
    # 预加载数据（所有任务共享）
    print(f"\n[数据加载] 读取数据集：{dataset_path}")
    
    results = []
    
    for idx, row in dml_jobs.iterrows():
        treatment = row["treatment"]
        treatment_var = row["treatment_variable"]
        
        print(f"\n{'─' * 70}")
        print(f"  任务 {idx + 1}/{len(dml_jobs)}：{treatment}")
        print(f"{'─' * 70}")
        
        try:
            # 加载数据（每次都重新加载，因为 treatment 不同）
            Y, T, X_raw, feat_cols, df_index = load_data(dataset_path, treatment)
            
            # 特征预处理
            X_full, vt, col_means = preprocess_features(X_raw)
            X_full = X_full.astype(np.float32)
            
            # 执行 DML
            result = run_dml(Y, T, X_full, n_folds=n_folds)
            
            # 记录结果
            results.append({
                "treatment": treatment,
                "treatment_variable": treatment_var,
                "theta": result["theta"],
                "se": result["se"],
                "ci_lo": result["ci_lo"],
                "ci_hi": result["ci_hi"],
                "n_effective": result["n_effective"],
                "status": "success",
                "error_msg": "",
            })
            
            print(f"  ✓ θ = {result['theta']:.6f}  SE = {result['se']:.6f}  "
                  f"95% CI = [{result['ci_lo']:.6f}, {result['ci_hi']:.6f}]")
        
        except Exception as e:
            print(f"  ✗ 失败：{str(e)}")
            results.append({
                "treatment": treatment,
                "treatment_variable": treatment_var,
                "theta": np.nan,
                "se": np.nan,
                "ci_lo": np.nan,
                "ci_hi": np.nan,
                "n_effective": 0,
                "status": "failed",
                "error_msg": str(e),
            })
    
    results_df = pd.DataFrame(results)
    print(f"\n[run_batch_dml] 完成 {len(results_df)} 个 DML 任务")
    print(f"  成功：{(results_df['status'] == 'success').sum()} 个")
    print(f"  失败：{(results_df['status'] == 'failed').sum()} 个")
    
    return results_df


# ═══════════════════════════════════════════════════════════════════════════
#  结果保存
# ═══════════════════════════════════════════════════════════════════════════

def save_batch_results(results_df: pd.DataFrame, output_dir: Path):
    """保存批量 DML 结果到 CSV。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "dml_theta_all_xin1.csv"
    results_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\n[保存] 批量 DML 结果：{output_path}")
    
    # 打印汇总统计
    success_df = results_df[results_df["status"] == "success"]
    if len(success_df) > 0:
        print("\n" + "=" * 70)
        print("  因果效应汇总（成功的任务）")
        print("=" * 70)
        print(success_df[["treatment_variable", "theta", "se", "ci_lo", "ci_hi"]].to_string(index=False))
        print("=" * 70)


# ═══════════════════════════════════════════════════════════════════════════
#  命令行入口
# ═══════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="批量 DML 因果效应估计（新一线浮选精矿品位）— 自动处理所有祖先节点"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="仿真 Parquet 数据集路径，覆盖脚本内默认路径",
    )
    parser.add_argument(
        "--edge-path",
        default=None,
        help="DAG 边表 CSV 路径（默认：data/features/global_edges.csv）",
    )
    parser.add_argument(
        "--target",
        default=TARGET_VARIABLE,
        help=f"目标变量名（默认：{TARGET_VARIABLE}）",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=N_FOLDS,
        help=f"Cross-fitting 折数（默认：{N_FOLDS}）",
    )
    return parser.parse_args()


def main():
    global N_FOLDS
    args = parse_args()
    N_FOLDS = args.n_folds
    
    dataset_path = args.dataset or SIMULATION_DATASET
    edge_path = Path(args.edge_path) if args.edge_path else EDGE_PATH
    target = args.target
    
    print("=" * 70)
    print("  批量 DML 因果效应估计 — 新一线浮选精矿品位")
    print(f"  目标变量：{target}")
    print(f"  DAG 边表：{edge_path}")
    print(f"  数据集：{dataset_path}")
    print(f"  Cross-fitting K：{N_FOLDS}")
    print(f"  结果目录：{RESULT_DIR}")
    print("=" * 70)
    
    t_start = time.time()
    
    # 步骤 1：因果投影
    projection_dir = RESULT_DIR / "causal_projection"
    try:
        run_causal_projection(edge_path, target, projection_dir)
    except Exception as e:
        print(f"\n[ERROR] 因果投影失败：{e}")
        sys.exit(1)
    
    # 步骤 2：批量 DML
    dml_jobs_path = projection_dir / "dml_jobs.csv"
    try:
        results_df = run_batch_dml(dml_jobs_path, dataset_path, N_FOLDS)
    except Exception as e:
        print(f"\n[ERROR] 批量 DML 失败：{e}")
        sys.exit(1)
    
    # 步骤 3：保存结果
    save_batch_results(results_df, RESULT_DIR)
    
    # 复制投影文件到结果目录（方便查看）
    import shutil
    for fname in ["dml_jobs.csv", "target_ancestors.csv", "target_parents.csv"]:
        src = projection_dir / fname
        if src.exists():
            dst = RESULT_DIR / fname
            shutil.copy(src, dst)
            print(f"[保存] {fname} -> {dst}")
    
    print(f"\n{'=' * 70}")
    print(f"  批量 DML 完成！总耗时：{time.time() - t_start:.1f}s")
    print(f"  结果已保存至：{RESULT_DIR}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
