"""
run_causal_discovery_by_stage.py
================================
按阶段运行因果发现，并生成相关性分析报告

改进措施：
1. 每次只分析一个阶段的 DCS 变量
2. 自动移除高度相关的变量（相关系数 > 0.95）
3. 生成详细的相关性分析报告
4. 使用改进的超参数
"""

import os
import sys
import time
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config_v2 import prepare_data_by_stage, can_cause, STAGES

# 导入因果发现算法（复用现有代码）
from run_innovation_real_data import (
    train_mb_cuts_real,
    extract_adj_and_save,
    build_windows,
    DEVICE,
    WINDOW_SIZE,
    BATCH_SIZE,
    LR,
    ALGO_THRESHOLDS,
    OUT_DIR
)

import torch

def build_topology_mask_v2(valid_vars, var_to_stage):
    """
    构建物理因果可行性掩码（简化版）。

    参数:
        valid_vars:   有效变量列表（不含 y_grade）
        var_to_stage: 变量名 → 工序阶段的映射

    返回:
        mask: (N+1, N+1) 物理可行性掩码
    """
    N = len(valid_vars)
    mask = np.zeros((N + 1, N + 1), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i != j and can_cause(
                var_to_stage[valid_vars[i]], 
                var_to_stage[valid_vars[j]]
            ):
                mask[i, j] = 1.0
        # i → y_grade
        if can_cause(var_to_stage[valid_vars[i]], "Y"):
            mask[i, N] = 1.0

    # y_grade 不影响任何其他节点
    mask[N, :] = 0.0
    # 消除对角线（自环）
    np.fill_diagonal(mask, 0.0)
    return mask


def run_stage_causal_discovery(stage="浮选", line="xin1", algo="mb_cuts", 
                                epochs=50, correlation_threshold=0.95):
    """
    对单个阶段运行因果发现。

    参数:
        stage:                 阶段名称（"磁选", "塔磨", "浮选"）
        line:                  产线名称（"xin1" 或 "xin2"）
        algo:                  算法名称（"mb_cuts"）
        epochs:                训练轮数
        correlation_threshold: 相关系数阈值
    """
    print(f"\n{'='*70}")
    print(f"阶段因果发现：{stage}  [产线={line}]  [算法={algo.upper()}]")
    print(f"{'='*70}")

    t0 = time.time()

    # ─── 1. 加载数据并移除高度相关变量 ────────────────────────────────────
    df, valid_vars, var_to_stage, removed_vars, correlation_report = prepare_data_by_stage(
        stage=stage,
        line=line,
        correlation_threshold=correlation_threshold
    )

    # 保存相关性报告
    report_path = os.path.join(OUT_DIR, f"correlation_report_{stage}_{line}.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(correlation_report)
    print(f"\n✓ 相关性报告已保存：{report_path}")

    N = len(valid_vars)
    print(f"\n变量数: {N}  样本数: {len(df)}")
    print(f"移除的高度相关变量数: {len(removed_vars)}")

    # ─── 2. 构建物理拓扑掩码 ──────────────────────────────────────────────
    topo_mask = build_topology_mask_v2(valid_vars, var_to_stage)
    n_feasible = int(topo_mask.sum())
    total_possible = (N + 1) ** 2 - (N + 1)
    print(f"物理可行边数: {n_feasible} / {total_possible}  "
          f"({n_feasible / total_possible * 100:.1f}%)")

    # ─── 3. 准备训练数据 ──────────────────────────────────────────────────
    all_vars = valid_vars + ["y_grade"]
    X_all = df[all_vars].values.astype(np.float32)  # (T, N+1)

    # ─── 4. 运行 MB-CUTS ──────────────────────────────────────────────────
    if algo == "mb_cuts":
        print(f"\n--- 训练 MB-CUTS ---")
        model = train_mb_cuts_real(X_all, topo_mask, verbose=True, epochs=epochs)
        
        # 提取邻接矩阵并保存
        threshold = ALGO_THRESHOLDS.get("mb_cuts", 0.02)
        G, W = extract_adj_and_save(
            model, valid_vars, topo_mask, line, 
            f"mb_cuts_{stage}", threshold=threshold
        )
        
        # 清理内存
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()
    else:
        raise ValueError(f"不支持的算法：{algo}")

    elapsed = time.time() - t0
    print(f"\n[{stage}] 完成，耗时 {elapsed:.1f}s")

    return G, W, removed_vars, correlation_report


def main():
    parser = argparse.ArgumentParser(
        description="按阶段运行因果发现（磁选、塔磨、浮选）"
    )
    parser.add_argument(
        "--stage",
        choices=["磁选", "塔磨", "浮选", "all"],
        default="浮选",
        help="阶段选择（默认: 浮选）",
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2"],
        default="xin1",
        help="产线选择（默认: xin1）",
    )
    parser.add_argument(
        "--algo",
        choices=["mb_cuts"],
        default="mb_cuts",
        help="算法选择（默认: mb_cuts）",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数（默认: 50）",
    )
    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=0.95,
        help="相关系数阈值（默认: 0.95）",
    )
    args = parser.parse_args()

    # 运行因果发现
    if args.stage == "all":
        stages = ["磁选", "塔磨", "浮选"]
    else:
        stages = [args.stage]

    results = {}
    for stage in stages:
        G, W, removed_vars, report = run_stage_causal_discovery(
            stage=stage,
            line=args.line,
            algo=args.algo,
            epochs=args.epochs,
            correlation_threshold=args.correlation_threshold
        )
        results[stage] = {
            "graph": G,
            "adjacency": W,
            "removed_vars": removed_vars,
            "report": report
        }

    # 生成总结报告
    print(f"\n{'='*70}")
    print("因果发现总结")
    print(f"{'='*70}")
    for stage, result in results.items():
        n_edges = result["graph"].number_of_edges()
        n_removed = len(result["removed_vars"])
        print(f"\n{stage}:")
        print(f"  发现的因果边数: {n_edges}")
        print(f"  移除的高度相关变量数: {n_removed}")
        if n_removed > 0:
            print(f"  移除的变量: {result['removed_vars'][:3]}...")

    print(f"\n✓ 所有阶段完成")


if __name__ == "__main__":
    main()
