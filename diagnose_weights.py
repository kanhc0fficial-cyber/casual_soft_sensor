"""
diagnose_weights.py
===================
诊断因果发现算法的权重分布，检查是否有指向 y_grade 的弱边被阈值过滤掉。

用法：
  python diagnose_weights.py --algo mb_cuts --line xin1 --sample_ratio 0.1
"""

import sys
import os
import argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import causal_discovery_config as config
import run_innovation_real_data as innovation


def diagnose_mb_cuts(line="xin1", sample_ratio=0.1):
    """诊断 MB-CUTS 的权重分布"""
    
    print("\n" + "="*70)
    print(f"诊断 MB-CUTS 权重分布  [产线={line}]  [采样={sample_ratio*100:.0f}%]")
    print("="*70)
    
    # 加载数据
    df, valid_vars, var_to_stage, var_to_group = config.prepare_data(line)
    
    if sample_ratio is not None and 0 < sample_ratio < 1:
        original_size = len(df)
        sample_size = int(original_size * sample_ratio)
        print(f"\n[采样] 从 {original_size} 个样本中采样 {sample_size} 个")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    N = len(valid_vars)
    print(f"变量数: {N}  样本数: {len(df)}")
    
    all_vars = valid_vars + ["y_grade"]
    X_all = df[all_vars].values.astype(np.float32)
    
    topo_mask = innovation.build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
    
    # 训练模型
    print(f"\n--- 训练 MB-CUTS ---")
    model = innovation.train_mb_cuts_real(X_all, topo_mask, verbose=True, epochs=50)
    
    # 提取权重矩阵
    with torch.no_grad():
        W = np.abs(model.W.detach().cpu().numpy())
    
    # 应用物理掩码
    W = W * topo_mask
    np.fill_diagonal(W, 0.0)
    
    # 分析权重分布
    print("\n" + "="*70)
    print("权重分析")
    print("="*70)
    
    # 全局统计
    all_weights = W.flatten()
    nonzero = all_weights[all_weights > 0]
    
    print(f"\n全局权重统计:")
    print(f"  总权重数: {len(all_weights)}")
    print(f"  非零权重数: {len(nonzero)}")
    print(f"  非零比例: {len(nonzero)/len(all_weights)*100:.2f}%")
    
    if len(nonzero) > 0:
        print(f"\n非零权重分布:")
        print(f"  最小值: {nonzero.min():.6f}")
        print(f"  最大值: {nonzero.max():.6f}")
        print(f"  平均值: {nonzero.mean():.6f}")
        print(f"  中位数: {np.median(nonzero):.6f}")
        
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        pct_values = np.percentile(nonzero, percentiles)
        print(f"\n分位数:")
        for p, v in zip(percentiles, pct_values):
            print(f"  {p:3d}%: {v:.6f}")
    
    # 重点：检查指向 y_grade 的权重
    print("\n" + "="*70)
    print("指向 y_grade 的权重（第 {} 列）".format(N))
    print("="*70)
    
    y_grade_col = W[:, N]  # 所有变量 → y_grade 的权重
    
    print(f"\n所有变量 → y_grade 的权重:")
    print(f"  非零权重数: {np.sum(y_grade_col > 0)}")
    print(f"  最大权重: {y_grade_col.max():.6f}")
    
    # 找出最大的 10 个权重
    top_indices = np.argsort(y_grade_col)[-10:][::-1]
    
    print(f"\nTop 10 变量 → y_grade 的权重:")
    for i, idx in enumerate(top_indices, 1):
        var_name = valid_vars[idx] if idx < N else "y_grade"
        weight = y_grade_col[idx]
        print(f"  {i:2d}. {var_name:30s}: {weight:.6f}")
    
    # 检查不同阈值下的边数
    print("\n" + "="*70)
    print("不同阈值下指向 y_grade 的边数")
    print("="*70)
    
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1]
    for thresh in thresholds:
        count = np.sum(y_grade_col > thresh)
        print(f"  阈值 {thresh:.3f}: {count} 条边")
    
    # 建议
    print("\n" + "="*70)
    print("建议")
    print("="*70)
    
    max_weight = y_grade_col.max()
    if max_weight < 0.001:
        print("⚠️  所有指向 y_grade 的权重都非常小（< 0.001）")
        print("   可能原因：")
        print("   1. 数据中 y_grade 与其他变量的因果关系确实很弱")
        print("   2. 需要更多训练轮数")
        print("   3. 需要更多数据（当前只用了 {}% 数据）".format(sample_ratio*100))
        print("   4. 特征工程：可能需要添加更多相关变量")
    elif max_weight < 0.01:
        print(f"✓  最大权重为 {max_weight:.6f}，建议使用阈值 < 0.01")
        print(f"   推荐阈值: {max_weight * 0.5:.6f}")
    elif max_weight < 0.05:
        print(f"✓  最大权重为 {max_weight:.6f}，建议使用阈值 < 0.05")
        print(f"   推荐阈值: {max_weight * 0.5:.6f}")
    else:
        print(f"✓  最大权重为 {max_weight:.6f}，当前阈值 0.05 应该能捕捉到边")
        print(f"   如果没有边，请检查 DAG 后处理是否移除了这些边")


def main():
    parser = argparse.ArgumentParser(
        description="诊断因果发现算法的权重分布"
    )
    parser.add_argument(
        "--algo",
        choices=["mb_cuts"],
        default="mb_cuts",
        help="算法选择（目前只支持 mb_cuts）"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2"],
        default="xin1",
        help="产线选择"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="采样比例（0.1 = 10%）"
    )
    args = parser.parse_args()
    
    if args.algo == "mb_cuts":
        diagnose_mb_cuts(args.line, args.sample_ratio)


if __name__ == "__main__":
    main()
