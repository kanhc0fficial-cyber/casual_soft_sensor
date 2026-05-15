"""
run_quick_test.py
=================
快速测试因果发现算法，使用 1/10 数据集。

用法：
  python run_quick_test.py --algo mb_cuts --line xin1
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import causal_discovery_config as config

# 保存原始函数
original_prepare_data = config.prepare_data


def prepare_data_sampled(line="xin1", sample_ratio=0.1):
    """
    修改版的 prepare_data，使用数据子集。
    
    参数：
      line: 产线名称
      sample_ratio: 采样比例（0.1 = 10%）
    """
    # 调用原始函数
    df, valid_vars, var_to_stage, var_to_group = original_prepare_data(line)
    
    # 计算采样大小
    sample_size = int(len(df) * sample_ratio)
    
    print(f"\n[SAMPLE] 从 {len(df)} 个样本中采样 {sample_size} 个 ({sample_ratio*100:.1f}%)")
    
    # 随机采样
    df_sampled = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    print(f"[SAMPLE] 采样后数据形状: {df_sampled.shape}")
    
    return df_sampled, valid_vars, var_to_stage, var_to_group


def run_mb_cuts_quick(line="xin1", sample_ratio=0.1):
    """运行 MB-CUTS 快速测试"""
    print("\n" + "="*70)
    print(f"快速测试：MB-CUTS  [产线={line}]  [采样比例={sample_ratio*100:.0f}%]")
    print("="*70)
    
    # 临时替换 prepare_data
    config.prepare_data = lambda l: prepare_data_sampled(l, sample_ratio)
    
    try:
        import run_innovation_real_data as innovation
        
        # 只运行 MB-CUTS
        t0 = innovation.time.time()
        df, valid_vars, var_to_stage, var_to_group = config.prepare_data(line)
        N = len(valid_vars)
        print(f"变量数: {N}  样本数: {len(df)}")
        
        all_vars = valid_vars + ["y_grade"]
        X_all = df[all_vars].values.astype(np.float32)
        
        topo_mask = innovation.build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
        n_feasible = int(topo_mask.sum())
        total_possible = (N + 1) ** 2 - (N + 1)
        print(f"物理可行边数: {n_feasible} / {total_possible}  "
              f"({n_feasible / total_possible * 100:.1f}%)")
        
        print(f"\n--- 训练 MB-CUTS ---")
        model_mb = innovation.train_mb_cuts_real(X_all, topo_mask, verbose=True, epochs=50)
        
        innovation.extract_adj_and_save(
            model_mb, valid_vars, topo_mask, line, "mb_cuts_quick",
            threshold=innovation.ALGO_THRESHOLDS.get("mb_cuts", 0.05)
        )
        
        elapsed = innovation.time.time() - t0
        print(f"\n[{line}] MB-CUTS 完成，耗时 {elapsed:.1f}s")
        
    finally:
        # 恢复原始函数
        config.prepare_data = original_prepare_data


def run_multiscale_nts_quick(line="xin1", sample_ratio=0.1):
    """运行 MultiScale-NTS 快速测试"""
    print("\n" + "="*70)
    print(f"快速测试：MultiScale-NTS  [产线={line}]  [采样比例={sample_ratio*100:.0f}%]")
    print("="*70)
    
    # 临时替换 prepare_data
    config.prepare_data = lambda l: prepare_data_sampled(l, sample_ratio)
    
    try:
        import run_innovation_real_data as innovation
        
        t0 = innovation.time.time()
        df, valid_vars, var_to_stage, var_to_group = config.prepare_data(line)
        N = len(valid_vars)
        print(f"变量数: {N}  样本数: {len(df)}")
        
        all_vars = valid_vars + ["y_grade"]
        X_all = df[all_vars].values.astype(np.float32)
        d = N + 1
        
        topo_mask = innovation.build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
        n_feasible = int(topo_mask.sum())
        total_possible = (N + 1) ** 2 - (N + 1)
        print(f"物理可行边数: {n_feasible} / {total_possible}  "
              f"({n_feasible / total_possible * 100:.1f}%)")
        
        print(f"\n--- 训练 MultiScale-NTS ---")
        model_ms = innovation.MultiScaleNTSNet(d).to(innovation.DEVICE)
        model_ms = innovation.train_with_topo_mask(
            model_ms, X_all, topo_mask, epochs=50, algo_name="MultiScale-NTS"
        )
        
        innovation.extract_adj_and_save(
            model_ms, valid_vars, topo_mask, line, "multiscale_nts_quick",
            threshold=innovation.ALGO_THRESHOLDS.get("multiscale_nts", 0.03)
        )
        
        elapsed = innovation.time.time() - t0
        print(f"\n[{line}] MultiScale-NTS 完成，耗时 {elapsed:.1f}s")
        
    finally:
        # 恢复原始函数
        config.prepare_data = original_prepare_data


def run_biattn_cuts_quick(line="xin1", sample_ratio=0.1):
    """运行 BiAttn-CUTS 快速测试"""
    print("\n" + "="*70)
    print(f"快速测试：BiAttn-CUTS  [产线={line}]  [采样比例={sample_ratio*100:.0f}%]")
    print("="*70)
    
    # 临时替换 prepare_data
    config.prepare_data = lambda l: prepare_data_sampled(l, sample_ratio)
    
    try:
        import run_innovation_real_data as innovation
        
        t0 = innovation.time.time()
        df, valid_vars, var_to_stage, var_to_group = config.prepare_data(line)
        N = len(valid_vars)
        print(f"变量数: {N}  样本数: {len(df)}")
        
        all_vars = valid_vars + ["y_grade"]
        X_all = df[all_vars].values.astype(np.float32)
        d = N + 1
        
        topo_mask = innovation.build_topology_mask(valid_vars, var_to_stage, var_to_group, line)
        n_feasible = int(topo_mask.sum())
        total_possible = (N + 1) ** 2 - (N + 1)
        print(f"物理可行边数: {n_feasible} / {total_possible}  "
              f"({n_feasible / total_possible * 100:.1f}%)")
        
        print(f"\n--- 训练 BiAttn-CUTS ---")
        model_biattn = innovation.BiAttnCUTSNet(d).to(innovation.DEVICE)
        model_biattn = innovation.train_with_topo_mask(
            model_biattn, X_all, topo_mask, epochs=50, algo_name="BiAttn-CUTS"
        )
        
        innovation.extract_adj_and_save(
            model_biattn, valid_vars, topo_mask, line, "biattn_cuts_quick",
            threshold=innovation.ALGO_THRESHOLDS.get("biattn_cuts", 0.30)
        )
        
        elapsed = innovation.time.time() - t0
        print(f"\n[{line}] BiAttn-CUTS 完成，耗时 {elapsed:.1f}s")
        
    finally:
        # 恢复原始函数
        config.prepare_data = original_prepare_data


def main():
    parser = argparse.ArgumentParser(
        description="快速测试因果发现算法（使用 1/10 数据集）"
    )
    parser.add_argument(
        "--algo",
        choices=["mb_cuts", "multiscale_nts", "biattn_cuts", "all"],
        default="mb_cuts",
        help="算法选择（默认: mb_cuts，最快）"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2"],
        default="xin1",
        help="产线选择（默认: xin1）"
    )
    parser.add_argument(
        "--sample_ratio",
        type=float,
        default=0.1,
        help="采样比例（默认: 0.1 = 10%%）"
    )
    args = parser.parse_args()
    
    if args.algo == "mb_cuts":
        run_mb_cuts_quick(args.line, args.sample_ratio)
    elif args.algo == "multiscale_nts":
        run_multiscale_nts_quick(args.line, args.sample_ratio)
    elif args.algo == "biattn_cuts":
        run_biattn_cuts_quick(args.line, args.sample_ratio)
    elif args.algo == "all":
        print("\n运行所有算法...")
        run_mb_cuts_quick(args.line, args.sample_ratio)
        run_multiscale_nts_quick(args.line, args.sample_ratio)
        run_biattn_cuts_quick(args.line, args.sample_ratio)
    
    print("\n" + "="*70)
    print("快速测试完成！")
    print("="*70)
    print("\n输出文件位于: 因果发现结果/")
    print("  - mb_cuts_quick_real_dag_xin1.graphml")
    print("  - multiscale_nts_quick_real_dag_xin1.graphml")
    print("  - biattn_cuts_quick_real_dag_xin1.graphml")


if __name__ == "__main__":
    main()
