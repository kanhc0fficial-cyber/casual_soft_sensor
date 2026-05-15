"""
run_dynotears_quick_test.py
============================
快速测试版本的 DYNOTEARS 因果发现，使用数据子集。

用法：
  python run_dynotears_quick_test.py --line xin1 --sample_size 10000 --epochs 50
"""

import sys
import os

# 临时修改 prepare_data 函数以支持采样
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 导入原始模块
import run_dynotears_dag as dyno
import causal_discovery_config as config
import pandas as pd
import numpy as np

# 保存原始 prepare_data 函数
original_prepare_data = config.prepare_data


def prepare_data_sampled(line="xin1", sample_size=10000):
    """
    修改版的 prepare_data，支持数据采样。
    
    参数：
      line: 产线名称
      sample_size: 采样大小（None 表示使用全部数据）
    """
    # 调用原始函数
    df, valid_vars, var_to_stage, var_to_group = original_prepare_data(line)
    
    # 如果指定了采样大小，进行采样
    if sample_size is not None and sample_size < len(df):
        print(f"[sample] 从 {len(df)} 个样本中随机采样 {sample_size} 个")
        df = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    
    return df, valid_vars, var_to_stage, var_to_group


def run_quick_test(line="xin1", sample_size=10000, epochs=50, threshold=0.03):
    """
    运行快速测试版本的 DYNOTEARS。
    
    参数：
      line: 产线名称
      sample_size: 采样大小
      epochs: 训练轮数
      threshold: 邻接矩阵阈值
    """
    # 临时替换 prepare_data 函数
    config.prepare_data = lambda l: prepare_data_sampled(l, sample_size)
    
    try:
        # 运行 DYNOTEARS
        dyno.run_dynotears(line=line, epochs=epochs, threshold=threshold)
    finally:
        # 恢复原始函数
        config.prepare_data = original_prepare_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="快速测试版本的 DYNOTEARS 因果发现（使用数据子集）"
    )
    parser.add_argument(
        "--line",
        choices=["xin1", "xin2"],
        default="xin1",
        help="产线选择（默认: xin1）"
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=10000,
        help="采样大小（默认: 10000）"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数（默认: 50）"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.03,
        help="邻接矩阵阈值（默认: 0.03）"
    )
    args = parser.parse_args()
    
    run_quick_test(
        line=args.line,
        sample_size=args.sample_size,
        epochs=args.epochs,
        threshold=args.threshold
    )
