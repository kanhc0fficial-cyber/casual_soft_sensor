"""
诊断 MB-CUTS 因果发现问题的脚本
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from causal_discovery_config import prepare_data, can_cause

def diagnose_data():
    """诊断数据质量"""
    print("=" * 70)
    print("数据诊断")
    print("=" * 70)
    
    df, valid_vars, var_to_stage, var_to_group = prepare_data("xin1")
    
    print(f"\n1. 数据基本信息")
    print(f"   样本数: {len(df)}")
    print(f"   变量数: {len(valid_vars)}")
    print(f"   缺失值: {df.isnull().sum().sum()}")
    
    print(f"\n2. 变量统计信息")
    print(df.describe())
    
    print(f"\n3. 变量方差分析")
    for var in valid_vars[:5]:
        std = df[var].std()
        mean = df[var].mean()
        print(f"   {var}: mean={mean:.4f}, std={std:.4f}, cv={std/abs(mean) if mean != 0 else 0:.4f}")
    
    print(f"\n4. Spearman 相关性分析")
    X = df[valid_vars].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    corr_matrix, _ = spearmanr(X_norm)
    corr = np.array(corr_matrix)
    
    # 找出最强的相关性
    np.fill_diagonal(corr, 0)
    top_indices = np.argsort(np.abs(corr).flatten())[-10:]
    
    print(f"   最强的 10 个相关性:")
    for idx in reversed(top_indices):
        i, j = divmod(idx, len(valid_vars))
        if i < j:
            print(f"   {valid_vars[i]} <-> {valid_vars[j]}: {corr[i, j]:.4f}")
    
    print(f"\n5. 与目标变量的相关性")
    y = df["y_grade"].values
    y_norm = (y - y.mean()) / (y.std() + 1e-8)
    
    correlations = []
    for i, var in enumerate(valid_vars):
        x_norm = X_norm[:, i]
        corr_val = np.corrcoef(x_norm, y_norm)[0, 1]
        correlations.append((var, corr_val))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"   与 y_grade 相关性最强的 10 个变量:")
    for var, corr_val in correlations[:10]:
        print(f"   {var}: {corr_val:.4f}")

def diagnose_model():
    """诊断模型训练"""
    print("\n" + "=" * 70)
    print("模型诊断")
    print("=" * 70)
    
    df, valid_vars, var_to_stage, var_to_group = prepare_data("xin1")
    
    # 构建数据
    all_vars = valid_vars + ["y_grade"]
    X_all = df[all_vars].values.astype(np.float32)
    
    # 标准化
    X_norm = (X_all - X_all.mean(axis=0)) / (X_all.std(axis=0) + 1e-8)
    
    # 构建窗口
    WINDOW_SIZE = 15
    T, d = X_norm.shape
    xs, ys = [], []
    for start in range(0, T - WINDOW_SIZE):
        xs.append(X_norm[start:start + WINDOW_SIZE, :])
        ys.append(X_norm[start + WINDOW_SIZE, :])
    
    xs = np.array(xs, dtype=np.float32)
    ys = np.array(ys, dtype=np.float32)
    
    print(f"\n1. 窗口化数据")
    print(f"   输入形状: {xs.shape}")
    print(f"   输出形状: {ys.shape}")
    print(f"   输入范围: [{xs.min():.4f}, {xs.max():.4f}]")
    print(f"   输出范围: [{ys.min():.4f}, {ys.max():.4f}]")
    
    print(f"\n2. 简单线性模型基准")
    # 尝试简单的线性回归
    from sklearn.linear_model import LinearRegression
    
    xs_flat = xs.reshape(xs.shape[0], -1)
    model = LinearRegression()
    model.fit(xs_flat, ys)
    
    pred = model.predict(xs_flat)
    mse = np.mean((pred - ys) ** 2)
    rmse = np.sqrt(mse)
    
    print(f"   线性模型 MSE: {mse:.6f}")
    print(f"   线性模型 RMSE: {rmse:.6f}")
    print(f"   线性模型权重范围: [{model.coef_.min():.6f}, {model.coef_.max():.6f}]")
    
    print(f"\n3. 邻接矩阵初始化测试")
    # 测试邻接矩阵初始化
    W = np.random.uniform(-0.01, 0.01, (d, d))
    np.fill_diagonal(W, 0)
    
    print(f"   随机 W 范围: [{W.min():.6f}, {W.max():.6f}]")
    print(f"   随机 W 非零比例: {(np.abs(W) > 1e-6).sum() / (d*d):.4f}")
    
    # 测试 W 的作用
    x_sample = xs[0]  # (15, 34)
    x_agg = np.dot(x_sample, W)  # (15, 34)
    
    print(f"   聚合后数据范围: [{x_agg.min():.6f}, {x_agg.max():.6f}]")
    print(f"   聚合后数据方差: {x_agg.var():.6f}")

def diagnose_topology():
    """诊断物理拓扑掩码"""
    print("\n" + "=" * 70)
    print("物理拓扑掩码诊断")
    print("=" * 70)
    
    df, valid_vars, var_to_stage, var_to_group = prepare_data("xin1")
    
    N = len(valid_vars)
    mask = np.zeros((N + 1, N + 1), dtype=np.float32)
    
    for i in range(N):
        for j in range(N):
            if i != j and can_cause(
                var_to_stage[valid_vars[i]], var_to_stage[valid_vars[j]],
                var_to_group.get(valid_vars[i]), var_to_group.get(valid_vars[j]),
                "xin1"
            ):
                mask[i, j] = 1.0
        # i → y_grade
        if can_cause(
            var_to_stage[valid_vars[i]], "Y",
            var_to_group.get(valid_vars[i]), None,
            "xin1"
        ):
            mask[i, N] = 1.0
    
    print(f"\n1. 拓扑掩码统计")
    print(f"   总维度: {N+1}")
    print(f"   可行边数: {int(mask.sum())}")
    print(f"   总可能边数: {(N+1)**2 - (N+1)}")
    print(f"   可行比例: {mask.sum() / ((N+1)**2 - (N+1)):.4f}")
    
    print(f"\n2. 按工序分析")
    stages = {}
    for var in valid_vars:
        stage = var_to_stage[var]
        if stage not in stages:
            stages[stage] = []
        stages[stage].append(var)
    
    for stage in sorted(stages.keys(), key=lambda x: ["粗选", "扫选", "精选"].index(x) if x in ["粗选", "扫选", "精选"] else 999):
        vars_in_stage = stages[stage]
        idx_in_stage = [valid_vars.index(v) for v in vars_in_stage]
        
        # 该工序内部的可行边
        internal_edges = 0
        for i in idx_in_stage:
            for j in idx_in_stage:
                if i != j and mask[i, j] > 0:
                    internal_edges += 1
        
        # 该工序指向其他工序的可行边
        outgoing_edges = 0
        for i in idx_in_stage:
            for j in range(N):
                if j not in idx_in_stage and mask[i, j] > 0:
                    outgoing_edges += 1
        
        print(f"   {stage}: {len(vars_in_stage)} 个变量, 内部边 {internal_edges}, 指向外部边 {outgoing_edges}")

if __name__ == "__main__":
    diagnose_data()
    diagnose_model()
    diagnose_topology()
    
    print("\n" + "=" * 70)
    print("诊断完成")
    print("=" * 70)
