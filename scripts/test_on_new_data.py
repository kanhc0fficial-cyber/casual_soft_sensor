"""
test_on_new_data.py
===================
在新数据上测试已训练的模型

用法：
  # 测试门控分组模型
  python scripts/test_on_new_data.py --model group_branch --checkpoint results/group_branch/model_checkpoint.pt --config configs/group_branch.yaml --data path/to/data.parquet
  
  # 测试DML残差模型
  python scripts/test_on_new_data.py --model dml_residual --checkpoint results/residual_soft_sensor/dml_residual_lstm_checkpoint.pt --config configs/residual_soft_sensor.yaml --data path/to/data.parquet
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.group_branch import CausalGroupBranchModel


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("test_on_new_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    fmt = logging.Formatter("%(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def make_windows(X: np.ndarray, y: np.ndarray, window_size: int):
    N = len(X)
    if N < window_size:
        raise ValueError(f"数据长度 {N} < window_size {window_size}")
    
    num_windows = N - window_size + 1
    Xw = np.zeros((num_windows, window_size, X.shape[1]), dtype=X.dtype)
    yw = np.zeros(num_windows, dtype=y.dtype)
    
    for i in range(num_windows):
        Xw[i] = X[i : i + window_size]
        yw[i] = y[i + window_size - 1]
    
    return Xw, yw


def test_group_branch_model(
    checkpoint_path: Path,
    config_path: str,
    data_path: str,
    output_dir: Path,
    logger: logging.Logger,
):
    """测试门控分组模型"""
    
    logger.info("=" * 60)
    logger.info("测试门控分组模型")
    logger.info("=" * 60)
    
    # 加载配置
    cfg = load_config(config_path)
    
    # 加载数据
    logger.info(f"读取数据: {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"数据形状: {df.shape}")
    
    # 预处理
    time_col = cfg.get("time_col")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        logger.info(f"已按时间列 '{time_col}' 排序")
    
    # 确定特征列
    target_col = cfg["target_col"]
    exclude_cols = set(cfg.get("exclude_cols", []))
    if time_col:
        exclude_cols.add(time_col)
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col and c not in exclude_cols]
    
    logger.info(f"特征列数量: {len(feature_cols)}")
    logger.info(f"目标变量: {target_col}")
    
    # 加载checkpoint
    logger.info(f"加载模型checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # 重建模型
    groups_cfg = cfg.get("groups", {})
    model_cfg = cfg.get("model", {})
    window_size = cfg["window_size"]
    num_features = len(feature_cols)
    
    model = CausalGroupBranchModel(
        groups_cfg=groups_cfg,
        model_cfg=model_cfg,
        window_size=window_size,
        num_features=num_features,
        allow_feature_overlap=bool(cfg.get("allow_feature_overlap", False)),
        warn_unused_features=False,
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    logger.info("模型加载成功")
    
    # 准备数据
    from sklearn.preprocessing import StandardScaler
    
    # 注意：这里使用新数据的统计量进行标准化
    # 如果需要使用训练时的统计量，需要在checkpoint中保存feature_scaler
    feat_scaler = StandardScaler()
    X = feat_scaler.fit_transform(df[feature_cols].values)
    y = df[target_col].values
    
    # Y的scaler从checkpoint加载
    y_scaler_mean = checkpoint['y_scaler_mean']
    y_scaler_scale = checkpoint['y_scaler_scale']
    y_scaled = (y - y_scaler_mean) / y_scaler_scale
    
    # 滑动窗口
    Xw, yw = make_windows(X, y_scaled, window_size)
    logger.info(f"窗口数量: {len(Xw)}")
    
    # 预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    Xw_t = torch.tensor(Xw, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        output = model(Xw_t)
        y_pred_scaled = output["y_hat"].squeeze(-1).cpu().numpy()
        branch_outputs = output["branch_outputs"].cpu().numpy()
        gates = output["gates"].cpu().numpy()
    
    # 反标准化
    y_pred = y_pred_scaled * y_scaler_scale + y_scaler_mean
    y_true = yw * y_scaler_scale + y_scaler_mean
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"测试指标: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    
    # 保存结果
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 指标
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {metrics_path}")
    
    # 2. 预测结果
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    
    group_names = list(groups_cfg.keys())
    for k, name in enumerate(group_names):
        pred_df[f"branch_{name}"] = branch_outputs[:, k]
        pred_df[f"contribution_{name}"] = gates[k] * branch_outputs[:, k]
    
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {pred_path}")
    
    # 3. Gate值
    gates_df = pd.DataFrame({"group": group_names, "gate_value": gates})
    gates_path = output_dir / "test_gates.csv"
    gates_df.to_csv(gates_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {gates_path}")
    
    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)


def test_dml_residual_model(
    checkpoint_path: Path,
    config_path: str,
    data_path: str,
    output_dir: Path,
    logger: logging.Logger,
):
    """测试DML残差模型"""
    
    logger.info("=" * 60)
    logger.info("测试DML残差模型")
    logger.info("=" * 60)
    
    logger.info("注意: DML残差模型需要完整的训练流程（包括g_model和q_models）")
    logger.info("当前仅加载LSTM部分，无法完整复现DML流程")
    logger.info("建议使用完整的训练脚本在新数据上重新训练")
    
    # 这里可以添加LSTM部分的测试逻辑
    # 但由于DML模型依赖于g_model和q_models，完整测试需要更复杂的逻辑
    
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="在新数据上测试已训练的模型")
    parser.add_argument("--model", required=True, choices=["group_branch", "dml_residual"], help="模型类型")
    parser.add_argument("--checkpoint", required=True, help="模型checkpoint路径")
    parser.add_argument("--config", required=True, help="配置文件路径")
    parser.add_argument("--data", required=True, help="测试数据路径")
    parser.add_argument("--output", default=None, help="输出目录（默认：results/test_on_new_data）")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"错误: checkpoint文件不存在: {checkpoint_path}")
        sys.exit(1)
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("results") / "test_on_new_data" / args.model
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_log.txt"
    logger = setup_logger(log_path)
    
    logger.info(f"模型类型: {args.model}")
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"测试数据: {args.data}")
    logger.info(f"输出目录: {output_dir}")
    
    if args.model == "group_branch":
        test_group_branch_model(checkpoint_path, args.config, args.data, output_dir, logger)
    elif args.model == "dml_residual":
        test_dml_residual_model(checkpoint_path, args.config, args.data, output_dir, logger)


if __name__ == "__main__":
    main()
