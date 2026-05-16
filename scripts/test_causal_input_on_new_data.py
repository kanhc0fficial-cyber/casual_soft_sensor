"""
test_causal_input_on_new_data.py
=================================
在新数据上测试因果输入模型（Model 1）

用法：
  python scripts/test_causal_input_on_new_data.py \\
    --checkpoint results/residual_soft_sensor_test_optuna/causal_input_lstm_checkpoint.pt \\
    --scalers results/residual_soft_sensor_test_optuna/causal_input_scalers.pkl \\
    --data "C:\\Users\\goldenwhale\\Downloads\\my_mining_simulation\\output\\multiregime_6x14400_seq_hybrid_normal_fast_sampling_noclip.parquet"
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_dml_residual_soft_sensor import LSTMRegressor


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("test_causal_input")
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


def main():
    parser = argparse.ArgumentParser(description="在新数据上测试因果输入模型")
    parser.add_argument("--checkpoint", required=True, help="LSTM checkpoint路径")
    parser.add_argument("--scalers", required=True, help="Scalers pickle文件路径")
    parser.add_argument("--data", required=True, help="测试数据路径")
    parser.add_argument("--output", default=None, help="输出目录（默认：results/test_causal_input_on_new_data）")
    parser.add_argument("--target", default="y_fx_xin1", help="目标变量列名")
    parser.add_argument("--time_col", default="t", help="时间列名")
    parser.add_argument("--window_size", type=int, default=12, help="滑动窗口大小")
    
    args = parser.parse_args()
    
    checkpoint_path = Path(args.checkpoint)
    scalers_path = Path(args.scalers)
    
    if not checkpoint_path.exists():
        print(f"错误: checkpoint文件不存在: {checkpoint_path}")
        sys.exit(1)
    
    if not scalers_path.exists():
        print(f"错误: scalers文件不存在: {scalers_path}")
        sys.exit(1)
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("results") / "test_causal_input_on_new_data"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_log.txt"
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info("测试因果输入模型（Model 1）")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Scalers: {args.scalers}")
    logger.info(f"测试数据: {args.data}")
    logger.info(f"输出目录: {output_dir}")
    
    # 1. 加载LSTM checkpoint
    logger.info("-" * 40)
    logger.info("加载LSTM模型...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    
    logger.info(f"  input_size: {input_size}")
    logger.info(f"  hidden_size: {hidden_size}")
    logger.info(f"  num_layers: {num_layers}")
    logger.info(f"  dropout: {dropout:.4f}")
    
    # 2. 重建LSTM模型
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model._model, model._device = model._build()
    model._model.load_state_dict(checkpoint['model_state_dict'])
    model._model.eval()
    logger.info("✅ LSTM模型加载成功")
    
    # 3. 加载scalers
    logger.info("-" * 40)
    logger.info("加载scalers...")
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    as_scaler = scalers['as_scaler']
    y_scaler = scalers['y_scaler']
    feature_cols = scalers['feature_cols']
    
    logger.info(f"  特征数量: {len(feature_cols)}")
    logger.info(f"  特征列: {feature_cols[:5]}... (显示前5个)")
    logger.info("✅ Scalers加载成功")
    
    # 4. 加载测试数据
    logger.info("-" * 40)
    logger.info(f"读取测试数据: {args.data}")
    df = pd.read_parquet(args.data)
    logger.info(f"  数据形状: {df.shape}")
    
    # 排序
    if args.time_col and args.time_col in df.columns:
        df = df.sort_values(args.time_col).reset_index(drop=True)
        logger.info(f"  已按时间列 '{args.time_col}' 排序")
    
    # 检查特征列是否存在
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"❌ 缺少特征列 ({len(missing_cols)}): {missing_cols[:10]}...")
        sys.exit(1)
    
    # 检查目标列是否存在
    if args.target not in df.columns:
        logger.error(f"❌ 缺少目标列: {args.target}")
        sys.exit(1)
    
    logger.info("✅ 所有必要列都存在")
    
    # 5. 准备数据
    logger.info("-" * 40)
    logger.info("准备数据...")
    
    X = as_scaler.transform(df[feature_cols].values)
    y = df[args.target].values
    
    # 标准化y（用于计算窗口）
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()
    
    # 滑动窗口
    Xw, yw = make_windows(X, y_scaled, args.window_size)
    logger.info(f"  窗口数量: {len(Xw)}")
    
    # 6. 预测
    logger.info("-" * 40)
    logger.info("开始预测...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model._device = device
    model._model = model._model.to(device)
    
    Xw_t = torch.tensor(Xw, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model._model(Xw_t).cpu().numpy()
    
    # 反标准化
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw.reshape(-1, 1)).ravel()
    
    # 7. 计算指标
    logger.info("-" * 40)
    logger.info("计算性能指标...")
    metrics = compute_metrics(y_true, y_pred)
    
    logger.info(f"  MAE:  {metrics['MAE']:.4f}")
    logger.info(f"  RMSE: {metrics['RMSE']:.4f}")
    logger.info(f"  R²:   {metrics['R2']:.4f}")
    
    # 8. 保存结果
    logger.info("-" * 40)
    logger.info("保存结果...")
    
    # 8.1 指标
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "test_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 已保存: {metrics_path}")
    
    # 8.2 预测结果
    pred_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "error": y_true - y_pred,
        "abs_error": np.abs(y_true - y_pred),
    })
    pred_path = output_dir / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 已保存: {pred_path}")
    
    # 8.3 误差统计
    error_stats = {
        "mean_error": float(np.mean(y_true - y_pred)),
        "std_error": float(np.std(y_true - y_pred)),
        "min_error": float(np.min(y_true - y_pred)),
        "max_error": float(np.max(y_true - y_pred)),
        "median_abs_error": float(np.median(np.abs(y_true - y_pred))),
    }
    error_stats_df = pd.DataFrame([error_stats])
    error_stats_path = output_dir / "error_statistics.csv"
    error_stats_df.to_csv(error_stats_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 已保存: {error_stats_path}")
    
    logger.info("=" * 60)
    logger.info("测试完成")
    logger.info("=" * 60)
    
    # 9. 打印总结
    print("\n" + "=" * 60)
    print("因果输入模型（Model 1）测试总结")
    print("=" * 60)
    print(f"测试数据: {args.data}")
    print(f"窗口数量: {len(Xw)}")
    print(f"特征数量: {len(feature_cols)}")
    print("-" * 60)
    print("性能指标:")
    print(f"  MAE:  {metrics['MAE']:.4f}")
    print(f"  RMSE: {metrics['RMSE']:.4f}")
    print(f"  R²:   {metrics['R2']:.4f}")
    print("-" * 60)
    print("误差统计:")
    print(f"  平均误差:     {error_stats['mean_error']:.4f}")
    print(f"  误差标准差:   {error_stats['std_error']:.4f}")
    print(f"  最小误差:     {error_stats['min_error']:.4f}")
    print(f"  最大误差:     {error_stats['max_error']:.4f}")
    print(f"  中位绝对误差: {error_stats['median_abs_error']:.4f}")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
