"""
test_all_models_on_multiregime.py
==================================
在多个测试集上测试所有4个模型（Model 0-3）

用法：
  python scripts/test_all_models_on_multiregime.py \\
    --model_dir results/residual_soft_sensor \\
    --data_dir "C:\\Users\\goldenwhale\\Downloads\\my_mining_simulation\\output\\multiregime_splits_noclip" \\
    --output results/multiregime_test_results

模型文件要求：
  Model 0: baseline_all_lstm_checkpoint.pt + baseline_all_lstm_scalers.pkl
  Model 1: as_lstm_checkpoint.pt + as_lstm_scalers.pkl
  Model 2: dml_effect_weight_lstm_checkpoint.pt + dml_effect_weight_lstm_scalers.pkl
  Model 3: dml_residual_lstm_checkpoint.pt + dml_residual_model_components.pkl
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_dml_residual_soft_sensor import LSTMRegressor


def setup_logger(log_path: Path) -> logging.Logger:
    """设置日志记录器"""
    logger = logging.getLogger("test_all_models")
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
    """计算评估指标"""
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"MAE": mae, "RMSE": rmse, "R2": r2}


def make_windows(X: np.ndarray, y: np.ndarray, window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """构造滑动窗口"""
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


def load_lstm_model(checkpoint_path: Path, logger: logging.Logger) -> LSTMRegressor:
    """加载LSTM模型"""
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    input_size = checkpoint['input_size']
    hidden_size = checkpoint['hidden_size']
    num_layers = checkpoint['num_layers']
    dropout = checkpoint['dropout']
    
    logger.info(f"    input_size: {input_size}, hidden_size: {hidden_size}, "
                f"num_layers: {num_layers}, dropout: {dropout:.4f}")
    
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )
    model._model, model._device = model._build()
    model._model.load_state_dict(checkpoint['model_state_dict'])
    model._model.eval()
    
    return model


def test_model_0_1_2(
    model_name: str,
    checkpoint_path: Path,
    scalers_path: Path,
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    logger: logging.Logger,
) -> Dict:
    """测试Model 0/1/2（简单LSTM模型）"""
    logger.info(f"  加载{model_name}...")
    
    # 加载模型
    model = load_lstm_model(checkpoint_path, logger)
    
    # 加载scalers
    with open(scalers_path, 'rb') as f:
        scalers = pickle.load(f)
    
    as_scaler = scalers['as_scaler']
    y_scaler = scalers['y_scaler']
    feature_cols = scalers['feature_cols']
    
    logger.info(f"    特征数量: {len(feature_cols)}")
    
    # 检查特征列
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        logger.error(f"    ❌ 缺少特征列 ({len(missing_cols)}): {missing_cols[:10]}...")
        return None
    
    # 准备数据
    X = as_scaler.transform(df[feature_cols].values)
    y = df[target_col].values
    y_scaled = y_scaler.transform(y.reshape(-1, 1)).ravel()
    
    # 滑动窗口
    Xw, yw = make_windows(X, y_scaled, window_size)
    logger.info(f"    窗口数量: {len(Xw)}")
    
    # 预测
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model._device = device
    model._model = model._model.to(device)
    
    Xw_t = torch.tensor(Xw, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_pred_scaled = model._model(Xw_t).cpu().numpy()
    
    # 反标准化
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw.reshape(-1, 1)).ravel()
    
    # 计算指标
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"    MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
    
    return {
        "model_name": model_name,
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "feature_cols": feature_cols,
    }


def test_model_3(
    checkpoint_path: Path,
    components_path: Path,
    df: pd.DataFrame,
    target_col: str,
    window_size: int,
    model_dir: Path,
    logger: logging.Logger,
) -> Dict:
    """测试Model 3（DML残差模型）"""
    logger.info(f"  加载Model 3 (DML残差)...")
    
    # 加载LSTM模型
    model = load_lstm_model(checkpoint_path, logger)
    
    # 加载组件（scalers, g_model, q_models等）
    with open(components_path, 'rb') as f:
        components = pickle.load(f)
    
    y_res_scaler = components['y_res_scaler']
    as_scaler = components['as_scaler']
    c_scaler = components['c_scaler']
    g_model = components['g_model']
    q_models = components['q_models']
    feature_cols = components['feature_cols']  # 这是残差化后的A+S列
    
    logger.info(f"    残差特征数量: {len(feature_cols)}")
    
    # 从variable_roles.csv读取C列名称
    roles_path = model_dir / "variable_roles.csv"
    if not roles_path.exists():
        logger.error(f"    ❌ 缺少variable_roles.csv文件")
        return None
    
    roles_df = pd.read_csv(roles_path)
    c_cols = roles_df[roles_df['role'] == 'confounder_C']['variable'].tolist()
    
    logger.info(f"    C变量数量: {len(c_cols)}")
    
    if len(c_cols) == 0:
        logger.error(f"    ❌ C列为空，无法进行DML残差预测")
        return None
    
    # 检查列是否存在
    missing_cols = [c for c in feature_cols if c not in df.columns]
    missing_c = [c for c in c_cols if c not in df.columns]
    
    if missing_cols:
        logger.error(f"    ❌ 缺少残差特征列 ({len(missing_cols)}): {missing_cols[:10]}...")
        return None
    
    if missing_c:
        logger.error(f"    ❌ 缺少C列 ({len(missing_c)}): {missing_c[:10]}...")
        return None
    
    # 步骤1: 计算y_base = g(C)
    C = df[c_cols].values
    C_scaled = c_scaler.transform(C)
    y_base = g_model.predict(C_scaled)
    
    # 步骤2: 残差化A和S
    X_residual = np.zeros((len(df), len(feature_cols)), dtype=np.float32)
    for j, col in enumerate(feature_cols):
        if col in q_models:
            X_j = df[col].values
            X_j_pred = q_models[col].predict(C_scaled)
            X_residual[:, j] = X_j - X_j_pred
        else:
            # 如果没有q模型，直接使用原始值
            X_residual[:, j] = df[col].values
    
    # 步骤3: 标准化残差特征
    X_residual_scaled = as_scaler.transform(X_residual)
    
    # 步骤4: 计算y_res_true
    y = df[target_col].values
    y_res_true = y - y_base
    y_res_scaled = y_res_scaler.transform(y_res_true.reshape(-1, 1)).ravel()
    
    # 步骤5: 滑动窗口
    Xw, yw = make_windows(X_residual_scaled, y_res_scaled, window_size)
    logger.info(f"    窗口数量: {len(Xw)}")
    
    # 步骤6: 预测y_res
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model._device = device
    model._model = model._model.to(device)
    
    Xw_t = torch.tensor(Xw, dtype=torch.float32).to(device)
    
    with torch.no_grad():
        y_res_pred_scaled = model._model(Xw_t).cpu().numpy()
    
    # 反标准化
    y_res_pred = y_res_scaler.inverse_transform(y_res_pred_scaled.reshape(-1, 1)).ravel()
    
    # 步骤7: 对齐y_base（去掉前window_size-1个点）
    align_offset = window_size - 1
    y_base_aligned = y_base[align_offset: align_offset + len(y_res_pred)]
    y_true_aligned = y[align_offset: align_offset + len(y_res_pred)]
    
    # 步骤8: 最终预测 y_hat = y_base + y_res_pred
    y_pred = y_base_aligned + y_res_pred
    
    # 计算指标
    metrics = compute_metrics(y_true_aligned, y_pred)
    logger.info(f"    MAE: {metrics['MAE']:.4f}, RMSE: {metrics['RMSE']:.4f}, R²: {metrics['R2']:.4f}")
    
    return {
        "model_name": "dml_residual_lstm",
        "metrics": metrics,
        "y_true": y_true_aligned,
        "y_pred": y_pred,
        "y_base": y_base_aligned,
        "y_res_pred": y_res_pred,
        "feature_cols": feature_cols,
    }


def save_results(
    output_dir: Path,
    dataset_name: str,
    results: Dict[str, Dict],
    logger: logging.Logger,
) -> None:
    """保存测试结果"""
    dataset_dir = output_dir / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存汇总指标
    metrics_rows = []
    for model_name, result in results.items():
        if result is None:
            continue
        metrics = result['metrics']
        metrics_rows.append({
            "model_name": model_name,
            "dataset": dataset_name,
            "MAE": metrics['MAE'],
            "RMSE": metrics['RMSE'],
            "R2": metrics['R2'],
        })
    
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = dataset_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info(f"    ✅ 已保存: {metrics_path}")
    
    # 2. 保存每个模型的预测结果
    for model_name, result in results.items():
        if result is None:
            continue
        
        pred_data = {
            "y_true": result['y_true'],
            "y_pred": result['y_pred'],
            "error": result['y_true'] - result['y_pred'],
            "abs_error": np.abs(result['y_true'] - result['y_pred']),
        }
        
        # Model 3 额外保存y_base和y_res
        if model_name == "dml_residual_lstm" and 'y_base' in result:
            pred_data['y_base'] = result['y_base']
            pred_data['y_res_pred'] = result['y_res_pred']
        
        pred_df = pd.DataFrame(pred_data)
        pred_path = dataset_dir / f"{model_name}_predictions.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"    ✅ 已保存: {pred_path}")
    
    # 3. 保存误差统计
    error_stats_rows = []
    for model_name, result in results.items():
        if result is None:
            continue
        
        errors = result['y_true'] - result['y_pred']
        error_stats_rows.append({
            "model_name": model_name,
            "dataset": dataset_name,
            "mean_error": float(np.mean(errors)),
            "std_error": float(np.std(errors)),
            "min_error": float(np.min(errors)),
            "max_error": float(np.max(errors)),
            "median_abs_error": float(np.median(np.abs(errors))),
        })
    
    error_stats_df = pd.DataFrame(error_stats_rows)
    error_stats_path = dataset_dir / "error_statistics.csv"
    error_stats_df.to_csv(error_stats_path, index=False, encoding="utf-8-sig")
    logger.info(f"    ✅ 已保存: {error_stats_path}")


def main():
    parser = argparse.ArgumentParser(description="在多个测试集上测试所有4个模型")
    parser.add_argument("--model_dir", required=True, help="模型文件目录")
    parser.add_argument("--data_dir", required=True, help="测试数据目录（包含6个parquet文件）")
    parser.add_argument("--output", default="results/multiregime_test_results", help="输出目录")
    parser.add_argument("--target", default="y_fx_xin1", help="目标变量列名")
    parser.add_argument("--time_col", default="t", help="时间列名")
    parser.add_argument("--window_size", type=int, default=12, help="滑动窗口大小")
    
    args = parser.parse_args()
    
    model_dir = Path(args.model_dir)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output)
    
    # 检查目录
    if not model_dir.exists():
        print(f"错误: 模型目录不存在: {model_dir}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"错误: 数据目录不存在: {data_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "test_log.txt"
    logger = setup_logger(log_path)
    
    logger.info("=" * 80)
    logger.info("在多个测试集上测试所有4个模型")
    logger.info("=" * 80)
    logger.info(f"模型目录: {model_dir}")
    logger.info(f"数据目录: {data_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"目标列: {args.target}")
    logger.info(f"窗口大小: {args.window_size}")
    
    # 定义模型配置
    model_configs = {
        "baseline_all_lstm": {
            "checkpoint": model_dir / "baseline_all_lstm_lstm_checkpoint.pt",
            "scalers": model_dir / "baseline_all_lstm_scalers.pkl",
            "type": "simple",
        },
        "as_lstm": {
            "checkpoint": model_dir / "as_lstm_lstm_checkpoint.pt",
            "scalers": model_dir / "as_lstm_scalers.pkl",
            "type": "simple",
        },
        "dml_effect_weight_lstm": {
            "checkpoint": model_dir / "dml_effect_weight_lstm_lstm_checkpoint.pt",
            "scalers": model_dir / "dml_effect_weight_lstm_scalers.pkl",
            "type": "simple",
        },
        "dml_residual_lstm": {
            "checkpoint": model_dir / "dml_residual_lstm_checkpoint.pt",
            "components": model_dir / "dml_residual_model_components.pkl",
            "type": "residual",
        },
    }
    
    # 检查模型文件是否存在
    logger.info("-" * 80)
    logger.info("检查模型文件...")
    missing_files = []
    for model_name, config in model_configs.items():
        if config['type'] == 'simple':
            if not config['checkpoint'].exists():
                missing_files.append(str(config['checkpoint']))
            if not config['scalers'].exists():
                missing_files.append(str(config['scalers']))
        else:  # residual
            if not config['checkpoint'].exists():
                missing_files.append(str(config['checkpoint']))
            if not config['components'].exists():
                missing_files.append(str(config['components']))
    
    if missing_files:
        logger.error("❌ 缺少以下模型文件:")
        for f in missing_files:
            logger.error(f"  - {f}")
        sys.exit(1)
    
    logger.info("✅ 所有模型文件都存在")
    
    # 查找所有测试数据集
    logger.info("-" * 80)
    logger.info("查找测试数据集...")
    data_files = sorted(data_dir.glob("*.parquet"))
    
    if not data_files:
        logger.error(f"❌ 在 {data_dir} 中未找到任何 .parquet 文件")
        sys.exit(1)
    
    logger.info(f"找到 {len(data_files)} 个测试数据集:")
    for f in data_files:
        logger.info(f"  - {f.name}")
    
    # 对每个数据集进行测试
    all_results = []
    
    for data_file in data_files:
        dataset_name = data_file.stem
        logger.info("=" * 80)
        logger.info(f"测试数据集: {dataset_name}")
        logger.info("=" * 80)
        
        # 加载数据
        logger.info(f"加载数据: {data_file}")
        df = pd.read_parquet(data_file)
        logger.info(f"  数据形状: {df.shape}")
        
        # 排序
        if args.time_col and args.time_col in df.columns:
            df = df.sort_values(args.time_col).reset_index(drop=True)
            logger.info(f"  已按时间列 '{args.time_col}' 排序")
        
        # 检查目标列
        if args.target not in df.columns:
            logger.error(f"  ❌ 缺少目标列: {args.target}")
            continue
        
        # 测试每个模型
        results = {}
        
        # Model 0, 1, 2
        for model_name in ["baseline_all_lstm", "as_lstm", "dml_effect_weight_lstm"]:
            logger.info(f"-" * 40)
            logger.info(f"测试 {model_name}...")
            try:
                config = model_configs[model_name]
                result = test_model_0_1_2(
                    model_name,
                    config['checkpoint'],
                    config['scalers'],
                    df,
                    args.target,
                    args.window_size,
                    logger,
                )
                results[model_name] = result
            except Exception as e:
                logger.error(f"  ❌ 测试失败: {e}")
                results[model_name] = None
        
        # Model 3
        logger.info(f"-" * 40)
        logger.info(f"测试 dml_residual_lstm...")
        try:
            config = model_configs["dml_residual_lstm"]
            result = test_model_3(
                config['checkpoint'],
                config['components'],
                df,
                args.target,
                args.window_size,
                model_dir,
                logger,
            )
            results["dml_residual_lstm"] = result
        except Exception as e:
            logger.error(f"  ❌ 测试失败: {e}")
            results["dml_residual_lstm"] = None
        
        # 保存结果
        logger.info(f"-" * 40)
        logger.info(f"保存结果...")
        save_results(output_dir, dataset_name, results, logger)
        
        # 收集汇总数据
        for model_name, result in results.items():
            if result is not None:
                all_results.append({
                    "dataset": dataset_name,
                    "model_name": model_name,
                    **result['metrics']
                })
    
    # 保存总汇总
    logger.info("=" * 80)
    logger.info("保存总汇总...")
    all_results_df = pd.DataFrame(all_results)
    all_results_path = output_dir / "all_results_summary.csv"
    all_results_df.to_csv(all_results_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 已保存: {all_results_path}")
    
    # 打印总结
    logger.info("=" * 80)
    logger.info("测试完成！")
    logger.info("=" * 80)
    logger.info(f"测试了 {len(data_files)} 个数据集")
    logger.info(f"测试了 4 个模型")
    logger.info(f"输出目录: {output_dir}")
    logger.info("=" * 80)
    
    # 打印性能汇总表
    if not all_results_df.empty:
        logger.info("\n性能汇总:")
        pivot_mae = all_results_df.pivot(index='dataset', columns='model_name', values='MAE')
        pivot_r2 = all_results_df.pivot(index='dataset', columns='model_name', values='R2')
        
        logger.info("\nMAE:")
        logger.info(pivot_mae.to_string())
        
        logger.info("\nR²:")
        logger.info(pivot_r2.to_string())


if __name__ == "__main__":
    main()
