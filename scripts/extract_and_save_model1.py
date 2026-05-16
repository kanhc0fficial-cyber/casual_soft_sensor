"""
从DML残差训练结果中提取并保存Model 1 (因果输入模型)
"""

import sys
from pathlib import Path
import torch
import pickle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_dml_residual_soft_sensor import *

def main():
    print("=" * 60)
    print("提取并保存因果输入模型 (Model 1)")
    print("=" * 60)
    
    # 加载配置
    config_path = "configs/residual_soft_sensor_test_best.yaml"
    cfg = load_config(config_path)
    
    # 修改输出目录
    output_dir = Path("results/causal_input_model")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "run_log.txt"
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info("因果输入模型 (Model 1) 训练")
    logger.info("=" * 60)
    logger.info(f"配置文件: {config_path}")
    
    set_seed(cfg["random_seed"])
    
    # 加载数据
    df = load_data(cfg, logger)
    df = preprocess(df, cfg, logger)
    
    # 变量角色推断
    roles_df = infer_variable_roles(df, cfg, logger)
    
    target_col = cfg["target_col"]
    time_col = cfg.get("time_col")
    
    a_cols = roles_df[roles_df["role"] == "operation_A"]["variable"].tolist()
    s_cols = roles_df[roles_df["role"] == "state_S"]["variable"].tolist()
    excluded = roles_df[roles_df["role"] == "excluded"]["variable"].tolist()
    
    logger.info(f"操作变量 A ({len(a_cols)}): {a_cols}")
    logger.info(f"状态变量 S ({len(s_cols)}): {s_cols}")
    
    # 数据切分
    skip_cols = {target_col}
    if time_col:
        skip_cols.add(time_col)
    all_feature_cols = [c for c in df.columns if c not in skip_cols and c not in excluded]
    
    model_cols = [c for c in [target_col] + all_feature_cols if c in df.columns]
    if time_col and time_col in df.columns:
        model_cols = [time_col] + model_cols
    df_model = df[[c for c in model_cols if c in df.columns]]
    num_cols_only = df_model.select_dtypes(include=np.number).columns.tolist()
    df_model = df_model[num_cols_only]
    
    # 更新列表
    a_cols = [c for c in a_cols if c in df_model.columns]
    s_cols = [c for c in s_cols if c in df_model.columns]
    
    train_df, val_df, test_df = split_data(df_model, cfg, logger)
    
    # 训练Model 1
    logger.info("=" * 60)
    logger.info("训练因果输入模型 (A + S -> LSTM -> y)")
    logger.info("=" * 60)
    
    model1 = run_model1_causal_input(
        train_df, val_df, test_df,
        a_cols, s_cols, target_col, cfg, logger,
    )
    
    # 保存Model 1的权重
    logger.info("=" * 60)
    logger.info("保存模型权重...")
    
    if "lstm" in model1 and model1["lstm"] is not None:
        lstm_model = model1["lstm"]
        if hasattr(lstm_model, '_model') and lstm_model._model is not None:
            model_path = output_dir / "causal_input_lstm_checkpoint.pt"
            torch.save({
                'model_state_dict': lstm_model._model.state_dict(),
                'input_size': lstm_model.input_size,
                'hidden_size': lstm_model.hidden_size,
                'num_layers': lstm_model.num_layers,
                'dropout': lstm_model.dropout,
            }, model_path)
            logger.info(f"✅ 已保存LSTM权重: {model_path}")
            print(f"✅ 已保存LSTM权重: {model_path}")
    
    # 保存scalers
    if "as_scaler" in model1 and "y_scaler" in model1:
        scalers_path = output_dir / "model_scalers.pkl"
        scalers_dict = {
            "as_scaler": model1["as_scaler"],
            "y_scaler": model1["y_scaler"],
            "a_cols": a_cols,
            "s_cols": s_cols,
            "window_size": cfg["window_size"],
        }
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers_dict, f)
        logger.info(f"✅ 已保存scalers: {scalers_path}")
        print(f"✅ 已保存scalers: {scalers_path}")
    
    # 保存性能指标
    metrics = model1.get("metrics", {})
    metrics_df = pd.DataFrame([{
        "model_name": "causal_input",
        "split": "test",
        "MAE": metrics.get("MAE", float("nan")),
        "RMSE": metrics.get("RMSE", float("nan")),
        "R2": metrics.get("R2", float("nan")),
    }])
    metrics_path = output_dir / "metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info(f"✅ 已保存性能指标: {metrics_path}")
    print(f"✅ 已保存性能指标: {metrics_path}")
    
    # 保存预测结果
    if "predictions_test" in model1:
        pred_path = output_dir / "predictions_test.csv"
        model1["predictions_test"].to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info(f"✅ 已保存预测结果: {pred_path}")
        print(f"✅ 已保存预测结果: {pred_path}")
    
    logger.info("=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)
    
    print("\n" + "=" * 60)
    print("✅ 因果输入模型保存完成！")
    print("=" * 60)
    print(f"输出目录: {output_dir}")
    print(f"性能: MAE={metrics.get('MAE', 0):.4f}, RMSE={metrics.get('RMSE', 0):.4f}, R2={metrics.get('R2', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
