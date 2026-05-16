"""
train_dml_residual_optuna.py
=============================
DML 正交残差软测量训练脚本 - 带Optuna超参数优化

用法：
  python scripts/train_dml_residual_optuna.py --config configs/residual_soft_sensor.yaml --trials 15

输出：
  results/residual_soft_sensor_optuna/
    - best_params.json
    - optimization_history.csv
    - [标准输出文件]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# 导入原始训练脚本的所有功能
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.train_dml_residual_soft_sensor import *

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# 确保导入必要的库
from sklearn.preprocessing import StandardScaler
import numpy as np


def create_optuna_objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    c_cols: List[str],
    a_cols: List[str],
    s_cols: List[str],
    target_col: str,
    cfg: dict,
    logger: logging.Logger,
):
    """创建Optuna优化目标函数"""
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna目标函数：最小化验证集RMSE"""
        
        # 超参数搜索空间
        trial_cfg = cfg.copy()
        trial_cfg['lstm_hidden_size'] = trial.suggest_categorical('lstm_hidden_size', [32, 64, 128])
        trial_cfg['lstm_num_layers'] = trial.suggest_int('lstm_num_layers', 1, 3)
        trial_cfg['lstm_dropout'] = trial.suggest_float('lstm_dropout', 0.0, 0.3)
        trial_cfg['lstm_lr'] = trial.suggest_float('lstm_lr', 1e-4, 1e-2, log=True)
        trial_cfg['lstm_batch_size'] = trial.suggest_categorical('lstm_batch_size', [64, 128, 256])
        
        residual_as_cols = a_cols + s_cols
        
        if not c_cols or not residual_as_cols:
            return float('inf')
        
        try:
            seed = trial_cfg["random_seed"]
            model_type = trial_cfg.get("residualization_model", "lightgbm")
            window_size = trial_cfg["window_size"]
            
            # C的标准化
            from sklearn.preprocessing import StandardScaler
            c_scaler = StandardScaler()
            C_train = c_scaler.fit_transform(train_df[c_cols].values)
            C_val = c_scaler.transform(val_df[c_cols].values)
            
            y_train_raw = train_df[target_col].values
            y_val_raw = val_df[target_col].values
            
            # g_model: C -> y
            g_model = build_residualization_model(model_type, seed)
            g_model.fit(C_train, y_train_raw)
            
            y_base_train = g_model.predict(C_train)
            y_base_val = g_model.predict(C_val)
            
            y_res_train = y_train_raw - y_base_train
            y_res_val = y_val_raw - y_base_val
            
            # q_j: C -> X_j
            as_scaler = StandardScaler()
            as_scaler.fit(train_df[residual_as_cols].values)
            AS_train_scaled = as_scaler.transform(train_df[residual_as_cols].values)
            AS_val_scaled = as_scaler.transform(val_df[residual_as_cols].values)
            
            AS_res_train = np.empty_like(AS_train_scaled)
            AS_res_val = np.empty_like(AS_val_scaled)
            
            for j, col in enumerate(residual_as_cols):
                qm = build_residualization_model(model_type, seed + j + 1)
                xj_train = AS_train_scaled[:, j]
                xj_val = AS_val_scaled[:, j]
                
                qm.fit(C_train, xj_train)
                
                xj_hat_train = qm.predict(C_train)
                xj_hat_val = qm.predict(C_val)
                
                AS_res_train[:, j] = xj_train - xj_hat_train
                AS_res_val[:, j] = xj_val - xj_hat_val
            
            # 残差化y的标准化
            y_res_scaler = StandardScaler()
            y_res_train_scaled = y_res_scaler.fit_transform(y_res_train.reshape(-1, 1)).ravel()
            y_res_val_scaled = y_res_scaler.transform(y_res_val.reshape(-1, 1)).ravel()
            
            # 滑动窗口
            Xw_tr, yw_tr = make_windows(AS_res_train, y_res_train_scaled, window_size)
            Xw_vl, yw_vl = make_windows(AS_res_val, y_res_val_scaled, window_size)
            
            # 训练残差LSTM
            residual_lstm = LSTMRegressor(
                input_size=len(residual_as_cols),
                hidden_size=trial_cfg["lstm_hidden_size"],
                num_layers=trial_cfg["lstm_num_layers"],
                dropout=trial_cfg["lstm_dropout"],
                epochs=20,  # 优化时减少epochs
                batch_size=trial_cfg["lstm_batch_size"],
                lr=trial_cfg["lstm_lr"],
                patience=5,
                seed=seed,
            )
            residual_lstm.fit(Xw_tr, yw_tr, Xw_vl, yw_vl, logger=None)
            
            # 验证集预测
            y_res_pred_scaled = residual_lstm.predict(Xw_vl)
            y_res_pred = y_res_scaler.inverse_transform(y_res_pred_scaled.reshape(-1, 1)).ravel()
            
            # 对齐
            align_offset = window_size - 1
            y_base_aligned = y_base_val[align_offset: align_offset + len(y_res_pred)]
            y_true_aligned = y_val_raw[align_offset: align_offset + len(y_res_pred)]
            
            y_hat = y_base_aligned + y_res_pred
            
            # 计算RMSE
            rmse = np.sqrt(np.mean((y_true_aligned - y_hat) ** 2))
            
            return rmse
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    return objective


def main():
    parser = argparse.ArgumentParser(
        description="DML 正交残差软测量 - Optuna优化"
    )
    parser.add_argument(
        "--config",
        default="configs/residual_soft_sensor.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=15,
        help="Optuna优化尝试次数（默认15）",
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"] + "_optuna")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "run_log.txt"
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info("DML 正交残差软测量 - Optuna优化")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"优化尝试次数: {args.trials}")
    
    set_seed(cfg["random_seed"])
    
    # 加载和预处理数据
    df = load_data(cfg, logger)
    df = preprocess(df, cfg, logger)
    
    # 变量角色推断
    logger.info("-" * 40)
    logger.info("推断变量角色...")
    roles_df = infer_variable_roles(df, cfg, logger)
    
    target_col = cfg["target_col"]
    time_col = cfg.get("time_col")
    
    a_cols = roles_df[roles_df["role"] == "operation_A"]["variable"].tolist()
    c_cols = roles_df[roles_df["role"] == "confounder_C"]["variable"].tolist()
    s_cols = roles_df[roles_df["role"] == "state_S"]["variable"].tolist()
    excluded = roles_df[roles_df["role"] == "excluded"]["variable"].tolist()
    
    logger.info(f"操作变量 A ({len(a_cols)}): {a_cols}")
    logger.info(f"工况变量 C ({len(c_cols)}): {c_cols}")
    logger.info(f"状态变量 S ({len(s_cols)}): {s_cols}")
    
    if not c_cols:
        logger.error("C candidates insufficient (n=0); 无法进行DML优化")
        sys.exit(1)
    
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
    c_cols = [c for c in c_cols if c in df_model.columns]
    s_cols = [c for c in s_cols if c in df_model.columns]
    
    train_df, val_df, test_df = split_data(df_model, cfg, logger)
    
    # 创建Optuna study
    logger.info("=" * 60)
    logger.info("开始Optuna超参数优化...")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg["random_seed"]),
        pruner=MedianPruner(n_startup_trials=3, n_warmup_steps=5),
    )
    
    objective_func = create_optuna_objective(
        train_df, val_df,
        c_cols, a_cols, s_cols,
        target_col, cfg, logger
    )
    
    study.optimize(objective_func, n_trials=args.trials, show_progress_bar=True)
    
    # 保存最佳参数
    logger.info("=" * 60)
    logger.info("优化完成！")
    logger.info(f"最佳验证RMSE: {study.best_value:.6f}")
    logger.info(f"最佳参数:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    
    best_params_path = output_dir / "best_params.json"
    with open(best_params_path, 'w') as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"已保存最佳参数: {best_params_path}")
    
    # 保存优化历史
    history_df = study.trials_dataframe()
    history_path = output_dir / "optimization_history.csv"
    history_df.to_csv(history_path, index=False)
    logger.info(f"已保存优化历史: {history_path}")
    
    # 使用最佳参数重新训练完整模型
    logger.info("=" * 60)
    logger.info("使用最佳参数训练最终模型...")
    
    # 更新配置
    cfg['lstm_hidden_size'] = study.best_params['lstm_hidden_size']
    cfg['lstm_num_layers'] = study.best_params['lstm_num_layers']
    cfg['lstm_dropout'] = study.best_params['lstm_dropout']
    cfg['lstm_lr'] = study.best_params['lstm_lr']
    cfg['lstm_batch_size'] = study.best_params['lstm_batch_size']
    cfg['lstm_epochs'] = 50  # 最终训练使用更多epochs
    cfg['lstm_patience'] = 8
    
    # 运行完整的三模型训练
    model0 = run_model0_baseline(
        train_df, val_df, test_df,
        all_feature_cols, target_col, cfg, logger,
    )
    
    model1 = run_model1_causal_input(
        train_df, val_df, test_df,
        a_cols, s_cols, target_col, cfg, logger,
    )
    
    model2 = run_model2_dml_residual(
        train_df, val_df, test_df,
        c_cols, a_cols, s_cols, target_col, cfg, logger, output_dir,
    )
    
    save_outputs(output_dir, roles_df, model0, model1, model2, logger)
    
    logger.info("=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
