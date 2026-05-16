"""
train_group_branch_optuna.py
=============================
工艺因果组分支软测量模型训练脚本 - 带Optuna超参数优化

用法：
  python scripts/train_group_branch_optuna.py --config configs/group_branch.yaml --trials 20

输出：
  results/group_branch_optuna/
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
from scripts.train_group_branch import *

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def create_optuna_objective(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    groups_cfg: dict,
    model_cfg: dict,
    window_size: int,
    num_features: int,
    cfg: dict,
    logger: logging.Logger,
):
    """创建Optuna优化目标函数"""
    
    def objective(trial: optuna.Trial) -> float:
        """Optuna目标函数：最小化验证集RMSE"""
        
        # 导入必要的库
        from sklearn.preprocessing import StandardScaler
        import torch
        from torch.utils.data import DataLoader, TensorDataset
        
        # 超参数搜索空间
        trial_cfg = cfg.copy()
        trial_cfg['lr'] = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        trial_cfg['batch_size'] = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
        
        # 为每个分组建议hidden_dim
        trial_groups_cfg = {}
        for group_name, group_config in groups_cfg.items():
            trial_groups_cfg[group_name] = group_config.copy()
            trial_groups_cfg[group_name]['hidden_dim'] = trial.suggest_int(
                f'hidden_dim_{group_name}', 16, 64, step=8
            )
        
        # Gate初始化值
        trial_model_cfg = model_cfg.copy()
        trial_model_cfg['gate_init'] = trial.suggest_float('gate_init', 0.3, 0.7)
        
        try:
            # 数据标准化
            feat_scaler = StandardScaler()
            y_scaler = StandardScaler()
            feat_scaler.fit(train_df[feature_cols].values)
            y_scaler.fit(train_df[[target_col]].values)
            
            def scale(split_df):
                X = feat_scaler.transform(split_df[feature_cols].values)
                y = y_scaler.transform(split_df[[target_col]].values).ravel()
                return X, y
            
            X_tr, y_tr = scale(train_df)
            X_vl, y_vl = scale(val_df)
            
            # 滑动窗口
            Xw_tr, yw_tr = make_windows(X_tr, y_tr, window_size)
            Xw_vl, yw_vl = make_windows(X_vl, y_vl, window_size)
            
            # 构建模型
            model = CausalGroupBranchModel(
                groups_cfg=trial_groups_cfg,
                model_cfg=trial_model_cfg,
                window_size=window_size,
                num_features=num_features,
                allow_feature_overlap=bool(cfg.get("allow_feature_overlap", False)),
                warn_unused_features=False,  # 优化时不显示警告
            )
            
            # 训练
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=trial_cfg['lr'])
            
            Xtr_t = torch.tensor(Xw_tr, dtype=torch.float32)
            ytr_t = torch.tensor(yw_tr, dtype=torch.float32)
            loader = DataLoader(
                TensorDataset(Xtr_t, ytr_t),
                batch_size=trial_cfg['batch_size'],
                shuffle=True,
            )
            
            Xvl_t = torch.tensor(Xw_vl, dtype=torch.float32).to(device)
            yvl_t = torch.tensor(yw_vl, dtype=torch.float32).to(device)
            
            best_val_loss = float("inf")
            patience = 5
            no_improve = 0
            max_epochs = 30  # 优化时减少epochs
            
            for epoch in range(1, max_epochs + 1):
                model.train()
                for xb, yb in loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out["y_hat"].squeeze(-1), yb)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                # 验证
                model.eval()
                with torch.no_grad():
                    val_out = model(Xvl_t)
                    val_loss = criterion(val_out["y_hat"].squeeze(-1), yvl_t).item()
                
                # Optuna剪枝
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        break
            
            return best_val_loss
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return float('inf')
    
    return objective


def main():
    parser = argparse.ArgumentParser(
        description="工艺因果组分支软测量模型训练 - Optuna优化"
    )
    parser.add_argument(
        "--config",
        default="configs/group_branch.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=20,
        help="Optuna优化尝试次数（默认20）",
    )
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"] + "_optuna")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_path = output_dir / "run_log.txt"
    logger = setup_logger(log_path)
    
    logger.info("=" * 60)
    logger.info("工艺因果组分支软测量 - Optuna优化")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"优化尝试次数: {args.trials}")
    
    set_seed(cfg["random_seed"])
    
    # 加载和预处理数据
    df = load_data(cfg, logger)
    df = preprocess(df, cfg, logger)
    
    # 确定特征列
    target_col = cfg["target_col"]
    time_col = cfg.get("time_col")
    exclude_cols = set(cfg.get("exclude_cols", []))
    if time_col:
        exclude_cols.add(time_col)
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col and c not in exclude_cols]
    
    if not feature_cols:
        logger.error("特征列为空，退出。")
        sys.exit(1)
    
    num_features = len(feature_cols)
    logger.info(f"特征列数量: {num_features}")
    
    # 获取配置
    groups_cfg = cfg.get("groups", {})
    model_cfg = cfg.get("model", {})
    window_size = cfg["window_size"]
    
    # 数据切分
    train_df, val_df, test_df = split_data(df, cfg, logger)
    
    # 创建Optuna study
    logger.info("=" * 60)
    logger.info("开始Optuna超参数优化...")
    
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=cfg["random_seed"]),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )
    
    objective_func = create_optuna_objective(
        train_df, val_df, test_df,
        feature_cols, target_col,
        groups_cfg, model_cfg,
        window_size, num_features,
        cfg, logger
    )
    
    study.optimize(objective_func, n_trials=args.trials, show_progress_bar=True)
    
    # 保存最佳参数
    logger.info("=" * 60)
    logger.info("优化完成！")
    logger.info(f"最佳验证损失: {study.best_value:.6f}")
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
    cfg['lr'] = study.best_params['lr']
    cfg['batch_size'] = study.best_params['batch_size']
    
    for group_name in groups_cfg.keys():
        groups_cfg[group_name]['hidden_dim'] = study.best_params[f'hidden_dim_{group_name}']
    
    model_cfg['gate_init'] = study.best_params['gate_init']
    
    # 标准化
    from sklearn.preprocessing import StandardScaler
    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()
    feat_scaler.fit(train_df[feature_cols].values)
    y_scaler.fit(train_df[[target_col]].values)
    
    def scale(split_df):
        X = feat_scaler.transform(split_df[feature_cols].values)
        y = y_scaler.transform(split_df[[target_col]].values).ravel()
        return X, y
    
    X_tr, y_tr = scale(train_df)
    X_vl, y_vl = scale(val_df)
    X_te, y_te = scale(test_df)
    
    Xw_tr, yw_tr = make_windows(X_tr, y_tr, window_size)
    Xw_vl, yw_vl = make_windows(X_vl, y_vl, window_size)
    Xw_te, yw_te = make_windows(X_te, y_te, window_size)
    
    # 构建最终模型
    model = CausalGroupBranchModel(
        groups_cfg=groups_cfg,
        model_cfg=model_cfg,
        window_size=window_size,
        num_features=num_features,
        allow_feature_overlap=bool(cfg.get("allow_feature_overlap", False)),
        warn_unused_features=bool(cfg.get("warn_unused_features", True)),
    )
    
    # 训练最终模型（更多epochs）
    cfg['epochs'] = 50
    cfg['patience'] = 8
    model = train_model(model, Xw_tr, yw_tr, Xw_vl, yw_vl, cfg, logger)
    
    # 预测和保存
    y_pred_scaled, branch_outputs_scaled, gates = predict_with_info(model, Xw_te)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()
    
    save_results(output_dir, model, y_true, y_pred, branch_outputs_scaled, gates, y_scaler, logger)
    
    logger.info("=" * 60)
    logger.info("完成！")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
