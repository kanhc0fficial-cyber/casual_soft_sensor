"""
从已完成的Optuna训练中重新保存模型权重

由于模型权重在训练时没有保存，我们需要使用最佳超参数重新训练并保存
"""

import json
import sys
from pathlib import Path
import torch
import pickle
import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train_group_branch import *
from scripts.train_dml_residual_soft_sensor import *


def save_group_branch_model():
    """重新训练并保存门控分组模型"""
    print("=" * 60)
    print("重新训练并保存门控分组模型")
    print("=" * 60)
    
    # 加载最佳超参数
    best_params_path = Path("results/group_branch_test_optuna/best_params.json")
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    print("最佳超参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 加载配置
    cfg = load_config("configs/group_branch_test.yaml")
    
    # 更新配置为最佳超参数
    cfg['lr'] = best_params['lr']
    cfg['batch_size'] = best_params['batch_size']
    
    groups_cfg = cfg.get("groups", {})
    for group_name in groups_cfg.keys():
        groups_cfg[group_name]['hidden_dim'] = best_params[f'hidden_dim_{group_name}']
    
    model_cfg = cfg.get("model", {})
    model_cfg['gate_init'] = best_params['gate_init']
    
    # 加载数据
    df = load_data(cfg, None)
    df = preprocess(df, cfg, None)
    
    # 确定特征列
    target_col = cfg["target_col"]
    time_col = cfg.get("time_col")
    exclude_cols = set(cfg.get("exclude_cols", []))
    if time_col:
        exclude_cols.add(time_col)
    
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [c for c in num_cols if c != target_col and c not in exclude_cols]
    
    num_features = len(feature_cols)
    window_size = cfg["window_size"]
    
    # 数据切分
    train_df, val_df, test_df = split_data(df, cfg, None)
    
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
    
    # 构建模型
    model = CausalGroupBranchModel(
        groups_cfg=groups_cfg,
        model_cfg=model_cfg,
        window_size=window_size,
        num_features=num_features,
        allow_feature_overlap=bool(cfg.get("allow_feature_overlap", False)),
        warn_unused_features=False,
    )
    
    # 训练
    cfg['epochs'] = 50
    cfg['patience'] = 8
    print("\n开始训练...")
    model = train_model(model, Xw_tr, yw_tr, Xw_vl, yw_vl, cfg, None)
    
    # 保存模型权重
    output_dir = Path("results/group_branch_test_optuna")
    model_path = output_dir / "model_checkpoint.pt"
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'y_scaler_mean': y_scaler.mean_,
        'y_scaler_scale': y_scaler.scale_,
        'feat_scaler_mean': feat_scaler.mean_,
        'feat_scaler_scale': feat_scaler.scale_,
        'feature_cols': feature_cols,
        'groups_cfg': groups_cfg,
        'model_cfg': model_cfg,
        'window_size': window_size,
        'num_features': num_features,
    }, model_path)
    
    print(f"\n✅ 模型权重已保存: {model_path}")
    print(f"文件大小: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
    
    return model, y_scaler, feat_scaler


def save_dml_residual_models():
    """重新训练并保存DML残差模型"""
    print("\n" + "=" * 60)
    print("重新训练并保存DML残差模型")
    print("=" * 60)
    
    # 加载最佳超参数
    best_params_path = Path("results/residual_soft_sensor_test_optuna/best_params.json")
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)
    
    print("最佳超参数:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    
    # 加载配置
    cfg = load_config("configs/residual_soft_sensor_test.yaml")
    
    # 更新配置为最佳超参数
    cfg['lstm_hidden_size'] = best_params['lstm_hidden_size']
    cfg['lstm_num_layers'] = best_params['lstm_num_layers']
    cfg['lstm_dropout'] = best_params['lstm_dropout']
    cfg['lstm_lr'] = best_params['lstm_lr']
    cfg['lstm_batch_size'] = best_params['lstm_batch_size']
    cfg['lstm_epochs'] = 50
    cfg['lstm_patience'] = 8
    
    # 加载数据
    df = load_data(cfg, None)
    df = preprocess(df, cfg, None)
    
    # 变量角色推断
    roles_df = infer_variable_roles(df, cfg, None)
    
    target_col = cfg["target_col"]
    time_col = cfg.get("time_col")
    
    a_cols = roles_df[roles_df["role"] == "operation_A"]["variable"].tolist()
    c_cols = roles_df[roles_df["role"] == "confounder_C"]["variable"].tolist()
    s_cols = roles_df[roles_df["role"] == "state_S"]["variable"].tolist()
    excluded = roles_df[roles_df["role"] == "excluded"]["variable"].tolist()
    
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
    
    train_df, val_df, test_df = split_data(df_model, cfg, None)
    
    # 运行三个模型
    print("\n训练Model 0 (baseline)...")
    model0 = run_model0_baseline(train_df, val_df, test_df, all_feature_cols, target_col, cfg, None)
    
    print("\n训练Model 1 (causal_input)...")
    model1 = run_model1_causal_input(train_df, val_df, test_df, a_cols, s_cols, target_col, cfg, None)
    
    print("\n训练Model 2 (dml_residual)...")
    output_dir = Path("results/residual_soft_sensor_test_optuna")
    model2 = run_model2_dml_residual(train_df, val_df, test_df, c_cols, a_cols, s_cols, target_col, cfg, None, output_dir)
    
    # 保存模型权重
    print("\n保存模型权重...")
    
    # 保存Model 0
    if "lstm" in model0 and model0["lstm"] is not None:
        lstm_model = model0["lstm"]
        if hasattr(lstm_model, '_model') and lstm_model._model is not None:
            model_path = output_dir / "baseline_lstm_checkpoint.pt"
            torch.save({
                'model_state_dict': lstm_model._model.state_dict(),
                'input_size': lstm_model.input_size,
                'hidden_size': lstm_model.hidden_size,
                'num_layers': lstm_model.num_layers,
                'dropout': lstm_model.dropout,
            }, model_path)
            print(f"✅ 已保存baseline LSTM: {model_path}")
    
    # 保存Model 1
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
            print(f"✅ 已保存causal_input LSTM: {model_path}")
    
    # 保存Model 2
    if "residual_lstm" in model2 and model2["residual_lstm"] is not None:
        lstm_model = model2["residual_lstm"]
        if hasattr(lstm_model, '_model') and lstm_model._model is not None:
            model_path = output_dir / "dml_residual_lstm_checkpoint.pt"
            torch.save({
                'model_state_dict': lstm_model._model.state_dict(),
                'input_size': lstm_model.input_size,
                'hidden_size': lstm_model.hidden_size,
                'num_layers': lstm_model.num_layers,
                'dropout': lstm_model.dropout,
            }, model_path)
            print(f"✅ 已保存dml_residual LSTM: {model_path}")
    
    # 保存scalers
    scalers_path = output_dir / "model_scalers.pkl"
    scalers_dict = {}
    
    if "y_res_scaler" in model2:
        scalers_dict["y_res_scaler"] = model2["y_res_scaler"]
    if "as_scaler" in model2:
        scalers_dict["as_scaler"] = model2["as_scaler"]
    if "c_scaler" in model2:
        scalers_dict["c_scaler"] = model2["c_scaler"]
    if "g_model" in model2:
        scalers_dict["g_model"] = model2["g_model"]
    if "q_models" in model2:
        scalers_dict["q_models"] = model2["q_models"]
    
    if scalers_dict:
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers_dict, f)
        print(f"✅ 已保存scalers: {scalers_path}")
    
    print("\n所有DML模型权重保存完成！")


if __name__ == "__main__":
    print("开始重新训练并保存模型权重...")
    print("这将使用Optuna找到的最佳超参数重新训练模型\n")
    
    # 保存门控分组模型
    save_group_branch_model()
    
    # 保存DML残差模型
    save_dml_residual_models()
    
    print("\n" + "=" * 60)
    print("✅ 所有模型权重保存完成！")
    print("=" * 60)
