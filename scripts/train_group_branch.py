"""
train_group_branch.py
=====================
工艺因果组分支软测量模型训练脚本。

用法：
  python scripts/train_group_branch.py --config configs/group_branch.yaml

输出（results/group_branch/）：
  group_branch_metrics.csv       - MAE / RMSE / R2
  group_branch_gates.csv         - 每个 group 的 gate 值
  group_branch_contributions.csv - 每个 group 的平均分支输出和平均绝对贡献
  predictions_test.csv           - 测试集逐条预测结果
  run_log.txt                    - 训练日志
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ─── 将 src 目录加入搜索路径 ──────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.group_branch import CausalGroupBranchModel  # noqa: E402


# ─── 随机种子 ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── 日志 ─────────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("group_branch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─── 配置加载 ─────────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    defaults = {
        "data_path": "data/modeling_dataset_xin2_final.parquet",
        "target_col": "concentrate_grade",
        "time_col": None,
        "exclude_cols": [],
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "window_size": 12,
        "epochs": 50,
        "batch_size": 64,
        "lr": 0.001,
        "patience": 8,
        "loss": "mse",
        "random_seed": 42,
        "output_dir": "results/group_branch",
        "allow_synthetic_demo": False,
        "allow_feature_overlap": False,
        "warn_unused_features": True,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    data_path = Path(cfg["data_path"])
    if data_path.exists():
        logger.info(f"读取数据文件: {data_path}")
        if data_path.suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif data_path.suffix in (".csv", ".tsv"):
            df = pd.read_csv(data_path)
        else:
            raise ValueError(f"不支持的数据格式: {data_path.suffix}")
        logger.info(f"数据形状: {df.shape}")
        return df
    if cfg.get("allow_synthetic_demo", False):
        logger.warning(
            f"数据文件不存在: {data_path}，使用合成演示数据（allow_synthetic_demo=true）"
        )
        return _generate_synthetic_data(cfg, logger)
    raise FileNotFoundError(
        f"数据文件不存在: {data_path}。"
        "若需使用合成演示数据，请在配置中设置 allow_synthetic_demo: true"
    )


def _generate_synthetic_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """生成与 train_dml_residual_soft_sensor.py 相同的合成演示数据。"""
    seed = cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)
    n = 2000
    t = np.arange(n)

    c1 = np.sin(t / 200) * 2 + rng.normal(0, 0.1, n)
    c2 = np.cos(t / 300) * 1.5 + rng.normal(0, 0.1, n)
    a1 = 0.5 * c1 + rng.normal(0, 0.3, n)
    a2 = 0.4 * c2 + rng.normal(0, 0.3, n)
    s1 = 0.6 * a1 + 0.3 * c1 + rng.normal(0, 0.2, n)
    s2 = 0.5 * a2 + 0.2 * c2 + rng.normal(0, 0.2, n)
    s3 = 0.4 * a1 + 0.4 * s1 + rng.normal(0, 0.2, n)
    y = (1.2 * a1 + 0.8 * a2 + 0.5 * s1 + 0.3 * s2
         + 0.4 * c1 + 0.2 * c2 + rng.normal(0, 0.3, n))

    target = cfg.get("target_col", "concentrate_grade")
    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "env_temperature": c1,
        "env_feed_rate": c2,
        "op_reagent_flow": a1,
        "op_air_flow": a2,
        "state_foam_thickness": s1,
        "state_pH_value": s2,
        "state_level_sensor": s3,
        target: y,
    })
    logger.info(f"合成数据形状: {df.shape}，列: {df.columns.tolist()}")
    return df


# ─── 预处理 ───────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    time_col = cfg.get("time_col")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        logger.info(f"已按时间列 '{time_col}' 排序")
    else:
        df = df.reset_index(drop=True)

    target_col = cfg["target_col"]
    before = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < before:
        logger.info(f"删除目标列缺失行: {before - len(df)}")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].ffill().bfill()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    for col in num_cols:
        if col == target_col:
            continue
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        if iqr > 0:
            df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    logger.info(f"预处理后数据形状: {df.shape}")
    return df


# ─── 数据切分与窗口化 ─────────────────────────────────────────────────────────

def split_data(
    df: pd.DataFrame, cfg: dict, logger: logging.Logger
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * cfg["train_ratio"])
    n_val = int(n * cfg["val_ratio"])
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train: n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    logger.info(f"数据切分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def make_windows(
    X: np.ndarray, y: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """构造滑动窗口：X[i:i+w] -> y[i+w-1]"""
    xs, ys = [], []
    for i in range(len(y) - window_size + 1):
        xs.append(X[i: i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─── 评估指标 ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": rmse, "R2": float(r2)}


# ─── 模型训练 ─────────────────────────────────────────────────────────────────

def train_model(
    model: CausalGroupBranchModel,
    Xw_tr: np.ndarray,
    yw_tr: np.ndarray,
    Xw_vl: np.ndarray,
    yw_vl: np.ndarray,
    cfg: dict,
    logger: logging.Logger,
) -> CausalGroupBranchModel:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    logger.info(f"训练设备: {device}")

    loss_name = cfg.get("loss", "mse").lower()
    if loss_name == "mae":
        criterion = torch.nn.L1Loss()
    else:
        criterion = torch.nn.MSELoss()
    logger.info(f"损失函数: {loss_name.upper()}")

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    Xtr_t = torch.tensor(Xw_tr, dtype=torch.float32)
    ytr_t = torch.tensor(yw_tr, dtype=torch.float32)
    loader = DataLoader(
        TensorDataset(Xtr_t, ytr_t),
        batch_size=cfg["batch_size"],
        shuffle=True,
    )

    Xvl_t = torch.tensor(Xw_vl, dtype=torch.float32).to(device)
    yvl_t = torch.tensor(yw_vl, dtype=torch.float32).to(device)

    best_val_loss = float("inf")
    best_state = None
    no_improve = 0
    patience = cfg["patience"]

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out["y_hat"].squeeze(-1), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item() * len(yb)
        epoch_loss /= len(yw_tr)

        # 验证集损失
        model.eval()
        with torch.no_grad():
            val_out = model(Xvl_t)
            val_loss = criterion(val_out["y_hat"].squeeze(-1), yvl_t).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            gates_np = model.get_gates().detach().cpu().numpy()
            gate_str = ", ".join(
                f"{n}={g:.3f}" for n, g in zip(model.group_names, gates_np)
            )
            logger.info(
                f"Epoch {epoch:3d}/{cfg['epochs']} | "
                f"train_loss={epoch_loss:.5f} | val_loss={val_loss:.5f} | "
                f"gates=[{gate_str}]"
            )

        if no_improve >= patience:
            logger.info(f"早停 @ epoch {epoch}，best_val_loss={best_val_loss:.6f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ─── 预测与收集分支信息 ───────────────────────────────────────────────────────

def predict_with_info(
    model: CausalGroupBranchModel,
    Xw: np.ndarray,
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    对 Xw 进行预测，同时收集各分支输出。

    Returns:
        y_pred:          [N]
        branch_outputs:  [N, num_groups]
        gates:           [num_groups]
    """
    import torch
    device = next(model.parameters()).device
    model.eval()

    all_preds = []
    all_branches = []
    n = len(Xw)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            xb = torch.tensor(Xw[i: i + batch_size], dtype=torch.float32).to(device)
            out = model(xb)
            all_preds.append(out["y_hat"].squeeze(-1).cpu().numpy())
            all_branches.append(out["branch_outputs"].cpu().numpy())
    gates = model.get_gates().detach().cpu().numpy()
    return (
        np.concatenate(all_preds, axis=0),
        np.concatenate(all_branches, axis=0),
        gates,
    )


# ─── 保存结果 ─────────────────────────────────────────────────────────────────

def save_results(
    output_dir: Path,
    model: CausalGroupBranchModel,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    branch_outputs: np.ndarray,
    gates: np.ndarray,
    y_scaler,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    group_names = model.group_names

    # ── 1. 指标 ──────────────────────────────────────────────────────────────
    metrics = compute_metrics(y_true, y_pred)
    logger.info(
        f"Test: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}"
    )
    metrics_df = pd.DataFrame([metrics])
    metrics_path = output_dir / "group_branch_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {metrics_path}")

    # ── 2. gate 值 ───────────────────────────────────────────────────────────
    gates_df = pd.DataFrame({"group": group_names, "gate_value": gates})
    gates_path = output_dir / "group_branch_gates.csv"
    gates_df.to_csv(gates_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {gates_path}")

    # ── 3. 分支贡献 ──────────────────────────────────────────────────────────
    # branch_outputs: [N, num_groups]（标准化空间）
    contrib_rows = []
    for k, name in enumerate(group_names):
        z_k = branch_outputs[:, k]            # [N]
        contribution_k = gates[k] * z_k       # gate * z_k
        contrib_rows.append({
            "group": name,
            "gate": float(gates[k]),
            "mean_branch_output": float(np.mean(z_k)),
            "mean_abs_contribution": float(np.mean(np.abs(contribution_k))),
        })
    contrib_df = pd.DataFrame(contrib_rows)
    contrib_path = output_dir / "group_branch_contributions.csv"
    contrib_df.to_csv(contrib_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {contrib_path}")
    logger.info("\n分支贡献：\n" + contrib_df.to_string(index=False))

    # ── 4. 逐样本预测 ────────────────────────────────────────────────────────
    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    for k, name in enumerate(group_names):
        pred_df[f"branch_{name}"] = branch_outputs[:, k]
        pred_df[f"contribution_{name}"] = gates[k] * branch_outputs[:, k]
    pred_path = output_dir / "predictions_test.csv"
    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {pred_path}")


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="工艺因果组分支软测量模型训练脚本"
    )
    parser.add_argument(
        "--config",
        default="configs/group_branch.yaml",
        help="配置文件路径（默认：configs/group_branch.yaml）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run_log.txt"
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("工艺因果组分支软测量  train_group_branch.py")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")

    set_seed(cfg["random_seed"])

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    df = load_data(cfg, logger)

    # ── 2. 预处理 ────────────────────────────────────────────────────────────
    df = preprocess(df, cfg, logger)

    # ── 3. 确定特征列 ────────────────────────────────────────────────────────
    target_col = cfg["target_col"]
    time_col = cfg.get("time_col")
    exclude_cols = set(cfg.get("exclude_cols", []))
    if time_col:
        exclude_cols.add(time_col)

    # 只保留数值列，排除目标列和指定排除列
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    feature_cols = [
        c for c in num_cols
        if c != target_col and c not in exclude_cols
    ]

    if not feature_cols:
        logger.error("特征列为空，退出。")
        sys.exit(1)

    num_features = len(feature_cols)
    logger.info(f"特征列数量: {num_features}")
    logger.info("特征列-索引对照表:")
    for i, col in enumerate(feature_cols):
        logger.info(f"  [{i:3d}] {col}")

    # ── 4. 校验 groups 配置 ──────────────────────────────────────────────────
    groups_cfg: dict = cfg.get("groups", {})
    if not groups_cfg:
        logger.error("配置中未定义 groups，退出。")
        sys.exit(1)

    logger.info("-" * 40)
    logger.info("变量分组配置：")
    for name, gcfg in groups_cfg.items():
        idxs = gcfg.get("indices", [])
        btype = gcfg.get("branch_type", "gru")
        hdim = gcfg.get("hidden_dim", 32)
        cols = [feature_cols[i] for i in idxs if 0 <= i < num_features]
        logger.info(f"  group={name}, indices={idxs}, cols={cols}, "
                    f"branch_type={btype}, hidden_dim={hdim}")

    model_cfg: dict = cfg.get("model", {})
    logger.info(f"模型配置: use_gate={model_cfg.get('use_gate', True)}, "
                f"trainable_gate={model_cfg.get('trainable_gate', True)}, "
                f"gate_init={model_cfg.get('gate_init', 0.5)}, "
                f"output_bias={model_cfg.get('output_bias', True)}")

    # ── 5. 数据切分 ──────────────────────────────────────────────────────────
    train_df, val_df, test_df = split_data(df, cfg, logger)

    # ── 6. 标准化 ────────────────────────────────────────────────────────────
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

    # ── 7. 滑动窗口 ──────────────────────────────────────────────────────────
    window_size = cfg["window_size"]
    Xw_tr, yw_tr = make_windows(X_tr, y_tr, window_size)
    Xw_vl, yw_vl = make_windows(X_vl, y_vl, window_size)
    Xw_te, yw_te = make_windows(X_te, y_te, window_size)
    logger.info(f"窗口化后样本数: train={len(yw_tr)}, val={len(yw_vl)}, test={len(yw_te)}")

    # ── 8. 构建模型 ──────────────────────────────────────────────────────────
    try:
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=model_cfg,
            window_size=window_size,
            num_features=num_features,
            allow_feature_overlap=bool(cfg.get("allow_feature_overlap", False)),
            warn_unused_features=bool(cfg.get("warn_unused_features", True)),
        )
    except (ValueError, KeyError) as e:
        logger.error(f"模型构建失败: {e}")
        sys.exit(1)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"模型参数总量: {total_params:,}")

    # ── 9. 训练 ──────────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("开始训练...")
    model = train_model(model, Xw_tr, yw_tr, Xw_vl, yw_vl, cfg, logger)

    # ── 10. 打印最终 gate 值 ─────────────────────────────────────────────────
    final_gates = model.get_gates().detach().cpu().numpy()
    logger.info("最终 gate 值：")
    for name, g in zip(model.group_names, final_gates):
        logger.info(f"  {name}: {g:.4f}")

    # ── 11. 在测试集预测 ─────────────────────────────────────────────────────
    y_pred_scaled, branch_outputs_scaled, gates = predict_with_info(model, Xw_te)

    # 反标准化
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()

    # ── 12. 保存结果 ─────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("保存结果...")
    save_results(
        output_dir=output_dir,
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        branch_outputs=branch_outputs_scaled,  # 标准化空间的分支输出
        gates=gates,
        y_scaler=y_scaler,
        logger=logger,
    )

    logger.info("=" * 60)
    logger.info("运行完成。输出文件：")
    for f in sorted(output_dir.iterdir()):
        logger.info(f"  {f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
