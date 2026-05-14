"""
train_dml_residual_soft_sensor.py
==================================
DML 正交残差软测量第二阶段脚本。

核心思想：
  普通软测量：X_seq -> LSTM -> y
  DML 残差化：
    C -> g(C) -> y_base          # 用工况变量预测 y 基线
    y_res = y - y_base           # 残差化 y

    C -> q_j(C) -> X_res_j = X_j - q_j(C)   # 残差化 A 和 S

    [A_res, S_res] 滑动窗口 -> LSTM -> y_res_hat
    y_hat = y_base + y_res_hat

用法：
  python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor.yaml

输出（results/residual_soft_sensor/）：
  variable_roles.csv
  residual_feature_summary.csv
  y_baseline_predictions.csv
  predictions_test.csv
  metrics_compare.csv
  run_log.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# ─── 懒加载重型依赖，避免导入时崩溃 ─────────────────────────────────────────

def _import_sklearn():
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return StandardScaler, RandomForestRegressor, MLPRegressor, mean_absolute_error, mean_squared_error, r2_score


def _import_lgbm():
    try:
        import lightgbm as lgb
        return lgb
    except ImportError:
        return None


def _import_torch():
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    return torch, nn, DataLoader, TensorDataset


def _import_networkx():
    import networkx as nx
    return nx


# ─── 随机种子 ─────────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch, *_ = _import_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ─── 日志设置 ─────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("dml_residual")
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
    # 默认值填充
    defaults = {
        "data_path": "data/modeling_dataset_xin2_final.parquet",
        "dag_path": "data/features/global_edges.csv",
        "target_col": "concentrate_grade",
        "time_col": None,
        "operation_vars": [],
        "exclude_cols": [],
        "train_ratio": 0.70,
        "val_ratio": 0.15,
        "test_ratio": 0.15,
        "window_size": 12,
        "residualization_model": "lightgbm",
        "residual_sequence_model": "lstm",
        "lstm_hidden_size": 64,
        "lstm_num_layers": 2,
        "lstm_dropout": 0.1,
        "lstm_epochs": 50,
        "lstm_batch_size": 64,
        "lstm_lr": 0.001,
        "lstm_patience": 8,
        "random_seed": 42,
        "output_dir": "results/residual_soft_sensor",
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
    else:
        logger.warning(f"数据文件不存在: {data_path}，使用合成演示数据")
        return _generate_synthetic_data(cfg, logger)


def _generate_synthetic_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    """生成合成演示数据，用于在无真实数据时验证流程可运行。"""
    seed = cfg.get("random_seed", 42)
    rng = np.random.default_rng(seed)
    n = 2000
    t = np.arange(n)

    # 工况变量 C（前置、慢变）
    c1 = np.sin(t / 200) * 2 + rng.normal(0, 0.1, n)
    c2 = np.cos(t / 300) * 1.5 + rng.normal(0, 0.1, n)

    # 操作变量 A（受工况影响）
    a1 = 0.5 * c1 + rng.normal(0, 0.3, n)
    a2 = 0.4 * c2 + rng.normal(0, 0.3, n)

    # 状态变量 S（受 A 和 C 影响）
    s1 = 0.6 * a1 + 0.3 * c1 + rng.normal(0, 0.2, n)
    s2 = 0.5 * a2 + 0.2 * c2 + rng.normal(0, 0.2, n)
    s3 = 0.4 * a1 + 0.4 * s1 + rng.normal(0, 0.2, n)

    # 目标变量 y（受 A、S、C 影响）
    y = (1.2 * a1 + 0.8 * a2 + 0.5 * s1 + 0.3 * s2
         + 0.4 * c1 + 0.2 * c2 + rng.normal(0, 0.3, n))

    target = cfg.get("target_col", "concentrate_grade")
    op_vars = cfg.get("operation_vars", [])
    if not op_vars:
        op_vars = ["op_reagent_flow", "op_air_flow"]

    df = pd.DataFrame({
        "timestamp": pd.date_range("2023-01-01", periods=n, freq="10min"),
        "env_temperature": c1,
        "env_feed_rate": c2,
        op_vars[0] if len(op_vars) > 0 else "op_reagent_flow": a1,
        op_vars[1] if len(op_vars) > 1 else "op_air_flow": a2,
        "state_foam_thickness": s1,
        "state_pH_value": s2,
        "state_level_sensor": s3,
        target: y,
    })
    logger.info(f"合成数据形状: {df.shape}，列: {df.columns.tolist()}")
    # 更新 config 中操作变量（若为空）
    if not cfg.get("operation_vars"):
        cfg["operation_vars"] = op_vars
    return df


# ─── DAG 加载与变量角色推断 ────────────────────────────────────────────────────

# 关键词黑名单：命名含这些词的列排除入 C
_LEAK_KEYWORDS = {
    "target", "label", "grade_y", "future", "after", "prediction",
    "output_quality", "result", "predict", "forecast",
}


def _contains_leak_keyword(name: str) -> bool:
    nl = name.lower()
    return any(kw in nl for kw in _LEAK_KEYWORDS)


def infer_variable_roles(
    df: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """
    推断每个变量的角色：target / operation_A / confounder_C / state_S / excluded

    规则（按问题说明）：
      C 候选：
        1. 非 y；2. 非 A；3. 非 y 后代；4. 非 A 后代；
        5. 不在 A->...->y 中介路径上；
        6. 优先前置/外生节点；
        7. 无明显泄漏关键词；
        8. 非对撞节点。
      S 候选：非 y、非 A、非 C、非明显泄漏。
    """
    target_col = cfg["target_col"]
    op_vars: List[str] = list(cfg.get("operation_vars", []))
    exclude_cols: List[str] = list(cfg.get("exclude_cols", []))
    time_col: Optional[str] = cfg.get("time_col")
    dag_path = Path(cfg.get("dag_path", ""))

    # 检测并排除 datetime 类型列（即使 time_col 未配置）
    datetime_cols = set(df.select_dtypes(include=["datetime64", "object"]).columns.tolist())

    all_cols = [c for c in df.columns if c != time_col and c not in datetime_cols]

    # ── 加载 DAG（可选） ──────────────────────────────────────────────────────
    dag_info: dict = {}
    if dag_path.exists():
        try:
            dag_info = _parse_dag(dag_path, target_col, op_vars, logger)
        except Exception as e:
            logger.warning(f"DAG 解析失败: {e}；退回保守规则推断")

    descendants_of_y: set = dag_info.get("descendants_of_y", set())
    descendants_of_any_a: set = dag_info.get("descendants_of_any_a", set())
    mediators_a_to_y: set = dag_info.get("mediators_a_to_y", set())
    colliders: set = dag_info.get("colliders", set())
    dag_ancestors_of_y: set = dag_info.get("ancestors_of_y", set())

    rows = []
    for col in all_cols:
        if col in exclude_cols:
            rows.append({"variable": col, "role": "excluded", "reason": "在 exclude_cols 中"})
            continue

        if col == target_col:
            rows.append({"variable": col, "role": "target", "reason": "目标变量"})
            continue

        if col in op_vars:
            rows.append({"variable": col, "role": "operation_A", "reason": "人工指定操作变量"})
            continue

        if _contains_leak_keyword(col):
            rows.append({"variable": col, "role": "excluded", "reason": "列名含泄漏关键词"})
            continue

        # 排除 y 后代
        if col in descendants_of_y:
            rows.append({"variable": col, "role": "excluded", "reason": "y 后代（数据泄漏）"})
            continue

        # 判断是否为 C（工况/混杂变量）
        is_c = _classify_as_C(
            col, dag_ancestors_of_y, descendants_of_any_a,
            mediators_a_to_y, colliders, op_vars, target_col, df,
        )
        if is_c is not None:
            rows.append({"variable": col, "role": "confounder_C", "reason": is_c})
            continue

        # 其余为 S（状态变量）
        rows.append({"variable": col, "role": "state_S",
                     "reason": "非目标/操作/工况/排除变量，归入状态变量"})

    roles_df = pd.DataFrame(rows)

    # ── 记录 C 是否充足 ─────────────────────────────────────────────────────
    n_c = (roles_df["role"] == "confounder_C").sum()
    if n_c == 0:
        logger.warning("C candidates insufficient (n=0); DML residualization will be skipped.")
    elif n_c < 2:
        logger.warning(f"C 变量仅有 {n_c} 个，残差化效果可能有限")
    else:
        logger.info(f"C（工况/混杂）变量: {n_c} 个")

    n_a = (roles_df["role"] == "operation_A").sum()
    n_s = (roles_df["role"] == "state_S").sum()
    n_ex = (roles_df["role"] == "excluded").sum()
    logger.info(f"变量角色汇总: target=1, A={n_a}, C={n_c}, S={n_s}, excluded={n_ex}")

    return roles_df


def _classify_as_C(
    col: str,
    dag_ancestors_of_y: set,
    descendants_of_any_a: set,
    mediators: set,
    colliders: set,
    op_vars: List[str],
    target_col: str,
    df: pd.DataFrame,
) -> Optional[str]:
    """
    返回 reason 字符串若应归类为 C，否则返回 None。
    """
    # 规则 4：是 A 的后代 -> 不是 C
    if col in descendants_of_any_a:
        return None

    # 规则 5：是 A->y 中介 -> 不是 C
    if col in mediators:
        return None

    # 规则 8：对撞节点 -> 不是 C
    if col in colliders:
        return None

    # 如果有 DAG 信息：规则 6，优先祖先节点
    if dag_ancestors_of_y and col in dag_ancestors_of_y:
        return "DAG 中 y 的祖先且非 A 后代/中介/对撞"

    # 若无 DAG，用启发式规则：
    # 慢变量（方差小、自相关高）、外生变量（名称含 env/feed/temp/speed/setting 等）
    heuristic = _heuristic_C_rule(col, df)
    if heuristic:
        return heuristic

    return None


_C_KEYWORDS = {"env", "feed", "temp", "speed", "setting", "condition",
               "pressure_inlet", "water", "slurry", "grade_feed",
               "temperature", "density", "humidity"}


def _heuristic_C_rule(col: str, df: pd.DataFrame) -> Optional[str]:
    """基于列名启发式判断是否为 C（当 DAG 信息不足时）。"""
    nl = col.lower()
    for kw in _C_KEYWORDS:
        if kw in nl:
            return f"列名含工况关键词 '{kw}'"
    return None


def _parse_dag(
    dag_path: Path,
    target_col: str,
    op_vars: List[str],
    logger: logging.Logger,
) -> dict:
    """从 DAG 边表 CSV 解析因果结构，提取各角色节点集。"""
    nx = _import_networkx()

    df_edges = pd.read_csv(dag_path)
    required = {"source", "target"}
    if not required.issubset(df_edges.columns):
        raise ValueError(f"DAG 边表缺少列: {required - set(df_edges.columns)}")

    G = nx.DiGraph()
    for _, row in df_edges.iterrows():
        src = str(row["source"]).strip()
        tgt = str(row["target"]).strip()
        G.add_edge(src, tgt)

    import re
    lag_pat = re.compile(r"^(.+?)_lag\d+$")

    def base_name(n: str) -> str:
        m = lag_pat.match(n)
        return m.group(1) if m else n

    # 找到 target 节点（支持 _lag0 形式）
    nodes = set(G.nodes())
    target_node = None
    for candidate in [target_col, f"{target_col}_lag0"]:
        if candidate in nodes:
            target_node = candidate
            break

    if target_node is None:
        logger.warning(f"目标变量 '{target_col}' 不在 DAG 中，退回保守规则")
        return {}

    descendants_of_y = {base_name(n) for n in nx.descendants(G, target_node)}
    ancestors_of_y = {base_name(n) for n in nx.ancestors(G, target_node)}

    descendants_of_any_a: set = set()
    mediators: set = set()
    for av in op_vars:
        a_node = av if av in nodes else f"{av}_lag0" if f"{av}_lag0" in nodes else None
        if a_node is None:
            continue
        desc_a = {base_name(n) for n in nx.descendants(G, a_node)}
        descendants_of_any_a |= desc_a
        # 中介：在 A 和 Y 之间的路径上
        if target_node in nx.descendants(G, a_node):
            for path in nx.all_simple_paths(G, a_node, target_node):
                for n in path[1:-1]:  # 去掉端点
                    mediators.add(base_name(n))

    # 对撞：被两个 A 同时指向的节点
    a_nodes = []
    for av in op_vars:
        an = av if av in nodes else f"{av}_lag0" if f"{av}_lag0" in nodes else None
        if an:
            a_nodes.append(an)
    colliders: set = set()
    if len(a_nodes) >= 2:
        for node in G.nodes():
            preds = set(G.predecessors(node))
            if len(preds & set(a_nodes)) >= 2:
                colliders.add(base_name(node))

    logger.info(f"DAG 解析: ancestors_of_y={len(ancestors_of_y)}, "
                f"descendants_of_y={len(descendants_of_y)}, "
                f"mediators={len(mediators)}, colliders={len(colliders)}")

    return {
        "ancestors_of_y": ancestors_of_y,
        "descendants_of_y": descendants_of_y,
        "descendants_of_any_a": descendants_of_any_a,
        "mediators_a_to_y": mediators,
        "colliders": colliders,
    }


# ─── 数据预处理 ───────────────────────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
) -> pd.DataFrame:
    """时间排序、缺失值、异常值处理。"""
    time_col = cfg.get("time_col")
    if time_col and time_col in df.columns:
        df = df.sort_values(time_col).reset_index(drop=True)
        logger.info(f"已按时间列 '{time_col}' 排序")
    else:
        df = df.reset_index(drop=True)

    # 删除目标列全为空的行
    target_col = cfg["target_col"]
    before = len(df)
    df = df.dropna(subset=[target_col])
    if len(df) < before:
        logger.info(f"删除目标列缺失行: {before - len(df)}")

    # 数值列：向前填充 + 均值填充
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    df[num_cols] = df[num_cols].ffill().bfill()
    df[num_cols] = df[num_cols].fillna(df[num_cols].mean())

    # IQR 异常值剪裁（只在数值列上，不动目标列）
    for col in num_cols:
        if col == target_col:
            continue
        q1, q3 = df[col].quantile(0.01), df[col].quantile(0.99)
        iqr = q3 - q1
        if iqr > 0:
            df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    logger.info(f"预处理后数据形状: {df.shape}")
    return df


def split_data(
    df: pd.DataFrame,
    cfg: dict,
    logger: logging.Logger,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """时序切分（不随机打乱）。"""
    n = len(df)
    tr = cfg["train_ratio"]
    vl = cfg["val_ratio"]
    n_train = int(n * tr)
    n_val = int(n * vl)
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train: n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    logger.info(f"数据切分: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")
    return train_df, val_df, test_df


def fit_scalers(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Tuple[object, object]:
    """在训练集上拟合特征和目标 scaler，返回 (feat_scaler, y_scaler)。"""
    from sklearn.preprocessing import StandardScaler
    feat_scaler = StandardScaler()
    y_scaler = StandardScaler()
    feat_scaler.fit(train_df[feature_cols].values)
    y_scaler.fit(train_df[[target_col]].values)
    return feat_scaler, y_scaler


def apply_scalers(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    feat_scaler,
    y_scaler,
) -> Tuple[np.ndarray, np.ndarray]:
    X = feat_scaler.transform(df[feature_cols].values)
    y = y_scaler.transform(df[[target_col]].values).ravel()
    return X, y


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """构造滑动窗口: X[i:i+w] -> y[i+w-1]。"""
    n = len(y)
    xs, ys = [], []
    for i in range(n - window_size + 1):
        xs.append(X[i: i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# ─── LSTM 模型 ────────────────────────────────────────────────────────────────

class LSTMRegressor:
    """
    轻量 LSTM 回归封装（PyTorch）。
    接口与 sklearn 类似：fit / predict。
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
        epochs: int = 50,
        batch_size: int = 64,
        lr: float = 0.001,
        patience: int = 8,
        seed: int = 42,
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.patience = patience
        self.seed = seed
        self._model = None
        self._device = None

    def _build(self):
        torch, nn, DataLoader, TensorDataset = _import_torch()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class _Net(nn.Module):
            def __init__(self, inp, hid, layers, drop):
                super().__init__()
                self.lstm = nn.LSTM(
                    inp, hid, layers,
                    batch_first=True,
                    dropout=drop if layers > 1 else 0.0,
                )
                self.fc = nn.Linear(hid, 1)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :]).squeeze(-1)

        model = _Net(self.input_size, self.hidden_size, self.num_layers, self.dropout)
        model = model.to(device)
        return model, device

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        logger: Optional[logging.Logger] = None,
    ) -> "LSTMRegressor":
        torch, nn, DataLoader, TensorDataset = _import_torch()
        set_seed(self.seed)
        model, device = self._build()
        self._device = device

        Xtr = torch.tensor(X_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)
        loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0

        for epoch in range(1, self.epochs + 1):
            model.train()
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            if X_val is not None and y_val is not None:
                model.eval()
                with torch.no_grad():
                    Xv = torch.tensor(X_val, dtype=torch.float32).to(device)
                    yv = torch.tensor(y_val, dtype=torch.float32).to(device)
                    val_loss = criterion(model(Xv), yv).item()
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1
                    if no_improve >= self.patience:
                        if logger:
                            logger.info(f"  早停 @ epoch {epoch}，best_val_loss={best_val_loss:.6f}")
                        break

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        torch, *_ = _import_torch()
        self._model.eval()
        with torch.no_grad():
            Xt = torch.tensor(X, dtype=torch.float32).to(self._device)
            return self._model(Xt).cpu().numpy()


# ─── 残差化基模型（g / q） ────────────────────────────────────────────────────

def build_residualization_model(model_type: str, seed: int):
    """返回支持 fit/predict 接口的回归模型实例。"""
    lgb = _import_lgbm()
    if model_type == "lightgbm" and lgb is not None:
        return lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    elif model_type == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    else:
        # 如果 lgb 未安装，退回随机森林
        if model_type == "lightgbm":
            logging.getLogger("dml_residual").warning(
                "lightgbm 未安装，退回 random_forest"
            )
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)


# ─── 评估指标 ─────────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


# ─── Model 0：基线软测量 ──────────────────────────────────────────────────────

def run_model0_baseline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    all_feature_cols: List[str],
    target_col: str,
    cfg: dict,
    logger: logging.Logger,
) -> dict:
    """Model 0: 全部特征 -> LSTM -> y。"""
    logger.info("=" * 60)
    logger.info("Model 0: 基线软测量 (all features -> LSTM -> y)")

    window_size = cfg["window_size"]
    feat_scaler, y_scaler = fit_scalers(train_df, all_feature_cols, target_col)

    X_tr, y_tr = apply_scalers(train_df, all_feature_cols, target_col, feat_scaler, y_scaler)
    X_vl, y_vl = apply_scalers(val_df, all_feature_cols, target_col, feat_scaler, y_scaler)
    X_te, y_te = apply_scalers(test_df, all_feature_cols, target_col, feat_scaler, y_scaler)

    Xw_tr, yw_tr = make_windows(X_tr, y_tr, window_size)
    Xw_vl, yw_vl = make_windows(X_vl, y_vl, window_size)
    Xw_te, yw_te = make_windows(X_te, y_te, window_size)

    model = LSTMRegressor(
        input_size=len(all_feature_cols),
        hidden_size=cfg["lstm_hidden_size"],
        num_layers=cfg["lstm_num_layers"],
        dropout=cfg["lstm_dropout"],
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch_size"],
        lr=cfg["lstm_lr"],
        patience=cfg["lstm_patience"],
        seed=cfg["random_seed"],
    )
    model.fit(Xw_tr, yw_tr, Xw_vl, yw_vl, logger=logger)

    y_pred_scaled = model.predict(Xw_te)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Model 0 Test: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    return {
        "model_name": "baseline",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ─── Model 1：因果输入软测量 ──────────────────────────────────────────────────

def run_model1_causal_input(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    a_cols: List[str],
    s_cols: List[str],
    target_col: str,
    cfg: dict,
    logger: logging.Logger,
) -> dict:
    """Model 1: A + S 特征 -> LSTM -> y。"""
    logger.info("=" * 60)
    logger.info("Model 1: 因果输入软测量 (A + S -> LSTM -> y)")

    causal_cols = a_cols + s_cols
    if not causal_cols:
        logger.warning("A + S 为空，跳过 Model 1")
        return {
            "model_name": "causal_input",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
        }

    window_size = cfg["window_size"]
    feat_scaler, y_scaler = fit_scalers(train_df, causal_cols, target_col)

    X_tr, y_tr = apply_scalers(train_df, causal_cols, target_col, feat_scaler, y_scaler)
    X_vl, y_vl = apply_scalers(val_df, causal_cols, target_col, feat_scaler, y_scaler)
    X_te, y_te = apply_scalers(test_df, causal_cols, target_col, feat_scaler, y_scaler)

    Xw_tr, yw_tr = make_windows(X_tr, y_tr, window_size)
    Xw_vl, yw_vl = make_windows(X_vl, y_vl, window_size)
    Xw_te, yw_te = make_windows(X_te, y_te, window_size)

    model = LSTMRegressor(
        input_size=len(causal_cols),
        hidden_size=cfg["lstm_hidden_size"],
        num_layers=cfg["lstm_num_layers"],
        dropout=cfg["lstm_dropout"],
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch_size"],
        lr=cfg["lstm_lr"],
        patience=cfg["lstm_patience"],
        seed=cfg["random_seed"],
    )
    model.fit(Xw_tr, yw_tr, Xw_vl, yw_vl, logger=logger)

    y_pred_scaled = model.predict(Xw_te)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Model 1 Test: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    return {
        "model_name": "causal_input",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ─── Model 2：DML 残差软测量 ──────────────────────────────────────────────────

def run_model2_dml_residual(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    c_cols: List[str],
    a_cols: List[str],
    s_cols: List[str],
    target_col: str,
    cfg: dict,
    logger: logging.Logger,
    output_dir: Path,
) -> dict:
    """
    Model 2: DML 残差软测量
      C -> g(C) -> y_base
      C -> q_j(C) -> X_res_j
      [A_res, S_res] 窗口序列 -> LSTM -> y_res
      y_hat = y_base + y_res_hat
    """
    logger.info("=" * 60)
    logger.info("Model 2: DML 残差软测量")

    from sklearn.preprocessing import StandardScaler

    residual_as_cols = a_cols + s_cols

    # ── C 不足时降级 ─────────────────────────────────────────────────────────
    if not c_cols:
        logger.warning("C candidates insufficient (n=0); Model 2 (DML 残差) 不可用，跳过残差化。")
        return {
            "model_name": "dml_residual_soft_sensor",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                        "note": "residual model unavailable: C is empty"},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "residual_feature_summary": pd.DataFrame(),
            "y_baseline_predictions": pd.DataFrame(),
            "predictions_test": pd.DataFrame(),
        }

    if not residual_as_cols:
        logger.warning("A + S 为空，Model 2 不可用")
        return {
            "model_name": "dml_residual_soft_sensor",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                        "note": "residual model unavailable: A+S is empty"},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "residual_feature_summary": pd.DataFrame(),
            "y_baseline_predictions": pd.DataFrame(),
            "predictions_test": pd.DataFrame(),
        }

    seed = cfg["random_seed"]
    model_type = cfg.get("residualization_model", "lightgbm")
    window_size = cfg["window_size"]

    # ── 步骤 1：C 的标准化（只在 train 上 fit） ────────────────────────────
    c_scaler = StandardScaler()
    C_train = c_scaler.fit_transform(train_df[c_cols].values)
    C_val = c_scaler.transform(val_df[c_cols].values)
    C_test = c_scaler.transform(test_df[c_cols].values)

    y_train_raw = train_df[target_col].values
    y_val_raw = val_df[target_col].values
    y_test_raw = test_df[target_col].values

    # ── 步骤 2：g_model: C -> y ───────────────────────────────────────────
    logger.info("训练 g_model: C -> y (残差化 y 的基线模型)")
    g_model = build_residualization_model(model_type, seed)
    g_model.fit(C_train, y_train_raw)

    y_base_train = g_model.predict(C_train)
    y_base_val = g_model.predict(C_val)
    y_base_test = g_model.predict(C_test)

    g_score_train = float(np.corrcoef(y_train_raw, y_base_train)[0, 1] ** 2)
    logger.info(f"  g_model R² (train): {g_score_train:.4f}")

    y_res_train = y_train_raw - y_base_train
    y_res_val = y_val_raw - y_base_val
    y_res_test = y_test_raw - y_base_test

    # 保存 y_baseline 预测（test 集）
    y_baseline_pred_df = pd.DataFrame({
        "index": test_df.index,
        "y_true": y_test_raw,
        "y_base": y_base_test,
        "y_res": y_res_test,
    })

    # ── 步骤 3：q_j: C -> X_j（对每个 A 和 S 列残差化）────────────────────
    logger.info("训练 q_model: C -> A_j / S_j（残差化操作变量和状态变量）")
    q_models: Dict[str, object] = {}
    residual_summary_rows = []

    as_scaler = StandardScaler()
    as_scaler.fit(train_df[residual_as_cols].values)
    AS_train_scaled = as_scaler.transform(train_df[residual_as_cols].values)
    AS_val_scaled = as_scaler.transform(val_df[residual_as_cols].values)
    AS_test_scaled = as_scaler.transform(test_df[residual_as_cols].values)

    AS_res_train = np.empty_like(AS_train_scaled)
    AS_res_val = np.empty_like(AS_val_scaled)
    AS_res_test = np.empty_like(AS_test_scaled)

    for j, col in enumerate(residual_as_cols):
        qm = build_residualization_model(model_type, seed + j + 1)
        xj_train = AS_train_scaled[:, j]
        xj_val = AS_val_scaled[:, j]
        xj_test = AS_test_scaled[:, j]

        qm.fit(C_train, xj_train)
        q_models[col] = qm

        xj_hat_train = qm.predict(C_train)
        xj_hat_val = qm.predict(C_val)
        xj_hat_test = qm.predict(C_test)

        xj_res_train = xj_train - xj_hat_train
        xj_res_val = xj_val - xj_hat_val
        xj_res_test = xj_test - xj_hat_test

        AS_res_train[:, j] = xj_res_train
        AS_res_val[:, j] = xj_res_val
        AS_res_test[:, j] = xj_res_test

        role = "operation_A" if col in a_cols else "state_S"
        q_r2 = float(np.corrcoef(xj_train, xj_hat_train)[0, 1] ** 2) if np.std(xj_hat_train) > 1e-8 else 0.0
        residual_summary_rows.append({
            "variable": col,
            "role": role,
            "original_mean": float(np.mean(xj_train)),
            "residual_mean": float(np.mean(xj_res_train)),
            "original_std": float(np.std(xj_train)),
            "residual_std": float(np.std(xj_res_train)),
            "q_model_score": q_r2,
        })

    residual_feature_summary = pd.DataFrame(residual_summary_rows)

    # ── 步骤 4：残差化 y 的标准化 ─────────────────────────────────────────
    y_res_scaler = StandardScaler()
    y_res_train_scaled = y_res_scaler.fit_transform(y_res_train.reshape(-1, 1)).ravel()
    y_res_val_scaled = y_res_scaler.transform(y_res_val.reshape(-1, 1)).ravel()
    y_res_test_scaled = y_res_scaler.transform(y_res_test.reshape(-1, 1)).ravel()

    # ── 步骤 5：滑动窗口（A_res, S_res -> y_res） ─────────────────────────
    Xw_tr, yw_tr = make_windows(AS_res_train, y_res_train_scaled, window_size)
    Xw_vl, yw_vl = make_windows(AS_res_val, y_res_val_scaled, window_size)
    Xw_te, yw_te = make_windows(AS_res_test, y_res_test_scaled, window_size)

    # ── 步骤 6：训练残差 LSTM ─────────────────────────────────────────────
    logger.info("训练残差 LSTM: [A_res, S_res] 序列 -> y_res")
    residual_lstm = LSTMRegressor(
        input_size=len(residual_as_cols),
        hidden_size=cfg["lstm_hidden_size"],
        num_layers=cfg["lstm_num_layers"],
        dropout=cfg["lstm_dropout"],
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch_size"],
        lr=cfg["lstm_lr"],
        patience=cfg["lstm_patience"],
        seed=cfg["random_seed"],
    )
    residual_lstm.fit(Xw_tr, yw_tr, Xw_vl, yw_vl, logger=logger)

    # ── 步骤 7：最终预测 y_hat = y_base + y_res_hat ───────────────────────
    y_res_pred_scaled = residual_lstm.predict(Xw_te)
    y_res_pred = y_res_scaler.inverse_transform(y_res_pred_scaled.reshape(-1, 1)).ravel()
    y_res_true = y_res_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()

    # 窗口对齐：去掉 y_base_test 前 window_size-1 个点
    align_offset = window_size - 1
    y_base_aligned = y_base_test[align_offset: align_offset + len(y_res_pred)]
    y_true_aligned = y_test_raw[align_offset: align_offset + len(y_res_pred)]

    y_hat = y_base_aligned + y_res_pred

    metrics_final = compute_metrics(y_true_aligned, y_hat)
    metrics_res = compute_metrics(y_res_true, y_res_pred)

    logger.info(f"Model 2 Test (原始 y): MAE={metrics_final['MAE']:.4f}, "
                f"RMSE={metrics_final['RMSE']:.4f}, R2={metrics_final['R2']:.4f}")
    logger.info(f"Model 2 Test (y_res):  MAE={metrics_res['MAE']:.4f}, "
                f"RMSE={metrics_res['RMSE']:.4f}, R2={metrics_res['R2']:.4f}")

    # 完整预测记录
    predictions_test = pd.DataFrame({
        "index": test_df.index[align_offset: align_offset + len(y_res_pred)],
        "y_true": y_true_aligned,
        "y_base": y_base_aligned,
        "y_res_true": y_res_true,
        "y_res_pred": y_res_pred,
        "y_pred": y_hat,
    })

    return {
        "model_name": "dml_residual_soft_sensor",
        "metrics": metrics_final,
        "y_true": y_true_aligned,
        "y_pred": y_hat,
        "residual_feature_summary": residual_feature_summary,
        "y_baseline_predictions": y_baseline_pred_df,
        "predictions_test": predictions_test,
    }


# ─── 保存所有输出 ─────────────────────────────────────────────────────────────

def save_outputs(
    output_dir: Path,
    roles_df: pd.DataFrame,
    model0: dict,
    model1: dict,
    model2: dict,
    logger: logging.Logger,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 变量角色表
    roles_path = output_dir / "variable_roles.csv"
    roles_df.to_csv(roles_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {roles_path}")

    # 2. y_baseline_predictions.csv
    if not model2.get("y_baseline_predictions", pd.DataFrame()).empty:
        bp_path = output_dir / "y_baseline_predictions.csv"
        model2["y_baseline_predictions"].to_csv(bp_path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {bp_path}")

    # 3. residual_feature_summary.csv
    if not model2.get("residual_feature_summary", pd.DataFrame()).empty:
        rs_path = output_dir / "residual_feature_summary.csv"
        model2["residual_feature_summary"].to_csv(rs_path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {rs_path}")

    # 4. predictions_test.csv
    if not model2.get("predictions_test", pd.DataFrame()).empty:
        pt_path = output_dir / "predictions_test.csv"
        model2["predictions_test"].to_csv(pt_path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {pt_path}")

    # 5. metrics_compare.csv
    rows = []
    for m_dict, split in [
        (model0, "test"),
        (model1, "test"),
        (model2, "test"),
    ]:
        mname = m_dict.get("model_name", "unknown")
        mvals = m_dict.get("metrics", {})
        row = {
            "model_name": mname,
            "split": split,
            "MAE": mvals.get("MAE", float("nan")),
            "RMSE": mvals.get("RMSE", float("nan")),
            "R2": mvals.get("R2", float("nan")),
        }
        note = mvals.get("note", "")
        if note:
            row["note"] = note
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    mc_path = output_dir / "metrics_compare.csv"
    metrics_df.to_csv(mc_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {mc_path}")
    logger.info("\n" + metrics_df.to_string(index=False))


# ─── 主流程 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DML 正交残差软测量 (Phase 2 DML Residual Soft Sensor)"
    )
    parser.add_argument(
        "--config",
        default="configs/residual_soft_sensor.yaml",
        help="配置文件路径（默认：configs/residual_soft_sensor.yaml）",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run_log.txt"
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("DML 正交残差软测量  train_dml_residual_soft_sensor.py")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"配置内容: {json.dumps(cfg, ensure_ascii=False, indent=2)}")

    set_seed(cfg["random_seed"])

    # ── 1. 加载数据 ──────────────────────────────────────────────────────────
    df = load_data(cfg, logger)

    # ── 2. 预处理 ────────────────────────────────────────────────────────────
    df = preprocess(df, cfg, logger)

    # ── 3. 变量角色推断 ──────────────────────────────────────────────────────
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
    logger.info(f"排除变量 ({len(excluded)}): {excluded}")

    if not c_cols:
        logger.warning("C candidates insufficient (n=0); DML residualization will be skipped.")

    # 所有特征列（排除 target 和 time_col）
    skip_cols = {target_col}
    if time_col:
        skip_cols.add(time_col)
    all_feature_cols = [c for c in df.columns if c not in skip_cols
                        and c not in excluded]

    logger.info(f"全部特征列 ({len(all_feature_cols)}): {all_feature_cols}")
    logger.info(f"窗口长度: {cfg['window_size']}")
    logger.info(f"数据切分方式: train={cfg['train_ratio']}, val={cfg['val_ratio']}, test={cfg['test_ratio']}")

    # ── 4. 数据切分 ──────────────────────────────────────────────────────────
    # 只保留数值列用于建模
    model_cols = [c for c in [target_col] + all_feature_cols if c in df.columns]
    if time_col and time_col in df.columns:
        model_cols = [time_col] + model_cols
    df_model = df[[c for c in model_cols if c in df.columns]].select_dtypes(include=[np.number, "datetime64"])

    # 再次只保留数值列
    df_model = df[[c for c in model_cols if c in df.columns]]
    num_cols_only = df_model.select_dtypes(include=np.number).columns.tolist()
    df_model = df_model[num_cols_only]

    # 更新 feature/role 列表为实际存在的数值列
    all_feature_cols = [c for c in all_feature_cols if c in df_model.columns]
    a_cols = [c for c in a_cols if c in df_model.columns]
    c_cols = [c for c in c_cols if c in df_model.columns]
    s_cols = [c for c in s_cols if c in df_model.columns]

    if target_col not in df_model.columns:
        logger.error(f"目标列 '{target_col}' 不在数值列中，退出")
        sys.exit(1)

    train_df, val_df, test_df = split_data(df_model, cfg, logger)

    # ── 5. Model 0：基线软测量 ────────────────────────────────────────────
    model0 = run_model0_baseline(
        train_df, val_df, test_df,
        all_feature_cols, target_col, cfg, logger,
    )

    # ── 6. Model 1：因果输入软测量 ────────────────────────────────────────
    model1 = run_model1_causal_input(
        train_df, val_df, test_df,
        a_cols, s_cols, target_col, cfg, logger,
    )

    # ── 7. Model 2：DML 残差软测量 ────────────────────────────────────────
    model2 = run_model2_dml_residual(
        train_df, val_df, test_df,
        c_cols, a_cols, s_cols, target_col, cfg, logger, output_dir,
    )

    # ── 8. 保存输出 ──────────────────────────────────────────────────────
    save_outputs(output_dir, roles_df, model0, model1, model2, logger)

    # ── 9. 最终日志摘要 ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("运行完成。输出文件：")
    for f in sorted(output_dir.iterdir()):
        logger.info(f"  {f}")
    logger.info("=" * 60)
    logger.info("DML 残差软测量区别于普通 LSTM 软测量：")
    logger.info("  普通 LSTM: X_seq -> y")
    logger.info("  DML 残差:  C->y_base + [A_res,S_res]_seq->y_res, y_hat=y_base+y_res_hat")
    logger.info("  核心优势: 消除工况变量 C 对 A/S 信号的混杂，使 LSTM 专注于因果相关残差信号。")


if __name__ == "__main__":
    main()
