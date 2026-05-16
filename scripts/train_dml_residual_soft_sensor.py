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
  baseline_predictions_test.csv
  as_lstm_predictions_test.csv
  dml_effect_weights.csv
  dml_effect_weight_predictions_test.csv
  residual_feature_summary.csv
  y_baseline_predictions.csv
  dml_residual_predictions_test.csv
  metrics_compare.csv
  run_log.txt
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import logging
import os
import platform
import random
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

def set_seed(seed: int, deterministic: bool = True, logger: Optional[logging.Logger] = None) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch, *_ = _import_torch()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            try:
                torch.use_deterministic_algorithms(True)
            except Exception as e:
                if logger is not None:
                    logger.warning(f"启用 torch.use_deterministic_algorithms(True) 失败: {e}")
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
        "allow_synthetic_demo": False,
        "external_dml_effect_dir": "casual_soft_sensor\\dml_causal_effect_value\\结果\\20260516_062800",
        "dml_effect_variable_col": None,
        "dml_effect_value_col": None,
        "dml_weight_clip_min": 0.3,
        "dml_weight_clip_max": 3.0,
        "state_weight_default": 1.0,
        "missing_effect_weight": 1.0,
        # ── 新增：人工 DML 权重配置 ──────────────────────────────────────────
        "manual_dml_weight_path": None,
        "dml_weight_use_manual_selected": False,
        "dml_weight_value_col": "theta_std",
        "dml_weight_variable_col": "resolved_treatment",
        "dml_weight_recommended_col": "recommended_for_weight",
        "constraints_enabled": True,
        "use_counterfactual_constraint": False,
        "counterfactual_lambda": 0.0,
        "counterfactual_delta_std": 0.05,
        "counterfactual_apply_to": "operation_A_only",
        "counterfactual_min_abs_effect": 0.0,
        "counterfactual_require_recommended": True,
        "counterfactual_use_last_step_only": True,
        "counterfactual_default_lag": 0,
        "use_process_constraint": False,
        "process_lambda": 0.0,
        "process_delta_std": 0.05,
        "process_require_dml_agree": True,
        "process_use_in_train_default": False,
        "process_use_in_eval_default": True,
        "process_constraints": [],
        "deterministic": True,
        "save_run_manifest": True,
    }
    for k, v in defaults.items():
        cfg.setdefault(k, v)
    return cfg


def _parse_bool_cli(value: Optional[int]) -> Optional[bool]:
    if value is None:
        return None
    return bool(int(value))


def apply_cli_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(cfg)
    if args.use_counterfactual_constraint is not None:
        cfg["use_counterfactual_constraint"] = _parse_bool_cli(args.use_counterfactual_constraint)
    if args.counterfactual_lambda is not None:
        cfg["counterfactual_lambda"] = float(args.counterfactual_lambda)
    if args.use_process_constraint is not None:
        cfg["use_process_constraint"] = _parse_bool_cli(args.use_process_constraint)
    if args.process_lambda is not None:
        cfg["process_lambda"] = float(args.process_lambda)
    if args.random_seed is not None:
        cfg["random_seed"] = int(args.random_seed)
    if args.output_dir:
        cfg["output_dir"] = str(args.output_dir)
    if args.run_name:
        cfg["run_name"] = str(args.run_name)
    return cfg


# ─── 数据加载 ─────────────────────────────────────────────────────────────────

def load_data(cfg: dict, logger: logging.Logger) -> pd.DataFrame:
    data_path = Path(cfg["data_path"])
    if data_path.exists():
        logger.info(f"读取数据文件: {data_path}")
        suffix = data_path.suffix.lower()
        if suffix == ".parquet":
            df = pd.read_parquet(data_path)
        elif suffix == ".csv":
            df = pd.read_csv(data_path)
        elif suffix == ".tsv":
            df = pd.read_csv(data_path, sep="\t")
        else:
            raise ValueError(f"不支持的数据格式: {data_path.suffix}")
        logger.info(f"数据形状: {df.shape}")
        return df
    else:
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

    # Bug 2 修复：operation_vars 留空时发出警告，当前版本不支持真实数据下自动推断
    if not op_vars:
        logger.warning(
            "operation_vars 为空，as_lstm / dml_effect_weight_lstm / dml_residual_lstm"
            " 的 A 列将为空，请在配置中手工填写操作变量名。"
        )

    # Bug 3 修复：只自动跳过 datetime 类型列；object 列保留在角色表并标为 excluded
    datetime_cols = set(df.select_dtypes(include=["datetime64"]).columns.tolist())
    object_cols = set(df.select_dtypes(include=["object"]).columns.tolist()) - datetime_cols

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

        # Bug 3 修复：object 列明确标为 excluded，不再无声消失
        if col in object_cols:
            rows.append({"variable": col, "role": "excluded", "reason": "非数值列，未参与建模"})
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
    total_ratio = cfg["train_ratio"] + cfg["val_ratio"] + cfg["test_ratio"]
    if not np.isclose(total_ratio, 1.0, atol=1e-4):
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio 之和应为 1.0，当前为 {total_ratio:.4f}"
        )
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
    if window_size <= 0:
        raise ValueError(f"window_size={window_size} 非法，必须为正整数。")
    n = len(y)
    if window_size > n:
        raise ValueError(
            f"window_size={window_size} 大于数据长度 {n}，无法构造任何窗口。"
            "请减小 window_size 或增加数据量。"
        )
    xs, ys = [], []
    for i in range(n - window_size + 1):
        xs.append(X[i: i + window_size])
        ys.append(y[i + window_size - 1])
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def build_train_stats(Xw_train: np.ndarray, feature_cols: List[str]) -> Dict[str, dict]:
    stats: Dict[str, dict] = {}
    if Xw_train.size == 0:
        return stats
    for j, col in enumerate(feature_cols):
        vals = Xw_train[:, :, j].reshape(-1)
        if vals.size == 0:
            continue
        stats[col] = {
            "q10": float(np.quantile(vals, 0.10)),
            "q30": float(np.quantile(vals, 0.30)),
            "q70": float(np.quantile(vals, 0.70)),
            "q90": float(np.quantile(vals, 0.90)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return stats


def _normalize_bool(v: Any, default: bool = False) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return default


def _infer_dml_sign_map(dml_effect_df: pd.DataFrame) -> Dict[str, dict]:
    dml_sign_map: Dict[str, dict] = {}
    if dml_effect_df is None or dml_effect_df.empty:
        return dml_sign_map
    for _, r in dml_effect_df.iterrows():
        var = str(r.get("variable", "")).strip()
        if not var:
            continue
        eff = pd.to_numeric(pd.Series([r.get("theta_std", np.nan)]), errors="coerce").iloc[0]
        if pd.isna(eff) or not np.isfinite(float(eff)):
            continue
        sign = int(np.sign(float(eff)))
        if sign == 0:
            continue
        dml_sign_map[var] = {
            "sign": sign,
            "abs_effect": float(abs(float(eff))),
            "recommended_for_weight": _normalize_bool(r.get("recommended_for_weight", False), default=False),
            "selected_lag_min": r.get("selected_lag_min", np.nan),
        }
    return dml_sign_map


def _resolve_rule_lag(raw_lag: Any, default_lag: int = 0) -> int:
    try:
        lag = int(float(raw_lag))
        return max(0, lag)
    except Exception:
        return max(0, int(default_lag))


def build_counterfactual_rules(
    cfg: dict,
    feature_cols: List[str],
    a_cols: List[str],
    dml_effect_df: pd.DataFrame,
) -> List[dict]:
    dml_sign_map = _infer_dml_sign_map(dml_effect_df)
    require_recommended = bool(cfg.get("counterfactual_require_recommended", True))
    min_abs_effect = float(cfg.get("counterfactual_min_abs_effect", 0.0))
    apply_to = str(cfg.get("counterfactual_apply_to", "operation_A_only"))
    default_lag = int(cfg.get("counterfactual_default_lag", 0))
    use_last_step_only = bool(cfg.get("counterfactual_use_last_step_only", True))
    delta_std = float(cfg.get("counterfactual_delta_std", 0.05))

    if apply_to == "operation_A_only":
        allowed = set(a_cols)
    else:
        allowed = set(feature_cols)

    rules: List[dict] = []
    for col in feature_cols:
        if col not in allowed:
            continue
        dml_info = dml_sign_map.get(col)
        if dml_info is None:
            continue
        if require_recommended and not dml_info.get("recommended_for_weight", False):
            continue
        if float(dml_info.get("abs_effect", 0.0)) < min_abs_effect:
            continue
        lag = _resolve_rule_lag(dml_info.get("selected_lag_min", default_lag), default_lag=default_lag)
        if use_last_step_only:
            lag = 0
        rules.append({
            "rule_name": f"cf_{col}",
            "variable": col,
            "direction": int(dml_info["sign"]),
            "lag": int(lag),
            "delta_std": delta_std,
            "rule_weight": float(dml_info.get("abs_effect", 1.0)),
            "use_in_train": True,
            "use_in_eval": True,
            "status": "trainable",
            "reason": "",
        })
    return rules


def _process_active_region_to_thresholds(active_region: dict, train_stats: Dict[str, dict], variable: str) -> dict:
    region = active_region if isinstance(active_region, dict) else {"type": "all"}
    rtype = str(region.get("type", "all"))
    out = {"type": rtype}
    st = train_stats.get(variable, {})
    q_grid = np.array([0.10, 0.30, 0.70, 0.90], dtype=np.float64)
    v_grid = np.array([
        st.get("q10", st.get("min", -1.0)),
        st.get("q30", st.get("q10", 0.0)),
        st.get("q70", st.get("q90", 1.0)),
        st.get("q90", st.get("max", 2.0)),
    ], dtype=np.float64)
    if np.any(~np.isfinite(v_grid)):
        v_grid = np.array([st.get("min", -1.0), st.get("q30", 0.0), st.get("q70", 1.0), st.get("max", 2.0)], dtype=np.float64)
    def _interp(q: float) -> float:
        return float(np.interp(np.clip(q, 0.0, 1.0), q_grid, v_grid))

    if rtype == "quantile_high":
        q = float(region.get("q", 0.70))
        out["min"] = _interp(q)
    elif rtype == "quantile_low":
        q = float(region.get("q", 0.30))
        out["max"] = _interp(q)
    elif rtype == "quantile_range":
        low = float(region.get("low", 0.10))
        high = float(region.get("high", 0.90))
        out["min"] = _interp(low)
        out["max"] = _interp(high)
    elif rtype == "value_range":
        out["min"] = region.get("min", None)
        out["max"] = region.get("max", None)
    return out


def _active_region_mask_np(values: np.ndarray, region: dict) -> np.ndarray:
    rtype = str((region or {}).get("type", "all"))
    if rtype == "all":
        return np.ones_like(values, dtype=bool)
    if rtype in {"quantile_high", "quantile_low", "quantile_range", "value_range"}:
        min_v = region.get("min", None)
        max_v = region.get("max", None)
        mask = np.ones_like(values, dtype=bool)
        if min_v is not None:
            mask = mask & (values >= float(min_v))
        if max_v is not None:
            mask = mask & (values <= float(max_v))
        return mask
    return np.ones_like(values, dtype=bool)


def screen_process_rules(
    cfg: dict,
    feature_cols: List[str],
    dml_effect_df: pd.DataFrame,
    train_stats: Dict[str, dict],
) -> pd.DataFrame:
    raw_rules = cfg.get("process_constraints", []) or []
    dml_sign_map = _infer_dml_sign_map(dml_effect_df)
    default_require = bool(cfg.get("process_require_dml_agree", True))
    default_lambda = float(cfg.get("process_lambda", 0.0))
    default_delta = float(cfg.get("process_delta_std", 0.05))
    default_train = bool(cfg.get("process_use_in_train_default", False))
    default_eval = bool(cfg.get("process_use_in_eval_default", True))
    rows: List[dict] = []

    for i, r in enumerate(raw_rules):
        rule = dict(r) if isinstance(r, dict) else {}
        name = str(rule.get("name", f"process_rule_{i}"))
        variable = str(rule.get("variable", "")).strip()
        direction = int(rule.get("direction", 0)) if str(rule.get("direction", "")).strip() else 0
        lag = _resolve_rule_lag(rule.get("lag", 0), default_lag=0)
        require_dml = _normalize_bool(rule.get("require_dml_agree", default_require), default=default_require)
        min_abs_dml = float(rule.get("min_abs_dml_effect", 0.0))
        use_in_train = _normalize_bool(rule.get("use_in_train", default_train), default=default_train)
        use_in_eval = _normalize_bool(rule.get("use_in_eval", default_eval), default=default_eval)
        status, reason = "trainable", ""
        dml_sign = None
        dml_abs = None

        if variable not in feature_cols:
            status, reason = "rejected", "variable_missing"
        elif direction not in (-1, 1):
            status, reason = "rejected", "invalid_direction"
        else:
            dml_info = dml_sign_map.get(variable)
            if dml_info is not None:
                dml_sign = int(dml_info["sign"])
                dml_abs = float(dml_info["abs_effect"])
            if require_dml:
                if dml_info is None:
                    status, reason = "eval_only", "missing_dml_effect"
                    use_in_train = False
                elif dml_abs is not None and dml_abs < min_abs_dml:
                    status, reason = "eval_only", "dml_effect_too_small"
                    use_in_train = False
                elif dml_sign != direction:
                    status, reason = "eval_only", "dml_direction_conflict"
                    use_in_train = False

        active_region = _process_active_region_to_thresholds(
            rule.get("active_region", {"type": "all"}),
            train_stats=train_stats,
            variable=variable,
        )
        rows.append({
            "rule_name": name,
            "variable": variable,
            "direction": direction,
            "lag": lag,
            "delta_std": float(rule.get("delta_std", default_delta)),
            "rule_lambda": float(rule.get("lambda", default_lambda)),
            "require_dml_agree": require_dml,
            "min_abs_dml_effect": min_abs_dml,
            "use_in_train": bool(use_in_train),
            "use_in_eval": bool(use_in_eval),
            "status": status,
            "reason": reason,
            "dml_sign": dml_sign,
            "dml_abs_effect": dml_abs,
            "active_region": json.dumps(active_region, ensure_ascii=False),
        })

    return pd.DataFrame(rows)


def evaluate_counterfactual_violations(
    model: "LSTMRegressor",
    X_eval: np.ndarray,
    counterfactual_rules: List[dict],
    feature_cols: List[str],
) -> pd.DataFrame:
    rows: List[dict] = []
    if X_eval.size == 0:
        return pd.DataFrame(rows)
    y_original = model.predict(X_eval)
    window_size = X_eval.shape[1]
    feat_idx = {c: i for i, c in enumerate(feature_cols)}
    for rule in counterfactual_rules:
        var = rule.get("variable")
        direction = int(rule.get("direction", 0))
        lag = _resolve_rule_lag(rule.get("lag", 0), 0)
        time_idx = window_size - 1 - lag
        use_in_eval = bool(rule.get("use_in_eval", True))
        status = str(rule.get("status", "trainable"))
        reason = str(rule.get("reason", ""))
        if not use_in_eval:
            rows.append({
                "rule_name": rule.get("rule_name", f"cf_{var}"),
                "variable": var,
                "direction": direction,
                "lag": lag,
                "effective_n": 0,
                "violation_rate": np.nan,
                "mean_violation_magnitude": np.nan,
                "mean_expected_response": np.nan,
                "use_in_train": bool(rule.get("use_in_train", True)),
                "use_in_eval": use_in_eval,
                "status": "disabled",
                "reason": "use_in_eval_false",
            })
            continue
        if (var not in feat_idx) or (direction not in (-1, 1)) or (time_idx < 0 or time_idx >= window_size):
            rows.append({
                "rule_name": rule.get("rule_name", f"cf_{var}"),
                "variable": var,
                "direction": direction,
                "lag": lag,
                "effective_n": 0,
                "violation_rate": np.nan,
                "mean_violation_magnitude": np.nan,
                "mean_expected_response": np.nan,
                "use_in_train": bool(rule.get("use_in_train", True)),
                "use_in_eval": use_in_eval,
                "status": "rejected" if status == "trainable" else status,
                "reason": reason or "invalid_rule_or_lag",
            })
            continue

        X_cf = X_eval.copy()
        X_cf[:, time_idx, feat_idx[var]] += float(rule.get("delta_std", 0.05))
        delta = model.predict(X_cf) - y_original
        expected = direction * delta
        violation = np.maximum(-expected, 0.0)
        rows.append({
            "rule_name": rule.get("rule_name", f"cf_{var}"),
            "variable": var,
            "direction": direction,
            "lag": lag,
            "effective_n": int(len(delta)),
            "violation_rate": float(np.mean(expected < 0)) if len(expected) > 0 else np.nan,
            "mean_violation_magnitude": float(np.mean(violation)) if len(violation) > 0 else np.nan,
            "mean_expected_response": float(np.mean(expected)) if len(expected) > 0 else np.nan,
            "use_in_train": bool(rule.get("use_in_train", True)),
            "use_in_eval": use_in_eval,
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def evaluate_process_violations(
    model: "LSTMRegressor",
    X_eval: np.ndarray,
    process_rules_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows: List[dict] = []
    if X_eval.size == 0:
        return pd.DataFrame(rows)
    if process_rules_df is None or process_rules_df.empty:
        return pd.DataFrame(rows)
    y_original = model.predict(X_eval)
    window_size = X_eval.shape[1]
    feat_idx = {c: i for i, c in enumerate(feature_cols)}
    for _, rr in process_rules_df.iterrows():
        rule = rr.to_dict()
        var = str(rule.get("variable", ""))
        direction = int(rule.get("direction", 0))
        lag = _resolve_rule_lag(rule.get("lag", 0), 0)
        time_idx = window_size - 1 - lag
        region = json.loads(rule.get("active_region", "{}") or "{}")
        status = str(rule.get("status", "trainable"))
        reason = str(rule.get("reason", ""))
        if not bool(rule.get("use_in_eval", True)):
            rows.append({
                "rule_name": rule.get("rule_name", f"process_{var}"),
                "variable": var,
                "direction": direction,
                "lag": lag,
                "effective_n": 0,
                "violation_rate": np.nan,
                "mean_violation_magnitude": np.nan,
                "mean_expected_response": np.nan,
                "use_in_train": bool(rule.get("use_in_train", False)),
                "use_in_eval": False,
                "status": "disabled",
                "reason": "use_in_eval_false",
            })
            continue
        if var not in feat_idx or direction not in (-1, 1) or (time_idx < 0 or time_idx >= window_size):
            rows.append({
                "rule_name": rule.get("rule_name", f"process_{var}"),
                "variable": var,
                "direction": direction,
                "lag": lag,
                "effective_n": 0,
                "violation_rate": np.nan,
                "mean_violation_magnitude": np.nan,
                "mean_expected_response": np.nan,
                "use_in_train": bool(rule.get("use_in_train", False)),
                "use_in_eval": bool(rule.get("use_in_eval", True)),
                "status": "rejected" if status == "trainable" else status,
                "reason": reason or "invalid_rule_or_lag",
            })
            continue
        v = X_eval[:, time_idx, feat_idx[var]]
        mask = _active_region_mask_np(v, region)
        effective_n = int(mask.sum())
        if effective_n == 0:
            rows.append({
                "rule_name": rule.get("rule_name", f"process_{var}"),
                "variable": var,
                "direction": direction,
                "lag": lag,
                "effective_n": 0,
                "violation_rate": np.nan,
                "mean_violation_magnitude": np.nan,
                "mean_expected_response": np.nan,
                "use_in_train": bool(rule.get("use_in_train", False)),
                "use_in_eval": bool(rule.get("use_in_eval", True)),
                "status": status,
                "reason": reason or "no_active_samples",
            })
            continue
        X_proc = X_eval.copy()
        X_proc[mask, time_idx, feat_idx[var]] += float(rule.get("delta_std", 0.05))
        delta = model.predict(X_proc) - y_original
        expected = direction * delta[mask]
        violation = np.maximum(-expected, 0.0)
        rows.append({
            "rule_name": rule.get("rule_name", f"process_{var}"),
            "variable": var,
            "direction": direction,
            "lag": lag,
            "effective_n": effective_n,
            "violation_rate": float(np.mean(expected < 0)) if len(expected) > 0 else np.nan,
            "mean_violation_magnitude": float(np.mean(violation)) if len(violation) > 0 else np.nan,
            "mean_expected_response": float(np.mean(expected)) if len(expected) > 0 else np.nan,
            "use_in_train": bool(rule.get("use_in_train", False)),
            "use_in_eval": bool(rule.get("use_in_eval", True)),
            "status": status,
            "reason": reason,
        })
    return pd.DataFrame(rows)


def _aggregate_violation_metrics(df: pd.DataFrame) -> Tuple[float, float]:
    if df is None or df.empty:
        return float("nan"), float("nan")
    valid = df[(df["effective_n"] > 0) & df["violation_rate"].notna()]
    if valid.empty:
        return float("nan"), float("nan")
    return float(valid["violation_rate"].mean()), float(valid["mean_violation_magnitude"].mean())


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
        self.loss_log_df = pd.DataFrame()

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
        constraint_context: Optional[dict] = None,
    ) -> "LSTMRegressor":
        torch, nn, DataLoader, TensorDataset = _import_torch()
        constraint_context = constraint_context or {}
        set_seed(
            self.seed,
            deterministic=bool(constraint_context.get("deterministic", True)),
            logger=logger,
        )
        model, device = self._build()
        self._device = device

        Xtr = torch.tensor(X_train, dtype=torch.float32)
        ytr = torch.tensor(y_train, dtype=torch.float32)
        dl_generator = torch.Generator()
        dl_generator.manual_seed(self.seed)
        loader = DataLoader(
            TensorDataset(Xtr, ytr),
            batch_size=self.batch_size,
            shuffle=True,
            generator=dl_generator,
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        relu = nn.ReLU()

        best_val_loss = float("inf")
        best_state = None
        no_improve = 0
        loss_rows: List[dict] = []

        feature_cols = constraint_context.get("feature_cols", [])
        window_size = X_train.shape[1] if X_train.ndim == 3 else 0
        feat_idx = {c: i for i, c in enumerate(feature_cols)}

        use_cf = bool(constraint_context.get("use_counterfactual_constraint", False))
        use_proc = bool(constraint_context.get("use_process_constraint", False))
        cf_rules_in = constraint_context.get("counterfactual_rules", [])
        if cf_rules_in is None:
            cf_rules_in = []
        proc_rules_in = constraint_context.get("process_rules", [])
        if proc_rules_in is None or (isinstance(proc_rules_in, pd.DataFrame) and proc_rules_in.empty):
            proc_rules_in = []
        cf_lambda = float(constraint_context.get("counterfactual_lambda", 0.0))
        process_lambda = float(constraint_context.get("process_lambda", 0.0))

        cf_rules = []
        for r in cf_rules_in:
            var = r.get("variable")
            lag = _resolve_rule_lag(r.get("lag", 0), 0)
            time_idx = window_size - 1 - lag
            if var in feat_idx and 0 <= time_idx < window_size and int(r.get("direction", 0)) in (-1, 1):
                rr = dict(r)
                rr["feature_idx"] = feat_idx[var]
                rr["time_idx"] = time_idx
                cf_rules.append(rr)

        proc_rules = []
        if isinstance(proc_rules_in, pd.DataFrame) and not proc_rules_in.empty:
            proc_iter = [row.to_dict() for _, row in proc_rules_in.iterrows()]
        elif isinstance(proc_rules_in, pd.DataFrame):
            proc_iter = []
        else:
            proc_iter = list(proc_rules_in) if proc_rules_in else []
        for r in proc_iter:
            var = r.get("variable")
            lag = _resolve_rule_lag(r.get("lag", 0), 0)
            time_idx = window_size - 1 - lag
            if var in feat_idx and 0 <= time_idx < window_size and int(r.get("direction", 0)) in (-1, 1):
                rr = dict(r)
                rr["feature_idx"] = feat_idx[var]
                rr["time_idx"] = time_idx
                try:
                    rr["active_region_parsed"] = json.loads(rr.get("active_region", "{}") or "{}")
                except Exception:
                    rr["active_region_parsed"] = {"type": "all"}
                proc_rules.append(rr)

        for epoch in range(1, self.epochs + 1):
            model.train()
            epoch_pred_loss = 0.0
            epoch_cf_loss = 0.0
            epoch_process_loss = 0.0
            epoch_total_loss = 0.0
            batches = 0
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                y_original = model(xb)
                pred_loss = criterion(y_original, yb)

                cf_loss = torch.tensor(0.0, dtype=pred_loss.dtype, device=device)
                if use_cf and cf_rules:
                    cf_terms = []
                    for r in cf_rules:
                        xcf = xb.clone()
                        xcf[:, r["time_idx"], r["feature_idx"]] += float(r.get("delta_std", 0.05))
                        y_cf = model(xcf)
                        delta_y = y_cf - y_original
                        direction = float(int(r.get("direction", 1)))
                        violation = -(direction * delta_y)
                        rule_loss = torch.mean(relu(violation))
                        weight = float(r.get("rule_weight", 1.0))
                        cf_terms.append(rule_loss * weight)
                    if cf_terms:
                        cf_loss = torch.stack(cf_terms).mean()

                process_loss = torch.tensor(0.0, dtype=pred_loss.dtype, device=device)
                if use_proc and proc_rules:
                    proc_terms = []
                    for r in proc_rules:
                        if str(r.get("status", "trainable")) != "trainable":
                            continue
                        if not bool(r.get("use_in_train", False)):
                            continue
                        values = xb[:, r["time_idx"], r["feature_idx"]]
                        region = r.get("active_region_parsed", {"type": "all"})
                        mask = torch.ones_like(values, dtype=torch.bool, device=device)
                        min_v = region.get("min", None)
                        max_v = region.get("max", None)
                        if min_v is not None:
                            mask = mask & (values >= float(min_v))
                        if max_v is not None:
                            mask = mask & (values <= float(max_v))
                        if torch.sum(mask).item() <= 0:
                            continue
                        x_proc = xb.clone()
                        x_proc[mask, r["time_idx"], r["feature_idx"]] += float(r.get("delta_std", 0.05))
                        y_proc = model(x_proc)
                        delta_y = y_proc - y_original
                        direction = float(int(r.get("direction", 1)))
                        violation = -(direction * delta_y[mask])
                        rule_loss = torch.mean(relu(violation))
                        proc_terms.append(rule_loss * float(r.get("rule_lambda", 0.0)))
                    if proc_terms:
                        process_loss = torch.stack(proc_terms).mean()

                total_loss = pred_loss + (cf_lambda * cf_loss) + (process_lambda * process_loss)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_pred_loss += float(pred_loss.item())
                epoch_cf_loss += float(cf_loss.item())
                epoch_process_loss += float(process_loss.item())
                epoch_total_loss += float(total_loss.item())
                batches += 1

            val_loss = float("nan")
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
                        denom = max(1, batches)
                        loss_rows.append({
                            "epoch": epoch,
                            "train_pred_loss": epoch_pred_loss / denom,
                            "train_cf_loss": epoch_cf_loss / denom,
                            "train_process_loss": epoch_process_loss / denom,
                            "train_total_loss": epoch_total_loss / denom,
                            "val_loss": val_loss,
                        })
                        break
            denom = max(1, batches)
            loss_rows.append({
                "epoch": epoch,
                "train_pred_loss": epoch_pred_loss / denom,
                "train_cf_loss": epoch_cf_loss / denom,
                "train_process_loss": epoch_process_loss / denom,
                "train_total_loss": epoch_total_loss / denom,
                "val_loss": val_loss,
            })

        if best_state is not None:
            model.load_state_dict(best_state)
        self._model = model
        self.loss_log_df = pd.DataFrame(loss_rows)
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
    model.fit(
        Xw_tr,
        yw_tr,
        Xw_vl,
        yw_vl,
        logger=logger,
        constraint_context={"deterministic": bool(cfg.get("deterministic", True))},
    )

    y_pred_scaled = model.predict(Xw_te)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Model 0 Test: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # 保存预测结果
    predictions_test = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    return {
        "model_name": "baseline_all_lstm",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "lstm": model,  # 返回LSTM模型对象
        "as_scaler": feat_scaler,  # 返回特征scaler
        "y_scaler": y_scaler,  # 返回目标scaler
        "feature_cols": all_feature_cols,  # 返回特征列名
        "predictions_test": predictions_test,  # 返回预测结果
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
            "model_name": "as_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "lstm": None,
            "as_scaler": None,
            "y_scaler": None,
            "feature_cols": [],
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
    model.fit(
        Xw_tr,
        yw_tr,
        Xw_vl,
        yw_vl,
        logger=logger,
        constraint_context={"deterministic": bool(cfg.get("deterministic", True))},
    )

    y_pred_scaled = model.predict(Xw_te)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()

    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Model 1 Test: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    # 保存预测结果
    predictions_test = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
    })

    return {
        "model_name": "as_lstm",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "lstm": model,  # 返回LSTM模型对象
        "as_scaler": feat_scaler,  # 返回特征scaler
        "y_scaler": y_scaler,  # 返回目标scaler
        "feature_cols": causal_cols,  # 返回特征列名
        "predictions_test": predictions_test,  # 返回预测结果
    }


# ─── Model 2：DML 效应权重软测量 ──────────────────────────────────────────────

def _resolve_external_effect_dir(effect_dir_cfg: str) -> Path:
    effect_dir = Path(str(effect_dir_cfg).replace("\\", os.sep))
    if effect_dir.exists():
        return effect_dir
    repo_root = Path(__file__).resolve().parents[1]
    alt_effect_dir = repo_root / str(effect_dir_cfg).replace("\\", os.sep)
    if alt_effect_dir.exists():
        return alt_effect_dir
    return effect_dir


def _read_effect_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"不支持的效应文件格式: {path}")


def _load_external_dml_effects(
    cfg: dict,
    a_cols: List[str],
    logger: logging.Logger,
) -> Tuple[Dict[str, float], Optional[Path], Optional[str], Optional[str]]:
    effect_dir = _resolve_external_effect_dir(cfg["external_dml_effect_dir"])
    if not effect_dir.exists():
        logger.warning(f"外部 DML 效应目录不存在: {effect_dir}")
        return {}, None, None, None

    candidates = sorted(
        [p for p in effect_dir.rglob("*") if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls", ".parquet"}]
    )
    if not candidates:
        logger.warning(f"外部 DML 效应目录中未找到表格文件: {effect_dir}")
        return {}, None, None, None

    var_candidates = ["variable", "treatment", "treatment_var", "operation_var", "treatment_variable", "变量名", "变量"]
    effect_candidates = ["effect", "theta", "causal_effect", "effect_value", "ate", "dml效应系数", "效应系数", "dml_effect"]

    best = None
    a_cols_lc = {c.lower(): c for c in a_cols}
    preferred_var = cfg.get("dml_effect_variable_col")
    preferred_effect = cfg.get("dml_effect_value_col")

    for fp in candidates:
        try:
            df = _read_effect_table(fp)
        except Exception as e:
            logger.warning(f"读取效应文件失败，跳过: {fp} | {e}")
            continue
        if df.empty:
            logger.warning(f"效应文件为空，跳过: {fp}")
            continue

        cols = list(df.columns)
        cols_lc = {str(c).strip().lower(): str(c) for c in cols}
        var_col = None
        effect_col = None

        if preferred_var is not None:
            var_col = cols_lc.get(str(preferred_var).strip().lower())
        if preferred_effect is not None:
            effect_col = cols_lc.get(str(preferred_effect).strip().lower())
        if var_col is None:
            for c in var_candidates:
                if c in cols_lc:
                    var_col = cols_lc[c]
                    break
        if effect_col is None:
            for c in effect_candidates:
                if c in cols_lc:
                    effect_col = cols_lc[c]
                    break

        if var_col is None or effect_col is None:
            logger.info(f"效应文件列不匹配，文件={fp}，可用列={cols}")
            continue

        tmp = df[[var_col, effect_col]].copy()
        tmp[var_col] = tmp[var_col].astype(str).str.strip()
        tmp[effect_col] = pd.to_numeric(tmp[effect_col], errors="coerce")
        tmp = tmp.dropna(subset=[var_col, effect_col])
        if tmp.empty:
            logger.info(f"效应文件无有效数据，文件={fp}，可用列={cols}")
            continue

        match_count = int(tmp[var_col].str.lower().isin(a_cols_lc.keys()).sum())
        score = (match_count, len(tmp))
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "path": fp,
                "df": tmp,
                "var_col": var_col,
                "effect_col": effect_col,
                "available_cols": cols,
            }

    if best is None:
        logger.warning(f"未找到可用 DML 效应表格文件，目录={effect_dir}")
        return {}, None, None, None

    logger.info(f"找到的 DML effect 文件路径: {best['path']}")
    logger.info(f"使用的 DML 列名: variable_col={best['var_col']}, effect_col={best['effect_col']}")

    grouped = best["df"].groupby(best["var_col"], as_index=False)[best["effect_col"]].mean()
    effect_dict = dict(zip(grouped[best["var_col"]].astype(str), grouped[best["effect_col"]].astype(float)))
    return effect_dict, best["path"], best["var_col"], best["effect_col"]


def _load_manual_dml_weights(
    cfg: dict,
    a_cols: List[str],
    logger: logging.Logger,
) -> Optional[pd.DataFrame]:
    """
    读取 manual_dml_theta_selected_for_weight.csv，返回 DataFrame 或 None。

    文件必须包含列：
      resolved_treatment, theta_std, recommended_for_weight
    可选列：treatment_group, selected_lag_min, reason

    若文件不存在或无法解析，发出 warning 并返回 None（调用方应回退为默认权重 1.0）。
    """
    raw_path = cfg.get("manual_dml_weight_path")
    if not raw_path:
        logger.warning(
            "manual_dml_weight_path 未配置，无法读取人工 DML 结果。"
            " Model 2 将回退为默认权重 1.0。"
        )
        return None

    weight_path = Path(str(raw_path).replace("\\", os.sep))
    if not weight_path.is_absolute():
        repo_root = Path(__file__).resolve().parents[1]
        alt = repo_root / weight_path
        if alt.exists():
            weight_path = alt

    if not weight_path.exists():
        logger.warning(
            f"manual_dml_weight_path 指定的文件不存在: {weight_path}。"
            " Model 2 将回退为默认权重 1.0，请检查路径或先运行人工 DML 批量估计脚本。"
        )
        return None

    logger.info(f"[Model 2] 读取人工 DML 权重文件: {weight_path}")

    try:
        df = pd.read_csv(weight_path)
    except Exception as e:
        logger.warning(f"读取人工 DML 权重文件失败: {e}。Model 2 将回退为默认权重 1.0。")
        return None

    logger.info(f"[Model 2] 文件列名: {df.columns.tolist()}")

    var_col = cfg.get("dml_weight_variable_col", "resolved_treatment")
    val_col = cfg.get("dml_weight_value_col", "theta_std")
    rec_col = cfg.get("dml_weight_recommended_col", "recommended_for_weight")

    missing_required = [c for c in [var_col, val_col, rec_col] if c not in df.columns]
    if missing_required:
        logger.warning(
            f"人工 DML 权重文件缺少必需列: {missing_required}。"
            " Model 2 将回退为默认权重 1.0。"
        )
        return None

    df[var_col] = df[var_col].astype(str).str.strip()
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")
    df[rec_col] = df[rec_col].astype(str).str.strip().str.lower().map(
        {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
    ).fillna(False).astype(bool)

    return df


def build_dml_effect_weights_table(
    cfg: dict,
    a_cols: List[str],
    s_cols: List[str],
    logger: logging.Logger,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Build the DML effect weight table without training Model 2.

    Model 3 uses this table to derive counterfactual and process constraints.
    Keeping it independent prevents --only-model3 from silently disabling those
    constraints when Model 2 is skipped.
    """
    use_manual = bool(cfg.get("dml_weight_use_manual_selected", False))
    clip_min = float(cfg.get("dml_weight_clip_min", 0.3))
    clip_max = float(cfg.get("dml_weight_clip_max", 3.0))
    state_weight = float(cfg.get("state_weight_default", 1.0))
    missing_weight = float(cfg.get("missing_effect_weight", 1.0))

    var_col = cfg.get("dml_weight_variable_col", "resolved_treatment")
    val_col = cfg.get("dml_weight_value_col", "theta_std")
    rec_col = cfg.get("dml_weight_recommended_col", "recommended_for_weight")

    rows: List[dict] = []
    raw_weights: Dict[str, float] = {}

    if use_manual:
        manual_df = _load_manual_dml_weights(cfg, a_cols, logger)
        if manual_df is not None:
            manual_map = {
                str(r[var_col]).strip().lower(): r
                for _, r in manual_df.iterrows()
                if pd.notna(r[var_col])
            }

            matched_a: List[str] = []
            missing_a: List[str] = []
            not_recommended_a: List[str] = []
            recommended_a: List[str] = []
            recommended_abs: List[float] = []

            for col in a_cols:
                match_row = manual_map.get(col.lower())
                if match_row is None:
                    missing_a.append(col)
                    rows.append({
                        "variable": col,
                        "role": "operation_A",
                        "matched_dml_variable": np.nan,
                        "treatment_group": np.nan,
                        "selected_lag_min": np.nan,
                        "theta_std": np.nan,
                        "recommended_for_weight": False,
                        "raw_weight": float(missing_weight),
                        "final_weight": float(np.clip(missing_weight, clip_min, clip_max)),
                        "weight_source": "missing_manual_theta_default_1",
                        "reason": "no matched resolved_treatment in manual DML file",
                    })
                    continue

                matched_a.append(col)
                theta_std = match_row.get(val_col, np.nan)
                theta_std_f = float(theta_std) if pd.notna(theta_std) else np.nan
                is_recommended = bool(match_row.get(rec_col, False))
                treatment_group = match_row.get("treatment_group", np.nan)
                selected_lag_min = match_row.get("selected_lag_min", np.nan)
                reason = match_row.get("reason", "")

                if is_recommended and pd.notna(theta_std_f) and np.isfinite(theta_std_f):
                    recommended_a.append(col)
                    recommended_abs.append(abs(theta_std_f))
                    rows.append({
                        "variable": col,
                        "role": "operation_A",
                        "matched_dml_variable": str(match_row[var_col]),
                        "treatment_group": treatment_group,
                        "selected_lag_min": selected_lag_min,
                        "theta_std": theta_std_f,
                        "recommended_for_weight": True,
                        "raw_weight": np.nan,
                        "final_weight": np.nan,
                        "weight_source": "manual_theta_std_recommended",
                        "reason": reason,
                    })
                else:
                    not_recommended_a.append(col)
                    rows.append({
                        "variable": col,
                        "role": "operation_A",
                        "matched_dml_variable": str(match_row[var_col]),
                        "treatment_group": treatment_group,
                        "selected_lag_min": selected_lag_min,
                        "theta_std": theta_std_f,
                        "recommended_for_weight": False,
                        "raw_weight": float(missing_weight),
                        "final_weight": float(np.clip(missing_weight, clip_min, clip_max)),
                        "weight_source": "manual_theta_std_not_recommended_default_1",
                        "reason": reason if reason else "recommended_for_weight is False or theta_std invalid",
                    })

            logger.info(f"[DML weights] matched A variables: {len(matched_a)}/{len(a_cols)}")
            logger.info(f"[DML weights] recommended_for_weight==true: {len(recommended_a)}")
            if not_recommended_a:
                logger.info(f"[DML weights] matched but not recommended A variables: {not_recommended_a}")
            if missing_a:
                logger.warning(f"[DML weights] missing A variables, default weight=1.0: {missing_a}")

            mean_abs = float(np.mean(recommended_abs)) if recommended_abs else 0.0
            if mean_abs <= 1e-12 and recommended_a:
                logger.warning("[DML weights] recommended theta_std mean abs is near zero; defaulting weights to 1.0")
                mean_abs = 0.0

            for row in rows:
                if row.get("weight_source") == "manual_theta_std_recommended":
                    if mean_abs > 1e-12:
                        rw = abs(float(row["theta_std"])) / mean_abs
                    else:
                        rw = float(missing_weight)
                    row["raw_weight"] = rw
                    row["final_weight"] = float(np.clip(rw, clip_min, clip_max))
                raw_weights[row["variable"]] = (
                    float(row["raw_weight"]) if pd.notna(row.get("raw_weight")) else float(missing_weight)
                )
        else:
            use_manual = False

    if not use_manual:
        effect_dict, _, _, _ = _load_external_dml_effects(cfg, a_cols, logger)
        effect_dict_lc = {str(k).strip().lower(): float(v) for k, v in effect_dict.items()}
        matched_a: List[str] = []
        missing_a: List[str] = []
        available_abs: List[float] = []

        for col in a_cols:
            eff = effect_dict_lc.get(col.lower())
            if eff is None or not np.isfinite(eff):
                missing_a.append(col)
                rows.append({
                    "variable": col,
                    "role": "operation_A",
                    "matched_dml_variable": np.nan,
                    "treatment_group": np.nan,
                    "selected_lag_min": np.nan,
                    "theta_std": np.nan,
                    "recommended_for_weight": False,
                    "raw_weight": float(missing_weight),
                    "final_weight": float(np.clip(missing_weight, clip_min, clip_max)),
                    "weight_source": "missing_manual_theta_default_1",
                    "reason": "no matched treatment in external DML effect dir",
                })
            else:
                matched_a.append(col)
                available_abs.append(abs(float(eff)))
                rows.append({
                    "variable": col,
                    "role": "operation_A",
                    "matched_dml_variable": col,
                    "treatment_group": np.nan,
                    "selected_lag_min": np.nan,
                    "theta_std": float(eff),
                    "recommended_for_weight": True,
                    "raw_weight": np.nan,
                    "final_weight": np.nan,
                    "weight_source": "external_dml_effect",
                    "reason": "",
                })

        mean_abs = float(np.mean(available_abs)) if available_abs else 0.0
        for row in rows:
            if row["role"] != "operation_A":
                continue
            if row["weight_source"] == "external_dml_effect" and mean_abs > 1e-12:
                row["raw_weight"] = float(abs(float(row["theta_std"])) / mean_abs)
            elif pd.isna(row.get("raw_weight", np.nan)):
                row["raw_weight"] = float(missing_weight)
            row["final_weight"] = float(np.clip(row["raw_weight"], clip_min, clip_max))
            raw_weights[row["variable"]] = float(row["raw_weight"])

        logger.info(f"[DML weights] matched external effect A variables: {len(matched_a)}/{len(a_cols)}")
        logger.info(f"[DML weights] missing external effect A variables: {missing_a}")

    for col in s_cols:
        rows.append({
            "variable": col,
            "role": "state_S",
            "matched_dml_variable": np.nan,
            "treatment_group": np.nan,
            "selected_lag_min": np.nan,
            "theta_std": np.nan,
            "recommended_for_weight": False,
            "raw_weight": float(state_weight),
            "final_weight": float(state_weight),
            "weight_source": "state_default_1",
            "reason": "state variable, weight fixed at 1",
        })
        raw_weights[col] = float(state_weight)

    weights_df = pd.DataFrame(rows, columns=[
        "variable", "role", "matched_dml_variable", "treatment_group", "selected_lag_min",
        "theta_std", "recommended_for_weight", "raw_weight", "final_weight", "weight_source", "reason",
    ])

    if output_dir is not None:
        weights_path = output_dir / "dml_effect_weights.csv"
        weights_df.to_csv(weights_path, index=False, encoding="utf-8-sig")
        logger.info(f"Saved DML effect weights table: {weights_path}")

    return weights_df


def run_model2_dml_effect_weight_lstm(
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
    """Model 2: A + S -> DML效应权重加权 -> LSTM -> y。"""
    logger.info("=" * 60)
    logger.info("Model 2: DML效应权重软测量 (A + S -> weighted -> LSTM -> y)")

    feature_cols = a_cols + s_cols
    if not feature_cols:
        logger.warning("A + S 为空，跳过 Model 2")
        return {
            "model_name": "dml_effect_weight_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "lstm": None,
            "as_scaler": None,
            "y_scaler": None,
            "feature_cols": [],
            "predictions_test": pd.DataFrame(),
            "dml_effect_weights": pd.DataFrame(),
            "counterfactual_rules": pd.DataFrame(),
            "process_rules": pd.DataFrame(),
            "constraint_loss_log": pd.DataFrame(),
            "counterfactual_metrics": pd.DataFrame(),
            "process_metrics": pd.DataFrame(),
            "use_counterfactual_constraint": False,
            "use_process_constraint": False,
        }

    window_size = cfg["window_size"]
    feat_scaler, y_scaler = fit_scalers(train_df, feature_cols, target_col)
    X_tr_scaled, y_tr = apply_scalers(train_df, feature_cols, target_col, feat_scaler, y_scaler)
    X_vl_scaled, y_vl = apply_scalers(val_df, feature_cols, target_col, feat_scaler, y_scaler)
    X_te_scaled, y_te = apply_scalers(test_df, feature_cols, target_col, feat_scaler, y_scaler)

    use_manual = bool(cfg.get("dml_weight_use_manual_selected", False))
    logger.info(f"[Model 2] 是否启用人工 DML 权重 (dml_weight_use_manual_selected): {use_manual}")

    clip_min = float(cfg.get("dml_weight_clip_min", 0.3))
    clip_max = float(cfg.get("dml_weight_clip_max", 3.0))
    state_weight = float(cfg.get("state_weight_default", 1.0))
    missing_weight = float(cfg.get("missing_effect_weight", 1.0))

    var_col = cfg.get("dml_weight_variable_col", "resolved_treatment")
    val_col = cfg.get("dml_weight_value_col", "theta_std")
    rec_col = cfg.get("dml_weight_recommended_col", "recommended_for_weight")

    rows: List[dict] = []
    raw_weights: Dict[str, float] = {}
    no_recommended_note = ""

    if use_manual:
        manual_df = _load_manual_dml_weights(cfg, a_cols, logger)
        if manual_df is not None:
            logger.info(
                f"[Model 2] 读取成功。A 变量数量: {len(a_cols)}，"
                f"权重文件行数: {len(manual_df)}"
            )
            # 构建 resolved_treatment -> row 映射（不区分大小写）
            manual_map = {
                str(r[var_col]).strip().lower(): r
                for _, r in manual_df.iterrows()
                if pd.notna(r[var_col])
            }

            matched_a: List[str] = []
            missing_a: List[str] = []
            not_recommended_a: List[str] = []
            recommended_a: List[str] = []
            recommended_abs: List[float] = []

            # 第一遍：确定匹配与 recommended 状态
            for col in a_cols:
                match_row = manual_map.get(col.lower())
                if match_row is None:
                    missing_a.append(col)
                    rows.append({
                        "variable": col,
                        "role": "operation_A",
                        "matched_dml_variable": np.nan,
                        "treatment_group": np.nan,
                        "selected_lag_min": np.nan,
                        "theta_std": np.nan,
                        "recommended_for_weight": False,
                        "raw_weight": float(missing_weight),
                        "final_weight": float(np.clip(missing_weight, clip_min, clip_max)),
                        "weight_source": "missing_manual_theta_default_1",
                        "reason": "no matched resolved_treatment in manual DML file",
                    })
                else:
                    matched_a.append(col)
                    theta_std = match_row.get(val_col, np.nan)
                    theta_std_f = float(theta_std) if pd.notna(theta_std) else np.nan
                    is_recommended = bool(match_row.get(rec_col, False))
                    treatment_group = match_row.get("treatment_group", np.nan)
                    selected_lag_min = match_row.get("selected_lag_min", np.nan)
                    reason = match_row.get("reason", "")

                    if is_recommended and pd.notna(theta_std_f) and np.isfinite(theta_std_f):
                        recommended_a.append(col)
                        recommended_abs.append(abs(theta_std_f))
                        rows.append({
                            "variable": col,
                            "role": "operation_A",
                            "matched_dml_variable": str(match_row[var_col]),
                            "treatment_group": treatment_group,
                            "selected_lag_min": selected_lag_min,
                            "theta_std": theta_std_f,
                            "recommended_for_weight": True,
                            "raw_weight": np.nan,  # 待第二遍填入
                            "final_weight": np.nan,
                            "weight_source": "manual_theta_std_recommended",
                            "reason": reason,
                        })
                    else:
                        not_recommended_a.append(col)
                        rows.append({
                            "variable": col,
                            "role": "operation_A",
                            "matched_dml_variable": str(match_row[var_col]),
                            "treatment_group": treatment_group,
                            "selected_lag_min": selected_lag_min,
                            "theta_std": theta_std_f,
                            "recommended_for_weight": False,
                            "raw_weight": float(missing_weight),
                            "final_weight": float(np.clip(missing_weight, clip_min, clip_max)),
                            "weight_source": "manual_theta_std_not_recommended_default_1",
                            "reason": reason if reason else "recommended_for_weight is False or theta_std invalid",
                        })

            # 日志
            logger.info(
                f"[Model 2] 成功匹配 A 变量数量: {len(matched_a)}/{len(a_cols)}"
            )
            logger.info(
                f"[Model 2] recommended_for_weight==true 的数量: {len(recommended_a)}"
            )
            if not_recommended_a:
                logger.info(
                    f"[Model 2] recommended_for_weight==false 的匹配变量: {not_recommended_a}"
                )
            if missing_a:
                logger.warning(
                    f"[Model 2] 未匹配到 theta_std 的 A 变量 (权重回退为 1.0): {missing_a}"
                )

            # 第二遍：对 recommended A 变量做均值归一化
            mean_abs = float(np.mean(recommended_abs)) if recommended_abs else 0.0
            if mean_abs <= 1e-12 and recommended_a:
                logger.warning(
                    "[Model 2] recommended A 变量 theta_std 绝对值均值接近 0，"
                    "A 变量权重全部回退为 1.0。"
                )
                mean_abs = 0.0

            if not recommended_a:
                logger.warning(
                    "[Model 2] 没有任何 A 变量满足 recommended_for_weight==true；"
                    " dml_effect_weight_lstm 退化为 as_lstm 风格的等权重。"
                )
                no_recommended_note = "no recommended theta_std matched; weights default to 1"
                # 将所有 recommended 行（此时为空）的 raw/final 设为 1
            else:
                theta_std_vals = [abs(r["theta_std"]) for r in rows if r.get("weight_source") == "manual_theta_std_recommended"]
                logger.info(
                    "[Model 2] theta_std 统计: min=%.6f, max=%.6f, mean=%.6f",
                    float(np.min(theta_std_vals)),
                    float(np.max(theta_std_vals)),
                    float(np.mean(theta_std_vals)),
                )

            for row in rows:
                if row.get("weight_source") == "manual_theta_std_recommended":
                    if mean_abs > 1e-12:
                        rw = abs(float(row["theta_std"])) / mean_abs
                    else:
                        rw = float(missing_weight)
                    row["raw_weight"] = rw
                    row["final_weight"] = float(np.clip(rw, clip_min, clip_max))
                raw_weights[row["variable"]] = float(row["raw_weight"]) if pd.notna(row.get("raw_weight")) else float(missing_weight)

        else:
            # manual_df 为 None：文件不存在或解析失败，已在 _load_manual_dml_weights 中发出 warning
            logger.warning("[Model 2] 人工 DML 权重文件无法读取，所有 A 变量使用默认权重 1.0。")
            use_manual = False  # 降级为旧逻辑

    if not use_manual:
        # 旧逻辑：读取外部 DML effect 目录
        effect_dict, effect_file, effect_var_col, effect_value_col = _load_external_dml_effects(cfg, a_cols, logger)
        if effect_file is None:
            logger.warning("未读取到外部 DML effect 文件，将使用默认权重。")

        effect_dict_lc = {str(k).strip().lower(): float(v) for k, v in effect_dict.items()}
        matched_a_old: List[str] = []
        missing_a_old: List[str] = []
        available_abs_old: List[float] = []

        for col in a_cols:
            eff = effect_dict_lc.get(col.lower())
            if eff is None or not np.isfinite(eff):
                missing_a_old.append(col)
                rows.append({
                    "variable": col,
                    "role": "operation_A",
                    "matched_dml_variable": np.nan,
                    "treatment_group": np.nan,
                    "selected_lag_min": np.nan,
                    "theta_std": np.nan,
                    "recommended_for_weight": False,
                    "raw_weight": float(missing_weight),
                    "final_weight": float(np.clip(missing_weight, clip_min, clip_max)),
                    "weight_source": "missing_manual_theta_default_1",
                    "reason": "no matched treatment in external DML effect dir",
                })
            else:
                matched_a_old.append(col)
                ae = abs(float(eff))
                available_abs_old.append(ae)
                rows.append({
                    "variable": col,
                    "role": "operation_A",
                    "matched_dml_variable": col,
                    "treatment_group": np.nan,
                    "selected_lag_min": np.nan,
                    "theta_std": float(eff),
                    "recommended_for_weight": True,
                    "raw_weight": np.nan,
                    "final_weight": np.nan,
                    "weight_source": "external_dml_effect",
                    "reason": "",
                })

        mean_abs_old = float(np.mean(available_abs_old)) if available_abs_old else 0.0
        if mean_abs_old <= 1e-12 and matched_a_old:
            logger.warning("匹配到的 A 变量效应绝对值均值接近 0，A 变量权重回退为默认值 1.0。")

        for row in rows:
            if row["role"] != "operation_A":
                continue
            if row["weight_source"] == "external_dml_effect" and mean_abs_old > 1e-12:
                row["raw_weight"] = float(abs(float(row["theta_std"])) / mean_abs_old)
            elif pd.isna(row.get("raw_weight", np.nan)):
                row["raw_weight"] = float(missing_weight)
            row["final_weight"] = float(np.clip(row["raw_weight"], clip_min, clip_max))
            raw_weights[row["variable"]] = float(row["raw_weight"])

        logger.info(f"成功匹配到 effect 的 A 变量数量: {len(matched_a_old)}/{len(a_cols)}")
        logger.info(f"缺失 effect 的 A 变量列表: {missing_a_old}")

    # S 变量：统一权重 1.0
    for col in s_cols:
        rows.append({
            "variable": col,
            "role": "state_S",
            "matched_dml_variable": np.nan,
            "treatment_group": np.nan,
            "selected_lag_min": np.nan,
            "theta_std": np.nan,
            "recommended_for_weight": False,
            "raw_weight": float(state_weight),
            "final_weight": float(state_weight),
            "weight_source": "state_default_1",
            "reason": "state variable, weight fixed at 1",
        })
        raw_weights[col] = float(state_weight)

    # 统计日志
    a_raw_values = [raw_weights.get(c, missing_weight) for c in a_cols]
    a_final_values = [float(r["final_weight"]) for r in rows if r["role"] == "operation_A"]
    if a_raw_values:
        logger.info(
            "A变量 raw_weight 统计: min=%.6f, max=%.6f, mean=%.6f",
            float(np.min(a_raw_values)),
            float(np.max(a_raw_values)),
            float(np.mean(a_raw_values)),
        )
    if a_final_values:
        logger.info(
            "A变量 final_weight 统计: min=%.6f, max=%.6f, mean=%.6f",
            float(np.min(a_final_values)),
            float(np.max(a_final_values)),
            float(np.mean(a_final_values)),
        )
    logger.info(f"S变量数量及默认权重: count={len(s_cols)}, weight={state_weight}")

    weights_df = pd.DataFrame(rows, columns=[
        "variable", "role", "matched_dml_variable", "treatment_group", "selected_lag_min",
        "theta_std", "recommended_for_weight", "raw_weight", "final_weight", "weight_source", "reason",
    ])
    weights_path = output_dir / "dml_effect_weights.csv"
    weights_df.to_csv(weights_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {weights_path}")

    feature_weights = np.array([
        float(weights_df.loc[weights_df["variable"] == c, "final_weight"].iloc[0]) for c in feature_cols
    ], dtype=np.float32)

    X_tr = X_tr_scaled * feature_weights
    X_vl = X_vl_scaled * feature_weights
    X_te = X_te_scaled * feature_weights

    Xw_tr, yw_tr = make_windows(X_tr, y_tr, window_size)
    Xw_vl, yw_vl = make_windows(X_vl, y_vl, window_size)
    Xw_te, yw_te = make_windows(X_te, y_te, window_size)

    train_stats = build_train_stats(Xw_tr, feature_cols)
    constraints_enabled = bool(cfg.get("constraints_enabled", True))
    use_cf = constraints_enabled and bool(cfg.get("use_counterfactual_constraint", False))
    use_process = constraints_enabled and bool(cfg.get("use_process_constraint", False))
    counterfactual_rules = build_counterfactual_rules(cfg, feature_cols, a_cols, weights_df)
    process_rules_df = screen_process_rules(cfg, feature_cols, weights_df, train_stats)
    constraint_context = {
        "feature_cols": feature_cols,
        "counterfactual_rules": counterfactual_rules,
        "process_rules": process_rules_df,
        "train_stats": train_stats,
        "use_counterfactual_constraint": use_cf,
        "use_process_constraint": use_process,
        "counterfactual_lambda": float(cfg.get("counterfactual_lambda", 0.0)),
        "process_lambda": float(cfg.get("process_lambda", 0.0)),
        "deterministic": bool(cfg.get("deterministic", True)),
    }

    model = LSTMRegressor(
        input_size=len(feature_cols),
        hidden_size=cfg["lstm_hidden_size"],
        num_layers=cfg["lstm_num_layers"],
        dropout=cfg["lstm_dropout"],
        epochs=cfg["lstm_epochs"],
        batch_size=cfg["lstm_batch_size"],
        lr=cfg["lstm_lr"],
        patience=cfg["lstm_patience"],
        seed=cfg["random_seed"],
    )
    model.fit(Xw_tr, yw_tr, Xw_vl, yw_vl, logger=logger, constraint_context=constraint_context)

    y_pred_scaled = model.predict(Xw_te)
    y_pred = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = y_scaler.inverse_transform(yw_te.reshape(-1, 1)).ravel()
    metrics = compute_metrics(y_true, y_pred)
    logger.info(f"Model 2 Test: MAE={metrics['MAE']:.4f}, RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")
    if no_recommended_note:
        metrics["note"] = no_recommended_note

    predictions_test = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pred_path = output_dir / "dml_effect_weight_predictions_test.csv"
    predictions_test.to_csv(pred_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {pred_path}")

    counterfactual_metrics = evaluate_counterfactual_violations(model, Xw_te, counterfactual_rules, feature_cols)
    process_metrics = evaluate_process_violations(model, Xw_te, process_rules_df, feature_cols)

    return {
        "model_name": "dml_effect_weight_lstm",
        "metrics": metrics,
        "y_true": y_true,
        "y_pred": y_pred,
        "lstm": model,
        "as_scaler": feat_scaler,
        "y_scaler": y_scaler,
        "feature_cols": feature_cols,
        "predictions_test": predictions_test,
        "dml_effect_weights": weights_df,
        "dml_effect_file": "",
        "dml_effect_variable_col": var_col,
        "dml_effect_value_col": val_col,
        "counterfactual_rules": pd.DataFrame(counterfactual_rules),
        "process_rules": process_rules_df,
        "constraint_loss_log": model.loss_log_df.copy(),
        "counterfactual_metrics": counterfactual_metrics,
        "process_metrics": process_metrics,
        "use_counterfactual_constraint": use_cf,
        "use_process_constraint": use_process,
    }


# ─── Model 3：DML 残差软测量 ──────────────────────────────────────────────────

def run_model3_dml_residual(
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
    dml_effect_weights_df: Optional[pd.DataFrame] = None,
) -> dict:
    """
    Model 3: DML 残差软测量
      C -> g(C) -> y_base
      C -> q_j(C) -> X_res_j
      [A_res, S_res] 窗口序列 -> LSTM -> y_res
      y_hat = y_base + y_res_hat
    """
    logger.info("=" * 60)
    logger.info("Model 3: DML 残差软测量")

    from sklearn.preprocessing import StandardScaler

    residual_as_cols = a_cols + s_cols

    # ── C 不足时降级 ─────────────────────────────────────────────────────────
    if not c_cols:
        logger.warning("C candidates insufficient (n=0); Model 3 (DML 残差) 不可用，跳过残差化。")
        return {
            "model_name": "dml_residual_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                        "note": "residual model unavailable: C is empty"},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "residual_lstm": None,
            "y_res_scaler": None,
            "as_scaler": None,
            "c_scaler": None,
            "g_model": None,
            "q_models": {},
            "feature_cols": residual_as_cols,
            "residual_feature_summary": pd.DataFrame(),
            "y_baseline_predictions": pd.DataFrame(),
            "predictions_test": pd.DataFrame(),
            "counterfactual_rules": pd.DataFrame(),
            "process_rules": pd.DataFrame(),
            "constraint_loss_log": pd.DataFrame(),
            "counterfactual_metrics": pd.DataFrame(),
            "process_metrics": pd.DataFrame(),
            "use_counterfactual_constraint": False,
            "use_process_constraint": False,
        }

    if not residual_as_cols:
        logger.warning("A + S 为空，Model 3 不可用")
        return {
            "model_name": "dml_residual_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan"),
                        "note": "residual model unavailable: A+S is empty"},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "residual_lstm": None,
            "y_res_scaler": None,
            "as_scaler": None,
            "c_scaler": None,
            "g_model": None,
            "q_models": {},
            "feature_cols": [],
            "residual_feature_summary": pd.DataFrame(),
            "y_baseline_predictions": pd.DataFrame(),
            "predictions_test": pd.DataFrame(),
            "counterfactual_rules": pd.DataFrame(),
            "process_rules": pd.DataFrame(),
            "constraint_loss_log": pd.DataFrame(),
            "counterfactual_metrics": pd.DataFrame(),
            "process_metrics": pd.DataFrame(),
            "use_counterfactual_constraint": False,
            "use_process_constraint": False,
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
    logger.info(f"  g_model R2 (train): {g_score_train:.4f}")

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

    dml_effect_weights_df = dml_effect_weights_df if dml_effect_weights_df is not None else pd.DataFrame()
    train_stats = build_train_stats(Xw_tr, residual_as_cols)
    constraints_enabled = bool(cfg.get("constraints_enabled", True))
    use_cf = constraints_enabled and bool(cfg.get("use_counterfactual_constraint", False))
    use_process = constraints_enabled and bool(cfg.get("use_process_constraint", False))
    counterfactual_rules = build_counterfactual_rules(cfg, residual_as_cols, a_cols, dml_effect_weights_df)
    process_rules_df = screen_process_rules(cfg, residual_as_cols, dml_effect_weights_df, train_stats)
    constraint_context = {
        "feature_cols": residual_as_cols,
        "counterfactual_rules": counterfactual_rules,
        "process_rules": process_rules_df,
        "train_stats": train_stats,
        "use_counterfactual_constraint": use_cf,
        "use_process_constraint": use_process,
        "counterfactual_lambda": float(cfg.get("counterfactual_lambda", 0.0)),
        "process_lambda": float(cfg.get("process_lambda", 0.0)),
        "deterministic": bool(cfg.get("deterministic", True)),
    }

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
    residual_lstm.fit(
        Xw_tr,
        yw_tr,
        Xw_vl,
        yw_vl,
        logger=logger,
        constraint_context=constraint_context,
    )

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

    logger.info(f"Model 3 Test (原始 y): MAE={metrics_final['MAE']:.4f}, "
                f"RMSE={metrics_final['RMSE']:.4f}, R2={metrics_final['R2']:.4f}")
    logger.info(f"Model 3 Test (y_res):  MAE={metrics_res['MAE']:.4f}, "
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

    counterfactual_metrics = evaluate_counterfactual_violations(
        residual_lstm, Xw_te, counterfactual_rules, residual_as_cols
    )
    process_metrics = evaluate_process_violations(
        residual_lstm, Xw_te, process_rules_df, residual_as_cols
    )

    return {
        "model_name": "dml_residual_lstm",
        "metrics": metrics_final,
        "y_true": y_true_aligned,
        "y_pred": y_hat,
        "residual_lstm": residual_lstm,
        "y_res_scaler": y_res_scaler,
        "as_scaler": as_scaler,
        "feature_scaler": as_scaler,
        "c_scaler": c_scaler,
        "g_model": g_model,
        "q_models": q_models,
        "feature_cols": residual_as_cols,
        "residual_feature_summary": residual_feature_summary,
        "y_baseline_predictions": y_baseline_pred_df,
        "predictions_test": predictions_test,
        "counterfactual_rules": pd.DataFrame(counterfactual_rules),
        "process_rules": process_rules_df,
        "constraint_loss_log": residual_lstm.loss_log_df.copy(),
        "counterfactual_metrics": counterfactual_metrics,
        "process_metrics": process_metrics,
        "use_counterfactual_constraint": use_cf,
        "use_process_constraint": use_process,
    }


def build_run_manifest(cfg: dict, output_dir: Path, script_name: str) -> dict:
    torch_version = None
    cuda_available = False
    try:
        torch, *_ = _import_torch()
        torch_version = torch.__version__
        cuda_available = bool(torch.cuda.is_available())
    except Exception:
        pass
    return {
        "cfg": cfg,
        "random_seed": int(cfg.get("random_seed", 42)),
        "deterministic": bool(cfg.get("deterministic", True)),
        "use_counterfactual_constraint": bool(cfg.get("use_counterfactual_constraint", False)),
        "use_process_constraint": bool(cfg.get("use_process_constraint", False)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "torch_version": torch_version,
        "cuda_available": cuda_available,
        "output_dir": str(output_dir),
        "run_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "script_name": script_name,
    }


def _ensure_csv(
    df: Optional[pd.DataFrame],
    path: Path,
    default_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    out_df = df.copy() if isinstance(df, pd.DataFrame) else pd.DataFrame()
    if out_df.empty and default_cols:
        out_df = pd.DataFrame(columns=default_cols)
    out_df.to_csv(path, index=False, encoding="utf-8-sig")
    return out_df


# ─── 保存所有输出 ─────────────────────────────────────────────────────────────

def save_outputs(
    output_dir: Path,
    cfg: dict,
    roles_df: pd.DataFrame,
    model0: dict,
    model1: dict,
    model2: dict,
    model3: dict,
    logger: logging.Logger,
    run_manifest: Optional[dict] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 变量角色表
    roles_path = output_dir / "variable_roles.csv"
    roles_df.to_csv(roles_path, index=False, encoding="utf-8-sig")
    logger.info(f"已保存: {roles_path}")

    # 2. 各模型预测文件
    pred_file_map = [
        (model0, "baseline_predictions_test.csv"),
        (model1, "as_lstm_predictions_test.csv"),
        (model2, "dml_effect_weight_predictions_test.csv"),
        (model3, "dml_residual_predictions_test.csv"),
    ]
    for m_dict, filename in pred_file_map:
        pred_df = m_dict.get("predictions_test", pd.DataFrame())
        if isinstance(pred_df, pd.DataFrame) and not pred_df.empty:
            p = output_dir / filename
            pred_df.to_csv(p, index=False, encoding="utf-8-sig")
            logger.info(f"已保存: {p}")

    # 3. dml_effect_weights.csv
    if not model2.get("dml_effect_weights", pd.DataFrame()).empty:
        weights_path = output_dir / "dml_effect_weights.csv"
        model2["dml_effect_weights"].to_csv(weights_path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {weights_path}")

    # 4. y_baseline_predictions.csv
    if not model3.get("y_baseline_predictions", pd.DataFrame()).empty:
        bp_path = output_dir / "y_baseline_predictions.csv"
        model3["y_baseline_predictions"].to_csv(bp_path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {bp_path}")

    # 5. residual_feature_summary.csv
    if not model3.get("residual_feature_summary", pd.DataFrame()).empty:
        rs_path = output_dir / "residual_feature_summary.csv"
        model3["residual_feature_summary"].to_csv(rs_path, index=False, encoding="utf-8-sig")
        logger.info(f"已保存: {rs_path}")

    # 6. metrics_compare.csv
    rows = []
    for m_dict, split in [
        (model0, "test"),
        (model1, "test"),
        (model2, "test"),
        (model3, "test"),
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

    if run_manifest is None and bool(cfg.get("save_run_manifest", True)):
        run_manifest = build_run_manifest(cfg, output_dir, Path(__file__).name)
    if run_manifest is not None and bool(cfg.get("save_run_manifest", True)):
        rm_path = output_dir / "run_manifest.json"
        with open(rm_path, "w", encoding="utf-8") as f:
            json.dump(run_manifest, f, ensure_ascii=False, indent=2)
        logger.info(f"已保存: {rm_path}")

    # 7. 约束相关输出（即使空也保存）
    cf_cols = [
        "rule_name", "variable", "direction", "lag", "effective_n",
        "violation_rate", "mean_violation_magnitude", "mean_expected_response",
        "use_in_train", "use_in_eval", "status", "reason",
    ]
    process_screen_cols = [
        "rule_name", "variable", "direction", "lag", "delta_std", "rule_lambda",
        "require_dml_agree", "min_abs_dml_effect", "use_in_train", "use_in_eval",
        "status", "reason", "dml_sign", "dml_abs_effect", "active_region",
    ]
    loss_cols = [
        "epoch", "train_pred_loss", "train_cf_loss", "train_process_loss",
        "train_total_loss", "val_loss",
    ]

    m2_cf = _ensure_csv(
        model2.get("counterfactual_metrics"),
        output_dir / "counterfactual_violation_metrics_model2.csv",
        default_cols=cf_cols,
    )
    m3_cf = _ensure_csv(
        model3.get("counterfactual_metrics"),
        output_dir / "counterfactual_violation_metrics_model3.csv",
        default_cols=cf_cols,
    )
    m2_proc = _ensure_csv(
        model2.get("process_metrics"),
        output_dir / "process_violation_metrics_model2.csv",
        default_cols=cf_cols,
    )
    m3_proc = _ensure_csv(
        model3.get("process_metrics"),
        output_dir / "process_violation_metrics_model3.csv",
        default_cols=cf_cols,
    )
    _ensure_csv(
        model2.get("process_rules"),
        output_dir / "process_constraint_screening_model2.csv",
        default_cols=process_screen_cols,
    )
    _ensure_csv(
        model3.get("process_rules"),
        output_dir / "process_constraint_screening_model3.csv",
        default_cols=process_screen_cols,
    )
    _ensure_csv(
        model2.get("constraint_loss_log"),
        output_dir / "constraint_loss_log_model2.csv",
        default_cols=loss_cols,
    )
    _ensure_csv(
        model3.get("constraint_loss_log"),
        output_dir / "constraint_loss_log_model3.csv",
        default_cols=loss_cols,
    )

    cf_vr_m2, cf_mv_m2 = _aggregate_violation_metrics(m2_cf)
    cf_vr_m3, cf_mv_m3 = _aggregate_violation_metrics(m3_cf)
    pr_vr_m2, pr_mv_m2 = _aggregate_violation_metrics(m2_proc)
    pr_vr_m3, pr_mv_m3 = _aggregate_violation_metrics(m3_proc)
    cm_rows = []
    for model_dict, cf_vr, pr_vr, cf_mv, pr_mv in [
        (model2, cf_vr_m2, pr_vr_m2, cf_mv_m2, pr_mv_m2),
        (model3, cf_vr_m3, pr_vr_m3, cf_mv_m3, pr_mv_m3),
    ]:
        mm = model_dict.get("metrics", {})
        cm_rows.append({
            "model_name": model_dict.get("model_name", "unknown"),
            "use_counterfactual_constraint": model_dict.get("use_counterfactual_constraint", False),
            "use_process_constraint": model_dict.get("use_process_constraint", False),
            "MAE": mm.get("MAE", np.nan),
            "RMSE": mm.get("RMSE", np.nan),
            "R2": mm.get("R2", np.nan),
            "CF_VR_mean": cf_vr,
            "Process_VR_mean": pr_vr,
            "CF_mean_violation": cf_mv,
            "Process_mean_violation": pr_mv,
            "random_seed": int(cfg.get("random_seed", 42)),
        })
    _ensure_csv(
        pd.DataFrame(cm_rows),
        output_dir / "constraint_metrics_compare.csv",
    )

    # 8. 保存模型权重和scalers
    import torch
    import pickle

    if "residual_lstm" in model3 and model3["residual_lstm"] is not None:
        lstm_model = model3["residual_lstm"]
        if hasattr(lstm_model, '_model') and lstm_model._model is not None:
            model_path = output_dir / "dml_residual_lstm_checkpoint.pt"
            torch.save({
                'model_state_dict': lstm_model._model.state_dict(),
                'input_size': lstm_model.input_size,
                'hidden_size': lstm_model.hidden_size,
                'num_layers': lstm_model.num_layers,
                'dropout': lstm_model.dropout,
            }, model_path)
            logger.info(f"已保存DML残差LSTM权重: {model_path}")

    # Model 3 组件
    scalers_path = output_dir / "dml_residual_model_components.pkl"
    scalers_dict = {}

    for key in ["y_res_scaler", "as_scaler", "feature_scaler", "c_scaler", "g_model", "q_models", "feature_cols"]:
        if key in model3:
            scalers_dict[key] = model3.get(key)

    if scalers_dict:
        with open(scalers_path, 'wb') as f:
            pickle.dump(scalers_dict, f)
        logger.info(f"已保存DML残差组件: {scalers_path}")

    # 保存Model 0/1/2 的LSTM权重和scalers
    for model_dict, model_name in [
        (model0, "baseline_all_lstm"),
        (model1, "as_lstm"),
        (model2, "dml_effect_weight_lstm"),
    ]:
        if "lstm" in model_dict and model_dict["lstm"] is not None:
            lstm_model = model_dict["lstm"]
            if hasattr(lstm_model, '_model') and lstm_model._model is not None:
                model_path = output_dir / f"{model_name}_lstm_checkpoint.pt"
                torch.save({
                    'model_state_dict': lstm_model._model.state_dict(),
                    'input_size': lstm_model.input_size,
                    'hidden_size': lstm_model.hidden_size,
                    'num_layers': lstm_model.num_layers,
                    'dropout': lstm_model.dropout,
                }, model_path)
                logger.info(f"已保存{model_name} LSTM权重: {model_path}")
                
                # 保存scalers和特征列
                if "as_scaler" in model_dict or "y_scaler" in model_dict:
                    scaler_path = output_dir / f"{model_name}_scalers.pkl"
                    scaler_dict = {
                        "as_scaler": model_dict.get("as_scaler"),
                        "y_scaler": model_dict.get("y_scaler"),
                        "feature_cols": model_dict.get("feature_cols", []),
                    }
                    with open(scaler_path, 'wb') as f:
                        pickle.dump(scaler_dict, f)
                    logger.info(f"已保存{model_name} scalers: {scaler_path}")


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
    parser.add_argument("--use-counterfactual-constraint", type=int, choices=[0, 1], default=None)
    parser.add_argument("--counterfactual-lambda", type=float, default=None)
    parser.add_argument("--use-process-constraint", type=int, choices=[0, 1], default=None)
    parser.add_argument("--process-lambda", type=float, default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--skip-model0", action="store_true", help="跳过 Model 0（基线软测量）")
    parser.add_argument("--skip-model1", action="store_true", help="跳过 Model 1（因果输入软测量）")
    parser.add_argument("--skip-model2", action="store_true", help="跳过 Model 2（DML效应权重软测量）")
    parser.add_argument("--only-model3", action="store_true", help="只运行 Model 3（DML残差软测量）")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    output_dir = Path(cfg["output_dir"])
    if cfg.get("run_name"):
        output_dir = output_dir / str(cfg["run_name"])
    cfg["output_dir"] = str(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "run_log.txt"
    logger = setup_logger(log_path)

    logger.info("=" * 60)
    logger.info("DML 正交残差软测量  train_dml_residual_soft_sensor.py")
    logger.info("=" * 60)
    logger.info(f"配置文件: {args.config}")
    logger.info(f"配置内容: {json.dumps(cfg, ensure_ascii=False, indent=2)}")

    set_seed(
        int(cfg["random_seed"]),
        deterministic=bool(cfg.get("deterministic", True)),
        logger=logger,
    )

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

    # ── 4. 数据切分 ──────────────────────────────────────────────────────────
    # 只保留数值列用于建模
    model_cols = [c for c in [target_col] + all_feature_cols if c in df.columns]
    if time_col and time_col in df.columns:
        model_cols = [time_col] + model_cols
    df_model = df[[c for c in model_cols if c in df.columns]]
    num_cols_only = df_model.select_dtypes(include=np.number).columns.tolist()
    df_model = df_model[num_cols_only]

    # 更新 feature/role 列表为实际存在的数值列
    all_feature_cols = [c for c in all_feature_cols if c in df_model.columns]
    a_cols = [c for c in a_cols if c in df_model.columns]
    c_cols = [c for c in c_cols if c in df_model.columns]
    s_cols = [c for c in s_cols if c in df_model.columns]

    # Bug 4 修复：在数值过滤完成后再打印全部特征列，保证与实际建模列一致
    logger.info(f"全部特征列（建模用数值列）({len(all_feature_cols)}): {all_feature_cols}")
    logger.info(f"窗口长度: {cfg['window_size']}")
    logger.info(f"数据切分方式: train={cfg['train_ratio']}, val={cfg['val_ratio']}, test={cfg['test_ratio']}")

    if target_col not in df_model.columns:
        logger.error(f"目标列 '{target_col}' 不在数值列中，退出")
        sys.exit(1)

    train_df, val_df, test_df = split_data(df_model, cfg, logger)

    # ── 确定要运行的模型 ──────────────────────────────────────────────────
    skip_model0 = args.skip_model0 or args.only_model3
    skip_model1 = args.skip_model1 or args.only_model3
    skip_model2 = args.skip_model2 or args.only_model3

    # ── 5. Model 0：基线软测量 ────────────────────────────────────────────
    if skip_model0:
        logger.info("=" * 60)
        logger.info("跳过 Model 0（基线软测量）")
        model0 = {
            "model_name": "baseline_all_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
        }
    else:
        model0 = run_model0_baseline(
            train_df, val_df, test_df,
            all_feature_cols, target_col, cfg, logger,
        )

    # ── 6. Model 1：因果输入软测量 ────────────────────────────────────────
    if skip_model1:
        logger.info("=" * 60)
        logger.info("跳过 Model 1（因果输入软测量）")
        model1 = {
            "model_name": "as_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
        }
    else:
        model1 = run_model1_causal_input(
            train_df, val_df, test_df,
            a_cols, s_cols, target_col, cfg, logger,
        )

    # ── 7. Model 2：DML 效应权重软测量 ────────────────────────────────────
    if skip_model2:
        logger.info("=" * 60)
        logger.info("跳过 Model 2（DML效应权重软测量）")
        logger.info("Model 2 skipped; building DML effect weights table for Model 3 constraints.")
        dml_effect_weights_df = build_dml_effect_weights_table(
            cfg, a_cols, s_cols, logger, output_dir=output_dir
        )
        model2 = {
            "model_name": "dml_effect_weight_lstm",
            "metrics": {"MAE": float("nan"), "RMSE": float("nan"), "R2": float("nan")},
            "y_true": np.array([]),
            "y_pred": np.array([]),
            "dml_effect_weights": dml_effect_weights_df,
        }
    else:
        model2 = run_model2_dml_effect_weight_lstm(
            train_df, val_df, test_df,
            c_cols, a_cols, s_cols, target_col, cfg, logger, output_dir,
        )

    # ── 8. Model 3：DML 残差软测量 ────────────────────────────────────────
    model3 = run_model3_dml_residual(
        train_df, val_df, test_df,
        c_cols, a_cols, s_cols, target_col, cfg, logger, output_dir,
        dml_effect_weights_df=model2.get("dml_effect_weights", pd.DataFrame()),
    )

    # ── 9. 保存输出 ──────────────────────────────────────────────────────
    run_manifest = None
    if bool(cfg.get("save_run_manifest", True)):
        run_manifest = build_run_manifest(cfg, output_dir, Path(__file__).name)
    save_outputs(output_dir, cfg, roles_df, model0, model1, model2, model3, logger, run_manifest=run_manifest)

    # ── 10. 四模型测试指标摘要 ────────────────────────────────────────────
    logger.info("-" * 60)
    logger.info("四个模型最终测试指标：")
    for m in [model0, model1, model2, model3]:
        mm = m.get("metrics", {})
        logger.info(
            "%s | MAE=%.4f, RMSE=%.4f, R2=%.4f",
            m.get("model_name", "unknown"),
            float(mm.get("MAE", np.nan)),
            float(mm.get("RMSE", np.nan)),
            float(mm.get("R2", np.nan)),
        )

    # ── 11. 最终日志摘要 ─────────────────────────────────────────────────
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
