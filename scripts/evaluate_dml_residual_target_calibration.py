from __future__ import annotations

import argparse
import json
import logging
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from train_dml_residual_soft_sensor import (
    LSTMRegressor,
    build_residualization_model,
    compute_metrics,
    infer_variable_roles,
    load_config,
    load_data,
    make_windows,
    preprocess,
    set_seed,
    setup_logger,
)


def parse_int_list(raw: str) -> List[int]:
    text = str(raw or "").strip()
    if not text:
        return []
    values: List[int] = []
    for part in text.split(","):
        token = part.strip()
        if not token:
            continue
        try:
            v = int(token)
        except Exception as e:
            raise ValueError(f"整数列表解析失败，非法值: '{token}'") from e
        if v <= 0:
            raise ValueError(f"整数列表解析失败，必须为正整数: {v}")
        values.append(v)
    return values


def build_temporal_c_features(
    df: pd.DataFrame,
    c_cols: List[str],
    c_scaler,
    lags: List[int],
    rollings: List[int],
    diffs: List[int],
    fill_mode: str,
) -> Tuple[np.ndarray, List[str]]:
    c_scaled = c_scaler.transform(df[c_cols].values)
    c_scaled_df = pd.DataFrame(c_scaled, columns=c_cols, index=df.index)

    blocks: List[pd.DataFrame] = [c_scaled_df]
    feature_names: List[str] = list(c_cols)

    for lag in lags:
        lag_df = c_scaled_df.shift(lag)
        lag_df.columns = [f"{c}_lag{lag}" for c in c_cols]
        blocks.append(lag_df)
        feature_names.extend(lag_df.columns.tolist())

    for w in rollings:
        mean_df = c_scaled_df.rolling(window=w, min_periods=1).mean()
        mean_df.columns = [f"{c}_rollmean{w}" for c in c_cols]
        blocks.append(mean_df)
        feature_names.extend(mean_df.columns.tolist())

        std_df = c_scaled_df.rolling(window=w, min_periods=1).std().fillna(0.0)
        std_df.columns = [f"{c}_rollstd{w}" for c in c_cols]
        blocks.append(std_df)
        feature_names.extend(std_df.columns.tolist())

    for d in diffs:
        diff_df = c_scaled_df - c_scaled_df.shift(d)
        diff_df.columns = [f"{c}_diff{d}" for c in c_cols]
        blocks.append(diff_df)
        feature_names.extend(diff_df.columns.tolist())

    features_df = pd.concat(blocks, axis=1)
    if fill_mode == "ffill_bfill":
        features_df = features_df.ffill().bfill()
    elif fill_mode == "zero":
        features_df = features_df.fillna(0.0)
    elif fill_mode == "drop":
        features_df = features_df.dropna()
    else:
        raise ValueError(f"不支持 temporal_g_fill: {fill_mode}")

    return features_df.values.astype(np.float32), feature_names


def fit_gt_lgbm_current(
    target_calib_df: pd.DataFrame,
    c_cols: List[str],
    target_col: str,
    c_scaler,
    residualization_model: str,
    seed: int,
    conservative: bool,
):
    C_calib_scaled = c_scaler.transform(target_calib_df[c_cols].values)
    y_calib_raw = target_calib_df[target_col].values

    if conservative:
        lgb = None
        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None
        if lgb is not None:
            model = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.03,
                num_leaves=7,
                max_depth=3,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
        else:
            model = build_residualization_model(residualization_model, seed)
    else:
        model = build_residualization_model(residualization_model, seed)
    model.fit(C_calib_scaled, y_calib_raw)
    return model


def fit_gt_ridge_current(
    target_calib_df: pd.DataFrame,
    c_cols: List[str],
    target_col: str,
    c_scaler,
    residualization_model: str,
    seed: int,
    alpha: float,
):
    del residualization_model, seed
    from sklearn.linear_model import Ridge

    C_calib_scaled = c_scaler.transform(target_calib_df[c_cols].values)
    y_calib_raw = target_calib_df[target_col].values
    model = Ridge(alpha=float(alpha))
    model.fit(C_calib_scaled, y_calib_raw)
    return model


def fit_gt_lgbm_temporal(
    target_calib_df: pd.DataFrame,
    c_cols: List[str],
    target_col: str,
    c_scaler,
    lags: List[int],
    rollings: List[int],
    diffs: List[int],
    fill_mode: str,
    seed: int,
    conservative: bool,
) -> Tuple[object, List[str]]:
    X_calib_temporal, temporal_feature_names = build_temporal_c_features(
        df=target_calib_df,
        c_cols=c_cols,
        c_scaler=c_scaler,
        lags=lags,
        rollings=rollings,
        diffs=diffs,
        fill_mode=fill_mode,
    )
    y_calib_raw = target_calib_df[target_col].values

    lgb = None
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None

    if lgb is None:
        model = build_residualization_model("random_forest", seed)
    elif conservative:
        model = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.03,
            num_leaves=7,
            max_depth=3,
            min_child_samples=30,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    else:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            random_state=seed,
            n_jobs=1,
            verbose=-1,
        )
    model.fit(X_calib_temporal, y_calib_raw)
    return model, temporal_feature_names


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate Model 3 DML residual soft sensor target calibration variants "
            "(g_S+q_S+h_S, g_T+q_S+h_S, g_T+q_T+h_S)."
        )
    )
    parser.add_argument("--config", default="configs/residual_soft_sensor.yaml")
    parser.add_argument("--source-model-dir", required=True)
    parser.add_argument("--target-data-path", required=True)
    parser.add_argument(
        "--output-dir",
        default="results/dml_residual_target_calibration",
    )
    parser.add_argument("--calib-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.8)
    parser.add_argument("--target-name", default=None)
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "source",
            "gt_lgbm_current",
            "gt_only",
            "gt_bias_only",
            "gt_ridge_current",
            "gt_lgbm_temporal",
            "gt_qt",
        ],
        default="all",
    )
    parser.add_argument("--residualization-model", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--allow-train-source-if-missing", action="store_true")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--temporal-g-lags", default="1,3,6,12")
    parser.add_argument("--temporal-g-rollings", default="3,6,12")
    parser.add_argument("--temporal-g-diffs", default="1,3")
    parser.add_argument(
        "--temporal-g-fill",
        choices=["ffill_bfill", "zero", "drop"],
        default="ffill_bfill",
    )
    parser.add_argument("--lgbm-calib-conservative", action="store_true")
    return parser.parse_args()


def _safe_name_from_path(path_str: str) -> str:
    p = Path(path_str)
    if p.stem:
        return p.stem
    normalized = path_str.replace("\\", "/").rstrip("/")
    if not normalized:
        return "target"
    return Path(normalized).name or "target"


def _load_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported target data format: {path.suffix}")


def _read_variable_roles(path: Path, cfg: dict, logger: logging.Logger) -> Tuple[List[str], List[str], List[str], str]:
    roles_df = pd.read_csv(path)
    if "variable" not in roles_df.columns or "role" not in roles_df.columns:
        raise ValueError(f"Invalid variable_roles.csv at {path}: missing 'variable' or 'role' columns")

    a_cols = roles_df.loc[roles_df["role"] == "operation_A", "variable"].astype(str).tolist()
    c_cols = roles_df.loc[roles_df["role"] == "confounder_C", "variable"].astype(str).tolist()
    s_cols = roles_df.loc[roles_df["role"] == "state_S", "variable"].astype(str).tolist()

    target_candidates = roles_df.loc[roles_df["role"] == "target", "variable"].astype(str).tolist()
    target_col = cfg.get("target_col")
    if target_candidates:
        target_col = target_candidates[0]
    if not target_col:
        raise ValueError("target_col cannot be determined from variable_roles.csv or config")

    logger.info(
        "Loaded variable roles from %s: A=%d, C=%d, S=%d, target=%s",
        path,
        len(a_cols),
        len(c_cols),
        len(s_cols),
        target_col,
    )
    return c_cols, a_cols, s_cols, target_col


def _infer_variable_roles_from_source_cfg(cfg: dict, logger: logging.Logger) -> Tuple[List[str], List[str], List[str], str]:
    source_df = load_data(cfg, logger)
    source_df = preprocess(source_df, cfg, logger)
    roles_df = infer_variable_roles(source_df, cfg, logger)
    a_cols = roles_df.loc[roles_df["role"] == "operation_A", "variable"].astype(str).tolist()
    c_cols = roles_df.loc[roles_df["role"] == "confounder_C", "variable"].astype(str).tolist()
    s_cols = roles_df.loc[roles_df["role"] == "state_S", "variable"].astype(str).tolist()
    target_col = cfg["target_col"]
    return c_cols, a_cols, s_cols, target_col


def _ensure_source_artifacts(
    source_model_dir: Path,
    args: argparse.Namespace,
    cfg: dict,
    logger: logging.Logger,
) -> Tuple[Path, Path]:
    del cfg
    ckpt = source_model_dir / "dml_residual_lstm_checkpoint.pt"
    components = source_model_dir / "dml_residual_model_components.pkl"
    if ckpt.exists() and components.exists():
        return ckpt, components

    if not args.allow_train_source_if_missing:
        missing = []
        if not ckpt.exists():
            missing.append("dml_residual_lstm_checkpoint.pt")
        if not components.exists():
            missing.append("dml_residual_model_components.pkl")
        raise FileNotFoundError("source_model_dir 缺少必需文件: " + ", ".join(missing))

    logger.warning("Source Model 3 artifacts missing, fallback training Model 3 only in source-model-dir")
    train_script = Path(__file__).resolve().parent / "train_dml_residual_soft_sensor.py"
    cmd = [
        sys.executable,
        str(train_script),
        "--config",
        str(args.config),
        "--only-model3",
        "--output-dir",
        str(source_model_dir),
    ]
    subprocess.run(cmd, check=True)

    if not ckpt.exists() or not components.exists():
        raise FileNotFoundError(
            "Fallback training finished but required artifacts still missing: "
            f"{ckpt}, {components}"
        )
    return ckpt, components


def _load_components(path: Path) -> dict:
    with open(path, "rb") as f:
        components = pickle.load(f)
    required = ["y_res_scaler", "as_scaler", "c_scaler", "q_models", "feature_cols"]
    missing = [k for k in required if k not in components]
    if missing:
        raise KeyError(f"Missing required keys in {path}: {missing}")
    return components


def _load_residual_lstm_from_checkpoint(path: Path) -> LSTMRegressor:
    import torch

    checkpoint = torch.load(path, map_location="cpu")
    for k in ["model_state_dict", "input_size", "hidden_size", "num_layers", "dropout"]:
        if k not in checkpoint:
            raise KeyError(f"Missing '{k}' in checkpoint: {path}")

    model = LSTMRegressor(
        input_size=int(checkpoint["input_size"]),
        hidden_size=int(checkpoint["hidden_size"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint["dropout"]),
    )
    net, device = model._build()
    net.load_state_dict(checkpoint["model_state_dict"])
    net.eval()
    model._model = net
    model._device = device
    return model


def _split_target_by_time(df: pd.DataFrame, calib_ratio: float, test_ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not np.isclose(calib_ratio + test_ratio, 1.0, atol=1e-8):
        raise ValueError(
            f"calib_ratio + test_ratio must be 1.0, got {calib_ratio + test_ratio:.6f}"
        )
    n = len(df)
    n_calib = int(n * calib_ratio)
    if n_calib <= 0 or n_calib >= n:
        raise ValueError(f"Invalid calibration split size: n={n}, n_calib={n_calib}")
    calib_df = df.iloc[:n_calib].copy()
    test_df = df.iloc[n_calib:].copy()
    return calib_df, test_df


def evaluate_with_custom_g_predictions(
    variant_name: str,
    df: pd.DataFrame,
    y_base_raw: np.ndarray,
    c_cols: List[str],
    residual_as_cols: List[str],
    target_col: str,
    c_scaler_source,
    as_scaler_source,
    y_res_scaler_source,
    residual_lstm: LSTMRegressor,
    q_models: Dict[str, object],
    window_size: int,
) -> Tuple[pd.DataFrame, dict]:
    C_scaled = c_scaler_source.transform(df[c_cols].values)
    AS_scaled = as_scaler_source.transform(df[residual_as_cols].values)

    AS_res = np.empty_like(AS_scaled)
    for j, col in enumerate(residual_as_cols):
        if col not in q_models:
            raise KeyError(f"Missing q model for feature '{col}'")
        xj_hat = np.asarray(q_models[col].predict(C_scaled), dtype=float)
        AS_res[:, j] = AS_scaled[:, j] - xj_hat

    dummy_y = np.zeros(len(AS_res), dtype=np.float32)
    Xw, _ = make_windows(AS_res, dummy_y, window_size)

    y_res_pred_scaled = residual_lstm.predict(Xw)
    y_res_pred = y_res_scaler_source.inverse_transform(y_res_pred_scaled.reshape(-1, 1)).ravel()

    align_offset = window_size - 1
    y_base_aligned = np.asarray(y_base_raw, dtype=float)[align_offset: align_offset + len(y_res_pred)]
    y_true_aligned = df[target_col].values[align_offset: align_offset + len(y_res_pred)]

    y_pred = y_base_aligned + y_res_pred
    error = y_pred - y_true_aligned

    metrics = compute_metrics(y_true_aligned, y_pred)
    metrics.update(
        {
            "mean_error": float(np.mean(error)),
            "median_error": float(np.median(error)),
            "error_std": float(np.std(error)),
            "abs_error_mean": float(np.mean(np.abs(error))),
            "y_base_mean": float(np.mean(y_base_aligned)),
            "y_res_pred_mean": float(np.mean(y_res_pred)),
            "y_true_mean": float(np.mean(y_true_aligned)),
            "residual_bias_proxy": float(np.mean(y_true_aligned - y_base_aligned)),
            "n_pred": int(len(y_pred)),
        }
    )

    pred_df = pd.DataFrame(
        {
            "index": df.index[align_offset: align_offset + len(y_pred)],
            "variant": variant_name,
            "y_true": y_true_aligned,
            "y_base": y_base_aligned,
            "y_res_pred": y_res_pred,
            "y_pred": y_pred,
            "error": error,
            "abs_error": np.abs(error),
        }
    )
    return pred_df, metrics


def evaluate_source_on_df(
    variant_name: str,
    target_df: pd.DataFrame,
    c_cols: List[str],
    residual_as_cols: List[str],
    target_col: str,
    c_scaler_source,
    as_scaler_source,
    y_res_scaler_source,
    residual_lstm: LSTMRegressor,
    g_model_source,
    q_models_source: Dict[str, object],
    window_size: int,
) -> Tuple[pd.DataFrame, dict]:
    C_scaled = c_scaler_source.transform(target_df[c_cols].values)
    y_base_raw = np.asarray(g_model_source.predict(C_scaled), dtype=float)
    return evaluate_with_custom_g_predictions(
        variant_name=variant_name,
        df=target_df,
        y_base_raw=y_base_raw,
        c_cols=c_cols,
        residual_as_cols=residual_as_cols,
        target_col=target_col,
        c_scaler_source=c_scaler_source,
        as_scaler_source=as_scaler_source,
        y_res_scaler_source=y_res_scaler_source,
        residual_lstm=residual_lstm,
        q_models=q_models_source,
        window_size=window_size,
    )


def _evaluate_variant(
    variant_name: str,
    target_test_df: pd.DataFrame,
    c_cols: List[str],
    residual_as_cols: List[str],
    target_col: str,
    c_scaler_source,
    as_scaler_source,
    y_res_scaler_source,
    residual_lstm: LSTMRegressor,
    g_model,
    q_models: Dict[str, object],
    window_size: int,
) -> Tuple[pd.DataFrame, dict]:
    C_test_scaled = c_scaler_source.transform(target_test_df[c_cols].values)
    y_base_raw = np.asarray(g_model.predict(C_test_scaled), dtype=float)
    return evaluate_with_custom_g_predictions(
        variant_name=variant_name,
        df=target_test_df,
        y_base_raw=y_base_raw,
        c_cols=c_cols,
        residual_as_cols=residual_as_cols,
        target_col=target_col,
        c_scaler_source=c_scaler_source,
        as_scaler_source=as_scaler_source,
        y_res_scaler_source=y_res_scaler_source,
        residual_lstm=residual_lstm,
        q_models=q_models,
        window_size=window_size,
    )


def _build_baseline_diagnostics(variant: str, pred_df: pd.DataFrame) -> dict:
    y_true = pred_df["y_true"].values
    y_base = pred_df["y_base"].values
    y_minus_base = y_true - y_base
    corr = np.nan
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_base) > 0:
        corr = float(np.corrcoef(y_true, y_base)[0, 1])
    base_metrics = compute_metrics(y_true, y_base)
    return {
        "variant": variant,
        "y_true_mean": float(np.mean(y_true)),
        "y_base_mean": float(np.mean(y_base)),
        "mean_y_minus_base": float(np.mean(y_minus_base)),
        "std_y_minus_base": float(np.std(y_minus_base)),
        "corr_y_base": corr,
        "r2_y_base_only": float(base_metrics["R2"]),
        "mae_y_base_only": float(base_metrics["MAE"]),
    }


def _update_global_comparison_csv(output_root: Path, metrics_df: pd.DataFrame) -> Path:
    out_path = output_root / "dml_residual_target_calibration_g_variants_comparison.csv"
    source_row = metrics_df.loc[metrics_df["variant"] == "source"]
    source_mae = float(source_row.iloc[0]["MAE"]) if not source_row.empty else np.nan
    out_rows = []
    for _, row in metrics_df.iterrows():
        mae = float(row["MAE"])
        if np.isfinite(source_mae) and source_mae != 0:
            improve = (source_mae - mae) / source_mae * 100.0
        else:
            improve = np.nan
        out_rows.append(
            {
                "target_name": row["target_name"],
                "variant": row["variant"],
                "MAE": row["MAE"],
                "RMSE": row["RMSE"],
                "R2": row["R2"],
                "mean_error": row["mean_error"],
                "residual_bias_proxy": row["residual_bias_proxy"],
                "mae_improvement_vs_source_pct": improve,
                "calib_ratio": row["calib_ratio"],
                "n_calib": row["n_calib"],
                "n_test": row["n_test"],
            }
        )
    new_df = pd.DataFrame(out_rows)
    if out_path.exists():
        old = pd.read_csv(out_path)
        old = old[old["target_name"] != new_df.iloc[0]["target_name"]]
        merged = pd.concat([old, new_df], ignore_index=True)
    else:
        merged = new_df
    merged.to_csv(out_path, index=False, encoding="utf-8-sig")
    return out_path


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.random_seed is not None:
        cfg["random_seed"] = int(args.random_seed)
    if args.residualization_model:
        cfg["residualization_model"] = args.residualization_model

    target_name = args.target_name or _safe_name_from_path(args.target_data_path)
    output_root = Path(args.output_dir)
    output_dir = output_root / target_name
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(output_dir / "run_log.txt")

    logger.info("=" * 60)
    logger.info("evaluate_dml_residual_target_calibration.py")
    logger.info("config=%s", args.config)
    logger.info("source_model_dir=%s", args.source_model_dir)
    logger.info("target_data_path=%s", args.target_data_path)
    logger.info("mode=%s", args.mode)
    logger.info("=" * 60)

    set_seed(int(cfg.get("random_seed", 42)), deterministic=bool(cfg.get("deterministic", True)), logger=logger)

    if not np.isclose(args.calib_ratio + args.test_ratio, 1.0, atol=1e-8):
        raise ValueError(
            f"calib_ratio + test_ratio must be 1.0, got {args.calib_ratio + args.test_ratio:.6f}"
        )

    temporal_lags = parse_int_list(args.temporal_g_lags)
    temporal_rollings = parse_int_list(args.temporal_g_rollings)
    temporal_diffs = parse_int_list(args.temporal_g_diffs)

    source_model_dir = Path(args.source_model_dir)
    source_model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, components_path = _ensure_source_artifacts(source_model_dir, args, cfg, logger)

    components = _load_components(components_path)
    residual_lstm = _load_residual_lstm_from_checkpoint(ckpt_path)

    roles_path = source_model_dir / "variable_roles.csv"
    if roles_path.exists():
        c_cols, a_cols, s_cols, target_col = _read_variable_roles(roles_path, cfg, logger)
    else:
        logger.warning("variable_roles.csv not found in source model dir, fallback to infer roles from source config/data")
        c_cols, a_cols, s_cols, target_col = _infer_variable_roles_from_source_cfg(cfg, logger)

    residual_as_cols = a_cols + s_cols
    feature_cols_components = list(components["feature_cols"])
    if residual_as_cols != feature_cols_components:
        logger.warning(
            "residual_as_cols mismatch with components['feature_cols']; using components feature_cols"
        )
    residual_as_cols = feature_cols_components

    for k in ["y_res_scaler", "as_scaler", "c_scaler", "q_models"]:
        if components.get(k) is None:
            raise ValueError(f"components[{k}] is None, cannot evaluate")

    target_path = Path(args.target_data_path)
    if not target_path.exists():
        raise FileNotFoundError(f"Target data file not found: {target_path}")

    target_df = _load_table(target_path)
    target_cfg = dict(cfg)
    target_cfg["target_col"] = target_col
    target_df = preprocess(target_df, target_cfg, logger)

    required_cols = [target_col] + c_cols + residual_as_cols
    missing_cols = [c for c in required_cols if c not in target_df.columns]
    if missing_cols:
        raise ValueError(f"Target data missing required columns: {missing_cols}")

    target_calib_df, target_test_df = _split_target_by_time(target_df, args.calib_ratio, args.test_ratio)
    logger.info("Target split: calib=%d, test=%d", len(target_calib_df), len(target_test_df))

    window_size = int(cfg["window_size"])
    if len(target_test_df) < window_size:
        raise ValueError(
            f"Target test size {len(target_test_df)} is smaller than window_size {window_size}"
        )
    if len(target_calib_df) < window_size:
        raise ValueError(
            f"Target calib size {len(target_calib_df)} is smaller than window_size {window_size}"
        )

    c_scaler_source = components["c_scaler"]
    as_scaler_source = components["as_scaler"]
    y_res_scaler_source = components["y_res_scaler"]
    q_models_source = components["q_models"]
    g_model_source = components.get("g_model")

    mode_to_variants = {
        "source": ["source"],
        "gt_lgbm_current": ["gt_lgbm_current"],
        "gt_only": ["gt_lgbm_current"],
        "gt_bias_only": ["gt_bias_only"],
        "gt_ridge_current": ["gt_ridge_current"],
        "gt_lgbm_temporal": ["gt_lgbm_temporal"],
        "gt_qt": ["gt_qt"],
        "all": ["source", "gt_lgbm_current", "gt_bias_only", "gt_ridge_current", "gt_lgbm_temporal"],
    }
    variants_to_run = mode_to_variants[args.mode]
    logger.info("本次运行变体: %s", variants_to_run)
    if args.mode == "all":
        logger.info("all 模式默认不运行 gt_qt。")

    if g_model_source is None and any(v in variants_to_run for v in ["source", "gt_bias_only"]):
        raise ValueError("g_model missing in source components, source/gt_bias_only 无法运行")

    model_type = cfg.get("residualization_model", "lightgbm")
    seed = int(cfg.get("random_seed", 42))

    g_t_current = None
    if any(v in variants_to_run for v in ["gt_lgbm_current", "gt_qt"]):
        g_t_current = fit_gt_lgbm_current(
            target_calib_df=target_calib_df,
            c_cols=c_cols,
            target_col=target_col,
            c_scaler=c_scaler_source,
            residualization_model=model_type,
            seed=seed,
            conservative=bool(args.lgbm_calib_conservative),
        )

    g_t_ridge = None
    if "gt_ridge_current" in variants_to_run:
        g_t_ridge = fit_gt_ridge_current(
            target_calib_df=target_calib_df,
            c_cols=c_cols,
            target_col=target_col,
            c_scaler=c_scaler_source,
            residualization_model=model_type,
            seed=seed,
            alpha=float(args.ridge_alpha),
        )

    temporal_feature_names: List[str] = []
    temporal_c_test = None
    g_t_temporal = None
    if "gt_lgbm_temporal" in variants_to_run:
        temporal_c_all, temporal_feature_names = build_temporal_c_features(
            df=target_df,
            c_cols=c_cols,
            c_scaler=c_scaler_source,
            lags=temporal_lags,
            rollings=temporal_rollings,
            diffs=temporal_diffs,
            fill_mode=args.temporal_g_fill,
        )
        if args.temporal_g_fill == "drop":
            raise ValueError("temporal_g_fill=drop 暂未支持（与 LSTM 窗口对齐复杂），请使用 ffill_bfill 或 zero")
        n_calib = len(target_calib_df)
        temporal_c_calib = temporal_c_all[:n_calib]
        temporal_c_test = temporal_c_all[n_calib:]
        y_calib_raw = target_calib_df[target_col].values

        lgb = None
        try:
            import lightgbm as lgb
        except ImportError:
            lgb = None
        if lgb is None:
            g_t_temporal = build_residualization_model("random_forest", seed)
        elif args.lgbm_calib_conservative:
            g_t_temporal = lgb.LGBMRegressor(
                n_estimators=100,
                learning_rate=0.03,
                num_leaves=7,
                max_depth=3,
                min_child_samples=30,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
        else:
            g_t_temporal = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=seed,
                n_jobs=1,
                verbose=-1,
            )
        g_t_temporal.fit(temporal_c_calib, y_calib_raw)

    q_t_models: Dict[str, object] = {}
    if "gt_qt" in variants_to_run:
        C_calib_scaled = c_scaler_source.transform(target_calib_df[c_cols].values)
        AS_calib_scaled = as_scaler_source.transform(target_calib_df[residual_as_cols].values)
        for j, col in enumerate(residual_as_cols):
            qm_t = build_residualization_model(model_type, seed + j + 1)
            qm_t.fit(C_calib_scaled, AS_calib_scaled[:, j])
            q_t_models[col] = qm_t

    source_calib_pred_df, _ = evaluate_source_on_df(
        variant_name="source",
        target_df=target_calib_df,
        c_cols=c_cols,
        residual_as_cols=residual_as_cols,
        target_col=target_col,
        c_scaler_source=c_scaler_source,
        as_scaler_source=as_scaler_source,
        y_res_scaler_source=y_res_scaler_source,
        residual_lstm=residual_lstm,
        g_model_source=g_model_source,
        q_models_source=q_models_source,
        window_size=window_size,
    )
    b_t = float(np.mean(source_calib_pred_df["y_true"].values - source_calib_pred_df["y_pred"].values))
    logger.info("gt_bias_only 校准偏置 b_T=%.6f（仅使用 calibration 段）", b_t)

    variant_rows = []
    baseline_rows = []
    all_pred_frames = []
    C_test_scaled = c_scaler_source.transform(target_test_df[c_cols].values)

    for variant in variants_to_run:
        if variant == "source":
            pred_df, metrics = evaluate_source_on_df(
                variant_name=variant,
                target_df=target_test_df,
                c_cols=c_cols,
                residual_as_cols=residual_as_cols,
                target_col=target_col,
                c_scaler_source=c_scaler_source,
                as_scaler_source=as_scaler_source,
                y_res_scaler_source=y_res_scaler_source,
                residual_lstm=residual_lstm,
                g_model_source=g_model_source,
                q_models_source=q_models_source,
                window_size=window_size,
            )
        elif variant == "gt_lgbm_current":
            y_base_raw = np.asarray(g_t_current.predict(C_test_scaled), dtype=float)
            pred_df, metrics = evaluate_with_custom_g_predictions(
                variant_name=variant,
                df=target_test_df,
                y_base_raw=y_base_raw,
                c_cols=c_cols,
                residual_as_cols=residual_as_cols,
                target_col=target_col,
                c_scaler_source=c_scaler_source,
                as_scaler_source=as_scaler_source,
                y_res_scaler_source=y_res_scaler_source,
                residual_lstm=residual_lstm,
                q_models=q_models_source,
                window_size=window_size,
            )
        elif variant == "gt_bias_only":
            y_base_raw = np.asarray(g_model_source.predict(C_test_scaled), dtype=float) + b_t
            pred_df, metrics = evaluate_with_custom_g_predictions(
                variant_name=variant,
                df=target_test_df,
                y_base_raw=y_base_raw,
                c_cols=c_cols,
                residual_as_cols=residual_as_cols,
                target_col=target_col,
                c_scaler_source=c_scaler_source,
                as_scaler_source=as_scaler_source,
                y_res_scaler_source=y_res_scaler_source,
                residual_lstm=residual_lstm,
                q_models=q_models_source,
                window_size=window_size,
            )
        elif variant == "gt_ridge_current":
            y_base_raw = np.asarray(g_t_ridge.predict(C_test_scaled), dtype=float)
            pred_df, metrics = evaluate_with_custom_g_predictions(
                variant_name=variant,
                df=target_test_df,
                y_base_raw=y_base_raw,
                c_cols=c_cols,
                residual_as_cols=residual_as_cols,
                target_col=target_col,
                c_scaler_source=c_scaler_source,
                as_scaler_source=as_scaler_source,
                y_res_scaler_source=y_res_scaler_source,
                residual_lstm=residual_lstm,
                q_models=q_models_source,
                window_size=window_size,
            )
        elif variant == "gt_lgbm_temporal":
            y_base_raw = np.asarray(g_t_temporal.predict(temporal_c_test), dtype=float)
            pred_df, metrics = evaluate_with_custom_g_predictions(
                variant_name=variant,
                df=target_test_df,
                y_base_raw=y_base_raw,
                c_cols=c_cols,
                residual_as_cols=residual_as_cols,
                target_col=target_col,
                c_scaler_source=c_scaler_source,
                as_scaler_source=as_scaler_source,
                y_res_scaler_source=y_res_scaler_source,
                residual_lstm=residual_lstm,
                q_models=q_models_source,
                window_size=window_size,
            )
        elif variant == "gt_qt":
            y_base_raw = np.asarray(g_t_current.predict(C_test_scaled), dtype=float)
            pred_df, metrics = evaluate_with_custom_g_predictions(
                variant_name=variant,
                df=target_test_df,
                y_base_raw=y_base_raw,
                c_cols=c_cols,
                residual_as_cols=residual_as_cols,
                target_col=target_col,
                c_scaler_source=c_scaler_source,
                as_scaler_source=as_scaler_source,
                y_res_scaler_source=y_res_scaler_source,
                residual_lstm=residual_lstm,
                q_models=q_t_models,
                window_size=window_size,
            )
        else:
            raise ValueError(f"Unknown variant: {variant}")

        variant_rows.append(
            {
                "variant": variant,
                **metrics,
                "n_calib": int(len(target_calib_df)),
                "n_test": int(len(target_test_df)),
                "calib_ratio": float(args.calib_ratio),
                "test_ratio": float(args.test_ratio),
                "target_name": target_name,
            }
        )
        baseline_rows.append(_build_baseline_diagnostics(variant, pred_df))
        all_pred_frames.append(pred_df)

        logger.info(
            "[%s] MAE=%.4f RMSE=%.4f R2=%.4f mean_error=%.4f residual_bias_proxy=%.4f",
            variant,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["R2"],
            metrics["mean_error"],
            metrics["residual_bias_proxy"],
        )

    if not variant_rows:
        raise RuntimeError("No variant executed successfully")

    metrics_df = pd.DataFrame(variant_rows).sort_values("variant").reset_index(drop=True)
    preds_df = pd.concat(all_pred_frames, ignore_index=True)
    baseline_df = pd.DataFrame(baseline_rows).sort_values("variant").reset_index(drop=True)

    metrics_path = output_dir / "target_calibration_metrics.csv"
    preds_path = output_dir / "target_calibration_predictions.csv"
    baseline_path = output_dir / "target_baseline_diagnostics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    preds_df.to_csv(preds_path, index=False, encoding="utf-8-sig")
    baseline_df.to_csv(baseline_path, index=False, encoding="utf-8-sig")

    cfg_payload = {
        "mode": args.mode,
        "calib_ratio": float(args.calib_ratio),
        "test_ratio": float(args.test_ratio),
        "target_name": target_name,
        "target_data_path": str(target_path),
        "source_model_dir": str(source_model_dir),
        "c_cols": c_cols,
        "residual_as_cols": residual_as_cols,
        "target_col": target_col,
        "window_size": window_size,
        "ridge_alpha": float(args.ridge_alpha),
        "temporal_g_lags": temporal_lags,
        "temporal_g_rollings": temporal_rollings,
        "temporal_g_diffs": temporal_diffs,
        "temporal_g_fill": args.temporal_g_fill,
        "lgbm_calib_conservative": bool(args.lgbm_calib_conservative),
        "random_seed": seed,
        "variants_executed": metrics_df["variant"].tolist(),
        "temporal_feature_names": temporal_feature_names,
        "temporal_use_full_target_context": True,
        "note": "temporal C 特征在 target_full 上先构造后按 split 切分，仅使用 C 的过去信息。",
    }
    config_path = output_dir / "target_calibration_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(cfg_payload, f, ensure_ascii=False, indent=2)

    global_summary_path = _update_global_comparison_csv(output_root, metrics_df)

    logger.info("Saved: %s", metrics_path)
    logger.info("Saved: %s", preds_path)
    logger.info("Saved: %s", baseline_path)
    logger.info("Saved: %s", config_path)
    logger.info("Saved: %s", global_summary_path)

    summary = metrics_df[["variant", "MAE", "RMSE", "R2", "mean_error"]].copy()
    print("\n" + summary.to_string(index=False, justify="left", float_format=lambda x: f"{x:.4f}"))

    source_row = metrics_df.loc[metrics_df["variant"] == "source"]
    if not source_row.empty:
        source_mae = float(source_row.iloc[0]["MAE"])
        for v in ["gt_lgbm_current", "gt_bias_only", "gt_ridge_current", "gt_lgbm_temporal"]:
            vr = metrics_df.loc[metrics_df["variant"] == v]
            if vr.empty:
                continue
            mae = float(vr.iloc[0]["MAE"])
            improve = float("nan") if source_mae == 0 else (source_mae - mae) / source_mae * 100.0
            print(f"{v} vs source: {improve:.1f}%")

    logger.info("Done.")


if __name__ == "__main__":
    main()
