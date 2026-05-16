from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        choices=["all", "source", "gt_only", "gt_qt"],
        default="all",
    )
    parser.add_argument("--residualization-model", default=None)
    parser.add_argument("--random-seed", type=int, default=None)
    parser.add_argument("--allow-train-source-if-missing", action="store_true")
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
    ckpt = source_model_dir / "dml_residual_lstm_checkpoint.pt"
    components = source_model_dir / "dml_residual_model_components.pkl"
    if ckpt.exists() and components.exists():
        return ckpt, components

    if not args.allow_train_source_if_missing:
        missing = [str(p) for p in [ckpt, components] if not p.exists()]
        raise FileNotFoundError(
            "Missing source Model 3 artifacts and fallback disabled: " + ", ".join(missing)
        )

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
    AS_test_scaled = as_scaler_source.transform(target_test_df[residual_as_cols].values)

    y_base_test = np.asarray(g_model.predict(C_test_scaled), dtype=float)

    AS_res_test = np.empty_like(AS_test_scaled)
    for j, col in enumerate(residual_as_cols):
        if col not in q_models:
            raise KeyError(f"Missing q model for feature '{col}'")
        xj_scaled = AS_test_scaled[:, j]
        xj_hat = np.asarray(q_models[col].predict(C_test_scaled), dtype=float)
        AS_res_test[:, j] = xj_scaled - xj_hat

    dummy_y = np.zeros(len(AS_res_test), dtype=np.float32)
    Xw_test, _ = make_windows(AS_res_test, dummy_y, window_size)

    y_res_pred_scaled = residual_lstm.predict(Xw_test)
    y_res_pred = y_res_scaler_source.inverse_transform(y_res_pred_scaled.reshape(-1, 1)).ravel()

    align_offset = window_size - 1
    y_base_aligned = y_base_test[align_offset: align_offset + len(y_res_pred)]
    y_true_aligned = target_test_df[target_col].values[align_offset: align_offset + len(y_res_pred)]

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
            "index": target_test_df.index[align_offset: align_offset + len(y_pred)],
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


def main() -> None:
    args = _parse_args()
    cfg = load_config(args.config)

    if args.random_seed is not None:
        cfg["random_seed"] = int(args.random_seed)
    if args.residualization_model:
        cfg["residualization_model"] = args.residualization_model

    target_name = args.target_name or _safe_name_from_path(args.target_data_path)

    output_dir = Path(args.output_dir) / target_name
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

    c_scaler_source = components["c_scaler"]
    as_scaler_source = components["as_scaler"]
    y_res_scaler_source = components["y_res_scaler"]
    q_models_source = components["q_models"]
    g_model_source = components.get("g_model")

    model_type = cfg.get("residualization_model", "lightgbm")
    seed = int(cfg.get("random_seed", 42))

    C_calib_scaled = c_scaler_source.transform(target_calib_df[c_cols].values)
    y_calib_raw = target_calib_df[target_col].values
    AS_calib_scaled = as_scaler_source.transform(target_calib_df[residual_as_cols].values)

    g_t = build_residualization_model(model_type, seed)
    g_t.fit(C_calib_scaled, y_calib_raw)

    q_t_models: Dict[str, object] = {}
    for j, col in enumerate(residual_as_cols):
        qm_t = build_residualization_model(model_type, seed + j + 1)
        qm_t.fit(C_calib_scaled, AS_calib_scaled[:, j])
        q_t_models[col] = qm_t

    mode_to_variants = {
        "source": ["source"],
        "gt_only": ["gt_only"],
        "gt_qt": ["gt_qt"],
        "all": ["source", "gt_only", "gt_qt"],
    }

    variant_rows = []
    all_pred_frames = []

    for variant in mode_to_variants[args.mode]:
        if variant == "source":
            if g_model_source is None:
                msg = "g_model missing in source components, source variant cannot run"
                if args.mode == "source":
                    raise ValueError(msg)
                logger.warning(msg)
                continue
            g_model = g_model_source
            q_models = q_models_source
        elif variant == "gt_only":
            g_model = g_t
            q_models = q_models_source
        elif variant == "gt_qt":
            g_model = g_t
            q_models = q_t_models
        else:
            raise ValueError(f"Unknown variant: {variant}")

        pred_df, metrics = _evaluate_variant(
            variant,
            target_test_df,
            c_cols,
            residual_as_cols,
            target_col,
            c_scaler_source,
            as_scaler_source,
            y_res_scaler_source,
            residual_lstm,
            g_model,
            q_models,
            window_size,
        )

        metrics_row = {
            "variant": variant,
            **metrics,
            "n_calib": int(len(target_calib_df)),
            "n_test": int(len(target_test_df)),
            "calib_ratio": float(args.calib_ratio),
            "test_ratio": float(args.test_ratio),
            "target_name": target_name,
        }
        variant_rows.append(metrics_row)
        all_pred_frames.append(pred_df)

        pred_path = output_dir / f"predictions_{variant}.csv"
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")
        logger.info("Saved: %s", pred_path)
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

    metrics_df = pd.DataFrame(variant_rows)
    metrics_df = metrics_df.sort_values("variant").reset_index(drop=True)
    metrics_path = output_dir / "metrics_summary.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    all_preds = pd.concat(all_pred_frames, ignore_index=True)
    all_preds_path = output_dir / "predictions_all_variants.csv"
    all_preds.to_csv(all_preds_path, index=False, encoding="utf-8-sig")

    run_info = {
        "config": str(args.config),
        "source_model_dir": str(source_model_dir),
        "target_data_path": str(target_path),
        "target_name": target_name,
        "mode": args.mode,
        "calib_ratio": float(args.calib_ratio),
        "test_ratio": float(args.test_ratio),
        "window_size": window_size,
        "residualization_model": model_type,
        "random_seed": seed,
        "variants_executed": metrics_df["variant"].tolist(),
        "required_source_components": ["y_res_scaler", "as_scaler", "c_scaler", "q_models", "feature_cols"],
    }
    run_info_path = output_dir / "run_info.json"
    with open(run_info_path, "w", encoding="utf-8") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    logger.info("Saved: %s", metrics_path)
    logger.info("Saved: %s", all_preds_path)
    logger.info("Saved: %s", run_info_path)
    logger.info("Done.")


if __name__ == "__main__":
    main()
