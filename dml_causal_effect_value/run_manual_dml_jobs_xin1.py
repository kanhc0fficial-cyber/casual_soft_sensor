#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于人工任务表执行批量 DML（xin1）。
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import dml_causal_effect_value.run_dml_xin2 as dml_core

DEFAULT_TARGET = "y_fx_xin1"
DEFAULT_JOBS = _PROJECT_ROOT / "data" / "features" / "manual_dml_jobs_xin1.csv"
DEFAULT_N_FOLDS = 5
LEAKAGE_KEYWORDS = (
    "target",
    "label",
    "predict",
    "pred",
    "forecast",
    "future",
    "result",
)


class RunLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.log_path.open("w", encoding="utf-8")

    def log(self, msg: str):
        print(msg)
        self._fp.write(msg + "\n")
        self._fp.flush()

    def close(self):
        self._fp.close()


def split_candidates(raw: object) -> list[str]:
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    parts = [p.strip() for p in str(raw).split(";")]
    return [p for p in parts if p]


def safe_int(value: object, default: int = 0) -> int:
    num = pd.to_numeric(value, errors="coerce")
    if pd.isna(num):
        return default
    return int(num)


def normalize_pattern(candidate: str) -> str:
    return re.sub(r"\{[^{}]+\}", "*", candidate)


def has_wildcard(candidate: str) -> bool:
    return any(ch in candidate for ch in "*?[]") or "{" in candidate


def match_candidate(candidate: str, columns: list[str]) -> list[str]:
    if not candidate:
        return []
    norm = normalize_pattern(candidate)
    if has_wildcard(candidate) or norm != candidate:
        return [c for c in columns if fnmatch.fnmatch(c, norm)]
    return [candidate] if candidate in columns else []


def resolve_treatment(candidates: list[str], columns: list[str]) -> tuple[str | None, list[str]]:
    missing = []
    for cand in candidates:
        matches = match_candidate(cand, columns)
        if matches:
            return matches[0], missing
        missing.append(cand)
    return None, missing


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for it in items:
        if it not in seen:
            seen.add(it)
            out.append(it)
    return out


def resolve_adjustments(candidates: list[str], columns: list[str]) -> tuple[list[str], list[str]]:
    used = []
    missing = []
    for cand in candidates:
        matches = match_candidate(cand, columns)
        if matches:
            used.extend(matches)
        else:
            missing.append(cand)
    return dedupe_keep_order(used), missing


def remove_forbidden(cols: list[str], forbidden_patterns: list[str]) -> tuple[list[str], list[str]]:
    if not forbidden_patterns:
        return cols, []
    removed = []
    kept = []
    for col in cols:
        ban = False
        for pat in forbidden_patterns:
            norm = normalize_pattern(pat)
            if fnmatch.fnmatch(col, norm):
                ban = True
                break
        if ban:
            removed.append(col)
        else:
            kept.append(col)
    return kept, dedupe_keep_order(removed)


def is_leakage_col(col: str, target: str) -> bool:
    c = col.lower()
    target_l = target.lower()
    if c == target_l:
        return True
    if c.startswith("y_"):
        return True
    if c.startswith("lab_flo_"):
        return True
    if target_l in c:
        return True
    return any(k in c for k in LEAKAGE_KEYWORDS)


def filter_adjustment_cols(
    adjustment_cols: list[str],
    target: str,
    treatment_col: str,
    treatment_candidates: list[str],
    treatment_lag_col: str,
) -> list[str]:
    flattened_family = set()
    for cand in treatment_candidates:
        for item in match_candidate(cand, adjustment_cols):
            flattened_family.add(item)

    out = []
    for col in adjustment_cols:
        if col == target or col == treatment_col or col == treatment_lag_col:
            continue
        if col in flattened_family:
            continue
        if is_leakage_col(col, target):
            continue
        out.append(col)
    return dedupe_keep_order(out)


def infer_lag_steps(df: pd.DataFrame, lag_min: int, logger: RunLogger, job_id: str) -> int:
    lag_min = max(int(lag_min), 1)
    if "t" not in df.columns:
        logger.log(f"[WARNING] job={job_id} 无 t 列，按 1 行=1分钟，lag_steps={lag_min}")
        return lag_min

    t_series = pd.to_numeric(df["t"], errors="coerce")
    diffs = np.diff(t_series.to_numpy(dtype=np.float64))
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        logger.log(f"[WARNING] job={job_id} t 列无法推断采样间隔，按 1 行=1分钟，lag_steps={lag_min}")
        return lag_min

    median_dt = float(np.median(diffs))
    if median_dt <= 0:
        logger.log(f"[WARNING] job={job_id} t 列采样间隔异常，按 1 行=1分钟，lag_steps={lag_min}")
        return lag_min

    steps = int(round(lag_min / median_dt))
    steps = max(steps, 1)
    return steps


def prepare_dataset(dataset_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(dataset_path)
    if "t" in df.columns:
        df = df.sort_values("t").reset_index(drop=True)
    else:
        df = df.sort_index().reset_index(drop=True)
    return df


def run_single_job(
    row: pd.Series,
    df: pd.DataFrame,
    target: str,
    n_folds: int,
    logger: RunLogger,
) -> tuple[dict, dict, pd.DataFrame | None]:
    job = row.to_dict()
    job_id = str(job.get("job_id", ""))
    group = str(job.get("treatment_group", ""))
    effect_type = str(job.get("effect_type", ""))

    base_result = {
        "job_id": job_id,
        "target": str(job.get("target", target)),
        "treatment_group": group,
        "resolved_treatment": "",
        "treatment_candidates": str(job.get("treatment_candidates", "")),
        "treatment_lag_min": safe_int(job.get("treatment_lag_min", 0), default=0),
        "treatment_role": str(job.get("treatment_role", "")),
        "effect_type": effect_type,
        "theta_raw": np.nan,
        "theta_std": np.nan,
        "se_raw": np.nan,
        "se_std": np.nan,
        "ci_lo_raw": np.nan,
        "ci_hi_raw": np.nan,
        "ci_lo_std": np.nan,
        "ci_hi_std": np.nan,
        "ci_cross_zero_raw": np.nan,
        "ci_cross_zero_std": np.nan,
        "n_effective": 0,
        "n_adjustment_cols": 0,
        "adjustment_cols_used": "",
        "adjustment_cols_missing": "",
        "forbidden_cols_removed": "",
        "missing_treatment_candidates": "",
        "status": "failed",
        "error_msg": "",
    }

    candidates = split_candidates(job.get("treatment_candidates"))
    adjustments = split_candidates(job.get("adjustment_candidates"))
    forbidden_patterns = split_candidates(job.get("forbidden_patterns"))
    columns = list(df.columns)

    treatment_col, missing_treat = resolve_treatment(candidates, columns)

    resolved_info = {
        "job_id": job_id,
        "target": str(job.get("target", target)),
        "treatment_group": group,
        "effect_type": effect_type,
        "treatment_candidates": ";".join(candidates),
        "resolved_treatment": treatment_col or "",
        "missing_treatment_candidates": ";".join(missing_treat),
        "treatment_lag_min": int(base_result["treatment_lag_min"]),
        "adjustment_candidates": ";".join(adjustments),
        "adjustment_cols_used": "",
        "adjustment_cols_missing": "",
        "forbidden_patterns": ";".join(forbidden_patterns),
        "forbidden_cols_removed": "",
        "status": "pending",
        "error_msg": "",
    }

    if not treatment_col:
        msg = "no treatment candidate found in dataset"
        base_result["error_msg"] = msg
        base_result["missing_treatment_candidates"] = ";".join(missing_treat)
        resolved_info["status"] = "failed"
        resolved_info["error_msg"] = msg
        logger.log(f"[JOB] {job_id} | group={group} | status=failed | reason={msg}")
        return base_result, resolved_info, None

    lag_steps = infer_lag_steps(df, base_result["treatment_lag_min"], logger, job_id)
    treatment_lag_col = f"{treatment_col}__lag_{lag_steps}"

    adjustment_cols_used, adjustment_cols_missing = resolve_adjustments(adjustments, columns)
    adjustment_cols_used, forbidden_cols_removed = remove_forbidden(adjustment_cols_used, forbidden_patterns)
    adjustment_cols_used = filter_adjustment_cols(
        adjustment_cols_used,
        target=target,
        treatment_col=treatment_col,
        treatment_candidates=candidates,
        treatment_lag_col=treatment_lag_col,
    )

    base_result["resolved_treatment"] = treatment_col
    base_result["missing_treatment_candidates"] = ";".join(missing_treat)
    base_result["adjustment_cols_used"] = ";".join(adjustment_cols_used)
    base_result["adjustment_cols_missing"] = ";".join(adjustment_cols_missing)
    base_result["forbidden_cols_removed"] = ";".join(forbidden_cols_removed)
    base_result["n_adjustment_cols"] = len(adjustment_cols_used)

    resolved_info["resolved_treatment"] = treatment_col
    resolved_info["missing_treatment_candidates"] = ";".join(missing_treat)
    resolved_info["adjustment_cols_used"] = ";".join(adjustment_cols_used)
    resolved_info["adjustment_cols_missing"] = ";".join(adjustment_cols_missing)
    resolved_info["forbidden_cols_removed"] = ";".join(forbidden_cols_removed)

    if target not in df.columns:
        msg = f"target '{target}' not found in dataset"
        base_result["error_msg"] = msg
        resolved_info["status"] = "failed"
        resolved_info["error_msg"] = msg
        logger.log(f"[JOB] {job_id} | group={group} | treatment={treatment_col} | status=failed | reason={msg}")
        return base_result, resolved_info, None

    if len(adjustment_cols_used) == 0:
        msg = "adjustment set empty after candidate resolving and filtering"
        base_result["error_msg"] = msg
        resolved_info["status"] = "failed"
        resolved_info["error_msg"] = msg
        logger.log(f"[JOB] {job_id} | group={group} | treatment={treatment_col} | lag_min={base_result['treatment_lag_min']} | status=failed | reason={msg}")
        return base_result, resolved_info, None

    try:
        Y = pd.to_numeric(df[target], errors="coerce")
        T_raw = pd.to_numeric(df[treatment_col], errors="coerce")
        T_lag = T_raw.shift(lag_steps)

        valid_mask = Y.notna() & T_lag.notna()
        n_valid = int(valid_mask.sum())
        if n_valid < n_folds * 10:
            raise ValueError(f"effective sample too small: {n_valid} < {n_folds * 10}")

        X_df = df[adjustment_cols_used].apply(pd.to_numeric, errors="coerce")
        X_raw = X_df.to_numpy(dtype=np.float32)
        X_full, _, _ = dml_core.preprocess_features(X_raw)
        X_full = X_full.astype(np.float32)

        Y_arr = Y.to_numpy(dtype=np.float32)
        T_arr = T_lag.to_numpy(dtype=np.float32)
        Y_arr[~valid_mask.to_numpy()] = np.nan

        dml_result = dml_core.run_dml(Y_arr, T_arr, X_full, n_folds=n_folds)

        valid_idx = dml_result["labeled_global_idx"]
        T_valid = T_arr[valid_idx]
        Y_valid = Y_arr[valid_idx]

        std_t = float(np.nanstd(T_valid))
        std_y = float(np.nanstd(Y_valid))
        if std_t <= 1e-8 or std_y <= 1e-8:
            raise ValueError(f"std too small for standardization: std_t={std_t:.3e}, std_y={std_y:.3e}")

        scale = std_t / std_y
        theta_raw = float(dml_result["theta"])
        se_raw = float(dml_result["se"])
        ci_lo_raw = float(dml_result["ci_lo"])
        ci_hi_raw = float(dml_result["ci_hi"])

        theta_std = theta_raw * scale
        se_std = se_raw * scale
        ci_lo_std = ci_lo_raw * scale
        ci_hi_std = ci_hi_raw * scale

        base_result.update(
            {
                "theta_raw": theta_raw,
                "theta_std": theta_std,
                "se_raw": se_raw,
                "se_std": se_std,
                "ci_lo_raw": ci_lo_raw,
                "ci_hi_raw": ci_hi_raw,
                "ci_lo_std": ci_lo_std,
                "ci_hi_std": ci_hi_std,
                "ci_cross_zero_raw": bool(ci_lo_raw <= 0 <= ci_hi_raw),
                "ci_cross_zero_std": bool(ci_lo_std <= 0 <= ci_hi_std),
                "n_effective": int(dml_result["n_effective"]),
                "status": "success",
                "error_msg": "",
            }
        )
        resolved_info["status"] = "success"

        time_source = df["t"].to_numpy() if "t" in df.columns else np.arange(len(df))
        residuals_df = pd.DataFrame(
            {
                "time": time_source[valid_idx],
                "Y_true": dml_result["Y_residuals"] + dml_result["Y_hat"],
                "Y_hat": dml_result["Y_hat"],
                "Y_residual": dml_result["Y_residuals"],
                "T_true_lag": dml_result["T_residuals"] + dml_result["T_hat"],
                "T_hat": dml_result["T_hat"],
                "T_residual": dml_result["T_residuals"],
            }
        )

        logger.log(
            f"[JOB] {job_id} | group={group} | treatment={treatment_col} | "
            f"lag_min={base_result['treatment_lag_min']} | effect={effect_type} | "
            f"adj={len(adjustment_cols_used)} | forbidden_removed={len(forbidden_cols_removed)} | "
            f"n_eff={base_result['n_effective']} | theta_raw={theta_raw:.6f} | "
            f"theta_std={theta_std:.6f} | ci_std=[{ci_lo_std:.6f},{ci_hi_std:.6f}] | status=success"
        )
        return base_result, resolved_info, residuals_df

    except Exception as exc:
        msg = str(exc)
        base_result["status"] = "failed"
        base_result["error_msg"] = msg
        resolved_info["status"] = "failed"
        resolved_info["error_msg"] = msg
        logger.log(
            f"[JOB] {job_id} | group={group} | treatment={treatment_col} | "
            f"lag_min={base_result['treatment_lag_min']} | effect={effect_type} | "
            f"adj={len(adjustment_cols_used)} | forbidden_removed={len(forbidden_cols_removed)} | "
            f"status=failed | error={msg}"
        )
        logger.log(traceback.format_exc().strip())
        return base_result, resolved_info, None


def enrich_and_select(theta_df: pd.DataFrame, n_folds: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = theta_df.copy()
    df["abs_theta_std"] = df["theta_std"].abs()
    df["is_success"] = df["status"] == "success"
    df["is_significant_std"] = df["is_success"] & (~df["ci_cross_zero_std"].astype("boolean").fillna(True))

    df["theta_std_rank_within_group"] = (
        df.groupby("treatment_group")["abs_theta_std"]
        .rank(method="dense", ascending=False)
        .astype("Int64")
    )

    min_effective = max(30, n_folds * 10)
    base_ok = df["is_success"] & (df["n_effective"] >= min_effective) & (~df["ci_cross_zero_std"].astype("boolean").fillna(True))
    df["recommended_for_weight"] = False
    for group, g_idx in df.groupby("treatment_group").groups.items():
        g = df.loc[list(g_idx)]
        g_ok = g[base_ok.loc[g.index]]
        if len(g_ok) == 0:
            continue
        top_idx = g_ok["abs_theta_std"].idxmax()
        df.loc[top_idx, "recommended_for_weight"] = True

    selected_rows = []
    for group, g in df.groupby("treatment_group", dropna=False):
        g2 = g.copy()
        sig = g2[g2["is_success"] & (~g2["ci_cross_zero_std"].astype("boolean").fillna(True))]
        if len(sig) > 0:
            chosen = sig.loc[sig["abs_theta_std"].idxmax()]
            recommended = bool(chosen["recommended_for_weight"])
            reason = "selected max abs(theta_std) among significant successful jobs"
        else:
            success = g2[g2["is_success"]]
            if len(success) > 0:
                chosen = success.loc[success["abs_theta_std"].idxmax()]
                recommended = False
                reason = "no significant job in group; selected max abs(theta_std) fallback"
            else:
                chosen = g2.iloc[0]
                recommended = False
                reason = "no successful job in group"

        selected_rows.append(
            {
                "treatment_group": group,
                "selected_job_id": chosen["job_id"],
                "resolved_treatment": chosen["resolved_treatment"],
                "selected_lag_min": chosen["treatment_lag_min"],
                "theta_raw": chosen["theta_raw"],
                "theta_std": chosen["theta_std"],
                "ci_lo_std": chosen["ci_lo_std"],
                "ci_hi_std": chosen["ci_hi_std"],
                "recommended_for_weight": recommended,
                "reason": reason,
            }
        )

    selected_df = pd.DataFrame(selected_rows)
    return df, selected_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run manual DML jobs for xin1 using curated adjustment sets")
    parser.add_argument("--dataset", required=True, help="Parquet dataset path")
    parser.add_argument("--jobs", default=str(DEFAULT_JOBS), help="Manual DML jobs CSV path")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Target column name")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS, help="Cross-fitting folds")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-train", type=int, default=None, help="Override HistGBDT max training samples")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_path = Path(args.dataset)
    jobs_path = Path(args.jobs)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = _THIS_DIR / "结果" / f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(out_dir / "run_log.txt")

    try:
        logger.log("=" * 80)
        logger.log("Manual DML batch run (xin1)")
        logger.log(f"dataset={dataset_path}")
        logger.log(f"jobs={jobs_path}")
        logger.log(f"target={args.target}")
        logger.log(f"n_folds={args.n_folds}")
        logger.log(f"output_dir={out_dir}")
        logger.log("=" * 80)

        if not dataset_path.exists():
            raise FileNotFoundError(f"dataset not found: {dataset_path}")
        if not jobs_path.exists():
            raise FileNotFoundError(f"jobs csv not found: {jobs_path}")

        if args.max_train is not None and args.max_train > 0:
            dml_core.HGBDT_MAX_TRAIN = int(args.max_train)
            logger.log(f"override HGBDT_MAX_TRAIN={dml_core.HGBDT_MAX_TRAIN}")

        df = prepare_dataset(dataset_path)
        jobs_df = pd.read_csv(jobs_path, encoding="utf-8-sig")

        if "target" in jobs_df.columns:
            jobs_df = jobs_df[jobs_df["target"].astype(str) == args.target].reset_index(drop=True)

        total_jobs = len(jobs_df)
        logger.log(f"total_jobs={total_jobs}")

        all_results = []
        resolved_rows = []

        for _, row in jobs_df.iterrows():
            res, resolved, residuals = run_single_job(
                row=row,
                df=df,
                target=args.target,
                n_folds=args.n_folds,
                logger=logger,
            )
            all_results.append(res)
            resolved_rows.append(resolved)
            if residuals is not None:
                job_id = str(res.get("job_id", "unknown"))
                residuals.to_csv(out_dir / f"residuals_{job_id}.csv", index=False, encoding="utf-8-sig")

        if total_jobs == 0:
            logger.log("[WARNING] no jobs matched target filter, writing empty outputs")
            theta_df = pd.DataFrame(
                columns=[
                    "job_id", "target", "treatment_group", "resolved_treatment", "treatment_candidates",
                    "treatment_lag_min", "treatment_role", "effect_type",
                    "theta_raw", "theta_std", "se_raw", "se_std",
                    "ci_lo_raw", "ci_hi_raw", "ci_lo_std", "ci_hi_std",
                    "ci_cross_zero_raw", "ci_cross_zero_std",
                    "n_effective", "n_adjustment_cols", "adjustment_cols_used", "adjustment_cols_missing",
                    "forbidden_cols_removed", "missing_treatment_candidates", "status", "error_msg",
                    "abs_theta_std", "is_success", "is_significant_std",
                    "theta_std_rank_within_group", "recommended_for_weight",
                ]
            )
            resolved_df = pd.DataFrame(
                columns=[
                    "job_id", "target", "treatment_group", "effect_type", "treatment_candidates",
                    "resolved_treatment", "missing_treatment_candidates", "treatment_lag_min",
                    "adjustment_candidates", "adjustment_cols_used", "adjustment_cols_missing",
                    "forbidden_patterns", "forbidden_cols_removed", "status", "error_msg",
                ]
            )
            selected_df = pd.DataFrame(
                columns=[
                    "treatment_group", "selected_job_id", "resolved_treatment", "selected_lag_min",
                    "theta_raw", "theta_std", "ci_lo_std", "ci_hi_std",
                    "recommended_for_weight", "reason",
                ]
            )
        else:
            theta_df = pd.DataFrame(all_results)
            resolved_df = pd.DataFrame(resolved_rows)
            theta_df, selected_df = enrich_and_select(theta_df, n_folds=args.n_folds)

        theta_path = out_dir / "manual_dml_theta_xin1.csv"
        resolved_path = out_dir / "manual_dml_jobs_resolved.csv"
        selected_path = out_dir / "manual_dml_theta_selected_for_weight.csv"

        theta_df.to_csv(theta_path, index=False, encoding="utf-8-sig")
        resolved_df.to_csv(resolved_path, index=False, encoding="utf-8-sig")
        selected_df.to_csv(selected_path, index=False, encoding="utf-8-sig")

        success_n = int((theta_df["status"] == "success").sum())
        failed_n = int((theta_df["status"] == "failed").sum())

        logger.log("=" * 80)
        logger.log(f"success_jobs={success_n}")
        logger.log(f"failed_jobs={failed_n}")
        for _, r in selected_df.iterrows():
            logger.log(
                f"[SELECT] group={r['treatment_group']} job={r['selected_job_id']} "
                f"lag={r['selected_lag_min']} recommended={r['recommended_for_weight']}"
            )
        logger.log(f"output_dir={out_dir}")
        logger.log(f"manual_dml_theta_xin1.csv={theta_path}")
        logger.log(f"manual_dml_theta_selected_for_weight.csv={selected_path}")
        logger.log("=" * 80)

    finally:
        logger.close()


if __name__ == "__main__":
    main()
