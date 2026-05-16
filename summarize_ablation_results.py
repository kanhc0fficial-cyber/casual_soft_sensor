#!/usr/bin/env python3
"""Summarize constraint ablation experiment results."""

from pathlib import Path
import sys

import pandas as pd


def main() -> int:
    experiments = [
        ("baseline_no_constraints", "results/ablation_constraints/baseline_no_constraints"),
        ("counterfactual_only", "results/ablation_constraints/counterfactual_only"),
        ("process_only", "results/ablation_constraints/process_only"),
        ("both_constraints", "results/ablation_constraints/both_constraints"),
    ]

    print("=" * 80)
    print("Constraint ablation result summary")
    print("=" * 80)
    print()

    results = []

    for exp_name, exp_dir in experiments:
        exp_path = Path(exp_dir)
        metrics_file = exp_path / "metrics_compare.csv"

        if not metrics_file.exists():
            print(f"[PENDING] {exp_name}: metrics_compare.csv not found")
            results.append({
                "experiment": exp_name,
                "MAE": "N/A",
                "RMSE": "N/A",
                "R2": "N/A",
                "cf_loss_mean": "N/A",
                "process_loss_mean": "N/A",
                "cf_rules": "N/A",
                "process_rules": "N/A",
                "status": "pending",
            })
            continue

        try:
            metrics_df = pd.read_csv(metrics_file)
            model3 = metrics_df[metrics_df["model_name"] == "dml_residual_lstm"]
            if model3.empty:
                raise ValueError("dml_residual_lstm row not found")

            loss_file = exp_path / "constraint_loss_log_model3.csv"
            loss_df = pd.read_csv(loss_file) if loss_file.exists() else pd.DataFrame()
            cf_loss_mean = (
                float(loss_df["train_cf_loss"].mean())
                if "train_cf_loss" in loss_df.columns and not loss_df.empty
                else 0.0
            )
            process_loss_mean = (
                float(loss_df["train_process_loss"].mean())
                if "train_process_loss" in loss_df.columns and not loss_df.empty
                else 0.0
            )

            cf_file = exp_path / "counterfactual_violation_metrics_model3.csv"
            process_file = exp_path / "process_constraint_screening_model3.csv"
            cf_rules = len(pd.read_csv(cf_file)) if cf_file.exists() else 0
            process_rules = len(pd.read_csv(process_file)) if process_file.exists() else 0

            mae = float(model3["MAE"].iloc[0])
            rmse = float(model3["RMSE"].iloc[0])
            r2 = float(model3["R2"].iloc[0])

            print(f"[OK] {exp_name}: MAE={mae:.6f}, RMSE={rmse:.6f}, R2={r2:.6f}")
            print(
                f"     mean losses: cf={cf_loss_mean:.6f}, "
                f"process={process_loss_mean:.6f}; rules: cf={cf_rules}, process={process_rules}"
            )
            print()

            results.append({
                "experiment": exp_name,
                "MAE": f"{mae:.6f}",
                "RMSE": f"{rmse:.6f}",
                "R2": f"{r2:.6f}",
                "cf_loss_mean": f"{cf_loss_mean:.6f}",
                "process_loss_mean": f"{process_loss_mean:.6f}",
                "cf_rules": cf_rules,
                "process_rules": process_rules,
                "status": "completed",
            })

        except Exception as e:
            print(f"[ERROR] {exp_name}: failed to read results - {e}")
            results.append({
                "experiment": exp_name,
                "MAE": "N/A",
                "RMSE": "N/A",
                "R2": "N/A",
                "cf_loss_mean": "N/A",
                "process_loss_mean": "N/A",
                "cf_rules": "N/A",
                "process_rules": "N/A",
                "status": "error",
            })

    results_df = pd.DataFrame(results)
    output_file = Path("results/ablation_constraints/summary.csv")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print(f"Saved summary to {output_file}")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print()

    completed = sum(1 for r in results if r["status"] == "completed")
    total = len(results)
    if completed == total:
        print(f"[OK] All {total} experiments are completed.")
        return 0

    print(f"[PENDING] {completed}/{total} experiments completed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
