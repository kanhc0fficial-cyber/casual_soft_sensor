from __future__ import annotations

import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from evaluate_dml_residual_target_calibration import (  # noqa: E402
    _evaluate_variant,
    _safe_name_from_path,
    _split_target_by_time,
)


class _IdentityScaler:
    def transform(self, x):
        return np.asarray(x)

    def inverse_transform(self, x):
        return np.asarray(x)


class _LinearGModel:
    def predict(self, c_scaled):
        c_scaled = np.asarray(c_scaled)
        return 2.0 * c_scaled[:, 0]


class _ZeroQModel:
    def predict(self, c_scaled):
        c_scaled = np.asarray(c_scaled)
        return np.zeros(c_scaled.shape[0], dtype=float)


class _ZeroResidualLSTM:
    def predict(self, xw):
        xw = np.asarray(xw)
        return np.zeros(xw.shape[0], dtype=float)


def test_safe_name_from_windows_style_path():
    path = r"C:\Users\goldenwhale\Downloads\xx\T1_high_feed_mild_coarse.parquet"
    assert _safe_name_from_path(path) == "T1_high_feed_mild_coarse"


def test_split_target_by_time_preserves_order_and_sizes():
    df = pd.DataFrame({"x": np.arange(10)})
    calib, test = _split_target_by_time(df, calib_ratio=0.2, test_ratio=0.8)
    assert len(calib) == 2
    assert len(test) == 8
    assert calib["x"].tolist() == [0, 1]
    assert test["x"].tolist() == [2, 3, 4, 5, 6, 7, 8, 9]


def test_evaluate_variant_outputs_expected_columns_and_perfect_scores():
    target_test_df = pd.DataFrame(
        {
            "c1": np.arange(6, dtype=float),
            "a1": np.linspace(0.0, 1.0, 6),
            "s1": np.linspace(1.0, 2.0, 6),
        }
    )
    target_test_df["y"] = 2.0 * target_test_df["c1"]

    pred_df, metrics = _evaluate_variant(
        variant_name="source",
        target_test_df=target_test_df,
        c_cols=["c1"],
        residual_as_cols=["a1", "s1"],
        target_col="y",
        c_scaler_source=_IdentityScaler(),
        as_scaler_source=_IdentityScaler(),
        y_res_scaler_source=_IdentityScaler(),
        residual_lstm=_ZeroResidualLSTM(),
        g_model=_LinearGModel(),
        q_models={"a1": _ZeroQModel(), "s1": _ZeroQModel()},
        window_size=3,
    )

    assert list(pred_df.columns) == [
        "index",
        "variant",
        "y_true",
        "y_base",
        "y_res_pred",
        "y_pred",
        "error",
        "abs_error",
    ]
    assert len(pred_df) == 4
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)
    assert metrics["residual_bias_proxy"] == pytest.approx(0.0)
