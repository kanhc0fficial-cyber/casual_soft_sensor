from __future__ import annotations

import logging
import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

from train_dml_residual_soft_sensor import load_data, make_windows, split_data


@pytest.fixture
def dummy_logger():
    logger = logging.getLogger("test_dml_residual")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    return logger


def test_load_data_tsv_uses_tab_separator(tmp_path: Path, dummy_logger):
    tsv_path = tmp_path / "sample.tsv"
    tsv_path.write_text("a\tb\n1\t2\n3\t4\n", encoding="utf-8")
    cfg = {"data_path": str(tsv_path), "allow_synthetic_demo": False}
    df = load_data(cfg, dummy_logger)
    assert list(df.columns) == ["a", "b"]
    assert df.shape == (2, 2)


def test_split_data_ratio_sum_must_be_one(dummy_logger):
    df = pd.DataFrame({"x": range(10)})
    cfg = {"train_ratio": 0.8, "val_ratio": 0.3, "test_ratio": 0.1}
    with pytest.raises(ValueError, match="之和应为 1.0"):
        split_data(df, cfg, dummy_logger)


def test_make_windows_raises_when_window_too_large():
    X = np.ones((5, 2), dtype=np.float32)
    y = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError, match="window_size=6 大于数据长度 5"):
        make_windows(X, y, window_size=6)


def test_make_windows_raises_when_window_non_positive():
    X = np.ones((5, 2), dtype=np.float32)
    y = np.ones(5, dtype=np.float32)
    with pytest.raises(ValueError, match="必须为正整数"):
        make_windows(X, y, window_size=0)
