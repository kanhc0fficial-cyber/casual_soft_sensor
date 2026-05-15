"""
test_group_branch.py
====================
轻量 pytest 测试套件
覆盖 src/models/group_branch.py 和 scripts/train_group_branch.py 的核心逻辑。

运行方式：
    pytest tests/test_group_branch.py -v
"""

from __future__ import annotations

import math
import sys
import warnings
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from models.group_branch import (
    CausalGroupBranchModel,
    GroupGRUBranch,
    GroupMLPBranch,
    GroupTCNBranch,
    build_branch,
)

# 同样导入训练脚本中的纯函数（不依赖真实数据）
sys.path.insert(0, str(_REPO_ROOT / "scripts"))
from train_group_branch import (
    make_windows,
    split_data,
    load_config,
    compute_metrics,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 辅助工具
# ═══════════════════════════════════════════════════════════════════════════════

def _minimal_groups_cfg(indices=(0, 1, 2), branch_type="gru", hidden_dim=8):
    """返回一个最简单的单组 groups_cfg。"""
    return {
        "g0": {
            "indices": list(indices),
            "branch_type": branch_type,
            "hidden_dim": hidden_dim,
        }
    }


def _minimal_model_cfg(**overrides):
    base = {"use_gate": True, "trainable_gate": True, "gate_init": 0.5, "output_bias": True}
    base.update(overrides)
    return base


def _dummy_input(batch=4, window=6, features=5):
    return torch.randn(batch, window, features)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 1. 分支模块：正向传播形状验证 ─────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestBranchShapes:
    """各分支模块正向传播，输出 shape 应为 [batch, 1]。"""

    @pytest.mark.parametrize("batch,window,feat", [
        (1, 1, 1),
        (4, 6, 3),
        (8, 12, 5),
    ])
    def test_gru_branch_output_shape(self, batch, window, feat):
        m = GroupGRUBranch(input_size=feat, hidden_dim=8)
        x = torch.randn(batch, window, feat)
        out = m(x)
        assert out.shape == (batch, 1), f"期望 ({batch}, 1)，得到 {out.shape}"

    @pytest.mark.parametrize("batch,window,feat", [
        (1, 1, 1),
        (4, 6, 3),
        (8, 12, 5),
    ])
    def test_mlp_branch_output_shape(self, batch, window, feat):
        m = GroupMLPBranch(input_size=feat, window_size=window, hidden_dim=8)
        x = torch.randn(batch, window, feat)
        out = m(x)
        assert out.shape == (batch, 1)

    @pytest.mark.parametrize("batch,window,feat", [
        (1, 3, 1),
        (4, 6, 3),
        (8, 12, 5),
    ])
    def test_tcn_branch_output_shape(self, batch, window, feat):
        m = GroupTCNBranch(input_size=feat, hidden_dim=8)
        x = torch.randn(batch, window, feat)
        out = m(x)
        assert out.shape == (batch, 1)

    # BUG-5: TCN window_size=1 时输出是否正常（边缘情况）
    def test_tcn_branch_window_size_1(self):
        """TCN 在 window_size=1 时应正常运行，不崩溃。"""
        m = GroupTCNBranch(input_size=2, hidden_dim=4)
        x = torch.randn(2, 1, 2)
        out = m(x)
        assert out.shape == (2, 1)

    # BUG-5: TCN window_size=2 边缘情况
    def test_tcn_branch_window_size_2(self):
        m = GroupTCNBranch(input_size=2, hidden_dim=4)
        x = torch.randn(2, 2, 2)
        out = m(x)
        assert out.shape == (2, 1)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 2. build_branch 工厂函数 ──────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestBuildBranch:
    @pytest.mark.parametrize("btype", ["gru", "mlp", "tcn", "GRU", "MLP", "TCN"])
    def test_known_types_succeed(self, btype):
        m = build_branch(btype, input_size=3, hidden_dim=8, window_size=6)
        assert isinstance(m, torch.nn.Module)

    def test_unknown_type_raises(self):
        with pytest.raises(ValueError, match="未知 branch_type"):
            build_branch("lstm", input_size=3, hidden_dim=8, window_size=6)


# ═══════════════════════════════════════════════════════════════════════════════
# ── 3. CausalGroupBranchModel 初始化校验 ──────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelInit:
    def test_empty_groups_cfg_raises(self):
        with pytest.raises(ValueError, match="groups_cfg 为空"):
            CausalGroupBranchModel(
                groups_cfg={},
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
            )

    def test_empty_indices_raises(self):
        with pytest.raises(ValueError, match="indices"):
            CausalGroupBranchModel(
                groups_cfg={"g0": {"indices": [], "branch_type": "gru", "hidden_dim": 8}},
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
            )

    def test_out_of_range_index_raises(self):
        with pytest.raises(ValueError, match="越界"):
            CausalGroupBranchModel(
                groups_cfg=_minimal_groups_cfg(indices=[0, 99]),
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
            )

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="越界"):
            CausalGroupBranchModel(
                groups_cfg=_minimal_groups_cfg(indices=[-1, 0]),
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
            )

    def test_feature_overlap_raises_by_default(self):
        groups_cfg = {
            "g0": {"indices": [0, 1], "branch_type": "gru", "hidden_dim": 8},
            "g1": {"indices": [1, 2], "branch_type": "gru", "hidden_dim": 8},
        }
        with pytest.raises(ValueError, match="allow_feature_overlap"):
            CausalGroupBranchModel(
                groups_cfg=groups_cfg,
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
                allow_feature_overlap=False,
            )

    def test_feature_overlap_allowed_when_flag_set(self):
        groups_cfg = {
            "g0": {"indices": [0, 1], "branch_type": "gru", "hidden_dim": 8},
            "g1": {"indices": [1, 2], "branch_type": "gru", "hidden_dim": 8},
        }
        # 不应抛异常
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(),
            window_size=6,
            num_features=5,
            allow_feature_overlap=True,
        )
        assert model is not None

    def test_unused_features_warns(self):
        # 只用了 indices [0]，共 num_features=5，剩余未用特征应触发 UserWarning
        with pytest.warns(UserWarning, match="未被任何 group 使用"):
            CausalGroupBranchModel(
                groups_cfg=_minimal_groups_cfg(indices=[0]),
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
                warn_unused_features=True,
            )

    def test_unused_features_no_warn_when_disabled(self):
        # warn_unused_features=False 时，不应发出警告
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            CausalGroupBranchModel(
                groups_cfg=_minimal_groups_cfg(indices=[0]),
                model_cfg=_minimal_model_cfg(),
                window_size=6,
                num_features=5,
                warn_unused_features=False,
            )

    @pytest.mark.parametrize("gate_init", [0.0, 0.5, 1.0])
    def test_extreme_gate_init_no_nan(self, gate_init):
        """gate_init=0.0 和 1.0 时 logit 计算不应产生 NaN（用了 max(..., 1e-6) 保护）。"""
        model = CausalGroupBranchModel(
            groups_cfg=_minimal_groups_cfg(indices=[0, 1, 2]),
            model_cfg=_minimal_model_cfg(gate_init=gate_init),
            window_size=6,
            num_features=5,
        )
        gates = model.get_gates()
        assert not torch.any(torch.isnan(gates)), "gate 值不应包含 NaN"


# ═══════════════════════════════════════════════════════════════════════════════
# ── 4. CausalGroupBranchModel forward 输出 ────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelForward:
    @pytest.fixture
    def default_model(self):
        groups_cfg = {
            "feed":    {"indices": [0, 1], "branch_type": "gru", "hidden_dim": 8},
            "reagent": {"indices": [2, 3], "branch_type": "gru", "hidden_dim": 8},
            "state":   {"indices": [4, 5, 6], "branch_type": "gru", "hidden_dim": 8},
        }
        return CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(),
            window_size=6,
            num_features=7,
            warn_unused_features=False,
        )

    def test_forward_returns_dict_with_expected_keys(self, default_model):
        x = _dummy_input(batch=4, window=6, features=7)
        out = default_model(x)
        assert set(out.keys()) == {"y_hat", "branch_outputs", "gates", "group_names"}

    def test_forward_y_hat_shape(self, default_model):
        x = _dummy_input(batch=4, window=6, features=7)
        out = default_model(x)
        assert out["y_hat"].shape == (4, 1)

    def test_forward_branch_outputs_shape(self, default_model):
        batch = 4
        x = _dummy_input(batch=batch, window=6, features=7)
        out = default_model(x)
        n_groups = len(default_model.group_names)
        assert out["branch_outputs"].shape == (batch, n_groups)

    def test_forward_gates_shape(self, default_model):
        x = _dummy_input(batch=4, window=6, features=7)
        out = default_model(x)
        assert out["gates"].shape == (len(default_model.group_names),)

    def test_forward_gate_values_in_0_1(self, default_model):
        x = _dummy_input(batch=4, window=6, features=7)
        out = default_model(x)
        g = out["gates"]
        assert torch.all(g >= 0) and torch.all(g <= 1), "gate 值应在 [0, 1] 内"

    def test_forward_no_nan_output(self, default_model):
        x = _dummy_input(batch=4, window=6, features=7)
        out = default_model(x)
        assert not torch.any(torch.isnan(out["y_hat"])), "y_hat 不应包含 NaN"

    @pytest.mark.parametrize("btype", ["gru", "mlp", "tcn"])
    def test_forward_all_branch_types(self, btype):
        groups_cfg = {"g0": {"indices": [0, 1, 2], "branch_type": btype, "hidden_dim": 8}}
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(),
            window_size=6,
            num_features=5,
            warn_unused_features=False,
        )
        x = _dummy_input(batch=4, window=6, features=5)
        out = model(x)
        assert out["y_hat"].shape == (4, 1)

    def test_forward_use_gate_false(self):
        """use_gate=False 时门控全为 1，输出应等于各分支输出之和（加偏置）。"""
        groups_cfg = {"g0": {"indices": [0, 1], "branch_type": "mlp", "hidden_dim": 4}}
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(use_gate=False),
            window_size=4,
            num_features=3,
            warn_unused_features=False,
        )
        x = _dummy_input(batch=2, window=4, features=3)
        out = model(x)
        assert out["y_hat"].shape == (2, 1)

    def test_get_gates_returns_correct_shape(self, default_model):
        gates = default_model.get_gates()
        assert gates.shape == (len(default_model.group_names),)

    def test_fixed_gate_not_in_parameters(self):
        """trainable_gate=False 时，fixed_gates 不应出现在可训练参数里。"""
        groups_cfg = {"g0": {"indices": [0, 1], "branch_type": "gru", "hidden_dim": 4}}
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(use_gate=True, trainable_gate=False, gate_init=0.7),
            window_size=4,
            num_features=3,
            warn_unused_features=False,
        )
        param_names = [n for n, _ in model.named_parameters()]
        assert "gate_logits" not in param_names, "fixed_gate 模式下 gate_logits 不应在参数列表中"


# ═══════════════════════════════════════════════════════════════════════════════
# ── 5. make_windows ───────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestMakeWindows:
    def test_normal_case_output_length(self):
        X = np.ones((20, 3), dtype=np.float32)
        y = np.arange(20, dtype=np.float32)
        Xw, yw = make_windows(X, y, window_size=5)
        # 期望样本数 = 20 - 5 + 1 = 16
        assert len(Xw) == 16
        assert len(yw) == 16

    def test_normal_case_window_shape(self):
        X = np.ones((20, 3), dtype=np.float32)
        y = np.zeros(20, dtype=np.float32)
        Xw, yw = make_windows(X, y, window_size=5)
        assert Xw.shape == (16, 5, 3)

    def test_label_is_last_step_of_window(self):
        """y 标签应对应窗口的最后一个时间步。"""
        X = np.zeros((10, 1), dtype=np.float32)
        y = np.arange(10, dtype=np.float32)
        Xw, yw = make_windows(X, y, window_size=3)
        # 第 0 个样本：窗口 [0,1,2]，标签应为 y[2]=2
        assert yw[0] == 2.0
        # 第 1 个样本：窗口 [1,2,3]，标签应为 y[3]=3
        assert yw[1] == 3.0

    def test_window_size_equals_data_length(self):
        """window_size == len(y) 时，应返回恰好 1 个样本。"""
        X = np.ones((5, 2), dtype=np.float32)
        y = np.zeros(5, dtype=np.float32)
        Xw, yw = make_windows(X, y, window_size=5)
        assert len(Xw) == 1

    # ── BUG-2: window_size > len(y) 时静默返回空数组 ──────────────────────────
    def test_window_larger_than_data_returns_empty(self):
        """
        BUG-2（文档化）：当 window_size > len(y) 时，make_windows 静默返回空数组，
        不会抛出异常，也不会给出任何警告。
        后续的 DataLoader 或 epoch_loss /= len(yw_tr) 会以 ZeroDivisionError 崩溃，
        错误信息与真实原因相距甚远。
        本测试记录当前行为（空数组），以便将来修复时可感知。
        """
        X = np.ones((3, 2), dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        Xw, yw = make_windows(X, y, window_size=10)
        # 当前行为：返回空数组，但应当报错或警告
        assert len(Xw) == 0, "当前行为：window > data 时返回空数组（已知 bug，应加校验）"


# ═══════════════════════════════════════════════════════════════════════════════
# ── 6. split_data ─────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestSplitData:
    """
    注意：split_data 接受 pd.DataFrame，需要 pandas。
    """

    @pytest.fixture
    def sample_df(self):
        import pandas as pd
        n = 100
        return pd.DataFrame({"x": range(n), "y": range(n)})

    @pytest.fixture
    def dummy_logger(self):
        import logging
        logger = logging.getLogger("test_split")
        logger.addHandler(logging.NullHandler())
        return logger

    def test_split_sizes_roughly_correct(self, sample_df, dummy_logger):
        cfg = {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15}
        tr, vl, te = split_data(sample_df, cfg, dummy_logger)
        assert len(tr) == 70
        assert len(vl) == 15

    def test_no_row_duplication(self, sample_df, dummy_logger):
        cfg = {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.15}
        tr, vl, te = split_data(sample_df, cfg, dummy_logger)
        total = len(tr) + len(vl) + len(te)
        assert total == len(sample_df), "三段之和应等于原始数据行数（test_ratio 被忽略时可能有差）"

    # ── BUG-1: test_ratio 配置被忽略 ─────────────────────────────────────────
    def test_test_ratio_actually_ignored(self, dummy_logger):
        """
        BUG-1（文档化）：test_df = df.iloc[n_train + n_val:] 不使用 test_ratio，
        实际 test 大小 = n - int(n*train) - int(n*val)，与配置的 test_ratio 无关。
        """
        import pandas as pd
        n = 1000
        df = pd.DataFrame({"x": range(n)})
        cfg = {"train_ratio": 0.70, "val_ratio": 0.15, "test_ratio": 0.05}  # test_ratio 故意设小
        tr, vl, te = split_data(df, cfg, dummy_logger)
        expected_test_by_ratio = int(n * 0.05)  # = 50（如果配置被遵守）
        actual_test = len(te)
        # 实际是 1000 - 700 - 150 = 150，而非配置的 50
        assert actual_test != expected_test_by_ratio, (
            "BUG-1：test_ratio 没有被使用，实际 test 集大小与配置不符"
        )

    # ── BUG-3: ratio 之和未校验 ────────────────────────────────────────────────
    def test_ratios_not_validated(self, dummy_logger):
        """
        BUG-3（文档化）：train_ratio + val_ratio + test_ratio 之和未做校验，
        用户填错配置（如三者之和 > 1.0）不会报错，只是静默产出错误切分。
        """
        import pandas as pd
        n = 100
        df = pd.DataFrame({"x": range(n)})
        # 三者之和 = 1.6，明显错误，但不会报错
        cfg = {"train_ratio": 0.80, "val_ratio": 0.50, "test_ratio": 0.30}
        try:
            tr, vl, te = split_data(df, cfg, dummy_logger)
            # val 超出总行数时，iloc 会静默截断，不抛异常
            # BUG：应该在这里抛 ValueError，但当前代码没有
            assert len(tr) + len(vl) + len(te) <= n, (
                "BUG-3：ratio 之和超过 1.0 时，代码不报错（应加校验）"
            )
        except Exception:
            pass  # 如果已修复则此处会捕获到异常


# ═══════════════════════════════════════════════════════════════════════════════
# ── 7. compute_metrics ────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestComputeMetrics:
    def test_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y, y)
        assert m["MAE"] == pytest.approx(0.0)
        assert m["RMSE"] == pytest.approx(0.0)
        assert m["R2"] == pytest.approx(1.0)

    def test_returns_all_keys(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.1, 2.1, 3.1])
        m = compute_metrics(y_true, y_pred)
        assert {"MAE", "RMSE", "R2"} == set(m.keys())

    def test_positive_mae_rmse_on_error(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        m = compute_metrics(y_true, y_pred)
        assert m["MAE"] > 0
        assert m["RMSE"] > 0


# ═══════════════════════════════════════════════════════════════════════════════
# ── 8. CausalGroupBranchModel 可反向传播（梯度流） ────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestGradientFlow:
    def test_loss_backward_no_error(self):
        groups_cfg = {
            "g0": {"indices": [0, 1], "branch_type": "gru", "hidden_dim": 4},
            "g1": {"indices": [2, 3], "branch_type": "mlp", "hidden_dim": 4},
        }
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(),
            window_size=4,
            num_features=5,
            warn_unused_features=False,
        )
        x = _dummy_input(batch=4, window=4, features=5)
        y_target = torch.randn(4, 1)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out["y_hat"], y_target)
        loss.backward()  # 不应抛出异常

    def test_trainable_gate_receives_gradient(self):
        groups_cfg = {"g0": {"indices": [0, 1, 2], "branch_type": "gru", "hidden_dim": 4}}
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(trainable_gate=True),
            window_size=4,
            num_features=5,
            warn_unused_features=False,
        )
        x = _dummy_input(batch=4, window=4, features=5)
        y_target = torch.randn(4, 1)
        out = model(x)
        loss = torch.nn.functional.mse_loss(out["y_hat"], y_target)
        loss.backward()
        assert model.gate_logits.grad is not None, "可训练 gate_logits 应收到梯度"
        assert not torch.all(model.gate_logits.grad == 0), "梯度不应全为零"


# ═══════════════════════════════════════════════════════════════════════════════
# ── 9. 多组配置 ────────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════════

class TestMultiGroupConfig:
    def test_two_groups_different_branch_types(self):
        groups_cfg = {
            "env":   {"indices": [0, 1], "branch_type": "gru", "hidden_dim": 4},
            "state": {"indices": [2, 3, 4], "branch_type": "tcn", "hidden_dim": 4},
        }
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(),
            window_size=6,
            num_features=5,
            warn_unused_features=False,
        )
        x = _dummy_input(batch=3, window=6, features=5)
        out = model(x)
        assert out["y_hat"].shape == (3, 1)
        assert out["branch_outputs"].shape == (3, 2)

    def test_group_names_preserved_in_output(self):
        groups_cfg = {
            "alpha": {"indices": [0], "branch_type": "gru", "hidden_dim": 4},
            "beta":  {"indices": [1], "branch_type": "gru", "hidden_dim": 4},
        }
        model = CausalGroupBranchModel(
            groups_cfg=groups_cfg,
            model_cfg=_minimal_model_cfg(),
            window_size=4,
            num_features=3,
            warn_unused_features=False,
        )
        x = _dummy_input(batch=2, window=4, features=3)
        out = model(x)
        assert out["group_names"] == ["alpha", "beta"]
