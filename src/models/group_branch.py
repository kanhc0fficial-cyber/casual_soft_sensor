"""
group_branch.py
===============
工艺因果组分支软测量模型 (Process-Causal Group Branch Soft Sensor)

核心思想：
    普通模型：y_hat = f(X_all)
    分支模型：z_k = f_k(X_group_k),  y_hat = bias + Σ gate_k * z_k

支持的分支类型：gru | mlp | tcn
通过配置文件（YAML）控制分组方案和超参，不需要修改本文件。

输入约定：X: [batch_size, window_size, num_features]
"""

from __future__ import annotations

import math
import warnings
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── 分支模块 ──────────────────────────────────────────────────────────────────

class GroupGRUBranch(nn.Module):
    """
    GRU 分支。
    输入：[batch, window_size, group_feature_num]
    输出：[batch, 1]
    """

    def __init__(self, input_size: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, feat]
        _, h = self.gru(x)        # h: [1, batch, hidden]
        return self.fc(h.squeeze(0))  # [batch, 1]


class GroupMLPBranch(nn.Module):
    """
    MLP 分支：先将时间窗口展平，再通过两层 MLP。
    输入：[batch, window_size, group_feature_num]
    输出：[batch, 1]
    """

    def __init__(self, input_size: int, window_size: int, hidden_dim: int = 32) -> None:
        super().__init__()
        flat_dim = input_size * window_size
        self.net = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, feat] -> flatten -> [batch, window*feat]
        batch = x.size(0)
        return self.net(x.reshape(batch, -1))  # [batch, 1]


class GroupTCNBranch(nn.Module):
    """
    轻量 TCN 分支：两层因果卷积，取最后时间步输出。
    输入：[batch, window_size, group_feature_num]
    输出：[batch, 1]
    """

    def __init__(self, input_size: int, hidden_dim: int = 32) -> None:
        super().__init__()
        # padding=0：在 forward 中使用显式左侧填充（kernel_size-1=2），保证因果性
        self.conv1 = nn.Conv1d(input_size, hidden_dim, kernel_size=3, padding=0)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=0)
        self.fc = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, window, feat] -> [batch, feat, window]
        x = x.permute(0, 2, 1)
        # 显式左侧填充 kernel_size-1=2，不填充右侧，保证卷积只看过去的时间步
        x = F.pad(x, (2, 0))
        x = self.relu(self.conv1(x))   # [batch, hidden, window]
        x = F.pad(x, (2, 0))
        x = self.relu(self.conv2(x))   # [batch, hidden, window]
        x = x[:, :, -1]               # [batch, hidden] 取最后时间步
        return self.fc(x)              # [batch, 1]


# ─── 工厂函数 ──────────────────────────────────────────────────────────────────

_BRANCH_REGISTRY = {
    "gru": GroupGRUBranch,
    "mlp": GroupMLPBranch,
    "tcn": GroupTCNBranch,
}


def build_branch(
    branch_type: str,
    input_size: int,
    hidden_dim: int,
    window_size: int,
) -> nn.Module:
    """
    工厂函数：根据 branch_type 创建对应分支模块。

    Args:
        branch_type: "gru" | "mlp" | "tcn"
        input_size:  该组的特征数量（= len(indices)）
        hidden_dim:  隐藏层维度
        window_size: 时间窗口大小（MLP 分支需要）
    """
    btype = branch_type.lower()
    if btype not in _BRANCH_REGISTRY:
        raise ValueError(
            f"未知 branch_type='{branch_type}'，"
            f"支持: {list(_BRANCH_REGISTRY.keys())}"
        )
    cls = _BRANCH_REGISTRY[btype]
    if btype == "mlp":
        return cls(input_size, window_size, hidden_dim)
    return cls(input_size, hidden_dim)


# ─── 主模型 ────────────────────────────────────────────────────────────────────

class CausalGroupBranchModel(nn.Module):
    """
    工艺因果组分支软测量模型。

    根据 groups_cfg 为每个变量组创建独立分支，最终加权求和：
        y_hat = bias + Σ gate_k * z_k

    Args:
        groups_cfg: 各组配置，格式：
            {group_name: {indices: [...], branch_type: str, hidden_dim: int}}
        model_cfg: 模型级配置：
            use_gate (bool): 是否启用门控，默认 True
            trainable_gate (bool): 门控是否可训练，默认 True
            gate_init (float): 门控初始值（sigmoid 前），默认 0.5
            output_bias (bool): 是否使用输出偏置，默认 True
        window_size: 滑动窗口大小
        num_features: 输入特征总数（用于越界检查）
        allow_feature_overlap: 同一特征是否允许出现在多个 group，默认 False
        warn_unused_features: 有特征未被任何 group 使用时是否发出警告，默认 True
    """

    def __init__(
        self,
        groups_cfg: dict,
        model_cfg: dict,
        window_size: int,
        num_features: int,
        allow_feature_overlap: bool = False,
        warn_unused_features: bool = True,
    ) -> None:
        super().__init__()

        if not groups_cfg:
            raise ValueError("groups_cfg 为空，至少需要定义一个变量组。")

        self.group_names: List[str] = list(groups_cfg.keys())
        self.group_indices: Dict[str, List[int]] = {}
        self.use_gate: bool = bool(model_cfg.get("use_gate", True))
        self.trainable_gate: bool = bool(model_cfg.get("trainable_gate", True))
        self.output_bias_flag: bool = bool(model_cfg.get("output_bias", True))
        gate_init: float = float(model_cfg.get("gate_init", 0.5))
        if self.use_gate and not (0.0 <= gate_init <= 1.0):
            raise ValueError(
                f"gate_init={gate_init} 非法，启用 gate 时取值范围应在 [0, 1]。"
            )

        # ── 配置校验 ─────────────────────────────────────────────────────────
        seen: Dict[int, str] = {}  # index -> first group that used it
        for name, gcfg in groups_cfg.items():
            idxs: List[int] = list(gcfg.get("indices", []))
            if not idxs:
                raise ValueError(f"Group '{name}' 的 indices 列表为空。")
            for idx in idxs:
                if not isinstance(idx, int) or idx < 0 or idx >= num_features:
                    raise ValueError(
                        f"Group '{name}' 的 index={idx} 越界，"
                        f"合法范围为 [0, {num_features - 1}]。"
                    )
                if not allow_feature_overlap and idx in seen:
                    raise ValueError(
                        f"特征索引 {idx} 同时出现在 group '{seen[idx]}' 和 "
                        f"group '{name}' 中（allow_feature_overlap=False）。"
                    )
                seen[idx] = name
            self.group_indices[name] = idxs

        # 检查未被任何 group 使用的特征
        if warn_unused_features:
            used = set(seen.keys())
            unused = [i for i in range(num_features) if i not in used]
            if unused:
                warnings.warn(
                    f"以下特征索引未被任何 group 使用：{unused}",
                    UserWarning,
                    stacklevel=2,
                )

        # ── 创建分支 ──────────────────────────────────────────────────────────
        branches: Dict[str, nn.Module] = {}
        for name, gcfg in groups_cfg.items():
            idxs = self.group_indices[name]
            btype = str(gcfg.get("branch_type", "gru"))
            hdim = int(gcfg.get("hidden_dim", 32))
            branches[name] = build_branch(btype, len(idxs), hdim, window_size)
        self.branches = nn.ModuleDict(branches)

        # ── 门控参数 ──────────────────────────────────────────────────────────
        n_groups = len(self.group_names)
        if self.use_gate and self.trainable_gate:
            # 将 gate_init 映射到 logit 空间：logit = ln(p / (1-p))
            init_logit = math.log(max(gate_init, 1e-6) / max(1.0 - gate_init, 1e-6))
            self.gate_logits = nn.Parameter(torch.full((n_groups,), init_logit))
        elif self.use_gate and not self.trainable_gate:
            # 固定门控，不参与梯度计算
            self.register_buffer(
                "fixed_gates", torch.full((n_groups,), gate_init)
            )

        # ── 输出偏置 ──────────────────────────────────────────────────────────
        if self.output_bias_flag:
            self.bias = nn.Parameter(torch.zeros(1))

    # ------------------------------------------------------------------

    def get_gates(self) -> torch.Tensor:
        """返回当前门控值（概率空间，[num_groups]）。"""
        if self.use_gate:
            if self.trainable_gate:
                return torch.sigmoid(self.gate_logits)
            else:
                return self.fixed_gates  # type: ignore[return-value]
        return torch.ones(len(self.group_names))

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: [batch_size, window_size, num_features]

        Returns:
            dict:
                y_hat:          [batch, 1]
                branch_outputs: [batch, num_groups]  各分支的原始输出
                gates:          [num_groups]          门控值（已过 sigmoid）
                group_names:    List[str]             分组名称列表
        """
        branch_outputs: List[torch.Tensor] = []
        for name in self.group_names:
            idxs = self.group_indices[name]
            x_group = x[:, :, idxs]                    # [batch, window, g_feat]
            z_k = self.branches[name](x_group)          # [batch, 1]
            branch_outputs.append(z_k)

        # [batch, num_groups]
        Z = torch.cat(branch_outputs, dim=1)

        # 门控 [num_groups]
        gates = self.get_gates().to(x.device)

        # 加权求和 -> [batch, 1]
        y_hat = (Z * gates.unsqueeze(0)).sum(dim=1, keepdim=True)

        if self.output_bias_flag:
            y_hat = y_hat + self.bias

        return {
            "y_hat": y_hat,
            "branch_outputs": Z,
            "gates": gates.detach(),
            "group_names": self.group_names,
        }
