"""
causal_discovery_config.py
==========================
因果发现配置模块：数据加载、物理拓扑掩码、产线定义。

关键功能：
  1. prepare_data(line)：从 Parquet 文件加载数据，返回 DataFrame、变量列表、阶段映射、分组映射
  2. can_cause(from_stage, to_stage, from_group, to_group, line)：判断物理因果可行性
  3. 产线定义：xin1 (Group A+C)、xin2 (Group B+C)
"""

import os
import pandas as pd
import numpy as np

# ─── 数据路径配置 ──────────────────────────────────────────────────────────────

# 新的数据来源路径
DATA_PATH = r"C:\Users\goldenwhale\Downloads\my_mining_simulation\output\simulation_2months_upstream_visible_clean_fast_sampling.parquet"

# ─── 产线定义 ──────────────────────────────────────────────────────────────────

# 产线 xin1：Group A + Group C
XINS_GROUPS = {
    "xin1": ["A", "C"],
    "xin2": ["B", "C"],
}

# 工序阶段定义（浮选工艺流程）
STAGES = {
    "粗选": 0,
    "扫选": 1,
    "精选": 2,
    "Y": 3,  # 目标变量（浓度等级）
}

# ─── 变量元数据 ────────────────────────────────────────────────────────────────

# 变量名 → (阶段, 分组) 映射
# 格式：变量名 → ("阶段名", "分组")
# 数据来自浮选工艺，包含两条产线 (s1, s2) 的多个工序单元
VARIABLE_METADATA = {
    # ─── 产线 1 (s1) ───────────────────────────────────────────────────────
    # 粗选单元 (Group A)
    "fx_s1_cx1_froth_h": ("粗选", "A"),
    "fx_s1_cx1_level": ("粗选", "A"),
    "fx_s1_cx1_air_flow": ("粗选", "A"),
    "fx_s1_cx2_froth_h": ("粗选", "A"),
    "fx_s1_cx2_level": ("粗选", "A"),
    "fx_s1_cx2_air_flow": ("粗选", "A"),
    "fx_s1_cx3_froth_h": ("粗选", "A"),
    "fx_s1_cx3_level": ("粗选", "A"),
    "fx_s1_cx3_air_flow": ("粗选", "A"),
    
    # 扫选单元 (Group C)
    "fx_s1_sx1_froth_h": ("扫选", "C"),
    "fx_s1_sx1_level": ("扫选", "C"),
    "fx_s1_sx1_air_flow": ("扫选", "C"),
    "fx_s1_sx2_froth_h": ("扫选", "C"),
    "fx_s1_sx2_level": ("扫选", "C"),
    "fx_s1_sx2_air_flow": ("扫选", "C"),
    "fx_s1_sx3_froth_h": ("扫选", "C"),
    "fx_s1_sx3_level": ("扫选", "C"),
    "fx_s1_sx3_air_flow": ("扫选", "C"),
    
    # 精选单元 (Group C)
    "fx_s1_jx_froth_h": ("精选", "C"),
    "fx_s1_jx_level": ("精选", "C"),
    "fx_s1_jx_air_flow": ("精选", "C"),
    
    # ─── 产线 2 (s2) ───────────────────────────────────────────────────────
    # 粗选单元 (Group B)
    "fx_s2_cx1_froth_h": ("粗选", "B"),
    "fx_s2_cx1_level": ("粗选", "B"),
    "fx_s2_cx1_air_flow": ("粗选", "B"),
    "fx_s2_cx2_froth_h": ("粗选", "B"),
    "fx_s2_cx2_level": ("粗选", "B"),
    "fx_s2_cx2_air_flow": ("粗选", "B"),
    "fx_s2_cx3_froth_h": ("粗选", "B"),
    "fx_s2_cx3_level": ("粗选", "B"),
    "fx_s2_cx3_air_flow": ("粗选", "B"),
    
    # 扫选单元 (Group C)
    "fx_s2_sx1_froth_h": ("扫选", "C"),
    "fx_s2_sx1_level": ("扫选", "C"),
    "fx_s2_sx1_air_flow": ("扫选", "C"),
    "fx_s2_sx2_froth_h": ("扫选", "C"),
    "fx_s2_sx2_level": ("扫选", "C"),
    "fx_s2_sx2_air_flow": ("扫选", "C"),
    "fx_s2_sx3_froth_h": ("扫选", "C"),
    "fx_s2_sx3_level": ("扫选", "C"),
    "fx_s2_sx3_air_flow": ("扫选", "C"),
    
    # 精选单元 (Group C)
    "fx_s2_jx_froth_h": ("精选", "C"),
    "fx_s2_jx_level": ("精选", "C"),
    "fx_s2_jx_air_flow": ("精选", "C"),
}

# ─── 物理因果可行性规则 ────────────────────────────────────────────────────────

def can_cause(from_stage, to_stage, from_group, to_group, line):
    """
    判断物理因果可行性。

    规则：
      1. 同一产线内，前序工序可以影响后序工序（粗选→扫选→精选→Y）
      2. 同一工序内，同组变量可以相互影响（Group A 内部、Group C 内部）
      3. 不同产线的变量不能相互影响
      4. 后序工序不能影响前序工序（无反向因果）
      5. 目标变量 Y 只能被其他变量指向，不能指向任何变量

    参数：
      from_stage:  源变量的工序阶段（"粗选", "扫选", "精选", "Y"）
      to_stage:    目标变量的工序阶段（"粗选", "扫选", "精选", "Y"）
      from_group:  源变量的分组（"A", "B", "C"）
      to_group:    目标变量的分组（"A", "B", "C"）
      line:        产线（"xin1", "xin2"）

    返回：
      True 表示物理上可行，False 表示不可行
    """
    
    # 规则 5：Y 不能指向任何变量
    if from_stage == "Y":
        return False
    
    # 规则 4：后序工序不能影响前序工序
    if STAGES.get(from_stage, -1) > STAGES.get(to_stage, -1):
        return False
    
    # 规则 1：同一产线内，前序工序可以影响后序工序
    if STAGES.get(from_stage, -1) < STAGES.get(to_stage, -1):
        # 检查分组兼容性
        line_groups = XINS_GROUPS.get(line, [])
        if from_group in line_groups and to_group in line_groups:
            return True
        return False
    
    # 规则 2：同一工序内，同组变量可以相互影响
    if from_stage == to_stage:
        # 同组内可以相互影响
        if from_group == to_group:
            return True
        # 不同组不能相互影响（除非是 Group C 的特殊情况）
        return False
    
    return False


# ─── 数据加载函数 ──────────────────────────────────────────────────────────────

def prepare_data(line="xin1"):
    """
    从 Parquet 文件加载数据，返回处理后的 DataFrame 和元数据。

    参数：
      line: 产线名称（"xin1" 或 "xin2"）

    返回：
      df:           DataFrame，包含该产线的所有变量和 y_grade
      valid_vars:   该产线的有效变量列表（不含 y_grade）
      var_to_stage: 变量名 → 工序阶段的映射
      var_to_group: 变量名 → 分组的映射
    """
    
    # 检查数据文件是否存在
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"数据文件不存在：{DATA_PATH}\n"
            f"请确保数据路径正确，或将 Parquet 文件放在指定位置。"
        )
    
    # 加载 Parquet 文件
    try:
        df_all = pd.read_parquet(DATA_PATH)
    except Exception as e:
        raise RuntimeError(f"无法读取 Parquet 文件：{e}")
    
    print(f"[prepare_data] 加载数据：{DATA_PATH}")
    print(f"[prepare_data] 数据形状：{df_all.shape}")
    print(f"[prepare_data] 列名：{list(df_all.columns)}")
    
    # 筛选该产线的变量
    line_groups = XINS_GROUPS.get(line, [])
    valid_vars = []
    var_to_stage = {}
    var_to_group = {}
    
    for var_name, (stage, group) in VARIABLE_METADATA.items():
        # 检查变量是否属于该产线
        if group in line_groups and var_name in df_all.columns:
            valid_vars.append(var_name)
            var_to_stage[var_name] = stage
            var_to_group[var_name] = group
    
    if not valid_vars:
        raise ValueError(
            f"产线 {line} 没有找到任何有效变量。\n"
            f"数据列：{list(df_all.columns)}\n"
            f"期望的变量：{[v for v, (s, g) in VARIABLE_METADATA.items() if g in line_groups]}"
        )
    
    # 检查目标变量（根据产线选择）
    target_col = f"y_fx_xin{1 if line == 'xin1' else 2}"
    if target_col not in df_all.columns:
        raise ValueError(
            f"数据中缺少目标变量 '{target_col}'。\n"
            f"数据列：{list(df_all.columns)}"
        )
    
    # 提取该产线的数据，并将目标变量重命名为 y_grade（便于下游统一处理）
    cols_to_use = valid_vars + [target_col]
    df = df_all[cols_to_use].copy()
    df = df.rename(columns={target_col: "y_grade"})
    
    # 移除缺失值
    df = df.dropna()
    
    print(f"[prepare_data] 产线 {line}：{len(valid_vars)} 个变量，{len(df)} 个样本")
    
    return df, valid_vars, var_to_stage, var_to_group


# ─── 测试函数 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试数据加载
    print("=" * 70)
    print("测试 causal_discovery_config.py")
    print("=" * 70)
    
    for line in ["xin1", "xin2"]:
        print(f"\n--- 产线 {line} ---")
        try:
            df, valid_vars, var_to_stage, var_to_group = prepare_data(line)
            print(f"[OK] 数据加载成功")
            print(f"  变量数：{len(valid_vars)}")
            print(f"  样本数：{len(df)}")
            print(f"  变量列表：{valid_vars[:3]}...")
        except Exception as e:
            print(f"[ERROR] 错误：{e}")
    
    # 测试物理因果可行性
    print(f"\n--- 物理因果可行性测试 ---")
    test_cases = [
        ("粗选", "扫选", "A", "C", "xin1", True),   # 前序→后序，同产线
        ("扫选", "粗选", "C", "A", "xin1", False),  # 后序→前序，不可行
        ("粗选", "粗选", "A", "A", "xin1", True),   # 同工序同组，可行
        ("粗选", "粗选", "A", "C", "xin1", False),  # 同工序不同组，不可行
        ("粗选", "Y", "A", None, "xin1", True),     # 任何变量→Y，可行
        ("Y", "粗选", None, "A", "xin1", False),    # Y→任何变量，不可行
    ]
    
    for from_stage, to_stage, from_group, to_group, line, expected in test_cases:
        result = can_cause(from_stage, to_stage, from_group, to_group, line)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} {from_stage}->{to_stage} ({from_group}->{to_group}, {line}): {result} (expected {expected})")
