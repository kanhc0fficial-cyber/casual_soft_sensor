"""
causal_discovery_config_v2.py
=============================
改进版因果发现配置模块：分阶段分析 + 去除对称变量

改进措施：
1. 每次只分析一个阶段的 DCS 变量，再加上精矿品位
2. 并行设备的对称变量只保留一份，且必须来自同一台机器
3. 生成高度相关变量移除报告

阶段划分：
- 磁选 (Magnetic Separation): agg_mag_*
- 塔磨 (Tower Mill): agg_tm_*, MC1_*
- 浮选 (Flotation): fx_* (分为 s1 和 s2 两条产线)
"""

import os
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

# ─── 数据路径配置 ──────────────────────────────────────────────────────────────

DATA_PATH = r"C:\Users\goldenwhale\Downloads\my_mining_simulation\output\simulation_2months_upstream_visible_clean_fast_sampling.parquet"

# ─── 阶段定义 ──────────────────────────────────────────────────────────────────

STAGES = {
    "磁选": 0,
    "塔磨": 1,
    "浮选": 2,
    "Y": 3,  # 目标变量（精矿品位）
}

# ─── 变量分组定义 ──────────────────────────────────────────────────────────────

# 磁选阶段变量（只有一台设备，无对称变量）
MAGNETIC_VARS = [
    "agg_mag_excit_voltage",      # 励磁电压
    "agg_mag_excit_current",      # 励磁电流
    "agg_mag_coil_temp",          # 线圈温度
    "agg_mag_tailings_valve1",    # 尾矿阀1
    "agg_mag_tailings_valve2",    # 尾矿阀2
    "agg_mag_blowdown_valve",     # 排污阀
    "agg_mag_pulsation_freq",     # 脉动频率
    "agg_mag_ring_freq",          # 环频
    "agg_mag_level",              # 液位
    "agg_mag_flush_water_pressure",  # 冲洗水压力
    "agg_mag_motor_current_rc",   # 环电机电流
    "agg_mag_motor_voltage_rc",   # 环电机电压
]

# 塔磨阶段变量（只有一台设备，无对称变量）
TOWER_MILL_VARS = [
    "agg_tm_cyclone_pool_level",           # 旋流器池液位
    "agg_tm_cyclone_pool_valve_setpoint",  # 旋流器池阀门设定值
    "MC1_FET503_AI",                       # 流量计
    "agg_tm_cyclone_feed_flow",            # 旋流器给料流量
    "agg_tm_cyclone_pump_freq",            # 旋流器泵频率
    "agg_tm_cyclone_pump_current",         # 旋流器泵电流
    "agg_tm_cyclone_sand_valve_setpoint",  # 旋流器沉砂阀设定值
    "agg_tm_cyclone_sand_valve_feedback",  # 旋流器沉砂阀反馈
    "agg_tm_cyclone_sand_water_flow",      # 旋流器沉砂水流量
    "agg_tm_motor_current",                # 塔磨电机电流
    "MC1_TM204_HDZC_1_WD_AI",             # 轴承1温度
    "MC1_TM206_HDZC_2_WD_AI",             # 轴承2温度
    "MC1_TM204_ZDJ_DZ_A_WD_AI",           # 定子A温度
    "MC1_TM206_ZDJ_DZ_B_WD_AI",           # 定子B温度
    "agg_tm_reducer_oil_temp",             # 减速机油温
    "agg_tm_reducer_outlet_temp",          # 减速机出口温度
    "agg_tm_cyclone_overflow_pool_level",  # 旋流器溢流池液位
    "agg_tm_overflow_pump_current",        # 溢流泵电流
]

# 浮选阶段变量 - 产线 s1（只保留 s1 的变量，去除 s2 的对称变量）
FLOTATION_S1_VARS = [
    # 浓缩机（只保留 nt1）
    "fx_nt1_motor_current",        # 浓缩机1电机电流
    "fx_nt1_underflow_density",    # 浓缩机1底流密度
    
    # 粗选槽（cx1, cx2, cx3）
    "fx_s1_cx1_froth_h",           # 粗选1泡沫高度
    "fx_s1_cx1_level",             # 粗选1液位
    "fx_s1_cx1_air_flow",          # 粗选1充气量
    "fx_s1_cx2_froth_h",           # 粗选2泡沫高度
    "fx_s1_cx2_level",             # 粗选2液位
    "fx_s1_cx2_air_flow",          # 粗选2充气量
    "fx_s1_cx3_froth_h",           # 粗选3泡沫高度
    "fx_s1_cx3_level",             # 粗选3液位
    "fx_s1_cx3_air_flow",          # 粗选3充气量
    
    # 扫选槽（sx1, sx2, sx3）
    "fx_s1_sx1_froth_h",           # 扫选1泡沫高度
    "fx_s1_sx1_level",             # 扫选1液位
    "fx_s1_sx1_air_flow",          # 扫选1充气量
    "fx_s1_sx2_froth_h",           # 扫选2泡沫高度
    "fx_s1_sx2_level",             # 扫选2液位
    "fx_s1_sx2_air_flow",          # 扫选2充气量
    "fx_s1_sx3_froth_h",           # 扫选3泡沫高度
    "fx_s1_sx3_level",             # 扫选3液位
    "fx_s1_sx3_air_flow",          # 扫选3充气量
    
    # 精选槽（jx）
    "fx_s1_jx_froth_h",            # 精选泡沫高度
    "fx_s1_jx_level",              # 精选液位
    "fx_s1_jx_air_flow",           # 精选充气量
    
    # 药剂控制
    "fx_s1_td_rough_freq",         # 粗选捕收剂频率
    "fx_s1_td_rough_curr",         # 粗选捕收剂电流
    "fx_s1_td_clean_freq",         # 精选捕收剂频率
    "fx_s1_td_clean_curr",         # 精选捕收剂电流
    "fx_s1_k6_rough_freq",         # 粗选起泡剂频率
    "fx_s1_k6_rough_curr",         # 粗选起泡剂电流
    "fx_s1_naoh_freq",             # NaOH频率
    "fx_s1_naoh_curr",             # NaOH电流
    "fx_s1_cao_freq",              # CaO频率
    "fx_s1_cao_curr",              # CaO电流
    "fx_s1_ph",                    # pH值
    
    # 调浆槽温度控制
    "fx_s1_tk1_temp",              # 调浆槽1温度
    "fx_s1_tk1_steam_sp",          # 调浆槽1蒸汽设定值
    "fx_s1_tk1_steam_fb",          # 调浆槽1蒸汽反馈
    "fx_s1_tk2_temp",              # 调浆槽2温度
    "fx_s1_tk2_steam_sp",          # 调浆槽2蒸汽设定值
    "fx_s1_tk2_steam_fb",          # 调浆槽2蒸汽反馈
    "fx_s1_tk3_temp",              # 调浆槽3温度
    "fx_s1_tk3_steam_sp",          # 调浆槽3蒸汽设定值
    "fx_s1_tk3_steam_fb",          # 调浆槽3蒸汽反馈
    
    # 泵池液位控制
    "fx_s1_pool1_level",           # 泵池1液位
    "fx_s1_pool1_pump_freq",       # 泵池1泵频率
    "fx_s1_pool1_pump_curr",       # 泵池1泵电流
    "fx_s1_pool2_level",           # 泵池2液位
    "fx_s1_pool2_pump_freq",       # 泵池2泵频率
    "fx_s1_pool2_pump_curr",       # 泵池2泵电流
    "fx_s1_pool3_level",           # 泵池3液位
    "fx_s1_pool3_pump_freq",       # 泵池3泵频率
    "fx_s1_pool3_pump_curr",       # 泵池3泵电流
    
    # 鼓风机（只保留 blower1）
    "fx_blower1_pressure",         # 鼓风机1压力
    
    # 搅拌器功率（只保留 ah5）
    "fx_ah5_power",                # 搅拌器5功率
    
    # 流量计
    "fx_s1_ft1701",                # 流量计1701
    "fx_s1_ft1702",                # 流量计1702
    
    # 药剂罐液位
    "fx_s1_k6_level",              # K6药剂罐液位
]

# 变量到阶段的映射
VARIABLE_TO_STAGE = {}
for var in MAGNETIC_VARS:
    VARIABLE_TO_STAGE[var] = "磁选"
for var in TOWER_MILL_VARS:
    VARIABLE_TO_STAGE[var] = "塔磨"
for var in FLOTATION_S1_VARS:
    VARIABLE_TO_STAGE[var] = "浮选"

# ─── 物理因果可行性规则 ────────────────────────────────────────────────────────

def can_cause(from_stage, to_stage):
    """
    判断物理因果可行性（简化版，只考虑阶段）。

    规则：
      1. 前序工序可以影响后序工序（磁选→塔磨→浮选→Y）
      2. 同一工序内，变量可以相互影响
      3. 后序工序不能影响前序工序（无反向因果）
      4. 目标变量 Y 只能被其他变量指向，不能指向任何变量

    参数：
      from_stage:  源变量的工序阶段（"磁选", "塔磨", "浮选", "Y"）
      to_stage:    目标变量的工序阶段（"磁选", "塔磨", "浮选", "Y"）

    返回：
      True 表示物理上可行，False 表示不可行
    """
    
    # 规则 4：Y 不能指向任何变量
    if from_stage == "Y":
        return False
    
    # 规则 3：后序工序不能影响前序工序
    if STAGES.get(from_stage, -1) > STAGES.get(to_stage, -1):
        return False
    
    # 规则 1 & 2：前序工序可以影响后序工序，同工序内可以相互影响
    return True


# ─── 数据加载函数 ──────────────────────────────────────────────────────────────

def prepare_data_by_stage(stage="浮选", line="xin1", correlation_threshold=0.95):
    """
    按阶段加载数据，并移除高度相关的变量。

    参数：
      stage: 阶段名称（"磁选", "塔磨", "浮选"）
      line:  产线名称（"xin1" 或 "xin2"）
      correlation_threshold: 相关系数阈值，超过此值的变量对将被移除

    返回：
      df:                DataFrame，包含该阶段的所有变量和 y_grade
      valid_vars:        该阶段的有效变量列表（不含 y_grade）
      var_to_stage:      变量名 → 工序阶段的映射
      removed_vars:      被移除的高度相关变量列表
      correlation_report: 相关性分析报告
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
    
    print(f"[prepare_data_by_stage] 加载数据：{DATA_PATH}")
    print(f"[prepare_data_by_stage] 数据形状：{df_all.shape}")
    
    # 根据阶段选择变量
    if stage == "磁选":
        stage_vars = MAGNETIC_VARS
    elif stage == "塔磨":
        stage_vars = TOWER_MILL_VARS
    elif stage == "浮选":
        stage_vars = FLOTATION_S1_VARS
    else:
        raise ValueError(f"未知的阶段：{stage}，支持的阶段：磁选、塔磨、浮选")
    
    # 筛选存在于数据中的变量
    valid_vars = [var for var in stage_vars if var in df_all.columns]
    
    if not valid_vars:
        raise ValueError(
            f"阶段 {stage} 没有找到任何有效变量。\n"
            f"数据列：{list(df_all.columns)}\n"
            f"期望的变量：{stage_vars}"
        )
    
    # 检查目标变量
    target_col = f"y_fx_{line}"
    if target_col not in df_all.columns:
        raise ValueError(
            f"数据中缺少目标变量 '{target_col}'。\n"
            f"数据列：{list(df_all.columns)}"
        )
    
    # 提取该阶段的数据
    cols_to_use = valid_vars + [target_col]
    df = df_all[cols_to_use].copy()
    df = df.rename(columns={target_col: "y_grade"})
    
    # 移除缺失值
    df = df.dropna()
    
    print(f"[prepare_data_by_stage] 阶段 {stage}：{len(valid_vars)} 个变量，{len(df)} 个样本")
    
    # ─── 分析高度相关的变量 ───────────────────────────────────────────────
    print(f"\n[相关性分析] 分析高度相关的变量（阈值 > {correlation_threshold}）...")
    
    X = df[valid_vars].values
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    
    # 计算 Spearman 相关系数
    corr_matrix, _ = spearmanr(X_norm)
    corr = np.array(corr_matrix)
    if corr.ndim == 0:
        corr = np.array([[1.0]])
    
    # 找出高度相关的变量对
    np.fill_diagonal(corr, 0.0)
    high_corr_pairs = []
    removed_vars = []
    
    for i in range(len(valid_vars)):
        for j in range(i + 1, len(valid_vars)):
            if abs(corr[i, j]) > correlation_threshold:
                high_corr_pairs.append((valid_vars[i], valid_vars[j], corr[i, j]))
    
    # 生成相关性报告
    correlation_report = []
    correlation_report.append(f"# 高度相关变量分析报告")
    correlation_report.append(f"\n## 阶段：{stage}")
    correlation_report.append(f"## 产线：{line}")
    correlation_report.append(f"## 相关系数阈值：{correlation_threshold}")
    correlation_report.append(f"\n## 发现的高度相关变量对：{len(high_corr_pairs)} 对\n")
    
    if high_corr_pairs:
        correlation_report.append("| 变量1 | 变量2 | 相关系数 | 处理方式 |")
        correlation_report.append("|-------|-------|----------|----------|")
        
        # 决定移除哪个变量（保留第一个，移除第二个）
        for var1, var2, corr_val in high_corr_pairs:
            if var2 not in removed_vars:
                removed_vars.append(var2)
                correlation_report.append(f"| {var1} | {var2} | {corr_val:.4f} | 保留 {var1}，移除 {var2} |")
            else:
                correlation_report.append(f"| {var1} | {var2} | {corr_val:.4f} | {var2} 已被移除 |")
    else:
        correlation_report.append("未发现高度相关的变量对。")
    
    correlation_report.append(f"\n## 移除的变量数：{len(removed_vars)}")
    correlation_report.append(f"## 保留的变量数：{len(valid_vars) - len(removed_vars)}")
    
    # 从数据中移除高度相关的变量
    if removed_vars:
        print(f"[相关性分析] 移除 {len(removed_vars)} 个高度相关的变量：{removed_vars}")
        valid_vars = [var for var in valid_vars if var not in removed_vars]
        df = df[valid_vars + ["y_grade"]]
    else:
        print(f"[相关性分析] 未发现需要移除的高度相关变量")
    
    # 构建变量到阶段的映射
    var_to_stage = {var: stage for var in valid_vars}
    
    print(f"[prepare_data_by_stage] 最终保留：{len(valid_vars)} 个变量，{len(df)} 个样本")
    
    return df, valid_vars, var_to_stage, removed_vars, "\n".join(correlation_report)


# ─── 测试函数 ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 测试数据加载
    print("=" * 70)
    print("测试 causal_discovery_config_v2.py")
    print("=" * 70)
    
    for stage in ["磁选", "塔磨", "浮选"]:
        print(f"\n--- 阶段 {stage} ---")
        try:
            df, valid_vars, var_to_stage, removed_vars, report = prepare_data_by_stage(
                stage=stage, 
                line="xin1",
                correlation_threshold=0.95
            )
            print(f"[OK] 数据加载成功")
            print(f"  变量数：{len(valid_vars)}")
            print(f"  样本数：{len(df)}")
            print(f"  移除变量数：{len(removed_vars)}")
            if removed_vars:
                print(f"  移除的变量：{removed_vars[:3]}...")
        except Exception as e:
            print(f"[ERROR] 错误：{e}")
    
    # 测试物理因果可行性
    print(f"\n--- 物理因果可行性测试 ---")
    test_cases = [
        ("磁选", "塔磨", True),   # 前序→后序，可行
        ("塔磨", "磁选", False),  # 后序→前序，不可行
        ("磁选", "磁选", True),   # 同工序，可行
        ("浮选", "Y", True),      # 任何变量→Y，可行
        ("Y", "浮选", False),     # Y→任何变量，不可行
    ]
    
    for from_stage, to_stage, expected in test_cases:
        result = can_cause(from_stage, to_stage)
        status = "[OK]" if result == expected else "[FAIL]"
        print(f"{status} {from_stage}->{to_stage}: {result} (expected {expected})")
