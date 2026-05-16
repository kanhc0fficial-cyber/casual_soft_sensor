"""
临时脚本：检查数据特征列及其索引
"""
import pandas as pd
import numpy as np

# 加载数据
data_path = r"C:\Users\goldenwhale\Downloads\my_mining_simulation\output\simulation_2months_seq_hybrid_normal_fast_sampling.parquet"
df = pd.read_parquet(data_path)

print(f"数据形状: {df.shape}")
print(f"\n总列数: {len(df.columns)}")

# 目标列和时间列
target_col = "y_fx_xin1"
time_col = "t"

# 排除列（实验室化验数据和另一个系统的目标）
exclude_cols = {
    "y_fx_xin2",
    "lab_1_eryi_f200", "lab_1_eryi_tfe", "lab_2_eryi_f200", "lab_2_eryi_tfe",
    "lab_3_eryi_f200", "lab_3_eryi_tfe", "lab_mag_wm_conc_tfe", "lab_mag_wm_tail_tfe",
    "lab_mag_hm_conc_tfe", "lab_mag_hm_tail_tfe", "lab_mag_sw_conc_tfe", "lab_mag_sw_tail_tfe",
    "lab_mag_mixed_conc_tfe", "lab_mag_tube_conc_tfe", "lab_mag_tube_yield",
    "lab_tm_feed_f325", "lab_tm_discharge_f325", "lab_tm_overflow_f325",
    "lab_tm_overflow_tfe", "lab_tm_overflow_conc", "lab_tm_sand_f325",
    "lab_flo_feed_tfe_s1", "lab_flo_feed_f325_s1", "lab_flo_conc_tfe_s1",
    "lab_flo_tail_tfe_s1", "lab_flo_rough_conc_tfe_s1", "lab_flo_rough_tail_tfe_s1",
    "lab_flo_clean_tail_tfe_s1", "lab_flo_scav1_conc_tfe_s1", "lab_flo_scav1_tail_tfe_s1",
    "lab_flo_scav2_conc_tfe_s1", "lab_flo_scav2_tail_tfe_s1", "lab_flo_scav3_conc_tfe_s1",
    "lab_flo_final_conc_yield_s1", "lab_flo_final_conc_recovery_s1",
    "lab_flo_feed_tfe_s2", "lab_flo_feed_f325_s2", "lab_flo_conc_tfe_s2",
    "lab_flo_tail_tfe_s2", "lab_flo_rough_conc_tfe_s2", "lab_flo_rough_tail_tfe_s2",
    "lab_flo_clean_tail_tfe_s2", "lab_flo_scav1_conc_tfe_s2", "lab_flo_scav1_tail_tfe_s2",
    "lab_flo_scav2_conc_tfe_s2", "lab_flo_scav2_tail_tfe_s2", "lab_flo_scav3_conc_tfe_s2",
    "lab_flo_final_conc_yield_s2", "lab_flo_final_conc_recovery_s2",
}

# 获取数值列
num_cols = df.select_dtypes(include=np.number).columns.tolist()

# 特征列：数值列 - 目标列 - 时间列 - 排除列
feature_cols = [
    c for c in num_cols
    if c != target_col and c != time_col and c not in exclude_cols
]

print(f"\n特征列数量: {len(feature_cols)}")
print("\n特征列索引对照表：")
print("=" * 80)

# 按类别分组显示
categories = {
    "磁选": "agg_mag_",
    "塔磨": "agg_tm_",
    "浮选系统1-浮选槽": "fx_s1_cx|fx_s1_jx|fx_s1_sx",
    "浮选系统1-药剂": "fx_s1_td_|fx_s1_k6_|fx_s1_naoh|fx_s1_cao",
    "浮选系统1-温度": "fx_s1_tk",
    "浮选系统1-泵池": "fx_s1_pool",
    "浮选系统1-其他": "fx_s1_",
    "浮选系统2": "fx_s2_",
    "浓密机": "fx_nt",
    "鼓风机": "fx_blower",
    "其他": "fx_ah|fx_ft",
}

import re

for cat_name, pattern in categories.items():
    cat_features = [(i, c) for i, c in enumerate(feature_cols) if re.search(pattern, c)]
    if cat_features:
        print(f"\n【{cat_name}】({len(cat_features)} 个)")
        for idx, col in cat_features:
            print(f"  [{idx:3d}] {col}")

print("\n" + "=" * 80)
print(f"\n总计特征列: {len(feature_cols)}")

# 保存到文件
with open("feature_index_mapping.txt", "w", encoding="utf-8") as f:
    f.write(f"特征列数量: {len(feature_cols)}\n")
    f.write("=" * 80 + "\n\n")
    for i, col in enumerate(feature_cols):
        f.write(f"[{i:3d}] {col}\n")

print("\n已保存特征索引映射到: feature_index_mapping.txt")
