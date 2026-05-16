# 高度相关变量分析报告

## 阶段：磁选
## 产线：xin1
## 相关系数阈值：0.95

## 发现的高度相关变量对：3 对

| 变量1 | 变量2 | 相关系数 | 处理方式 |
|-------|-------|----------|----------|
| agg_mag_excit_voltage | agg_mag_excit_current | 0.9952 | 保留 agg_mag_excit_voltage，移除 agg_mag_excit_current |
| agg_mag_excit_voltage | agg_mag_coil_temp | 0.9528 | 保留 agg_mag_excit_voltage，移除 agg_mag_coil_temp |
| agg_mag_excit_current | agg_mag_coil_temp | 0.9682 | agg_mag_coil_temp 已被移除 |

## 移除的变量数：2
## 保留的变量数：10