# 高度相关变量分析报告

## 阶段：塔磨
## 产线：xin1
## 相关系数阈值：0.95

## 发现的高度相关变量对：7 对

| 变量1 | 变量2 | 相关系数 | 处理方式 |
|-------|-------|----------|----------|
| agg_tm_cyclone_pool_level | agg_tm_cyclone_pool_valve_setpoint | -0.9963 | 保留 agg_tm_cyclone_pool_level，移除 agg_tm_cyclone_pool_valve_setpoint |
| agg_tm_cyclone_pool_level | MC1_FET503_AI | -0.9881 | 保留 agg_tm_cyclone_pool_level，移除 MC1_FET503_AI |
| agg_tm_cyclone_pool_valve_setpoint | MC1_FET503_AI | 0.9917 | MC1_FET503_AI 已被移除 |
| agg_tm_cyclone_feed_flow | agg_tm_cyclone_pump_freq | 0.9703 | 保留 agg_tm_cyclone_feed_flow，移除 agg_tm_cyclone_pump_freq |
| agg_tm_cyclone_feed_flow | agg_tm_cyclone_pump_current | 0.9884 | 保留 agg_tm_cyclone_feed_flow，移除 agg_tm_cyclone_pump_current |
| agg_tm_cyclone_pump_freq | agg_tm_cyclone_pump_current | 0.9769 | agg_tm_cyclone_pump_current 已被移除 |
| agg_tm_cyclone_sand_valve_setpoint | agg_tm_cyclone_sand_valve_feedback | 0.9969 | 保留 agg_tm_cyclone_sand_valve_setpoint，移除 agg_tm_cyclone_sand_valve_feedback |

## 移除的变量数：5
## 保留的变量数：13