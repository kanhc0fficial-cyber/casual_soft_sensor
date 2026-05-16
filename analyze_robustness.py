"""
分析Model 3的鲁棒性：是否在工况变化时性能下降幅度更小？
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print('=' * 100)
print('Model 3 (DML残差) 鲁棒性分析')
print('=' * 100)

# 读取数据
train_metrics = pd.read_csv('results/residual_soft_sensor/metrics_compare.csv')
train_metrics = train_metrics[train_metrics['split'] == 'test'].copy()

transfer_results = pd.read_csv('results/transfer_regimes_test_results/all_results_summary.csv')

# 计算性能下降
print('\n' + '=' * 100)
print('1. 性能下降幅度对比 (训练集测试 → 迁移测试)')
print('=' * 100)

models = ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']
degradation_data = []

for model in models:
    # 训练集性能
    train_mae = train_metrics[train_metrics['model_name'] == model]['MAE'].values[0]
    train_r2 = train_metrics[train_metrics['model_name'] == model]['R2'].values[0]
    
    # 迁移测试性能
    transfer_data = transfer_results[transfer_results['model_name'] == model]
    transfer_mae_mean = transfer_data['MAE'].mean()
    transfer_mae_std = transfer_data['MAE'].std()
    transfer_r2_mean = transfer_data['R2'].mean()
    transfer_r2_std = transfer_data['R2'].std()
    
    # 计算下降幅度
    mae_degradation_pct = (transfer_mae_mean - train_mae) / train_mae * 100
    mae_degradation_abs = transfer_mae_mean - train_mae
    r2_degradation = transfer_r2_mean - train_r2
    
    degradation_data.append({
        'Model': model,
        'Train_MAE': train_mae,
        'Transfer_MAE_mean': transfer_mae_mean,
        'Transfer_MAE_std': transfer_mae_std,
        'MAE_degradation_%': mae_degradation_pct,
        'MAE_degradation_abs': mae_degradation_abs,
        'Train_R2': train_r2,
        'Transfer_R2_mean': transfer_r2_mean,
        'Transfer_R2_std': transfer_r2_std,
        'R2_degradation': r2_degradation,
    })

degradation_df = pd.DataFrame(degradation_data)

print('\nMAE性能下降排序 (从小到大 = 从鲁棒到脆弱):')
print('-' * 100)
sorted_by_mae_pct = degradation_df.sort_values('MAE_degradation_%')
for idx, row in sorted_by_mae_pct.iterrows():
    print(f"{row['Model']:<30} {row['MAE_degradation_%']:>6.1f}%  "
          f"(训练: {row['Train_MAE']:.4f} → 迁移: {row['Transfer_MAE_mean']:.4f})")

print('\nR²性能下降排序 (从小到大 = 从鲁棒到脆弱):')
print('-' * 100)
sorted_by_r2 = degradation_df.sort_values('R2_degradation', ascending=False)
for idx, row in sorted_by_r2.iterrows():
    print(f"{row['Model']:<30} {row['R2_degradation']:>7.3f}  "
          f"(训练: {row['Train_R2']:.3f} → 迁移: {row['Transfer_R2_mean']:.3f})")

# 分析：Model 3是否最鲁棒？
print('\n' + '=' * 100)
print('2. Model 3 (dml_residual_lstm) 鲁棒性评估')
print('=' * 100)

model3_data = degradation_df[degradation_df['Model'] == 'dml_residual_lstm'].iloc[0]
model3_mae_rank = (degradation_df['MAE_degradation_%'] < model3_data['MAE_degradation_%']).sum() + 1
model3_r2_rank = (degradation_df['R2_degradation'] > model3_data['R2_degradation']).sum() + 1

print(f"\nModel 3性能下降幅度:")
print(f"  MAE下降: {model3_data['MAE_degradation_%']:.1f}% (排名: {model3_mae_rank}/4, 1=最鲁棒)")
print(f"  R²下降:  {model3_data['R2_degradation']:.3f} (排名: {model3_r2_rank}/4, 1=最鲁棒)")

if model3_mae_rank == 4 and model3_r2_rank == 4:
    print("\n❌ 结论: Model 3在工况变化时性能下降幅度**最大**，不是最鲁棒的模型")
elif model3_mae_rank == 1 and model3_r2_rank == 1:
    print("\n✅ 结论: Model 3在工况变化时性能下降幅度**最小**，是最鲁棒的模型")
else:
    print(f"\n⚠️ 结论: Model 3鲁棒性排名中等 (MAE排名{model3_mae_rank}, R²排名{model3_r2_rank})")

# 分析：相对性能排名是否稳定？
print('\n' + '=' * 100)
print('3. 模型排名稳定性分析')
print('=' * 100)

print('\n训练集测试排名 (按MAE):')
train_ranking = train_metrics.sort_values('MAE')[['model_name', 'MAE', 'R2']]
for rank, (idx, row) in enumerate(train_ranking.iterrows(), 1):
    print(f"  {rank}. {row['model_name']:<30} MAE={row['MAE']:.4f}, R²={row['R2']:.3f}")

print('\n迁移测试平均排名 (按MAE):')
transfer_avg = transfer_results.groupby('model_name').agg({
    'MAE': 'mean',
    'R2': 'mean'
}).sort_values('MAE')
for rank, (model, row) in enumerate(transfer_avg.iterrows(), 1):
    print(f"  {rank}. {model:<30} MAE={row['MAE']:.4f}, R²={row['R2']:.3f}")

# 计算排名变化
print('\n排名变化:')
train_rank_dict = {row['model_name']: rank for rank, (idx, row) in enumerate(train_ranking.iterrows(), 1)}
transfer_rank_dict = {model: rank for rank, model in enumerate(transfer_avg.index, 1)}

for model in models:
    train_rank = train_rank_dict[model]
    transfer_rank = transfer_rank_dict[model]
    rank_change = transfer_rank - train_rank
    direction = "↓" if rank_change < 0 else "↑" if rank_change > 0 else "→"
    print(f"  {model:<30} {train_rank} → {transfer_rank} ({direction}{abs(rank_change)})")

# 分析：跨数据集的稳定性
print('\n' + '=' * 100)
print('4. 跨数据集稳定性分析 (变异系数 CV = std/mean)')
print('=' * 100)

print('\nMAE变异系数 (越小越稳定):')
for model in models:
    transfer_data = transfer_results[transfer_results['model_name'] == model]
    mae_mean = transfer_data['MAE'].mean()
    mae_std = transfer_data['MAE'].std()
    cv = mae_std / mae_mean
    print(f"  {model:<30} CV={cv:.4f} (mean={mae_mean:.4f}, std={mae_std:.4f})")

model3_transfer = transfer_results[transfer_results['model_name'] == 'dml_residual_lstm']
model3_cv = model3_transfer['MAE'].std() / model3_transfer['MAE'].mean()
best_cv = min([transfer_results[transfer_results['model_name'] == m]['MAE'].std() / 
               transfer_results[transfer_results['model_name'] == m]['MAE'].mean() 
               for m in models])

if model3_cv == best_cv:
    print("\n✅ Model 3在不同工况下的MAE变异最小，表现最稳定")
else:
    print(f"\n⚠️ Model 3的MAE变异系数为{model3_cv:.4f}，不是最稳定的")

# 分析：绝对性能 vs 相对鲁棒性
print('\n' + '=' * 100)
print('5. 绝对性能 vs 相对鲁棒性权衡')
print('=' * 100)

print('\n综合评分 (归一化后的加权平均):')
print('  权重: 训练集性能40% + 迁移性能30% + 鲁棒性30%')

# 归一化
train_mae_norm = 1 - (train_metrics.set_index('model_name')['MAE'] - train_metrics['MAE'].min()) / (train_metrics['MAE'].max() - train_metrics['MAE'].min())
transfer_mae_norm = 1 - (transfer_avg['MAE'] - transfer_avg['MAE'].min()) / (transfer_avg['MAE'].max() - transfer_avg['MAE'].min())
robustness_norm = 1 - (degradation_df.set_index('Model')['MAE_degradation_%'] - degradation_df['MAE_degradation_%'].min()) / (degradation_df['MAE_degradation_%'].max() - degradation_df['MAE_degradation_%'].min())

scores = []
for model in models:
    score = (0.4 * train_mae_norm[model] + 
             0.3 * transfer_mae_norm[model] + 
             0.3 * robustness_norm[model])
    scores.append({
        'Model': model,
        'Train_score': train_mae_norm[model],
        'Transfer_score': transfer_mae_norm[model],
        'Robustness_score': robustness_norm[model],
        'Total_score': score
    })

scores_df = pd.DataFrame(scores).sort_values('Total_score', ascending=False)
print()
for rank, (idx, row) in enumerate(scores_df.iterrows(), 1):
    print(f"  {rank}. {row['Model']:<30} 总分={row['Total_score']:.3f} "
          f"(训练={row['Train_score']:.3f}, 迁移={row['Transfer_score']:.3f}, 鲁棒={row['Robustness_score']:.3f})")

# 最终结论
print('\n' + '=' * 100)
print('6. 最终结论')
print('=' * 100)

print('\n关于"Model 3在工况变化时性能下降幅度更小"的观点:')
print('-' * 100)

if model3_mae_rank == 4:
    print('\n❌ **数据不支持该观点**')
    print('\n证据:')
    print(f'  1. Model 3的MAE下降幅度为{model3_data["MAE_degradation_%"]:.1f}%，在4个模型中排名第{model3_mae_rank}（最差）')
    print(f'  2. Model 3的R²下降幅度为{model3_data["R2_degradation"]:.3f}，在4个模型中排名第{model3_r2_rank}（最差）')
    print(f'  3. 最鲁棒的模型是: {sorted_by_mae_pct.iloc[0]["Model"]} (MAE下降{sorted_by_mae_pct.iloc[0]["MAE_degradation_%"]:.1f}%)')
    
    print('\n可能的原因:')
    print('  - Model 3依赖C变量进行残差化，但C变量在不同工况下的分布可能差异很大')
    print('  - g_model和q_models在训练集上拟合，在新工况下泛化能力不足')
    print('  - 残差化过程引入了额外的误差累积')
    
    print('\n但是，Model 3有其他优势:')
    model3_cv = model3_transfer['MAE'].std() / model3_transfer['MAE'].mean()
    print(f'  - 跨数据集稳定性: CV={model3_cv:.4f} (变异系数)')
    print(f'  - 可解释性: 提供y_base和y_res的分解')
    
else:
    print('\n✅ **数据支持该观点**')
    print(f'\n证据: Model 3的性能下降幅度排名第{model3_mae_rank}，表现出较好的鲁棒性')

print('\n推荐:')
if model3_mae_rank <= 2:
    print('  ✅ 如果看重鲁棒性和可解释性，推荐使用Model 3')
else:
    print('  ⚠️ 如果看重绝对性能，推荐使用Model 1或Model 2')
    print('  ⚠️ Model 3在当前数据上的鲁棒性优势不明显')

# 保存分析结果
degradation_df.to_csv('results/robustness_analysis.csv', index=False, encoding='utf-8-sig')
scores_df.to_csv('results/comprehensive_scores.csv', index=False, encoding='utf-8-sig')

print('\n' + '=' * 100)
print('分析结果已保存:')
print('  - results/robustness_analysis.csv')
print('  - results/comprehensive_scores.csv')
print('=' * 100)
