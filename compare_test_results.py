"""
对比两种测试集的性能
"""
import pandas as pd
import numpy as np

# 读取两个测试结果
df_old = pd.read_csv('results/multiregime_test_results/all_results_summary.csv')
df_new = pd.read_csv('results/transfer_regimes_test_results/all_results_summary.csv')

# 添加数据集类型标签
df_old['test_type'] = 'multiregime_splits_noclip'
df_new['test_type'] = 'transfer_regimes_same_mechanism'

# 合并
df_all = pd.concat([df_old, df_new], ignore_index=True)

# 保存合并结果
df_all.to_csv('results/combined_test_results.csv', index=False, encoding='utf-8-sig')

# 计算每个模型在两种测试集上的平均性能
print('=' * 80)
print('两种测试集的性能对比')
print('=' * 80)

for model in ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']:
    print(f'\n{model}:')
    
    old_data = df_old[df_old['model_name'] == model]
    new_data = df_new[df_new['model_name'] == model]
    
    print(f'  multiregime_splits_noclip (6个数据集):')
    print(f'    平均MAE: {old_data["MAE"].mean():.4f} ± {old_data["MAE"].std():.4f}')
    print(f'    平均R²:  {old_data["R2"].mean():.4f} ± {old_data["R2"].std():.4f}')
    
    print(f'  transfer_regimes_same_mechanism (5个数据集):')
    print(f'    平均MAE: {new_data["MAE"].mean():.4f} ± {new_data["MAE"].std():.4f}')
    print(f'    平均R²:  {new_data["R2"].mean():.4f} ± {new_data["R2"].std():.4f}')
    
    # 计算改善
    mae_improve = (old_data['MAE'].mean() - new_data['MAE'].mean()) / old_data['MAE'].mean() * 100
    r2_improve = new_data['R2'].mean() - old_data['R2'].mean()
    
    print(f'  改善: MAE {mae_improve:+.1f}%, R² {r2_improve:+.2f}')

print('\n' + '=' * 80)
print('最佳模型统计 (按MAE)')
print('=' * 80)

# 找出每个数据集上的最佳模型
for test_type in ['multiregime_splits_noclip', 'transfer_regimes_same_mechanism']:
    print(f'\n{test_type}:')
    df_subset = df_all[df_all['test_type'] == test_type]
    
    for dataset in df_subset['dataset'].unique():
        df_ds = df_subset[df_subset['dataset'] == dataset]
        best_idx = df_ds['MAE'].idxmin()
        best_model = df_ds.loc[best_idx, 'model_name']
        best_mae = df_ds.loc[best_idx, 'MAE']
        best_r2 = df_ds.loc[best_idx, 'R2']
        
        print(f'  {dataset}: {best_model} (MAE={best_mae:.4f}, R²={best_r2:.4f})')

print('\n' + '=' * 80)
print('模型排名统计')
print('=' * 80)

# 统计每个模型获得最佳MAE的次数
model_wins = {}
for model in ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']:
    model_wins[model] = 0

for test_type in ['multiregime_splits_noclip', 'transfer_regimes_same_mechanism']:
    df_subset = df_all[df_all['test_type'] == test_type]
    
    for dataset in df_subset['dataset'].unique():
        df_ds = df_subset[df_subset['dataset'] == dataset]
        best_idx = df_ds['MAE'].idxmin()
        best_model = df_ds.loc[best_idx, 'model_name']
        model_wins[best_model] += 1

print('\n各模型获得最佳MAE的次数 (总共11个数据集):')
for model, wins in sorted(model_wins.items(), key=lambda x: x[1], reverse=True):
    print(f'  {model}: {wins}次')
