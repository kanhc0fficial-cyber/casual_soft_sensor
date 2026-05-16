"""
全面对比：训练集测试 vs 迁移测试
"""
import pandas as pd
import numpy as np

print('=' * 100)
print('全面性能对比：训练集测试 vs 迁移测试')
print('=' * 100)

# 1. 读取训练集测试结果
train_metrics = pd.read_csv('results/residual_soft_sensor/metrics_compare.csv')
train_metrics = train_metrics[train_metrics['split'] == 'test'].copy()
train_metrics['dataset_type'] = 'train_test_split'
train_metrics['dataset_name'] = 'simulation_2months (test 15%)'

# 2. 读取迁移测试结果
transfer_results = pd.read_csv('results/transfer_regimes_test_results/all_results_summary.csv')
transfer_results['dataset_type'] = 'transfer_test'
transfer_results.rename(columns={'dataset': 'dataset_name'}, inplace=True)

# 3. 合并
all_results = pd.concat([
    train_metrics[['model_name', 'dataset_type', 'dataset_name', 'MAE', 'RMSE', 'R2']],
    transfer_results[['model_name', 'dataset_type', 'dataset_name', 'MAE', 'RMSE', 'R2']]
], ignore_index=True)

# 4. 按模型分组展示
print('\n' + '=' * 100)
print('按模型分组的性能对比')
print('=' * 100)

for model in ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']:
    print(f'\n{model}:')
    print('-' * 100)
    
    model_data = all_results[all_results['model_name'] == model].copy()
    
    # 训练集测试
    train_data = model_data[model_data['dataset_type'] == 'train_test_split']
    if not train_data.empty:
        print(f'\n  训练集测试 (simulation_2months test split):')
        print(f'    MAE:  {train_data["MAE"].values[0]:.6f}')
        print(f'    RMSE: {train_data["RMSE"].values[0]:.6f}')
        print(f'    R²:   {train_data["R2"].values[0]:.6f}')
    
    # 迁移测试
    transfer_data = model_data[model_data['dataset_type'] == 'transfer_test']
    if not transfer_data.empty:
        print(f'\n  迁移测试 (transfer_regimes, 5个数据集平均):')
        print(f'    平均MAE:  {transfer_data["MAE"].mean():.6f} ± {transfer_data["MAE"].std():.6f}')
        print(f'    平均RMSE: {transfer_data["RMSE"].mean():.6f} ± {transfer_data["RMSE"].std():.6f}')
        print(f'    平均R²:   {transfer_data["R2"].mean():.6f} ± {transfer_data["R2"].std():.6f}')
        
        print(f'\n  各迁移数据集详情:')
        for _, row in transfer_data.iterrows():
            print(f'    {row["dataset_name"]:<35} MAE={row["MAE"]:.6f}, R²={row["R2"]:.6f}')
        
        # 计算性能下降
        train_mae = train_data["MAE"].values[0]
        transfer_mae = transfer_data["MAE"].mean()
        mae_degradation = (transfer_mae - train_mae) / train_mae * 100
        
        train_r2 = train_data["R2"].values[0]
        transfer_r2 = transfer_data["R2"].mean()
        r2_degradation = transfer_r2 - train_r2
        
        print(f'\n  性能变化:')
        print(f'    MAE: {mae_degradation:+.1f}% ({"恶化" if mae_degradation > 0 else "改善"})')
        print(f'    R²:  {r2_degradation:+.4f} ({"恶化" if r2_degradation < 0 else "改善"})')

# 5. 创建汇总表
print('\n' + '=' * 100)
print('汇总表')
print('=' * 100)

summary_rows = []
for model in ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']:
    model_data = all_results[all_results['model_name'] == model]
    
    train_data = model_data[model_data['dataset_type'] == 'train_test_split']
    transfer_data = model_data[model_data['dataset_type'] == 'transfer_test']
    
    if not train_data.empty and not transfer_data.empty:
        summary_rows.append({
            'Model': model,
            'Train_MAE': train_data["MAE"].values[0],
            'Train_R2': train_data["R2"].values[0],
            'Transfer_MAE_mean': transfer_data["MAE"].mean(),
            'Transfer_MAE_std': transfer_data["MAE"].std(),
            'Transfer_R2_mean': transfer_data["R2"].mean(),
            'Transfer_R2_std': transfer_data["R2"].std(),
            'MAE_degradation_%': (transfer_data["MAE"].mean() - train_data["MAE"].values[0]) / train_data["MAE"].values[0] * 100,
            'R2_degradation': transfer_data["R2"].mean() - train_data["R2"].values[0],
        })

summary_df = pd.DataFrame(summary_rows)
print('\n' + summary_df.to_string(index=False))

# 6. 保存完整对比结果
all_results.to_csv('results/comprehensive_comparison.csv', index=False, encoding='utf-8-sig')
print('\n' + '=' * 100)
print('完整对比结果已保存到: results/comprehensive_comparison.csv')
print('=' * 100)

# 7. 关键发现
print('\n' + '=' * 100)
print('关键发现')
print('=' * 100)

print('\n1. 训练集测试性能 (同分布):')
print('   - 所有模型R²都为正 (0.37-0.48)')
print('   - Model 2 (dml_effect_weight_lstm) 最佳: MAE=0.0175, R²=0.477')

print('\n2. 迁移测试性能 (跨工况):')
print('   - R²大多为负，说明跨工况泛化困难')
print('   - Model 1 (as_lstm) 最稳定: 平均R²=-0.42')
print('   - 在T2数据集上，Model 1和Model 2达到正R² (~0.09)')

print('\n3. 性能下降分析:')
for model in ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']:
    row = summary_df[summary_df['Model'] == model].iloc[0]
    print(f'   {model}:')
    print(f'     MAE增加: {row["MAE_degradation_%"]:.1f}%')
    print(f'     R²下降: {row["R2_degradation"]:.4f}')

print('\n4. 建议:')
print('   - 训练集测试结果可靠，模型在同分布数据上表现良好')
print('   - 迁移测试揭示了跨工况泛化的挑战')
print('   - Model 1 (as_lstm) 在迁移场景下最稳健')
print('   - 考虑使用域适应或在线学习来提升跨工况性能')
