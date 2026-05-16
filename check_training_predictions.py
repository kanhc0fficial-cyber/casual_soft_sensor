"""
查看训练集测试结果
"""
import pandas as pd

print('=' * 80)
print('训练集测试结果文件概览 (simulation_2months数据集的test split)')
print('=' * 80)

files = [
    ('Model 0: baseline_all_lstm', 'results/residual_soft_sensor/baseline_predictions_test.csv'),
    ('Model 1: as_lstm', 'results/residual_soft_sensor/as_lstm_predictions_test.csv'),
    ('Model 2: dml_effect_weight_lstm', 'results/residual_soft_sensor/dml_effect_weight_predictions_test.csv'),
    ('Model 3: dml_residual_lstm', 'results/residual_soft_sensor/dml_residual_predictions_test.csv'),
]

for name, path in files:
    df = pd.read_csv(path)
    print(f'\n{name}:')
    print(f'  文件: {path}')
    print(f'  形状: {df.shape}')
    print(f'  列: {list(df.columns)}')
    
    # 计算基本统计
    if 'y_true' in df.columns and 'y_pred' in df.columns:
        mae = (df['y_true'] - df['y_pred']).abs().mean()
        rmse = ((df['y_true'] - df['y_pred'])**2).mean()**0.5
        print(f'  MAE: {mae:.6f}')
        print(f'  RMSE: {rmse:.6f}')
        print(f'  y_true范围: [{df["y_true"].min():.4f}, {df["y_true"].max():.4f}]')
        print(f'  y_pred范围: [{df["y_pred"].min():.4f}, {df["y_pred"].max():.4f}]')
        
        # 显示前5行
        print(f'  前5行:')
        print(df.head().to_string(index=False))

print('\n' + '=' * 80)
print('metrics_compare.csv (官方指标):')
print('=' * 80)
metrics = pd.read_csv('results/residual_soft_sensor/metrics_compare.csv')
print(metrics.to_string(index=False))

print('\n' + '=' * 80)
print('数据集信息:')
print('=' * 80)
print('训练数据: simulation_2months_seq_hybrid_normal_fast_sampling.parquet')
print('  总样本数: 86,400')
print('  训练集: 70% = 60,480')
print('  验证集: 15% = 12,960')
print('  测试集: 15% = 12,960')
print('\n注意: 预测文件中的样本数是12,950，因为滑动窗口(window_size=12)会损失11个样本')
print('      12,960 - 11 = 12,949 ≈ 12,950')
