"""
可视化鲁棒性分析
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 读取数据
train_metrics = pd.read_csv('results/residual_soft_sensor/metrics_compare.csv')
train_metrics = train_metrics[train_metrics['split'] == 'test'].copy()
transfer_results = pd.read_csv('results/transfer_regimes_test_results/all_results_summary.csv')

models = ['baseline_all_lstm', 'as_lstm', 'dml_effect_weight_lstm', 'dml_residual_lstm']
model_labels = ['Model 0\nBaseline', 'Model 1\nAS-LSTM', 'Model 2\nDML-Weight', 'Model 3\nDML-Residual']

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Model 3 Robustness Analysis: Performance Degradation under Regime Shifts', 
             fontsize=16, fontweight='bold')

# 1. 性能下降幅度对比
ax1 = axes[0, 0]
mae_degradations = []
for model in models:
    train_mae = train_metrics[train_metrics['model_name'] == model]['MAE'].values[0]
    transfer_mae = transfer_results[transfer_results['model_name'] == model]['MAE'].mean()
    degradation = (transfer_mae - train_mae) / train_mae * 100
    mae_degradations.append(degradation)

colors = ['#2ecc71' if d < 30 else '#e74c3c' for d in mae_degradations]
bars1 = ax1.bar(range(len(models)), mae_degradations, color=colors, alpha=0.7, edgecolor='black')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(model_labels, fontsize=10)
ax1.set_ylabel('MAE Degradation (%)', fontsize=11, fontweight='bold')
ax1.set_title('(A) MAE Performance Degradation\n(Lower is Better)', fontsize=12, fontweight='bold')
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax1.grid(axis='y', alpha=0.3)

# 添加数值标签
for i, (bar, val) in enumerate(zip(bars1, mae_degradations)):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)

# 标注Model 3
ax1.text(3, mae_degradations[3] + 3, '❌ Worst', ha='center', fontsize=10, 
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# 2. R²下降对比
ax2 = axes[0, 1]
r2_degradations = []
for model in models:
    train_r2 = train_metrics[train_metrics['model_name'] == model]['R2'].values[0]
    transfer_r2 = transfer_results[transfer_results['model_name'] == model]['R2'].mean()
    degradation = transfer_r2 - train_r2
    r2_degradations.append(degradation)

colors2 = ['#2ecc71' if d > -1.0 else '#e74c3c' for d in r2_degradations]
bars2 = ax2.bar(range(len(models)), r2_degradations, color=colors2, alpha=0.7, edgecolor='black')
ax2.set_xticks(range(len(models)))
ax2.set_xticklabels(model_labels, fontsize=10)
ax2.set_ylabel('R² Degradation', fontsize=11, fontweight='bold')
ax2.set_title('(B) R² Performance Degradation\n(Higher is Better)', fontsize=12, fontweight='bold')
ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1)
ax2.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars2, r2_degradations)):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.1, 
             f'{val:.2f}', ha='center', va='top', fontweight='bold', fontsize=10)

ax2.text(3, r2_degradations[3] - 0.15, '❌ Worst', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))

# 3. 训练 vs 迁移性能对比
ax3 = axes[1, 0]
x = np.arange(len(models))
width = 0.35

train_maes = [train_metrics[train_metrics['model_name'] == m]['MAE'].values[0] for m in models]
transfer_maes = [transfer_results[transfer_results['model_name'] == m]['MAE'].mean() for m in models]

bars3a = ax3.bar(x - width/2, train_maes, width, label='Train Test', color='#3498db', alpha=0.7, edgecolor='black')
bars3b = ax3.bar(x + width/2, transfer_maes, width, label='Transfer Test', color='#e67e22', alpha=0.7, edgecolor='black')

ax3.set_xticks(x)
ax3.set_xticklabels(model_labels, fontsize=10)
ax3.set_ylabel('MAE', fontsize=11, fontweight='bold')
ax3.set_title('(C) Absolute Performance Comparison', fontsize=12, fontweight='bold')
ax3.legend(fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# 标注Model 3的性能差距
ax3.annotate('', xy=(3 + width/2, transfer_maes[3]), xytext=(3 - width/2, train_maes[3]),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax3.text(3, (train_maes[3] + transfer_maes[3])/2, f'+{(transfer_maes[3]-train_maes[3])/train_maes[3]*100:.1f}%',
         ha='right', va='center', fontsize=9, color='red', fontweight='bold')

# 4. 跨数据集稳定性 (变异系数)
ax4 = axes[1, 1]
cvs = []
for model in models:
    transfer_data = transfer_results[transfer_results['model_name'] == model]
    cv = transfer_data['MAE'].std() / transfer_data['MAE'].mean()
    cvs.append(cv)

colors4 = ['#2ecc71' if cv < 0.15 else '#f39c12' for cv in cvs]
bars4 = ax4.bar(range(len(models)), cvs, color=colors4, alpha=0.7, edgecolor='black')
ax4.set_xticks(range(len(models)))
ax4.set_xticklabels(model_labels, fontsize=10)
ax4.set_ylabel('Coefficient of Variation (CV)', fontsize=11, fontweight='bold')
ax4.set_title('(D) Cross-Dataset Stability\n(Lower is Better)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)

for i, (bar, val) in enumerate(zip(bars4, cvs)):
    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{val:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

ax4.text(3, cvs[3] + 0.015, '✅ Best', ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))

plt.tight_layout()
plt.savefig('results/robustness_analysis.png', dpi=300, bbox_inches='tight')
print('图表已保存: results/robustness_analysis.png')

# 创建第二个图：每个数据集上的详细性能
fig2, ax = plt.subplots(figsize=(14, 6))

datasets = transfer_results['dataset'].unique()
x = np.arange(len(datasets))
width = 0.2

for i, model in enumerate(models):
    model_data = transfer_results[transfer_results['model_name'] == model]
    maes = [model_data[model_data['dataset'] == ds]['MAE'].values[0] for ds in datasets]
    ax.bar(x + i*width, maes, width, label=model_labels[i], alpha=0.7, edgecolor='black')

ax.set_xlabel('Transfer Test Datasets', fontsize=12, fontweight='bold')
ax.set_ylabel('MAE', fontsize=12, fontweight='bold')
ax.set_title('Performance Across Different Regimes', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(datasets, rotation=15, ha='right', fontsize=10)
ax.legend(fontsize=10, loc='upper left')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/performance_by_dataset.png', dpi=300, bbox_inches='tight')
print('图表已保存: results/performance_by_dataset.png')

print('\n可视化完成！')
