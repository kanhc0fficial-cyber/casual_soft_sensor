"""
visualize_group_branch_results.py
==================================
门控分组模型结果可视化脚本

用法：
  python scripts/visualize_group_branch_results.py

输出：
  results/group_branch/visualizations/
    - prediction_vs_actual.png       预测值vs真实值散点图
    - residual_distribution.png      残差分布直方图
    - branch_contributions.png       各分支贡献箱线图
    - time_series_comparison.png     时间序列对比图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置绘图风格
sns.set_style("whitegrid")
sns.set_palette("husl")

def load_data():
    """加载预测结果数据"""
    data_path = Path("results/group_branch/predictions_test.csv")
    df = pd.read_csv(data_path)
    return df

def plot_prediction_vs_actual(df, output_dir):
    """绘制预测值vs真实值散点图"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 计算指标
    r2 = r2_score(df['y_true'], df['y_pred'])
    mae = mean_absolute_error(df['y_true'], df['y_pred'])
    rmse = np.sqrt(mean_squared_error(df['y_true'], df['y_pred']))
    
    # 散点图
    ax.scatter(df['y_true'], df['y_pred'], alpha=0.5, s=20, edgecolors='none')
    
    # 对角线（完美预测线）
    min_val = min(df['y_true'].min(), df['y_pred'].min())
    max_val = max(df['y_true'].max(), df['y_pred'].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='完美预测线')
    
    # 标注指标
    textstr = f'R² = {r2:.4f}\nMAE = {mae:.4f}\nRMSE = {rmse:.4f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    ax.set_xlabel('真实值 (y_fx_xin1)', fontsize=14)
    ax.set_ylabel('预测值 (y_fx_xin1)', fontsize=14)
    ax.set_title('门控分组模型：预测值 vs 真实值', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_dir / 'prediction_vs_actual.png'}")

def plot_residual_distribution(df, output_dir):
    """绘制残差分布直方图"""
    residuals = df['y_pred'] - df['y_true']
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 残差直方图
    axes[0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2, label='零残差线')
    axes[0].set_xlabel('残差 (预测值 - 真实值)', fontsize=12)
    axes[0].set_ylabel('频数', fontsize=12)
    axes[0].set_title('残差分布直方图', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # 统计信息
    mean_res = residuals.mean()
    std_res = residuals.std()
    textstr = f'均值 = {mean_res:.6f}\n标准差 = {std_res:.6f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    axes[0].text(0.70, 0.95, textstr, transform=axes[0].transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    
    # 残差vs预测值散点图
    axes[1].scatter(df['y_pred'], residuals, alpha=0.5, s=20, edgecolors='none')
    axes[1].axhline(y=0, color='r', linestyle='--', linewidth=2, label='零残差线')
    axes[1].set_xlabel('预测值', fontsize=12)
    axes[1].set_ylabel('残差', fontsize=12)
    axes[1].set_title('残差 vs 预测值', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'residual_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_dir / 'residual_distribution.png'}")

def plot_branch_contributions(df, output_dir):
    """绘制各分支贡献箱线图"""
    # 提取贡献列
    contribution_cols = [col for col in df.columns if col.startswith('contribution_')]
    
    # 准备数据
    contributions_data = []
    branch_names = []
    
    # 中文分支名称映射
    branch_name_map = {
        'preprocessing': '预处理',
        'reagent': '药剂',
        'roughing': '粗选',
        'cleaning': '精选',
        'scavenging': '扫选',
        'temperature': '温度',
        'auxiliary': '辅助'
    }
    
    for col in contribution_cols:
        branch_name = col.replace('contribution_', '')
        contributions_data.append(df[col].values)
        branch_names.append(branch_name_map.get(branch_name, branch_name))
    
    # 计算平均贡献并排序
    mean_contributions = [np.abs(data).mean() for data in contributions_data]
    sorted_indices = np.argsort(mean_contributions)[::-1]  # 降序
    
    contributions_data = [contributions_data[i] for i in sorted_indices]
    branch_names = [branch_names[i] for i in sorted_indices]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 箱线图
    bp = ax.boxplot(contributions_data, labels=branch_names, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # 设置颜色
    colors = plt.cm.Set3(np.linspace(0, 1, len(contributions_data)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # 添加平均绝对贡献值标注
    for i, (data, name) in enumerate(zip(contributions_data, branch_names)):
        mean_abs = np.abs(data).mean()
        ax.text(i+1, ax.get_ylim()[1]*0.95, f'{mean_abs:.3f}', 
                ha='center', va='top', fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('工艺分组', fontsize=14)
    ax.set_ylabel('分支贡献值', fontsize=14)
    ax.set_title('各分支贡献分布（按平均绝对贡献排序）', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'branch_contributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_dir / 'branch_contributions.png'}")

def plot_time_series_comparison(df, output_dir):
    """绘制时间序列对比图"""
    # 只显示前500个样本，避免图表过于密集
    n_samples = min(500, len(df))
    df_subset = df.head(n_samples)
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 上图：预测值vs真实值时间序列
    x = np.arange(n_samples)
    axes[0].plot(x, df_subset['y_true'].values, label='真实值', linewidth=1.5, alpha=0.8)
    axes[0].plot(x, df_subset['y_pred'].values, label='预测值', linewidth=1.5, alpha=0.8)
    axes[0].set_xlabel('样本序号', fontsize=12)
    axes[0].set_ylabel('浮选精矿品位 (y_fx_xin1)', fontsize=12)
    axes[0].set_title(f'时间序列对比（前{n_samples}个样本）', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, loc='best')
    axes[0].grid(True, alpha=0.3)
    
    # 下图：残差时间序列
    residuals = df_subset['y_pred'] - df_subset['y_true']
    axes[1].plot(x, residuals, color='red', linewidth=1, alpha=0.7)
    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
    axes[1].fill_between(x, residuals, 0, alpha=0.3, color='red')
    axes[1].set_xlabel('样本序号', fontsize=12)
    axes[1].set_ylabel('残差 (预测值 - 真实值)', fontsize=12)
    axes[1].set_title('预测残差时间序列', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'time_series_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_dir / 'time_series_comparison.png'}")

def plot_gate_values(output_dir):
    """绘制Gate值柱状图"""
    # 读取gate值
    gates_path = Path("results/group_branch/group_branch_gates.csv")
    df_gates = pd.read_csv(gates_path)
    
    # 中文分支名称映射
    branch_name_map = {
        'preprocessing': '预处理',
        'reagent': '药剂',
        'roughing': '粗选',
        'cleaning': '精选',
        'scavenging': '扫选',
        'temperature': '温度',
        'auxiliary': '辅助'
    }
    
    df_gates['group_cn'] = df_gates['group'].map(branch_name_map)
    
    # 按gate值排序
    df_gates = df_gates.sort_values('gate_value', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_gates)))
    bars = ax.bar(df_gates['group_cn'], df_gates['gate_value'], color=colors, alpha=0.8, edgecolor='black')
    
    # 添加数值标注
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, alpha=0.5, label='初始值 (0.5)')
    ax.set_xlabel('工艺分组', fontsize=14)
    ax.set_ylabel('Gate值', fontsize=14)
    ax.set_title('各分支Gate值（训练后）', fontsize=16, fontweight='bold')
    ax.set_ylim([0.45, 0.55])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'gate_values.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ 已保存: {output_dir / 'gate_values.png'}")

def main():
    print("=" * 60)
    print("门控分组模型结果可视化")
    print("=" * 60)
    
    # 创建输出目录
    output_dir = Path("results/group_branch/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    print("\n加载数据...")
    df = load_data()
    print(f"数据形状: {df.shape}")
    
    # 生成各类图表
    print("\n生成可视化图表...")
    print("-" * 60)
    
    plot_prediction_vs_actual(df, output_dir)
    plot_residual_distribution(df, output_dir)
    plot_branch_contributions(df, output_dir)
    plot_time_series_comparison(df, output_dir)
    plot_gate_values(output_dir)
    
    print("-" * 60)
    print(f"\n✓ 所有图表已保存到: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
