"""
使用Optuna最佳超参数重新训练并保存模型权重
"""

import json
import sys
from pathlib import Path

# 简单方法：直接调用训练脚本，但修改配置文件使用最佳超参数

def update_config_with_best_params():
    """更新配置文件使用最佳超参数"""
    
    print("=" * 60)
    print("步骤1: 更新门控分组模型配置")
    print("=" * 60)
    
    # 读取最佳超参数
    gb_best_params_path = Path("results/group_branch_test_optuna/best_params.json")
    with open(gb_best_params_path, 'r') as f:
        gb_best_params = json.load(f)
    
    print("门控分组最佳超参数:")
    for key, value in gb_best_params.items():
        print(f"  {key}: {value}")
    
    # 读取配置文件
    import yaml
    config_path = Path("configs/group_branch_test.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 更新超参数
    cfg['lr'] = gb_best_params['lr']
    cfg['batch_size'] = gb_best_params['batch_size']
    
    if 'groups' in cfg:
        for group_name in cfg['groups'].keys():
            if f'hidden_dim_{group_name}' in gb_best_params:
                cfg['groups'][group_name]['hidden_dim'] = gb_best_params[f'hidden_dim_{group_name}']
    
    if 'model' in cfg:
        cfg['model']['gate_init'] = gb_best_params['gate_init']
    
    # 保存更新后的配置
    updated_config_path = Path("configs/group_branch_test_best.yaml")
    with open(updated_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    
    print(f"✅ 已保存更新后的配置: {updated_config_path}")
    
    print("\n" + "=" * 60)
    print("步骤2: 更新DML残差模型配置")
    print("=" * 60)
    
    # 读取DML最佳超参数
    dml_best_params_path = Path("results/residual_soft_sensor_test_optuna/best_params.json")
    with open(dml_best_params_path, 'r') as f:
        dml_best_params = json.load(f)
    
    print("DML残差最佳超参数:")
    for key, value in dml_best_params.items():
        print(f"  {key}: {value}")
    
    # 读取配置文件
    config_path = Path("configs/residual_soft_sensor_test.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    
    # 更新超参数
    cfg['lstm_hidden_size'] = dml_best_params['lstm_hidden_size']
    cfg['lstm_num_layers'] = dml_best_params['lstm_num_layers']
    cfg['lstm_dropout'] = dml_best_params['lstm_dropout']
    cfg['lstm_lr'] = dml_best_params['lstm_lr']
    cfg['lstm_batch_size'] = dml_best_params['lstm_batch_size']
    
    # 保存更新后的配置
    updated_config_path = Path("configs/residual_soft_sensor_test_best.yaml")
    with open(updated_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    
    print(f"✅ 已保存更新后的配置: {updated_config_path}")
    
    return True


if __name__ == "__main__":
    print("准备使用Optuna最佳超参数重新训练模型...")
    print("这将创建新的配置文件，然后需要手动运行训练脚本\n")
    
    update_config_with_best_params()
    
    print("\n" + "=" * 60)
    print("✅ 配置文件已更新！")
    print("=" * 60)
    
    print("\n下一步：运行以下命令训练并保存模型\n")
    
    print("1. 训练门控分组模型:")
    print("   python scripts/train_group_branch.py --config configs/group_branch_test_best.yaml")
    print()
    
    print("2. 训练DML残差模型:")
    print("   python scripts/train_dml_residual_soft_sensor.py --config configs/residual_soft_sensor_test_best.yaml")
    print()
    
    print("这两个脚本已经包含了模型保存功能，会自动保存权重文件。")
