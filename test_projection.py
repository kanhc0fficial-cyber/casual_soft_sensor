"""
test_projection.py
==================
测试 target_causal_projection.py 脚本的功能。

创建一个包含 y_grade 边的示例 CSV 文件，然后运行解析脚本。
"""

import pandas as pd
from pathlib import Path
import subprocess
import sys


def create_test_edges():
    """创建测试用的边表 CSV 文件"""
    
    # 创建示例边数据（包含指向 y_grade 的边）
    edges = [
        # 粗选单元内部
        {'source': 'fx_s1_cx1_level', 'target': 'fx_s1_cx2_level', 'score': 0.9, 'lag': 0, 'edge_type': 'directed'},
        {'source': 'fx_s1_cx2_level', 'target': 'fx_s1_cx3_level', 'score': 0.85, 'lag': 0, 'edge_type': 'directed'},
        
        # 粗选 → 扫选
        {'source': 'fx_s1_cx3_level', 'target': 'fx_s1_sx1_level', 'score': 0.8, 'lag': 1, 'edge_type': 'directed'},
        {'source': 'fx_s1_cx3_level', 'target': 'fx_s1_sx2_level', 'score': 0.75, 'lag': 1, 'edge_type': 'directed'},
        
        # 扫选 → 精选
        {'source': 'fx_s1_sx1_level', 'target': 'fx_s1_jx_level', 'score': 0.7, 'lag': 1, 'edge_type': 'directed'},
        {'source': 'fx_s1_sx2_level', 'target': 'fx_s1_jx_level', 'score': 0.65, 'lag': 1, 'edge_type': 'directed'},
        
        # 关键：指向 y_grade 的边
        {'source': 'fx_s1_cx3_level', 'target': 'y_grade', 'score': 0.6, 'lag': 2, 'edge_type': 'directed'},
        {'source': 'fx_s1_jx_level', 'target': 'y_grade', 'score': 0.75, 'lag': 1, 'edge_type': 'directed'},
        {'source': 'fx_s1_sx1_level', 'target': 'y_grade', 'score': 0.55, 'lag': 2, 'edge_type': 'directed'},
    ]
    
    df = pd.DataFrame(edges)
    
    # 保存为 CSV
    output_path = Path("data/features/test_edges.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"[test] 创建测试边表: {output_path}")
    print(f"[test] 边数: {len(df)}")
    print(f"\n边表内容:")
    print(df)
    
    return output_path


def run_projection_script(edge_path):
    """运行 target_causal_projection.py 脚本"""
    
    cmd = [
        sys.executable,
        "src/target_causal_projection.py",
        "--edge_path", str(edge_path),
        "--target", "y_grade",
        "--output_dir", "data/features/test_projection"
    ]
    
    print(f"\n[test] 运行命令: {' '.join(cmd)}")
    print("=" * 70)
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    return result.returncode


def check_outputs():
    """检查输出文件"""
    
    output_dir = Path("data/features/test_projection")
    
    expected_files = [
        "target_parents.csv",
        "target_ancestors.csv",
        "excluded_descendants.csv",
        "dml_jobs.csv",
        "causal_feature_config.json",
        "projection_summary.json"
    ]
    
    print("\n" + "=" * 70)
    print("[test] 检查输出文件:")
    
    all_exist = True
    for filename in expected_files:
        filepath = output_dir / filename
        exists = filepath.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {filename}")
        
        if exists and filename.endswith('.csv'):
            df = pd.read_csv(filepath)
            print(f"       -> {len(df)} 行")
        
        all_exist = all_exist and exists
    
    return all_exist


def main():
    print("=" * 70)
    print("测试 target_causal_projection.py")
    print("=" * 70)
    
    # 1. 创建测试数据
    edge_path = create_test_edges()
    
    # 2. 运行解析脚本
    returncode = run_projection_script(edge_path)
    
    # 3. 检查输出
    if returncode == 0:
        all_exist = check_outputs()
        if all_exist:
            print("\n[OK] 所有测试通过！")
        else:
            print("\n[FAIL] 部分输出文件缺失")
    else:
        print(f"\n[FAIL] 脚本执行失败，返回码: {returncode}")


if __name__ == "__main__":
    main()
