"""
merge_dags.py
=============
合并多个GraphML DAG文件为单个edges CSV文件

用法：
  python merge_dags.py
"""

import networkx as nx
import pandas as pd
from pathlib import Path

def merge_graphml_to_csv(
    graphml_files: list,
    output_csv: str,
    target_mapping: dict = None,
    min_weight: float = 0.0,
    verbose: bool = True
):
    """
    合并多个GraphML DAG文件为单个edges CSV
    
    Args:
        graphml_files: GraphML文件路径列表
        output_csv: 输出CSV路径
        target_mapping: 节点名称映射字典 {旧名: 新名}
        min_weight: 最小权重阈值（过滤弱边）
        verbose: 是否打印详细信息
    
    Returns:
        DataFrame: 合并后的边表
    """
    all_edges = []
    stats = {}
    
    for gml_file in graphml_files:
        if verbose:
            print(f"\n读取: {gml_file}")
        
        G = nx.read_graphml(gml_file)
        edge_count = 0
        filtered_count = 0
        
        for source, target, data in G.edges(data=True):
            weight = data.get('weight', 1.0)
            
            # 过滤低权重边
            if weight < min_weight:
                filtered_count += 1
                continue
            
            # 应用节点名称映射
            original_source = source
            original_target = target
            if target_mapping:
                source = target_mapping.get(source, source)
                target = target_mapping.get(target, target)
            
            all_edges.append({
                'source': source,
                'target': target,
                'score': float(weight),
                'lag': 0,  # GraphML中没有lag信息，默认0
                'edge_type': 'directed'
            })
            edge_count += 1
        
        stats[gml_file] = {
            'nodes': G.number_of_nodes(),
            'edges': G.number_of_edges(),
            'kept': edge_count,
            'filtered': filtered_count
        }
        
        if verbose:
            print(f"  节点数: {G.number_of_nodes()}")
            print(f"  边数: {G.number_of_edges()}")
            print(f"  保留: {edge_count}, 过滤: {filtered_count}")
    
    # 创建DataFrame
    df = pd.DataFrame(all_edges)
    
    if verbose:
        print(f"\n合并前总边数: {len(df)}")
    
    # 去重（保留第一次出现的边）
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['source', 'target'], keep='first')
    after_dedup = len(df)
    
    if verbose:
        print(f"去重后边数: {after_dedup} (移除 {before_dedup - after_dedup} 条重复边)")
    
    # 保存
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, encoding='utf-8-sig')
    
    if verbose:
        print(f"\n✅ 合并完成！")
        print(f"输出文件: {output_csv}")
        print(f"\n边表统计:")
        print(f"  唯一源节点数: {df['source'].nunique()}")
        print(f"  唯一目标节点数: {df['target'].nunique()}")
        print(f"  总边数: {len(df)}")
        print(f"  权重范围: [{df['score'].min():.4f}, {df['score'].max():.4f}]")
        print(f"  平均权重: {df['score'].mean():.4f}")
    
    return df, stats


def main():
    """主函数"""
    print("=" * 70)
    print("DAG合并工具 - 将GraphML文件合并为edges CSV")
    print("=" * 70)
    
    # 配置
    graphml_files = [
        "因果发现结果/mb_cuts_磁选_real_dag_xin1.graphml",
        "因果发现结果/mb_cuts_塔磨_real_dag_xin1.graphml",
        "因果发现结果/mb_cuts_浮选_real_dag_xin1.graphml",
    ]
    
    output_csv = "data/features/global_edges.csv"
    
    # 节点名称映射：将GraphML中的y_grade映射为实际数据中的y_fx_xin1
    target_mapping = {
        'y_grade': 'y_fx_xin1'
    }
    
    # 最小权重阈值（0表示不过滤）
    min_weight = 0.0
    
    print(f"\n配置:")
    print(f"  输入文件: {len(graphml_files)} 个")
    for f in graphml_files:
        print(f"    - {f}")
    print(f"  输出文件: {output_csv}")
    print(f"  节点映射: {target_mapping}")
    print(f"  最小权重: {min_weight}")
    
    # 执行合并
    df, stats = merge_graphml_to_csv(
        graphml_files=graphml_files,
        output_csv=output_csv,
        target_mapping=target_mapping,
        min_weight=min_weight,
        verbose=True
    )
    
    # 显示前几条边
    print(f"\n前10条边预览:")
    print(df.head(10).to_string(index=False))
    
    # 检查目标变量相关的边
    target_edges = df[df['target'] == 'y_fx_xin1']
    print(f"\n指向目标变量 y_fx_xin1 的边: {len(target_edges)} 条")
    if len(target_edges) > 0:
        print(target_edges.to_string(index=False))
    
    print("\n" + "=" * 70)
    print("✅ 完成！可以在DML残差模型中使用此DAG文件了。")
    print("=" * 70)


if __name__ == "__main__":
    main()
