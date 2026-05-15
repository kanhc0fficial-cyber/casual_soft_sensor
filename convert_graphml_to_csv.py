"""
convert_graphml_to_csv.py
=========================
将 GraphML 格式的因果图转换为 CSV 边表格式，供 target_causal_projection.py 使用。

用法：
  python convert_graphml_to_csv.py --input 因果发现结果/dynotears_real_dag_xin1.graphml --output data/features/global_edges.csv
"""

import argparse
import networkx as nx
import pandas as pd
from pathlib import Path


def convert_graphml_to_csv(graphml_path, csv_path):
    """
    将 GraphML 文件转换为 CSV 边表。
    
    参数：
      graphml_path: GraphML 文件路径
      csv_path: 输出 CSV 文件路径
    """
    # 读取 GraphML
    G = nx.read_graphml(graphml_path)
    
    print(f"[convert] 读取 GraphML: {graphml_path}")
    print(f"[convert] 节点数: {G.number_of_nodes()}")
    print(f"[convert] 边数: {G.number_of_edges()}")
    
    # 提取边信息
    edges = []
    for source, target, data in G.edges(data=True):
        edge = {
            'source': source,
            'target': target,
            'score': data.get('weight', 1.0),
            'lag': data.get('lag', None),
            'edge_type': 'directed'
        }
        edges.append(edge)
    
    # 转换为 DataFrame
    df = pd.DataFrame(edges)
    
    # 确保输出目录存在
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存为 CSV
    df.to_csv(csv_path, index=False)
    
    print(f"[convert] 保存 CSV: {csv_path}")
    print(f"[convert] CSV 行数: {len(df)}")
    print(f"\n前 5 条边:")
    print(df.head())
    
    return df


def main():
    parser = argparse.ArgumentParser(
        description="将 GraphML 格式的因果图转换为 CSV 边表"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="输入 GraphML 文件路径"
    )
    parser.add_argument(
        "--output",
        default="data/features/global_edges.csv",
        help="输出 CSV 文件路径（默认：data/features/global_edges.csv）"
    )
    args = parser.parse_args()
    
    convert_graphml_to_csv(args.input, args.output)
    print("\n[OK] 转换完成")


if __name__ == "__main__":
    main()
