"""
target_causal_projection.py
===========================
从全局 DAG 边表（global_edges.csv）中提取目标变量的因果投影，
生成软测量建模所需的父节点列表、祖先节点、后代节点、DML 任务表和模型配置。

不重新运行因果发现算法。只做已有全局 DAG 的解析、投影和导出。

用法：
  python src/target_causal_projection.py \\
    --edge_path data/features/global_edges.csv \\
    --target concentrate_grade \\
    --output_dir data/features/causal_projection
"""

import argparse
import json
import re
import sys
from pathlib import Path

import networkx as nx
import pandas as pd


# ─── 数据加载 ──────────────────────────────────────────────────────────────────

def load_edges(edge_path: str) -> pd.DataFrame:
    """
    读取全局 DAG 边表 CSV 文件。

    必须包含列：source, target
    可选列：score（缺失时默认 1.0）、lag、edge_type（缺失时默认 "directed"）

    返回标准化后的 DataFrame，列顺序：source, target, score, lag, edge_type
    """
    path = Path(edge_path)
    if not path.exists():
        raise FileNotFoundError(f"边表文件不存在：{edge_path}")

    df = pd.read_csv(path)
    print(f"[load_edges] 读取边表：{path}，共 {len(df)} 条边")

    required_cols = {"source", "target"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"边表缺少必要列：{missing}")

    # 填充可选列
    if "score" not in df.columns:
        df["score"] = 1.0
        print("[load_edges] 缺少 score 列，默认设为 1.0")
    else:
        df["score"] = df["score"].fillna(1.0)

    if "edge_type" not in df.columns:
        df["edge_type"] = "directed"
        print("[load_edges] 缺少 edge_type 列，默认设为 'directed'")
    else:
        df["edge_type"] = df["edge_type"].fillna("directed")

    if "lag" not in df.columns:
        df["lag"] = None

    # 去空行
    df = df.dropna(subset=["source", "target"])
    df["source"] = df["source"].astype(str).str.strip()
    df["target"] = df["target"].astype(str).str.strip()

    return df[["source", "target", "score", "lag", "edge_type"]].reset_index(drop=True)


# ─── 构建图 ────────────────────────────────────────────────────────────────────

def build_graph(df: pd.DataFrame) -> nx.DiGraph:
    """
    从边表 DataFrame 构建 networkx.DiGraph。

    每条边带属性：score, lag, edge_type
    """
    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(
            row["source"],
            row["target"],
            score=float(row["score"]),
            lag=row["lag"],
            edge_type=str(row["edge_type"]),
        )
    print(f"[build_graph] 图节点数：{G.number_of_nodes()}，边数：{G.number_of_edges()}")
    return G


# ─── DAG 检查 ──────────────────────────────────────────────────────────────────

def check_dag(G: nx.DiGraph, log_path: Path) -> bool:
    """
    检查图是否为 DAG。如果不是，将环信息写入 log_path，返回 False。
    不崩溃，继续按有向图解析。
    """
    is_dag = nx.is_directed_acyclic_graph(G)
    if is_dag:
        print("[check_dag] 图为合法 DAG ✓")
    else:
        print("[check_dag] ⚠ 图不是 DAG，检测到环！详情见：" + str(log_path))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        cycles = list(nx.simple_cycles(G))
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(f"检测到 {len(cycles)} 个环：\n\n")
            for i, cycle in enumerate(cycles, 1):
                f.write(f"环 {i}: {' -> '.join(cycle)} -> {cycle[0]}\n")
        print(f"[check_dag] 共 {len(cycles)} 个环，已写入 {log_path}")
    return is_dag


# ─── 节点名解析 ────────────────────────────────────────────────────────────────

_LAG_PATTERN = re.compile(r"^(.+?)_lag(\d+)$")


def parse_node_name(node: str) -> dict:
    """
    解析节点名中的变量名和滞后阶数。

    规则：
      reagent_lag3          -> variable=reagent,          lag=3
      pH_lag2               -> variable=pH,               lag=2
      concentrate_grade_lag0 -> variable=concentrate_grade, lag=0
      concentrate_grade      -> variable=concentrate_grade, lag=None
    """
    m = _LAG_PATTERN.match(node)
    if m:
        return {"variable": m.group(1), "lag": int(m.group(2))}
    return {"variable": node, "lag": None}


# ─── 目标节点解析 ──────────────────────────────────────────────────────────────

def resolve_target_node(G: nx.DiGraph, target: str) -> str:
    """
    确定图中的目标节点名。

    优先级：
      1. {target}_lag0（如 concentrate_grade_lag0）
      2. {target}（如 concentrate_grade）
      3. 报错，并列出图中可能相关的节点
    """
    nodes = set(G.nodes())
    lag0_name = f"{target}_lag0"

    if lag0_name in nodes:
        print(f"[resolve_target_node] 使用目标节点：{lag0_name}")
        return lag0_name

    if target in nodes:
        print(f"[resolve_target_node] 使用目标节点：{target}")
        return target

    # 没找到，列出可能相关的节点
    related = sorted(n for n in nodes if target in n)
    raise ValueError(
        f"目标变量 '{target}' 和 '{lag0_name}' 均不在图中。\n"
        f"图中可能相关的节点：{related}\n"
        f"所有节点：{sorted(nodes)}"
    )


# ─── 父节点提取 ────────────────────────────────────────────────────────────────

def extract_target_parents(G: nx.DiGraph, target_node: str) -> pd.DataFrame:
    """
    提取目标节点的直接父节点 Pa(Y)。

    返回 DataFrame，列：node, variable, lag, edge_score, edge_type
    """
    rows = []
    for parent in G.predecessors(target_node):
        edge_data = G[parent][target_node]
        parsed = parse_node_name(parent)
        rows.append({
            "node": parent,
            "variable": parsed["variable"],
            "lag": parsed["lag"],
            "edge_score": edge_data.get("score", 1.0),
            "edge_type": edge_data.get("edge_type", "directed"),
        })

    df = pd.DataFrame(rows, columns=["node", "variable", "lag", "edge_score", "edge_type"])
    df = df.sort_values("edge_score", ascending=False).reset_index(drop=True)
    print(f"[extract_target_parents] 目标节点 '{target_node}' 的直接父节点数：{len(df)}")
    return df


# ─── 祖先节点提取 ──────────────────────────────────────────────────────────────

def extract_ancestors(G: nx.DiGraph, target_node: str) -> pd.DataFrame:
    """
    提取目标节点的所有祖先节点 Anc(Y)，以及到目标节点的最短有向路径长度。

    返回 DataFrame，列：node, variable, lag, distance_to_target
    """
    ancestors = nx.ancestors(G, target_node)
    rows = []
    for anc in ancestors:
        try:
            dist = nx.shortest_path_length(G, anc, target_node)
        except nx.NetworkXNoPath:
            # nx.ancestors() guarantees a path exists; this branch indicates
            # an unexpected graph inconsistency (e.g., graph mutated mid-run).
            print(f"[extract_ancestors] ⚠ 无法计算 '{anc}' 到 '{target_node}' 的路径长度，设为 None")
            dist = None
        parsed = parse_node_name(anc)
        rows.append({
            "node": anc,
            "variable": parsed["variable"],
            "lag": parsed["lag"],
            "distance_to_target": dist,
        })

    df = pd.DataFrame(rows, columns=["node", "variable", "lag", "distance_to_target"])
    df = df.sort_values("distance_to_target").reset_index(drop=True)
    print(f"[extract_ancestors] 目标节点 '{target_node}' 的祖先节点数：{len(df)}")
    return df


# ─── 后代节点提取 ──────────────────────────────────────────────────────────────

def extract_descendants(G: nx.DiGraph, target_node: str) -> pd.DataFrame:
    """
    提取目标节点的所有后代节点 Desc(Y)。

    返回 DataFrame，列：node, variable, lag, reason
    """
    descendants = nx.descendants(G, target_node)
    rows = []
    for desc in descendants:
        parsed = parse_node_name(desc)
        rows.append({
            "node": desc,
            "variable": parsed["variable"],
            "lag": parsed["lag"],
            "reason": "target_descendant_excluded_to_avoid_leakage",
        })

    df = pd.DataFrame(rows, columns=["node", "variable", "lag", "reason"])
    df = df.sort_values("node").reset_index(drop=True)
    print(f"[extract_descendants] 目标节点 '{target_node}' 的后代节点数：{len(df)}")
    return df


# ─── DML 任务表构建 ────────────────────────────────────────────────────────────

def build_dml_jobs(
    G: nx.DiGraph,
    target_node: str,
    parents_df: pd.DataFrame,
    descendants_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    为每个 target parent 生成一条 DML 任务。

    adjustment_set = Pa(T) ∪ Pa(Y) - {T} - Desc(Y) - {Y}

    字段：treatment, treatment_variable, treatment_lag, outcome, adjustment_set
    """
    parent_nodes_of_target = set(parents_df["node"].tolist())
    desc_nodes = set(descendants_df["node"].tolist())
    # excluded = Desc(Y) ∪ {Y}  (covers the "- {T} - Desc(Y) - {Y}" part of the formula)
    excluded = desc_nodes | {target_node}

    rows = []
    for _, row in parents_df.iterrows():
        treatment = row["node"]

        # Pa(T)：treatment 的父节点
        parents_of_treatment = set(G.predecessors(treatment))

        # adjustment_set = Pa(T) ∪ Pa(Y) - {T} - Desc(Y) - {Y}
        adj_set = (parents_of_treatment | parent_nodes_of_target) - {treatment} - excluded
        adj_str = ";".join(sorted(adj_set)) if adj_set else ""

        rows.append({
            "treatment": treatment,
            "treatment_variable": row["variable"],
            "treatment_lag": row["lag"],
            "outcome": target_node,
            "adjustment_set": adj_str,
        })

    df = pd.DataFrame(
        rows,
        columns=["treatment", "treatment_variable", "treatment_lag", "outcome", "adjustment_set"],
    )
    print(f"[build_dml_jobs] 生成 DML 任务数：{len(df)}")
    return df


# ─── 输出保存 ──────────────────────────────────────────────────────────────────

def save_outputs(
    output_dir: Path,
    target_variable: str,
    target_node: str,
    G: nx.DiGraph,
    is_dag: bool,
    parents_df: pd.DataFrame,
    ancestors_df: pd.DataFrame,
    descendants_df: pd.DataFrame,
    dml_jobs_df: pd.DataFrame,
) -> dict:
    """
    将所有结果写入 output_dir，返回已写入文件路径字典。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. target_parents.csv
    p = output_dir / "target_parents.csv"
    parents_df.to_csv(p, index=False)
    paths["target_parents"] = str(p)

    # 2. target_ancestors.csv
    p = output_dir / "target_ancestors.csv"
    ancestors_df.to_csv(p, index=False)
    paths["target_ancestors"] = str(p)

    # 3. excluded_descendants.csv
    p = output_dir / "excluded_descendants.csv"
    descendants_df.to_csv(p, index=False)
    paths["excluded_descendants"] = str(p)

    # 4. dml_jobs.csv
    p = output_dir / "dml_jobs.csv"
    dml_jobs_df.to_csv(p, index=False)
    paths["dml_jobs"] = str(p)

    # 5. causal_feature_config.json
    config = {
        "target_node": target_node,
        "target_variable": target_variable,
        "causal_parent_nodes": parents_df["node"].tolist(),
        "causal_parent_variables": parents_df["variable"].tolist(),
        "ancestor_nodes": ancestors_df["node"].tolist(),
        "excluded_descendant_nodes": descendants_df["node"].tolist(),
        "dml_jobs_file": "dml_jobs.csv",
    }
    p = output_dir / "causal_feature_config.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    paths["causal_feature_config"] = str(p)

    # 6. projection_summary.json
    summary = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "is_dag": is_dag,
        "target_node": target_node,
        "num_target_parents": len(parents_df),
        "num_target_ancestors": len(ancestors_df),
        "num_target_descendants": len(descendants_df),
        "num_dml_jobs": len(dml_jobs_df),
    }
    p = output_dir / "projection_summary.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    paths["projection_summary"] = str(p)

    return paths


# ─── 主函数 ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="从全局 DAG 边表提取目标变量的因果投影，生成软测量建模配置文件。"
    )
    parser.add_argument(
        "--edge_path",
        default="data/features/global_edges.csv",
        help="全局 DAG 边表 CSV 文件路径（默认：data/features/global_edges.csv）",
    )
    parser.add_argument(
        "--target",
        default="concentrate_grade",
        help="目标变量名（默认：concentrate_grade）",
    )
    parser.add_argument(
        "--output_dir",
        default="data/features/causal_projection",
        help="输出目录（默认：data/features/causal_projection）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    cycle_log = Path("results/logs/dag_cycles.txt")

    print("=" * 60)
    print("目标因果投影  target_causal_projection.py")
    print("=" * 60)
    print(f"边表路径  ：{args.edge_path}")
    print(f"目标变量  ：{args.target}")
    print(f"输出目录  ：{output_dir}")
    print()

    # 1. 读取边表
    try:
        edges_df = load_edges(args.edge_path)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    except ValueError as e:
        print(f"[ERROR] 边表格式错误：{e}")
        sys.exit(1)

    # 2. 构建图
    G = build_graph(edges_df)

    # 3. 检查 DAG
    is_dag = check_dag(G, cycle_log)

    # 4. 确定目标节点
    try:
        target_node = resolve_target_node(G, args.target)
    except ValueError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    # 5. 提取父节点、祖先节点、后代节点
    parents_df = extract_target_parents(G, target_node)
    ancestors_df = extract_ancestors(G, target_node)
    descendants_df = extract_descendants(G, target_node)

    # 6. 构建 DML 任务表
    dml_jobs_df = build_dml_jobs(G, target_node, parents_df, descendants_df)

    # 7. 保存所有输出
    paths = save_outputs(
        output_dir=output_dir,
        target_variable=args.target,
        target_node=target_node,
        G=G,
        is_dag=is_dag,
        parents_df=parents_df,
        ancestors_df=ancestors_df,
        descendants_df=descendants_df,
        dml_jobs_df=dml_jobs_df,
    )

    # 8. 打印输出文件路径
    print()
    print("─" * 60)
    print("输出文件：")
    for key, fpath in paths.items():
        print(f"  {key:30s}: {fpath}")
    print("─" * 60)
    print("✓ 因果投影完成")


if __name__ == "__main__":
    main()
