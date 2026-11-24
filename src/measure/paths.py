"""源点、汇点的判定和路径计数（不调用大模型）。口径见 src/measure/variables.py 的模块说明。

源点      入度为 0 的点（在给定目标的子图里，就是能到达目标、自身入度为 0 的祖先）
严格汇点  入度 > 0、出度 = 0 的点（孤立点不算汇点）
路径计数  按拓扑序动态规划（每个点到目标的路径 = 各直接后继到目标的路径各加一条边），条数和边数之和都是整数精确计数，
          与穷举简单路径的结果相同（DAG 里的有向路径都是简单路径）。
"""
import networkx as nx


def is_strict_sink(g, v):
    """v 在图里且入度 > 0、出度 = 0。"""
    return v in g and g.in_degree(v) > 0 and g.out_degree(v) == 0


def strict_sinks(g):
    """图里入度 > 0、出度 = 0 的节点。"""
    return [v for v in g if g.in_degree(v) > 0 and g.out_degree(v) == 0]


def path_counts(g, targets):
    """每个能到达 targets 的点（targets 的祖先）到 targets 的路径（至少一条边）：{点: (条数, 边数之和)}。
    多个目标时是到各目标的路径合在一起（一个目标在另一个目标的上游时，经过它到下游目标的路径也算）。"""
    tset = set(targets)
    full, anc = {}, {}  # full：含长度为 0 的路径（点本身是目标）
    for v in reversed(list(nx.topological_sort(g))):
        n = s = 0
        for u in g.successors(v):
            fu = full.get(u)
            if fu:
                n, s = n + fu[0], s + fu[1] + fu[0]
        if n:
            anc[v] = (n, s)
        if v in tset:
            full[v] = (n + 1, s)
        elif n:
            full[v] = (n, s)
    return anc


def sources(g, anc):
    """path_counts 的结果 anc 里的源点（入度为 0 的祖先）。"""
    return [v for v in anc if g.in_degree(v) == 0]
