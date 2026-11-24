"""读 graph 步骤的成品：各次运行（output/graph/<运行名>/data/）的概念表、边表、研报表，并建图。下游步骤通过这里读，不自己拼路径。

同一篇研报（report_id）在几次运行里都有时，用 runs 里靠前的那次运行（默认 RUN_ORDER：先 deep 后 pilot）。
"""
from collections import defaultdict

import networkx as nx
import pandas as pd

from src.config import out_dir
from src.graph.common import DagOut

RUN_ORDER = ('deep', 'pilot')  # report_id 在几次运行里都有时用靠前的
TABLES = ('concepts', 'edges', 'reports')


def data_dir(run):
    """一次运行的结果数据目录 output/graph/<运行名>/data/。"""
    return DagOut(out_dir('graph', run)).data


def owners(runs=RUN_ORDER, report_ids=None):
    """{report_id: 运行名}：每篇研报归哪次运行（在 runs 里靠前的优先）；report_ids 给出时只看这些研报。"""
    want, owner = None if report_ids is None else set(report_ids), {}
    for run in runs:
        for rid in pd.read_parquet(data_dir(run) / 'reports.parquet', columns=['report_id']).report_id:
            if (want is None or rid in want) and rid not in owner:
                owner[rid] = run
    return owner


def read_table(name, runs=RUN_ORDER, report_ids=None, columns=None):
    """几次运行合并后的一张表（name 是 concepts、edges、reports 或统一名称的 concept_unify、unified_edges 等 data/ 下的
    parquet 文件名）：每篇研报只取它所属运行（owners）的行，各次运行按 runs 的顺序上下拼接、运行内保持文件里的行序；
    最后加一列 run（运行名）。columns 只读这些列（report_id 总会读）。"""
    owner = owners(runs, report_ids)
    cols = None if columns is None else list(dict.fromkeys(['report_id', *columns]))
    parts = []
    for run in runs:
        t = pd.read_parquet(data_dir(run) / f'{name}.parquet', columns=cols)
        t = t[[owner.get(r) == run for r in t.report_id]]
        parts.append(t.assign(run=run))
    return pd.concat(parts, ignore_index=True)


def read_tables(runs=RUN_ORDER, report_ids=None, names=TABLES):
    """{表名: read_table(表名)}，默认是概念表、边表、研报表三张。"""
    return {name: read_table(name, runs, report_ids) for name in names}


def build_graph(edges):
    """[(原因, 结果, 引文数)] -> 无环有向图，返回 (图, 为去环删掉的边数)。"""
    g = nx.DiGraph()
    for a, b, s in edges:
        if a != b:
            w = g.edges[a, b]['support'] + s if g.has_edge(a, b) else s
            g.add_edge(a, b, support=w)
    removed = 0
    while not nx.is_directed_acyclic_graph(g):  # 先用快的判断（无环时 find_cycle 要遍历全图，慢约 10 倍），有环时同原来逐个删
        cycle = nx.find_cycle(g)
        a, b = min(((u, v) for u, v, *_ in cycle), key=lambda e: (g.edges[e]['support'], e))
        g.remove_edge(a, b)
        removed += 1
    return g, removed


def load_graphs(report_ids=None, runs=RUN_ORDER):
    """每篇研报的图：{report_id: networkx.DiGraph}，用边表里 in_dag 为真的原始边、按 build_graph 建图（去环）。
    report_ids 给出时顺序同它（没有边的研报是空图），否则是 runs 里全部研报（按 owners 的顺序）。"""
    owner = owners(runs, report_ids)
    report_ids = list(owner) if report_ids is None else list(report_ids)
    E = read_table('edges', runs, report_ids, ['cause', 'effect', 'support', 'in_dag'])
    edges = defaultdict(list)
    for rid, a, b, s in E[E.in_dag][['report_id', 'cause', 'effect', 'support']].itertuples(index=False):
        edges[rid].append((a, b, s))
    return {rid: build_graph(edges[rid])[0] for rid in report_ids}
