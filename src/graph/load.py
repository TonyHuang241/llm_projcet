"""读 graph 步骤的成品：按样本从各抽取版本的库（output/graph/<版本>/data/）里取概念表、边表、研报表，并建图。
下游步骤通过这里读，不自己拼路径。

样本清单（src/samples/<样本>.csv）决定每篇研报取自哪个版本；各版本按样本登记的版本顺序上下拼接（hs300_deep：v1 在前、
v0 在后），版本内保持库里文件的行序——回归对求和顺序敏感，这个行序不能变。样本里还没有抽取结果的研报不出现。
version 参数见 src/samples/__init__.py 的 assignment：不给（或给样本的主版本）时照清单；给别的版本时样本的全部研报都取自它。
"""
from collections import defaultdict

import networkx as nx
import pandas as pd

from src import samples
from src.config import DEFAULT_SAMPLE, graph_dir
from src.graph.store import DagOut

TABLES = ('concepts', 'edges', 'reports')


def data_dir(version):
    """一个版本的库的结果数据目录 output/graph/<版本>/data/。"""
    return DagOut(graph_dir(version)).data


def library_table(name, version, columns=None):
    """一个版本的库里的整张表（不按样本筛选），name 是 data/ 下的 parquet 文件名（不带扩展名）。"""
    return pd.read_parquet(data_dir(version) / f'{name}.parquet', columns=columns)


def owners(sample=DEFAULT_SAMPLE, report_ids=None, version=None):
    """{report_id: 版本}：样本里每篇研报取自哪个版本（只含库里已有结果的），按读出的行序（版本按登记顺序，版本内按库的研报表
    的行序）；report_ids 给出时只看这些研报。"""
    t, order = samples.assignment(sample, version)
    want = None if report_ids is None else set(report_ids)
    plan = {rid: v for rid, v in zip(t.report_id, t.version) if want is None or rid in want}
    owner = {}
    for v in order:
        for rid in library_table('reports', v, ['report_id']).report_id:
            if plan.get(rid) == v and rid not in owner:
                owner[rid] = v
    return owner


def read_table(name, sample=DEFAULT_SAMPLE, report_ids=None, columns=None, version=None):
    """样本的一张表（name 是 concepts、edges、reports 或统一名称的 concept_unify、unified_edges 等 data/ 下的 parquet
    文件名）：每篇研报只取它所属版本（owners）的行，各版本按登记顺序上下拼接、版本内保持文件里的行序；最后加一列 version。
    columns 只读这些列（report_id 总会读）。"""
    owner = owners(sample, report_ids, version)
    cols = None if columns is None else list(dict.fromkeys(['report_id', *columns]))
    parts = []
    for v in dict.fromkeys(samples.assignment(sample, version)[1]):
        t = library_table(name, v, cols)
        t = t[[owner.get(r) == v for r in t.report_id]]
        parts.append(t.assign(version=v))
    return pd.concat(parts, ignore_index=True)


def read_tables(sample=DEFAULT_SAMPLE, report_ids=None, names=TABLES, version=None):
    """{表名: read_table(表名)}，默认是概念表、边表、研报表三张。"""
    return {name: read_table(name, sample, report_ids, version=version) for name in names}


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


def graphs_from_edges(E, report_ids):
    """边表 E（含 report_id、cause、effect、support、in_dag）-> {report_id: 图}，用 in_dag 为真的原始边、按 build_graph 建图；
    顺序同 report_ids，没有边的研报是空图。"""
    edges = defaultdict(list)
    for rid, a, b, s in E[E.in_dag][['report_id', 'cause', 'effect', 'support']].itertuples(index=False):
        edges[rid].append((a, b, s))
    return {rid: build_graph(edges[rid])[0] for rid in report_ids}


def load_graphs(report_ids=None, sample=DEFAULT_SAMPLE, version=None):
    """样本里每篇研报的图：{report_id: networkx.DiGraph}。report_ids 给出时顺序同它（没有边的研报是空图），否则是样本里
    全部已有结果的研报（按 owners 的顺序）。"""
    owner = owners(sample, report_ids, version)
    report_ids = list(owner) if report_ids is None else list(report_ids)
    E = read_table('edges', sample, report_ids, ['cause', 'effect', 'support', 'in_dag'], version)
    return graphs_from_edges(E, report_ids)


def library_graphs(version):
    """一个版本的库里全部研报的图：{report_id: 图}，顺序同库的研报表（measure 步骤按版本算变量时用）。"""
    report_ids = list(library_table('reports', version, ['report_id']).report_id)
    E = library_table('edges', version, ['report_id', 'cause', 'effect', 'support', 'in_dag'])
    return graphs_from_edges(E, report_ids)
