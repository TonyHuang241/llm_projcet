"""概念 DAG 的读图和结构指标（不调用大模型）。预测准确度回归（src/dag_forecast.py）和实验脚本共用。

图        每篇研报一张图：DAG 输出目录 edges.parquet 里 in_dag 为真的原始边，用 src/viz/plot_dag_structure.build_graph 建图（去环）；
          report_id 在多个目录里都有时用前面的目录（DAG_DIRS 的顺序）。
路径计数  按拓扑序动态规划（每个点到目标的路径 = 各直接后继到目标的路径各加一条边），条数、边数之和、最长边数都是整数精确计数，
          与穷举简单路径的结果相同（DAG 里的有向路径都是简单路径）。

五个结构指标（graph_measures 一张图、five_measures 一批图）：给定目标节点集合 T（一个或多个节点名）
  子图      T 加上它们的全部祖先（能到达 T 中任一节点的点）
  NSRC      子图里的源点数（入度为 0 的祖先）
  N_PATHS   从各源点到 T 的不同有向路径条数（多个目标时以各目标为终点的路径合在一起）；LNPATH = ln(1 + N_PATHS)
  PATHLEN   这些路径的平均边数
  NNODE     子图节点数（含 T）
  DEPTH     这些路径里最长的边数（= 子图的最长路径边数）
不指定目标（target=None）= 整张图，T = 图里全部严格汇点（入度 > 0、出度 = 0）：
  NSRC = 入度 0 且出度 > 0 的节点数；N_PATHS = 全部源点到全部严格汇点的简单路径条数（路径长度按边数）；
  PATHLEN = 这些路径的平均边数；NNODE = 图的节点数；DEPTH = 最长路径的边数。
  与 --x dag 用的 report_panel（src/viz/plot_dag_structure.measure：源点 = 入度 0、汇点 = 出度 0，路径 = 源点到汇点）口径相同
  （791 张图都没有孤立点，两种源点/汇点定义一致）；但 report_panel 的 PATHLEN 四舍五入到 4 位小数，RATIO 是源点数 ÷ 汇点数（不是 NSRC），
  这里不取整。
"""
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import dag_outcomes_reg as R
from src.config import ROOT_DIR

DAG_DIRS = (ROOT_DIR / 'output_dag' / 'deep' / 'data', ROOT_DIR / 'output_dag' / 'data')  # report_id 重复时用前面的
RATING_NODE = '投资评级'
MEAS = ['NSRC', 'LNPATH', 'PATHLEN', 'NNODE', 'DEPTH']
FINAL_LIST = ROOT_DIR / 'output_dag' / 'forecast' / 'final_indicators' / 'final_indicator_list.csv'
# 汇点按名称归类（先判估值类，命中且不命中排除词就是 VAL；否则判现金流类；都不命中为 OTHER）
VAL_RE = r'评级|目标价|估值|市值|股价|市盈率|市净率|市销率|投资价值|配置价值|上涨空间|合理价值|每股价值|安全边际|EV/|(?<![A-Za-z])(?:PE|PB|PS|PEG)(?![A-Za-z])'
VAL_EXCLUDE = r'市值管理'
CF_RE = r'收入|营收|净利润|利润|毛利|净利率|费用|盈利|EPS|每股收益|现金流|业绩|ROE|ROIC|ROA|净资产收益率|分红|股息|派息'
CFCORE_RE = r'收入|营收|利润|业绩'  # 核心现金流类（core=True 时代替 CF_RE）
CFCORE_EXCLUDE = r'利润率'


# ---------------------------------------------------------------- 读图
@lru_cache(maxsize=4)
def _load_graphs(report_ids, dag_dirs):
    from src.viz.plot_dag_structure import build_graph
    want, owner = set(report_ids), {}
    for i, d in enumerate(dag_dirs):
        for rid in pq.read_table(Path(d) / 'reports.parquet', columns=['report_id']).column(0).to_pylist():
            if rid in want and rid not in owner:
                owner[rid] = i
    edges = defaultdict(list)
    for i, d in enumerate(dag_dirs):
        for e in pq.read_table(Path(d) / 'edges.parquet', columns=['report_id', 'cause', 'effect', 'support', 'in_dag']).to_pylist():
            if e['in_dag'] and owner.get(e['report_id']) == i:
                edges[e['report_id']].append((e['cause'], e['effect'], e['support']))
    return {rid: build_graph(edges[rid])[0] for rid in report_ids}


def load_graphs(report_ids, dag_dirs=DAG_DIRS):
    """每篇研报的图：{report_id: networkx.DiGraph}（顺序同 report_ids）。同一进程里同样的参数只读一次（有缓存，不要原地修改返回的图）。"""
    return dict(_load_graphs(tuple(report_ids), tuple(str(d) for d in dag_dirs)))


def strict_sinks(g):
    """图里入度 > 0、出度 = 0 的节点。"""
    return [v for v in g if g.in_degree(v) > 0 and g.out_degree(v) == 0]


def classify_sink(name, core=False):
    """汇点名称 -> 'VAL'（估值类）/ 'CF'（现金流类）/ 'OTHER'；core=True 时现金流类换成核心口径 'CFCORE'
    （匹配 CFCORE_RE 且不匹配 CFCORE_EXCLUDE），估值类的判断不变。"""
    if re.search(VAL_RE, name) and not re.search(VAL_EXCLUDE, name):
        return 'VAL'
    if core:
        return 'CFCORE' if re.search(CFCORE_RE, name) and not re.search(CFCORE_EXCLUDE, name) else 'OTHER'
    return 'CF' if re.search(CF_RE, name) else 'OTHER'


def final_names(path=FINAL_LIST, exclude=()):
    """最终指标清单里 status == '入选' 的节点名称（tier A、B 都要）；exclude 为要去掉的 category（如 ('评级',)）。"""
    f = pd.read_csv(path, encoding='utf-8-sig', dtype={'name': str})
    return frozenset(f.loc[(f.status == '入选') & ~f.category.isin(list(exclude)), 'name'])


# ---------------------------------------------------------------- 路径计数
def path_counts(g, targets):
    """每个能到达 targets 的点（targets 的祖先）到 targets 的路径（至少一条边）：{点: (条数, 边数之和, 最长边数)}。
    多个目标时是到各目标的路径合在一起（一个目标在另一个目标的上游时，经过它到下游目标的路径也算）。"""
    tset = set(targets)
    full, anc = {}, {}  # full：含长度为 0 的路径（点本身是目标）
    for v in reversed(list(nx.topological_sort(g))):
        n = s = 0
        mx = -1
        for u in g.successors(v):
            fu = full.get(u)
            if fu:
                n, s, mx = n + fu[0], s + fu[1] + fu[0], max(mx, fu[2] + 1)
        if n:
            anc[v] = (n, s, mx)
        if v in tset:
            full[v] = (n + 1, s, max(mx, 0))
        elif n:
            full[v] = (n, s, mx)
    return anc


def _resolve(g, target):
    """target -> 图里的目标节点列表（None = 全部严格汇点；字符串 = 一个节点名；其余当作节点名的集合）。"""
    if target is None:
        return strict_sinks(g)
    return [target] if isinstance(target, str) else [t for t in target if t in g]


def graph_measures(g, target=None):
    """一张图的五个结构指标（定义见模块说明）：target 为节点名、节点名的元组/集合，或 None（整张图）。
    返回 dict：NSRC、N_PATHS、LNPATH、PATHLEN、NNODE、DEPTH（原始值，不缩尾）和 N_EDGES（子图边数）、N_ANC（祖先数，不含目标）；
    目标不在图里或没有入边（没有路径）时返回 None。是否要求目标是严格汇点由调用方决定（five_measures 的 strict）。"""
    ts = [t for t in _resolve(g, target) if t in g]
    anc = path_counts(g, ts) if ts else {}
    srcs = [v for v in anc if g.in_degree(v) == 0]
    n = sum(anc[v][0] for v in srcs)
    if not n:
        return None
    nodes = set(anc) | set(ts)
    return {'NSRC': len(srcs), 'N_PATHS': n, 'LNPATH': float(np.log1p(n)), 'PATHLEN': sum(anc[v][1] for v in srcs) / n,
            'NNODE': g.number_of_nodes() if target is None else len(nodes), 'DEPTH': max(anc[v][2] for v in srcs),
            'N_EDGES': g.number_of_edges() if target is None else sum(1 for u in nodes for w in g.successors(u) if w in nodes),
            'N_ANC': len(set(anc) - set(ts))}


def five_measures(graphs, target=RATING_NODE, strict=True):
    """一批图的五个指标：graphs = {report_id: 图}（load_graphs 的结果）。
    target 为节点名：strict=True 时只保留该节点是严格汇点（入度 > 0、出度 = 0）的研报，strict=False 时保留它在图里且有入边的研报；
    target=None：整张图（全部有路径的研报）。
    返回 DataFrame：report_id、N_PATHS、N_EDGES、各指标原始值 *_RAW，以及在返回的研报里按 1%/99% 缩尾的同名列（NSRC、LNPATH、…）。"""
    rows = []
    for rid, g in graphs.items():
        if strict and target is not None and not all(t in g and g.in_degree(t) > 0 and g.out_degree(t) == 0
                                                     for t in _resolve(g, target)):
            continue
        m = graph_measures(g, target)
        if m is None:
            continue
        rows.append({'report_id': rid, 'N_PATHS': m['N_PATHS'], 'N_EDGES': m['N_EDGES'],
                     **{x + '_RAW': float(m[x]) for x in MEAS}})
    out = pd.DataFrame(rows, columns=['report_id', 'N_PATHS', 'N_EDGES'] + [x + '_RAW' for x in MEAS])
    for x in MEAS:
        out[x] = R.winsor(out[x + '_RAW'])
    return out


# ---------------------------------------------------------------- 预测回归用的汇点指标（R_ 列）
def sink_measures(report_ids, dag_dirs=DAG_DIRS, targets=(RATING_NODE,), count=False, strict=False, final_exclude=()):
    """每篇研报的目标节点（汇点）指标：R_HAS_NODE（任一目标节点在图里）、R_N_ANC（祖先数，不含目标节点本身）、R_N_SOURCES、
    R_N_PATHS、R_OUT_DEG（目标节点出度之和），未缩尾的 R_RATIO_RAW、R_LNPATH_RAW、R_PATHLEN_RAW（没有路径记缺失），
    缩尾后的 R_RATIO、R_LNPATH、R_PATHLEN（在有值的研报里 1%/99%）和 has_rating（R_LNPATH_RAW 非缺失）。
    targets 是节点名元组时按名称取节点（R_RATIO = 源点数）；是类别名（'VAL'、'CF'、'CFCORE'）时取图里该类的全部严格汇点，
    R_RATIO = 源点数 ÷ 汇点个数，另存 R_N_SINKS；count=True 时 R_RATIO 直接用源点数。'CFCORE' 用 classify_sink(core=True)；
    'FINAL' 不归类，取名称在 final_names(exclude=final_exclude) 里的严格汇点。
    strict=True（用于按名称取节点）：只保留入度 > 0、出度 = 0 的目标节点，有出边的当作不在图里（R_HAS_NODE 为假、指标缺失）。
    路径计数同 graph_measures（R_N_SOURCES = NSRC、R_N_PATHS = N_PATHS、R_PATHLEN_RAW = PATHLEN、R_N_ANC = NNODE − 目标个数）。"""
    by_class = isinstance(targets, str)
    core = targets == 'CFCORE'
    names = final_names(exclude=final_exclude) if targets == 'FINAL' else None
    in_class = (lambda v: v in names) if names is not None else (lambda v: classify_sink(v, core) == targets)  # noqa: E731
    rows = []
    for rid, g in load_graphs(report_ids, dag_dirs).items():
        present = [v for v in strict_sinks(g) if in_class(v)] if by_class else [t for t in targets if t in g]
        if strict:
            present = [t for t in present if g.in_degree(t) > 0 and g.out_degree(t) == 0]
        rec = {'report_id': rid, 'R_HAS_NODE': bool(present)}
        if by_class:
            rec['R_N_SINKS'] = len(present)
        if present:
            anc = path_counts(g, present)
            srcs = [v for v in anc if g.in_degree(v) == 0]
            n = sum(anc[v][0] for v in srcs)
            rec.update(R_N_ANC=len(set(anc) - set(present)), R_N_SOURCES=len(srcs), R_N_PATHS=n,
                       R_OUT_DEG=sum(g.out_degree(t) for t in present))
            if n:
                rec.update(R_RATIO_RAW=float(len(srcs)) / (len(present) if by_class and not count else 1), R_LNPATH_RAW=float(np.log1p(n)),
                           R_PATHLEN_RAW=sum(anc[v][1] for v in srcs) / n)
        rows.append(rec)
    out = pd.DataFrame(rows)
    for x in ('R_RATIO', 'R_LNPATH', 'R_PATHLEN'):
        out[x] = R.winsor(out[x + '_RAW'])
    out['has_rating'] = out.R_LNPATH_RAW.notna()
    return out


def sink_classification(report_ids, dag_dirs=DAG_DIRS, core=False):
    """样本研报里每个严格汇点名称的类别和出现的研报篇数（core=True 时用核心现金流类口径）。"""
    cnt = Counter(v for g in load_graphs(report_ids, dag_dirs).values() for v in set(strict_sinks(g)))
    out = pd.DataFrame({'name': list(cnt), 'n_reports': list(cnt.values())})
    out['class'] = out.name.map(lambda n: classify_sink(n, core))
    return out.sort_values(['class', 'n_reports', 'name'], ascending=[True, False, True])[['name', 'class', 'n_reports']]
