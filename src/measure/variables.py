"""一张概念 DAG 的逻辑结构变量（不调用大模型）。图来自 src/graph/load.py 的 load_graphs：每篇研报一张图，用边表里 in_dag
为真的原始边建图（去环）。

基本概念
  源点      入度为 0 的点（没有其他概念指向它）
  严格汇点  入度 > 0、出度 = 0 的点（有概念指向它、它不再指向其他概念；孤立点不算）
  路径      从源点沿边走到目标节点的有向路径（至少一条边）；经过的概念序列不同就算不同的路径，长度按边数计。
            按拓扑序动态规划精确计数（src/measure/paths.py 的 path_counts），与穷举简单路径的结果相同

单张图的三个量（graph_measures）：给定目标节点集合 T
  子图      T 加上它们的全部祖先（能到达 T 中任一节点的点）
  NSRC      子图里的源点数（入度为 0 的祖先）
  N_PATHS   从各源点到 T 的不同有向路径条数；多个目标时以各目标为终点的路径合在一起
            （一个目标在另一个目标的上游时，经过它到下游目标的路径也算）；LNPATH = ln(1 + N_PATHS)
  PATHLEN   这些路径的平均边数（每条路径权重相同）
  没有路径（目标不在图里、没有入边，或空图）时三个量都缺失。

两套变量（report_variables）
  全文逻辑结构（WG_*）           整张图：T = 图里全部严格汇点。NSRC = 入度 0 且出度 > 0 的节点数；N_PATHS = 全部源点到全部
                                 严格汇点的路径条数；PATHLEN = 这些路径的平均边数。
  “投资评级”对应的逻辑结构（RT_*）  T = {“投资评级”}，子图是“投资评级”节点及其全部上游。只在“投资评级”在图里且是严格汇点时
                                 有定义；不在图里、或有出边（不是汇点）时三个量都缺失。
  文档里的符号：RateSrc、RatePath、RateLen = RT_NSRC、RT_LNPATH、RT_PATHLEN；AllSrc、AllPath、AllLen = WG_NSRC、WG_LNPATH、
  WG_PATHLEN。这里都是原始值，不缩尾、不标准化、不筛样本（这些依赖样本，由后面的研究代码做）。
"""
import numpy as np

from src.measure.paths import is_strict_sink, path_counts, sources, strict_sinks

RATING_NODE = '投资评级'
MEAS = ('NSRC', 'LNPATH', 'PATHLEN')  # 每套变量的三个量
PREFIX = {'RT': RATING_NODE, 'WG': None}  # 变量前缀 -> 目标节点（None 为整张图）


def graph_measures(g, target=None):
    """一张图的源点数、路径数、平均路径长度（定义见模块说明）：target 为节点名，或 None（整张图，目标是全部严格汇点）。
    返回 dict：NSRC、N_PATHS、LNPATH、PATHLEN（原始值）；目标不在图里或没有入边（没有路径）时返回 None。
    是否要求目标是严格汇点由调用方决定（report_variables 要求是）。"""
    ts = strict_sinks(g) if target is None else [target]
    ts = [t for t in ts if t in g]
    anc = path_counts(g, ts) if ts else {}
    srcs = sources(g, anc)
    n = sum(anc[v][0] for v in srcs)
    if not n:
        return None
    return {'NSRC': len(srcs), 'N_PATHS': n, 'LNPATH': float(np.log1p(n)), 'PATHLEN': sum(anc[v][1] for v in srcs) / n}


def report_variables(g):
    """一篇研报（一张图）的两套变量：RT_NSRC、RT_LNPATH、RT_PATHLEN、WG_NSRC、WG_LNPATH、WG_PATHLEN（原始值，NSRC 也记成
    浮点数，没有定义时是 NaN），各自的路径条数 RT_N_PATHS、WG_N_PATHS（没有定义时是 None），以及 HAS_RATING_NODE
    （图里有没有“投资评级”节点）、RATING_STRICT_SINK（它是不是严格汇点）。"""
    strict = is_strict_sink(g, RATING_NODE)
    out = {}
    for pre, target in PREFIX.items():
        m = graph_measures(g, target) if target is None or strict else None
        for x in MEAS:
            out[f'{pre}_{x}'] = float(m[x]) if m else np.nan
        out[f'{pre}_N_PATHS'] = m['N_PATHS'] if m else None
    out['HAS_RATING_NODE'] = RATING_NODE in g
    out['RATING_STRICT_SINK'] = strict
    return out
