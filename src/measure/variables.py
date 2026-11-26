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

变量（report_variables）
  全文逻辑结构（WG_*）           整张图：T = 图里全部严格汇点。NSRC = 入度 0 且出度 > 0 的节点数；N_PATHS = 全部源点到全部
                                 严格汇点的路径条数；PATHLEN = 这些路径的平均边数。
  目标对应的逻辑结构             T = 目标节点集合（登记表 TARGETS 的一项给出），子图是 T 及其全部上游；T 为空时三个量都缺失。
                                 现有两个目标：
    rating（RT_*）  “投资评级”对应的逻辑结构：T = {“投资评级”}，只在“投资评级”在图里且是严格汇点时非空；不在图里、
                    或有出边（不是汇点）时三个量都缺失。
    revenue（RV_*） 营收类汇点对应的逻辑结构：T = 图里的严格汇点中名称匹配 REVENUE_PATTERN（收入|营收）且不匹配排除词
                    REVENUE_EXCLUDE（成本收入比|可支配收入|居民收入|财政收入）的节点（re.search，按名称归类，如“营业收入”
                    “利息净收入”“营业收入增速”“海外收入占比”都算）；营收类节点有出边（如“营业收入 → 归母净利润”）时
                    不是汇点，不算。T 为空时缺失。
  文档里的符号：RateSrc、RatePath、RateLen = RT_NSRC、RT_LNPATH、RT_PATHLEN；RevSrc、RevPath、RevLen = RV_NSRC、RV_LNPATH、
  RV_PATHLEN；AllSrc、AllPath、AllLen = WG_NSRC、WG_LNPATH、WG_PATHLEN。这里都是原始值，不缩尾、不标准化、不筛样本（这些
  依赖样本，由后面的研究代码做）。
  以后加别的目标：在 TARGETS 里加一项（列名前缀、文档符号前缀、中文名、名称判定、取目标节点集合的函数），并在
  src/measure/__init__.py 的 COLUMNS 末尾追加它的列。
"""
import re

import numpy as np

from src.measure.paths import is_strict_sink, path_counts, sources, strict_sinks

RATING_NODE = '投资评级'
REVENUE_PATTERN = re.compile(r'收入|营收')  # 营收类节点的名称（re.search）
REVENUE_EXCLUDE = re.compile(r'成本收入比|可支配收入|居民收入|财政收入')  # 名称匹配上面的、但不是营收的排除词
MEAS = ('NSRC', 'LNPATH', 'PATHLEN')  # 每套变量的三个量


def is_revenue_name(name):
    """名称是不是营收类：匹配 REVENUE_PATTERN 且不匹配 REVENUE_EXCLUDE。"""
    return bool(REVENUE_PATTERN.search(name)) and not REVENUE_EXCLUDE.search(name)


def rating_nodes(g):
    """rating 目标的节点集合：“投资评级”是严格汇点时为 [“投资评级”]，否则为空。"""
    return [RATING_NODE] if is_strict_sink(g, RATING_NODE) else []


def revenue_nodes(g):
    """revenue 目标的节点集合：名称是营收类（is_revenue_name）的严格汇点。"""
    return [v for v in strict_sinks(g) if is_revenue_name(v)]


# 目标登记表：目标名 -> prefix 列名前缀、symbol 文档符号前缀（RateSrc / RevSrc …）、name 中文名（“…对应的逻辑结构”）、
# doc 研究表格和文档里的称呼、short 简称（图 1 的例子、正文数字的说明用）、tag 分布工具图表里的两字标签、
# match 概念名称是否属于该目标（研究里数“抽取出该概念的研报”用）、nodes 一张图里的目标节点集合（都是严格汇点，空表示该研报
# 没有定义）、defined 是否有定义的标记列、flags 这个目标在 measure 表里的标记列（研究的 scope 一并读入）、count 目标节点
# 个数的列（目标节点可以不止一个时才有，研究里报告每篇个数的分布和名称频数；rating 只有一个节点，为 None）；
# text 是研究输出（text_numbers.csv 等）里描述这个目标的几句话：rule 范围条件、concept 概念的称呼、match 概念名称的判定、
# sink_item / sink_how “有定义”的篇数的内容和算法说明。
TARGETS = {
    'rating': dict(prefix='RT', symbol='Rate', name='“投资评级”对应的逻辑结构', doc='投资评级概念对应的逻辑结构',
                   short='投资评级', tag='评级', match=lambda n: n == RATING_NODE, nodes=rating_nodes,
                   defined='RATING_STRICT_SINK', flags=['HAS_RATING_NODE', 'RATING_STRICT_SINK'], count=None,
                   text=dict(rule='“投资评级”是严格汇点', concept='“投资评级”', match='name == “投资评级”',
                             sink_item='“投资评级”是汇点的研报篇数', sink_how='投资评级是严格汇点')),
    'revenue': dict(prefix='RV', symbol='Rev', name='营收类汇点对应的逻辑结构', doc='营收类汇点对应的逻辑结构',
                    short='营收类汇点', tag='营收', match=is_revenue_name, nodes=revenue_nodes,
                    defined='HAS_REVENUE_SINK', flags=['N_REVENUE_SINKS', 'HAS_REVENUE_SINK'], count='N_REVENUE_SINKS',
                    text=dict(rule='至少有一个营收类汇点', concept='营收类',
                              match=f'name 匹配 {REVENUE_PATTERN.pattern} 且不匹配 {REVENUE_EXCLUDE.pattern}',
                              sink_item='有营收类汇点的研报篇数', sink_how='至少有一个营收类汇点')),
}
DEFAULT_TARGET = 'rating'


def graph_measures(g, target=None):
    """一张图的源点数、路径数、平均路径长度（定义见模块说明）：target 为节点名、节点名的列表（目标节点集合），或 None（整张图，
    目标是全部严格汇点）。返回 dict：NSRC、N_PATHS、LNPATH、PATHLEN（原始值）；目标不在图里或没有入边（没有路径）时返回 None。
    是否要求目标是严格汇点由调用方决定（report_variables 要求是）。"""
    ts = strict_sinks(g) if target is None else [target] if isinstance(target, str) else list(target)
    ts = [t for t in ts if t in g]
    anc = path_counts(g, ts) if ts else {}
    srcs = sources(g, anc)
    n = sum(anc[v][0] for v in srcs)
    if not n:
        return None
    return {'NSRC': len(srcs), 'N_PATHS': n, 'LNPATH': float(np.log1p(n)), 'PATHLEN': sum(anc[v][1] for v in srcs) / n}


def report_variables(g):
    """一篇研报（一张图）的变量：RT_NSRC、RT_LNPATH、RT_PATHLEN、WG_NSRC、WG_LNPATH、WG_PATHLEN（原始值，NSRC 也记成
    浮点数，没有定义时是 NaN），各自的路径条数 RT_N_PATHS、WG_N_PATHS（没有定义时是 None），HAS_RATING_NODE
    （图里有没有“投资评级”节点）、RATING_STRICT_SINK（它是不是严格汇点）；之后是 revenue 目标的 RV_NSRC、RV_LNPATH、
    RV_PATHLEN、RV_N_PATHS（缺失同上）、N_REVENUE_SINKS（营收类汇点个数，没有时为 0）、HAS_REVENUE_SINK。"""
    out = {}

    def put(pre, m):
        for x in MEAS:
            out[f'{pre}_{x}'] = float(m[x]) if m else np.nan
        out[f'{pre}_N_PATHS'] = m['N_PATHS'] if m else None

    rate = rating_nodes(g)
    put('RT', graph_measures(g, rate) if rate else None)
    put('WG', graph_measures(g, None))
    out['HAS_RATING_NODE'] = RATING_NODE in g
    out['RATING_STRICT_SINK'] = bool(rate)
    rev = revenue_nodes(g)
    put('RV', graph_measures(g, rev) if rev else None)
    out['N_REVENUE_SINKS'] = len(rev)
    out['HAS_REVENUE_SINK'] = bool(rev)
    return out
