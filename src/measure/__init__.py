"""步骤 measure：每篇研报概念 DAG 的逻辑结构变量（不调用大模型）。对一个抽取版本的库（src/graph/load.py 的 library_graphs：
库里全部研报，不只是某个样本）每篇研报算一行，写 output/measure/<版本>/report_measures.parquet；与样本无关。
下游按样本读用 src/measure/load.py（清单决定每篇取自哪个版本的表）。

变量的口径见 src/measure/variables.py（源点、严格汇点、路径的判定和计数在 src/measure/paths.py）。输出表的列：
  report_id                 研报
  version                   抽取版本（这张表里都相同；按样本读出的表里标明每篇取自哪个版本）
  RT_NSRC、RT_LNPATH、RT_PATHLEN   “投资评级”对应的逻辑结构：源点数、ln(1 + 路径数)、平均路径长度；“投资评级”不是严格汇点时缺失
  WG_NSRC、WG_LNPATH、WG_PATHLEN   全文逻辑结构的同样三个量；空图（没有进图的边）时缺失
  RT_N_PATHS、WG_N_PATHS    两套各自的路径条数（整数，缺失同上）
  HAS_RATING_NODE           图里有没有“投资评级”节点（图只由进 DAG 的边构成：概念表里有“投资评级”、但它没有进图的边时
                            这里是 False，所以比按概念表数的篇数少）
  RATING_STRICT_SINK        “投资评级”是不是严格汇点（入度 > 0、出度 = 0）
表里一切都是原始值：不缩尾、不标准化、不筛样本（这些依赖样本，是后面研究代码的事）。行序同库的研报表。

在仓库根目录运行：
    python run.py measure                  默认版本 v1
    python run.py measure --version v0     早期试跑的库（只读的版本也可以算变量）
"""
from src.config import DEFAULT_VERSION, measure_dir

FILE = 'report_measures.parquet'
COLUMNS = ['report_id', 'version', 'RT_NSRC', 'RT_LNPATH', 'RT_PATHLEN', 'WG_NSRC', 'WG_LNPATH', 'WG_PATHLEN',
           'RT_N_PATHS', 'WG_N_PATHS', 'HAS_RATING_NODE', 'RATING_STRICT_SINK']


def out_path(version):
    """一个版本的变量表 output/measure/<版本>/report_measures.parquet（下游用 src/measure/load.py 读）。"""
    return measure_dir(version) / FILE


def add_args(ap):
    from src.graph import VERSIONS
    ap.description = '一个抽取版本的库 -> 每篇研报的逻辑结构变量表（原始值，不缩尾、不筛样本），写 output/measure/<版本>/'
    ap.add_argument('--version', default=DEFAULT_VERSION, help=f'抽取版本（{"、".join(VERSIONS)}），默认 {DEFAULT_VERSION}')
    ap.add_argument('--output', default=None, help='输出 parquet 路径，默认 output/measure/<版本>/report_measures.parquet')


def measure_all(version=DEFAULT_VERSION):
    """一个版本的库里全部研报的变量表（列见模块说明，行序同库的研报表）。"""
    import pandas as pd

    from src.graph.load import library_graphs
    from src.measure.variables import report_variables
    graphs = library_graphs(version)
    rows = [{'report_id': rid, 'version': version, **report_variables(g)} for rid, g in graphs.items()]
    out = pd.DataFrame(rows, columns=COLUMNS)
    for c in ('RT_N_PATHS', 'WG_N_PATHS'):
        out[c] = out[c].astype('Int64')
    return out


def run(args):
    import time
    from pathlib import Path

    from src.graph import version_spec
    version_spec(args.version)
    t0 = time.time()
    out = measure_all(args.version)
    path = Path(args.output) if args.output else out_path(args.version)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f'版本 {args.version} 的库：{len(out)} 篇研报，写到 {path}')
    print(f'图里有“投资评级”节点 {int(out.HAS_RATING_NODE.sum())} 篇，其中是严格汇点 {int(out.RATING_STRICT_SINK.sum())} 篇；'
          f'全文变量有定义 {int(out.WG_NSRC.notna().sum())} 篇；用时 {time.time() - t0:.1f} 秒')
