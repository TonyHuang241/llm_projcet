"""步骤 measure：每篇研报概念 DAG 的逻辑结构变量（不调用大模型）。读 graph 步骤的成品（src/graph/load.py：各次运行按
优先级合并后的全部研报，不只是回归样本），每篇研报一行，写 output/measure/report_measures.parquet。下游读这张表用
src/measure/load.py。

变量的口径见 src/measure/variables.py（源点、严格汇点、路径的判定和计数在 src/measure/paths.py）。输出表的列：
  report_id                 研报
  run                       图来自 graph 的哪次运行（同一篇在几次运行里都有时用靠前的，见 src/graph/load.py 的 RUN_ORDER）
  RT_NSRC、RT_LNPATH、RT_PATHLEN   “投资评级”对应的逻辑结构：源点数、ln(1 + 路径数)、平均路径长度；“投资评级”不是严格汇点时缺失
  WG_NSRC、WG_LNPATH、WG_PATHLEN   全文逻辑结构的同样三个量；空图（没有进图的边）时缺失
  RT_N_PATHS、WG_N_PATHS    两套各自的路径条数（整数，缺失同上）
  HAS_RATING_NODE           图里有没有“投资评级”节点（图只由进 DAG 的边构成：概念表里有“投资评级”、但它没有进图的边时
                            这里是 False，所以比按概念表数的篇数少）
  RATING_STRICT_SINK        “投资评级”是不是严格汇点（入度 > 0、出度 = 0）
表里一切都是原始值：不缩尾、不标准化、不筛样本（这些依赖样本，是后面研究代码的事）。

在仓库根目录运行：
    python run.py measure                  按下面的运行参数运行
    python run.py measure --runs deep      只用 deep 这次运行的研报（默认用 src/graph/load.py 的 RUN_ORDER：先 deep 后 pilot）
"""
from src.config import out_dir

# ==================== 运行参数 ====================
OUT_PATH = out_dir('measure') / 'report_measures.parquet'  # 输出表（下游用 src/measure/load.py 读）
# ==================================================

COLUMNS = ['report_id', 'run', 'RT_NSRC', 'RT_LNPATH', 'RT_PATHLEN', 'WG_NSRC', 'WG_LNPATH', 'WG_PATHLEN',
           'RT_N_PATHS', 'WG_N_PATHS', 'HAS_RATING_NODE', 'RATING_STRICT_SINK']


def add_args(ap):
    ap.description = '概念 DAG -> 每篇研报的逻辑结构变量表（原始值，不缩尾、不筛样本）'
    ap.add_argument('--runs', default=None,
                    help='读 graph 的哪几次运行，逗号分隔，同一篇研报在几次运行里都有时用靠前的；默认 src/graph/load.py 的 RUN_ORDER（deep,pilot）')
    ap.add_argument('--output', default=OUT_PATH, help=f'输出 parquet 路径，默认 {OUT_PATH}')


def measure_all(runs=None):
    """runs（None 时用 src/graph/load.py 的 RUN_ORDER）里全部研报的变量表（列见模块说明，行序同 src/graph/load.py 的 owners）。"""
    import pandas as pd

    from src.graph.load import RUN_ORDER, load_graphs, owners
    from src.measure.variables import report_variables
    runs = RUN_ORDER if runs is None else tuple(runs)
    owner = owners(runs)
    graphs = load_graphs(list(owner), runs)
    rows = [{'report_id': rid, 'run': owner[rid], **report_variables(g)} for rid, g in graphs.items()]
    out = pd.DataFrame(rows, columns=COLUMNS)
    for c in ('RT_N_PATHS', 'WG_N_PATHS'):
        out[c] = out[c].astype('Int64')
    return out


def run(args):
    import time
    from pathlib import Path
    t0 = time.time()
    runs = None if args.runs is None else tuple(x.strip() for x in args.runs.split(',') if x.strip())
    out = measure_all(runs)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    print(f'{len(out)} 篇研报（' + '、'.join(f'{k} {v} 篇' for k, v in out.run.value_counts().items()) + f'），写到 {path}')
    print(f'图里有“投资评级”节点 {int(out.HAS_RATING_NODE.sum())} 篇，其中是严格汇点 {int(out.RATING_STRICT_SINK.sum())} 篇；'
          f'全文变量有定义 {int(out.WG_NSRC.notna().sum())} 篇；用时 {time.time() - t0:.1f} 秒')
