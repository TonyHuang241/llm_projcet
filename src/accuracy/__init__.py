"""步骤 accuracy：分析师盈利预测准确度和控制变量（变量层，只出原始值）。

研报范围 = graph 步骤能读到的全部研报（src/graph/load.py，deep 和 pilot 两次运行合并，同一篇取 deep 的），公司代码 = 研报表
stock_code 的前 6 位，发布日 = 研报表的 date。对每篇研报匹配它本身在 data/EST_COMP_TN 里的预测行（RPT_YR_T 记为 T），按预测窗口
H1、H2、H3（第 h 年预测 PROFIT_Th，目标年度 T + h，逐篇不同）分别算绝对预测误差、相对同行的预测误差 PMAFE、排名分数 Score、
同行时滞 PeerLag，另算六个控制变量；对研报出现过的每个（公司，券商）按窗口 h 算券商 2024 年内最后一条第 h 年预测的 PMAFE、
时效差 DAGE 和该预测发布日上的控制变量。口径见 forecasts.py、peers.py、controls.py 的模块说明。窗口相关的列都带后缀 _Hh，
主窗口由研究层选（src/studies/forecast_accuracy/sample.py 的 MAIN_H），这里三个窗口都给出。

这里不做任何依赖样本的处理：不缩尾、不标准化、不选回归样本，也不给券商分组、不配行业（这些在研究层 src/studies/ 里做）；
基础样本标记 base 逐篇可判定，作为一列给出。

结果写到 output/accuracy/：report_accuracy.parquet（每篇研报一行）、report_peers.parquet（同行明细）、pair_accuracy.parquet
（每个公司-券商对一行）。各列的含义见 src/accuracy/load.py，下游读结果也用它。

在仓库根目录运行：
    python run.py accuracy                     全部重算（十几秒），覆盖 output/accuracy/ 下的三张表
    python run.py accuracy --output 目录        另写到别的目录（核对用）
"""
from src.config import out_dir

STEP = 'accuracy'
FILES = {'report': 'report_accuracy.parquet', 'peers': 'report_peers.parquet', 'pair': 'pair_accuracy.parquet'}


def build():
    """算出三张表，返回 (report_accuracy, report_peers, pair_accuracy)。"""
    import pandas as pd

    from src.accuracy import controls, forecasts, peers
    from src.accuracy.forecasts import HORIZONS
    from src.graph.load import read_table

    r = read_table('reports', columns=['stock_code', 'date', 'broker'])
    rep = pd.DataFrame({'report_id': r.report_id, 'code': r.stock_code.str[:6], 'date': r.date, 'broker': r.broker})
    assert rep.report_id.is_unique
    codes = sorted(set(rep.code))
    est = forecasts.load_est(codes)
    lg = forecasts.to_long(est)
    ann = forecasts.load_annual(codes)

    d = forecasts.report_forecasts(rep, est, ann)
    b = d[d.base].reset_index(drop=True)
    pm, detail = peers.report_peers(b, lg)
    ctl = controls.controls_at(d.assign(key=d.report_id)[['key', 'code', 'date', 'T']], ann).rename(columns={'key': 'report_id'})
    R = (rep.merge(d.drop(columns=['code', 'date']), on='report_id', how='left', validate='1:1')
         .merge(pm, on='report_id', how='left', validate='1:1').merge(ctl, on='report_id', how='left', validate='1:1'))
    report_cols, pair_cols = columns()
    R = R[report_cols]

    P = peers.pair_table(R[['report_id', 'code', 'ORG_CL', 'date']], lg, ann)
    ev = pd.concat([P[['code', 'ORG_CL', f'EVAL_DT_H{h}', f'EVAL_T_H{h}']].dropna()
                    .rename(columns={f'EVAL_DT_H{h}': 'date', f'EVAL_T_H{h}': 'T'}).assign(h=h) for h in HORIZONS], ignore_index=True)
    ev['key'] = [f'{c}|{o}|{h}' for c, o, h in zip(ev.code, ev.ORG_CL, ev.h)]
    ctl = controls.controls_at(ev[['key', 'code', 'date', 'T']], ann)  # 三个窗口的被评价预测一起算（日行情只读一次）
    ctl = ev[['code', 'ORG_CL', 'h', 'key']].merge(ctl, on='key', validate='1:1')
    for h in HORIZONS:
        c = ctl[ctl.h == h].drop(columns=['h', 'key']).rename(columns={v: f'{v}_H{h}' for v in controls.CTRL})
        P = P.merge(c, on=['code', 'ORG_CL'], how='left', validate='1:1')
    return R, detail, P[pair_cols]


def columns():
    """(report_accuracy 的列, pair_accuracy 的列)；窗口相关的列按 h = 1、2、3 依次排。"""
    from src.accuracy.controls import CTRL
    from src.accuracy.forecasts import HORIZONS
    rep = (['report_id', 'code', 'date', 'broker', 'ORG_CL', 'RPT_YR_T', 'matched', 'base']
           + [f'{v}_H{h}' for h in HORIZONS for v in ('TY', 'LNHORIZON', 'F', 'A', 'ABSERR')]
           + [f'{v}_H{h}' for h in HORIZONS for v in ('PMAFE', 'SCORE', 'N_PEERS', 'PEERLAG', 'LAGMED', 'LNNPEERS',
                                                       'PMAFEW30', 'N_PEERSW30', 'PEERLAGW30')]
           + CTRL)
    pair = (['code', 'ORG_CL', 'n_reports', 'last_deep_dt']
            + [f'{v}_H{h}' for h in HORIZONS for v in ['EVAL_RPT_ID', 'EVAL_DT', 'EVAL_F', 'EVAL_T', 'TY', 'A', 'AFE', 'N_GROUP', 'AGE',
                                                       'MAFE', 'PMAFE', 'DAGE', 'EVAL_LAST_TY', 'EVAL_IS_DEEP', 'EVAL_GAP'] + CTRL])
    return rep, pair



def add_args(ap):
    ap.description = '分析师预测准确度（PMAFE、Score、同行时滞）和控制变量的原始值：研报层面和公司-券商层面'
    ap.add_argument('--output', default=None, help=f'输出目录，默认 output/{STEP}/')


def run(args):
    import time
    from pathlib import Path

    from src.accuracy.forecasts import HORIZONS
    t0 = time.perf_counter()
    out = Path(args.output) if args.output else out_dir(STEP)
    out.mkdir(parents=True, exist_ok=True)
    tables = build()
    for key, t in zip(FILES, tables):
        t.to_parquet(out / FILES[key], index=False)
        print(f'{FILES[key]}：{len(t)} 行、{t.shape[1]} 列')
    R, _, P = tables
    print(f'研报 {len(R)} 篇：匹配到本篇预测 {int(R.matched.sum())} 篇，基础样本 {int(R.base.sum())} 篇；公司-券商对 {len(P)} 个')
    for h in HORIZONS:
        ev = P[f'EVAL_LAST_TY_H{h}'].dropna()
        print(f'  H{h}：研报 PMAFE_H{h} 非缺失 {int(R[f"PMAFE_H{h}"].notna().sum())} 篇；公司-券商对 PMAFE_H{h} 非缺失 '
              f'{int(P[f"PMAFE_H{h}"].notna().sum())} 个，有被评价预测 {len(ev)} 个，其中被评价预测不是本券商当年对目标年度的'
              f'最后一条 {int((~ev.astype(bool)).sum())} 个')
    print(f'输出在 {out}，耗时 {time.perf_counter() - t0:.1f} 秒')
