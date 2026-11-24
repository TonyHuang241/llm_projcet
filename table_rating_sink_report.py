"""v11：研报层面回归作为主回归，控制变量加“同行平均时滞”PEERLAG。复用 table_rating_sink_pair（v10）的样本、X、因变量和回归函数。

观测    研报（基础样本 base；回归样本 = F.y_sample 后去缺失、反复剔除固定效应单例，R.fit）。
Y       PMAFE_FY25 / PMAFE_FY24 = v10 研报层面的 PMAFE0 口径（T.pmafe_v5：其他券商在 [d−90, d] 内对同公司同目标年度的最新一条预测，
        ≥ 3 家，MAFE 含本篇；基础样本内 1%/99% 缩尾）；SCORE_FY25 = 100 × (1 − RANK_FY25)；rwin30 的 PMAFE_FY25 = [d−90, d+30] 口径。
        全部取自 P.report_level13（与 v10 相同的缓存）。
PEERLAG 比较组内各同行 (d − peer_dt) 的均值（天）。同行 = T.pick_peers（与 T.pmafe_v5 同一函数）选出的同行，比较组不足 3 家记缺失。
        PEERLAG25：FY2025、[d−90, d]，用于 PMAFE_FY25、SCORE_FY25；PEERLAG24：FY2024、[d−90, d]，用于 PMAFE_FY24；
        PEERLAGW30：FY2025、[d−90, d+30]（d − peer_dt 可为负），用于 rwin30。另存 LAGMED25（中位数）、N_PEERS25/24/W30、
        LNNPEERS25 = ln(N_PEERS25)。这些时滞控制变量与六个控制变量的处理相同：基础样本内 1%/99% 缩尾（R.winsor，同 F.winsorize），
        不标准化（R.fit 只标准化 X）。
X       13 个（P.XS_13）及 RATE_LNSRC、ALL_LNSRC，原始值取自 P.new13_measures，基础样本内 1%/99% 缩尾（P.report_level13），回归样本内标准化。
控制    六个（T.CTRL，研报日，MOM/VOL/ROA_T × 100）+ 时滞控制变量；固定效应 T.PANS 的 N/A/B/C；按公司聚类。
块      rmain、rladder、rref、rjoint、rnolag（= v10 rladder 的 PMAFE_FY25、SCORE_FY25 部分，不放 PEERLAG）、rmonth（+ 发布月份固定效应）、
        rlagalt（spec 末尾 |LAGMED25：LAGMED25 代替 PEERLAG；|PEERLAG25+LNNPEERS25：再加 ln 同行数）、rlogsrc、rwin30、
        rex（剔除 000858 五粮液的 rmain B/C 和 rref PMAFE_FY25）。reg_all.csv 的 role：x / control（六个控制变量）/ peerlag（时滞类控制变量）。
输出    output_dag/forecast_newdata/rating_table10_v11/：reg_all.csv、report_sample.parquet、descriptives.csv、corr.csv、checks.txt
用法    C:/Users/hung/projects/.venv/Scripts/python.exe -X utf8 table_rating_sink_report.py [--ver v11|v12] [--jobs 16] [--out 目录]

v12（--ver v12）：X 换成 6 个（P.XS_RT6：RT_NSRC、RT_LNPATH、RT_PATHLEN 为“投资评级”严格汇点子图的源点数、ln(1+路径数)、平均路径长度，
        非严格汇点缺失；WG_* 为整张图的同名指标；P.rt6_measures = D.five_measures 的原始值），另存 RT_LNSRC、WG_LNSRC = ln(源点数)；
        X 在基础样本内 1%/99% 缩尾（R.winsor，同 v11），回归样本内标准化。研报层面其余设定同 v11；块：rmain、rladder、rref（RT 与对应 WG
        成对）、rjoint（rt3 / wg3 / joint6）、rsame（WG 三个单独放入，样本限 RT_NSRC 非缺失）、rnolag、rmonth、rlagalt、rlogsrc、rwin30、
        rex（剔除 000858 的 rmain B/C、rref 与 joint6 的 PMAFE_FY25）。公司-券商层面 = P.main(agg='last', xset='rt6')（v10 pair 层设定，
        块 ladder / ref / joint），输出在 rating_table10_v12/pair/，其 reg_all 并入 v12 reg_all.csv（level = pair）。
v13（--ver v13）：同 v12，固定效应换成 P.PANS13（N 无、A 行业、B 公司、D 分析师、C 公司 + 分析师；analyst = broker_g|EST_COMP_TN AUTHOR
        第一作者，P.analyst_keys），单例按各固定效应维度迭代剔除；不跑 rjoint；rteam 用团队口径（CT 公司 + 团队、DT 只有团队）；
        rbroker3（公司 + 券商 + 分析师）只在 checks 里与 C 比较。公司-券商层面 = P.main(agg='last', xset='rt6a')（块 ladder / ref）。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stat_peer_lag as L  # noqa: E402
import table_rating_sink as T  # noqa: E402
import table_rating_sink_pair as P  # noqa: E402
from src import dag_forecast as F  # noqa: E402
from src import dag_outcomes_reg as R  # noqa: E402

OUT = P.ND / 'rating_table10_v11'
OUTS = {'v11': OUT, 'v12': P.ND / 'rating_table10_v12', 'v13': P.ND / 'rating_table10_v13'}
P13 = P.PANS13  # v13：N/A/B 同前，D = 分析师，C = 公司 + 分析师
PTEAM = {'CT': (['code_ty', 'team'], []), 'DT': (['team', 'ty'], [])}  # rteam：公司 + 团队、只有团队
V10 = P.OUT_DEFAULT[('new13', 'last')]
X6, X6V = P.XS_RT6, P.XS_RT6V
XS, XV, CTRL, PANS = P.XS_13, P.XS_13V, T.CTRL, T.PANS
LAGCOL = {'PMAFE_FY25': 'PEERLAG25', 'SCORE_FY25': 'PEERLAG25', 'PMAFE_FY24': 'PEERLAG24'}
LAG_W = ['PEERLAG25', 'PEERLAG24', 'LAGMED25', 'LNNPEERS25', 'PEERLAGW30']  # 缩尾的时滞控制变量
WINDOWS = (('25', 2025, 90, 0), ('24', 2024, 90, 0), ('W30', 2025, 90, 30))
X858 = T.X858
log = T.log


def inputs():
    """与 P.main（--x new13 --agg last）相同的缓存和函数：面板 d2、预测长表 lg、13 个指标 nm。"""
    def _sample():
        d2_, s_, _ = T.build_sample_v2()
        return d2_, s_, F.build_panel(main_only=False)[0]
    (d2, _, _), h1 = P.cached('sample_v2', {}, _sample, 3)
    codes = sorted(set(d2.code))
    (_, lg), h2 = P.cached('est_long', {'years': [2018, 2025], 'codes': codes}, lambda: (lambda e: (e, T.long_v5(e)))(
        T.load_est_eps(codes, years=range(2018, 2026))), 2)
    (nm,), h3 = P.cached('new_measures', {'n': len(d2), 'res': [P.EARN_RE, P.VALX_RE, P.EXT_RE]},
                         lambda: (P.new_report_measures(list(d2.report_id)),), 1)
    nm = P.new13_measures(nm)
    v10 = pd.read_parquet(V10 / 'new_report_measures.parquet')
    same = v10.set_index('report_id').loc[nm.report_id, XS + XV].reset_index(drop=True).equals(nm[XS + XV].reset_index(drop=True))
    log(f'缓存：面板 {"命中" if h1 else "重算"}、预测长表 {"命中" if h2 else "重算"}、指标 {"命中" if h3 else "重算"}；'
        f'15 个 X 原始值与 v10 new_report_measures.parquet 逐篇相同：{same}')
    return d2, lg, nm


def lag_frame(d2, lg, Rr):
    """基础样本每篇研报在三个比较组上的同行数、时滞均值（和 FY25 的中位数），并逐篇核对同行数、重算 PMAFE。"""
    rep = d2[d2.base][['report_id', 'code', 'ORG_CL', 'date', 'ABSERR25', 'ABSERR24', 'A25', 'A24']].reset_index(drop=True)
    out = rep[['report_id']].copy()
    ref = Rr.set_index('report_id')
    for tag, ty, back, fwd in WINDOWS:
        pl = L.peer_lags(rep, lg, ty, back, fwd)
        g = pl.groupby('report_id').LAG
        n = rep.report_id.map(g.size()).fillna(0).astype(int)
        ok = n >= F.MIN_PEERS
        out[f'N_PEERS{tag}'] = n
        out[f'PEERLAG{tag}'] = rep.report_id.map(g.mean()).where(ok)
        if tag == '25':
            out['LAGMED25'] = rep.report_id.map(g.median()).where(ok)
            out['LNNPEERS25'] = np.log(n.where(ok).astype(float))
        # 核对：重算 PMAFE（MAFE 含本篇）与 report_level13 的缩尾前值
        yy = {'25': 'PMAFE0_FY25_RAW', '24': 'PMAFE0_FY24_RAW', 'W30': 'PMAFEW30_FY25_RAW'}[tag]
        own, a = rep[f'ABSERR{ty % 100}'].to_numpy(float), rep[f'A{ty % 100}'].to_numpy(float)
        ae = (pl.F - pl.report_id.map(dict(zip(rep.report_id, a)))).abs().groupby(pl.report_id).sum()
        mafe = (rep.report_id.map(ae).fillna(0).to_numpy() + own) / (n.to_numpy() + 1)
        re_ = np.where(ok & np.isfinite(own) & np.isfinite(a) & (mafe > 0), (own - mafe) / mafe, np.nan)
        y0 = ref.loc[rep.report_id, yy].to_numpy(float)
        bad = int((~(np.isclose(re_, y0, rtol=1e-12, atol=1e-12) | (np.isnan(re_) & np.isnan(y0)))).sum())
        msg = (f'比较组 FY{ty} [d−{back}, d+{fwd}]：基础样本 {len(rep)} 篇，同行 ≥ 3 的 {int(ok.sum())} 篇；同行明细 {len(pl)} 行，'
               f'LAG 范围 [{pl.LAG.min()}, {pl.LAG.max()}]，同一研报同一券商重复 {int(pl.duplicated(["report_id", "ORG_CL"]).sum())} 行；'
               f'重算 PMAFE 与 {yy} 不一致（含缺失与否）{bad} 篇')
        if tag != 'W30':
            ng = ref.loc[rep.report_id, f'N_GRP0_FY{tag}'].to_numpy()
            has = ng > 0
            msg += f'；N_GRP0_FY{tag} > 0 的 {int(has.sum())} 篇中同行数 + 1 ≠ N_GRP0 的 {int((n.to_numpy()[has] + 1 != ng[has]).sum())} 篇'
        log(msg)
        if tag == '25':  # 与 stat_peer_lag 已存的明细（709 篇）逐条比较
            old = pd.read_parquet(L.OUT / 'peer_lags.parquet')
            j = old.merge(pl, on=['report_id', 'ORG_CL'], how='left', suffixes=('', '_new'))
            log(f'  与 rating_table10_v10_timing/peer_lags.parquet（{len(old)} 行）逐条比较：同行和日期不一致 '
                f'{int((j.peer_dt != j.peer_dt_new).sum() + (j.LAG != j.LAG_new).sum())} 行；新明细里这 709 篇的行数 '
                f'{int(pl.report_id.isin(old.report_id).sum())}')
    return out


def build_data(d2, lg, nm, extra=None):
    """extra：另加的 X 原始值（report_id + 列），并入后存为 *_RAW，并在基础样本内 1%/99% 缩尾（同 P.report_level13 对 X 的处理）。"""
    Rr, _ = P.report_level13(d2, lg, nm)
    lags = lag_frame(d2, lg, Rr)
    D = Rr.merge(d2[['report_id', 'ORG_CL', 'date']], on='report_id', how='left', validate='1:1').merge(
        lags, on='report_id', how='left', validate='1:1')
    D['month'] = D.date.str[:4] + '-' + D.date.str[4:6]
    b = D.base.astype(bool)
    if extra is not None:
        xw = [c for c in extra.columns if c != 'report_id']
        D = D.merge(extra.rename(columns={x: x + '_RAW' for x in xw}), on='report_id', how='left', validate='1:1')
        for x in xw:
            D[x] = np.nan
            D.loc[b, x] = R.winsor(D.loc[b, x + '_RAW'])
        log('另加 X（基础样本内 1%/99% 缩尾）：' + '；'.join(f'{x} 非缺失 {int(D.loc[b, x].notna().sum())} 篇' for x in xw))
    for c in LAG_W:
        D[c + '_RAW'] = D[c]
        D[c] = np.nan
        D.loc[b, c] = R.winsor(D.loc[b, c + '_RAW'])
    log('时滞控制变量：与六个控制变量相同，在基础样本内按 1%/99% 缩尾（R.winsor，同 F.winsorize），不标准化；缩尾界 '
        + '；'.join(f'{c} [{D.loc[b, c].min():.3f}, {D.loc[b, c].max():.3f}]（原始 [{D.loc[b, c + "_RAW"].min():.3f}, '
                   f'{D.loc[b, c + "_RAW"].max():.3f}]）' for c in LAG_W))
    for y, lc in LAGCOL.items():
        m = b & D[y].notna()
        log(f'  {y} 非缺失 {int(m.sum())} 篇，其中 {lc} 缺失 {int(D.loc[m, lc].isna().sum())} 篇')
    m = b & D.PMAFEW30_FY25.notna()
    log(f'  rwin30 的 PMAFE_FY25 非缺失 {int(m.sum())} 篇，其中 PEERLAGW30 缺失 {int(D.loc[m, "PEERLAGW30"].isna().sum())} 篇')
    return D


# ---------------------------------------------------------------- 回归
def _adder(tasks, tags):
    """往 tasks / tags 里加任务的函数：每个（块，Y，面板）一个 P.run_tasks 任务；控制 = CTRL + lag(y)；tag 会接到 spec 末尾。"""
    def add(block, data, ys, pans, xs_, lag=lambda y: [LAGCOL[y]], joint=False, spec_name=None, tag=None):
        for y in ys:
            ctrl = CTRL + lag(y)
            fe_cols = [c for fe, _ in pans.values() for c in fe if c in data.columns and c not in ('sw', 'code', 'broker_g')]
            cols_ = list(dict.fromkeys(['report_id', 'code', 'base', 'broker_g', 'sw', 'TY25', 'TY24', 'LNHORIZON25', 'LNHORIZON24', y]
                                       + fe_cols + list(xs_.values()) + ctrl))
            for p, fe_ctrl in pans.items():
                tasks.append((block, spec_name, (data[cols_], xs_, y, {p: fe_ctrl}, ctrl, joint)))
                tags.append(tag)
    return add


def make_tasks_v12(D):
    tasks, tags = [], []
    add = _adder(tasks, tags)
    D30 = D.assign(PMAFE_FY25=D.PMAFEW30_FY25)
    Dex = D[D.code != X858]
    pbc = {p: PANS[p] for p in ('B', 'C')}
    pmon = {p: (PANS[p][0] + ['month'], PANS[p][1]) for p in ('B', 'C')}
    xs = {x: x for x in X6}
    pairs = {'+'.join(c): {x: x for x in c} for c in P.RT6_PAIRS}
    joints = {'rt3': {x: x for x in X6[:3]}, 'wg3': {x: x for x in X6[3:]}, 'joint6': xs}
    y2 = ['PMAFE_FY25', 'SCORE_FY25']
    add('rmain', D, ['PMAFE_FY25'], {'C': PANS['C']}, xs)
    add('rladder', D, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'], PANS, xs)
    for name, c in pairs.items():
        add('rref', D, y2, pbc, c, joint=True, spec_name=name)
    for name, c in joints.items():
        add('rjoint', D, y2, pbc, c, joint=True, spec_name=name)
    add('rsame', D[D.RT_NSRC_RAW.notna()], y2, pbc, {x: x for x in X6[3:]})
    add('rnolag', D, y2, PANS, xs, lag=lambda y: [])
    add('rmonth', D, y2, pmon, xs)
    add('rlagalt', D, ['PMAFE_FY25'], pbc, xs, lag=lambda y: ['LAGMED25'], tag='LAGMED25')
    add('rlagalt', D, ['PMAFE_FY25'], pbc, xs, lag=lambda y: ['PEERLAG25', 'LNNPEERS25'], tag='PEERLAG25+LNNPEERS25')
    add('rlogsrc', D, ['PMAFE_FY25'], pbc, {x: x for x in X6V})
    add('rlogsrc', D, ['PMAFE_FY25'], pbc, {x: x for x in X6V}, joint=True, spec_name='+'.join(X6V))
    add('rwin30', D30, ['PMAFE_FY25'], pbc, xs, lag=lambda y: ['PEERLAGW30'])
    add('rex', Dex, ['PMAFE_FY25'], pbc, xs)
    for name, c in list(pairs.items()) + [('joint6', xs)]:
        add('rex', Dex, ['PMAFE_FY25'], pbc, c, joint=True, spec_name=name)
    return tasks, tags


def make_tasks_v13(D):
    """v13：同 v12 的研报层面，固定效应换成 P13（N/A/B/D/C），不跑 rjoint，另加 rteam（PTEAM）。"""
    tasks, tags = [], []
    add = _adder(tasks, tags)
    D30 = D.assign(PMAFE_FY25=D.PMAFEW30_FY25)
    Dex = D[D.code != X858]
    sub = lambda *ps: {p: P13[p] for p in ps}  # noqa: E731
    pdc, pbdc = sub('D', 'C'), sub('B', 'D', 'C')
    pmon = {p: (P13[p][0] + ['month'], P13[p][1]) for p in ('D', 'C')}
    xs = {x: x for x in X6}
    pairs = {'+'.join(c): {x: x for x in c} for c in P.RT6_PAIRS}
    y2 = ['PMAFE_FY25', 'SCORE_FY25']
    add('rmain', D, ['PMAFE_FY25'], sub('C'), xs)
    add('rladder', D, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'], P13, xs)
    for name, c in pairs.items():
        add('rref', D, y2, pbdc, c, joint=True, spec_name=name)
    add('rsame', D[D.RT_NSRC_RAW.notna()], y2, pdc, {x: x for x in X6[3:]})
    add('rnolag', D, y2, P13, xs, lag=lambda y: [])
    add('rmonth', D, y2, pmon, xs)
    add('rlagalt', D, ['PMAFE_FY25'], pdc, xs, lag=lambda y: ['LAGMED25'], tag='LAGMED25')
    add('rlagalt', D, ['PMAFE_FY25'], pdc, xs, lag=lambda y: ['PEERLAG25', 'LNNPEERS25'], tag='PEERLAG25+LNNPEERS25')
    add('rlogsrc', D, ['PMAFE_FY25'], pdc, {x: x for x in X6V})
    add('rlogsrc', D, ['PMAFE_FY25'], pdc, {x: x for x in X6V}, joint=True, spec_name='+'.join(X6V))
    add('rwin30', D30, ['PMAFE_FY25'], pdc, xs, lag=lambda y: ['PEERLAGW30'])
    add('rex', Dex, ['PMAFE_FY25'], pdc, xs)
    for name, c in pairs.items():
        add('rex', Dex, ['PMAFE_FY25'], pbdc, c, joint=True, spec_name=name)
    add('rteam', D, y2, PTEAM, xs)
    return tasks, tags


def v13_checks(D, res):
    """v13 的样本与分析师分布、与 v12 相同面板的逐格比较、rbroker3（公司 + 券商 + 分析师）与 C 的比较。"""
    b = D[D.base.astype(bool)]
    vc = b.analyst.value_counts()
    log(f'基础样本 {len(b)} 篇：AUTHOR 缺失 {int(b.AUTHOR.isna().sum())} 篇；第一作者（analyst）{vc.size} 位：1 篇的 {int((vc == 1).sum())} 位（{int(vc[vc == 1].sum())} 篇）、'
        f'2 篇的 {int((vc == 2).sum())} 位（{int(vc[vc == 2].sum())} 篇）、≥3 篇的 {int((vc >= 3).sum())} 位（{int(vc[vc >= 3].sum())} 篇），最多 {vc.max()} 篇；'
        f'团队 {b.team.nunique()} 个；同一研报 AUTHOR 串不止一个的 {int((b.n_author > 1).sum())} 篇')
    fc = b.groupby('analyst').code.nunique().value_counts().sort_index()
    log('  基础样本每位分析师覆盖的公司数：' + '，'.join(f'{k} 家 {v} 位' for k, v in fc.items()))
    ys = F.y_sample(D, 'PMAFE_FY25')
    for x in ('RT_NSRC', 'WG_NSRC'):
        for p in ('N', 'A', 'B', 'D', 'C'):
            fe = P13[p][0]
            pre = ys[['PMAFE_FY25', x, 'code', 'analyst'] + CTRL + ['PEERLAG25'] + [c for c in fe if c not in ('code', 'analyst')]].dropna()
            _, smp = R.fit(ys, 'PMAFE_FY25', [x], CTRL + ['PEERLAG25'], fe, 'code')
            S = ys.loc[smp.index]
            log(f'  PMAFE_FY25 / {x} / 面板 {p}（{"+".join(fe) or "无"}）：去缺失后 {len(pre)} 篇，剔除单例 {len(pre) - len(S)} 篇，估计样本 {len(S)} 篇、'
                f'{S.code.nunique()} 家公司、{S.analyst.nunique()} 位分析师')
            if p == 'C':
                both = lambda d: int(((d.groupby('code').code.transform('size') > 1) & (d.groupby('analyst').analyst.transform('size') > 1)).sum())  # noqa: E731
                fcs = S.groupby('analyst').code.nunique().value_counts().sort_index()
                log(f'    C 档：既有同公司其他研报、又有同分析师其他研报的研报 估计样本 {both(S)} / {len(S)} 篇（单例剔除后按构造全部满足），'
                    f'剔除单例前 {both(pre)} / {len(pre)} 篇；估计样本中每位分析师覆盖的公司数：' + '，'.join(f'{k} 家 {v} 位' for k, v in fcs.items()))
    # 与 v12 相同面板（N/A/B）逐格比较：rladder、rnolag
    v12 = pd.read_csv(OUTS['v12'] / 'reg_all.csv')
    num = ['coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons', 'adj_r2', 'r2', 'k']
    for blk in ('rladder', 'rnolag'):
        a = v12[(v12.level == 'report') & (v12.block == blk) & v12.panel.isin(['N', 'A', 'B'])]
        c = res[(res.block == blk) & res.panel.isin(['N', 'A', 'B'])]
        j = a.merge(c, on=['Y', 'panel', 'X', 'term'], how='outer', suffixes=('_v12', ''), indicator=True)
        bj = j[j._merge == 'both']
        mx = max(float(np.nanmax(np.abs(bj[k].to_numpy(float) - bj[k + '_v12'].to_numpy(float)))) for k in num)
        log(f'{blk} 面板 N/A/B 与 v12 同块逐行比较：v12 {len(a)} 行、v13 {len(c)} 行，只在一边的 {int((j._merge != "both").sum())} 行，各数值列最大绝对差 {mx:.1e}')
    # rbroker3：公司 + 券商 + 分析师，应与 C 相同（分析师嵌套在券商内）
    pb3 = {'CB': (['code_ty', 'broker_g', 'analyst'], [])}
    r3 = F.regress_each(D, {x: x for x in X6}, ['PMAFE_FY25'], pans=pb3, controls=CTRL + ['PEERLAG25'], terms=True)
    rc = res[(res.block == 'rladder') & (res.Y == 'PMAFE_FY25') & (res.panel == 'C')]
    j = (r3.assign(key=r3.spec + '|' + r3.X)[['key', 'coef', 'se', 'n']]
         .merge(rc.assign(key=rc.X + '|' + rc.term)[['key', 'coef', 'se', 'n']], on='key', suffixes=('_b3', '')))
    log(f'rbroker3（公司 + 券商 + 分析师，PMAFE_FY25，6 个 X 单独放入）与 rladder 面板 C：{len(j)} / {len(r3)} 项对上，系数最大绝对差 '
        f'{float(np.max(np.abs(j.coef_b3 - j.coef))):.1e}，se 最大绝对差 {float(np.max(np.abs(j.se_b3 - j.se))):.1e}，N 不同的项 {int((j.n_b3 != j.n).sum())}（不输出）')


def make_tasks(D):
    tasks, tags = [], []
    D30 = D.assign(PMAFE_FY25=D.PMAFEW30_FY25)
    Dex = D[D.code != X858]
    pbc = {p: PANS[p] for p in ('B', 'C')}
    pmon = {p: (PANS[p][0] + ['month'], PANS[p][1]) for p in ('B', 'C')}
    xs = {x: x for x in XS}
    add = _adder(tasks, tags)
    combos = {'+'.join(c): {x: x for x in c} for c in P.REF_COMBOS}
    add('rmain', D, ['PMAFE_FY25'], {'C': PANS['C']}, xs)
    add('rladder', D, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'], PANS, xs)
    for name, c in combos.items():
        add('rref', D, ['PMAFE_FY25', 'SCORE_FY25'], pbc, c, joint=True, spec_name=name)
    add('rjoint', D, ['PMAFE_FY25', 'SCORE_FY25'], pbc, xs, joint=True, spec_name='joint13')
    add('rnolag', D, ['PMAFE_FY25', 'SCORE_FY25'], PANS, xs, lag=lambda y: [])
    add('rmonth', D, ['PMAFE_FY25', 'SCORE_FY25'], pmon, xs)
    add('rlagalt', D, ['PMAFE_FY25'], pbc, xs, lag=lambda y: ['LAGMED25'], tag='LAGMED25')
    add('rlagalt', D, ['PMAFE_FY25'], pbc, xs, lag=lambda y: ['PEERLAG25', 'LNNPEERS25'], tag='PEERLAG25+LNNPEERS25')
    add('rlogsrc', D, ['PMAFE_FY25'], pbc, {x: x for x in XV})
    add('rlogsrc', D, ['PMAFE_FY25'], pbc, {x: x for x in XV + ['ALL_LNNODE']}, joint=True, spec_name='+'.join(XV + ['ALL_LNNODE']))
    add('rwin30', D30, ['PMAFE_FY25'], pbc, xs, lag=lambda y: ['PEERLAGW30'])
    add('rex', Dex, ['PMAFE_FY25'], pbc, xs)
    for name, c in combos.items():
        add('rex', Dex, ['PMAFE_FY25'], pbc, c, joint=True, spec_name=name)
    return tasks, tags


def check_rnolag(res):
    v10 = pd.read_csv(V10 / 'reg_all.csv')
    a = v10[(v10.level == 'report') & (v10.block == 'rladder') & v10.Y.isin(['PMAFE_FY25', 'SCORE_FY25'])]
    b = res[res.block == 'rnolag']
    key = ['Y', 'panel', 'X', 'term']
    j = a.merge(b, on=key, how='outer', suffixes=('_v10', ''), indicator=True)
    num = ['coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons', 'adj_r2', 'r2', 'k']
    both = j[j._merge == 'both']
    diff = np.zeros(len(both), dtype=bool)
    for c in num:
        diff |= ~np.isclose(both[c].to_numpy(float), both[c + '_v10'].to_numpy(float), rtol=1e-10, atol=1e-12)
    cells = both.assign(d=diff).groupby(['Y', 'panel', 'X']).d.any()
    mx = max(float(np.nanmax(np.abs(both[c].to_numpy(float) - both[c + '_v10'].to_numpy(float)))) for c in num)
    log(f'rnolag vs v10 report 层 rladder（PMAFE_FY25、SCORE_FY25）：v10 {a.groupby(key[:3]).ngroups} 格 / {len(a)} 行，'
        f'rnolag {b.groupby(key[:3]).ngroups} 格 / {len(b)} 行；只在一边的行 {int((j._merge != "both").sum())}；'
        f'数值不一致（rtol 1e-10）的格 {int(cells.sum())}；各数值列最大绝对差 {mx:.1e}')


def sig_counts(res, by_level=False):
    lv = ['level'] if by_level else []
    q = res[res.role == 'x'].assign(firm_fe=lambda t: np.where(t.panel.isin(['B', 'C', 'CT']), 'B/C（含公司）', 'N/A（不含公司）'))  # v13：D 不含公司，CT = 公司 + 团队
    g = q.groupby(lv + ['block', 'firm_fe']).agg(cells=('p', 'size'), p10=('p', lambda p: int((p < .1).sum())), p05=('p', lambda p: int((p < .05).sum())))
    tot = q.groupby(lv + ['firm_fe']).agg(cells=('p', 'size'), p10=('p', lambda p: int((p < .1).sum())), p05=('p', lambda p: int((p < .05).sum())))
    log('显著格计数（role = x 的系数格，p < 0.10 / p < 0.05）：\n' + g.to_string() + '\n合计：\n' + tot.to_string())


def check_v12_vs_v11(D, res):
    """v12：RT_NSRC = RATE_NSRC、WG_NSRC = ALL_NSRC（原始值和缩尾值逐篇），rmain 中两者的各项结果与 v11 rmain 相同。"""
    b = D.base.astype(bool)
    for a, c in (('RT_NSRC', 'RATE_NSRC'), ('WG_NSRC', 'ALL_NSRC')):
        for suf, m in (('_RAW', slice(None)), ('', b)):
            x, y = D.loc[m, a + suf].to_numpy(float), D.loc[m, c + suf].to_numpy(float)
            same = bool(((x == y) | (np.isnan(x) & np.isnan(y))).all())
            log(f'{a}{suf} 与 v11 的 {c}{suf} 逐篇相同（含缺失位置，{"全部 791 篇" if suf else "基础样本"}）：{same}')
    v11 = pd.read_csv(OUT / 'reg_all.csv')
    num = ['coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons', 'adj_r2', 'r2', 'k']
    for a, c in (('RT_NSRC', 'RATE_NSRC'), ('WG_NSRC', 'ALL_NSRC')):
        x = res[(res.block == 'rmain') & (res.X == a)].assign(term=lambda t: t.term.replace(a, c)).set_index('term')
        y = v11[(v11.block == 'rmain') & (v11.X == c)].set_index('term').loc[x.index]
        mx = max(float(np.max(np.abs(x[k].to_numpy(float) - y[k].to_numpy(float)))) for k in num)
        log(f'rmain：v12 {a} 一列（{len(x)} 项）与 v11 rmain {c} 一列的各数值列最大绝对差 {mx:.1e}')


def describe(D, xmain='ALL_NSRC', xs=XS, x6=None, fe=None):
    ys = F.y_sample(D, 'PMAFE_FY25')
    _, smp = R.fit(ys, 'PMAFE_FY25', [xmain], CTRL + ['PEERLAG25'], fe or PANS['C'][0], 'code')
    S = ys.loc[smp.index]
    log(f'主回归估计样本（rmain，面板 C，X = {xmain}，控制 + PEERLAG25）：{len(S)} 篇、{S.code.nunique()} 家')
    vars_ = (['PMAFE_FY25', 'PMAFE0_FY25_RAW', 'PMAFE_FY24', 'SCORE_FY25'] + [x + '_RAW' for x in xs] + CTRL
             + ['PEERLAG25', 'PEERLAG25_RAW', 'N_PEERS25', 'LAGMED25_RAW', 'PEERLAG24_RAW'])
    rows = []
    for v in vars_:
        z = S[v].dropna().astype(float)
        rows.append({'var': v, 'N': len(z), 'mean': z.mean(), 'sd': z.std(), 'p10': z.quantile(.1), 'p50': z.median(), 'p90': z.quantile(.9),
                     'min': z.min(), 'max': z.max()})
    desc = pd.DataFrame(rows)
    cv = ['PMAFE_FY25', 'SCORE_FY25', 'PEERLAG25', 'LAGMED25', 'N_PEERS25'] + xs + CTRL
    cs = []
    if x6:  # 六个 X（缩尾后，两两非缺失）的相关矩阵单独列出
        for meth in ('pearson', 'spearman'):
            m_ = S[x6].corr() if meth == 'pearson' else S[x6].rank().corr()
            m_.insert(0, 'var', m_.index)
            m_.insert(0, 'method', meth)
            m_.insert(0, 'table', 'x6')
            cs.append(m_)
        log('六个 X 的相关（主估计样本，缩尾后，Pearson，两两非缺失）：\n' + S[x6].corr().round(3).to_string())
    for meth in ('pearson', 'spearman'):
        m_ = S[cv].corr() if meth == 'pearson' else S[cv].rank().corr()
        m_.insert(0, 'var', m_.index)
        m_.insert(0, 'method', meth)
        if x6:
            m_.insert(0, 'table', 'all')
        cs.append(m_)
    b = D[D.base.astype(bool)]
    for c in ('PEERLAG25_RAW', 'PEERLAG24_RAW', 'PEERLAGW30_RAW', 'LAGMED25_RAW', 'N_PEERS25'):
        z = b[c] if c != 'N_PEERS25' else b.N_PEERS25.where(b.N_PEERS25 >= F.MIN_PEERS)
        z = z.dropna()
        log(f'  {c}（基础样本，比较组 ≥ 3 家）：{len(z)} 篇，均值 {z.mean():.3f}、标准差 {z.std():.3f}、p1/p10/p50/p90/p99 = '
            + '/'.join(f'{v:.2f}' for v in z.quantile([.01, .1, .5, .9, .99])) + f'、最小 {z.min():.2f}、最大 {z.max():.2f}')
    return desc, pd.concat(cs, ignore_index=True), S


def main(jobs=16, ver='v11', out=None):
    v13 = ver == 'v13'
    v12 = ver in ('v12', 'v13')  # 以下 v12 指“6 个 X 且含公司-券商层面”，v13 也走这些分支
    xset_pair = 'rt6a' if v13 else 'rt6'
    OUT_ = Path(out) if out else OUTS[ver]
    OUT_.mkdir(parents=True, exist_ok=True)
    pair_log = []
    if v12:  # 公司-券商层面：P.main 会清空 T.log_lines，先跑，日志另存
        P.main('last', str(OUT_ / 'pair'), xset_pair, jobs)
        pair_log = list(T.log_lines)
    T.log_lines.clear()
    log('（v13）研报层面：同 v12，券商固定效应换成分析师（D = 分析师，C = 公司 + 分析师），另加 rteam；以下为构造和核对' if v13 else
        '（v12）研报层面：X = 6 个（RT_/WG_ 的源点数、ln(1+路径数)、平均路径长度），其余同 v11；以下为构造和核对' if v12 else
        '（v11）研报层面主回归 + 同行平均时滞 PEERLAG；以下为构造和核对')
    d2, lg, nm = inputs()
    D = build_data(d2, lg, nm, P.rt6_measures(list(d2.report_id)) if v12 else None)
    if v13:
        D = D.merge(P.analyst_keys(D[['report_id', 'code', 'broker_g']]), on='report_id', how='left', validate='1:1')
    tasks, tags = make_tasks_v13(D) if v13 else make_tasks_v12(D) if v12 else make_tasks(D)
    res = P.run_tasks(tasks, jobs)
    for i, tg in enumerate(tags):
        if tg and len(res[i]):
            res[i] = res[i].assign(spec=res[i].spec + '|' + tg)
    res = pd.concat(res, ignore_index=True)[T.COLS]
    res.loc[res.term.isin(LAG_W), 'role'] = 'peerlag'
    res.insert(0, 'level', 'report')
    assert not res.duplicated(['spec', 'term']).any()
    log(f'{"研报层面" if v12 else "reg_all.csv"}：{len(res)} 行；任务 {len(tasks)} 个；各块回归数 {res.groupby("block").spec.nunique().to_dict()}')
    for (blk, y, p), g in res[res.role == 'x'].groupby(['block', 'Y', 'panel'], sort=False):
        log(f'  N/公司数/单例 {blk} {y} {p}：' + '；'.join(sorted({f'{int(r.n)}/{int(r.n_clusters)}/{int(r.n_singletons)}' for r in g.itertuples()})))
    if v12:
        if v13:
            v13_checks(D, res)
        else:
            check_v12_vs_v11(D, res)
        pr = pd.read_csv(OUT_ / 'pair' / 'reg_all.csv')
        pr.insert(0, 'level', 'pair')
        log(f'公司-券商层面（P.main agg=last xset={xset_pair}，输出在 pair/）：{len(pr)} 行；各块回归数 {pr.groupby("block").spec.nunique().to_dict()}')
        res = pd.concat([res, pr], ignore_index=True)
        assert not res.duplicated(['level', 'spec', 'term']).any()
        pd.read_parquet(OUT_ / 'pair' / 'pair_sample.parquet').to_parquet(OUT_ / 'pair_sample.parquet', index=False)
        res.to_csv(OUT_ / 'reg_all.csv', index=False, encoding='utf-8-sig')
        log(f'reg_all.csv：{len(res)} 行（report {int((res.level == "report").sum())}、pair {int((res.level == "pair").sum())}）')
        sig_counts(res, by_level=True)
    else:
        res.to_csv(OUT_ / 'reg_all.csv', index=False, encoding='utf-8-sig')
        check_rnolag(res)
        sig_counts(res)

    # 自检：独立 OLS（T.own_ols）复算两格
    sc = ((('rmain', D, 'PMAFE_FY25', 'C', ['RT_NSRC'], CTRL + ['PEERLAG25']), ('rladder', D, 'PMAFE_FY24', 'B', ['WG_PATHLEN'], CTRL + ['PEERLAG24']))
          if v12 else (('rmain', D, 'PMAFE_FY25', 'C', ['RATE_NSRC'], CTRL + ['PEERLAG25']),
                       ('rladder', D, 'PMAFE_FY24', 'B', ['ALL_LNNODE'], CTRL + ['PEERLAG24'])))
    if v13:
        sc = (('rmain', D, 'PMAFE_FY25', 'C', ['RT_NSRC'], CTRL + ['PEERLAG25']), ('rladder', D, 'SCORE_FY25', 'D', ['WG_PATHLEN'], CTRL + ['PEERLAG25']))
    for blk, data, y, pan, xx, ctrl in sc:
        o_ = T.own_ols(F.y_sample(data, y), y, xx, ctrl, (P13 if v13 else PANS)[pan][0])
        r_ = res[(res.level == 'report') & (res.block == blk) & (res.Y == y) & (res.panel == pan) & (res.X == xx[0])].set_index('term').loc[xx + ctrl]
        dc = np.max(np.abs(r_.coef.to_numpy() - o_['coef']) / np.maximum(np.abs(o_['coef']), 1e-8))
        ds = np.max(np.abs(r_.se.to_numpy() - o_['se']) / o_['se'])
        log(f'  独立 OLS [{blk} {y} {pan} {xx[0]}]：系数相对差 {dc:.1e}，se 相对差 {ds:.1e}，N {int(r_.n.iloc[0])} vs {o_["n"]}')

    desc, corr, _ = describe(D, 'WG_NSRC', X6, X6, P13['C'][0] if v13 else None) if v12 else describe(D)
    desc.to_csv(OUT_ / 'descriptives.csv', index=False, encoding='utf-8-sig')
    corr.to_csv(OUT_ / 'corr.csv', index=False, encoding='utf-8-sig')
    xk = X6 + X6V if v12 else XS + XV
    keep = (['report_id', 'code', 'ORG_CL', 'broker_g', 'sw', 'date', 'month', 'PMAFE0_FY25_RAW', 'PMAFE_FY25', 'PMAFE0_FY24_RAW', 'PMAFE_FY24',
             'SCORE_FY25', 'PMAFEW30_FY25_RAW', 'PMAFEW30_FY25', 'N_PEERS25', 'N_PEERS24', 'N_PEERSW30']
            + [c for v in LAG_W for c in (v + '_RAW', v)] + CTRL + [c for x in xk for c in (x + '_RAW', x)]
            + (['analyst', 'team', 'AUTHOR'] if v13 else []))
    D[D.base.astype(bool)][keep].rename(columns={'PMAFE0_FY25_RAW': 'PMAFE_FY25_RAW', 'PMAFE0_FY24_RAW': 'PMAFE_FY24_RAW'}).to_parquet(
        OUT_ / 'report_sample.parquet', index=False)
    if pair_log:
        T.log_lines.extend(['', '=' * 30 + f' 公司-券商层面（P.main --x {xset_pair} --agg last 的日志，同 pair/checks.txt）', *pair_log])
    (OUT_ / 'checks.txt').write_text('\n'.join(T.log_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ver', choices=['v11', 'v12', 'v13'], default='v11')
    ap.add_argument('--jobs', type=int, default=16)
    ap.add_argument('--out', default=None, help='另指定输出目录（核对用）')
    a = ap.parse_args()
    main(a.jobs, a.ver, a.out)
