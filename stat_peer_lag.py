"""v10 研报层面比较组的“时效差”：同行预测比研报早多少天（只描述，不回归）。

比较组    与 v10 研报层面 PMAFE_FY25 完全相同（table_rating_sink_pair.report_level13 → T.pmafe_v5 的 PMAFE0 口径）：
          其他券商（ORG_CL ≠ 本篇）、同公司、目标年度 2025、发布日在 [d − 90, d]（含两端），每家取区间内最新一条
          （长表 T.long_v5 已按 RPT_DT、CREAT_TM、SEQ 排序，T.pick_peers 取每家最后一条），预测表只含 IS_VLD = 1、预测值非缺失；
          ≥ 3 家（F.MIN_PEERS）才有 PMAFE。本篇不算同行。
样本      v10 研报层面回归（rmain / rladder，PMAFE_FY25，面板 B、C，X = ALL_NSRC 等 12 个全非缺失的 X）的估计样本：
          基础样本、PMAFE_FY25 与六个控制变量非缺失、剔除固定效应单例（R.fit 返回的回归样本）。
LAG       report_dt − peer_dt（自然日）。
输出      output_dag/forecast_newdata/rating_table10_v10_timing/：peer_lags.parquet、report_lags.parquet、summary.csv、corr.csv、
          rate_nsrc_tertiles.csv、lag_distribution.png、checks.txt
用法      C:/Users/hung/projects/.venv/Scripts/python.exe -X utf8 stat_peer_lag.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_rating_sink as T  # noqa: E402
import table_rating_sink_pair as P  # noqa: E402
from src import dag_forecast as F  # noqa: E402
from src import dag_outcomes_reg as R  # noqa: E402

V10 = P.OUT_DEFAULT[('new13', 'last')]
OUT = P.ND / 'rating_table10_v10_timing'
BACK, FWD = T.WINDOWS5['0']  # (90, 0)：PMAFE0 的窗口
TY = 2025
Y = 'PMAFE_FY25'
XS3 = ['RATE_NSRC', 'ALL_NSRC', 'ALL_LNNODE']
LAGV = ['LAG_MEAN', 'LAG_MEDIAN', 'LAG_MIN', 'n_peers']
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(str(s))


def load_inputs():
    """与 table_rating_sink_pair.main 相同的缓存键取面板 d2 和预测长表 lg；指标表用 v10 输出的 new_report_measures.parquet。"""
    def _sample():
        d2_, s_, _ = T.build_sample_v2()
        return d2_, s_, F.build_panel(main_only=False)[0]
    (d2, _, _), hit1 = P.cached('sample_v2', {}, _sample, 3)
    codes = sorted(set(d2.code))
    (_, lg), hit2 = P.cached('est_long', {'years': [2018, 2025], 'codes': codes}, lambda: (lambda e: (e, T.long_v5(e)))(
        T.load_est_eps(codes, years=range(2018, 2026))), 2)
    nm = pd.read_parquet(V10 / 'new_report_measures.parquet')
    log(f'缓存：面板 {"命中" if hit1 else "重算"}，预测长表 {"命中" if hit2 else "重算"}；指标表 {len(nm)} 篇')
    return d2, lg, nm


def regression_sample(Rr):
    """v10 研报层面 PMAFE_FY25 回归（X = ALL_NSRC，控制 = CTRL）在面板 B、C 下的估计样本（R.fit 返回的样本）。"""
    ys = F.y_sample(Rr, Y)
    ids = {}
    for p in ('B', 'C'):
        _, smp = R.fit(ys, Y, ['ALL_NSRC'], T.CTRL, T.PANS[p][0], 'code')
        ids[p] = set(ys.loc[smp.index, 'report_id'])
    reg = pd.read_csv(V10 / 'reg_all.csv')
    n_v10 = reg[(reg.block == 'rmain') & (reg.Y == Y) & (reg.X == 'ALL_NSRC') & (reg.role == 'x')].n.iloc[0]
    b = Rr.base.astype(bool)
    log(f'研报层面：基础样本 {int(b.sum())} 篇，PMAFE_FY25 非缺失 {int(Rr[Y][b].notna().sum())} 篇，再要求控制变量非缺失 '
        f'{int(Rr.loc[b, [Y, "ALL_NSRC"] + T.CTRL].notna().all(axis=1).sum())} 篇；剔除单例后 面板 B {len(ids["B"])}、面板 C {len(ids["C"])} 篇'
        f'（两者相同：{ids["B"] == ids["C"]}）；v10 reg_all.csv 中 rmain ALL_NSRC 面板 C 的 N = {int(n_v10)}')
    assert ids['B'] == ids['C'] and len(ids['C']) == n_v10
    return ids['C']


def peer_lags(rep, lg, ty=TY, back=BACK, fwd=FWD):
    """每篇研报的同行（T.pick_peers，窗口 [d − back, d + fwd]，目标年度 ty）明细；rep 需有 report_id、code、ORG_CL、date。"""
    g = lg[lg.ty == ty].reset_index(drop=True)  # 仍按公司、券商、时间排序
    code_a = g.SEC_CD.to_numpy()
    start = np.flatnonzero(np.r_[True, code_a[1:] != code_a[:-1]])
    stop = np.r_[start[1:], len(g)]
    dt, org = g.RPT_DT.astype(int).to_numpy(), g.ORG_CL.to_numpy()
    slices = {code_a[a]: (dt[a:b], org[a:b], a) for a, b in zip(start, stop)}
    rows = []
    for r in rep.itertuples():
        p = T.pick_peers(slices[r.code], r.ORG_CL, r.date, back, fwd) if r.code in slices else np.array([], int)
        rows.append(pd.DataFrame({'report_id': r.report_id, 'code': r.code, 'report_dt': r.date, 'report_ORG_CL': r.ORG_CL,
                                  'ORG_CL': g.ORG_CL.to_numpy()[p], 'peer_RPT_ID': g.RPT_ID.to_numpy()[p],
                                  'peer_dt': g.RPT_DT.to_numpy()[p], 'F': g.F.to_numpy(float)[p]}))
    pl = pd.concat(rows, ignore_index=True)
    pl['LAG'] = (pd.to_datetime(pl.report_dt, format='%Y%m%d') - pd.to_datetime(pl.peer_dt, format='%Y%m%d')).dt.days
    return pl


def describe(z):
    z = pd.Series(z, dtype=float).dropna()
    q = z.quantile([.05, .1, .25, .5, .75, .9, .95])
    return {'N': len(z), 'mean': z.mean(), 'sd': z.std(), 'p5': q.iloc[0], 'p10': q.iloc[1], 'p25': q.iloc[2], 'p50': q.iloc[3],
            'p75': q.iloc[4], 'p90': q.iloc[5], 'p95': q.iloc[6], 'min': z.min(), 'max': z.max()}


def summary(pl, rl):
    rows = []
    for name, ids in (('全部', set(rl.report_id)), ('RATE_NSRC非缺失', set(rl.report_id[rl.RATE_NSRC.notna()]))):
        lag = pl.LAG[pl.report_id.isin(ids)]
        r = rl[rl.report_id.isin(ids)]
        rows.append({'sample': name, 'var': 'LAG（全部同行合并）', **describe(lag)})
        for v in LAGV:
            rows.append({'sample': name, 'var': v, **describe(r[v])})
        for lo, hi in ((0, 7), (8, 30), (31, 60), (61, 90)):
            m = lag.between(lo, hi)
            rows.append({'sample': name, 'var': f'LAG {lo}–{hi} 天的占比', 'N': int(m.sum()), 'share': m.mean()})
    return pd.DataFrame(rows)


def corr_p(x, y):
    """Pearson r 和双侧 p 值（t = r·sqrt((n−2)/(1−r²))，t(n−2)；Spearman 传入平均名次，同 scipy.stats.spearmanr 的近似）。"""
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else np.inf
    return r, R.t_pvalue(t, n - 2)


def corr_table(rl):
    rows = []
    for spec in ('raw', 'within_firm'):
        for x in XS3:
            for v in LAGV:
                d = rl[['code', x, v]].dropna()
                if spec == 'within_firm':
                    d = d[d.groupby('code').code.transform('size') >= 2]
                    d = d.assign(**{c: d[c] - d.groupby('code')[c].transform('mean') for c in (x, v)})
                for meth in ('pearson', 'spearman'):
                    a, b = (d[x], d[v]) if meth == 'pearson' else (d[x].rank(), d[v].rank())
                    r, p = corr_p(a.to_numpy(float), b.to_numpy(float))
                    rows.append({'spec': spec, 'x': x, 'lag_var': v, 'method': meth, 'r': r, 'p': p,
                                 'N': len(d), 'n_firms': d.code.nunique()})
    return pd.DataFrame(rows)


def tertiles(rl):
    """RATE_NSRC 非缺失的研报按 RATE_NSRC 排序（同值按 report_id）等分三组。"""
    d = rl[rl.RATE_NSRC.notna()].sort_values(['RATE_NSRC', 'report_id']).reset_index(drop=True)
    d['tertile'] = pd.qcut(np.arange(len(d)), 3, labels=['T1（低）', 'T2', 'T3（高）'])
    out = d.groupby('tertile', observed=True).agg(n_reports=('report_id', 'size'), RATE_NSRC_min=('RATE_NSRC', 'min'),
                                                  RATE_NSRC_max=('RATE_NSRC', 'max'), RATE_NSRC_mean=('RATE_NSRC', 'mean'),
                                                  LAG_MEAN=('LAG_MEAN', 'mean'), LAG_MIN=('LAG_MIN', 'mean'), n_peers=('n_peers', 'mean'))
    return out.reset_index()


def plot(pl, rl, path, titles=None, suptitle=None):
    """titles：两个子图标题（None 用默认）；suptitle：总标题（None 用默认，'' 不要总标题）。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    surf, ink, ink2, grid, blue = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#2a78d6'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=surf)
    bins = np.arange(0, 95, 5)
    titles = titles or (f'全部同行预测的 LAG（{len(pl)} 条）', f'每篇研报同行 LAG 的均值（{len(rl)} 篇）')
    if suptitle is None:
        suptitle = 'v10 研报层面比较组：同行预测比研报早多少天（窗口 [d−90, d]，5 天一格）'
    for ax, z, title, ylab in ((axes[0], pl.LAG, titles[0], '同行预测条数'),
                               (axes[1], rl.LAG_MEAN, titles[1], '研报篇数')):
        ax.set_facecolor(surf)
        ax.hist(z, bins=bins, color=blue, edgecolor=surf, linewidth=1.5, zorder=2)
        med = float(np.median(z))
        ax.axvline(med, color=ink, linestyle='--', linewidth=1.2, zorder=3)
        ax.text(med + 1.5, ax.get_ylim()[1] * 0.95, f'中位数 {med:.0f} 天' if med == int(med) else f'中位数 {med:.1f} 天',
                color=ink, fontsize=10, va='top', ha='left')
        ax.set_xlim(0, 90)
        ax.set_xticks(range(0, 91, 15))
        ax.set_xlabel('研报发布日 − 同行预测发布日（天）', color=ink2)
        ax.set_ylabel(ylab, color=ink2)
        ax.set_title(title, color=ink, fontsize=12, loc='left')
        ax.grid(axis='y', color=grid, linewidth=0.6, zorder=0)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            ax.spines[s_].set_color(ink2)
        ax.tick_params(colors=ink2)
    if suptitle:
        fig.suptitle(suptitle, color=ink, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94) if suptitle else None)
    fig.savefig(path, dpi=200, facecolor=surf)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d2, lg, nm = load_inputs()
    Rr, _ = P.report_level13(d2, lg, nm)
    ids = regression_sample(Rr)

    rep = d2[d2.report_id.isin(ids)][['report_id', 'code', 'ORG_CL', 'date', 'ABSERR25', 'A25']].reset_index(drop=True)
    log(f'样本研报 {len(rep)} 篇、{rep.code.nunique()} 家；本篇 ORG_CL 缺失的 {int(rep.ORG_CL.isna().sum())} 篇；发布日范围 {rep.date.min()}–{rep.date.max()}')
    pl = peer_lags(rep, lg)

    # 核对
    log(f'同行明细 {len(pl)} 行；LAG 范围 [{pl.LAG.min()}, {pl.LAG.max()}]，全部在 [0, 90]：{bool(pl.LAG.between(0, 90).all())}；'
        f'同一研报内同一同行券商重复的行 {int(pl.duplicated(["report_id", "ORG_CL"]).sum())}；同行 ORG_CL 等于本篇 ORG_CL 的行 '
        f'{int((pl.ORG_CL == pl.report_ORG_CL).sum())}；同行 ORG_CL 缺失 {int(pl.ORG_CL.isna().sum())}')
    n = pl.groupby('report_id').size()
    chk = rep.merge(Rr[['report_id', 'N_GRP0_FY25', 'PMAFE0_FY25_RAW']], on='report_id', how='left')
    chk['n_peers'] = chk.report_id.map(n).fillna(0).astype(int)
    pa = pl.assign(AE=(pl.F - pl.report_id.map(rep.set_index('report_id').A25)).abs())
    s_ae = pa.groupby('report_id').AE.agg(['sum', 'size'])
    chk['PMAFE_RE'] = [(own - m) / m if m > 0 else np.nan for own, m in
                       zip(chk.ABSERR25, (chk.report_id.map(s_ae['sum']) + chk.ABSERR25) / (chk.n_peers + 1))]
    bad_n = int((chk.n_peers + 1 != chk.N_GRP0_FY25).sum())
    dif = (chk.PMAFE_RE - chk.PMAFE0_FY25_RAW).abs()
    bad_p = int((~np.isclose(chk.PMAFE_RE, chk.PMAFE0_FY25_RAW, rtol=1e-12, atol=1e-12)).sum())
    log(f'逐篇核对（与 report_level13 的 N_GRP0_FY25 = 同行数 + 1、PMAFE0_FY25_RAW 缩尾前值比较）：比较组人数不一致 {bad_n} 篇；'
        f'PMAFE 不一致 {bad_p} 篇（最大绝对差 {dif.max():.1e}）；同行数 < 3 的 {int((chk.n_peers < F.MIN_PEERS).sum())} 篇')
    rw = Rr.set_index('report_id').loc[chk.report_id, Y]
    log(f'  缩尾后 PMAFE_FY25（回归所用）与重算值落在缩尾界外的篇数：{int((~np.isclose(rw.to_numpy(), chk.PMAFE_RE.to_numpy())).sum())}（缩尾造成，属正常）')

    pl = pl[['report_id', 'code', 'report_dt', 'report_ORG_CL', 'ORG_CL', 'peer_RPT_ID', 'peer_dt', 'LAG', 'F']]
    pl.to_parquet(OUT / 'peer_lags.parquet', index=False)

    g = pl.groupby('report_id').LAG
    rl = rep[['report_id', 'code', 'ORG_CL', 'date']].rename(columns={'date': 'report_dt'})
    rl['month'] = rl.report_dt.str[:4] + '-' + rl.report_dt.str[4:6]
    agg = pd.DataFrame({'n_peers': g.size(), 'LAG_MEAN': g.mean(), 'LAG_MEDIAN': g.median(), 'LAG_MIN': g.min(), 'LAG_MAX': g.max(),
                        'SHARE_LE7': g.apply(lambda z: (z <= 7).mean()), 'SHARE_LE30': g.apply(lambda z: (z <= 30).mean())})
    rl = rl.merge(agg.reset_index(), on='report_id', how='left', validate='1:1').merge(nm[['report_id'] + XS3], on='report_id', how='left',
                                                                                       validate='1:1')
    rl = rl.sort_values(['report_dt', 'report_id']).reset_index(drop=True)
    rl.to_parquet(OUT / 'report_lags.parquet', index=False)
    log(f'report_lags：{len(rl)} 篇；RATE_NSRC 非缺失 {int(rl.RATE_NSRC.notna().sum())} 篇；ALL_NSRC、ALL_LNNODE 缺失 '
        f'{int(rl.ALL_NSRC.isna().sum())}/{int(rl.ALL_LNNODE.isna().sum())}')
    log('按月篇数：' + '，'.join(f'{k} {v}' for k, v in rl.month.value_counts().sort_index().items()))

    sm = summary(pl, rl)
    sm.to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')
    cr = corr_table(rl)
    cr.to_csv(OUT / 'corr.csv', index=False, encoding='utf-8-sig')
    tt = tertiles(rl)
    tt.to_csv(OUT / 'rate_nsrc_tertiles.csv', index=False, encoding='utf-8-sig')
    plot(pl, rl, OUT / 'lag_distribution.png')
    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        log(sm.round(3).to_string(index=False))
        log(cr.round(4).to_string(index=False))
        log(tt.round(3).to_string(index=False))
    (OUT / 'checks.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def doc_figure(path=None):
    """文档用的同一张分布图（读已存的 peer_lags / report_lags，不重算、不覆盖 OUT 里的文件）。"""
    path = Path(path) if path else T.ROOT / 'docs' / 'figures' / 'peer_lag_distribution.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    pl, rl = pd.read_parquet(OUT / 'peer_lags.parquet'), pd.read_parquet(OUT / 'report_lags.parquet')
    plot(pl, rl, path, titles=(f'(a) 全部同行预测（{len(pl)} 条）', f'(b) 每篇研报的同行平均时滞（{len(rl)} 篇）'),
         suptitle='研报发布前 90 天内同行预测的时滞分布')
    print(f'已写 {path}')


if __name__ == '__main__':
    if '--doc-figure' in sys.argv:
        doc_figure()
    else:
        main()
