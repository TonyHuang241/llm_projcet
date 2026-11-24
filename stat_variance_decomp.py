"""v12 六个逻辑结构变量的固定效应方差分解（仿 Liu, Liu & Sui 2026 Table 5）：变量对各组固定效应虚拟变量 + 常数项做 OLS，报告 R² 和调整 R²。

变量    RT_NSRC、RT_LNPATH、RT_PATHLEN、WG_NSRC、WG_LNPATH、WG_PATHLEN；wins = rating_table10_v12/report_sample.parquet 的缩尾值，raw = *_RAW。
设定    月份、券商（broker_g）、公司（code）、行业（sw）及其可加组合，行业×月份、券商×行业交互，第一作者及第一作者 + 公司，
        另加分析师团队（完整作者串）及团队 + 公司。作者取自 data/EST_COMP_TN/2024 的 AUTHOR（RPT_ID = report_id 大写、SEC_CD = code），
        第一作者 = AUTHOR 按逗号分开的第一个名字；作者和团队都与 broker_g 拼接成键，避免不同券商的同名分析师并为一组。
口径    R² = 1 − RSS/TSS；调整 R² = 1 − (1 − R²)(N − 1)/(N − K)，K = 设计矩阵的秩（含常数项），N − K ≤ 0 记缺失；不剔除单例。
样本    S1：基础样本中该变量非缺失；S2：S1 中公司至少 2 篇；S3：v12 rmain（FE = C，控制 + PEERLAG25）对该变量的估计样本（R.fit 返回）。
        WG 三个变量另在 RT_NSRC 非缺失的研报上各算一遍（S1_RT、S2_RT、S3_RT；S3_RT = RT_NSRC 的 rmain 估计样本）。
输出    output_dag/forecast_newdata/rating_table10_v12_vardecomp/：decomp.csv、decomp_wide_S1/S2/S3.csv（调整 R²，%）、
        decomp_wide_r2_S1/S2/S3.csv（未调整 R²，%）、checks.txt
用法    C:/Users/hung/projects/.venv/Scripts/python.exe -X utf8 stat_variance_decomp.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_rating_sink as T  # noqa: E402
import table_rating_sink_pair as P  # noqa: E402
from src import dag_outcomes_reg as R  # noqa: E402

V12 = P.ND / 'rating_table10_v12'
OUT = P.ND / 'rating_table10_v12_vardecomp'
X6 = P.XS_RT6
SPECS = {
    'month': ['month'], 'broker': ['broker_g'], 'firm': ['code'], 'ind': ['sw'],
    'month+broker': ['month', 'broker_g'], 'month+firm': ['month', 'code'], 'broker+firm': ['broker_g', 'code'],
    'month+broker+firm': ['month', 'broker_g', 'code'], 'ind+broker': ['sw', 'broker_g'],
    'ind×month': ['ind_month'], 'broker×ind': ['broker_ind'],
    'author1': ['author1'], 'author1+firm': ['author1', 'code'], 'team': ['team'], 'team+firm': ['team', 'code'],
}
DIMS = {'n_month': 'month', 'n_broker': 'broker_g', 'n_firm': 'code', 'n_ind': 'sw'}
lines = []


def log(s=''):
    print(s, flush=True)
    lines.append(str(s))


def authors(s):
    """EST_COMP_TN 2024 年文件里各研报的 AUTHOR（不限 IS_VLD；同一研报各行的 AUTHOR 相同时取之）。"""
    fs = sorted(glob.glob(str(T.ROOT / 'data' / 'EST_COMP_TN' / '2024' / '2024-[0-9][0-9].parquet')))
    ids = set(s.report_id.str.upper())
    e = pd.concat([pq.read_table(f, columns=['RPT_ID', 'SEC_CD', 'AUTHOR']).to_pandas() for f in fs], ignore_index=True)
    e = e[e.RPT_ID.str.upper().isin(ids)].assign(RID=lambda t: t.RPT_ID.str.upper())
    m = s[['report_id', 'code']].assign(RID=s.report_id.str.upper()).merge(e, left_on=['RID', 'code'], right_on=['RID', 'SEC_CD'], how='left')
    g = m.groupby('report_id').AUTHOR.agg(lambda x: sorted(set(x.dropna())))
    log(f'作者：EST_COMP_TN 2024 年文件按 RPT_ID（= report_id 大写）+ SEC_CD（= code）匹配，{len(s)} 篇中有 AUTHOR 的 {int((g.str.len() > 0).sum())} 篇，'
        f'同一研报有多个不同 AUTHOR 串的 {int((g.str.len() > 1).sum())} 篇')
    a = g.map(lambda v: v[0] if v else np.nan)
    hs = pd.read_parquet(T.ROOT / 'data' / 'hs300_2024_reports.parquet', columns=['report_id', 'analysts'])
    hs = hs.set_index('report_id').analysts.replace('', np.nan)
    split = lambda v: [n.strip() for n in str(v).replace('，', ',').replace('、', ',').split(',') if n.strip()]  # noqa: E731
    lh = s.report_id.map(hs).map(lambda v: split(v) if pd.notna(v) else np.nan)
    le = s.report_id.map(a).map(lambda v: split(v) if isinstance(v, str) else np.nan)
    both = lh.notna() & le.notna()
    first_same = sum(h[0] == e[0] for h, e in zip(lh[both], le[both]))
    in_est = sum(h[0] in e for h, e in zip(lh[both], le[both]))
    log(f'  对照 data/hs300_2024_reports.parquet 的 analysts 列（从研报首页解析，多数为空串）：非空 {int(lh.notna().sum())} 篇；其中第一作者与 AUTHOR '
        f'第一个名字相同 {first_same} 篇，在 AUTHOR 名单里 {in_est} 篇。因覆盖不全，固定效应只用 EST_COMP_TN 的 AUTHOR')
    log('  其他查过的文件：output_dag/deep/data/reports.parquet、output_dag/data/reports.parquet、output_dag/forecast/forecast_panel_raw.parquet '
        '都没有作者或分析师字段')
    return s.report_id.map(a)


def fit_r2(y, fe_frame):
    """常数项 + 各固定效应虚拟变量（每个去掉一组）的 OLS：返回 (R², 调整 R², K)。"""
    X = np.column_stack([np.ones(len(y)), R.fe_dummies(fe_frame, list(fe_frame.columns))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    k = int(np.linalg.matrix_rank(X))
    n = len(y)
    r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n - k > 0 else np.nan
    return r2, adj, k


def samples(D, s3fes=None, s2=True):
    """{样本名: {变量: 布尔掩码}}。s3fes：{样本名: rmain 的固定效应列表}（估计样本 = R.fit 返回的样本），默认 {'S3': v12 的面板 C}。"""
    s3fes = s3fes or {'S3': T.PANS['C'][0]}
    ys = D.assign(base=True, code_ty=D.code + '_2025', ty='2025')
    s3 = {n: {} for n in s3fes}
    for n, fe in s3fes.items():
        for x in X6:
            _, smp = R.fit(ys, 'PMAFE_FY25', [x], T.CTRL + ['PEERLAG25'], fe, 'code')
            s3[n][x] = D.index.isin(smp.index)
    names = ['S1'] + (['S2'] if s2 else []) + list(s3fes)
    out = {**{n: {} for n in names}, **{n + '_RT': {} for n in names}}
    rt_ok = D.RT_NSRC.notna()
    for x in X6:
        m1 = D[x].notna()
        out['S1'][x] = m1.to_numpy()
        if s2:
            out['S2'][x] = (m1 & (D.code.map(D[m1].code.value_counts()) >= 2)).to_numpy()
        for n in s3fes:
            out[n][x] = s3[n][x]
        if x.startswith('WG'):
            m1r = m1 & rt_ok
            out['S1_RT'][x] = m1r.to_numpy()
            if s2:
                out['S2_RT'][x] = (m1r & (D.code.map(D[m1r].code.value_counts()) >= 2)).to_numpy()
            for n in s3fes:
                out[n + '_RT'][x] = s3[n]['RT_NSRC'] & m1.to_numpy()
    return out


def group_stats(D, m, tag, cols=('month', 'broker_g', 'code', 'sw', 'author1', 'team', 'broker_ind', 'ind_month')):
    d = D[m]
    parts = []
    for c in cols:
        vc = d[c].value_counts()
        parts.append(f'{c} {vc.size} 组（1 篇的 {int((vc == 1).sum())}、2 篇 {int((vc == 2).sum())}、≥3 篇 {int((vc >= 3).sum())}，最多 {vc.max()}）')
    log(f'  {tag}：N {len(d)}；' + '；'.join(parts))


V13 = P.ND / 'rating_table10_v13'
SPECS13 = {
    'month': ['month'], 'ind': ['sw'], 'firm': ['code'], 'analyst': ['analyst'], 'month+analyst': ['month', 'analyst'],
    'ind+analyst': ['sw', 'analyst'], 'firm+analyst': ['code', 'analyst'], 'month+firm+analyst': ['month', 'code', 'analyst'],
    'team': ['team'], 'firm+team': ['code', 'team'],
}
CFG = {
    'v12': dict(src=V12, out=OUT, specs=SPECS, dims=DIMS, s3fes=None, s2=True, wide=('S1', 'S2', 'S3'),
                spot=(('S1', 'WG_NSRC', 'firm'), ('S3', 'RT_LNPATH', 'broker+firm'), ('S2', 'WG_PATHLEN', 'month+broker+firm'))),
    # v13：券商换成分析师（analyst = broker_g|第一作者，report_sample 里已有）；S3 / S3D / S3B = v13 rmain 在面板 C / D / B 下的估计样本
    'v13': dict(src=V13, out=P.ND / 'rating_table10_v13_vardecomp', specs=SPECS13,
                dims={'n_month': 'month', 'n_analyst': 'analyst', 'n_firm': 'code', 'n_ind': 'sw'},
                s3fes={'S3': P.PANS13['C'][0], 'S3D': P.PANS13['D'][0], 'S3B': P.PANS13['B'][0]}, s2=False, wide=('S1', 'S3', 'S3D', 'S3B'),
                spot=(('S1', 'WG_NSRC', 'analyst'), ('S3', 'RT_LNPATH', 'firm+analyst'), ('S3D', 'WG_PATHLEN', 'month+firm+analyst'))),
}


def main(ver='v12'):
    cfg = CFG[ver]
    OUT_, SPECS_, DIMS_ = cfg['out'], cfg['specs'], cfg['dims']
    OUT_.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(cfg['src'] / 'report_sample.parquet').reset_index(drop=True)
    log(f'report_sample.parquet：{len(D)} 篇（基础样本）；sw 缺失 {int(D.sw.isna().sum())}、broker_g 缺失 {int(D.broker_g.isna().sum())}、month 缺失 {int(D.month.isna().sum())}')
    D['author'] = authors(D)
    D['author1'] = D.broker_g.astype(str) + '|' + D.author.str.split(',').str[0].str.strip()
    if ver == 'v13':
        tm = D.broker_g.astype(str) + '|' + D.author.map(lambda v: ','.join(sorted(n.strip() for n in v.split(','))) if isinstance(v, str) else np.nan)
        log(f'v13 report_sample 的 analyst / team 与本脚本按 AUTHOR 重建的第一作者 / 团队逐篇相同：{bool((D.analyst == D.author1).all())} / '
            f'{bool((D.team == tm).all())}；AUTHOR 列相同：{bool((D.AUTHOR == D.author).all())}')
    else:
        D['team'] = D.broker_g.astype(str) + '|' + D.author.map(lambda v: ','.join(sorted(n.strip() for n in v.split(','))) if isinstance(v, str) else np.nan)
        D.loc[D.author.isna(), ['author1', 'team']] = np.nan
    D['ind_month'] = D.sw.astype(str) + '|' + D.month
    D['broker_ind'] = D.broker_g.astype(str) + '|' + D.sw.astype(str)
    SM = samples(D, cfg['s3fes'], cfg['s2'])
    reg = pd.read_csv(cfg['src'] / 'reg_all.csv')
    if ver == 'v12':
        n_v12 = reg[(reg.level == 'report') & (reg.block == 'rmain') & (reg.role == 'x')].set_index('X').n
        log('S3 篇数与 v12 rmain 的 N：' + '；'.join(f'{x} {int(SM["S3"][x].sum())} / {int(n_v12[x])}' for x in X6))
    else:
        q = reg[(reg.level == 'report') & (reg.block == 'rladder') & (reg.Y == 'PMAFE_FY25') & (reg.role == 'x')]
        for n, p in (('S3', 'C'), ('S3D', 'D'), ('S3B', 'B')):
            nn = q[q.panel == p].set_index('X').n
            log(f'{n} 篇数与 v13 rladder PMAFE_FY25 面板 {p} 的 N：' + '；'.join(f'{x} {int(SM[n][x].sum())} / {int(nn[x])}' for x in X6))

    log('各样本的组数和组内篇数分布：')
    for sname, d_ in SM.items():
        for grp, xs_ in (('RT', [x for x in X6 if x.startswith('RT')]), ('WG', [x for x in X6 if x.startswith('WG')])):
            xs_ = [x for x in xs_ if x in d_]
            if not xs_:
                continue
            masks = [d_[x] for x in xs_]
            same = all((m_ == masks[0]).all() for m_ in masks)
            gcols = (('month', 'broker_g', 'code', 'sw', 'author1', 'team', 'broker_ind', 'ind_month') if ver == 'v12'
                     else ('month', 'code', 'sw', 'analyst', 'team', 'broker_g'))
            group_stats(D, masks[0], f'{sname} {grp}（{"/".join(xs_)}{"，三个变量样本相同" if same else "，样本不同，按第一个变量"}）', gcols)

    rows = []
    for sname, d_ in SM.items():
        for x, m in d_.items():
            sub = D[m]
            for vt, col in (('wins', x), ('raw', x + '_RAW')):
                y = sub[col].to_numpy(float)
                for spec, fes in SPECS_.items():
                    fe = sub[fes]
                    ok = fe.notna().all(axis=1).to_numpy()
                    r2, adj, k = fit_r2(y[ok], fe[ok])
                    rows.append({'sample': sname, 'variable': x, 'value_type': vt, 'spec': spec, 'N': int(ok.sum()),
                                 **{k_: int(sub.loc[ok, c].nunique()) for k_, c in DIMS_.items()}, 'K': k, 'r2': r2, 'adj_r2': adj})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_ / 'decomp.csv', index=False, encoding='utf-8-sig')

    # 第二种算法抽查：按组去均值（R.within 交替投影）后的 RSS 与 lstsq 的 R² 对比
    for sname, x, spec in cfg['spot']:
        sub = D[SM[sname][x]][[x] + SPECS_[spec]].copy()
        w = R.within(sub, [x], SPECS_[spec], iters=20000, tol=1e-14)[:, 0]
        y = sub[x].to_numpy(float)
        r2_w = 1 - (w ** 2).sum() / ((y - y.mean()) ** 2).sum()
        r2_l = res[(res['sample'] == sname) & (res.variable == x) & (res.value_type == 'wins') & (res.spec == spec)].r2.iloc[0]
        if len(SPECS_[spec]) == 1:
            gm = sub.groupby(SPECS_[spec][0])[x].transform('mean')
            r2_g = 1 - ((y - gm) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            extra = f'，groupby 去均值 R² {r2_g:.10f}'
        else:
            extra = ''
        log(f'抽查 {sname} {x} {spec}：lstsq R² {r2_l:.10f}，R.within 交替投影 R² {r2_w:.10f}{extra}，差 {abs(r2_l - r2_w):.1e}')

    for sname in cfg['wide']:
        for val, fn in (('adj_r2', f'decomp_wide_{sname}.csv'), ('r2', f'decomp_wide_r2_{sname}.csv')):
            q = res[res['sample'].isin([sname, sname + '_RT']) & (res.value_type == 'wins')].copy()
            q['row'] = np.where(q['sample'] == sname, q.variable, q.variable + '（RT 样本）')
            wide = (q.pivot_table(index='row', columns='spec', values=val, sort=False) * 100).round(1)[list(SPECS_)]
            wide.insert(0, 'N', q.groupby('row').N.max())
            wide.to_csv(OUT_ / fn, encoding='utf-8-sig')
            log(f'\n{fn}（%）：\n' + wide.to_string())
    (OUT_ / 'checks.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--ver', choices=['v12', 'v13'], default='v12')
    ap.add_argument('--out', default=None, help='另指定输出目录（核对用）')
    a = ap.parse_args()
    if a.out:
        CFG[a.ver]['out'] = Path(a.out)
    main(a.ver)
