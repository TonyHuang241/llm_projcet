"""预测准确度回归的三个稳健性检验（表格在 src/dag_forecast_tables.py，入口 regress_dag_forecast.py）：

一、同行基准（--peer-variants，peer_variants）：PMAFE_FY24、PMAFE_FY25 换 8 种同行定义（V0–V7）× 2 种同行来源（全部预测行 / 只用主体公司行）
    重算并回归，写 <输出目录>/peer_variants.csv、peer_variants_panel.parquet、tables/f11–f12。
    同行的共同口径同 PMAFE1：其他券商（ORG_CL 不同）、同一公司、目标年度相同（同行那行的 RPT_YR_T + k 等于目标年度，k = 1、2、3）、
    每家券商取窗口内最近一条。本篇预测和绝对误差同 AFE_FY24 / AFE_FY25（元，不除 STD_PAST）。
    V0–V5、V7：Y = (本篇绝对误差 − 同行基准) ÷ 同行基准，同行基准 > 0 才算，在基础样本内 1%/99% 缩尾；
    V6：Y = (同行绝对误差严格小于本篇的家数 + 0.5 × 相等的家数) ÷ 同行家数，不缩尾。V0 与 PMAFE_FY24 / PMAFE_FY25 的算法逐步相同。
    回归：三个 Panel、三个 X 各自单独进入，控制变量、固定效应、聚类同主回归（预测期限用 LNHORIZON24 / LNHORIZON25）。

二、两种汇点口径的联合回归（--joint，joint）：只用两套指标都非缺失的研报，把两套同类指标放进同一个回归。读已有输出（不重算变量），
    写 <输出目录>/joint_panel.parquet、joint_results.csv、tables/j01–j04。
    数据：A 口径的 forecast_panel.parquet（A 指标、控制变量、全部预测行口径的 PMAFE）和 main_only/forecast_panel.parquet（只用主体公司行的 PMAFE），
    B 口径的 forecast_panel.parquet（B 指标）。两套指标都用各自口径已缩尾的值（不在交集里重缩）。
    每种指标类型三个回归：(1) 只放 A 的；(2) 只放 B 的；(3) 两个一起放；三个回归同一样本，X 在回归样本内标准化。
    系数相等的检验（第 3 个回归）：两个标准化后的 X 记为 a、b，y = β_a·a + β_b·b + … 等价于 y = γ_S·(a + b) + γ_D·(a − b) + …，
    γ_D = (β_a − β_b) / 2，其按公司聚类（CR1）的 t 和 p（t(G−1)）就是 H0: β_a = β_b 的检验。

三、比值回归（--ratio rating_rev，ratio）：“投资评级”节点指标 ÷ “营业收入”节点指标，对 PMAFE_FY24、PMAFE_FY25（同行基准 V0–V7 × 两种来源）回归，
    写 ratio_panel.parquet、ratio_results.csv、tables/（比值口径）和 tables/log/（对数口径）。
    样本：基础样本里两个节点都是终点（入度 > 0、出度 = 0）的研报（rating_strict 与 sink_rev_strict 两个口径都有指标）。
    解释变量各自单独进回归，都用未缩尾的原始值构造，再在本样本内按 1%/99% 缩尾：
      比值口径  Q_NSRC = 评级源点数 ÷ 营收源点数；Q_LNPATH = 评级 ln(1+路径条数) ÷ 营收 ln(1+路径条数)；Q_PATHLEN = 平均路径长度之比
      对数口径  LQ_NSRC = ln(源点数之比)；LQ_NPATH = ln(路径条数之比)；LQ_PATHLEN = ln(平均路径长度之比)
    被解释变量：variant_panel 的 16 种口径；Panel N、A、B、C。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import dag_forecast as F
from src import dag_outcomes_reg as R

# ---------------------------------------------------------------- 一、同行基准
VARIANTS = {  # 代号: (窗口天数, 是否含研报当天, 同行基准, 至少几家)
    'V0': (90, True, 'mean', 3), 'V1': (30, True, 'mean', 3), 'V2': (60, True, 'mean', 3), 'V3': (180, True, 'mean', 3),
    'V4': (90, True, 'median', 3), 'V5': (90, True, 'mean', 5), 'V6': (90, True, 'rank', 3), 'V7': (90, False, 'mean', 3)}
SOURCES = {'all': '全部预测行', 'main': '主体公司行'}
TARGETS = {'FY25': '25', 'FY24': '24'}  # Y 的标签 -> 年度后缀


def variant_panel(w, lgs):
    """基础样本每篇研报一行：每个 (年度, 定义, 来源) 的 Y（列名 FY25_V1_all 等，已按规则缩尾）、未缩尾值（*_raw）、同行家数（N_*）、同行基准（BENCH_*）。"""
    b = w[w.base]
    out = {'report_id': b.report_id.to_numpy(), 'code': b.code.to_numpy()}
    for src, lg in lgs.items():
        peers = F.Peers(lg)
        for yk, t in TARGETS.items():
            ty = int('20' + t)
            res = {v: ([], [], []) for v in VARIANTS}
            for code, org, date, own, a in zip(b.code, b.ORG_CL, b.date, b[f'ABSERR{t}'], b[f'A{t}']):
                cache = {}
                for v, (days, today, stat, mn) in VARIANTS.items():
                    y, n, bench = np.nan, 0, np.nan
                    if (days, today) not in cache:
                        pos = peers.last_each(code, ty, org, date, days, today)
                        cache[(days, today)] = None if pos is None else peers.abs_errors(pos, a)
                    e = cache[(days, today)]
                    if e is not None:
                        n = len(e)
                        if stat == 'rank':
                            if n >= mn and np.isfinite(own):
                                y = ((e < own).sum() + 0.5 * (e == own).sum()) / n
                        else:
                            bench = e.mean() if stat == 'mean' else e.median()
                            if n >= mn and bench > 0:
                                y = (own - bench) / bench
                    res[v][0].append(y); res[v][1].append(n); res[v][2].append(bench)  # noqa: E702
            for v in VARIANTS:
                c = f'{yk}_{v}_{src}'
                raw = pd.Series(res[v][0], dtype=float).replace([np.inf, -np.inf], np.nan)
                out[c + '_raw'], out[f'N_{c}'], out[f'BENCH_{c}'] = raw.to_numpy(), res[v][1], res[v][2]
                out[c] = (raw if v == 'V6' else R.winsor(raw)).to_numpy()
    return pd.DataFrame(out)


def variant_regress(w, vp, xs, pans=F.PANELS):
    """每个 (年度, 定义, 来源) × Panel（pans，默认 A、B、C）× X（单独进入）。"""
    d0 = w[w.base].merge(vp.drop(columns='code'), on='report_id', how='left')
    rows = []
    for yk, t in TARGETS.items():
        ys = [f'{yk}_{v}_{src}' for v in VARIANTS for src in SOURCES]
        d = F.slim(F.set_target_year(d0, f'LNHORIZON{t}', f'TY{t}'), ys, xs, pans)
        for v in VARIANTS:
            for src in SOURCES:
                y = f'{yk}_{v}_{src}'
                for pan, (fe, ctrl) in pans.items():
                    for x in xs:
                        res, _ = R.fit(d, y, [x], ctrl, fe, 'code')
                        rows += [{'Y': f'PMAFE_{yk}', 'variant': v, 'source': src, 'panel': pan, 'X': x,
                                  **F.pick(r, ['coef', 'se', 't', 'p', 'n', 'n_clusters', 'y_mean', 'y_sd'])} for r in res]
    return pd.DataFrame(rows)


def peer_variants(out, w, lgs, xs, sink):
    from src import dag_forecast_tables as T
    vp = variant_panel(w, lgs)
    vp.to_parquet(out / 'peer_variants_panel.parquet', index=False)
    res = variant_regress(w, vp, xs)
    res.to_csv(out / 'peer_variants.csv', index=False, encoding='utf-8-sig')
    T.write(out / 'tables' / 'f11_peer_variants_FY25.md', T.peer_table(res, 'FY25', xs, sink))
    T.write(out / 'tables' / 'f12_peer_variants_FY24.md', T.peer_table(res, 'FY24', xs, sink))
    print(f'同行基准稳健性：{len(res)} 个系数 -> {out}', flush=True)


# ---------------------------------------------------------------- 二、联合回归
JOINTS = {  # 组合名: 两个口径的目录、列名前缀、长表里的 X 标签、表里的行名、指标类型的中文名、样本和指标定义的表注文字
    'joint_rating_rev': dict(
        dirs=('rating_sink', 'sink_rev'), prefix=('RT_', 'RV_'), xlab=('rating', 'rev'), names=('投资评级节点', '营业收入节点'),
        kinds={'RATIO': '源点数', 'LNPATH': '路径数', 'PATHLEN': '平均路径长度'},
        sample='“投资评级”节点和“营业收入”节点都有路径',
        defs=('投资评级节点（营业收入节点）的指标只看图中名称恰好为“投资评级”（“营业收入”）的节点及其全部祖先：'
              '源点数 = 能到达该节点的源点个数，路径数 = ln(1 + 以该节点为终点的路径条数)，平均路径长度 = 这些路径的平均边数；')),
    'joint_val_cf': dict(
        dirs=('sink_val', 'sink_cf'), prefix=('VAL_', 'CF_'), xlab=('val', 'cf'), names=('估值类汇点', '现金流类汇点'),
        kinds={'RATIO': '源汇比', 'LNPATH': '路径数', 'PATHLEN': '平均路径长度'},
        sample='估值类汇点和现金流类汇点都有',
        defs=('汇点为图中入度大于 0、出度为 0 的节点，按名称先判估值类（评级、目标价、估值、市值、股价、PE/PB/PS 等，不含“市值管理”），'
              '否则判现金流类（收入、利润、毛利、费用、盈利、EPS、现金流、业绩、ROE、分红等）；对一篇研报里某一类的全部汇点 T：'
              '源汇比 = 能到达 T 中任一汇点的源点个数 ÷ T 的汇点个数，路径数 = ln(1 + 以 T 中各汇点为终点的路径条数之和)，'
              '平均路径长度 = 这些路径合在一起的平均边数；')),
    'joint_val_cfcore': dict(
        dirs=('sink_val_n', 'sink_cf_core'), prefix=('VAL_', 'CFC_'), xlab=('val', 'cfcore'), names=('估值类汇点', '核心现金流类汇点'),
        kinds={'RATIO': '源点数', 'LNPATH': '路径数', 'PATHLEN': '平均路径长度'},
        sample='估值类汇点和核心现金流类汇点都有',
        defs=('汇点为图中入度大于 0、出度为 0 的节点，按名称先判估值类（评级、目标价、估值、市值、股价、PE/PB/PS 等，不含“市值管理”），'
              '否则判核心现金流类（名称含收入、营收、利润或业绩，且不含“利润率”），其余不用；对一篇研报里某一类的全部汇点 T：'
              '源点数 = 能到达 T 中任一汇点的源点个数（不除以汇点个数），路径数 = ln(1 + 以 T 中各汇点为终点的路径条数之和)，'
              '平均路径长度 = 这些路径合在一起的平均边数；')),
}
JOINT_YS = {'PMAFE_FY25': '25', 'PMAFE_FY24': '24'}
JOINT_CTRL = ['LNHORIZON24', 'LNHORIZON25', 'LNCHARS_R', 'LNNODES_R', 'SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']


def joint_panel(fc_dir, cfg):
    """交集样本：A、B 两套指标（列名前缀见 cfg）、PMAFE_FY25/24 的 _all 和 _main、控制变量、固定效应用的列。"""
    (da, db), (pa, pb) = cfg['dirs'], cfg['prefix']
    rt = pd.read_parquet(fc_dir / da / 'forecast_panel.parquet')
    rm = pd.read_parquet(fc_dir / da / 'main_only' / 'forecast_panel.parquet').set_index('report_id')
    rv = pd.read_parquet(fc_dir / db / 'forecast_panel.parquet').set_index('report_id')
    d = rt[rt.base][['report_id', 'code', 'company', 'date', 'broker', 'broker_g', 'sw', 'TY24', 'TY25'] + JOINT_CTRL].copy()
    for k in cfg['kinds']:
        d[pa + k] = rt.loc[rt.base, 'R_' + k].to_numpy()
        d[pb + k] = rv.loc[d.report_id, 'R_' + k].to_numpy()
    for y in JOINT_YS:
        d[y + '_all'] = rt.loc[rt.base, y].to_numpy()
        d[y + '_main'] = rm.loc[d.report_id, y].to_numpy()
    keep = d[[p + k for p in (pa, pb) for k in cfg['kinds']]].notna().all(axis=1)
    return d[keep].reset_index(drop=True)


def joint_regress(d, cfg, pans=F.PANELS):
    rows = []
    for y, t in JOINT_YS.items():
        e = F.set_target_year(d, f'LNHORIZON{t}', f'TY{t}')
        for src in SOURCES:
            yc = f'{y}_{src}'
            for pan, (fe, ctrl) in pans.items():
                for k in cfg['kinds']:
                    a, b = cfg['prefix'][0] + k, cfg['prefix'][1] + k
                    for reg, xs in ((1, [a]), (2, [b]), (3, [a, b])):
                        res, s = R.fit(e, yc, xs, ctrl, fe, 'code')
                        diff = {}
                        if reg == 3 and res:  # 重参数化：标准化后的 a、b -> 和 S、差 D
                            s = s.assign(S=s[a] + s[b], D=s[a] - s[b])
                            rd, _ = R.fit(s, yc, ['S', 'D'], ctrl, fe, 'code', standardize=False)
                            dr = [r for r in rd if r['X'] == 'D'][0]
                            diff = {'diff_coef': 2 * dr['coef'], 'diff_t': dr['t'], 'diff_p': dr['p']}
                        rows += [{'Y': y, 'source': src, 'panel': pan, 'kind': k, 'reg': reg,
                                  'X': cfg['xlab'][0] if r['X'] == a else cfg['xlab'][1], 'var': r['X'],
                                  **F.pick(r, ['coef', 'se', 't', 'p', 'n', 'n_clusters', 'adj_r2', 'y_mean', 'y_sd']), **diff}
                                 for r in res]
    return pd.DataFrame(rows)


def joint(fc_dir, name='joint_rating_rev', with_nofe=False, out_dir=None):
    """读 fc_dir 下两个口径的已有输出。with_nofe=True：多一个不加固定效应的 Panel N（N、A、B、C），两个口径的输入改读
    --with-nofe 的输出目录（<口径目录>_n/），默认写到 <组合名>_n/。out_dir 指定输出目录（默认 fc_dir/<组合名>[_n]）。"""
    from src import dag_forecast_tables as T
    fc_dir, cfg = Path(fc_dir), JOINTS[name]
    pans = F.panels(with_nofe)
    if with_nofe:
        cfg = {**cfg, 'dirs': tuple(x + '_n' for x in cfg['dirs'])}
    out = Path(out_dir) if out_dir else fc_dir / (name + ('_n' if with_nofe else ''))
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    d = joint_panel(fc_dir, cfg)
    d.to_parquet(out / 'joint_panel.parquet', index=False)
    res = joint_regress(d, cfg, pans)
    res.to_csv(out / 'joint_results.csv', index=False, encoding='utf-8-sig')
    n_obs, n_firms = len(d), d.code.nunique()
    for i, (y, src) in enumerate([('PMAFE_FY25', 'all'), ('PMAFE_FY24', 'all'), ('PMAFE_FY25', 'main'), ('PMAFE_FY24', 'main')], start=1):
        T.write(out / 'tables' / f'j0{i}_{y}{"" if src == "all" else "_main"}.md', T.joint_table(res, y, src, n_obs, n_firms, cfg, pans))
    print(f'交集样本 {n_obs} 篇、{n_firms} 家；{len(res)} 个系数 -> {out}', flush=True)


# ---------------------------------------------------------------- 三、比值回归
RATIO_NODES = {'RT': 'rating_strict', 'RV': 'sink_rev_strict'}  # 列名前缀 -> 口径（分子、分母）
RATIO_RAW = ['R_N_SOURCES', 'R_N_PATHS', 'R_LNPATH_RAW', 'R_PATHLEN_RAW']
RATIO_X = {'Q_NSRC': '源点数比值', 'Q_LNPATH': '路径数比值', 'Q_PATHLEN': '平均路径长度比值'}
LOG_X = {'LQ_NSRC': 'ln 源点数比值', 'LQ_NPATH': 'ln 路径条数比值', 'LQ_PATHLEN': 'ln 平均路径长度比值'}
FORMS = {'ratio': RATIO_X, 'log': LOG_X}


def ratio_panel(d, w, dag_dirs=F.DAG_DIRS):
    """两个节点都是终点的研报一行：两个节点的原始指标（RT_*、RV_*）、比值和对数口径的原始值（*_RAW）与缩尾值（只在 base 内有值）、base。"""
    m = d[['report_id', 'code']].copy()
    for pre, sink in RATIO_NODES.items():
        sm = F.sink_frame(d.report_id, sink, dag_dirs)
        m = m.merge(sm[['report_id', 'has_rating'] + RATIO_RAW].rename(columns={c: f'{pre}_{c}' for c in ['has_rating'] + RATIO_RAW}),
                    on='report_id', how='left')
    m = m[m.RT_has_rating.fillna(False) & m.RV_has_rating.fillna(False)].drop(columns=['RT_has_rating', 'RV_has_rating'])
    m = m.merge(w[['report_id', 'base']], on='report_id', how='left').reset_index(drop=True)
    raw = {
        'Q_NSRC': m.RT_R_N_SOURCES / m.RV_R_N_SOURCES, 'Q_LNPATH': m.RT_R_LNPATH_RAW / m.RV_R_LNPATH_RAW,
        'Q_PATHLEN': m.RT_R_PATHLEN_RAW / m.RV_R_PATHLEN_RAW,
        'LQ_NSRC': np.log(m.RT_R_N_SOURCES / m.RV_R_N_SOURCES), 'LQ_NPATH': np.log(m.RT_R_N_PATHS / m.RV_R_N_PATHS),
        'LQ_PATHLEN': np.log(m.RT_R_PATHLEN_RAW / m.RV_R_PATHLEN_RAW)}
    for x, v in raw.items():
        m[x + '_RAW'] = v.astype(float)
        m[x] = np.nan
        m.loc[m.base, x] = R.winsor(m.loc[m.base, x + '_RAW'])  # 在本样本（base 内两个节点都是终点）里缩尾
    return m


def ratio(out_dir, dag_dirs=F.DAG_DIRS, **paths):
    """paths 可传 panel_path、est_dir、fin_path、daily_path（同 dag_forecast.build_panel）。"""
    from src import dag_forecast_tables as T
    out = Path(out_dir)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    d, _, _, lgs = F.build_panel(main_only=False, **paths)
    w = F.winsorize(d)
    m = ratio_panel(d, w, dag_dirs)
    vp = variant_panel(w, lgs)
    ycols = [c for c in vp.columns if c not in ('report_id', 'code')]
    panel = m.merge(vp[['report_id'] + ycols], on='report_id', how='left')
    panel.to_parquet(out / 'ratio_panel.parquet', index=False)
    s = m[m.base]
    print(f'两个节点都是终点：全部 {len(m)} 篇、{m.code.nunique()} 家；base 内 {len(s)} 篇、{s.code.nunique()} 家', flush=True)

    ws = w[w.report_id.isin(s.report_id)].merge(s[['report_id'] + list(RATIO_X) + list(LOG_X)], on='report_id', how='left')
    pans = F.panels(with_nofe=True)
    res = pd.concat([variant_regress(ws, vp, list(xs), pans).assign(form=form) for form, xs in FORMS.items()], ignore_index=True)
    res = res[['variant', 'source', 'Y', 'panel', 'form', 'X', 'coef', 'se', 't', 'p', 'n', 'n_clusters', 'y_mean', 'y_sd']]
    res.to_csv(out / 'ratio_results.csv', index=False, encoding='utf-8-sig')
    for form in FORMS:
        sub = out / 'tables' / ('' if form == 'ratio' else 'log')
        sub.mkdir(parents=True, exist_ok=True)
        for v in VARIANTS:
            for src in SOURCES:
                T.write(sub / f'{v}_{src}.md', T.ratio_table(res, v, src, form))
    print(f'{len(res)} 个系数 -> {out}', flush=True)
    return panel, res
