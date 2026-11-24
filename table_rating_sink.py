"""“投资评级”是严格汇点的研报：DAG 结构指标与分析师预测准确度的最终回归（只出数据文件，不写报告）。

样本    791 篇 DAG 研报里“投资评级”是严格汇点（入度 > 0、出度 = 0）的 606 篇；回归样本再限基础样本 base（F.y_sample）。
X（11 个，一次进一个）
        RT_NSRC、RT_LNPATH、RT_PATHLEN、RT_NNODE、RT_DEPTH   评级子图五指标（D.five_measures(graphs, '投资评级', strict=True) 的 *_RAW）
        LNCHARS = ln(n_chars)                                  n_chars 来自 output_dag/regression/report_panel.parquet 的 n_chars 列
                                                               （清理后正文段落字符数之和，src/concept_dag.py 里 sum(len(p['text']))）
        WG_NSRC、WG_LNPATH、WG_PATHLEN、WG_NNODE、WG_DEPTH   整张图五指标（D.five_measures(graphs, None) 的 *_RAW）
        11 个变量都从原始值出发，在这 606 篇内按 1%/99% 缩尾；回归时在估计样本内标准化（R.fit 里做，去缺失和单例之后）。
Y       PMAFE_FY25（主），PMAFE_FY24、AFE_FY25、AFE_FY24（F.forecast_panel 的缩尾值，基础样本内 1%/99%）。
控制    SIZE_M、BM、MOM、VOL、ROA_T、LNAGE（所有回归统一；oldctrl 块例外，用 F.panels(True) 各面板自带的控制变量）。
固定效应 F.panels(True)：N 无；A 申万行业 + 目标年度；B 公司 + 目标年度；C 公司×目标年度 + 券商（broker_g）。按公司聚类（CR1）。
输出    output_dag/forecast_newdata/rating_table10/：reg_all.csv、sample.parquet、descriptives.csv、corr.csv、checks.txt
用法    C:/Users/hung/projects/.venv/Scripts/python.exe -X utf8 table_rating_sink.py [--version v1|v2]
        v1（默认）即上面的口径；v2 的口径见文件后半部分 v2 段的说明，输出到 output_dag/forecast_newdata/rating_table10_v2/。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src import dag_forecast as F  # noqa: E402
from src import dag_measures as D  # noqa: E402
from src import dag_outcomes_reg as R  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_table10'
REPORT_PANEL = ROOT / 'output_dag' / 'regression' / 'report_panel.parquet'
OLD_RES = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_subgraph' / 'reg_forecast.csv'
SAVED_PANEL = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_strict_n' / 'forecast_panel.parquet'

MEAS = D.MEAS  # NSRC, LNPATH, PATHLEN, NNODE, DEPTH
RT = ['RT_' + m for m in MEAS]
WG = ['WG_' + m for m in MEAS]
XS = RT + ['LNCHARS'] + WG  # 11 个 X（缩尾后的列名；原始值列名加 _RAW）
CTRL = ['SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']
YS = ['PMAFE_FY25', 'PMAFE_FY24', 'AFE_FY25', 'AFE_FY24']
PANS = F.panels(True)  # N、A、B、C
FE_LABEL = {'N': '无', 'A': '申万行业+目标年度', 'B': '公司+目标年度', 'C': '公司×目标年度+券商'}
COLS = ['block', 'Y', 'panel', 'fe', 'fe_label', 'controls', 'spec', 'X', 'term', 'role', 'coef', 'se', 't', 'p',
        'n', 'n_clusters', 'n_singletons', 'adj_r2', 'r2', 'k']

log_lines = []


def log(s=''):
    print(s, flush=True)
    log_lines.append(str(s))


# ---------------------------------------------------------------- 样本和变量
def build_sample():
    d, w = F.forecast_panel('rating_strict', main_only=False)
    graphs = D.load_graphs(list(w.report_id))
    m_rt = D.five_measures(graphs, D.RATING_NODE, strict=True)
    m_wg = D.five_measures(graphs, None)
    log(f'five_measures：评级子图 {len(m_rt)} 篇，整张图 {len(m_wg)} 篇')
    rp = pd.read_parquet(REPORT_PANEL, columns=['report_id', 'n_chars'])

    s = (m_rt[['report_id'] + [m + '_RAW' for m in MEAS]].rename(columns={m + '_RAW': 'RT_' + m + '_RAW' for m in MEAS})
         .merge(m_wg[['report_id'] + [m + '_RAW' for m in MEAS]].rename(columns={m + '_RAW': 'WG_' + m + '_RAW' for m in MEAS}),
                on='report_id', how='left', validate='1:1')
         .merge(rp, on='report_id', how='left', validate='1:1'))
    assert s[[c + '_RAW' for c in RT + WG]].notna().all().all() and s.n_chars.notna().all()
    s['LNCHARS_RAW'] = np.log(s.n_chars.astype(float))
    for x in XS:  # 在这 606 篇内 1%/99% 缩尾
        s[x] = R.winsor(s[x + '_RAW'])

    # 核对：评级子图缩尾值 = five_measures 返回的缩尾值；整张图（在 791 篇内缩尾）与本口径（606 篇内）不同之处计数
    chk = s[['report_id'] + RT].merge(m_rt[['report_id'] + MEAS], on='report_id')
    for m in MEAS:
        assert np.allclose(chk['RT_' + m], chk[m], rtol=0, atol=1e-12), m
    log('核对：RT_ 五指标（606 篇内缩尾）与 five_measures 返回的缩尾值逐篇相同')
    chk = s[['report_id'] + WG].merge(m_wg[['report_id'] + MEAS], on='report_id')
    log('整张图五指标：606 篇内缩尾 vs five_measures 在 791 篇内缩尾，取值不同的篇数 ' +
        ', '.join(f'{m} {int((~np.isclose(chk["WG_" + m], chk[m])).sum())}' for m in MEAS))

    keep = ['report_id', 'code', 'company', 'date', 'broker', 'broker_g', 'sw', 'base', 'TY24', 'TY25',
            'LNHORIZON24', 'LNHORIZON25', 'LNCHARS_R', 'LNNODES_R'] + YS + CTRL
    keep = list(dict.fromkeys(keep))
    s = w[keep].merge(s, on='report_id', how='inner', validate='1:1')
    assert len(s) == len(m_rt) == 606, len(s)
    return d, w, s


def check_saved_panel(w):
    """重建的面板与已存的 rating_strict_n/forecast_panel.parquet 的 Y、控制变量是否一致。"""
    if not SAVED_PANEL.exists():
        log('（已存面板不存在，跳过核对）')
        return
    o = pd.read_parquet(SAVED_PANEL)
    cols = YS + CTRL + ['LNHORIZON24', 'LNHORIZON25', 'LNCHARS_R', 'LNNODES_R', 'base']
    a = w.set_index('report_id')[cols]
    b = o.set_index('report_id').loc[a.index, cols]
    bad = [c for c in cols if not ((a[c] == b[c]) | (a[c].isna() & b[c].isna()) |
                                   (np.isclose(a[c].astype(float), b[c].astype(float), rtol=1e-12, atol=0))).all()]
    log(f'核对：重建面板 vs 已存 forecast_panel.parquet（{len(a)} 篇）的 Y、控制变量、base：' + ('全部一致' if not bad else f'不一致 {bad}'))


# ---------------------------------------------------------------- 回归
def reshape(res, block, spec_name=None):
    """regress_each 的结果 -> reg_all 的列。regress_each 的 X 列是本行系数对应的变量（term），spec 是单独进入时的 X 标签。"""
    if res.empty:
        return res
    r = res.rename(columns={'X': 'term'}).copy()
    r['block'] = block
    r['X'] = spec_name if spec_name is not None else r['spec']
    r['fe_label'] = r.panel.map(FE_LABEL)
    r['spec'] = block + '|' + r.Y + '|' + r.panel + '|' + r.X
    return r


def run_all(s):
    out = []
    xs = {x: x for x in XS}
    # 1. main：PMAFE_FY25，面板 C
    out.append(reshape(F.regress_each(s, xs, ['PMAFE_FY25'], pans={'C': PANS['C']}, controls=CTRL, terms=True), 'main'))
    # 2. ladder：4 Y × N/A/B/C × 11 X
    out.append(reshape(F.regress_each(s, xs, YS, pans=PANS, controls=CTRL, terms=True), 'ladder'))
    # 3. horse / horse2：PMAFE_FY25/24 × B/C × 五个指标
    pbc = {p: PANS[p] for p in ('B', 'C')}
    for m in MEAS:
        for block, cols in (('horse', ['RT_' + m, 'WG_' + m, 'LNCHARS']), ('horse2', ['RT_' + m, 'LNCHARS'])):
            r = F.regress_each(s, {c: c for c in cols}, ['PMAFE_FY25', 'PMAFE_FY24'], pans=pbc, controls=CTRL, joint=True, terms=True)
            out.append(reshape(r, block, '+'.join(cols)))
    # 4. oldctrl：PMAFE_FY25 × N/A/B/C × 评级子图五指标，控制变量用各面板自带的
    out.append(reshape(F.regress_each(s, {x: x for x in RT}, ['PMAFE_FY25'], pans=PANS, controls=None, terms=True), 'oldctrl'))
    res = pd.concat(out, ignore_index=True)[COLS]
    assert not res.duplicated(['spec', 'term']).any()
    return res


# ---------------------------------------------------------------- 自检
def own_ols(s, y, xs, ctrl, fe, cluster='code'):
    """独立实现（不调用 R.fit / R.ols）：去缺失、反复剔除单例、X 标准化、虚拟变量 OLS（lstsq）、CR1 聚类标准误。"""
    cols = list(dict.fromkeys([y] + xs + ctrl + fe + [cluster]))
    t = s[cols].dropna().copy()
    n0 = len(t)
    while fe:
        bad = pd.Series(False, index=t.index)
        for c in fe:
            bad |= t.groupby(c)[c].transform('size') == 1
        if not bad.any():
            break
        t = t[~bad]
    for x in xs:
        t[x] = (t[x] - t[x].mean()) / t[x].std(ddof=1)
    parts = [np.ones((len(t), 1)), t[xs + ctrl].to_numpy(float)]
    for c in fe:
        lev = sorted(t[c].astype(str).unique())
        parts.append((t[c].astype(str).to_numpy()[:, None] == np.array(lev[1:])[None, :]).astype(float))
    X = np.hstack(parts)
    yv = t[y].to_numpy(float)
    b, *_ = np.linalg.lstsq(X, yv, rcond=None)
    e = yv - X @ b
    n, k = X.shape[0], np.linalg.matrix_rank(X)
    XtXi = np.linalg.pinv(X.T @ X)
    g = pd.factorize(t[cluster])[0]
    G = g.max() + 1
    S = np.zeros((G, X.shape[1]))
    np.add.at(S, g, X * e[:, None])
    V = XtXi @ S.T @ S @ XtXi * G / (G - 1) * (n - 1) / (n - k)
    se = np.sqrt(np.diag(V))
    r2 = 1 - e @ e / ((yv - yv.mean()) ** 2).sum()
    return {'coef': b[1:1 + len(xs) + len(ctrl)], 'se': se[1:1 + len(xs) + len(ctrl)], 'n': n, 'G': G, 'r2': r2,
            'adj_r2': 1 - (1 - r2) * (n - 1) / (n - k), 'n_single': n0 - n}


def self_checks(s, res):
    # (a) 每个 Y 的估计样本里目标年度是否常数
    for y in YS:
        ys = F.y_sample(s, y)
        ys = ys[ys[y].notna()]
        log(f'目标年度：{y} 的样本（{len(ys)} 篇）里 ty 取值 {sorted(ys.ty.unique())}')
    # (b) 独立 OLS 复算若干格
    cells = [('main', 'PMAFE_FY25', 'C', ['RT_NSRC'], CTRL), ('main', 'PMAFE_FY25', 'C', ['LNCHARS'], CTRL),
             ('ladder', 'PMAFE_FY24', 'A', ['WG_NNODE'], CTRL), ('ladder', 'AFE_FY25', 'N', ['RT_DEPTH'], CTRL),
             ('ladder', 'AFE_FY24', 'B', ['RT_PATHLEN'], CTRL),
             ('horse', 'PMAFE_FY25', 'C', ['RT_NNODE', 'WG_NNODE', 'LNCHARS'], CTRL),
             ('oldctrl', 'PMAFE_FY25', 'C', ['RT_NNODE'], PANS['C'][1])]
    maxdiff = 0.0
    for block, y, pan, xs, ctrl in cells:
        o = own_ols(F.y_sample(s, y), y, xs, ctrl, PANS[pan][0])
        X = '+'.join(xs)
        r = res[(res.block == block) & (res.Y == y) & (res.panel == pan) & (res.X == X)].set_index('term').loc[xs + ctrl]
        # 相对差（控制变量量纲差异大，pinv 与 lstsq 的数值误差按相对量比较）
        dc = np.max(np.abs(r.coef.to_numpy() - o['coef']) / np.maximum(np.abs(o['coef']), 1e-8))
        ds = np.max(np.abs(r.se.to_numpy() - o['se']) / o['se'])
        meta_ok = (r.n.iloc[0] == o['n']) and (r.n_clusters.iloc[0] == o['G']) and (r.n_singletons.iloc[0] == o['n_single'])
        dr = max(abs(r.r2.iloc[0] - o['r2']), abs(r.adj_r2.iloc[0] - o['adj_r2']))
        maxdiff = max(maxdiff, dc, ds, dr)
        log(f'  独立 OLS 复算 [{block} {y} {pan} {X}]：系数最大相对差 {dc:.1e}，se 最大相对差 {ds:.1e}，R²/调整R² 差 {dr:.1e}，'
            f'N/聚类数/单例 {"一致" if meta_ok else "不一致"}（N={o["n"]}, G={o["G"]}, 单例={o["n_single"]}）')
        assert meta_ok and dc < 1e-6 and ds < 1e-6 and dr < 1e-10
    log(f'独立 OLS 复算 {len(cells)} 格全部一致（最大相对差 {maxdiff:.1e}）')

    # (c) oldctrl 复现 rating_subgraph/reg_forecast.csv（scope == 'all'、Y == 'PMAFE_FY25'）
    old = pd.read_csv(OLD_RES)
    old = old[(old.scope == 'all') & (old.Y == 'PMAFE_FY25')].assign(term=lambda t: 'RT_' + t.X)
    new = res[(res.block == 'oldctrl') & (res.role == 'x')]
    j = new.merge(old[['panel', 'term', 'coef', 'se', 't', 'n', 'n_clusters', 'controls', 'fe']], on=['panel', 'term'],
                  suffixes=('', '_old'), how='outer', indicator=True)
    ok = ((j._merge == 'both').all() and np.allclose(j.coef, j.coef_old, rtol=1e-9, atol=1e-12) and np.allclose(j.se, j.se_old, rtol=1e-9)
          and np.allclose(j.t, j.t_old, rtol=1e-9) and (j.n == j.n_old).all() and (j.n_clusters == j.n_clusters_old).all()
          and (j.controls == j.controls_old).all())
    log(f'oldctrl 复现 reg_forecast.csv：{len(j)} 格（4 面板 × 5 指标），系数/se/t/N/聚类数/控制变量 ' + ('逐格一致' if ok else '有不一致'))
    log(f'  系数最大差 {np.max(np.abs(j.coef - j.coef_old)):.1e}，t 最大差 {np.max(np.abs(j.t - j.t_old)):.1e}')
    if not ok:
        log(j.to_string())
    return j


# ---------------------------------------------------------------- 描述统计、相关
def describe(s):
    rows = []

    def add(sample, version, v, x):
        x = x.dropna()
        q = x.quantile([.1, .5, .9])
        rows.append({'sample': sample, 'version': version, 'var': v, 'n': len(x), 'mean': x.mean(), 'sd': x.std(), 'p10': q.iloc[0],
                     'p50': q.iloc[1], 'p90': q.iloc[2], 'min': x.min(), 'max': x.max()})

    for sample, t in (('rating606', s), ('rating606_base', s[s.base])):
        for x in XS:
            add(sample, 'raw', x, t[x + '_RAW'])
        add(sample, 'raw', 'n_chars', t.n_chars.astype(float))
        for x in XS:
            add(sample, 'winsorized_606', x, t[x])
        for v in YS + CTRL:  # 回归用的值（基础样本内 1%/99% 缩尾）
            add(sample, 'panel_winsorized', v, t[v])
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d, w, s = build_sample()
    check_saved_panel(w)
    log(f'样本：{len(s)} 篇、{s.code.nunique()} 家；其中基础样本 {int(s.base.sum())} 篇、{s[s.base].code.nunique()} 家；'
        + '，'.join(f'{y} 非缺失 {int(s[y].notna().sum())}' for y in YS))

    res = run_all(s)
    res.to_csv(OUT / 'reg_all.csv', index=False, encoding='utf-8-sig')
    log(f'reg_all.csv：{len(res)} 行，{res.spec.nunique()} 个回归；各块回归数 ' +
        ', '.join(f'{b} {res[res.block == b].spec.nunique()}' for b in res.block.unique()))
    exp = {'main': 11, 'ladder': 4 * 4 * 11, 'horse': 2 * 2 * 5, 'horse2': 2 * 2 * 5, 'oldctrl': 20}
    miss = {b: (n, res[res.block == b].spec.nunique()) for b, n in exp.items() if res[res.block == b].spec.nunique() != n}
    log('回归数与设计相符' if not miss else f'回归数与设计不符（期望, 实际）：{miss}')

    self_checks(s, res)

    cols = (['report_id', 'code', 'company', 'date', 'broker', 'broker_g', 'sw', 'base'] + YS + CTRL + ['n_chars']
            + [x + '_RAW' for x in XS] + XS + ['LNCHARS_R', 'LNNODES_R'])
    s[cols].to_parquet(OUT / 'sample.parquet', index=False)
    describe(s).to_csv(OUT / 'descriptives.csv', index=False, encoding='utf-8-sig')
    c = s[XS].corr()
    c.insert(0, 'var', c.index)
    c.to_csv(OUT / 'corr.csv', index=False, encoding='utf-8-sig')
    log('相关系数（缩尾后，606 篇）：RT_m vs WG_m：' + ', '.join(f'{m} {s["RT_" + m].corr(s["WG_" + m]):.3f}' for m in MEAS))
    log('  与 LNCHARS：' + ', '.join(f'{x} {s[x].corr(s.LNCHARS):.3f}' for x in RT + WG))
    log(f'  LNCHARS vs 面板 LNCHARS_R：{s.LNCHARS.corr(s.LNCHARS_R):.4f}；LNCHARS_RAW 与 LNCHARS_R 不同的篇数 '
        f'{int((~np.isclose(s.LNCHARS_RAW, s.LNCHARS_R)).sum())}，LNCHARS 与 LNCHARS_R 不同的篇数 {int((~np.isclose(s.LNCHARS, s.LNCHARS_R)).sum())}')
    (OUT / 'checks.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')


# ================================================================ v2：新口径（输出到 rating_table10_v2/，不碰 v1 的目录）
"""v2 与 v1 的差别
X     概念数改对数：RT_LNNODE = ln(评级子图节点数)、WG_LNNODE = ln(整张图节点数)，替换 RT_NNODE / WG_NNODE；先取对数再在 606 篇内缩尾。
Y     AFEH_FY{t}  = ABSERR{t} / |A2023|（A2023 = load_annual 取 2023 年 1231 行的 np_parent_company_owners_ttm，同面板 A24/A25 的取法）
      AFEHT_FY{t} = ABSERR{t} / |A_T|（T = RPT_YR_T，即面板的 AT）；AFEA_FY{t} = ABSERR{t} / |A{t}|
      以上在全部 791 篇的基础样本（786 篇）内 1%/99% 缩尾，同 F.winsorize 对 PMAFE / AFE 的做法；分母为 0 记缺失。
      RANK_FY{t}    = (同行绝对误差 < 本篇的家数 + 0.5 × 相等的家数) / 同行家数；同行 = PMAFE 的同行（peer_detail，F.Peers.last_each），
                      至少 3 家；不缩尾。
      RANKALL_FY{t} = 同上，同行 = 2024 年内其他券商（ORG_CL 不同）对同公司同目标年度的全部预测，每家取全年最新一条，至少 3 家。
      都只在基础样本内、本篇绝对误差非缺失时有值。PMAFE_FY24/25 保留。
控制  MOM、VOL、ROA_T × 100（百分数），其余不变；列名不变。
"""
OUT_V2 = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_table10_v2'
MEAS2 = ['NSRC', 'LNPATH', 'PATHLEN', 'LNNODE', 'DEPTH']
RT2 = ['RT_' + m for m in MEAS2]
WG2 = ['WG_' + m for m in MEAS2]
XS2 = RT2 + ['LNCHARS'] + WG2
PCT = ['MOM', 'VOL', 'ROA_T']
Y_RATIO = ['AFEH_FY25', 'AFEH_FY24', 'AFEHT_FY25', 'AFEHT_FY24', 'AFEA_FY25', 'AFEA_FY24']  # 缩尾
Y_RANK = ['RANK_FY25', 'RANK_FY24', 'RANKALL_FY25', 'RANKALL_FY24']  # 不缩尾
# 主因变量 AFEA_FY25（分母 = 目标年度自身实际值）；AFEH、AFEHT 只在 ladder 作对照
# AFEA_FY25_X858：AFEA_FY25 缩尾后的值去掉五粮液（000858）的研报（不重新缩尾），只进 ladder
MAIN_Y = 'AFEA_FY25'
X858 = '000858'
YS_LADDER = ['AFEA_FY25', 'AFEA_FY24', 'RANK_FY25', 'RANK_FY24', 'PMAFE_FY25', 'PMAFE_FY24', 'AFEH_FY25', 'AFEH_FY24',
             'AFEHT_FY25', 'AFEHT_FY24', 'RANKALL_FY25', 'RANKALL_FY24', 'AFEA_FY25_X858']
YS_HORSE = ['AFEA_FY25', 'AFEA_FY24', 'RANK_FY25', 'RANK_FY24', 'PMAFE_FY25']
YS_ALL2 = YS_LADDER + ['AFE_FY25', 'AFE_FY24']
OLD_RANK = ROOT / 'output_dag' / 'forecast_newdata' / 'horizon_check' / 'rank_panel.parquet'

# 新因变量的 LNHORIZON / 目标年度列（regress_each → y_sample 按名字查这两个字典）。只在本进程内往字典里加键，不改 src 文件
F.Y_HORIZON['AFEA_FY25_X858'], F.Y_TY['AFEA_FY25_X858'] = 'LNHORIZON25', 'TY25'
for _y in Y_RATIO + Y_RANK:
    F.Y_HORIZON[_y] = 'LNHORIZON' + _y[-2:]
    F.Y_TY[_y] = 'TY' + _y[-2:]


def rank_of(own, peers):
    peers = np.asarray(peers, float)
    return ((peers < own).sum() + 0.5 * (peers == own).sum()) / len(peers)


def new_ys(d, detail, lgs):
    """在未缩尾的全部面板 d 上算新因变量（缩尾前），返回加了列的副本和核对信息。"""
    d = d.copy()
    codes = sorted(set(d.code))
    ann = F.load_annual(F.FINANCIAL, codes)
    # 核对：面板的 A24 / A25 / AT 就是 load_annual 的 A（同一来源、同一取法）
    for col, fy in (('A24', pd.Series(2024, index=d.index)), ('A25', pd.Series(2025, index=d.index)), ('AT', d['T'])):
        ref = [ann.A.get((c, int(y)), np.nan) if pd.notna(y) else np.nan for c, y in zip(d.code, fy)]
        ok = np.allclose(d[col].astype(float), np.array(ref, float), equal_nan=True, rtol=0, atol=0)
        log(f'核对：面板 {col} 与 load_annual 的 A 逐篇相同：{ok}')
        assert ok
    d['A23'] = [ann.A.get((c, 2023), np.nan) for c in d.code]
    base = d.base.astype(bool)
    for t in ('24', '25'):
        ae = d[f'ABSERR{t}']
        for name, den in (('AFEH', d.A23), ('AFEHT', d.AT), ('AFEA', d[f'A{t}'])):
            v = ae / den.abs()
            d[f'{name}_FY{t}'] = v.where(base & (den.abs() > 0)).replace([np.inf, -np.inf], np.nan)
        # RANK：PMAFE 的同行
        det = detail[t].copy()
        g = {rid: x.F.to_numpy(float) for rid, x in det.groupby('report_id')}
        rk, npr, bad = [], [], 0
        for rid, own, a, n0, pe, ok in zip(d.report_id, ae, d[f'A{t}'], d[f'N_PEER{t}'], d[f'PEER_ERR{t}'], base):
            pa = np.abs(g[rid] - a) if rid in g else np.array([])
            if ok and len(pa):
                # 实际值缺失时同行误差均值为 NaN，两边都 NaN 算一致
                bad += (len(pa) != n0) or not np.isclose(pa.mean(), pe, rtol=1e-12, atol=0, equal_nan=True)
            npr.append(len(pa))
            rk.append(rank_of(own, pa) if ok and pd.notna(own) and len(pa) >= F.MIN_PEERS else np.nan)
        assert bad == 0, f'FY{t} 同行与面板 N_PEER/PEER_ERR 不一致 {bad} 篇'
        d[f'RANK_FY{t}'], d[f'N_RANKPEER{t}'] = rk, npr
        # RANKALL：2024 年内其他券商的全年最新一条
        lg = lgs['all']
        lg = lg[(lg.RPT_DT.str[:4] == '2024') & (lg.ty == int('20' + t))]
        last = lg.groupby(['SEC_CD', 'ORG_CL']).F.last().reset_index()  # lg 已按时间排序
        byc = {c: x for c, x in last.groupby('SEC_CD')}
        ra, nall = [], []
        for code, org, own, a, ok in zip(d.code, d.ORG_CL, ae, d[f'A{t}'], base):
            x = byc.get(code)
            pa = np.abs(x.F.to_numpy(float)[x.ORG_CL.to_numpy() != org] - a) if x is not None else np.array([])
            nall.append(len(pa))
            ra.append(rank_of(own, pa) if ok and pd.notna(own) and len(pa) >= F.MIN_PEERS else np.nan)
        d[f'RANKALL_FY{t}'], d[f'N_ALLPEER{t}'] = ra, nall
    log('核对：RANK 的同行家数和同行绝对误差均值与面板 N_PEER24/25、PEER_ERR24/25 逐篇一致')
    return d


def build_sample_v2():
    d, w = F.forecast_panel('rating_strict', main_only=False)
    _, detail, _, lgs = F.build_panel(main_only=False)
    check_saved_panel(w)
    d2 = new_ys(d, detail, lgs)
    # 缩尾：比值类在全部基础样本内 1%/99%（同 F.winsorize）；RANK 不缩尾
    wq = {}
    for y in Y_RATIO:
        raw = d2.loc[d2.base, y]
        lo, hi = raw.quantile([.01, .99])
        wq[y] = (lo, hi, raw.max(), int(raw.notna().sum()))
        d2[y + '_RAW'] = d2[y]
        d2.loc[d2.base, y] = R.winsor(raw)
    # PMAFE / AFE 缩尾后的值取自 w（与 v1 相同）；缩尾前的值取自 d
    for y in ['PMAFE_FY25', 'PMAFE_FY24', 'AFE_FY25', 'AFE_FY24']:
        d2[y + '_RAW'] = d[y]
        d2[y] = w[y].to_numpy()
    assert (d2.report_id.to_numpy() == w.report_id.to_numpy()).all()
    for c in CTRL:  # 控制变量用缩尾后的值（同 v1）
        d2[c] = w[c].to_numpy()

    graphs = D.load_graphs(list(w.report_id))
    m_rt = D.five_measures(graphs, D.RATING_NODE, strict=True)
    m_wg = D.five_measures(graphs, None)
    rp = pd.read_parquet(REPORT_PANEL, columns=['report_id', 'n_chars'])

    def raw5(m, pre):
        o = pd.DataFrame({'report_id': m.report_id})
        for x in ['NSRC', 'LNPATH', 'PATHLEN', 'DEPTH']:
            o[f'{pre}_{x}_RAW'] = m[x + '_RAW'].to_numpy()
        o[f'{pre}_LNNODE_RAW'] = np.log(m['NNODE_RAW'].to_numpy(float))
        o[f'{pre}_NNODE_RAW'] = m['NNODE_RAW'].to_numpy()
        return o

    s = raw5(m_rt, 'RT').merge(raw5(m_wg, 'WG'), on='report_id', how='left', validate='1:1').merge(rp, on='report_id', how='left')
    s['LNCHARS_RAW'] = np.log(s.n_chars.astype(float))
    for x in XS2:
        s[x] = R.winsor(s[x + '_RAW'])
    for m in ['NSRC', 'LNPATH', 'PATHLEN', 'DEPTH']:  # 与 five_measures 缩尾值一致
        assert np.allclose(s['RT_' + m], m_rt.set_index('report_id').loc[s.report_id, m].to_numpy(), rtol=0, atol=1e-12)
    log(f'X：RT_LNNODE 原始值范围 [{s.RT_LNNODE_RAW.min():.3f}, {s.RT_LNNODE_RAW.max():.3f}]，'
        f'WG_LNNODE [{s.WG_LNNODE_RAW.min():.3f}, {s.WG_LNNODE_RAW.max():.3f}]')

    keep = (['report_id', 'code', 'company', 'date', 'broker', 'broker_g', 'ORG_CL', 'sw', 'base', 'RPT_YR_T', 'TY24', 'TY25',
             'LNHORIZON24', 'LNHORIZON25', 'A23', 'AT', 'A24', 'A25', 'ABSERR24', 'ABSERR25', 'N_PEER24', 'N_PEER25',
             'N_RANKPEER24', 'N_RANKPEER25', 'N_ALLPEER24', 'N_ALLPEER25']
            + [y for y in YS_ALL2 if y != 'AFEA_FY25_X858'] + [y + '_RAW' for y in Y_RATIO + ['PMAFE_FY25', 'PMAFE_FY24', 'AFE_FY25', 'AFE_FY24']] + CTRL)
    s = d2[list(dict.fromkeys(keep))].merge(s, on='report_id', how='inner', validate='1:1')
    assert len(s) == 606
    s['AFEA_FY25_X858'] = s.AFEA_FY25.where(s.code != X858)
    for c in CTRL:
        s[c + '_ORIG'] = s[c]
    for c in PCT:
        s[c] = s[c] * 100
    log(f'控制变量：VOL 原值（缩尾后）606 篇均值 {s.VOL_ORIG.mean():.5f}、中位数 {s.VOL_ORIG.median():.5f}（小数形式的日收益标准差），'
        f'MOM 均值 {s.MOM_ORIG.mean():.4f}，ROA_T 均值 {s.ROA_T_ORIG.mean():.4f}；三者 × 100')
    return d2, s, wq


def denominators_report(d2, s, wq):
    b = d2[d2.base]
    sb = s[s.base]
    for lab, t in (('全部基础样本(786)', b), ('606 篇中的基础样本', sb)):
        for col in ('A25', 'A24', 'A23', 'AT'):
            a = t[col]
            log(f'分母 {col} [{lab}]：缺失 {int(a.isna().sum())} 篇；<0 {int((a < 0).sum())} 篇 / {t.code[a < 0].nunique()} 家；'
                f'=0 {int((a == 0).sum())}；|A|<1 亿 {int((a.abs() < 1e8).sum())} 篇 / {t.code[a.abs() < 1e8].nunique()} 家；'
                f'|A|<5 亿 {int((a.abs() < 5e8).sum())} 篇 / {t.code[a.abs() < 5e8].nunique()} 家')
    for y, (lo, hi, mx, n) in wq.items():
        n606 = int(sb[y + '_RAW'].notna().sum())
        log(f'  {y}：缩尾前 {n} 篇（606 中 {n606}），最大值 {mx:.4f}，缩尾下限 {lo:.5f}、上限 {hi:.4f}；'
            f'606 中被顶到上限的 {int((sb[y + "_RAW"] > hi).sum())} 篇')
    for t in ('25', '24'):
        small = sb[(sb[f'A{t}'].abs() < 1e8) | (sb[f'A{t}'] < 0)][['code', 'company', 'date', 'broker', f'A{t}', f'ABSERR{t}',
                                                                  f'AFEA_FY{t}_RAW', f'AFEA_FY{t}']]
        log(f'  606 篇（基础样本）里 A{t} < 0 或 |A{t}| < 1 亿的研报 {len(small)} 篇' + ('：\n' + small.to_string(index=False) if len(small) else ''))
    for t in ('25', '24'):
        top = sb.nlargest(8, f'AFEA_FY{t}_RAW')[['code', 'company', 'date', f'A{t}', f'ABSERR{t}', f'AFEA_FY{t}_RAW', f'AFEA_FY{t}']]
        log(f'  606 篇里 AFEA_FY{t} 缩尾前最大的 8 篇：\n' + top.to_string(index=False))
    wl = s[s.code == X858][['report_id', 'date', 'broker', 'base', 'A24', 'A25', 'ABSERR25', 'AFEA_FY25_RAW', 'AFEA_FY25',
                            'AFEA_FY24', 'PMAFE_FY25', 'RANK_FY25']]
    log(f'五粮液（{X858}）在 606 篇里 {len(wl)} 篇：\n' + wl.to_string(index=False))
    loss = d2[d2.base]
    log(f'亏损公司（全部基础样本）：A25<0 {loss.code[loss.A25 < 0].nunique()} 家，A24<0 {loss.code[loss.A24 < 0].nunique()} 家；'
        f'606 篇基础样本：A25<0 {sb.code[sb.A25 < 0].nunique()} 家，A24<0 {sb.code[sb.A24 < 0].nunique()} 家')


def rank_compare(d2):
    if not OLD_RANK.exists():
        log('（旧 rank_panel 不存在，跳过）')
        return
    o = pd.read_parquet(OLD_RANK)
    j = d2[['report_id', 'RANK_FY24', 'RANK_FY25', 'RANKALL_FY24', 'RANKALL_FY25', 'PMAFE_FY24_RAW', 'PMAFE_FY25_RAW', 'base']].merge(
        o, on='report_id', suffixes=('', '_old'))
    for t in ('24', '25'):
        a, b = j[f'RANK_FY{t}'], j[f'RANK_FY{t}_old']
        m = a.notna() & b.notna()
        log(f'排名口径比较 FY{t}：新 RANK 非缺失 {int(a.notna().sum())}，旧 {int(b.notna().sum())}，共同 {int(m.sum())}；'
            f'Pearson {np.corrcoef(a[m], b[m])[0, 1]:.6f}，最大绝对差 {np.abs(a[m] - b[m]).max():.2e}；'
            f'只有新口径有值 {int((a.notna() & b.isna()).sum())}，只有旧口径有值 {int((a.isna() & b.notna()).sum())}')
        m2 = j[f'RANK_FY{t}'].notna() & j[f'RANKALL_FY{t}'].notna()
        log(f'  RANK vs RANKALL FY{t}：共同 {int(m2.sum())}，Pearson {np.corrcoef(j.loc[m2, f"RANK_FY{t}"], j.loc[m2, f"RANKALL_FY{t}"])[0, 1]:.4f}')


def run_all_v2(s):
    out = []
    xs = {x: x for x in XS2}
    out.append(reshape(F.regress_each(s, xs, [MAIN_Y], pans={'C': PANS['C']}, controls=CTRL, terms=True), 'main'))
    out.append(reshape(F.regress_each(s, xs, YS_LADDER, pans=PANS, controls=CTRL, terms=True), 'ladder'))
    pbc = {p: PANS[p] for p in ('B', 'C')}
    for m in MEAS2:
        for block, cols in (('horse', ['RT_' + m, 'WG_' + m, 'LNCHARS']), ('horse2', ['RT_' + m, 'LNCHARS'])):
            r = F.regress_each(s, {c: c for c in cols}, YS_HORSE, pans=pbc, controls=CTRL, joint=True, terms=True)
            out.append(reshape(r, block, '+'.join(cols)))
    res = pd.concat(out, ignore_index=True)[COLS]
    assert not res.duplicated(['spec', 'term']).any()
    return res


def self_checks_v2(s, res):
    for y in ['AFEA_FY25', 'AFEA_FY24', 'AFEH_FY25', 'RANK_FY25', 'RANKALL_FY24', 'AFEA_FY25_X858']:
        ys = F.y_sample(s, y)
        log(f'目标年度：{y} 样本（{int(ys[y].notna().sum())} 篇）ty 取值 {sorted(ys.loc[ys[y].notna(), "ty"].unique())}')
    cells = [('main', 'AFEA_FY25', 'C', ['RT_NSRC']), ('main', 'AFEA_FY25', 'C', ['WG_LNNODE']), 
             ('ladder', 'AFEH_FY25', 'C', ['RT_LNNODE']), ('ladder', 'AFEA_FY25_X858', 'C', ['RT_NSRC']),
             ('ladder', 'RANK_FY25', 'B', ['RT_LNNODE']), ('ladder', 'AFEA_FY24', 'A', ['LNCHARS']),
             ('ladder', 'RANKALL_FY24', 'N', ['RT_DEPTH']), ('horse', 'RANK_FY25', 'C', ['RT_LNNODE', 'WG_LNNODE', 'LNCHARS']),
             ('horse2', 'AFEA_FY24', 'B', ['RT_LNPATH', 'LNCHARS'])]
    mx = 0.0
    for block, y, pan, xs in cells:
        o = own_ols(F.y_sample(s, y), y, xs, CTRL, PANS[pan][0])
        r = res[(res.block == block) & (res.Y == y) & (res.panel == pan) & (res.X == '+'.join(xs))].set_index('term').loc[xs + CTRL]
        dc = np.max(np.abs(r.coef.to_numpy() - o['coef']) / np.maximum(np.abs(o['coef']), 1e-8))
        ds = np.max(np.abs(r.se.to_numpy() - o['se']) / o['se'])
        dr = max(abs(r.r2.iloc[0] - o['r2']), abs(r.adj_r2.iloc[0] - o['adj_r2']))
        meta = (r.n.iloc[0] == o['n']) and (r.n_clusters.iloc[0] == o['G']) and (r.n_singletons.iloc[0] == o['n_single'])
        mx = max(mx, dc, ds)
        log(f'  独立 OLS [{block} {y} {pan} {"+".join(xs)}]：系数相对差 {dc:.1e}，se 相对差 {ds:.1e}，R² 差 {dr:.1e}，'
            f'N/G/单例 {"一致" if meta else "不一致"}（{o["n"]}/{o["G"]}/{o["n_single"]}）')
        assert meta and dc < 1e-6 and ds < 1e-6 and dr < 1e-10
    log(f'独立 OLS 复算 {len(cells)} 格全部一致（最大相对差 {mx:.1e}）')
    # 换单位前后：main 块用原单位重跑
    s0 = s.copy()
    for c in PCT:
        s0[c] = s0[c + '_ORIG']
    r0 = reshape(F.regress_each(s0, {x: x for x in XS2}, [MAIN_Y], pans={'C': PANS['C']}, controls=CTRL, terms=True), 'main')
    a = res[res.block == 'main'].set_index(['X', 'term'])
    b = r0.set_index(['X', 'term']).loc[a.index]
    dt = np.abs(a.t - b.t).max()
    scale = np.where(a.index.get_level_values('term').isin(PCT), 100.0, 1.0)
    dcoef = np.max(np.abs(a.coef * scale - b.coef) / np.maximum(np.abs(b.coef), 1e-12))
    log(f'换单位核对（main 块 {len(a)} 个系数）：t 最大差 {dt:.1e}；MOM/VOL/ROA_T 系数 × 100 = 原单位系数（最大相对差 {dcoef:.1e}），'
        f'其余系数相同；调整 R² 最大差 {np.abs(a.adj_r2 - b.adj_r2).max():.1e}')


def describe_v2(s):
    rows = []

    def add(sample, version, v, x):
        x = x.dropna()
        q = x.quantile([.01, .1, .25, .5, .75, .9, .99])
        rows.append({'sample': sample, 'version': version, 'var': v, 'n': len(x), 'mean': x.mean(), 'sd': x.std(), 'min': x.min(),
                     'p1': q.iloc[0], 'p10': q.iloc[1], 'p25': q.iloc[2], 'p50': q.iloc[3], 'p75': q.iloc[4], 'p90': q.iloc[5],
                     'p99': q.iloc[6], 'max': x.max()})

    for sample, t in (('rating606', s), ('rating606_base', s[s.base])):
        for x in XS2:
            add(sample, 'raw', x, t[x + '_RAW'])
        add(sample, 'raw', 'n_chars', t.n_chars.astype(float))
        for x in XS2:
            add(sample, 'winsorized_606', x, t[x])
        for y in Y_RATIO + ['PMAFE_FY25', 'PMAFE_FY24', 'AFE_FY25', 'AFE_FY24']:
            add(sample, 'raw', y, t[y + '_RAW'])
        for y in YS_ALL2:
            add(sample, 'regression' if y not in Y_RANK else 'regression(未缩尾)', y, t[y])
        for c in CTRL:
            add(sample, 'regression(MOM/VOL/ROA_T 为 %)', c, t[c])
        for c in ['A23', 'AT', 'A24', 'A25', 'N_RANKPEER25', 'N_RANKPEER24', 'N_ALLPEER25', 'N_ALLPEER24']:
            add(sample, 'raw', c, t[c].astype(float))
    return pd.DataFrame(rows)


def main_v2():
    OUT_V2.mkdir(parents=True, exist_ok=True)
    d2, s, wq = build_sample_v2()
    log(f'样本：{len(s)} 篇、{s.code.nunique()} 家；基础样本 {int(s.base.sum())} 篇；非缺失：' +
        '，'.join(f'{y} {int(s[y].notna().sum())}' for y in YS_LADDER))
    denominators_report(d2, s, wq)
    rank_compare(d2)
    res = run_all_v2(s)
    res.to_csv(OUT_V2 / 'reg_all.csv', index=False, encoding='utf-8-sig')
    exp = {'main': 11, 'ladder': 13 * 4 * 11, 'horse': 5 * 2 * 5, 'horse2': 5 * 2 * 5}
    got = {b: res[res.block == b].spec.nunique() for b in exp}
    log(f'reg_all.csv：{len(res)} 行；回归数 {got}' + ('，与设计相符' if got == exp else f'，与设计不符（期望 {exp}）'))
    self_checks_v2(s, res)

    s.to_parquet(OUT_V2 / 'sample.parquet', index=False)
    describe_v2(s).to_csv(OUT_V2 / 'descriptives.csv', index=False, encoding='utf-8-sig')
    c = s[XS2].corr()
    c.insert(0, 'var', c.index)
    c.to_csv(OUT_V2 / 'corr.csv', index=False, encoding='utf-8-sig')
    sb = s[s.base]
    yc = []
    for meth in ('pearson', 'spearman'):
        m = sb[YS_ALL2].corr(method='pearson') if meth == 'pearson' else sb[YS_ALL2].rank().corr()
        m.insert(0, 'var', m.index)
        m.insert(0, 'method', meth)
        yc.append(m)
    pd.concat(yc, ignore_index=True).to_csv(OUT_V2 / 'y_corr.csv', index=False, encoding='utf-8-sig')
    log('相关（缩尾后，606 篇）：RT_LNNODE–WG_LNNODE ' + f'{s.RT_LNNODE.corr(s.WG_LNNODE):.3f}，RT_LNNODE–RT_NSRC {s.RT_LNNODE.corr(s.RT_NSRC):.3f}，'
        f'WG_LNNODE–LNCHARS {s.WG_LNNODE.corr(s.LNCHARS):.3f}，RT_LNNODE–LNCHARS {s.RT_LNNODE.corr(s.LNCHARS):.3f}')
    (OUT_V2 / 'checks.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')


# ================================================================ v3：EPS 口径的 AFE（输出到 rating_table10_v3/）
"""v3 = v2 的样本、X、控制变量、固定效应，因变量加 EPS 口径的 AFE（文献 Table 10 的算法）。
隐含股本  每条预测 S = PROFIT_Tk / EPS_Tk。EST_COMP_TN 的 EPS 按 SHR_CAP_CHG_DT（股本变动日）那天的股本折算（数据库在之后发生
          送转时会回溯重算 EPS 并把 SHR_CAP_CHG_DT 改到送转之后），所以 S 的股本时点取 c = SHR_CAP_CHG_DT（缺失时用 RPT_DT）。
送转因子  data/market_data/stock_exrights.csv 的 allotted_ps（每股送转股数，10 送 10 = 1.0）；CUM(t) = Π_{除权日 ≤ t}(1 + allotted_ps)，
          k(a→b) = CUM(b) / CUM(a)。配股（rationed_ps）、现金分红不计入。用日行情（总股本 = 总市值 × 复权因子 / 后复权收盘价；
          复权因子跳变 = 前收盘 / 除权参考价）逐个事件核对。
本篇      S_i^d = S_i × k(c_i→d)（本篇隐含股本折到研报日 d 的送转口径；c_i = d 或之间没有送转时就是 S_i），k 取目标年度对应的那个预测期。
          本篇每股误差 = ABSERR_t / S_i^d。
一致预期  沿用 F.consensus_errors 的入选预测（日历年 s 内发布、目标年度 s，每家券商当年最后一条）和 ERR_s；
          E1：S̄_s = median_j(S_j × k(c_j→r_j))（各条预测折到它自己发布日的股本口径），E_s = ERR_s / S̄_s；
          E2：B_s(d) = median_j(S_j × k(c_j→d))（送转调整到研报日 d，增发回购不调整），E_s^adj = ERR_s / B_s(d)；
          E0：ERR_s / S_i^d（全部折到本篇股本）。分母 = T−4..T 上的标准差（ddof=1，至少 3 年，sd > 0，同 STD_PAST）。
因变量    AFEE_FY{t} = (ABSERR_t / S_i^d) / SD(E2)；AFEE1_FY{t} = 同分子 / SD(E1)；AFE_FY{t}（现有 = E0）。
          隐含股本与日行情总股本之比不在 [0.5, 2] 或 EPS 为 0 / 缺失的预测不用（一致预期里剔除该条，本篇则记缺失）。
          AFEE、AFEE1 在全部基础样本内 1%/99% 缩尾。
"""
OUT_V3 = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_table10_v3'
EXRIGHTS = ROOT / 'data' / 'market_data' / 'stock_exrights.csv'
Y_EPS = ['AFEE_FY25', 'AFEE_FY24', 'AFEE1_FY25', 'AFEE1_FY24']
YS_LADDER3 = ['AFEE_FY25', 'AFEE_FY24', 'AFEE1_FY25', 'AFEE1_FY24', 'AFE_FY25', 'AFE_FY24', 'RANK_FY25', 'RANK_FY24',
              'AFEA_FY25', 'AFEA_FY24', 'PMAFE_FY25', 'PMAFE_FY24']
YS_HORSE3 = ['AFEE_FY25', 'AFEE_FY24', 'RANK_FY25']
S_BAND = (0.5, 2.0)  # 隐含股本 / 日行情总股本 的合理范围
for _y in Y_EPS:
    F.Y_HORIZON[_y] = 'LNHORIZON' + _y[-2:]
    F.Y_TY[_y] = 'TY' + _y[-2:]


def load_est_eps(codes, years=None):
    """同 F._load_est 的文件和筛选（IS_VLD == 1、SEC_CD 在样本里），多读 EPS_T1–T3 和 SHR_CAP_CHG_DT。"""
    import glob
    import pyarrow.parquet as pq
    cols = (['SEQ', 'CREAT_TM', 'RPT_ID', 'SEC_CD', 'RPT_MAIN_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'SHR_CAP_CHG_DT']
            + [f'{v}_T{k}' for k in (1, 2, 3) for v in ('PROFIT', 'EPS')])
    fs = [f for y in (years or F.EST_YEARS) for f in sorted(glob.glob(str(F.EST_DIR / str(y) / f'{y}-[0-9][0-9].parquet')))]
    e = pd.concat([pq.read_table(f, columns=cols, filters=[('IS_VLD', '=', 1), ('SEC_CD', 'in', list(codes))]).to_pandas()
                   for f in fs], ignore_index=True)
    e['RPT_DT'] = e.RPT_DT.dt.strftime('%Y%m%d')
    e['SCD'] = pd.to_datetime(e.SHR_CAP_CHG_DT).dt.strftime('%Y%m%d')
    return e


class Shares:
    """日行情总股本（总市值 × 复权因子 / 后复权收盘价）和送转累计因子 CUM。"""

    def __init__(self, codes):
        import pyarrow.parquet as pq
        d = pq.read_table(F.DAILY, columns=['date', 'code', 'close', 'mkcap', 'adj_factor'],
                          filters=[('code', 'in', list(codes)), ('date', '>=', '20160101')]).to_pandas()
        d = d[(d.close > 0) & (d.mkcap > 0) & (d.adj_factor > 0)].sort_values(['code', 'date']).reset_index(drop=True)
        d['shares'] = d.mkcap * d.adj_factor / d.close
        d['raw_close'] = d.close / d.adj_factor
        self.daily = d
        self.by = {c: (g.date.to_numpy(), g.shares.to_numpy()) for c, g in d.groupby('code')}
        x = pd.read_csv(EXRIGHTS, dtype={'code': str, 'date': str})
        self.ex = x[x.code.isin(set(codes))].sort_values(['code', 'date']).reset_index(drop=True)
        sp = self.ex[self.ex.allotted_ps > 0]
        self.cum = {c: (g.date.to_numpy(), np.cumprod(1 + g.allotted_ps.to_numpy())) for c, g in sp.groupby('code')}

    def shares_at(self, code, day):
        if code not in self.by or not isinstance(day, str):
            return np.nan
        dt, s = self.by[code]
        j = np.searchsorted(dt, day, 'right') - 1
        return s[j] if j >= 0 else np.nan

    def cum_at(self, code, day):
        if code not in self.cum:
            return 1.0
        dt, c = self.cum[code]
        j = np.searchsorted(dt, day, 'right') - 1
        return c[j] if j >= 0 else 1.0

    def k(self, code, a, b):
        """a 到 b 的送转累计因子（b 晚于 a 时 ≥ 1，早于 a 时 ≤ 1）。"""
        return self.cum_at(code, b) / self.cum_at(code, a)


def split_events(sh, start='20170101'):
    """送转事件表：stock_exrights 的每个事件（含配股、纯分红），附日行情核对；另加行情识别出的股本跳变。"""
    d = sh.daily
    rows = []
    for (code, g) in d.groupby('code'):
        dt, shr, adj, rc = g.date.to_numpy(), g.shares.to_numpy(), g.adj_factor.to_numpy(), g.raw_close.to_numpy()
        ex = sh.ex[(sh.ex.code == code) & (sh.ex.date >= start)]
        for r in ex.itertuples():
            i1 = np.searchsorted(dt, r.date, 'left')
            if i1 == 0 or i1 >= len(dt):
                continue
            i0 = i1 - 1
            p0 = rc[i0]
            ref = (p0 - r.bonus_ps + r.rationed_px * r.rationed_ps) / (1 + r.allotted_ps + r.rationed_ps)
            i5 = min(i1 + 5, len(dt) - 1)
            rows.append({'code': code, 'ex_date': r.date, 'source': 'exrights', 'allotted_ps': r.allotted_ps, 'rationed_ps': r.rationed_ps,
                         'rationed_px': r.rationed_px, 'bonus_ps': r.bonus_ps, 'prev_day': dt[i0], 'ex_trade_day': dt[i1],
                         'raw_close_prev': p0, 'ref_price': ref, 'adj_ratio': adj[i1] / adj[i0], 'adj_ratio_pred': p0 / ref,
                         'shares_ratio_d0': shr[i1] / shr[i0], 'shares_ratio_d5': shr[i5] / shr[i0]})
    ev = pd.DataFrame(rows)
    ev['is_split'] = ev.allotted_ps > 0
    ev['adj_ok'] = np.isclose(ev.adj_ratio, ev.adj_ratio_pred, rtol=0.01)
    ev['shares_ok'] = np.isclose(ev.shares_ratio_d0, 1 + ev.allotted_ps, rtol=0.01) | np.isclose(ev.shares_ratio_d5, 1 + ev.allotted_ps, rtol=0.01)
    # 行情规则：股本跳变 > 1%
    g = d.copy()
    same = g.code.eq(g.code.shift())
    g['sr'] = np.where(same, g.shares / g.shares.shift(), np.nan)
    g['ar'] = np.where(same, g.adj_factor / g.adj_factor.shift(), np.nan)
    j = g[(g.date >= start) & (np.abs(g.sr - 1) > 0.01)].copy()
    # 送转：股本增加 > 1% 且复权因子同比例跳变（复权因子 / 股本比 在 [0.99, 1.08]，多出的部分是同日现金分红）
    j['rule'] = np.where((j.sr > 1.01) & (j.ar > 1.01) & (j.ar / j.sr).between(0.99, 1.08), '送转（股本与价格同比例）',
                         np.where(np.abs(j.ar - 1) < 0.01, '非送转股本变动（价格未调整）', '其他'))
    sp = ev[ev.is_split]
    near = []
    for r in j.itertuples():
        e = sp[(sp.code == r.code)]
        dd = (pd.to_datetime(e.ex_date) - pd.Timestamp(r.date)).dt.days.abs() if len(e) else pd.Series(dtype=float)
        near.append(bool((dd <= 10).any()))
    j['near_exrights_split'] = near
    jumps = j[['code', 'date', 'sr', 'ar', 'rule', 'near_exrights_split']].rename(columns={'date': 'ex_date', 'sr': 'shares_ratio_d0', 'ar': 'adj_ratio'})
    jumps['source'] = 'daily_rule'
    return ev, jumps


def eps_measures(d2, sh, est):
    """全部面板（791 篇）上算 AFEE / AFEE1 / E0，返回 (加列的面板, 公司-财年表, 核对信息)。"""
    # 预测长表（同 F.to_long 的排序），保留 EPS 和股本时点
    keep = ['RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'CREAT_TM', 'SEQ', 'SCD']
    lg = pd.concat([est[keep + [f'PROFIT_T{k}', f'EPS_T{k}']].rename(columns={f'PROFIT_T{k}': 'F', f'EPS_T{k}': 'EPS'}).assign(k=k)
                    for k in (1, 2, 3)], ignore_index=True)
    lg['ty'] = lg.RPT_YR_T + lg.k
    lg = lg[lg.F.notna()].sort_values(['SEC_CD', 'ty', 'ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ']).reset_index(drop=True)
    c = lg[lg.RPT_DT.str[:4].astype(int) == lg.ty]
    sel = c.groupby(['SEC_CD', 'ty', 'ORG_CL']).tail(1).copy()  # 每家券商当年最后一条（同 consensus_errors）
    sel['c'] = sel.SCD.fillna(sel.RPT_DT)
    sel['S'] = (sel.F / sel.EPS).where((sel.EPS != 0) & sel.EPS.notna())
    sel['S_act_c'] = [sh.shares_at(a, b) for a, b in zip(sel.SEC_CD, sel.c)]
    sel['S_ratio'] = sel.S / sel.S_act_c
    sel['S_ok'] = sel.S.gt(0) & sel.S_ratio.between(*S_BAND)
    sel['k_c_r'] = [sh.k(a, x, y) for a, x, y in zip(sel.SEC_CD, sel.c, sel.RPT_DT)]
    sel['S_r'] = sel.S * sel.k_c_r  # 折到发布日的股本口径
    sel['S_act_r'] = [sh.shares_at(a, b) for a, b in zip(sel.SEC_CD, sel.RPT_DT)]
    ok = sel[sel.S_ok]
    cons = sel.groupby(['SEC_CD', 'ty']).agg(CONS=('F', 'mean'), N_BROKER=('F', 'size'))
    cons['N_EPS_OK'] = ok.groupby(['SEC_CD', 'ty']).size().reindex(cons.index).fillna(0).astype(int)
    cons['SBAR'] = ok.groupby(['SEC_CD', 'ty']).S_r.median().reindex(cons.index)
    cons['SBAR_RAW'] = ok.groupby(['SEC_CD', 'ty']).S.median().reindex(cons.index)
    cons['S_ACT_MED'] = ok.groupby(['SEC_CD', 'ty']).S_act_r.median().reindex(cons.index)
    cons.index.names = ['code', 'fy']
    ann = F.load_annual(F.FINANCIAL, sorted(set(d2.code)))
    cons['A'] = ann.A.reindex(cons.index)
    cons['ERR'] = (cons.CONS - cons.A).abs()
    cons['E1'] = cons.ERR / cons.SBAR
    info = {'sel': sel}
    # 核对：一致预期与 F.consensus_errors 相同
    ref = F.build_panel(main_only=False)[2]
    j = cons[['CONS', 'N_BROKER', 'ERR']].join(ref[['CONS', 'N_BROKER', 'ERR']].add_suffix('_ref'), how='outer')
    info['cons_ok'] = bool(np.allclose(j.CONS, j.CONS_ref, rtol=1e-12, equal_nan=True) and (j.N_BROKER == j.N_BROKER_ref).all()
                           and np.allclose(j.ERR, j.ERR_ref, rtol=1e-12, equal_nan=True))
    info['cons_n'] = len(j)

    # 本篇的预测行（同 _build_panel 的匹配）
    own = est.rename(columns={'RPT_ID': 'RID', 'SEC_CD': 'code'})
    own['RID'] = own.RID.str.upper()
    own = own.sort_values(['RPT_DT', 'CREAT_TM', 'SEQ'], kind='mergesort').drop_duplicates(['RID', 'code'], keep='last')
    own = own[['RID', 'code', 'SCD'] + [f'{v}_T{k}' for k in (1, 2, 3) for v in ('PROFIT', 'EPS')]].add_prefix('own_').rename(
        columns={'own_RID': 'RID', 'own_code': 'code'})
    d = d2.assign(RID=d2.report_id.str.upper()).merge(own, on=['RID', 'code'], how='left')
    assert len(d) == len(d2)
    for k in (1, 2, 3):
        assert np.allclose(d[f'own_PROFIT_T{k}'], d[f'PROFIT_T{k}'], equal_nan=True, rtol=0, atol=0), k
    for k in (1, 2, 3):
        d[f'S_T{k}'] = (d[f'own_PROFIT_T{k}'] / d[f'own_EPS_T{k}']).where(d[f'own_EPS_T{k}'].ne(0) & d[f'own_EPS_T{k}'].notna())
    sk = d[['S_T1', 'S_T2', 'S_T3']]
    info['own_s_spread'] = (sk.max(axis=1) / sk.min(axis=1) - 1)[d.base]
    d['c_own'] = d.own_SCD.fillna(d.date)
    d['S_ACT_D'] = [sh.shares_at(a, b) for a, b in zip(d.code, d.date)]
    d['S_ACT_C'] = [sh.shares_at(a, b) for a, b in zip(d.code, d.c_own)]
    d['K_C_D'] = [sh.k(a, x, y) for a, x, y in zip(d.code, d.c_own, d.date)]
    errs = cons.ERR.to_dict()
    e1 = cons.E1.to_dict()
    sel_ok = {key: (g.S.to_numpy(), g.c.to_numpy()) for key, g in ok.groupby(['SEC_CD', 'ty'])}
    for t in ('24', '25'):
        kk = int('20' + t) - d.RPT_YR_T  # 目标年度对应的预测期
        s_i = pd.Series([r[f'S_T{int(k)}'] if pd.notna(k) and 1 <= k <= 3 else np.nan for k, (_, r) in zip(kk, d.iterrows())], index=d.index)
        bad = ~(s_i.gt(0) & (s_i / d.S_ACT_C).between(*S_BAND))
        d[f'S_I{t}'] = s_i.where(~bad)
        d[f'S_I{t}_BAD'] = bad & s_i.notna()
        d[f'S_ID{t}'] = d[f'S_I{t}'] * d.K_C_D
        d[f'PSE{t}'] = d[f'ABSERR{t}'] / d[f'S_ID{t}']  # 本篇每股误差
    # 各篇的 E0 / E1 / E2 和三个分母
    out = {f'{n}': [] for n in ['SD_E0_25', 'SD_E0_24', 'SD_E1', 'SD_E2', 'NY_E1', 'NY_E2'] + [f'E2_L{j}' for j in range(5)]
           + [f'E1_L{j}' for j in range(5)] + [f'K_L{j}' for j in range(5)]}
    for code, T, day, sid25, sid24, ok_b in zip(d.code, d['T'], d.date, d.S_ID25, d.S_ID24, d.base):
        vals = {n: [] for n in ('E0_25', 'E0_24', 'E1', 'E2', 'K')}
        if pd.notna(T) and ok_b:
            T = int(T)
            for s in range(T - F.PAST_YEARS + 1, T + 1):
                er = errs.get((code, s), np.nan)
                vals['E0_25'].append(er / sid25 if pd.notna(sid25) else np.nan)  # E0：全部折到本篇股本
                vals['E0_24'].append(er / sid24 if pd.notna(sid24) else np.nan)
                vals['E1'].append(e1.get((code, s), np.nan))
                if (code, s) in sel_ok and np.isfinite(er):
                    S, cc = sel_ok[(code, s)]
                    B = np.median(S * np.array([sh.k(code, x, day) for x in cc]))
                    vals['E2'].append(er / B)
                    vals['K'].append(B / cons.SBAR.get((code, s), np.nan))
                else:
                    vals['E2'].append(np.nan)
                    vals['K'].append(np.nan)
        else:
            for n in vals:
                vals[n] = [np.nan] * F.PAST_YEARS

        def sd(v):
            v = np.array([x for x in v if np.isfinite(x)])
            s_ = v.std(ddof=1) if len(v) >= F.MIN_PAST else np.nan
            return (s_ if s_ > 0 else np.nan), len(v)
        out['SD_E0_25'].append(sd(vals['E0_25'])[0])
        out['SD_E0_24'].append(sd(vals['E0_24'])[0])
        for n in ('E1', 'E2'):
            s_, nv = sd(vals[n])
            out[f'SD_{n}'].append(s_)
            out[f'NY_{n}'].append(nv)
        for j in range(5):  # L4 = T−4，L0 = T
            out[f'E2_L{4 - j}'].append(vals['E2'][j])
            out[f'E1_L{4 - j}'].append(vals['E1'][j])
            out[f'K_L{4 - j}'].append(vals['K'][j])
    for n, v in out.items():
        d[n] = v
    for t in ('24', '25'):
        d[f'AFEE0_FY{t}'] = (d[f'PSE{t}'] / d[f'SD_E0_{t}']).where(d.base)
        d[f'AFEE_FY{t}'] = (d[f'PSE{t}'] / d.SD_E2).where(d.base).replace([np.inf, -np.inf], np.nan)
        d[f'AFEE1_FY{t}'] = (d[f'PSE{t}'] / d.SD_E1).where(d.base).replace([np.inf, -np.inf], np.nan)
    return d.drop(columns='RID'), cons.reset_index(), info


def main_v3():
    OUT_V3.mkdir(parents=True, exist_ok=True)
    d2, s, wq = build_sample_v2()
    log_lines.clear()
    log('（v3）样本、X、控制变量同 v2；以下为 EPS 口径的构造和核对')
    codes = sorted(set(d2.code))
    sh = Shares(codes)
    est = load_est_eps(codes)
    log(f'预测表：{len(est)} 行（IS_VLD=1、样本公司、2018–2024 年）；SHR_CAP_CHG_DT 缺失 {int(est.SCD.isna().sum())} 行')

    # 隐含股本的股本时点：与日行情总股本比
    smp = est[est.EPS_T1.ne(0) & est.EPS_T1.notna() & est.PROFIT_T1.notna()].sample(20000, random_state=20261122)
    smp = smp.assign(S=smp.PROFIT_T1 / smp.EPS_T1)
    for lab, col in (('研报日', 'RPT_DT'), ('SHR_CAP_CHG_DT', 'SCD')):
        r = smp.S / np.array([sh.shares_at(a, b) for a, b in zip(smp.SEC_CD, smp[col])])
        log(f'隐含股本 / 日行情总股本（{lab}），随机 20000 条：相差 2% 以内 {float((np.abs(r - 1) < .02).mean()):.3f}，'
            f'5% 以内 {float((np.abs(r - 1) < .05).mean()):.3f}，分位数 p1/p5/p50/p95/p99 = '
            + '/'.join(f'{q:.3f}' for q in r.quantile([.01, .05, .5, .95, .99])))
    scd_after = (pd.to_datetime(est.SCD) > pd.to_datetime(est.RPT_DT))
    log(f'SHR_CAP_CHG_DT 晚于研报日的预测行 {float(scd_after.mean()):.3f}（数据库在之后的股本变动时回溯重算 EPS）')

    ev, jumps = split_events(sh)
    pd.concat([ev, jumps], ignore_index=True).to_csv(OUT_V3 / 'split_events.csv', index=False, encoding='utf-8-sig')
    spv = ev[ev.is_split]
    log(f'送转事件（stock_exrights allotted_ps > 0，样本 {len(codes)} 家，2017 年起，有行情）：{len(spv)} 个、{spv.code.nunique()} 家；'
        f'复权因子跳变与理论值相差 1% 以内 {int(spv.adj_ok.sum())}，总股本在除权日（或其后 5 个交易日内）按 1+送转比例跳变 {int(spv.shares_ok.sum())}；'
        f'配股事件 {int((ev.rationed_ps > 0).sum())} 个（不计入送转因子）；纯现金分红 {int(((ev.allotted_ps == 0) & (ev.rationed_ps == 0)).sum())} 个，'
        f'其复权因子与理论值相差 1% 以内 {int(ev[(ev.allotted_ps == 0) & (ev.rationed_ps == 0)].adj_ok.sum())}')
    jr = jumps.rule.value_counts().to_dict()
    log(f'行情规则（股本日跳变 > 1%）：{len(jumps)} 次，{jr}；其中“送转”规则的 {int((jumps.rule.str.startswith("送转")).sum())} 次里'
        f'在 stock_exrights 送转事件 ±10 天内的 {int((jumps.rule.str.startswith("送转") & jumps.near_exrights_split).sum())} 次；'
        f'送转事件被行情规则识别到的 {int(sum(((jumps.code == r.code) & jumps.rule.str.startswith("送转") & (abs((pd.to_datetime(jumps.ex_date) - pd.Timestamp(r.ex_date)).dt.days) <= 10)).any() for r in spv.itertuples()))}')
    chk = spv.sample(min(8, len(spv)), random_state=7)[['code', 'ex_date', 'allotted_ps', 'bonus_ps', 'raw_close_prev', 'ref_price',
                                                        'adj_ratio', 'adj_ratio_pred', 'shares_ratio_d0', 'shares_ratio_d5']]
    log('送转事件抽查 8 个：\n' + chk.to_string(index=False))
    bad = spv[~(spv.adj_ok & spv.shares_ok)][['code', 'ex_date', 'allotted_ps', 'bonus_ps', 'adj_ratio', 'adj_ratio_pred', 'shares_ratio_d0', 'shares_ratio_d5']]
    log(f'送转事件中行情核对不通过的 {len(bad)} 个：\n' + (bad.to_string(index=False) if len(bad) else '（无）'))

    d3, cons, info = eps_measures(d2, sh, est)
    d3 = d3.copy()
    sel = info['sel']
    log(f'一致预期（公司-财年 {info["cons_n"]} 个）与 F.consensus_errors 的 CONS、N_BROKER、ERR 相同：{info["cons_ok"]}')
    log(f'入选一致预期的预测 {len(sel)} 条：隐含股本缺失（EPS 为 0 或缺失）{int(sel.S.isna().sum())}，'
        f'与 SHR_CAP_CHG_DT 当日总股本之比不在 [{S_BAND[0]}, {S_BAND[1]}] 的 {int((sel.S.notna() & ~sel.S_ok).sum())}；'
        f'股本时点到发布日之间有送转（k ≠ 1）的 {int((sel.k_c_r != 1).sum())}')
    r = (cons.SBAR / cons.S_ACT_MED).dropna()
    log(f'S̄_s（折到发布日口径）/ 发布日日行情总股本中位数：{len(r)} 个公司-财年，2% 以内 {float((np.abs(r - 1) < .02).mean()):.3f}，'
        f'p1/p50/p99 = {r.quantile(.01):.3f}/{r.median():.3f}/{r.quantile(.99):.3f}')
    b = d3[d3.base]
    sp_ = info['own_s_spread']
    log(f'本篇 T1/T2/T3 隐含股本一致性（基础样本 {len(b)} 篇）：最大/最小 − 1 > 0.1% 的 {int((sp_ > 1e-3).sum())} 篇，> 1% 的 {int((sp_ > .01).sum())} 篇，'
        f'最大 {sp_.max():.4%}；取目标年度对应预测期的那个')
    for t in ('25', '24'):
        log(f'本篇 FY{t}：S_i 缺失 {int(b[f"S_I{t}"].isna().sum() - b[f"S_I{t}_BAD"].sum())}（EPS 0 / 缺失或无该预测期），'
            f'隐含股本异常（与股本时点总股本之比不在范围内）{int(b[f"S_I{t}_BAD"].sum())}')
    log(f'本篇股本时点（SHR_CAP_CHG_DT）与研报日之间有送转、S_i 需折算的 {int((b.K_C_D != 1).sum())} 篇；'
        f'S_i^d / 研报日日行情总股本：2% 以内 {float((np.abs(b.S_ID25 / b.S_ACT_D - 1) < .02).mean()):.3f}')
    # E0 核对
    for t in ('25', '24'):
        a_ = f'AFE_FY{t}_RAW'  # 现有 AFE 缩尾前的值（d2 里 AFE_FY{t} 已换成缩尾后的值）
        m = b[f'AFEE0_FY{t}'].notna() & b[a_].notna()
        diff = np.abs(b.loc[m, f'AFEE0_FY{t}'] - b.loc[m, a_])
        log(f'E0 核对 FY{t}：E0 口径与现有 AFE_FY{t}（缩尾前）共同 {int(m.sum())} 篇，最大绝对差 {diff.max():.2e}，最大相对差 '
            f'{(diff / b.loc[m, a_].abs()).max():.2e}；AFE 有值而 E0 缺失 {int((b[a_].notna() & b[f"AFEE0_FY{t}"].isna()).sum())} 篇，'
            f'E0 有值而 AFE 缺失 {int((b[a_].isna() & b[f"AFEE0_FY{t}"].notna()).sum())} 篇')

    # 缩尾（全部基础样本内）
    for y in Y_EPS:
        raw = d3.loc[d3.base, y]
        lo, hi = raw.quantile([.01, .99])
        log(f'{y}：缩尾前 {int(raw.notna().sum())} 篇，最大 {raw.max():.3f}，缩尾下限 {lo:.5f}、上限 {hi:.3f}')
        d3[y + '_RAW'] = d3[y]
        d3.loc[d3.base, y] = R.winsor(raw)
    add = (['report_id'] + Y_EPS + [y + '_RAW' for y in Y_EPS] + ['AFEE0_FY25', 'AFEE0_FY24', 'S_T1', 'S_T2', 'S_T3', 'S_I25', 'S_I24',
           'S_ID25', 'S_ID24', 'c_own', 'K_C_D', 'S_ACT_D', 'S_ACT_C', 'PSE25', 'PSE24', 'SD_E0_25', 'SD_E0_24', 'SD_E1', 'SD_E2', 'NY_E1', 'NY_E2', 'STD_PAST']
           + [f'{n}_L{j}' for n in ('E1', 'E2', 'K') for j in range(5)] + [f'PAST_ERR_L{j}' for j in range(5)])
    s = s.merge(d3[add], on='report_id', how='left', validate='1:1')

    # 窗口内送转、增发回购（606 篇）
    win = []
    for code, T, day in zip(s.code, s.RPT_YR_T, s.date):
        st = f'{int(T) - 4}0101' if pd.notna(T) else None
        if st is None:
            win.append((np.nan, np.nan))
            continue
        kk = sh.k(code, st, day)
        a0, a1 = sh.shares_at(code, st), sh.shares_at(code, day)
        win.append((kk, a1 / (a0 * kk) - 1 if a0 > 0 else np.nan))
    s['K_WINDOW'] = [x[0] for x in win]
    s['NONSPLIT_CHG_WINDOW'] = [x[1] for x in win]
    w6 = s[s.RPT_YR_T.notna()]
    log(f'606 篇里五年窗口（T−4 年初到研报日）内有送转的：{int((w6.K_WINDOW > 1).sum())} 篇 / {w6.code[w6.K_WINDOW > 1].nunique()} 家；'
        f'剔除送转后总股本变动超过 5% 的（增发、回购、可转债转股等）：{int((w6.NONSPLIT_CHG_WINDOW.abs() > .05).sum())} 篇 / '
        f'{w6.code[w6.NONSPLIT_CHG_WINDOW.abs() > .05].nunique()} 家（其中增加 {int((w6.NONSPLIT_CHG_WINDOW > .05).sum())} 篇、减少 {int((w6.NONSPLIT_CHG_WINDOW < -.05).sum())} 篇）')
    sb = s[s.base]
    log('非缺失（606 篇）：' + '，'.join(f'{y} {int(s[y].notna().sum())}' for y in YS_LADDER3))
    for t in ('25', '24'):
        for meth in ('pearson', 'spearman'):
            for lab, tag in (('缩尾后', ''), ('缩尾前', '_RAW')):
                a = sb[[f'AFEE_FY{t}' + tag, f'AFEE1_FY{t}' + tag, f'AFE_FY{t}' + ('_RAW' if tag else '')]].dropna()
                cm = a.corr() if meth == 'pearson' else a.rank().corr()
                log(f'  FY{t} {meth} {lab}（{len(a)} 篇）：AFEE–AFEE1 {cm.iloc[0, 1]:.4f}，AFEE–AFE {cm.iloc[0, 2]:.4f}，AFEE1–AFE {cm.iloc[1, 2]:.4f}')
    cons.to_csv(OUT_V3 / 'eps_consensus_errors.csv', index=False, encoding='utf-8-sig')

    res = []
    xs = {x: x for x in XS2}
    res.append(reshape(F.regress_each(s, xs, ['AFEE_FY25'], pans={'C': PANS['C']}, controls=CTRL, terms=True), 'main'))
    res.append(reshape(F.regress_each(s, xs, YS_LADDER3, pans=PANS, controls=CTRL, terms=True), 'ladder'))
    pbc = {p: PANS[p] for p in ('B', 'C')}
    for m in MEAS2:
        for block, cols in (('horse', ['RT_' + m, 'WG_' + m, 'LNCHARS']), ('horse2', ['RT_' + m, 'LNCHARS'])):
            r_ = F.regress_each(s, {c_: c_ for c_ in cols}, YS_HORSE3, pans=pbc, controls=CTRL, joint=True, terms=True)
            res.append(reshape(r_, block, '+'.join(cols)))
    res = pd.concat(res, ignore_index=True)[COLS]
    assert not res.duplicated(['spec', 'term']).any()
    res.to_csv(OUT_V3 / 'reg_all.csv', index=False, encoding='utf-8-sig')
    exp = {'main': 11, 'ladder': 12 * 4 * 11, 'horse': 3 * 2 * 5, 'horse2': 3 * 2 * 5}
    got = {b_: res[res.block == b_].spec.nunique() for b_ in exp}
    log(f'reg_all.csv：{len(res)} 行；回归数 {got}' + ('，与设计相符' if got == exp else f'，与设计不符（期望 {exp}）'))
    # v2 共有的因变量应与 v2 结果相同
    v2 = OUT_V2 / 'reg_all.csv'
    if v2.exists():
        o = pd.read_csv(v2)
        jj = res[res.block == 'ladder'].merge(o[o.block == 'ladder'], on=['Y', 'panel', 'X', 'term'], suffixes=('', '_v2'))
        log(f'与 v2 共有的 ladder 系数 {len(jj)} 个：系数最大差 {np.abs(jj.coef - jj.coef_v2).max():.1e}，t 最大差 {np.abs(jj.t - jj.t_v2).max():.1e}')
    # 独立复算
    for block, y, pan, xx in [('main', 'AFEE_FY25', 'C', ['RT_NSRC']), ('ladder', 'AFEE1_FY24', 'B', ['WG_LNNODE']),
                              ('ladder', 'AFEE_FY24', 'N', ['LNCHARS']), ('horse', 'AFEE_FY25', 'C', ['RT_LNPATH', 'WG_LNPATH', 'LNCHARS'])]:
        o = own_ols(F.y_sample(s, y), y, xx, CTRL, PANS[pan][0])
        r_ = res[(res.block == block) & (res.Y == y) & (res.panel == pan) & (res.X == '+'.join(xx))].set_index('term').loc[xx + CTRL]
        dc = np.max(np.abs(r_.coef.to_numpy() - o['coef']) / np.maximum(np.abs(o['coef']), 1e-8))
        ds = np.max(np.abs(r_.se.to_numpy() - o['se']) / o['se'])
        meta = (r_.n.iloc[0] == o['n']) and (r_.n_clusters.iloc[0] == o['G']) and (r_.n_singletons.iloc[0] == o['n_single'])
        log(f'  独立 OLS [{block} {y} {pan} {"+".join(xx)}]：系数相对差 {dc:.1e}，se 相对差 {ds:.1e}，N/G/单例 {"一致" if meta else "不一致"}')
        assert meta and dc < 1e-6 and ds < 1e-6

    s.to_parquet(OUT_V3 / 'sample.parquet', index=False)
    ds_ = describe_v2(s)
    rows = []
    for sample, t in (('rating606', s), ('rating606_base', s[s.base])):
        for v, x in [(y + '_RAW', 'raw') for y in Y_EPS] + [(y, 'regression') for y in Y_EPS] + \
                    [(v_, 'raw') for v_ in ['SD_E0_25', 'SD_E1', 'SD_E2', 'STD_PAST', 'S_ID25', 'K_WINDOW', 'NONSPLIT_CHG_WINDOW']]:
            z = t[v].dropna()
            q = z.quantile([.01, .1, .25, .5, .75, .9, .99])
            rows.append({'sample': sample, 'version': x, 'var': v.replace('_RAW', ''), 'n': len(z), 'mean': z.mean(), 'sd': z.std(),
                         'min': z.min(), 'p1': q.iloc[0], 'p10': q.iloc[1], 'p25': q.iloc[2], 'p50': q.iloc[3], 'p75': q.iloc[4],
                         'p90': q.iloc[5], 'p99': q.iloc[6], 'max': z.max()})
    pd.concat([ds_, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT_V3 / 'descriptives.csv', index=False, encoding='utf-8-sig')
    c_ = s[XS2].corr()
    c_.insert(0, 'var', c_.index)
    c_.to_csv(OUT_V3 / 'corr.csv', index=False, encoding='utf-8-sig')
    yy = YS_LADDER3 + ['AFEH_FY25', 'AFEH_FY24', 'RANKALL_FY25', 'RANKALL_FY24']
    yc = []
    for meth in ('pearson', 'spearman'):
        m_ = sb[yy].corr() if meth == 'pearson' else sb[yy].rank().corr()
        m_.insert(0, 'var', m_.index)
        m_.insert(0, 'method', meth)
        yc.append(m_)
    pd.concat(yc, ignore_index=True).to_csv(OUT_V3 / 'y_corr.csv', index=False, encoding='utf-8-sig')
    (OUT_V3 / 'checks.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')


# ================================================================ v4：相对准确度（Hirshleifer, Levi, Lourie & Teoh 2019）
"""v4 = v2/v3 的样本、X、控制变量、固定效应，主因变量换成相对准确度（越大越准）。误差 e = |F − A_t|（归母净利润）。
RA_FY{t}   比较组 = PMAFE 的同行（peer_detail：其他券商、同公司同目标年度、[d−90, d]、每家最新一条、≥3 家）∪ 本篇；
           RA = (median(e_组) − e_i) / SD(e_组)，ddof=1；SD = 0 或组内 < 4 条记缺失。
RAX_FY{t}  中位数、标准差只用同行（不含本篇）。
RAC_FY{t}  同行改为 [d−45, d+45] 内其他券商，每家取离研报日最近的一条（并列取较早；同一天多条取最后一条），≥3 家，组含本篇。
           RA / RAX / RAC 在全部基础样本内 1%/99% 缩尾。
HK_FY{t}   = 100 × (1 − RANK_FY{t})（v2 的排名，同 PMAFE 的同行），不缩尾。
AFEE_FY{t} 读 v3 的 sample.parquet（缩尾后的值和 _RAW）；PMAFE 同前。
"""
OUT_V4 = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_table10_v4'
Y_RA = ['RA_FY25', 'RA_FY24', 'RAX_FY25', 'RAX_FY24', 'RAC_FY25', 'RAC_FY24']
Y_HK = ['HK_FY25', 'HK_FY24']
YS_LADDER4 = ['RA_FY25', 'RA_FY24', 'RAX_FY25', 'RAX_FY24', 'RAC_FY25', 'RAC_FY24', 'HK_FY25', 'HK_FY24',
              'AFEE_FY25', 'AFEE_FY24', 'PMAFE_FY25', 'PMAFE_FY24']
YS_HORSE4 = ['RA_FY25', 'RA_FY24', 'HK_FY25']
CENTER_DAYS = 45
for _y in Y_RA + Y_HK:
    F.Y_HORIZON[_y] = 'LNHORIZON' + _y[-2:]
    F.Y_TY[_y] = 'TY' + _y[-2:]


def ra_measures(d2, detail, lgs):
    """全部面板上算 RA / RAX / RAC（缩尾前），返回加列的副本和核对计数。"""
    d = d2.copy()
    base = d.base.astype(bool)
    chk = {}
    lg = lgs['all']
    lgd = pd.to_datetime(lg.RPT_DT, format='%Y%m%d')
    for t in ('24', '25'):
        ty = int('20' + t)
        g = {rid: x.F.to_numpy(float) for rid, x in detail[t].groupby('report_id')}
        cols = {k: [] for k in ('RA', 'RAX', 'N_GRP', 'MED_GRP', 'SD_GRP', 'MED_PEER', 'SD_PEER', 'RAC', 'N_GRPC', 'MED_GRPC', 'SD_GRPC')}
        bad = 0
        sub = lg[lg.ty == ty]
        sl = {k: v for k, v in sub.groupby('SEC_CD')}
        for rid, code, org, day, own, a, n0, pe, ok in zip(d.report_id, d.code, d.ORG_CL, d.date, d[f'ABSERR{t}'], d[f'A{t}'],
                                                            d[f'N_PEER{t}'], d[f'PEER_ERR{t}'], base):
            pa = np.abs(g[rid] - a) if rid in g else np.array([])
            use = ok and pd.notna(own) and np.isfinite(a)
            if ok and len(pa):
                bad += (len(pa) != n0) or not np.isclose(pa.mean(), pe, rtol=1e-12, atol=0, equal_nan=True)
            # 主口径和 RAX
            if use and len(pa) >= F.MIN_PEERS:
                grp = np.r_[pa, own]
                med, sd = np.median(grp), grp.std(ddof=1)
                medp, sdp = np.median(pa), pa.std(ddof=1)
                cols['RA'].append((med - own) / sd if sd > 0 else np.nan)
                cols['RAX'].append((medp - own) / sdp if sdp > 0 else np.nan)
                cols['N_GRP'].append(len(grp)); cols['MED_GRP'].append(med); cols['SD_GRP'].append(sd)
                cols['MED_PEER'].append(medp); cols['SD_PEER'].append(sdp)
            else:
                for k in ('RA', 'RAX', 'MED_GRP', 'SD_GRP', 'MED_PEER', 'SD_PEER'):
                    cols[k].append(np.nan)
                cols['N_GRP'].append(len(pa) + 1 if use else 0)
            # RAC：以研报日为中心的 ±45 天，其他券商每家离研报日最近的一条
            rac = (np.nan, 0, np.nan, np.nan)
            if use and code in sl:
                x = sl[code]
                dd = (lgd[x.index] - pd.Timestamp(day)).dt.days
                m = (x.ORG_CL != org).to_numpy() & (dd.abs() <= CENTER_DAYS).to_numpy()
                if m.any():
                    c = x[m].assign(dist=dd[m].abs().to_numpy(), pos=np.flatnonzero(m))
                    c = c.sort_values(['ORG_CL', 'dist', 'RPT_DT', 'pos'], ascending=[True, True, True, False])
                    pc = np.abs(c.drop_duplicates('ORG_CL').F.to_numpy(float) - a)
                    if len(pc) >= F.MIN_PEERS:
                        grp = np.r_[pc, own]
                        med, sd = np.median(grp), grp.std(ddof=1)
                        rac = ((med - own) / sd if sd > 0 else np.nan, len(grp), med, sd)
                    else:
                        rac = (np.nan, len(pc) + 1, np.nan, np.nan)
            for k, v in zip(('RAC', 'N_GRPC', 'MED_GRPC', 'SD_GRPC'), rac):
                cols[k].append(v)
        chk[t] = bad
        for k, v in cols.items():
            name = {'RA': f'RA_FY{t}', 'RAX': f'RAX_FY{t}', 'RAC': f'RAC_FY{t}'}.get(k, f'{k}{t}')
            d[name] = v
        for y in (f'RA_FY{t}', f'RAX_FY{t}', f'RAC_FY{t}'):
            d[y] = d[y].where(base).replace([np.inf, -np.inf], np.nan)
    return d, chk


def cluster_mean_t(y, g):
    """均值的 t：普通 t 和按公司聚类（CR1）的 t。"""
    y = np.asarray(y, float)
    n = len(y)
    e = y - y.mean()
    _, idx = np.unique(np.asarray(g), return_inverse=True)
    G = idx.max() + 1
    s = np.bincount(idx, weights=e)
    se_cl = np.sqrt((s ** 2).sum() * G / (G - 1)) / n
    return y.mean() / (y.std(ddof=1) / np.sqrt(n)), y.mean() / se_cl, G


def main_v4():
    OUT_V4.mkdir(parents=True, exist_ok=True)
    d2, s, _ = build_sample_v2()
    log_lines.clear()
    log('（v4）样本、X、控制变量同 v2/v3；以下为相对准确度的构造和核对')
    _, detail, _, lgs = F.build_panel(main_only=False)
    d4, chk = ra_measures(d2, detail, lgs)
    log(f'同行集合核对：RA 用的同行家数和同行误差均值与面板 N_PEER、PEER_ERR 不一致的篇数 FY24 {chk["24"]}、FY25 {chk["25"]}')
    b = d4[d4.base]
    for t in ('25', '24'):
        log(f'FY{t}：基础样本 {len(b)} 篇；RA 非缺失 {int(b[f"RA_FY{t}"].notna().sum())}（PMAFE 非缺失 {int(b[f"PMAFE_FY{t}"].notna().sum())}），'
            f'组内 < 4 条 {int(((b[f"N_GRP{t}"] < 4) & b[f"ABSERR{t}"].notna()).sum())}，SD = 0 {int((b[f"SD_GRP{t}"] == 0).sum())}；'
            f'RAX 非缺失 {int(b[f"RAX_FY{t}"].notna().sum())}；RAC 非缺失 {int(b[f"RAC_FY{t}"].notna().sum())}，'
            f'RAC 组内条数中位数 {b[f"N_GRPC{t}"][b[f"RAC_FY{t}"].notna()].median():.0f}（RA {b[f"N_GRP{t}"][b[f"RA_FY{t}"].notna()].median():.0f}）')
        # RA 与 PMAFE 的同行一致：RA 有值 ⇔ PMAFE 有值（PMAFE 另要求 PEER_ERR > 0）
        log(f'  RA 有值而 PMAFE 缺失 {int((b[f"RA_FY{t}"].notna() & b[f"PMAFE_FY{t}"].isna()).sum())}，'
            f'PMAFE 有值而 RA 缺失 {int((b[f"RA_FY{t}"].isna() & b[f"PMAFE_FY{t}"].notna()).sum())}')
    # 缩尾
    wq = {}
    for y in Y_RA:
        raw = d4.loc[d4.base, y]
        lo, hi = raw.quantile([.01, .99])
        wq[y] = (lo, hi)
        d4[y + '_RAW'] = d4[y]
        d4.loc[d4.base, y] = R.winsor(raw)
    d4 = d4.copy()
    add = ['report_id'] + Y_RA + [y + '_RAW' for y in Y_RA] + [f'{k}{t}' for t in ('25', '24') for k in
                                                              ('N_GRP', 'MED_GRP', 'SD_GRP', 'MED_PEER', 'SD_PEER', 'N_GRPC', 'MED_GRPC', 'SD_GRPC')]
    s = s.merge(d4[add], on='report_id', how='left', validate='1:1')
    for t in ('25', '24'):
        s[f'HK_FY{t}'] = 100 * (1 - s[f'RANK_FY{t}'])
    v3 = pd.read_parquet(OUT_V3 / 'sample.parquet', columns=['report_id', 'AFEE_FY25', 'AFEE_FY24', 'AFEE_FY25_RAW', 'AFEE_FY24_RAW'])
    s = s.merge(v3, on='report_id', how='left', validate='1:1')
    s = s.copy()
    sb = s[s.base]

    # 描述：缩尾前分布、缩尾界、均值 > 0 的检验、极端值
    for y in Y_RA:
        x = sb[y + '_RAW'].dropna()
        q = x.quantile([.01, .1, .5, .9, .99])
        t_, tcl, G = cluster_mean_t(x, sb.loc[x.index, 'code'])
        log(f'{y} 缩尾前（606 篇基础样本 {len(x)} 篇）：均值 {x.mean():.4f}、标准差 {x.std():.4f}、p1/p10/p50/p90/p99 = '
            + '/'.join(f'{v:.3f}' for v in q) + f'、最小 {x.min():.3f}、最大 {x.max():.3f}；缩尾界（全部基础样本）[{wq[y][0]:.3f}, {wq[y][1]:.3f}]；'
            f'均值 t = {t_:.2f}（按公司聚类 t = {tcl:.2f}，{G} 家）；|值| > 3 的 {int((x.abs() > 3).sum())} 篇')
    for t in ('25', '24'):
        x = sb[[f'RA_FY{t}_RAW', f'SD_GRP{t}', f'MED_GRP{t}', f'N_GRP{t}', f'ABSERR{t}', 'company', 'date', f'A{t}']].dropna(subset=[f'RA_FY{t}_RAW'])
        x = x.assign(SD_REL=x[f'SD_GRP{t}'] / x[f'A{t}'].abs())
        log(f'  RA_FY{t}：组内 SD / |实际值| < 1% 的 {int((x.SD_REL < .01).sum())} 篇；其中 |RA| > 3 的 {int(((x.SD_REL < .01) & (x[f"RA_FY{t}_RAW"].abs() > 3)).sum())} 篇；'
            f'|RA| 最大的 5 篇：\n' + x.reindex(x[f'RA_FY{t}_RAW'].abs().sort_values(ascending=False).index).head(5).to_string(index=False))
    log('非缺失（606 篇）：' + '，'.join(f'{y} {int(s[y].notna().sum())}' for y in YS_LADDER4))
    for t in ('25', '24'):
        cols = [f'RA_FY{t}', f'RAX_FY{t}', f'RAC_FY{t}', f'HK_FY{t}', f'PMAFE_FY{t}', f'AFEE_FY{t}']
        a = sb[cols]
        p, sp = a.corr(), a.rank().corr()
        log(f'FY{t} 相关（606 篇基础样本，缩尾后，两两非缺失）：' + '；'.join(
            f'RA–{c.split("_")[0]} {p.loc[cols[0], c]:.3f}/{sp.loc[cols[0], c]:.3f}' for c in cols[1:]) + '（Pearson/Spearman）')

    res = []
    xs = {x: x for x in XS2}
    res.append(reshape(F.regress_each(s, xs, ['RA_FY25'], pans={'C': PANS['C']}, controls=CTRL, terms=True), 'main'))
    res.append(reshape(F.regress_each(s, xs, YS_LADDER4, pans=PANS, controls=CTRL, terms=True), 'ladder'))
    pbc = {p_: PANS[p_] for p_ in ('B', 'C')}
    for m in MEAS2:
        for block, cols in (('horse', ['RT_' + m, 'WG_' + m, 'LNCHARS']), ('horse2', ['RT_' + m, 'LNCHARS'])):
            r_ = F.regress_each(s, {c_: c_ for c_ in cols}, YS_HORSE4, pans=pbc, controls=CTRL, joint=True, terms=True)
            res.append(reshape(r_, block, '+'.join(cols)))
    res = pd.concat(res, ignore_index=True)[COLS]
    assert not res.duplicated(['spec', 'term']).any()
    res.to_csv(OUT_V4 / 'reg_all.csv', index=False, encoding='utf-8-sig')
    exp = {'main': 11, 'ladder': 12 * 4 * 11, 'horse': 3 * 2 * 5, 'horse2': 3 * 2 * 5}
    got = {b_: res[res.block == b_].spec.nunique() for b_ in exp}
    log(f'reg_all.csv：{len(res)} 行；回归数 {got}' + ('，与设计相符' if got == exp else f'，与设计不符（期望 {exp}）'))
    # 与 v3 共有的因变量（AFEE、PMAFE）应相同；HK 与 v2 的 RANK 系数应为 −100 倍、t 相反
    o = pd.read_csv(OUT_V3 / 'reg_all.csv')
    jj = res[res.block == 'ladder'].merge(o[o.block == 'ladder'], on=['Y', 'panel', 'X', 'term'], suffixes=('', '_v3'))
    log(f'与 v3 共有的 ladder 系数 {len(jj)} 个：系数最大差 {np.abs(jj.coef - jj.coef_v3).max():.1e}，t 最大差 {np.abs(jj.t - jj.t_v3).max():.1e}')
    o2 = pd.read_csv(OUT_V2 / 'reg_all.csv')
    o2 = o2[(o2.block == 'ladder') & o2.Y.isin(['RANK_FY25', 'RANK_FY24']) & (o2.role == 'x')].assign(Y=lambda t: t.Y.str.replace('RANK', 'HK'))
    jh = res[(res.block == 'ladder') & (res.role == 'x')].merge(o2, on=['Y', 'panel', 'X'], suffixes=('', '_v2'))
    log(f'HK 与 v2 RANK（{len(jh)} 个 X 系数）：coef + 100 × coef_RANK 最大 {np.abs(jh.coef + 100 * jh.coef_v2).max():.1e}，'
        f't + t_RANK 最大 {np.abs(jh.t + jh.t_v2).max():.1e}')
    for block, y, pan, xx in [('main', 'RA_FY25', 'C', ['RT_NSRC']), ('ladder', 'RAC_FY24', 'B', ['WG_LNNODE']),
                              ('ladder', 'RAX_FY25', 'N', ['LNCHARS']), ('horse', 'RA_FY24', 'C', ['RT_LNPATH', 'WG_LNPATH', 'LNCHARS'])]:
        o_ = own_ols(F.y_sample(s, y), y, xx, CTRL, PANS[pan][0])
        r_ = res[(res.block == block) & (res.Y == y) & (res.panel == pan) & (res.X == '+'.join(xx))].set_index('term').loc[xx + CTRL]
        dc = np.max(np.abs(r_.coef.to_numpy() - o_['coef']) / np.maximum(np.abs(o_['coef']), 1e-8))
        ds = np.max(np.abs(r_.se.to_numpy() - o_['se']) / o_['se'])
        meta = (r_.n.iloc[0] == o_['n']) and (r_.n_clusters.iloc[0] == o_['G']) and (r_.n_singletons.iloc[0] == o_['n_single'])
        log(f'  独立 OLS [{block} {y} {pan} {"+".join(xx)}]：系数相对差 {dc:.1e}，se 相对差 {ds:.1e}，N/G/单例 {"一致" if meta else "不一致"}')
        assert meta and dc < 1e-6 and ds < 1e-6

    s.to_parquet(OUT_V4 / 'sample.parquet', index=False)
    rows = []
    for sample, t in (('rating606', s), ('rating606_base', s[s.base])):
        for v, ver in [(y + '_RAW', 'raw') for y in Y_RA] + [(y, 'regression') for y in Y_RA + Y_HK + ['AFEE_FY25', 'AFEE_FY24', 'PMAFE_FY25', 'PMAFE_FY24']] \
                + [(f'{k}{t_}', 'raw') for t_ in ('25', '24') for k in ('N_GRP', 'SD_GRP', 'N_GRPC')]:
            z = t[v].dropna()
            q = z.quantile([.01, .1, .25, .5, .75, .9, .99])
            rows.append({'sample': sample, 'version': ver, 'var': v.replace('_RAW', ''), 'n': len(z), 'mean': z.mean(), 'sd': z.std(),
                         'min': z.min(), 'p1': q.iloc[0], 'p10': q.iloc[1], 'p25': q.iloc[2], 'p50': q.iloc[3], 'p75': q.iloc[4],
                         'p90': q.iloc[5], 'p99': q.iloc[6], 'max': z.max()})
    pd.concat([describe_v2(s), pd.DataFrame(rows)], ignore_index=True).to_csv(OUT_V4 / 'descriptives.csv', index=False, encoding='utf-8-sig')
    c_ = s[XS2].corr()
    c_.insert(0, 'var', c_.index)
    c_.to_csv(OUT_V4 / 'corr.csv', index=False, encoding='utf-8-sig')
    yy = YS_LADDER4 + ['RANK_FY25', 'RANK_FY24', 'AFEA_FY25', 'AFEA_FY24', 'AFE_FY25', 'AFE_FY24']
    yc = []
    for meth in ('pearson', 'spearman'):
        m_ = sb[yy].corr() if meth == 'pearson' else sb[yy].rank().corr()
        m_.insert(0, 'var', m_.index)
        m_.insert(0, 'method', meth)
        yc.append(m_)
    pd.concat(yc, ignore_index=True).to_csv(OUT_V4 / 'y_corr.csv', index=False, encoding='utf-8-sig')
    (OUT_V4 / 'checks.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')


# ================================================================ v5：Harford, Jiang, Wang & Xie (2019) 的 PMAFE，比较组向后延伸 30 天
"""v5 = v4 的样本、X、控制变量、固定效应。误差 AFE_i = |F_i − A_t|（= ABSERR）。预测行读 2018–2025 年的 EST_COMP_TN（含 2025 年文件）。
比较组    其他券商（ORG_CL ≠ 本篇）、同公司、同目标年度、发布日在 [d − 90, d + 30]（含两端），每家取区间内最新一条
          （按 RPT_DT、CREAT_TM、SEQ 排序取最后），≥ 3 家；本篇券商只用本篇。
PMAFE     (AFE_i − MAFE_i) / MAFE_i，MAFE = 同行 + 本篇的绝对误差均值；MAFE ≤ 0 记缺失。
变体      PMAFEX（MAFE 不含本篇）、PMAFE180（[d−180, d+30]）、PMAFEALL（d+30 及之前全部）、PMAFE0（[d−90, d]，含本篇）；
          PMAFEOLD = v1–v4 的 PMAFE（[d−90, d]、不含本篇）。以上在全部基础样本内 1%/99% 缩尾（PMAFEOLD 沿用面板的缩尾值）。
SCORE     主口径比较组上的 Hong-Kubik 分数 100 − (r − 1)/(N − 1) × 100（r 为本篇在组内的名次，最准为 1，并列取平均名次），不缩尾。
RA        主口径比较组上的 (median − AFE_i) / SD（含本篇，ddof=1），缩尾。HK = 100 × (1 − RANK)（旧比较组）。
DAGE      比较组（含本篇）平均发布日 − 本篇发布日（天）。
"""
OUT_V5 = ROOT / 'output_dag' / 'forecast_newdata' / 'rating_table10_v5'
WINDOWS5 = {'': (90, 30), '180': (180, 30), 'ALL': (None, 30), '0': (90, 0)}  # 名字后缀: (回看天数, 向后天数)
Y_W5 = ['PMAFE_FY25', 'PMAFE_FY24', 'PMAFEX_FY25', 'PMAFEX_FY24', 'PMAFE180_FY25', 'PMAFE180_FY24', 'PMAFEALL_FY25',
        'PMAFEALL_FY24', 'PMAFE0_FY25', 'PMAFE0_FY24', 'RA_FY25', 'RA_FY24']  # 缩尾
YS_LADDER5 = ['PMAFE_FY25', 'PMAFE_FY24', 'PMAFEX_FY25', 'PMAFEX_FY24', 'PMAFE180_FY25', 'PMAFEALL_FY25', 'PMAFE0_FY25',
              'PMAFEOLD_FY25', 'PMAFEOLD_FY24', 'SCORE_FY25', 'SCORE_FY24', 'HK_FY25', 'RA_FY25']
YS_HORSE5 = ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25']
for _y in Y_W5 + ['PMAFEOLD_FY25', 'PMAFEOLD_FY24', 'SCORE_FY25', 'SCORE_FY24', 'HK_FY25', 'HK_FY24']:
    F.Y_HORIZON[_y] = 'LNHORIZON' + _y[-2:]
    F.Y_TY[_y] = 'TY' + _y[-2:]


def long_v5(est):
    """预测长表（同 F.to_long：一行一个预测行 × k，按公司、目标年度、券商、时间排序）。"""
    keep = ['RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'CREAT_TM', 'SEQ']
    lg = pd.concat([est[keep + [f'PROFIT_T{k}']].rename(columns={f'PROFIT_T{k}': 'F'}).assign(k=k) for k in (1, 2, 3)], ignore_index=True)
    lg['ty'] = lg.RPT_YR_T + lg.k
    return lg[lg.F.notna()].sort_values(['SEC_CD', 'ty', 'ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ']).reset_index(drop=True)


def pick_peers(sl, org, day, back, fwd):
    """sl = (日期 int 数组, 券商数组, 起始位置)；其他券商在 [day − back, day + fwd] 内每家最新一条的位置（back=None 不设下界）。"""
    dt, o, a0 = sl
    ts = pd.Timestamp(day)
    hi = int((ts + pd.Timedelta(days=fwd)).strftime('%Y%m%d'))
    m = (o != org) & (dt <= hi)
    if back is not None:
        m &= dt >= int((ts - pd.Timedelta(days=back)).strftime('%Y%m%d'))
    idx = np.flatnonzero(m)
    if not len(idx):
        return idx
    oo = o[idx]
    return a0 + idx[np.r_[oo[1:] != oo[:-1], True]]  # 同一券商的最后一条（切片已按券商、时间排序）


def pmafe_v5(d2, lg):
    """全部面板上算 v5 的因变量（缩尾前），返回 (加列的面板, 新比较组明细 {t: DataFrame}, 交叉核对计数)。"""
    d = d2.copy()
    base = d.base.astype(bool)
    code_a, ty_a = lg.SEC_CD.to_numpy(), lg.ty.to_numpy()
    start = np.flatnonzero(np.r_[True, (code_a[1:] != code_a[:-1]) | (ty_a[1:] != ty_a[:-1])])
    stop = np.r_[start[1:], len(lg)]
    dt_all, org_all, f_all = lg.RPT_DT.astype(int).to_numpy(), lg.ORG_CL.to_numpy(), lg.F.to_numpy(float)
    slices = {(code_a[a], int(ty_a[a])): (dt_all[a:b], org_all[a:b], a) for a, b in zip(start, stop)}
    details, chk = {}, {}
    for t in ('24', '25'):
        ty = int('20' + t)
        out = {k: [] for k in ['PMAFE', 'PMAFEX', 'PMAFE180', 'PMAFEALL', 'PMAFE0', 'SCORE', 'RA', 'N_GRP', 'N_GRP180', 'N_GRPALL',
                               'N_GRP0', 'MAFE', 'MAFEX', 'LATE_SHARE', 'SAME_DAY_SHARE', 'DAGE']}
        det, bad, n_chk = [], 0, 0
        for rid, code, org, day, own, a, n0, pe, ok in zip(d.report_id, d.code, d.ORG_CL, d.date, d[f'ABSERR{t}'], d[f'A{t}'],
                                                            d[f'N_PEER{t}'], d[f'PEER_ERR{t}'], base):
            use = ok and pd.notna(own) and np.isfinite(a) and (code, ty) in slices
            res = {}
            if ok and (code, ty) in slices:  # 旧窗口交叉核对：[d−90, d] 与 Peers.last_each（面板 N_PEER、PEER_ERR）
                p0 = pick_peers(slices[(code, ty)], org, day, 90, 0)
                n_chk += 1
                bad += (len(p0) != n0) or (len(p0) > 0 and not np.isclose(np.abs(f_all[p0] - a).mean(), pe, rtol=1e-12, atol=0,
                                                                           equal_nan=True))
            elif ok:
                bad += n0 != 0
            for suf, (back, fwd) in WINDOWS5.items():
                if not use:
                    break
                p = pick_peers(slices[(code, ty)], org, day, back, fwd)
                res[f'N_GRP{suf}'] = len(p) + 1
                if len(p) < F.MIN_PEERS:
                    continue
                pa = np.abs(f_all[p] - a)
                grp = np.r_[pa, own]
                mafe = grp.mean()
                res[f'PMAFE{suf}'] = (own - mafe) / mafe if mafe > 0 else np.nan
                if suf == '':
                    res['MAFE'], res['MAFEX'] = mafe, pa.mean()
                    res['PMAFEX'] = (own - pa.mean()) / pa.mean() if pa.mean() > 0 else np.nan
                    less, eq = (pa < own).sum(), (pa == own).sum()
                    r = 1 + less + eq / 2  # 本篇的平均名次（最准为 1）
                    res['SCORE'] = 100 - (r - 1) / (len(grp) - 1) * 100
                    sd = grp.std(ddof=1)
                    res['RA'] = (np.median(grp) - own) / sd if sd > 0 else np.nan
                    pd_ = dt_all[p]
                    res['LATE_SHARE'] = float((pd_ > int(day)).mean())
                    res['SAME_DAY_SHARE'] = float((pd_ == int(day)).mean())
                    days = (pd.to_datetime(pd_.astype(str), format='%Y%m%d') - pd.Timestamp(day)).days.to_numpy()
                    res['DAGE'] = float(np.r_[days, 0].mean())
                    det.append(pd.DataFrame({'report_id': rid, 'ty': ty, 'peer_RPT_ID': lg.RPT_ID.to_numpy()[p], 'ORG_CL': org_all[p],
                                             'RPT_DT': dt_all[p].astype(str), 'F': f_all[p], 'abs_err': pa}))
            for k in out:
                out[k].append(res.get(k, 0 if k.startswith('N_GRP') else np.nan))
        for k, v in out.items():
            d[f'{k}_FY{t}'] = v
        for y in [f'{k}_FY{t}' for k in ('PMAFE', 'PMAFEX', 'PMAFE180', 'PMAFEALL', 'PMAFE0', 'SCORE', 'RA', 'DAGE')]:
            d[y] = d[y].where(base).replace([np.inf, -np.inf], np.nan)
        details[t] = pd.concat(det, ignore_index=True)
        chk[t] = (bad, n_chk)
    return d, details, chk


def main_v5():
    OUT_V5.mkdir(parents=True, exist_ok=True)
    d2, s, _ = build_sample_v2()
    log_lines.clear()
    log('（v5）样本、X、控制变量同 v2–v4；以下为新比较组 PMAFE 的构造和核对')
    codes = sorted(set(d2.code))
    est = load_est_eps(codes, years=range(2018, 2026))
    log(f'预测表（IS_VLD=1、样本公司、2018–2025 年）：{len(est)} 行，其中 2025 年文件读入的 RPT_DT 在 2025 年的 {int((est.RPT_DT.str[:4] == "2025").sum())} 行，'
        f'RPT_DT 范围 {est.RPT_DT.min()}–{est.RPT_DT.max()}')
    lg = long_v5(est)
    d5, details, chk = pmafe_v5(d2, lg)
    for t in ('25', '24'):
        log(f'交叉核对 FY{t}：用本脚本的取数在 [d−90, d] 上与面板 N_PEER{t}/PEER_ERR{t}（Peers.last_each）比较，{chk[t][1]} 篇中不一致 {chk[t][0]} 篇')
        details[t].to_parquet(OUT_V5 / f'peer_detail_v5_fy{t}.parquet', index=False)
    b = d5[d5.base]
    cross = b.date > '20241201'
    log(f'研报日晚于 2024-12-01（d + 30 跨到 2025 年）的：全部基础样本 {int(cross.sum())} 篇；606 篇中 {int(s[s.base & (s.date > "20241201")].shape[0])} 篇；'
        f'新比较组明细里 RPT_DT 在 2025 年的同行预测 FY25 {int((details["25"].RPT_DT.str[:4] == "2025").sum())} 条、FY24 {int((details["24"].RPT_DT.str[:4] == "2025").sum())} 条')
    # 缩尾
    wq = {}
    for y in Y_W5:
        raw = d5.loc[d5.base, y]
        wq[y] = tuple(raw.quantile([.01, .99]))
        d5[y + '_RAW'] = d5[y]
        d5.loc[d5.base, y] = R.winsor(raw)
    d5 = d5.copy()
    add = (['report_id'] + Y_W5 + [y + '_RAW' for y in Y_W5] + ['SCORE_FY25', 'SCORE_FY24', 'DAGE_FY25', 'DAGE_FY24']
           + [f'{k}_FY{t}' for t in ('25', '24') for k in ('N_GRP', 'N_GRP180', 'N_GRPALL', 'N_GRP0', 'MAFE', 'MAFEX', 'LATE_SHARE', 'SAME_DAY_SHARE')])
    s = s.rename(columns={'PMAFE_FY25': 'PMAFEOLD_FY25', 'PMAFE_FY24': 'PMAFEOLD_FY24',
                          'PMAFE_FY25_RAW': 'PMAFEOLD_FY25_RAW', 'PMAFE_FY24_RAW': 'PMAFEOLD_FY24_RAW'})
    s = s.merge(d5[add], on='report_id', how='left', validate='1:1')
    for t in ('25', '24'):
        s[f'HK_FY{t}'] = 100 * (1 - s[f'RANK_FY{t}'])
    s = s.copy()
    sb = s[s.base]

    for t in ('25', '24'):
        has = sb[f'PMAFE_FY{t}'].notna()
        x = sb[has]
        g = x[f'N_GRP_FY{t}']
        log(f'FY{t} 新比较组（606 篇基础样本，PMAFE 有值 {int(has.sum())} 篇）：条数（含本篇）均值 {g.mean():.2f}、中位数 {g.median():.0f}、最小 {g.min()}、最大 {g.max()}；'
            f'旧窗口同行家数 N_PEER 均值 {x[f"N_PEER{t}"].mean():.2f}，新比较组同行家数 − 旧 的均值 {(g - 1 - x[f"N_PEER{t}"]).mean():.2f}、中位数 {(g - 1 - x[f"N_PEER{t}"]).median():.0f}、'
            f'最小 {(g - 1 - x[f"N_PEER{t}"]).min()}、最大 {(g - 1 - x[f"N_PEER{t}"]).max()}')
        ls = x[f'LATE_SHARE_FY{t}']
        log(f'  晚于本篇发布的同行占比：均值 {ls.mean():.3f}、中位数 {ls.median():.3f}、p90 {ls.quantile(.9):.3f}、为 0 的 {int((ls == 0).sum())} 篇；'
            f'与本篇同一天的同行占比均值 {x[f"SAME_DAY_SHARE_FY{t}"].mean():.3f}')
        dg = x[f'DAGE_FY{t}']
        log(f'  DAGE（天）：均值 {dg.mean():.2f}、标准差 {dg.std():.2f}、p10/p50/p90 = {dg.quantile(.1):.1f}/{dg.median():.1f}/{dg.quantile(.9):.1f}、'
            f'最小 {dg.min():.1f}、最大 {dg.max():.1f}；DAGE < 0（本篇比组内平均晚）{int((dg < 0).sum())} 篇')
        nn = sb[f'PMAFE0_FY{t}'].notna()
        log(f'  对照：PMAFE0（[d−90, d]）比较组条数中位数 {sb.loc[nn, f"N_GRP0_FY{t}"].median():.0f}；PMAFE180 {sb.loc[sb[f"PMAFE180_FY{t}"].notna(), f"N_GRP180_FY{t}"].median():.0f}；'
            f'PMAFEALL {sb.loc[sb[f"PMAFEALL_FY{t}"].notna(), f"N_GRPALL_FY{t}"].median():.0f}')
    for y in ['PMAFE_FY25', 'PMAFE_FY24', 'PMAFEX_FY25', 'PMAFEX_FY24', 'PMAFE180_FY25', 'PMAFE180_FY24', 'PMAFEALL_FY25', 'PMAFEALL_FY24',
              'PMAFE0_FY25', 'PMAFE0_FY24', 'PMAFEOLD_FY25', 'PMAFEOLD_FY24', 'RA_FY25', 'RA_FY24', 'SCORE_FY25', 'SCORE_FY24', 'HK_FY25', 'HK_FY24']:
        col = y + '_RAW' if y + '_RAW' in sb else y
        x = sb[col].dropna()
        q = x.quantile([.01, .1, .5, .9, .99])
        t_, tcl, G = cluster_mean_t(x, sb.loc[x.index, 'code'])
        wb = f'，缩尾界 [{wq[y][0]:.4f}, {wq[y][1]:.4f}]' if y in wq else ''
        log(f'{y}{"（缩尾前）" if col != y else ""}：{len(x)} 篇，均值 {x.mean():.4f}（t {t_:.2f}，按公司聚类 t {tcl:.2f}，{G} 家），标准差 {x.std():.4f}，'
            f'p1/p10/p50/p90/p99 = ' + '/'.join(f'{v:.3f}' for v in q) + f'，最小 {x.min():.3f}，最大 {x.max():.3f}{wb}')
    for t in ('25', '24'):
        cols = [f'PMAFE_FY{t}', f'PMAFEOLD_FY{t}', f'PMAFEX_FY{t}', f'PMAFE0_FY{t}', f'SCORE_FY{t}', f'HK_FY{t}', f'RA_FY{t}']
        a = sb[cols]
        p, sp = a.corr(), a.rank().corr()
        log(f'FY{t} 相关（606 篇基础样本，回归用值，两两非缺失；Pearson/Spearman）：' + '；'.join(
            f'PMAFE–{c.rsplit("_", 1)[0]} {p.loc[cols[0], c]:.3f}/{sp.loc[cols[0], c]:.3f}' for c in cols[1:]))
    log('非缺失（606 篇）：' + '，'.join(f'{y} {int(s[y].notna().sum())}' for y in YS_LADDER5))

    res = []
    xs = {x: x for x in XS2}
    res.append(reshape(F.regress_each(s, xs, ['PMAFE_FY25'], pans={'C': PANS['C']}, controls=CTRL, terms=True), 'main'))
    res.append(reshape(F.regress_each(s, xs, YS_LADDER5, pans=PANS, controls=CTRL, terms=True), 'ladder'))
    pbc = {p_: PANS[p_] for p_ in ('B', 'C')}
    for m in MEAS2:
        for block, cols in (('horse', ['RT_' + m, 'WG_' + m, 'LNCHARS']), ('horse2', ['RT_' + m, 'LNCHARS'])):
            r_ = F.regress_each(s, {c_: c_ for c_ in cols}, YS_HORSE5, pans=pbc, controls=CTRL, joint=True, terms=True)
            res.append(reshape(r_, block, '+'.join(cols)))
    res.append(reshape(F.regress_each(s, xs, ['PMAFE_FY25', 'SCORE_FY25'], pans=pbc, controls=CTRL + ['DAGE_FY25'], terms=True), 'dage'))
    res.append(reshape(F.regress_each(s, {x: x for x in RT2}, ['PMAFE_FY25', 'SCORE_FY25'], pans=pbc,
                                      controls=CTRL + ['DAGE_FY25', 'LNCHARS'], terms=True), 'dage2'))
    res = pd.concat(res, ignore_index=True)[COLS]
    assert not res.duplicated(['spec', 'term']).any()
    res.to_csv(OUT_V5 / 'reg_all.csv', index=False, encoding='utf-8-sig')
    exp = {'main': 11, 'ladder': 13 * 4 * 11, 'horse': 3 * 2 * 5, 'horse2': 3 * 2 * 5, 'dage': 2 * 2 * 11, 'dage2': 2 * 2 * 5}
    got = {b_: res[res.block == b_].spec.nunique() for b_ in exp}
    log(f'reg_all.csv：{len(res)} 行；回归数 {got}' + ('，与设计相符' if got == exp else f'，与设计不符（期望 {exp}）'))
    # 对照：PMAFEOLD / HK 与 v4 的 PMAFE / HK 相同
    o = pd.read_csv(OUT_V4 / 'reg_all.csv')
    o = o[(o.block == 'ladder') & o.Y.isin(['PMAFE_FY25', 'PMAFE_FY24', 'HK_FY25'])].assign(Y=lambda t: t.Y.str.replace('PMAFE_', 'PMAFEOLD_'))
    jj = res[res.block == 'ladder'].merge(o, on=['Y', 'panel', 'X', 'term'], suffixes=('', '_v4'))
    log(f'PMAFEOLD、HK_FY25 与 v4 的 PMAFE、HK 系数 {len(jj)} 个：系数最大差 {np.abs(jj.coef - jj.coef_v4).max():.1e}，t 最大差 {np.abs(jj.t - jj.t_v4).max():.1e}')
    for block, y, pan, xx, ctrl in [('main', 'PMAFE_FY25', 'C', ['RT_NSRC'], CTRL), ('ladder', 'SCORE_FY24', 'B', ['WG_LNNODE'], CTRL),
                                    ('ladder', 'PMAFEALL_FY25', 'N', ['LNCHARS'], CTRL),
                                    ('dage', 'PMAFE_FY25', 'C', ['RT_LNNODE'], CTRL + ['DAGE_FY25']),
                                    ('dage2', 'SCORE_FY25', 'B', ['RT_NSRC'], CTRL + ['DAGE_FY25', 'LNCHARS']),
                                    ('horse', 'PMAFE_FY24', 'C', ['RT_LNPATH', 'WG_LNPATH', 'LNCHARS'], CTRL)]:
        o_ = own_ols(F.y_sample(s, y), y, xx, ctrl, PANS[pan][0])
        r_ = res[(res.block == block) & (res.Y == y) & (res.panel == pan) & (res.X == '+'.join(xx))].set_index('term').loc[xx + ctrl]
        dc = np.max(np.abs(r_.coef.to_numpy() - o_['coef']) / np.maximum(np.abs(o_['coef']), 1e-8))
        ds = np.max(np.abs(r_.se.to_numpy() - o_['se']) / o_['se'])
        meta = (r_.n.iloc[0] == o_['n']) and (r_.n_clusters.iloc[0] == o_['G']) and (r_.n_singletons.iloc[0] == o_['n_single'])
        log(f'  独立 OLS [{block} {y} {pan} {"+".join(xx)}]：系数相对差 {dc:.1e}，se 相对差 {ds:.1e}，N/G/单例 {"一致" if meta else "不一致"}')
        assert meta and dc < 1e-6 and ds < 1e-6
    # 手算一篇
    rid = sb[sb.PMAFE_FY25.notna()].report_id.iloc[0]
    r0 = s.set_index('report_id').loc[rid]
    dd = details['25'][details['25'].report_id == rid]
    grp = np.r_[dd.abs_err.to_numpy(), r0.ABSERR25]
    log(f'  手算一篇（{rid}）：同行 {len(dd)} 家，PMAFE = {(r0.ABSERR25 - grp.mean()) / grp.mean():.6f}，表中缩尾前 {r0.PMAFE_FY25_RAW:.6f}；'
        f'同行日期 {dd.RPT_DT.min()}–{dd.RPT_DT.max()}，研报日 {r0.date}')

    s.to_parquet(OUT_V5 / 'sample.parquet', index=False)
    rows = []
    for sample, t in (('rating606', s), ('rating606_base', s[s.base])):
        for v, ver in [(y + '_RAW', 'raw') for y in Y_W5] + [(y, 'regression') for y in Y_W5 + ['PMAFEOLD_FY25', 'PMAFEOLD_FY24', 'SCORE_FY25',
                                                                                              'SCORE_FY24', 'HK_FY25', 'HK_FY24']] \
                + [(f'{k}_FY{t_}', 'raw') for t_ in ('25', '24') for k in ('N_GRP', 'LATE_SHARE', 'SAME_DAY_SHARE', 'MAFE')] \
                + [(f'DAGE_FY{t_}', 'raw') for t_ in ('25', '24')]:
            z = t[v].dropna()
            q = z.quantile([.01, .1, .25, .5, .75, .9, .99])
            rows.append({'sample': sample, 'version': ver, 'var': v.replace('_RAW', ''), 'n': len(z), 'mean': z.mean(), 'sd': z.std(),
                         'min': z.min(), 'p1': q.iloc[0], 'p10': q.iloc[1], 'p25': q.iloc[2], 'p50': q.iloc[3], 'p75': q.iloc[4],
                         'p90': q.iloc[5], 'p99': q.iloc[6], 'max': z.max()})
    s_d = s.rename(columns={'PMAFEOLD_FY25': 'PMAFE_FY25', 'PMAFEOLD_FY24': 'PMAFE_FY24', 'PMAFEOLD_FY25_RAW': 'PMAFE_FY25_RAW',
                            'PMAFEOLD_FY24_RAW': 'PMAFE_FY24_RAW'}).loc[:, lambda t: ~t.columns.duplicated()]
    dv2 = describe_v2(s_d)
    dv2 = dv2[~dv2['var'].str.startswith('PMAFE')]  # 旧 PMAFE 在下面以 PMAFEOLD 名义给出
    pd.concat([dv2, pd.DataFrame(rows)], ignore_index=True).to_csv(OUT_V5 / 'descriptives.csv', index=False, encoding='utf-8-sig')
    c_ = s[XS2].corr()
    c_.insert(0, 'var', c_.index)
    c_.to_csv(OUT_V5 / 'corr.csv', index=False, encoding='utf-8-sig')
    yy = YS_LADDER5 + ['PMAFE180_FY24', 'PMAFEALL_FY24', 'PMAFE0_FY24', 'HK_FY24', 'RA_FY24', 'AFEA_FY25', 'AFE_FY25', 'DAGE_FY25', 'DAGE_FY24']
    yc = []
    for meth in ('pearson', 'spearman'):
        m_ = sb[yy].corr() if meth == 'pearson' else sb[yy].rank().corr()
        m_.insert(0, 'var', m_.index)
        m_.insert(0, 'method', meth)
        yc.append(m_)
    pd.concat(yc, ignore_index=True).to_csv(OUT_V5 / 'y_corr.csv', index=False, encoding='utf-8-sig')
    (OUT_V5 / 'checks.txt').write_text('\n'.join(log_lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--version', choices=['v1', 'v2', 'v3', 'v4', 'v5'], default='v1',
                    help='v1 → rating_table10/；v2–v5 → rating_table10_v2/ … _v5/')
    ver = ap.parse_args().version
    {'v1': main, 'v2': main_v2, 'v3': main_v3, 'v4': main_v4, 'v5': main_v5}[ver]()
