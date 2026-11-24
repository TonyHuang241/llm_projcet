"""概念 DAG 结构指标（源汇比、唯一有向路径数、平均路径长度）与 2025 年股价、财务指标的回归：
按 docs/DAG结构指标与2025年股价财务回归方案.md 构造公司层面、研报层面、季度面板，做 M1（公司截面，主）、M2（研报层面）、
M3（季度财务面板）三个模型和稳健性，写出 output_dag/regression/ 下的全部结果。由根目录 regress_dag_outcomes.py 调用。
变量构造在 src/dag_outcomes_vars.py，估计在 src/dag_outcomes_reg.py。

输出（<输出目录>/）：
  firm_panel.parquet、report_panel.parquet、quarter_panel.parquet   回归用的面板（连续变量已按 1%/99% 缩尾）；
  *_raw.parquet                                                      同样的面板，未缩尾（核对用）
  market_returns.parquet                                             市值加权市场组合的周收益、日收益（核对用）
  descriptives.csv、correlations.csv、reg_results.csv、回归结果.md
  reg_coefs_full.csv                                                 M1/M2/M3 main 口径（单独、联合）的全部系数（含控制变量）
  rank_ic.csv、group_sort.csv、influence.csv                          M1 主样本的秩相关、按 X 三等分、逐一剔除
  tables/t01_*.md … t19_*.md                                         论文式表格片段（src/dag_outcomes_tables.py）
  fe_r2_decomposition.csv、tables/s_fe_r2.md                          研报层面 DAG 指标对月份/行业/券商固定效应的 R² Shapley 分解
  fe_r2_decomposition_firm.csv、tables/s_fe_r2_firm.md                再加公司固定效应的四组分解（csv 里另有研报不少于 2 篇的公司子样本）
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src import dag_outcomes_robust as RB
from src import dag_outcomes_tables as T
from src import dag_outcomes_vars as V

DAG_X = ['RATIO', 'LNPATH', 'PATHLEN']
X_LABEL = {'RATIO': '源汇比', 'LNPATH': 'ln(1+路径数)', 'PATHLEN': '平均路径长度'}
# 公司层面被解释变量（2025）及其上年值（Y24）；None 表示不加（CAR 类，或上年值已是常规控制变量、或和常规控制变量完全共线）
FIRM_Y = {
    'SYNCH': 'SYNCH24', 'SYNCH_SW': 'SYNCH_SW24', 'SYNCH_MKT': 'SYNCH_MKT24', 'NCSKEW': 'NCSKEW24', 'DUVOL': 'DUVOL24',
    'IVOL': 'IVOL24', 'SIGMA': None, 'RET': None, 'BHAR': None,
    'CAR_AR24_1': None, 'CAR_AR24_5': None, 'ABSCAR_AR24': None, 'CAR_AR25_1': None, 'CAR_AR25_5': None, 'ABSCAR_AR25': None,
    'TURN': None, 'ILLIQ': 'ILLIQ24',
    'ROA': None, 'ROE': 'ROE24', 'DROA': 'DROA24', 'REVG': 'REVG24', 'NPG': 'NPG24', 'GM': 'GM24', 'CFO': 'CFO24',
    'ASSETG': 'ASSETG24', 'DLEV': 'DLEV24', 'RD': 'RD24', 'EARNVOL': 'EARNVOL24',
    'NPM': 'NPM24', 'ATO': 'ATO24', 'OPG': 'OPG24', 'ACC': 'ACC24'}
ANNUAL_FIN = ['ROA', 'ROE', 'DROA', 'REVG', 'NPG', 'GM', 'CFO', 'ASSETG', 'DLEV', 'RD', 'EARNVOL', 'NPM', 'ATO', 'OPG', 'ACC']
# 表格里的顺序和分组（BHAR 在截面里和 RET 完全等价：照跑、留在 csv 里，表格不列）
PRICE_GROUPS = [('同步性', ['SYNCH', 'SYNCH_SW', 'SYNCH_MKT']), ('崩盘风险', ['NCSKEW', 'DUVOL']), ('波动', ['IVOL', 'SIGMA']),
                ('回报', ['RET', 'CAR_AR24_1', 'CAR_AR24_5', 'ABSCAR_AR24', 'CAR_AR25_1', 'CAR_AR25_5', 'ABSCAR_AR25']),
                ('流动性', ['TURN', 'ILLIQ'])]
FIN_GROUPS = [('盈利能力', ['ROA', 'ROE', 'NPM', 'GM', 'ATO']), ('成长与业绩变化', ['DROA', 'REVG', 'NPG', 'OPG', 'ASSETG']),
              ('现金流与盈余质量', ['CFO', 'ACC', 'EARNVOL']), ('资本结构与投入', ['DLEV', 'RD'])]
PRICE_Y = [y for _, vs in PRICE_GROUPS for y in vs]
FIN_Y = [y for _, vs in FIN_GROUPS for y in vs]
Y_LABEL = {'CAR_AR24_1': 'CAR_AR24[-1,1]', 'CAR_AR24_5': 'CAR_AR24[-1,5]', 'CAR_AR25_1': 'CAR_AR25[-1,1]', 'CAR_AR25_5': 'CAR_AR25[-1,5]'}
QUARTER_Y = ['ROA_Q', 'REVG_Q', 'NPG_Q', 'GM_Q', 'CFO_Q', 'SUE_Q']
FIRM_CTRL = ['SIZE', 'LEV', 'BM', 'ROA24', 'TURN24', 'RET24', 'SIGMA24', 'LNCOV']  # 公司控制变量
CONTROLS = FIRM_CTRL + ['LNNODES', 'LNCHARS', 'LNNDAG']  # + DAG/文本控制（M1、M3）
REPORT_CONTROLS = FIRM_CTRL + ['LNNDAG', 'LNCHARS_R', 'LNNODES_R']  # + DAG/文本控制（M2）
MIN_BROKER = 5  # M2：样本里不足这么多篇的券商并成“其他”
# 逐步加控制：c1 只有固定效应；c2 + 公司控制；c3 + DAG/文本控制（M1 里就是 noY24）；main + 上年同一指标 Y24
STEP_OF = {'c1': 'c1', 'c2': 'c2', 'c3': 'c3', 'noY24': 'c3', 'main': 'main'}
FE_DECOMP = ['month', 'sw', 'broker_g']  # 研报层面 DAG 指标 R² 分解的三组固定效应：发布月份、申万一级、券商（不足 5 篇并成“其他”）


# ---------------------------------------------------------------- 面板
def build_panels(sample_path, industry_path, dag_dirs, daily_path, fin_path, reports_meta_path):
    """构造三个面板（未缩尾）和核对信息。"""
    t0 = time.time()
    info = {}
    sample = pd.read_csv(sample_path, encoding='utf-8-sig', dtype=str)
    sample['code'] = sample.stock_code.str[:6]
    codes = sorted(sample.code)
    ind = V.load_industry(industry_path)
    ind_s = ind.set_index('code')
    info['industry_rows'] = len(ind)
    info['sample_no_csrc'] = sorted(c for c in codes if c not in ind_s.index or pd.isna(ind_s.csrc.get(c)))
    info['sample_no_sw'] = sorted(c for c in codes if c not in ind_s.index or pd.isna(ind_s.sw.get(c)))

    rep = V.load_dag(dag_dirs, set(codes))
    info['n_reports'] = len(rep)
    info['n_reports_by_src'] = rep.src.value_counts().sort_index().to_dict()
    info['n_reports_unified'] = int(rep.unified.sum())
    info['n_reports_no_edge'] = int((rep.n_edges == 0).sum())
    fdag = V.firm_dag(rep)
    print(f'DAG：{len(rep)} 篇研报、{rep.code.nunique()} 家公司（{time.time() - t0:.0f} 秒）', flush=True)

    d = V.load_daily(daily_path, '20231101', '20260531')
    t, mkt_d = V.daily_returns(d)
    week_of, week_year = V.make_weeks(d.date)
    wk = V.weekly_returns(t, week_of)
    all_weeks = pd.Index(sorted(set(week_of.values())), name='week')
    m = V.portfolio_sums(wk)['ALL']
    mkt_w = (m[0] / m[1]).reindex(all_weeks)
    csrc_map = ind.dropna(subset=['csrc']).set_index('code').csrc.to_dict()
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    ind_sums = {'csrc': V.portfolio_sums(wk, csrc_map), 'sw': V.portfolio_sums(wk, sw_map)}
    info['n_stocks_2025'] = int(d[(d.date >= '20250101') & (d.date <= '20251231')].code.nunique())
    traded25 = set(t[(t.date >= '20250101') & (t.date <= '20251231')].code)
    info['n_stocks_2025_no_csrc'] = sum(1 for c in traded25 if c not in csrc_map)
    print(f'行情：{len(d)} 行、{d.code.nunique()} 只股票、{len(wk)} 个股票周（{time.time() - t0:.0f} 秒）', flush=True)

    wm = V.stock_week_measures(wk, codes, mkt_w, ind_sums, {'csrc': csrc_map, 'sw': sw_map}, week_year, [2024, 2025])
    dm, mkt_bh = V.stock_day_measures(t, d, mkt_d, codes, [2024, 2025])
    info['mkt_bh'] = mkt_bh
    info['n_weeks_market'] = {y: int((week_year == y).sum()) for y in (2024, 2025)}
    mw = pd.DataFrame({'week': all_weeks, 'year': week_year.reindex(all_weeks).to_numpy(), 'mkt_ret': mkt_w.to_numpy()})
    mw['week_last_date'] = mw.week.map(pd.Series(list(week_of.keys()), index=list(week_of.values())).groupby(level=0).max())
    info['mkt_w_bh'] = {y: float((1 + mw.mkt_ret[mw.year == y]).prod() - 1) for y in (2024, 2025)}
    market = {'weekly': mw, 'daily': mkt_d.rename('mkt_ret').reset_index()}

    fin, info['fin_dup'] = V.load_fin(fin_path, codes)
    fin_codes = {c for c in codes if csrc_map.get(c) == V.FIN_INDUSTRY}
    info['n_fin'] = len(fin_codes)
    ann = {y: {c: v for c, v in V.fin_get(fin, codes, f'{y}1231', 'date').items() if isinstance(v, str)} for y in (2024, 2025)}
    car = {y: V.event_car(t, mkt_d, codes, ann[y]).set_index('code') for y in (2024, 2025)}
    a = {y: V.annual_fin(fin, codes, y, fin_codes).set_index('code') for y in (2024, 2025)}
    mk = {y: V.last_traded_value(t, codes, f'{y}1231', 'mkcap') for y in (2023, 2024)}
    q = {y: V.quarter_fin(fin, codes, y, fin_codes, mk[y - 1]) for y in (2024, 2025)}
    print(f'股价和财务指标（{time.time() - t0:.0f} 秒）', flush=True)

    meta = pd.read_parquet(reports_meta_path, columns=['stock_code', 'is_stock_report', 'date'])
    cov = meta[meta.is_stock_report & meta.date.str.startswith('2024')].stock_code.str[:6].value_counts()

    # 公司层面
    f = sample[['code', 'stock_code', 'company']].set_index('code')
    f['csrc'] = [csrc_map.get(c) for c in f.index]
    f['sw'] = [sw_map.get(c) for c in f.index]
    f['fin'] = f.index.isin(fin_codes)
    f = f.join(fdag.set_index('code'))
    w25, w24 = (wm[wm.year == y].set_index('code') for y in (2025, 2024))
    for c in ('SYNCH', 'SYNCH_SW', 'SYNCH_MKT', 'NCSKEW', 'DUVOL', 'IVOL', 'SIGMA', 'N_WEEKS', 'N_WEEKS_SYNCH', 'R2_SYNCH'):
        f[c] = w25[c]
        f[c + '24'] = w24[c]
    d25, d24 = (dm[dm.year == y].set_index('code') for y in (2025, 2024))
    for c in ('RET', 'BHAR', 'TURN', 'ILLIQ', 'N_DAYS'):
        f[c] = d25[c]
        f[c + '24'] = d24[c]
    for y, tag in ((2024, 'AR24'), (2025, 'AR25')):
        f[f'ANN_{tag}'] = car[y].ann_date
        f[f'DAY0_{tag}'] = car[y].day0
        f[f'CAR_{tag}_1'] = car[y].CAR_1
        f[f'CAR_{tag}_5'] = car[y].CAR_5
        f[f'ABSCAR_{tag}'] = car[y].CAR_1.abs()
    for c in ANNUAL_FIN:
        f[c] = a[2025][c]
        f[c + '24'] = a[2024][c]
    f['SIZE'], f['LEV'] = a[2024].SIZE, a[2024].LEV
    f['MKCAP24'] = mk[2024]
    f['BM'] = a[2024].equity / f.MKCAP24
    f['LNCOV'] = np.log1p(cov.reindex(f.index).fillna(0))
    f['N_COV'] = cov.reindex(f.index).fillna(0).astype(int)
    firm = f.reset_index()

    # 研报层面
    r = rep.rename(columns={'ratio': 'RATIO', 'lnpath': 'LNPATH', 'pathlen': 'PATHLEN'}).copy()
    r = r[r.n_edges > 0]
    r['LNCHARS_R'] = np.log(r.n_chars.astype(float))
    r['LNNODES_R'] = np.log(r.n_nodes)
    bc = r.broker.value_counts()
    r['broker_g'] = np.where(r.broker.map(bc) >= MIN_BROKER, r.broker, '其他')
    r['month'] = r.date.str[:6]
    info['n_brokers'] = int(r.broker.nunique())
    info['n_broker_groups'] = int(r.broker_g.nunique())
    keep_firm = ['code', 'csrc', 'sw', 'fin', 'N_DAG'] + [c for c in firm.columns if c in FIRM_Y or c in FIRM_Y.values()] + \
        [c for c in CONTROLS if c not in ('LNNODES', 'LNCHARS')]
    keep_firm = list(dict.fromkeys(c for c in keep_firm if c is not None))
    report = r[['report_id', 'code', 'company', 'date', 'month', 'broker', 'broker_g', 'n_chars', 'n_nodes', 'n_paths', 'unified',
                'RATIO', 'LNPATH', 'PATHLEN', 'LNCHARS_R', 'LNNODES_R']].merge(firm[keep_firm], on='code', how='left')

    # 季度面板
    qp = q[2025].merge(q[2024], on=['code', 'quarter'], suffixes=('', '24'))
    qp = qp.merge(firm[['code', 'csrc', 'sw', 'fin', 'N_DAG'] + DAG_X + CONTROLS], on='code', how='left')
    qp['ind_q'] = qp.csrc + '_Q' + qp.quarter.astype(str)
    qp['ind_q_sw'] = qp.sw + '_Q' + qp.quarter.astype(str)
    print(f'面板：公司 {len(firm)}、研报 {len(report)}、公司-季度 {len(qp)}（{time.time() - t0:.0f} 秒）', flush=True)
    return firm, report, qp, market, info


def winsorize_panels(firm, report, qp):
    """连续变量按 1%/99% 缩尾：公司层面变量在公司层面缩尾；研报层面的 X 和篇幅变量在研报层面缩尾，公司变量用公司层面缩尾后的值；
    季度面板的单季 Y 和上年同季值在公司-季度层面缩尾。计数、日期、代码类不缩尾。"""
    skip = {'N_DAG', 'N_WEEKS', 'N_WEEKS24', 'N_WEEKS_SYNCH', 'N_WEEKS_SYNCH24', 'N_DAYS', 'N_DAYS24', 'N_COV', 'MKCAP24', 'R2_SYNCH', 'R2_SYNCH24'}
    fw = firm.copy()
    num = [c for c in fw.columns if pd.api.types.is_float_dtype(fw[c]) and c not in skip]
    for c in num:
        fw[c] = R.winsor(fw[c])
    rw = report.copy()
    for c in ['RATIO', 'LNPATH', 'PATHLEN', 'LNCHARS_R', 'LNNODES_R']:
        rw[c] = R.winsor(rw[c])
    firm_cols = [c for c in rw.columns if c in num and c not in ('RATIO', 'LNPATH', 'PATHLEN')]
    rw = rw.drop(columns=firm_cols).merge(fw[['code'] + firm_cols], on='code', how='left')[report.columns]
    qw = qp.copy()
    for c in QUARTER_Y + [c + '24' for c in QUARTER_Y]:
        qw[c] = R.winsor(qw[c])
    firm_cols = [c for c in qw.columns if c in num]
    qw = qw.drop(columns=firm_cols).merge(fw[['code'] + firm_cols], on='code', how='left')[qp.columns]
    return fw, rw, qw


# ---------------------------------------------------------------- 回归
M1_SPECS = [  # (名称, 说明)
    ('main', '主回归：证监会门类固定效应和聚类，原始边，各篇均值'),
    ('sw', '申万一级固定效应和聚类'),
    ('nonfin', '剔除金融业'),
    ('unified', '统一名称后的边'),
    ('median', '公司层面取中位数'),
    ('ndag2', 'DAG 研报不少于 2 篇的公司'),
    ('ndag3', 'DAG 研报不少于 3 篇的公司'),
    ('noY24', '不加 Y24（即逐步加控制 (3)）'),
    ('noNODES', '不加 LNNODES'),
    ('c1', '逐步加控制 (1)：只有行业固定效应'),
    ('c2', '逐步加控制 (2)：行业固定效应 + 公司控制变量'),
]
STEP_LABEL = {'c1': '逐步加控制 (1)：只有固定效应', 'c2': '逐步加控制 (2)：固定效应 + 公司控制变量',
              'c3': '逐步加控制 (3)：+ DAG/文本控制，不加 Y24'}
M2_SPECS = [('main', '研报层面：行业+券商+月份固定效应，按公司聚类'), ('sw', '行业固定效应换申万一级'), ('nonfin', '剔除金融业'),
            ('twoway', '按公司和券商（原始券商名）双向聚类')] + list(STEP_LABEL.items())
M3_SPECS = [('main', '季度面板：行业×季度固定效应，按公司聚类'), ('sw', '申万一级×季度固定效应'), ('nonfin', '剔除金融业')] + \
    list(STEP_LABEL.items())


def step_controls(spec, ctrl):
    """逐步加控制的口径 -> (控制变量, 是否加 Y24)；不是逐步口径时返回 None。"""
    return {'c1': ([], False), 'c2': (list(FIRM_CTRL), False), 'c3': (list(ctrl), False)}.get(spec)


def m1_data(fw, spec):
    """M1 某个口径的数据、控制变量（不含 Y24）、固定效应、聚类维度、是否加 Y24。"""
    d, ctrl, fe, cl, y24 = fw, list(CONTROLS), 'csrc', 'csrc', True
    if spec in ('c1', 'c2'):
        ctrl, y24 = step_controls(spec, ctrl)
    elif spec == 'sw':
        fe = cl = 'sw'
    elif spec == 'nonfin':
        d = fw[~fw.fin]
    elif spec == 'unified':
        d = fw.drop(columns=DAG_X + ['LNNODES']).rename(columns={'RATIO_U': 'RATIO', 'LNPATH_U': 'LNPATH', 'PATHLEN_U': 'PATHLEN',
                                                                 'LNNODES_U': 'LNNODES'})
    elif spec == 'median':
        d = fw.drop(columns=DAG_X).rename(columns={'RATIO_MED': 'RATIO', 'LNPATH_MED': 'LNPATH', 'PATHLEN_MED': 'PATHLEN'})
    elif spec in ('ndag2', 'ndag3'):
        d = fw[fw.N_DAG >= int(spec[-1])]
    elif spec == 'noY24':
        y24 = False
    elif spec == 'noNODES':
        ctrl.remove('LNNODES')
    return d, ctrl, fe, cl, y24


def m2_data(rw, spec):
    """M2 某个口径的数据、控制变量（不含 Y24）、固定效应列表、聚类维度（列名或两个列名）、是否加 Y24。"""
    d, ctrl, fe, cl, y24 = rw, list(REPORT_CONTROLS), ['csrc', 'broker_g', 'month'], 'code', True
    if spec == 'sw':
        fe = ['sw', 'broker_g', 'month']
    elif spec == 'nonfin':
        d = rw[~rw.fin.astype(bool)]
    elif spec == 'twoway':
        cl = ['code', 'broker']
    elif spec in STEP_LABEL:
        ctrl, y24 = step_controls(spec, ctrl)
    return d, ctrl, fe, cl, y24


def m3_data(qw, spec):
    """M3 某个口径的数据、控制变量（不含 Y24）、固定效应列表、聚类维度、是否加 Y24（上年同季）。"""
    d, ctrl, fe, cl, y24 = qw, list(CONTROLS), ['ind_q'], 'code', True
    if spec == 'sw':
        fe = ['ind_q_sw']
    elif spec == 'nonfin':
        d = qw[~qw.fin.astype(bool)]
    elif spec in STEP_LABEL:
        ctrl, y24 = step_controls(spec, ctrl)
    return d, ctrl, fe, cl, y24


def run_set(d, ys, y24_of, ctrl, fe, cl, model, spec, label, full=None):
    """每个 Y：三个 X 分别单独回归，再三个一起回归。full 是列表时把全部系数（含控制变量）追加进去。"""
    rows = []
    for y in ys:
        c = list(ctrl) + ([y24_of[y]] if y24_of.get(y) else [])
        for xs, kind in [([x], '单独') for x in DAG_X] + [(DAG_X, '联合')]:
            res, _ = R.fit(d, y, xs, c, fe, cl, terms=full is not None)
            for row in res:
                if row['X'] in xs:
                    rows.append({'model': model, 'spec': spec, 'spec_label': label, 'Y': y, 'reg': kind, **row,
                                 'fe': '+'.join(fe), 'cluster': cl if isinstance(cl, str) else '+'.join(cl), 'controls': '、'.join(c)})
                if full is not None:
                    full.append({'model': model, 'spec': spec, 'Y': y, 'reg': kind, 'X_set': xs[0] if kind == '单独' else '联合',
                                 'term': row['X'], **{k: row[k] for k in ('coef', 'se', 't', 'p', 'n', 'r2', 'adj_r2', 'n_clusters')}})
    return rows


def run_perm(fw):
    """M1 main、sw 口径单独回归的 Freedman–Lane 行业内置换检验 -> DataFrame[model, spec, Y, reg, X, p_perm, dt]（dt：FWL t 与 fit t 之差）。"""
    out = []
    for spec in ('main', 'sw'):
        d, ctrl, fe, cl, _ = m1_data(fw, spec)
        for y in FIRM_Y:
            c = ctrl + ([FIRM_Y[y]] if FIRM_Y[y] else [])
            for x in DAG_X:
                r = RB.perm_test(d, y, x, c, fe, cl, f'M1|{spec}|{y}|{x}')
                if r:
                    out.append({'model': 'M1', 'spec': spec, 'Y': y, 'reg': '单独', 'X': x, 'p_perm': r[2], 'dt': abs(r[0] - r[1])})
    return pd.DataFrame(out)


def run_all(fw, rw, qw):
    """全部回归 -> (reg_results 长表, 全部系数表)。"""
    rows, full = [], []
    t0 = time.time()
    for spec, label in M1_SPECS:
        d, ctrl, fe, cl, y24 = m1_data(fw, spec)
        rows += run_set(d, list(FIRM_Y), FIRM_Y if y24 else {}, ctrl, [fe], cl, 'M1', spec, label, full if spec == 'main' else None)
    print(f'M1：{len(rows)} 个系数（{time.time() - t0:.0f} 秒）', flush=True)
    n = len(rows)
    for spec, label in M2_SPECS:
        d, ctrl, fe, cl, y24 = m2_data(rw, spec)
        rows += run_set(d, list(FIRM_Y), FIRM_Y if y24 else {}, ctrl, fe, cl, 'M2', spec, label, full if spec == 'main' else None)
    print(f'M2：{len(rows) - n} 个系数（{time.time() - t0:.0f} 秒）', flush=True)
    n = len(rows)
    for spec, label in M3_SPECS:
        d, ctrl, fe, cl, y24 = m3_data(qw, spec)
        rows += run_set(d, QUARTER_Y, {y: y + '24' for y in QUARTER_Y} if y24 else {}, ctrl, fe, cl, 'M3', spec, label,
                        full if spec == 'main' else None)
    print(f'M3：{len(rows) - n} 个系数（{time.time() - t0:.0f} 秒）', flush=True)
    res = pd.DataFrame(rows)
    res['step'] = res.spec.map(STEP_OF)
    perm = run_perm(fw)
    print(f'置换检验：{len(perm)} 个回归、每个 B = {RB.B_PERM}；FWL t 与完整回归 t 的最大差 {perm.dt.max():.2e}'
          f'（{time.time() - t0:.0f} 秒）', flush=True)
    res = res.merge(perm.drop(columns='dt'), on=['model', 'spec', 'Y', 'reg', 'X'], how='left')
    order = ['model', 'spec', 'spec_label', 'step', 'Y', 'reg', 'X', 'coef', 'se', 't', 'p', 'se_hc1', 't_hc1', 'p_hc1', 'p_perm',
             'n', 'k', 'r2', 'adj_r2', 'n_clusters', 'n_singletons', 'fe', 'cluster', 'y_mean', 'y_sd', 'controls']
    return res[order], pd.DataFrame(full)


def run_m1_extra(fw):
    """M1 主样本的秩相关、三等分、逐一剔除。"""
    d, ctrl, fe, cl, _ = m1_data(fw, 'main')
    ric, grp = RB.rank_group(d, list(FIRM_Y), FIRM_Y, ctrl, fe, cl, DAG_X)
    infl, diff = RB.influence(d, list(FIRM_Y), FIRM_Y, ctrl, fe, cl, DAG_X)
    print(f'逐一剔除：numpy 重估与 R.fit 重估的最大差 {diff:.2e}', flush=True)
    return ric, grp, infl


# ---------------------------------------------------------------- 描述性统计和相关系数
def descriptives(fw, rw, qw):
    out = []
    sets = [('公司', fw, [c for c in fw.columns if pd.api.types.is_numeric_dtype(fw[c]) and fw[c].dtype != bool]),
            ('研报', rw, ['RATIO', 'LNPATH', 'PATHLEN', 'LNCHARS_R', 'LNNODES_R', 'n_chars', 'n_nodes', 'n_paths']),
            ('季度', qw, QUARTER_Y + [c + '24' for c in QUARTER_Y])]
    for panel, d, cols in sets:
        for c in cols:
            s = pd.to_numeric(d[c], errors='coerce').dropna()
            q = s.quantile([0, .01, .25, .5, .75, .99, 1]).to_numpy() if len(s) else [np.nan] * 7
            out.append({'panel': panel, 'var': c, 'n': len(s), 'mean': s.mean(), 'sd': s.std(), 'min': q[0], 'p1': q[1], 'p25': q[2],
                        'p50': q[3], 'p75': q[4], 'p99': q[5], 'max': q[6]})
    return pd.DataFrame(out)


def correlations(fw):
    rows = []
    for x in DAG_X:
        for y in FIRM_Y:
            s = fw[[x, y]].dropna()
            rows.append({'X': x, 'Y': y, 'n': len(s), 'pearson': s[x].corr(s[y]),
                         'spearman': s[x].rank().corr(s[y].rank())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 回归结果.md
def _cell(r, hc=True):
    s = f'{r.coef:.4g}{R.stars(r.p)} ({r.t:.2f})'
    return s + (f' [{r.t_hc1:.2f}]' if hc else '')


def table(res, model, spec, hc=True):
    r = res[(res.model == model) & (res.spec == spec)]
    ys = list(dict.fromkeys(r.Y))
    head = '| Y | ' + ' | '.join(X_LABEL[x] for x in DAG_X) + ' | N | G | 联合：' + ' / '.join(X_LABEL[x] for x in DAG_X) + ' | N |'
    lines = [head, '|' + '---|' * (len(DAG_X) + 5)]
    for y in ys:
        a = r[(r.Y == y) & (r.reg == '单独')].set_index('X')
        j = r[(r.Y == y) & (r.reg == '联合')].set_index('X')
        cells = [_cell(a.loc[x], hc) if x in a.index else '' for x in DAG_X]
        n = a.n.iloc[0] if len(a) else ''
        g = a.n_clusters.iloc[0] if len(a) else ''
        jc = ' / '.join(f'{j.loc[x].coef:.4g}{R.stars(j.loc[x].p)} ({j.loc[x].t:.2f})' for x in DAG_X) if len(j) == 3 else ''
        lines.append(f'| {Y_LABEL.get(y, y)} | ' + ' | '.join(cells) + f' | {n} | {g} | {jc} | {j.n.iloc[0] if len(j) else ""} |')
    return '\n'.join(lines)


def robustness_table(res):
    r = res[(res.model == 'M1') & (res.reg == '单独')]
    specs = [s for s, _ in M1_SPECS]
    lines = ['| Y | X | ' + ' | '.join(specs) + ' |', '|' + '---|' * (len(specs) + 2)]
    piv = r.set_index(['Y', 'X', 'spec'])
    for y in FIRM_Y:
        for x in DAG_X:
            cells = []
            for s in specs:
                k = (y, x, s)
                cells.append(f'{piv.loc[k].coef:.3g}{R.stars(piv.loc[k].p)}' if k in piv.index else '')
            lines.append(f'| {Y_LABEL.get(y, y)} | {X_LABEL[x]} | ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_md(path, res, info, desc):
    main = res[(res.model == 'M1') & (res.spec == 'main')]
    sig = main[(main.reg == '单独') & (main.p < 0.1)]
    fd = desc[desc.panel == '公司'].set_index('var')
    pick = lambda v: f'{fd.loc[v, "mean"]:.4g}（{fd.loc[v, "sd"]:.3g}，{int(fd.loc[v, "n"])}）' if v in fd.index else ''  # noqa: E731
    lines = [
        '# DAG 结构指标与 2025 年股价、财务指标的回归结果', '',
        f'生成时间 {time.strftime("%Y-%m-%d %H:%M")}；方案见 docs/DAG结构指标与2025年股价财务回归方案.md。'
        '全部数字由 regress_dag_outcomes.py 写出，长表见 reg_results.csv。', '',
        '## 样本', '',
        f'- 公司 241 家，DAG 研报 {info["n_reports"]} 篇（deep {info["n_reports_by_src"].get(0, 0)} 篇、试跑目录 {info["n_reports_by_src"].get(1, 0)} 篇；'
        f'有统一结果的 {info["n_reports_unified"]} 篇）；金融业 {info["n_fin"]} 家。',
        f'- 市值加权市场组合（沪深 A 股，总市值加权）买入持有收益：2024 年 {info["mkt_bh"][2024]:.2%}，2025 年 {info["mkt_bh"][2025]:.2%}'
        f'（日收益复利；周收益复利 2024 {info["mkt_w_bh"][2024]:.2%}、2025 {info["mkt_w_bh"][2025]:.2%}）。',
        '- 数据里没有机构持股、产权性质、交叉上市，这几个文献常用的控制变量没有加；组合权重用总市值（没有流通市值）。', '',
        '## 主要变量（缩尾后，均值（标准差，观测数））', '',
        '| 变量 | 均值（标准差，N） |', '|---|---|']
    for v in DAG_X + ['SYNCH', 'NCSKEW', 'DUVOL', 'IVOL', 'SIGMA', 'RET', 'BHAR', 'CAR_AR24_1', 'CAR_AR25_1', 'TURN', 'ILLIQ',
                      'ROA', 'ROE', 'REVG', 'NPG', 'EARNVOL']:
        lines.append(f'| {Y_LABEL.get(v, v)} | {pick(v)} |')
    lines += ['', '## M1 公司层面截面回归（主）', '',
              '每格：系数（X 在回归样本内标准化，指标高 1 个标准差时 Y 的变化）、星号（按证监会门类聚类 CR1，p 值用 t(G−1)：* 10%、** 5%、*** 1%）、'
              '(聚类 t)、[HC1 t]。控制 SIZE、LEV、BM、ROA24、TURN24、RET24、SIGMA24、LNCOV、LNNODES、LNCHARS、LNNDAG 和 Y24（能算的），'
              '证监会门类固定效应。“联合”是三个 X 一起进回归。', '',
              table(res, 'M1', 'main'), '',
              f'按聚类 t 在 10% 水平显著的单独回归系数 {len(sig)} 个（共 {len(main[main.reg == "单独"])} 个）。', '',
              '## M2 研报层面回归', '',
              '每篇研报一个观测，X 是这一篇的指标；控制 ln 研报字数、ln 进图概念数和公司层面控制变量（不含 LNNODES、LNCHARS）、Y24，'
              '证监会门类 + 券商（不足 5 篇并成“其他”）+ 发布月份固定效应，按公司聚类。', '',
              table(res, 'M2', 'main'), '',
              '## M3 季度财务面板（2025Q1–Q4）', '',
              'X 是公司层面指标；控制同 M1，Y24 是 2024 年同季度的值；行业×季度固定效应，按公司聚类。', '',
              table(res, 'M3', 'main'), '',
              '## 稳健性（M1，单独回归的系数和聚类显著性）', '',
              '口径：' + '；'.join(f'{s} = {lab}' for s, lab in M1_SPECS) + '。', '',
              robustness_table(res), '',
              '## 解读时要注意', '',
              '- 约 240 家公司、二十多个 Y、三个 X：按 10% 水平会有不少系数碰巧显著，要看方向和显著性在不同口径下是否一致。',
              '- 证监会门类聚类只有十几类，聚类标准误本身不稳定；HC1 t 值一并列出作对照。',
              '- 这是相关性，不是因果；概念命名不稳定带来的测量误差会把系数压向 0。', '']
    Path(path).write_text('\n'.join(lines), encoding='utf-8')


def main(out_dir, sample_path, industry_path, dag_dirs, daily_path, fin_path, reports_meta_path):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    firm, report, qp, market, info = build_panels(sample_path, industry_path, [Path(p) for p in dag_dirs], daily_path, fin_path,
                                                  reports_meta_path)
    fw, rw, qw = winsorize_panels(firm, report, qp)
    for name, raw, w in (('firm', firm, fw), ('report', report, rw), ('quarter', qp, qw)):
        w.to_parquet(out_dir / f'{name}_panel.parquet', index=False)
        raw.to_parquet(out_dir / f'{name}_panel_raw.parquet', index=False)
    cols = ['freq', 'year', 'week', 'date', 'mkt_ret']  # 周收益的 date 是这一周最后一个交易日
    wk = market['weekly'].rename(columns={'week_last_date': 'date'}).assign(freq='week')[cols]
    dy = market['daily'].assign(freq='day', year=lambda x: x.date.str[:4].astype(int), week=np.nan)[cols]
    pd.concat([wk, dy], ignore_index=True).to_parquet(out_dir / 'market_returns.parquet', index=False)
    desc = descriptives(fw, rw, qw)
    desc.to_csv(out_dir / 'descriptives.csv', index=False, encoding='utf-8-sig')
    correlations(fw).to_csv(out_dir / 'correlations.csv', index=False, encoding='utf-8-sig')
    res, full = run_all(fw, rw, qw)
    res.to_csv(out_dir / 'reg_results.csv', index=False, encoding='utf-8-sig')
    full.to_csv(out_dir / 'reg_coefs_full.csv', index=False, encoding='utf-8-sig')
    ric, grp, infl = run_m1_extra(fw)
    ric.to_csv(out_dir / 'rank_ic.csv', index=False, encoding='utf-8-sig')
    grp.to_csv(out_dir / 'group_sort.csv', index=False, encoding='utf-8-sig')
    infl.to_csv(out_dir / 'influence.csv', index=False, encoding='utf-8-sig')
    write_md(out_dir / '回归结果.md', res, info, desc)
    T.write_tables(out_dir / 'tables', res, full, desc, fw, ric, grp, infl, info)
    dec, chk = RB.fe_r2_decomposition(rw, DAG_X, FE_DECOMP)
    dec.to_csv(out_dir / 'fe_r2_decomposition.csv', index=False, encoding='utf-8-sig')
    T.write_fe_r2(out_dir / 'tables' / 's_fe_r2.md', dec, FE_DECOMP)
    print('固定效应 R² 分解核对（R² 贡献和 − 总 R²，总 R² − 组内去均值法，调整后 R² 贡献和 − 总调整后 R²）：',
          {y: tuple(f'{v:.1e}' for v in c) for y, c in chk.items()}, flush=True)
    fe_r2_firm(out_dir, rw)
    info['seconds'] = round(time.time() - t0)
    print('核对信息：', {k: v for k, v in info.items()}, flush=True)
    print(f'完成：{out_dir}（{time.time() - t0:.0f} 秒）')
    return res, info


def fe_r2_firm(out_dir, rw):
    """在三组分解上加公司固定效应（code）的四组分解：全样本出表 s_fe_r2_firm.md；研报不少于 2 篇的公司子样本上的四组、三组分解只进 csv。"""
    fe4 = FE_DECOMP + ['code']
    n_rep = rw.code.value_counts()
    multi = rw[rw.code.map(n_rep) >= 2]
    parts, lines = [], []
    for name, d, fes in (('全样本', rw, fe4), ('研报≥2篇的公司', multi, fe4), ('研报≥2篇的公司', multi, FE_DECOMP)):
        dec, chk = RB.fe_r2_decomposition(d, DAG_X, fes)
        s = dec[dec.kind == 'subset'].set_index('subset')
        parts.append(dec.assign(sample=name, decomp=f'{len(fes)}组'))
        msg = (f'{name}{len(fes)}组：R² 贡献和 − 总 R² 最大 {max(abs(c[0]) for c in chk.values()):.1e}，总 R² − 组内去均值法最大 '
               f'{max(abs(c[1]) for c in chk.values()):.1e}，调整后 R² 贡献和 − 总调整后 R² 最大 {max(abs(c[2]) for c in chk.values()):.1e}')
        if 'code' in fes:
            alt = max(abs(s.loc[(s.index == '+'.join(fes)) & (s.Y == y), c].iloc[0] -
                          s.loc[(s.index == 'month+broker_g+code') & (s.Y == y), c].iloc[0]) for y in DAG_X for c in ('r2', 'adj_r2'))
            msg += (f'，含公司的子集加不加行业：R² 差最大 {RB.nested_gap(dec, fes, "sw", "code"):.1e}、调整后 R² 差最大 '
                    f'{RB.nested_gap(dec, fes, "sw", "code", "adj_r2"):.1e}，四组 vs month+broker_g+code（R² 和调整后 R²）差最大 {alt:.1e}')
        lines.append(msg)
    cols = ['sample', 'decomp']
    out = pd.concat(parts, ignore_index=True)
    out[cols + [c for c in out.columns if c not in cols]].to_csv(out_dir / 'fe_r2_decomposition_firm.csv', index=False, encoding='utf-8-sig')
    n1 = int((n_rep == 1).sum())
    note = ('公司固定效应嵌套了申万一级行业固定效应：每家公司只属于一个行业，已经放了公司时再加行业，拟合优度和设计矩阵的秩都不变，'
            '调整后 R² 也不再变化，所以行业的贡献只来自它在公司之前加入的那些顺序里的增量的平均；四组都放时的模型等价于只放发布月份、券商、公司。'
            f'样本里 {len(n_rep)} 家公司中有 {n1} 家只有 1 篇研报，这 {n1} 篇研报会被公司固定效应完全拟合。')
    T.write_fe_r2(out_dir / 'tables' / 's_fe_r2_firm.md', parts[0], fe4, note)
    print('四组（含公司）R² 分解核对：' + '；'.join(lines), flush=True)
