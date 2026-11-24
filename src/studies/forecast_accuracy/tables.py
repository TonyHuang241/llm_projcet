"""文档的表 1–8 和正文、第 4 节、图注里的统计数字（都从回归样本和 reg_all 长表算出）。

表的 csv（tables/ 下）行列形状同文档里的表，数值是文档显示的口径（PMAFE 的系数乘 100，Score 的系数不乘，表 1 是调整 R² × 100），
保留 6 位小数以便核对；系数格后面跟显著性星号（* p < 0.10、** p < 0.05、*** p < 0.01，p 值用 t(公司数 − 1)），
t 值在括号里；表 3、5、6、7 的一格是“系数星号 (t)”，N 一列为“投资评级概念对应的逻辑结构各列 / 全文逻辑结构各列”的观测数。

text_numbers.csv：正文、图注和第 4 节出现的数字，每个数字一行：位置、内容、数值（原始精度）、小数位（文档显示的位数）、
显示（按小数位四舍五入）、星号、算法（怎么算的）。

方差分解（表 1）：每个逻辑结构变量（缩尾后）对一组固定效应虚拟变量 + 常数项做 OLS（不剔除单例），调整 R² =
1 − (1 − R²)(N − 1)/(N − K)，K 为设计矩阵的秩；样本为该变量在基准回归（表 2 那一列）里的估计样本。
"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from src.ols import fe_dummies, stars, t_pvalue
from src.studies.forecast_accuracy.regressions import estimation_sample
from src.studies.forecast_accuracy.sample import X6

LABEL = dict(zip(X6, ['RateSrc', 'RatePath', 'RateLen', 'AllSrc', 'AllPath', 'AllLen']))
GROUP = {x: '投资评级概念对应的逻辑结构' if x.startswith('RT') else '全文逻辑结构' for x in X6}
FE_ROW = {'none': '无', 'industry': '行业', 'firm': '公司', 'firm+broker': '公司 + 券商'}
DECOMP_SPECS = [  # (名称, 文档列名, 固定效应列；列的先后决定设计矩阵的列序)
    ('month', '(1) 月份', ['month']), ('industry', '(2) 行业', ['sw']), ('firm', '(3) 公司', ['code']),
    ('broker', '(4) 券商', ['broker_g']), ('firm+broker', '(5) 公司 + 券商', ['broker_g', 'code']),
    ('month+firm+broker', '(6) 月份 + 公司 + 券商', ['month', 'broker_g', 'code']),
]
TABLE2_TERMS = [('G', None), ('Size', 'SIZE_M'), ('BM', 'BM'), ('Mom', 'MOM'), ('Vol', 'VOL'), ('ROA', 'ROA_T'), ('LnAge', 'LNAGE'),
                ('PeerLag', 'PEERLAG25')]
PAIRS = ['RT_NSRC+WG_NSRC', 'RT_LNPATH+WG_LNPATH', 'RT_PATHLEN+WG_PATHLEN']
PAIR_COLS = ['(1) 源点数', '(2) 路径数', '(3) 平均路径长度']
FILES = {1: 'table1_fe_decomposition.csv', 2: 'table2_benchmark.csv', 3: 'table3_fe_steps.csv', 4: 'table4_paired.csv',
         5: 'table5_fy2024.csv', 6: 'table6_score.csv', 7: 'table7_firm_broker.csv', 8: 'table8_firm_broker_paired.csv'}
NUMBERS_FILE = 'text_numbers.csv'


def rnd(x, nd):
    """四舍五入（十进制、半数进位）。"""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


# ---------------------------------------------------------------- 方差分解（表 1）
def fit_r2(y, fe_frame):
    """常数项 + 各固定效应虚拟变量（每个去掉一组）的 OLS：返回 (R², 调整 R², K)。"""
    X = np.column_stack([np.ones(len(y)), fe_dummies(fe_frame, list(fe_frame.columns))])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    k = int(np.linalg.matrix_rank(X))
    n = len(y)
    r2 = 1 - (e @ e) / ((y - y.mean()) ** 2).sum()
    adj = 1 - (1 - r2) * (n - 1) / (n - k) if n - k > 0 else np.nan
    return r2, adj, k


def variance_decomposition(D):
    """六个变量 × 六组固定效应的 R² 和调整 R²（样本 = 该变量的基准回归估计样本）。"""
    rows = []
    for x in X6:
        sub = estimation_sample(D, x=x)
        y = sub[x].to_numpy(float)
        for spec, _, fes in DECOMP_SPECS:
            fe = sub[fes]
            ok = fe.notna().all(axis=1).to_numpy()
            r2, adj, k = fit_r2(y[ok], fe[ok])
            rows.append({'variable': x, 'spec': spec, 'fe_cols': '+'.join(fes), 'N': int(ok.sum()), 'K': k, 'r2': r2, 'adj_r2': adj})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 表
class Reg:
    """reg_all 长表的取格子工具。"""

    def __init__(self, reg):
        self.reg = reg

    def q(self, **kw):
        m = np.ones(len(self.reg), dtype=bool)
        for k, v in kw.items():
            m &= (self.reg[k] == v).to_numpy()
        return self.reg[m]

    def cell(self, level, block, y, fe, X, term=None):
        r = self.q(level=level, block=block, Y=y, fe=fe, X=X)
        r = r[r.term == (term or X)]
        assert len(r) == 1, (level, block, y, fe, X, term, len(r))
        return r.iloc[0]

    def n_of(self, level, block, y, fe, xs):
        ns = sorted({int(self.cell(level, block, y, fe, x).n) for x in xs})
        return str(ns[0]) if len(ns) == 1 else ','.join(map(str, ns))


def coef_str(c, scale):
    return f'{c.coef * scale:.6f}{stars(c.p)}'


def table1(dec):
    rows = []
    for x in X6:
        z = dec[dec.variable == x].set_index('spec')
        rows.append({'逻辑结构': GROUP[x] if x in ('RT_NSRC', 'WG_NSRC') else '', '变量': LABEL[x],
                     **{col: f'{z.loc[spec, "adj_r2"] * 100:.6f}' for spec, col, _ in DECOMP_SPECS}})
    return pd.DataFrame(rows)


def table2(G):
    cols = {x: f'({i}) {GROUP[x]} {LABEL[x]}' for i, x in enumerate(X6, 1)}
    rows = []
    for lab, term in TABLE2_TERMS:
        cs = {x: G.cell('report', 'fe_steps', 'PMAFE_FY25', 'firm+broker', x, term) for x in X6}
        rows.append({'': lab, **{cols[x]: coef_str(c, 100) for x, c in cs.items()}})
        rows.append({'': '(t)', **{cols[x]: f'({c.t:.6f})' for x, c in cs.items()}})
    cs = {x: G.cell('report', 'fe_steps', 'PMAFE_FY25', 'firm+broker', x) for x in X6}
    rows.append({'': '调整 R²', **{cols[x]: f'{c.adj_r2:.6f}' for x, c in cs.items()}})
    rows.append({'': 'N', **{cols[x]: str(int(c.n)) for x, c in cs.items()}})
    for lab in ('券商固定效应', '公司固定效应', '年度固定效应'):
        rows.append({'': lab, **{cols[x]: '是' for x in X6}})
    return pd.DataFrame(rows)


def ladder(G, level, block, y, scale):
    """表 3、5、6、7：行 = 固定效应四档，列 = 六个变量 + N。"""
    rows = []
    for fe, lab in FE_ROW.items():
        r = {'固定效应': lab}
        for x in X6:
            c = G.cell(level, block, y, fe, x)
            r[f'{GROUP[x]} {LABEL[x]}'] = f'{coef_str(c, scale)} ({c.t:.6f})'
        r['N'] = f'{G.n_of(level, block, y, fe, X6[:3])} / {G.n_of(level, block, y, fe, X6[3:])}'
        rows.append(r)
    return pd.DataFrame(rows)


def paired(G, level, block, ctrl_label):
    """表 4、8：两类逻辑结构同名变量成对放入（PMAFE_FY25，公司 + 券商固定效应）。"""
    cs = {spec: (G.cell(level, block, 'PMAFE_FY25', 'firm+broker', spec, spec.split('+')[0]),
                 G.cell(level, block, 'PMAFE_FY25', 'firm+broker', spec, spec.split('+')[1])) for spec in PAIRS}
    rows = []
    for i, lab in ((0, '投资评级概念对应的逻辑结构'), (1, '全文逻辑结构')):
        rows.append({'': lab, **{col: coef_str(cs[s][i], 100) for s, col in zip(PAIRS, PAIR_COLS)}})
        rows.append({'': '(t)', **{col: f'({cs[s][i].t:.6f})' for s, col in zip(PAIRS, PAIR_COLS)}})
    for lab in (ctrl_label, '公司固定效应', '券商固定效应'):
        rows.append({'': lab, **{col: '是' for col in PAIR_COLS}})
    rows.append({'': '调整 R²', **{col: f'{cs[s][0].adj_r2:.6f}' for s, col in zip(PAIRS, PAIR_COLS)}})
    rows.append({'': 'N', **{col: str(int(cs[s][0].n)) for s, col in zip(PAIRS, PAIR_COLS)}})
    return pd.DataFrame(rows)


def all_tables(reg, dec):
    """{表号: DataFrame}。"""
    G = Reg(reg)
    return {1: table1(dec), 2: table2(G), 3: ladder(G, 'report', 'fe_steps', 'PMAFE_FY25', 100), 4: paired(G, 'report', 'paired', '控制变量 X 和 PeerLag'),
            5: ladder(G, 'report', 'fe_steps', 'PMAFE_FY24', 100), 6: ladder(G, 'report', 'fe_steps', 'SCORE_FY25', 1),
            7: ladder(G, 'firm_broker', 'fe_steps', 'PMAFE_FY25', 100), 8: paired(G, 'firm_broker', 'paired', '控制变量 X 和 DAge')}


# ---------------------------------------------------------------- 正文数字
def corr_p(x, y):
    """Pearson r 和双侧 p 值（t = r·sqrt((n−2)/(1−r²))，t(n−2)）。"""
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else np.inf
    return r, t_pvalue(t, n - 2)


def text_numbers(S, D, P, reg, dec, est, pl, rl, schematic):
    """S 研究范围（791 篇），D 研报层面样本，P 公司-券商层面样本，reg 回归长表，dec 方差分解，est 基准回归估计样本，
    pl / rl 图 2 的同行明细和每篇研报，schematic 图 1 例子的指标。返回 text_numbers.csv 的 DataFrame。"""
    from src.graph.load import read_table
    from src.measure.variables import RATING_NODE
    G = Reg(reg)
    rows = []

    def add(loc, item, value, nd, how, star=None):
        v = float(value)
        rows.append({'位置': loc, '内容': item, '数值': v, '小数位': nd, '显示': rnd(v, nd) if np.isfinite(v) else 'NA',
                     '星号': '' if star is None else star, '算法': how})

    def cell2(loc, item, block, y, fe, X, term=None, scale=100):
        c = G.cell('report', block, y, fe, X, term)
        add(loc, item + ' 系数', c.coef * scale, 2, f'reg_all report / {block} / {y} / {fe} / X={X} / term={term or X} 的 coef'
            + (' × 100' if scale == 100 else ''), stars(c.p))
        add(loc, item + ' t', c.t, 2, '同上的 t（按公司聚类）')

    # ---- 1.1、1.2 样本
    rid = list(S.report_id)
    base = G.cell('report', 'fe_steps', 'PMAFE_FY25', 'firm+broker', 'WG_NSRC')
    rate = G.cell('report', 'fe_steps', 'PMAFE_FY25', 'firm+broker', 'RT_NSRC')
    add('1.1', '样本研报篇数 791', len(S), 0, '研究范围：公司在覆盖名单（241 家）里、graph 研报表 n_edges > 0 的研报篇数')
    add('1.1', '满足条件的研报 771', int(D.PMAFE_FY25.notna().sum()), 0, '基础样本（786 篇）中 PMAFE_FY25 非缺失的篇数')
    add('1.1', '基准回归样本 709 篇', base.n, 0, 'reg_all report / fe_steps / PMAFE_FY25 / firm+broker / WG_* 的 n')
    add('1.1', '基准回归 180 家公司', base.n_clusters, 0, '同上的 n_clusters')
    C = read_table('concepts', report_ids=rid, columns=['name'])
    n_has = C[C.name == RATING_NODE].report_id.nunique()
    add('1.2', '抽取出“投资评级”概念的研报 767 篇', n_has, 0,
        'graph 概念表（src.graph.load.read_table("concepts")，各次运行按 RUN_ORDER 合并，含没有进图边的孤立概念）中 name == “投资评级” 的研报数（791 篇内）')
    add('1.2', '767 篇占 791 篇的 97%', n_has / len(S) * 100, 0, '上一行 ÷ 791 × 100')
    add('1.2（对照，非文档数字）', '图里有“投资评级”节点的研报篇数', int(S.HAS_RATING_NODE.sum()), 0,
        'measure 的 HAS_RATING_NODE（图只由进图的边构成）之和；说明 767 是按概念表数的')
    for name, key, nd, val in (('全文', '全文', 0, 'NSRC'), ('全文', '全文', 0, 'N_SINKS'), ('全文', '全文', 0, 'N_PATHS'), ('全文', '全文', 1, 'PATHLEN'),
                               ('投资评级', '投资评级', 0, 'NSRC'), ('投资评级', '投资评级', 0, 'N_PATHS'), ('投资评级', '投资评级', 1, 'PATHLEN')):
        lab = {'NSRC': '源点数', 'N_SINKS': '汇点数', 'N_PATHS': '路径数', 'PATHLEN': '平均路径长度'}[val]
        v = schematic[key]['N_SINKS'] if val == 'N_SINKS' else schematic[key][val]
        add('1.2 图 1 的例子', f'{name}逻辑结构的{lab}', v, nd, '图 1 虚构小图上 src.measure.variables.graph_measures 的结果（汇点数 = 严格汇点个数）')
    add('1.2', '“投资评级”是汇点的研报 606 篇', int(S.RT_NSRC.notna().sum()), 0, '791 篇中 measure 的 RT_NSRC 非缺失（投资评级是严格汇点）的篇数')
    add('1.2', '基准回归中 RT 各列 520 篇', rate.n, 0, 'reg_all report / fe_steps / PMAFE_FY25 / firm+broker / RT_* 的 n')
    add('1.2', '基准回归中 RT 各列 149 家公司', rate.n_clusters, 0, '同上的 n_clusters')

    # ---- 1.3、1.4 描述统计（基准回归估计样本 709 篇）
    add('1.3', '同行家数中位数 20', est.N_PEERS25.dropna().median(), 0, '基准回归估计样本（709 篇）N_PEERS25 的中位数')
    add('1.4', 'PMAFE 标准差 33 个百分点', est.PMAFE_FY25.dropna().std() * 100, 0, '709 篇缩尾后 PMAFE_FY25 的标准差 × 100')
    add('1.4', 'Score 标准差约 30', est.SCORE_FY25.dropna().std(), 0, '709 篇 SCORE_FY25 的标准差')
    add('1.4', 'PMAFE 与 Score 的相关系数 −0.79', est[['PMAFE_FY25', 'SCORE_FY25']].corr().iloc[0, 1], 2, '709 篇 Pearson 相关（两两非缺失）')

    # ---- 2.1
    d3 = dec.set_index(['variable', 'spec']).adj_r2 * 100
    for x in X6:
        add('表 1 注', f'{LABEL[x]} 的样本篇数', dec[dec.variable == x].N.iloc[0], 0, '方差分解样本 = 该变量的基准回归估计样本（六个设定相同）')
    add('2.1', '月份固定效应的调整 R² 不超过 1.6%', max(d3[(x, 'month')] for x in X6), 1, '表 1 第 (1) 列六个变量的最大值')
    for spec, lab in (('firm', '公司'), ('broker', '券商'), ('firm+broker', '公司 + 券商')):
        vals = [d3[(x, spec)] for x in X6[:3]]
        add('2.1', f'RT 三个变量{lab}固定效应的调整 R² 下限', min(vals), 1, f'表 1 {lab} 列 RT 三行的最小值')
        add('2.1', f'RT 三个变量{lab}固定效应的调整 R² 上限', max(vals), 1, f'表 1 {lab} 列 RT 三行的最大值')
    add('2.1', 'AllSrc 公司 + 券商 36.2%', d3[('WG_NSRC', 'firm+broker')], 1, '表 1 AllSrc 第 (5) 列')
    add('2.1', 'AllPath 公司 + 券商 38.5%', d3[('WG_LNPATH', 'firm+broker')], 1, '表 1 AllPath 第 (5) 列')
    cx = est[X6].corr()
    add('2.1', 'RateSrc 与 RatePath 的相关系数 0.92', cx.loc['RT_NSRC', 'RT_LNPATH'], 2, '709 篇（缩尾后，两两非缺失，RT 变量非缺失 546 篇）Pearson 相关')
    add('2.1', 'RateSrc 与 RateLen 的相关系数 0.70', cx.loc['RT_NSRC', 'RT_PATHLEN'], 2, '同上')

    # ---- 图 2、2.2
    add('图 2 注', '样本研报 709 篇', rl.report_id.nunique(), 0, '图 2 的研报 = 基准回归估计样本')
    add('图 2 注', '同行预测 15266 条', len(pl), 0, '这些研报在 accuracy report_peers 里的 FY2025、[d − 90, d] 比较组同行条数')
    add('2.2', '全部同行预测时滞的中位数 23 天', pl.LAG.median(), 0, '15266 条 LAG 的中位数')
    add('2.2', '7 天以内的占 26%', pl.LAG.between(0, 7).mean() * 100, 0, 'LAG 在 0–7 天的占比 × 100')
    add('2.2', '61 至 90 天的占 15%', pl.LAG.between(61, 90).mean() * 100, 0, 'LAG 在 61–90 天的占比 × 100')
    add('2.2', '96% 的研报发布前 7 天内有同行更新', (rl.LAG_MIN <= 7).mean() * 100, 0, '709 篇中同行最小时滞 ≤ 7 天的占比 × 100（现算）')
    add('2.2', 'PeerLag 中位数 29.4 天', rl.LAG_MEAN.median(), 1, '709 篇 PeerLag（缩尾前）的中位数')
    add('2.2', 'PeerLag 下四分位 21.6 天', rl.LAG_MEAN.quantile(.25), 1, '709 篇 PeerLag（缩尾前）的 25% 分位')
    add('2.2', 'PeerLag 上四分位 37.3 天', rl.LAG_MEAN.quantile(.75), 1, '709 篇 PeerLag（缩尾前）的 75% 分位')
    d = rl[['code', 'RT_NSRC', 'LAG_MEAN']].dropna()
    r, p = corr_p(d.RT_NSRC.to_numpy(float), d.LAG_MEAN.to_numpy(float))
    add('2.2', 'PeerLag 与 RateSrc 的相关系数 −0.03', r, 2, f'709 篇中 RT_NSRC 非缺失的 {len(d)} 篇，两者缩尾前的 Pearson 相关（p = {p:.3f}）')
    d = d[d.groupby('code').code.transform('size') >= 2]
    d = d.assign(**{c: d[c] - d.groupby('code')[c].transform('mean') for c in ('RT_NSRC', 'LAG_MEAN')})
    r, p = corr_p(d.RT_NSRC.to_numpy(float), d.LAG_MEAN.to_numpy(float))
    add('2.2', '按公司去均值后的相关系数 −0.02', r, 2, f'同上，只留至少 2 篇的公司（{len(d)} 篇），按公司去均值后的 Pearson 相关（p = {p:.3f}）')

    # ---- 3.1、3.4、3.6
    add('3.1', 'PeerLag 的 t 值绝对值最大 0.54',
        max(abs(G.cell('report', 'fe_steps', 'PMAFE_FY25', 'firm+broker', x, 'PEERLAG25').t) for x in X6), 2, '表 2 六列 PeerLag 的 |t| 最大值')
    q = G.q(level='report', block='fe_steps', Y='PMAFE_FY24', role='x')
    add('3.4', '含公司固定效应时 t 值绝对值最大 0.76', q[q.fe.isin(['firm', 'firm+broker'])].t.abs().max(), 2, '表 5 公司、公司 + 券商两行 12 格的 |t| 最大值')
    pc = G.cell('firm_broker', 'fe_steps', 'PMAFE_FY25', 'firm+broker', 'WG_NSRC')
    add('3.6', '公司-券商层面 701 个观测', pc.n, 0, 'reg_all firm_broker / fe_steps / PMAFE_FY25 / firm+broker / WG_* 的 n')
    add('3.6', '公司-券商层面 182 家公司', pc.n_clusters, 0, '同上的 n_clusters')
    add('3.6', '公司-券商层面 RT 各列 518 个观测', G.cell('firm_broker', 'fe_steps', 'PMAFE_FY25', 'firm+broker', 'RT_NSRC').n, 0, '同上 RT_* 的 n')
    e = P[P.EVAL_DT25.notna()]
    add('3.6', '被评价的预测正好出自深度研报的占 24%', e.EVAL_IS_DEEP.mean() * 100, 0,
        f'公司-券商层面样本中有被评价预测的 {len(e)} 对（全部样本对，不是 701 个估计样本）里 EVAL_IS_DEEP 的占比 × 100')
    add('3.6', '最后一次预测晚于深度研报的间隔中位数 146 天', e.loc[e.EVAL_GAP > 0, 'EVAL_GAP'].median(), 0, '同上各对中 EVAL_GAP > 0 的中位数（天）')
    q = G.q(level='firm_broker', block='fe_steps', Y='PMAFE_FY25', role='x')
    add('3.6', '含公司固定效应时 t 值绝对值最大 1.14', q[q.fe.isin(['firm', 'firm+broker'])].t.abs().max(), 2, '表 7 公司、公司 + 券商两行的 |t| 最大值')

    # ---- 第 4 节
    for fe in ('firm', 'firm+broker'):
        for x in ('RT_NSRC', 'RT_LNPATH'):
            cell2('4.1', f'去掉 PeerLag：{LABEL[x]}（{FE_ROW[fe]}）', 'no_peerlag', 'PMAFE_FY25', fe, x)
    ts = [G.cell('report', b, 'PMAFE_FY25', fe, 'RT_NSRC').t for b in ('peerlag_median', 'peerlag_npeers') for fe in ('firm', 'firm+broker')]
    add('4.1', 'PeerLag 换成中位数或加 ln 同行家数后 RateSrc 的 t 值下界 −1.87', min(ts), 2,
        'peerlag_median、peerlag_npeers 两块 × 公司、公司 + 券商的 RT_NSRC 四个 t 的最小值（' + '、'.join(f'{t:.3f}' for t in ts) + '）')
    add('4.1', '同上 t 值上界 −1.82', max(ts), 2, '同上四个 t 的最大值')
    for x in X6[:3]:
        for fe in ('firm+month', 'firm+broker+month'):
            c = G.cell('report', 'month_fe', 'PMAFE_FY25', fe, x)
            add('4.1', f'加月份固定效应：{LABEL[x]}（{fe}）在 5% 水平显著（1 = 是）', float(c.p < 0.05), 0, f'month_fe / PMAFE_FY25 / {fe} / {x} 的 p = {c.p:.4f} < 0.05')
    for fe in ('firm+month', 'firm+broker+month'):
        cell2('4.1', f'加月份固定效应：RateSrc（{fe}）', 'month_fe', 'PMAFE_FY25', fe, 'RT_NSRC')
    for x in ('RT_NSRC', 'RT_LNPATH'):
        for fe in ('firm', 'firm+broker'):
            cell2('4.2', f'窗口延伸 30 天：{LABEL[x]}（{FE_ROW[fe]}）', 'window_plus30', 'PMAFEW30_FY25', fe, x)
    for spec in ('RT_NSRC+WG_NSRC', 'RT_LNPATH+WG_LNPATH'):
        x = spec.split('+')[0]
        cell2('4.3', f'Score 下成对放入：{LABEL[x]}（公司 + 券商）', 'paired', 'SCORE_FY25', 'firm+broker', spec, x, scale=1)
    q = G.q(level='report', block='same_sample', Y='PMAFE_FY25', role='x')
    add('4.4', 'WG 变量限 520 篇时 t 值绝对值最大 1.50', q.t.abs().max(), 2, 'same_sample / PMAFE_FY25 / 公司、公司 + 券商 × WG 三个的 |t| 最大值')
    add('4.4', '限定样本 520 篇', G.cell('report', 'same_sample', 'PMAFE_FY25', 'firm+broker', 'WG_NSRC').n, 0, 'same_sample / PMAFE_FY25 / firm+broker 的 n')
    for fe in ('firm', 'firm+broker'):
        cell2('4.5', f'源点数取对数：RateSrc（{FE_ROW[fe]}）', 'log_sources', 'PMAFE_FY25', fe, 'RT_LNSRC')
    for fe in ('firm', 'firm+broker'):
        cell2('4.5', f'源点数取对数、与全文成对放入：RateSrc（{FE_ROW[fe]}）', 'log_sources', 'PMAFE_FY25', fe, 'RT_LNSRC+WG_LNSRC', 'RT_LNSRC')
    lad = G.q(level='report', block='fe_steps', role='x')
    bc = lad[lad.fe.isin(['firm', 'firm+broker'])]
    add('4.6', '表 3、5、6 共 72 个系数', len(lad), 0, 'fe_steps（3 个 Y × 4 档固定效应 × 6 个变量）的 role = x 行数')
    add('4.6', '含公司固定效应的 36 个', len(bc), 0, '其中公司、公司 + 券商两档的行数')
    add('4.6', '其中 10% 水平显著的 8 个', int((bc.p < 0.1).sum()), 0, '这 36 个中 p < 0.10 的个数')
    add('4.6', '全文逻辑结构变量显著的个数 0', int((bc[bc.X.str.startswith('WG')].p < 0.1).sum()), 0, '这 36 个中 WG 变量 p < 0.10 的个数')
    fac = np.sqrt((bc.n - (bc.k - (bc.n_clusters - 1))) / (bc.n - bc.k))
    add('4.7', '不计嵌套的公司固定效应时 t 值约增大两成', float(np.median(fac)), 1,
        f'推导：CR1 的自由度因子为 (N−1)/(N−K)，K 计入全部固定效应虚拟变量；不计嵌套在聚类内的公司固定效应时 K′ = K − (公司数 − 1)，'
        f't 放大 √((N−K′)/(N−K))。表 3、5、6 含公司固定效应的 36 格：{fac.min():.3f}–{fac.max():.3f}，取中位数')
    w = D[D.code == '000858']
    assert w.A25.nunique() == 1 and w.A24.nunique() == 1
    add('4.8', '五粮液 2025 年归母净利润 89.5 亿元', w.A25.iloc[0] / 1e8, 1, '研报层面样本中 code = 000858 的 A25 ÷ 1e8')
    add('4.8', '五粮液 2024 年归母净利润 318.5 亿元', w.A24.iloc[0] / 1e8, 1, '同上的 A24 ÷ 1e8')
    for fe in ('firm', 'firm+broker'):
        cell2('4.8', f'剔除五粮液：RateSrc（{FE_ROW[fe]}）', 'excl_000858', 'PMAFE_FY25', fe, 'RT_NSRC')
    for fe in ('firm', 'firm+broker'):
        cell2('4.8', f'剔除五粮液、与 AllSrc 成对放入：RateSrc（{FE_ROW[fe]}）', 'excl_000858', 'PMAFE_FY25', fe, 'RT_NSRC+WG_NSRC', 'RT_NSRC')
    return pd.DataFrame(rows)
