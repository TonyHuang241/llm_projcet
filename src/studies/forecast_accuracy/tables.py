"""文档的表 1–13 和正文、第 4 节、图注里的统计数字（都从回归样本、数据文件的行、描述性统计、方差分解和 reg_all 长表算出）。

表的 csv（tables/ 下）行列形状同文档里的表，数值是文档显示的口径，保留 6 位小数以便核对：
  系数：RFE、PMED、自算 PMAFE 无量纲，表里乘 100；IFE、IFEALL、RAWFE 是占 D（市值或股价）的比例，乘 100 后是“占市值（或股价）
  的百分点”；自算 Score 不乘（分数）；表 4 是调整 R² × 100。
  系数格后面跟显著性星号（* p < 0.10、** p < 0.05、*** p < 0.01，p 值用 t(公司数 − 1)），t 值在括号里；梯子表（表 6、8、12）和
  不同误差定义表（表 9–11）的一格是“系数星号 (t)”，N 一列为“投资评级概念对应的逻辑结构各列 / 全文逻辑结构各列”的观测数；
  某一格回归样本不足（reg_all 里没有）时写 NA。
被解释变量都用主窗口（sample.MAIN_H，现为 H2）的主变量 Y_MAIN（sample.MAIN_DEF，现为 PMED180_NP）；表 8“其他预测窗口”是其余
两个窗口（sample.OTHER_H，现为 H1、H3）的同名变量 ymain(h) 各一个 Panel，csv 第一列“窗口”区分。下面 Ym 指主变量。

文档表号 -> 文件 -> 函数
  表 1   预测误差变量的描述性统计（H2）    table1_error_descriptive.csv    error_stats、table1（describe 环节）
  表 2   与自算 PMAFE 的一致性（H2 各行）  consistency.csv（全部窗口）      errors.consistency（describe 环节）
         字段层面的一致性和中位数核对        consistency_fields.csv           errors.consistency_fields、errors.median_check
  表 3   逻辑结构变量的描述性统计           table3_descriptive.csv           descriptive_stats、table3（describe 环节）
  表 4   固定效应分解                       table4_fe_decomposition.csv      variance_decomposition、table4（decomp 环节）
  表 5   基准结果                           table5_benchmark.csv             reg_all：report / fe_steps / Ym_Hm / firm+broker
  表 6   不同固定效应                       table6_fe_steps.csv              reg_all：report / fe_steps / Ym_Hm
  表 7   成对放入（Panel A 公司 + 券商，B 公司）table7_paired.csv           reg_all：report / paired / Ym_Hm
  表 8   其他预测窗口                       table8_other_horizons.csv        reg_all：report / fe_steps / Ym_Ho
  表 9   不同误差定义（H2，公司 + 年度 + 券商）table9_definitions_firm_broker.csv  reg_all：fe_steps、definitions、own_measures
  表 10  不同误差定义（H2，公司 + 年度）     table10_definitions_firm.csv     同上
  表 11  不同误差定义（H1、H3，两档固定效应）table11_definitions_other_horizons.csv  同上
  表 12  公司-券商层面                      table12_firm_broker.csv          reg_all：firm_broker / fe_steps / Ym_Hm
  表 13  公司-券商层面成对放入              table13_firm_broker_paired.csv   reg_all：firm_broker / paired / Ym_Hm / firm+broker

text_numbers.csv：正文、图注和第 4 节出现的数字，每个数字一行：位置、内容、数值（原始精度）、小数位（文档显示的位数）、
显示（按小数位四舍五入）、星号、算法（怎么算的）。

预测误差变量的描述性统计（error_stats.csv、表 1）：786 篇基础样本里各列各自的非缺失值（缩尾后 / 缩尾前），统计量同下；表 1 用
缩尾后的值，IFE、IFEALL、RAWFE 乘 100（占 D 的百分点），其余原样。
逻辑结构变量的描述性统计（表 3，descriptive_stats.csv）：源点数、路径数（取对数之前的条数 *_N_PATHS）、平均路径长度三个变量 ×
两类逻辑结构，样本为基准回归（表 5，主变量 Ym_Hm）的估计样本：RT 三个变量用 RT_NSRC 那一列的估计样本，WG 三个用 WG_NSRC
那一列的。descriptive_stats.csv 是长表，每个变量 × 口径（缩尾后 = 回归样本里的列；缩尾前 = <变量>_RAW）一行：N、均值、
标准差（n − 1）、最小值、P1、P5、P25、中位数、P75、P95、P99（pandas quantile，线性插值）、最大值、偏度、超额峰度（pandas skew、
kurt）；表 3 用缩尾后的值。--log1p 时照样对输出目录里的列计算，那时源点数、平均路径长度是 ln(1 + x)，路径数仍是条数。

方差分解（表 4）：每个逻辑结构变量（缩尾后）对一组固定效应虚拟变量 + 常数项做 OLS（不剔除单例），调整 R² =
1 − (1 − R²)(N − 1)/(N − K)，K 为设计矩阵的秩；样本为该变量在基准回归（表 5 那一列）里的估计样本。

各环节读回 csv 用 regressions.read_csv（数值与写出前逐位相同），所以分环节运行和一次跑完的结果一样。
"""
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd

from src.ols import fe_dummies, stars, t_pvalue
from src.studies.forecast_accuracy.errors import DEFS, METRICS, yname
from src.studies.forecast_accuracy.regressions import estimation_sample
from src.studies.forecast_accuracy.sample import (HORIZONS, LAG_MAIN, LNN_MAIN, MAIN_DEF, MAIN_H, MAIN_WIN, N_MAIN, NPEERS_MAIN,
                                                  OTHER_H, OWN_MAIN, SCORE_MAIN, X6, Y_MAIN, hv, ymain)

LABEL = dict(zip(X6, ['RateSrc', 'RatePath', 'RateLen', 'AllSrc', 'AllPath', 'AllLen']))
GROUP = {x: '投资评级概念对应的逻辑结构' if x.startswith('RT') else '全文逻辑结构' for x in X6}
DESC_X = ['RT_NSRC', 'RT_N_PATHS', 'RT_PATHLEN', 'WG_NSRC', 'WG_N_PATHS', 'WG_PATHLEN']  # 表 3 的行
DESC_LABEL = dict(zip(DESC_X, ['源点数（RateSrc）', '路径数（条）', '平均路径长度（RateLen）',
                               '源点数（AllSrc）', '路径数（条）', '平均路径长度（AllLen）']))
QUANTILES = {'P1': .01, 'P5': .05, 'P25': .25, '中位数': .5, 'P75': .75, 'P95': .95, 'P99': .99}
TABLE1_STATS = ['N', '均值', '标准差', '最小值', 'P25', '中位数', 'P75', '最大值']
FE_ROW = {'none': '无', 'industry': '行业', 'firm': '公司', 'firm+broker': '公司 + 券商'}
FE_LABEL = {'firm': '公司 + 年度', 'firm+broker': '公司 + 年度 + 券商'}
DECOMP_SPECS = [  # (名称, 文档列名, 固定效应列；列的先后决定设计矩阵的列序)
    ('month', '(1) 月份', ['month']), ('industry', '(2) 行业', ['sw']), ('firm', '(3) 公司', ['code']),
    ('broker', '(4) 券商', ['broker_g']), ('firm+broker', '(5) 公司 + 券商', ['broker_g', 'code']),
    ('month+firm+broker', '(6) 月份 + 公司 + 券商', ['month', 'broker_g', 'code']),
]
TABLE5_TERMS = [('G', None), ('Size', 'SIZE_M'), ('BM', 'BM'), ('Mom', 'MOM'), ('Vol', 'VOL'), ('ROA', 'ROA_T'), ('LnAge', 'LNAGE'),
                ('PeerLag', LAG_MAIN)]
PAIRS = ['RT_NSRC+WG_NSRC', 'RT_LNPATH+WG_LNPATH', 'RT_PATHLEN+WG_PATHLEN']
PAIR_COLS = ['(1) 源点数', '(2) 路径数', '(3) 平均路径长度']
FILES = {1: 'table1_error_descriptive.csv', 2: 'consistency.csv', 3: 'table3_descriptive.csv', 4: 'table4_fe_decomposition.csv',
         5: 'table5_benchmark.csv', 6: 'table6_fe_steps.csv', 7: 'table7_paired.csv', 8: 'table8_other_horizons.csv',
         9: 'table9_definitions_firm_broker.csv', 10: 'table10_definitions_firm.csv',
         11: 'table11_definitions_other_horizons.csv', 12: 'table12_firm_broker.csv', 13: 'table13_firm_broker_paired.csv'}
FIELDS_FILE = 'consistency_fields.csv'
NUMBERS_FILE = 'text_numbers.csv'
PCT_DEFS = {'IFE75', 'IFE180', 'IFEALL75', 'IFEALL180', 'RAWFE'}  # 占 D 的比例：表 1 乘 100


def rnd(x, nd):
    """四舍五入（十进制、半数进位）。"""
    return str(Decimal(repr(float(x))).quantize(Decimal(1).scaleb(-nd), rounding=ROUND_HALF_UP))


def defn_of(y):
    """被解释变量的定义名：PMED180_NP_H2 -> PMED180；PMAFE_H2 -> PMAFE；SCORE_H2 -> SCORE。"""
    return y.split('_')[0]


def scale_of(y):
    """表里系数的倍数：Score 不乘，其余乘 100。"""
    return 1 if defn_of(y) == 'SCORE' else 100


def def_rows(h):
    """“不同误差定义”表的行：数据文件 9 个定义 × 2 个口径（每个口径内主变量的定义 MAIN_DEF 在最前，主变量本身在第一行）；
    主窗口再加自算的 PMAFE、SCORE
    （own_measures 块只估计了主窗口）。返回 [(变量, 块, 定义, 口径)]。"""
    out = []
    keys = [MAIN_DEF] + [k for k in DEFS if k != MAIN_DEF]
    for mk in METRICS:
        for k in keys:
            y = yname(k, mk, h)
            out.append((y, 'fe_steps' if (k, mk) == (MAIN_DEF, 'NP') else 'definitions', k, mk))
    if h == MAIN_H:
        out += [(hv('PMAFE', h), 'own_measures', 'PMAFE（自算）', 'NP'), (hv('SCORE', h), 'own_measures', 'Score（自算）', 'NP')]
    return out


# ---------------------------------------------------------------- 描述性统计（表 1、表 3）
def stats_row(z):
    z = z.astype(float).dropna()
    q = z.quantile(list(QUANTILES.values()))
    return {'N': int(z.count()), '均值': z.mean(), '标准差': z.std(), '最小值': z.min(), **{k: q[v] for k, v in QUANTILES.items()},
            '最大值': z.max(), '偏度': z.skew(), '超额峰度': z.kurt()}


def error_stats(D):
    """预测误差变量（数据文件 54 列、自算 PMAFE_Hh、SCORE_Hh）和同行条数在 786 篇基础样本里的描述性统计长表。"""
    from src.studies.forecast_accuracy.errors import ALL_Y
    rows = []
    for y in ALL_Y + [hv(v, h) for h in HORIZONS for v in ('PMAFE', 'SCORE')]:
        for kind, col in (('缩尾后', y), ('缩尾前', y + '_RAW')):
            if col in D.columns:
                rows.append({'变量': y, '口径': kind, '列': col, **stats_row(D[col])})
    for y in [f'{v}_{m}_H{h}' for h in HORIZONS for m in METRICS for v in ('N75', 'N180')] + [hv('N_PEERS', h) for h in HORIZONS]:
        rows.append({'变量': y, '口径': '原值', '列': y, **stats_row(D[y])})
    return pd.DataFrame(rows)


def table1(es, h=MAIN_H):
    """表 1：H2 的各定义 × 口径、自算 PMAFE 和 Score、同行条数（缩尾后；IFE、IFEALL、RAWFE 乘 100）。"""
    z = es.set_index(['变量', '口径'])
    rows = []
    for y, _, k, mk in def_rows(h):
        sc = 100 if k in PCT_DEFS else 1
        r = z.loc[(y, '缩尾后')]
        rows.append({'变量': y, '定义': k, '口径': mk,
                     **{s: str(int(r[s])) if s == 'N' else f'{r[s] * sc:.6f}' for s in TABLE1_STATS}})
    for y, k, mk in ((f'N75_NP_H{h}', 'n75（条）', 'NP'), (f'N180_NP_H{h}', 'n180（条）', 'NP'), (hv('N_PEERS', h), '自算同行家数', 'NP')):
        r = z.loc[(y, '原值')]
        rows.append({'变量': y, '定义': k, '口径': mk, **{s: str(int(r[s])) if s == 'N' else f'{r[s]:.6f}' for s in TABLE1_STATS}})
    return pd.DataFrame(rows)


def descriptive_stats(D):
    """六个逻辑结构变量 × 口径（缩尾后、缩尾前）的描述性统计长表（样本见模块说明）。"""
    est = {g: estimation_sample(D, x=g + '_NSRC') for g in ('RT', 'WG')}
    rows = []
    for x in DESC_X:
        for kind, col in (('缩尾后', x), ('缩尾前', x + '_RAW')):
            rows.append({'变量': x, '口径': kind, '列': col, **stats_row(est[x[:2]][col])})
    return pd.DataFrame(rows)


def table3(desc):
    rows = []
    for x in DESC_X:
        z = desc[(desc.变量 == x) & (desc.口径 == '缩尾后')].iloc[0]
        rows.append({'逻辑结构': GROUP[x] if x in ('RT_NSRC', 'WG_NSRC') else '', '变量': DESC_LABEL[x],
                     **{k: str(int(z[k])) if k == 'N' else f'{z[k]:.6f}' for k in TABLE1_STATS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 方差分解（表 4）
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


def table4(dec):
    rows = []
    for x in X6:
        z = dec[dec.variable == x].set_index('spec')
        rows.append({'逻辑结构': GROUP[x] if x in ('RT_NSRC', 'WG_NSRC') else '', '变量': LABEL[x],
                     **{col: f'{z.loc[spec, "adj_r2"] * 100:.6f}' for spec, col, _ in DECOMP_SPECS}})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 回归表
class Reg:
    """reg_all 长表的取格子工具。"""

    def __init__(self, reg):
        self.reg = reg

    def q(self, **kw):
        m = np.ones(len(self.reg), dtype=bool)
        for k, v in kw.items():
            m &= (self.reg[k] == v).to_numpy()
        return self.reg[m]

    def cell(self, level, block, y, fe, X, term=None, missing_ok=False):
        """一格（一个系数）；missing_ok=True 时这一回归不在 reg_all 里（样本不足、src.ols.fit 没有结果）返回 None。"""
        r = self.q(level=level, block=block, Y=y, fe=fe, X=X)
        r = r[r.term == (term or X)]
        if missing_ok and not len(r):
            return None
        assert len(r) == 1, (level, block, y, fe, X, term, len(r))
        return r.iloc[0]

    def n_of(self, level, block, y, fe, xs):
        cs = [self.cell(level, block, y, fe, x, missing_ok=True) for x in xs]
        ns = sorted({int(c.n) for c in cs if c is not None})
        return 'NA' if not ns else str(ns[0]) if len(ns) == 1 else ','.join(map(str, ns))


def coef_str(c, scale):
    return f'{c.coef * scale:.6f}{stars(c.p)}'


def cell_str(c, scale):
    return 'NA' if c is None else f'{coef_str(c, scale)} ({c.t:.6f})'


def table5(G):
    cols = {x: f'({i}) {GROUP[x]} {LABEL[x]}' for i, x in enumerate(X6, 1)}
    rows = []
    for lab, term in TABLE5_TERMS:
        cs = {x: G.cell('report', 'fe_steps', Y_MAIN, 'firm+broker', x, term) for x in X6}
        rows.append({'': lab, **{cols[x]: coef_str(c, 100) for x, c in cs.items()}})
        rows.append({'': '(t)', **{cols[x]: f'({c.t:.6f})' for x, c in cs.items()}})
    cs = {x: G.cell('report', 'fe_steps', Y_MAIN, 'firm+broker', x) for x in X6}
    rows.append({'': '调整 R²', **{cols[x]: f'{c.adj_r2:.6f}' for x, c in cs.items()}})
    rows.append({'': 'N', **{cols[x]: str(int(c.n)) for x, c in cs.items()}})
    for lab in ('券商固定效应', '公司固定效应', '年度固定效应'):
        rows.append({'': lab, **{cols[x]: '是' for x in X6}})
    return pd.DataFrame(rows)


def ladder(G, level, block, y, scale):
    """表 6、8、12：行 = 固定效应四档，列 = 六个变量 + N；回归样本不足（reg_all 里没有）的格子写 NA。"""
    rows = []
    for fe, lab in FE_ROW.items():
        r = {'固定效应': lab}
        for x in X6:
            r[f'{GROUP[x]} {LABEL[x]}'] = cell_str(G.cell(level, block, y, fe, x, missing_ok=True), scale)
        r['N'] = f'{G.n_of(level, block, y, fe, X6[:3])} / {G.n_of(level, block, y, fe, X6[3:])}'
        rows.append(r)
    return pd.DataFrame(rows)


def other_horizons(G):
    """表 8：其他预测窗口（OTHER_H 依次为 Panel A、B）的主变量 ymain(h)，设定同表 6；第一列“窗口”标明 Panel 和窗口。"""
    parts = [ladder(G, 'report', 'fe_steps', ymain(h), 100).assign(窗口=f'Panel {"AB"[i]}：H{h}')
             for i, h in enumerate(OTHER_H)]
    t = pd.concat(parts, ignore_index=True)
    return t[['窗口'] + [c for c in t.columns if c != '窗口']]


def paired(G, level, block, ctrl_label, fes=('firm+broker',)):
    """表 7、13：两类逻辑结构同名变量成对放入（主变量）；fes 有多档时每档一个 Panel（第一列“固定效应”）。"""
    parts = []
    for fe in fes:
        cs = {spec: (G.cell(level, block, Y_MAIN, fe, spec, spec.split('+')[0]),
                     G.cell(level, block, Y_MAIN, fe, spec, spec.split('+')[1])) for spec in PAIRS}
        rows = []
        for i, lab in ((0, '投资评级概念对应的逻辑结构'), (1, '全文逻辑结构')):
            rows.append({'': lab, **{col: coef_str(cs[s][i], 100) for s, col in zip(PAIRS, PAIR_COLS)}})
            rows.append({'': '(t)', **{col: f'({cs[s][i].t:.6f})' for s, col in zip(PAIRS, PAIR_COLS)}})
        for lab in (ctrl_label, '公司固定效应') + (('券商固定效应',) if fe == 'firm+broker' else ()):
            rows.append({'': lab, **{col: '是' for col in PAIR_COLS}})
        rows.append({'': '调整 R²', **{col: f'{cs[s][0].adj_r2:.6f}' for s, col in zip(PAIRS, PAIR_COLS)}})
        rows.append({'': 'N', **{col: str(int(cs[s][0].n)) for s, col in zip(PAIRS, PAIR_COLS)}})
        parts.append(pd.DataFrame(rows).assign(固定效应=FE_LABEL[fe]))
    t = pd.concat(parts, ignore_index=True)
    return t if len(fes) > 1 else t.drop(columns='固定效应')


def definitions(G, h, fe):
    """表 9–11 的一块：窗口 h、固定效应 fe 下各定义 × 口径（加自算 PMAFE、SCORE）一行，列 = 六个 X + N。"""
    rows = []
    for y, block, k, mk in def_rows(h):
        r = {'变量': y, '定义': k, '口径': mk}
        for x in X6:
            r[f'{GROUP[x]} {LABEL[x]}'] = cell_str(G.cell('report', block, y, fe, x, missing_ok=True), scale_of(y))
        r['N'] = f'{G.n_of("report", block, y, fe, X6[:3])} / {G.n_of("report", block, y, fe, X6[3:])}'
        rows.append(r)
    return pd.DataFrame(rows)


def definitions_other(G):
    """表 11：H1、H3（OTHER_H）× 两档固定效应的各定义，前两列“窗口”“固定效应”。"""
    parts = [definitions(G, h, fe).assign(窗口=f'H{h}', 固定效应=FE_LABEL[fe]) for h in OTHER_H for fe in ('firm+broker', 'firm')]
    t = pd.concat(parts, ignore_index=True)
    return t[['窗口', '固定效应'] + [c for c in t.columns if c not in ('窗口', '固定效应')]]


def reg_tables(reg):
    """回归表 {表号: DataFrame}（表 5–13；表 1–4 由 describe、decomp 环节生成）。"""
    G = Reg(reg)
    return {5: table5(G), 6: ladder(G, 'report', 'fe_steps', Y_MAIN, 100),
            7: paired(G, 'report', 'paired', '控制变量 X 和 PeerLag', ('firm+broker', 'firm')),
            8: other_horizons(G), 9: definitions(G, MAIN_H, 'firm+broker'), 10: definitions(G, MAIN_H, 'firm'),
            11: definitions_other(G), 12: ladder(G, 'firm_broker', 'fe_steps', Y_MAIN, 100),
            13: paired(G, 'firm_broker', 'paired', '控制变量 X 和 DAge')}


# ---------------------------------------------------------------- 正文数字
def corr_p(x, y):
    """Pearson r 和双侧 p 值（t = r·sqrt((n−2)/(1−r²))，t(n−2)）。"""
    r = float(np.corrcoef(x, y)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 2) / (1 - r * r)) if abs(r) < 1 else np.inf
    return r, t_pvalue(t, n - 2)


def actual_of(D, code, year):
    """研报层面样本里公司 code 在财年 year 的归母净利润实际值：在各窗口的列里找目标年度 TY_Hh == year 的 A_Hh（窗口和 T 不同的研报
    取到的都是同一个财年的值，assert 唯一）。"""
    w = D[D.code == code]
    vals = pd.concat([w.loc[w[hv('TY', h)] == year, hv('A', h)] for h in HORIZONS]).dropna().unique()
    assert len(vals) == 1, (code, year, vals)
    return float(vals[0])


def text_numbers(S, D, P, reg, desc, dec, est, pl, rl, schematic):
    """S 研究范围，D 研报层面样本，P 公司-券商层面样本，reg 回归长表，desc 逻辑结构变量的描述性统计，dec 方差分解，
    est 基准回归估计样本，pl / rl 图 2 的同行明细和每篇研报，schematic 图 1 例子的指标。返回 text_numbers.csv 的 DataFrame。
    “内容”一栏不写具体数字（数字在“数值”“显示”里），主窗口的变量名都由 sample.MAIN_H 派生。"""
    from src.graph.load import read_table
    from src.measure.variables import RATING_NODE
    G = Reg(reg)
    rows = []
    Hm = f'H{MAIN_H}'

    def add(loc, item, value, nd, how, star=None):
        v = float(value)
        rows.append({'位置': loc, '内容': item, '数值': v, '小数位': nd, '显示': rnd(v, nd) if np.isfinite(v) else 'NA',
                     '星号': '' if star is None else star, '算法': how})

    def cell2(loc, item, block, y, fe, X, term=None, scale=100, level='report'):
        c = G.cell(level, block, y, fe, X, term)
        add(loc, item + ' 系数', c.coef * scale, 2, f'reg_all {level} / {block} / {y} / {fe} / X={X} / term={term or X} 的 coef'
            + (' × 100' if scale == 100 else ''), stars(c.p))
        add(loc, item + ' t', c.t, 2, '同上的 t（按公司聚类）')

    # ---- 1.1、1.2 样本
    rid = list(S.report_id)
    n_est = len(est)
    base = G.cell('report', 'fe_steps', Y_MAIN, 'firm+broker', 'WG_NSRC')
    rate = G.cell('report', 'fe_steps', Y_MAIN, 'firm+broker', 'RT_NSRC')
    add('1.1', '样本研报篇数（研究范围）', len(S), 0, '研究范围：公司在覆盖名单（data/hs300_2024_dag_covered.csv）里、graph 研报表 n_edges > 0 的研报篇数')
    add('1.1', '基础样本篇数', len(D), 0, '研报层面样本（匹配到本篇预测、有 H1 预测和实际值、发布于 H1 财年年报公告之前）')
    add('1.1', f'满足条件的研报篇数（主变量 {Y_MAIN} 非缺失）', int(D[Y_MAIN].notna().sum()), 0, f'基础样本中 {Y_MAIN} 非缺失的篇数')
    add('1.1（对照，非文档数字）', f'自算 {OWN_MAIN} 非缺失的篇数', int(D[OWN_MAIN].notna().sum()), 0, f'基础样本中 {OWN_MAIN} 非缺失的篇数')
    add('1.1', '基准回归样本篇数', base.n, 0, f'reg_all report / fe_steps / {Y_MAIN} / firm+broker / WG_* 的 n')
    add('1.1', '基准回归公司数', base.n_clusters, 0, '同上的 n_clusters')
    for T, g in est.groupby('RPT_YR_T'):
        add('1.1（对照，非文档数字）', f'基准回归样本中 T = {int(T)}（{Hm} 目标年度 {int(T) + MAIN_H}）的篇数', len(g), 0,
            f'基准回归估计样本中 RPT_YR_T = {int(T)} 的篇数')
    C = read_table('concepts', report_ids=rid, columns=['name'])
    n_has = C[C.name == RATING_NODE].report_id.nunique()
    add('1.2', '抽取出“投资评级”概念的研报篇数', n_has, 0,
        f'graph 概念表（src.graph.load.read_table("concepts")，各次运行按 RUN_ORDER 合并，含没有进图边的孤立概念）中 name == “投资评级” 的研报数（研究范围 {len(S)} 篇内）')
    add('1.2', '抽取出“投资评级”概念的研报占研究范围的比例（%）', n_has / len(S) * 100, 0, '上一行 ÷ 研究范围篇数 × 100')
    for name, key, nd, val in (('全文', '全文', 0, 'NSRC'), ('全文', '全文', 0, 'N_SINKS'), ('全文', '全文', 0, 'N_PATHS'), ('全文', '全文', 1, 'PATHLEN'),
                               ('投资评级', '投资评级', 0, 'NSRC'), ('投资评级', '投资评级', 0, 'N_PATHS'), ('投资评级', '投资评级', 1, 'PATHLEN')):
        lab = {'NSRC': '源点数', 'N_SINKS': '汇点数', 'N_PATHS': '路径数', 'PATHLEN': '平均路径长度'}[val]
        v = schematic[key]['N_SINKS'] if val == 'N_SINKS' else schematic[key][val]
        add('1.2 图 1 的例子', f'{name}逻辑结构的{lab}', v, nd, '图 1 虚构小图上 src.measure.variables.graph_measures 的结果（汇点数 = 严格汇点个数）')
    add('1.2', '“投资评级”是汇点的研报篇数', int(S.RT_NSRC.notna().sum()), 0, '研究范围中 measure 的 RT_NSRC 非缺失（投资评级是严格汇点）的篇数')
    add('1.2', '基准回归中 RT 各列的篇数', rate.n, 0, f'reg_all report / fe_steps / {Y_MAIN} / firm+broker / RT_* 的 n')
    add('1.2', '基准回归中 RT 各列的公司数', rate.n_clusters, 0, '同上的 n_clusters')

    # ---- 1.3 窗口与财年、同行条数
    for T, g in D.groupby('RPT_YR_T'):
        T = int(T)
        add('1.3 窗口与财年', f'预测从 {T + 1} 财年开始（RPT_YR_T = {T}）的研报篇数（基础样本）', len(g), 0, f'研报层面样本（基础样本 {len(D)} 篇）中 RPT_YR_T = {T} 的篇数')
        mon = g.date.str[:6]
        assert set(mon.str[:4]) == {'2024'}, (T, sorted(set(mon)))
        add('1.3 窗口与财年', f'预测从 {T + 1} 财年开始的研报最早发布月份（2024 年的第几月）', int(mon.min()[4:]), 0, '同上各篇发布日（均在 2024 年）的最小月份')
        add('1.3 窗口与财年', f'预测从 {T + 1} 财年开始的研报最晚发布月份（2024 年的第几月）', int(mon.max()[4:]), 0, '同上各篇发布日的最大月份')
        for h in HORIZONS:
            add('1.3 窗口与财年', f'预测从 {T + 1} 财年开始的研报中 H{h}（{T + h} 财年）的主变量 {ymain(h)} 非缺失篇数', int(g[ymain(h)].notna().sum()), 0,
                f'同上各篇中 {ymain(h)} 非缺失的篇数')
            add('1.3 窗口与财年（对照，非文档数字）', f'预测从 {T + 1} 财年开始的研报中 H{h} 的自算 PMAFE 非缺失篇数', int(g[hv('PMAFE', h)].notna().sum()), 0,
                f'同上各篇中 {hv("PMAFE", h)} 非缺失的篇数')
    add('1.3', 'n75（净利润，H2）的中位数（条）', est[hv('N75_NP')].dropna().median(), 0, f'基准回归估计样本（{n_est} 篇）{hv("N75_NP")} 的中位数')
    add('1.3', 'n180（净利润，H2）的中位数（条）', est[hv('N180_NP')].dropna().median(), 0, f'基准回归估计样本 {hv("N180_NP")} 的中位数')
    add('1.3', '自算比较组同行家数的中位数', est[NPEERS_MAIN].dropna().median(), 0, f'基准回归估计样本 {NPEERS_MAIN} 的中位数')
    d = D[D[Y_MAIN].notna()]
    add('1.3', f'主变量非缺失研报中 n{MAIN_WIN} ≤ 2 的篇数', int((d[N_MAIN] <= 2).sum()), 0, f'基础样本中 {Y_MAIN} 非缺失且 {N_MAIN} ≤ 2 的篇数（数据文件不要求至少 3 条同行）')
    add('1.3', '主变量缺失、自算 PMAFE 非缺失的篇数', int((D[Y_MAIN].isna() & D[OWN_MAIN].notna()).sum()), 0, f'基础样本中 {Y_MAIN} 缺失而 {OWN_MAIN} 非缺失的篇数')
    add('1.3', '主变量非缺失、自算 PMAFE 缺失的篇数', int((D[Y_MAIN].notna() & D[OWN_MAIN].isna()).sum()), 0, f'基础样本中 {Y_MAIN} 非缺失而 {OWN_MAIN} 缺失的篇数')

    # ---- 1.4 被解释变量的量级
    add('1.4', f'{Y_MAIN} 的标准差', est[Y_MAIN].std(), 2, f'基准回归估计样本（{n_est} 篇）缩尾后 {Y_MAIN} 的标准差')
    add('1.4', f'{Y_MAIN} 的标准差 × 100', est[Y_MAIN].std() * 100, 0, '同上 × 100（表里系数乘 100 后的单位）')
    add('1.4', f'自算 {OWN_MAIN} 的标准差 × 100', est[OWN_MAIN].dropna().std() * 100, 0, f'同一样本中缩尾后 {OWN_MAIN} 的标准差 × 100')
    add('1.4', f'{Y_MAIN} 与自算 {OWN_MAIN} 的相关系数', est[[Y_MAIN, OWN_MAIN]].dropna().corr().iloc[0, 1], 2, f'同一样本两两非缺失的 Pearson 相关')
    add('1.4', f'{Y_MAIN} 与自算 {SCORE_MAIN} 的相关系数', est[[Y_MAIN, SCORE_MAIN]].dropna().corr().iloc[0, 1], 2, '同上')

    # ---- 2.1 描述性统计（表 3）
    ds = desc.set_index(['变量', '口径'])
    ws = lambda x, k: ds.loc[(x, '缩尾后'), k]  # noqa: E731
    add('表 3 注', 'RT 各行的研报篇数', ws('RT_NSRC', 'N'), 0, 'descriptive_stats.csv RT_NSRC（缩尾后）的 N：基准回归 RT_NSRC 那一列的估计样本')
    add('表 3 注', 'WG 各行的研报篇数', ws('WG_NSRC', 'N'), 0, 'descriptive_stats.csv WG_NSRC（缩尾后）的 N：基准回归 WG_NSRC 那一列的估计样本')
    for x, unit, nd in (('RT_NSRC', '个', 0), ('WG_NSRC', '个', 0), ('RT_N_PATHS', '条', 0), ('WG_N_PATHS', '条', 0)):
        add('2.1', f'{x} 的中位数（{unit}）', ws(x, '中位数'), nd, f'descriptive_stats.csv {x}（缩尾后）的中位数（= 表 3 中位数列）')
    rt = estimation_sample(D, x='RT_NSRC')
    add('2.1', '只有 1 个源点的研报占比（%）', (rt.RT_NSRC_RAW == 1).mean() * 100, 0,
        f'基准回归 RT_NSRC 那一列的估计样本（{len(rt)} 篇）中 RT_NSRC_RAW（缩尾前）== 1 的占比 × 100')
    for x in ('RT_PATHLEN', 'WG_PATHLEN'):
        add('2.1', f'{x} 的中位数', ws(x, '中位数'), 2, f'descriptive_stats.csv {x}（缩尾后）的中位数（= 表 3 中位数列）')

    # ---- 2.2 方差分解（表 4）
    d3 = dec.set_index(['variable', 'spec']).adj_r2 * 100
    add('2.2', '月份固定效应调整 R² 的最大值（%）', max(d3[(x, 'month')] for x in X6), 1, '表 4 第 (1) 列六个变量的最大值')
    for spec, lab in (('firm', '公司'), ('broker', '券商'), ('firm+broker', '公司 + 券商')):
        vals = [d3[(x, spec)] for x in X6[:3]]
        add('2.2', f'RT 三个变量{lab}固定效应的调整 R² 下限', min(vals), 1, f'表 4 {lab} 列 RT 三行的最小值')
        add('2.2', f'RT 三个变量{lab}固定效应的调整 R² 上限', max(vals), 1, f'表 4 {lab} 列 RT 三行的最大值')
    add('2.2', 'AllSrc 公司 + 券商固定效应的调整 R²（%）', d3[('WG_NSRC', 'firm+broker')], 1, '表 4 AllSrc 第 (5) 列')
    add('2.2', 'AllPath 公司 + 券商固定效应的调整 R²（%）', d3[('WG_LNPATH', 'firm+broker')], 1, '表 4 AllPath 第 (5) 列')
    cx = est[X6].corr()
    add('2.2', 'RateSrc 与 RatePath 的相关系数', cx.loc['RT_NSRC', 'RT_LNPATH'], 2,
        f'基准回归估计样本（{n_est} 篇；缩尾后，两两非缺失，RT 变量非缺失 {int(est.RT_NSRC.notna().sum())} 篇）Pearson 相关')
    add('2.2', 'RateSrc 与 RateLen 的相关系数', cx.loc['RT_NSRC', 'RT_PATHLEN'], 2, '同上')

    # ---- 图 2、2.3
    add('图 2 注', '样本研报篇数', rl.report_id.nunique(), 0, '图 2 的研报 = 基准回归估计样本')
    add('图 2 注', '同行预测条数', len(pl), 0,
        f'这些研报在 accuracy report_peers 里 h = {MAIN_H}（目标年度 TY_{Hm}）、[d − 90, d] 比较组的同行条数')
    add('2.3', '全部同行预测时滞的中位数（天）', pl.LAG.median(), 0, f'{len(pl)} 条 LAG 的中位数')
    add('2.3', '时滞 7 天以内的占比（%）', pl.LAG.between(0, 7).mean() * 100, 0, 'LAG 在 0–7 天的占比 × 100')
    add('2.3', '时滞 61 至 90 天的占比（%）', pl.LAG.between(61, 90).mean() * 100, 0, 'LAG 在 61–90 天的占比 × 100')
    add('2.3', '发布前 7 天内有同行更新的研报占比（%）', (rl.LAG_MIN <= 7).mean() * 100, 0, f'{len(rl)} 篇中同行最小时滞 ≤ 7 天的占比 × 100（现算）')
    add('2.3', 'PeerLag 中位数（天）', rl.LAG_MEAN.median(), 1, f'{len(rl)} 篇 PeerLag（缩尾前）的中位数')
    add('2.3', 'PeerLag 下四分位（天）', rl.LAG_MEAN.quantile(.25), 1, f'{len(rl)} 篇 PeerLag（缩尾前）的 25% 分位')
    add('2.3', 'PeerLag 上四分位（天）', rl.LAG_MEAN.quantile(.75), 1, f'{len(rl)} 篇 PeerLag（缩尾前）的 75% 分位')
    d = rl[['code', 'RT_NSRC', 'LAG_MEAN']].dropna()
    r, p = corr_p(d.RT_NSRC.to_numpy(float), d.LAG_MEAN.to_numpy(float))
    add('2.3', 'PeerLag 与 RateSrc 的相关系数', r, 2, f'{len(rl)} 篇中 RT_NSRC 非缺失的 {len(d)} 篇，两者缩尾前的 Pearson 相关（p = {p:.3f}）')
    add('2.3', 'PeerLag 与 RateSrc 的相关系数的 p 值', p, 3, '上一行的双侧 p 值（t(n − 2)）')
    d = d[d.groupby('code').code.transform('size') >= 2]
    d = d.assign(**{c: d[c] - d.groupby('code')[c].transform('mean') for c in ('RT_NSRC', 'LAG_MEAN')})
    r, p = corr_p(d.RT_NSRC.to_numpy(float), d.LAG_MEAN.to_numpy(float))
    add('2.3', '按公司去均值后的相关系数', r, 2, f'同上，只留至少 2 篇的公司（{len(d)} 篇），按公司去均值后的 Pearson 相关（p = {p:.3f}）')
    add('2.3', '按公司去均值后的相关系数的 p 值', p, 3, '上一行的双侧 p 值（t(n − 2)）')

    # ---- 3.1、3.4
    add('3.1', 'PeerLag 的 t 值绝对值最大值',
        max(abs(G.cell('report', 'fe_steps', Y_MAIN, 'firm+broker', x, LAG_MAIN).t) for x in X6), 2, '表 5 六列 PeerLag 的 |t| 最大值')
    for i, h in enumerate(OTHER_H):
        q = G.q(level='report', block='fe_steps', Y=ymain(h), role='x')
        qf = q[q.fe.isin(['firm', 'firm+broker'])]
        add('3.4', f'表 8 Panel {"AB"[i]}（H{h}）含公司固定效应时 t 值绝对值最大值', qf.t.abs().max() if len(qf) else np.nan, 2,
            f'表 8 Panel {"AB"[i]}（{ymain(h)}）公司、公司 + 券商两行 {len(qf)} 格的 |t| 最大值')
        add('3.4', f'表 8 Panel {"AB"[i]}（H{h}）含公司固定效应时 10% 水平显著的格数', int((qf.p < 0.1).sum()), 0, f'同上 {len(qf)} 格中 p < 0.10 的个数')

    # ---- 3.5 不同误差定义：每个窗口 × 固定效应 × 定义族的显著格数
    for h in HORIZONS:
        for fe in ('firm+broker', 'firm'):
            for fam, keys in (('RFE', ('RFE75', 'RFE180')), ('IFE', ('IFE75', 'IFE180', 'IFEALL75', 'IFEALL180')), ('RAWFE', ('RAWFE',)),
                              ('PMED', ('PMED75', 'PMED180'))):
                ys = [yname(k, m, h) for m in METRICS for k in keys]
                q = G.reg[(G.reg.level == 'report') & G.reg.block.isin(['fe_steps', 'definitions']) & G.reg.Y.isin(ys)
                          & (G.reg.fe == fe) & (G.reg.role == 'x')]
                for grp, xs in (('RT', X6[:3]), ('WG', X6[3:])):
                    qq = q[q.X.isin(xs)]
                    add(f'3.5（H{h}）', f'{fam} 族（{len(ys)} 个变量）{FE_LABEL[fe]} 下 {grp} 三个变量 10% 水平显著的格数', int((qq.p < 0.1).sum()), 0,
                        f'{"、".join(ys)} × {"、".join(xs)} 共 {len(qq)} 格（无结果的不计）中 p < 0.10 的个数')
                    add(f'3.5（H{h}）', f'{fam} 族 {FE_LABEL[fe]} 下 {grp} 三个变量的格数', len(qq), 0, '同上的格数')
                    add(f'3.5（H{h}）', f'{fam} 族 {FE_LABEL[fe]} 下 {grp} 三个变量系数为负的格数', int((qq.coef < 0).sum()), 0, '同上系数 < 0 的个数')

    # ---- 3.6 公司-券商层面
    pc = G.cell('firm_broker', 'fe_steps', Y_MAIN, 'firm+broker', 'WG_NSRC')
    add('3.6', '公司-券商层面观测数', pc.n, 0, f'reg_all firm_broker / fe_steps / {Y_MAIN} / firm+broker / WG_* 的 n')
    add('3.6', '公司-券商层面公司数', pc.n_clusters, 0, '同上的 n_clusters')
    add('3.6', '公司-券商层面 RT 各列观测数', G.cell('firm_broker', 'fe_steps', Y_MAIN, 'firm+broker', 'RT_NSRC').n, 0, '同上 RT_* 的 n')
    e = P[P[hv('EVAL_DT')].notna()]
    add('3.6', '样本对数', len(P), 0, '公司-券商层面样本对数')
    add('3.6', '有被评价预测的样本对数', len(e), 0, f'公司-券商层面样本（{len(P)} 对）中 {hv("EVAL_DT")} 非缺失的对数')
    add('3.6', '被评价预测在数据文件里找到对应行的对数', int(P.EVAL_FILE_FOUND.sum()), 0,
        'EVAL_RPT_ID 大写相同、sec == code、PROFIT、target_year == TY 的行存在的对数')
    add('3.6', f'数据文件的 {Y_MAIN} 非缺失的对数', int(P[Y_MAIN].notna().sum()), 0, f'公司-券商层面样本中 {Y_MAIN} 非缺失的对数')
    add('3.6', '匹配到的行 horizon = 2 的对数', int((P.EVAL_FILE_HORIZON == MAIN_H).sum()), 0, '匹配到的行中 horizon == MAIN_H 的对数')
    add('3.6（对照，非文档数字）', f'自算 {OWN_MAIN} 非缺失的对数', int(P[OWN_MAIN].notna().sum()), 0, f'公司-券商层面样本中自算 {OWN_MAIN} 非缺失的对数')
    em = e[hv('EVAL_DT')].str[:6] == '202412'
    add('3.6', '被评价预测发布于 2024 年 12 月的对数', int(em.sum()), 0, f'有被评价预测的 {len(e)} 对中 {hv("EVAL_DT")} 在 2024 年 12 月的对数')
    add('3.6', '其中在数据文件里找到的对数', int(e.loc[em, 'EVAL_FILE_FOUND'].sum()), 0, '同上各对中 EVAL_FILE_FOUND 的对数')
    add('3.6', '被评价预测发布于 2024 年 1–11 月的对数', int((~em).sum()), 0, '同上，1–11 月')
    add('3.6', '其中在数据文件里找到的对数（1–11 月）', int(e.loc[~em, 'EVAL_FILE_FOUND'].sum()), 0, '同上各对中 EVAL_FILE_FOUND 的对数')
    add('3.6', '被评价的预测正好出自深度研报的占比（%）', e.EVAL_IS_DEEP.mean() * 100, 0,
        f'公司-券商层面样本中有被评价预测的 {len(e)} 对（全部样本对，不是估计样本）里 EVAL_IS_DEEP（{Hm} 的被评价预测）的占比 × 100')
    est_p = P[P[Y_MAIN].notna() & P.WG_NSRC.notna()]
    add('3.6', f'{Y_MAIN} 非缺失的对中被评价预测出自深度研报的占比（%）', est_p.EVAL_IS_DEEP.mean() * 100, 0,
        f'{Y_MAIN} 非缺失的 {len(est_p)} 对里 EVAL_IS_DEEP 的占比 × 100')
    add('3.6', '最后一次预测晚于深度研报的间隔中位数（天）', e.loc[e.EVAL_GAP > 0, 'EVAL_GAP'].median(), 0, '有被评价预测的各对中 EVAL_GAP > 0 的中位数（天）')
    add('3.6', f'{Y_MAIN} 非缺失的对中间隔中位数（天）', est_p.loc[est_p.EVAL_GAP > 0, 'EVAL_GAP'].median(), 0, f'{Y_MAIN} 非缺失的各对中 EVAL_GAP > 0 的中位数')
    for ty, n in est_p[hv('TY')].value_counts().sort_index().items():
        add('3.6', f'{Y_MAIN} 非缺失的对中被评价预测目标年度为 {int(ty)} 的对数', n, 0, f'同上各对中 {hv("TY")} = {int(ty)} 的对数')
    q = G.q(level='firm_broker', block='fe_steps', Y=Y_MAIN, role='x')
    add('3.6', '含公司固定效应时 t 值绝对值最大值', q[q.fe.isin(['firm', 'firm+broker'])].t.abs().max(), 2, '表 12 公司、公司 + 券商两行的 |t| 最大值')

    # ---- 第 4 节
    for fe in ('firm', 'firm+broker'):
        for x in X6[:3]:
            cell2('4.1', f'去掉 PeerLag：{LABEL[x]}（{FE_ROW[fe]}）', 'no_peerlag', Y_MAIN, fe, x)
    for b in ('peerlag_median', 'peerlag_npeers'):
        for fe in ('firm', 'firm+broker'):
            for x in X6[:3]:
                cell2('4.1', f'{b}：{LABEL[x]}（{FE_ROW[fe]}）', b, Y_MAIN, fe, x)
    for fe in ('firm', 'firm+broker'):
        cell2('4.1', f'peerlag_npeers：ln(n{MAIN_WIN})（{FE_ROW[fe]}，RateSrc 那一列）', 'peerlag_npeers', Y_MAIN, fe, 'RT_NSRC', LNN_MAIN)
    for x in X6[:3]:
        for fe in ('firm+month', 'firm+broker+month'):
            cell2('4.1', f'加月份固定效应：{LABEL[x]}（{fe}）', 'month_fe', Y_MAIN, fe, x)
    for spec in PAIRS:
        for fe in ('firm', 'firm+broker'):
            for j in (0, 1):
                x = spec.split('+')[j]
                cell2('4.3', f'自算 PMAFE 下成对放入：{LABEL[x]}（{FE_ROW[fe]}）', 'paired', OWN_MAIN, fe, spec, x)
    q = G.q(level='report', block='same_sample', Y=Y_MAIN, role='x')
    add('4.4', 'WG 变量限 RT_NSRC 有定义的研报时 t 值绝对值最大值', q.t.abs().max(), 2, f'same_sample / {Y_MAIN} / 公司、公司 + 券商 × WG 三个的 |t| 最大值')
    add('4.4', 'WG 变量限 RT_NSRC 有定义的研报时 10% 水平显著的格数', int((q.p < 0.1).sum()), 0, '同上 6 格中 p < 0.10 的个数')
    for fe in ('firm', 'firm+broker'):
        for x in X6[3:]:
            cell2('4.4', f'WG 变量限 RT_NSRC 有定义的研报：{LABEL[x]}（{FE_ROW[fe]}）', 'same_sample', Y_MAIN, fe, x)
    add('4.4', '限定样本篇数', G.cell('report', 'same_sample', Y_MAIN, 'firm+broker', 'WG_NSRC').n, 0, f'same_sample / {Y_MAIN} / firm+broker 的 n')
    for fe in ('firm', 'firm+broker'):
        cell2('4.5', f'源点数取对数：RateSrc（{FE_ROW[fe]}）', 'log_sources', Y_MAIN, fe, 'RT_LNSRC')
    for fe in ('firm', 'firm+broker'):
        cell2('4.5', f'源点数取对数、与全文成对放入：RateSrc（{FE_ROW[fe]}）', 'log_sources', Y_MAIN, fe, 'RT_LNSRC+WG_LNSRC', 'RT_LNSRC')
    lad = G.q(level='report', block='fe_steps', role='x')
    bc = lad[lad.fe.isin(['firm', 'firm+broker'])]
    ys = list(dict.fromkeys(lad.Y))
    add('4.6', '表 6、8 的系数个数', len(lad), 0,
        f'fe_steps（{len(ys)} 个 Y：{"、".join(ys)} × 4 档固定效应 × 6 个变量；样本不足没有结果的不计）的 role = x 行数')
    add('4.6', '其中含公司固定效应的系数个数', len(bc), 0, '其中公司、公司 + 券商两档的行数')
    add('4.6', '无法估计（样本不足、没有结果）的格数', len(ys) * len(FE_ROW) * len(X6) - len(lad), 0,
        f'{len(ys)} 个 Y × {len(FE_ROW)} 档固定效应 × {len(X6)} 个变量 − 上面的系数个数')
    add('4.6', '其中 10% 水平显著的个数', int((bc.p < 0.1).sum()), 0, f'这 {len(bc)} 个中 p < 0.10 的个数')
    add('4.6', '不加固定效应和只加行业固定效应时 10% 水平显著的个数', int((lad[~lad.fe.isin(['firm', 'firm+broker'])].p < 0.1).sum()), 0,
        'fe_steps 无、行业两档中 p < 0.10 的个数')
    dq = G.reg[(G.reg.level == 'report') & G.reg.block.isin(['fe_steps', 'definitions']) & (G.reg.role == 'x')
               & G.reg.fe.isin(['firm', 'firm+broker'])]
    add('4.6', '数据文件全部定义（54 个 Y）含公司固定效应的系数个数', len(dq), 0, 'fe_steps、definitions 两块 × 公司、公司 + 券商 × 六个 X（无结果的不计）')
    add('4.6', '其中 10% 水平显著的个数', int((dq.p < 0.1).sum()), 0, '同上 p < 0.10 的个数')
    add('4.6', '其中 RT 变量 10% 水平显著的个数', int((dq[dq.X.str.startswith('RT')].p < 0.1).sum()), 0, '同上 RT 三个变量中 p < 0.10 的个数')
    add('4.6', '其中 RT 变量的系数个数', int(dq.X.str.startswith('RT').sum()), 0, '同上 RT 三个变量的格数')
    add('4.6', '其中 WG 变量 10% 水平显著的个数', int((dq[dq.X.str.startswith('WG')].p < 0.1).sum()), 0, '同上 WG 三个变量中 p < 0.10 的个数')
    fac = np.sqrt((bc.n - (bc.k - (bc.n_clusters - 1))) / (bc.n - bc.k))
    add('4.7', '不计嵌套的公司固定效应时 t 值增大的倍数（中位数）', float(np.median(fac)), 1,
        f'推导：CR1 的自由度因子为 (N−1)/(N−K)，K 计入全部固定效应虚拟变量；不计嵌套在聚类内的公司固定效应时 K′ = K − (公司数 − 1)，'
        f't 放大 √((N−K′)/(N−K))。表 6、8 含公司固定效应的 {len(bc)} 格：{fac.min():.3f}–{fac.max():.3f}，取中位数')
    add('4.8', '五粮液 2025 年归母净利润（亿元）', actual_of(D, '000858', 2025) / 1e8, 1,
        '研报层面样本中 code = 000858 的研报在目标年度 TY_Hh = 2025 的窗口上的 A_Hh ÷ 1e8（各篇、各窗口取到的值相同）')
    add('4.8', '五粮液 2024 年归母净利润（亿元）', actual_of(D, '000858', 2024) / 1e8, 1, '同上，目标年度 = 2024')
    for fe in ('firm', 'firm+broker'):
        cell2('4.8', f'剔除五粮液：RateSrc（{FE_ROW[fe]}）', 'excl_000858', Y_MAIN, fe, 'RT_NSRC')
    for fe in ('firm', 'firm+broker'):
        cell2('4.8', f'剔除五粮液、与 AllSrc 成对放入：RateSrc（{FE_ROW[fe]}）', 'excl_000858', Y_MAIN, fe, 'RT_NSRC+WG_NSRC', 'RT_NSRC')
    for fe, lab in (('firmxyear', '公司 × 年度'), ('firmxyear+broker', '公司 × 年度 + 券商')):
        for x in X6[:3]:
            cell2('4.9', f'{LABEL[x]}（{lab}）', 'firm_x_year', Y_MAIN, fe, x)
        add('4.9', f'（{lab}）RT 各列的观测数', G.cell('report', 'firm_x_year', Y_MAIN, fe, 'RT_NSRC').n, 0,
            f'reg_all report / firm_x_year / {Y_MAIN} / {fe} / RT_NSRC 的 n')
    return pd.DataFrame(rows)
