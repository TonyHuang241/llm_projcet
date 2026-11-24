"""DAG 结构指标回归的论文式表格片段：全部数字由 reg_results / reg_coefs_full / 描述统计 / 秩相关 / 三等分 / 逐一剔除的结果写出，
不手填。每个片段一个 markdown 文件（只有表格和表下注释，没有标题行），写到 <输出目录>/tables/。由 src/dag_outcomes.py 调用。

格式：系数 3 位有效数字、不用科学计数法；t 值两位小数；星号按该表主标准误的 p 值（* 10%、** 5%、*** 1%）；
变量写成“中文名（代码名）”。BHAR 在截面里和 RET 完全等价，表格都不列。
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.dag_outcomes_reg import stars

NAME = {
    'RATIO': '源汇比', 'LNPATH': '路径数', 'PATHLEN': '平均路径长度',
    'SYNCH': '股价同步性', 'SYNCH_SW': '股价同步性·申万行业', 'SYNCH_MKT': '股价同步性·仅市场', 'NCSKEW': '负收益偏态系数',
    'DUVOL': '收益上下波动比', 'IVOL': '特质波动率', 'SIGMA': '总波动率', 'RET': '年度回报', 'BHAR': '超额买入持有回报',
    'CAR_AR24_1': '2024 年报公告 CAR', 'CAR_AR24_5': '2024 年报公告 CAR', 'ABSCAR_AR24': '2024 年报公告 CAR 绝对值',
    'CAR_AR25_1': '2025 年报公告 CAR', 'CAR_AR25_5': '2025 年报公告 CAR', 'ABSCAR_AR25': '2025 年报公告 CAR 绝对值',
    'TURN': '换手率', 'ILLIQ': '非流动性',
    'ROA': '总资产收益率', 'ROE': '净资产收益率', 'NPM': '销售净利率', 'GM': '毛利率', 'ATO': '总资产周转率',
    'DROA': 'ROA 变化', 'REVG': '营业收入增速', 'NPG': '归母净利润增速', 'OPG': '营业利润增速', 'ASSETG': '总资产增速',
    'CFO': '经营现金流', 'ACC': '应计利润', 'EARNVOL': '盈利波动性', 'DLEV': '资产负债率变化', 'RD': '研发强度',
    'SIZE': '公司规模', 'LEV': '资产负债率', 'BM': '账面市值比', 'ROA24': '上年 ROA', 'TURN24': '上年换手率', 'RET24': '上年回报',
    'SIGMA24': '上年总波动率', 'LNCOV': '分析师关注度', 'LNNODES': '概念数', 'LNCHARS': '研报字数', 'LNNDAG': 'DAG 研报篇数',
    'LNCHARS_R': '研报字数·单篇', 'LNNODES_R': '概念数·单篇',
    'ROA_Q': '单季 ROA', 'REVG_Q': '单季营收同比', 'NPG_Q': '单季归母净利润同比', 'GM_Q': '单季毛利率', 'CFO_Q': '单季经营现金流',
    'SUE_Q': '未预期盈余',
}
CODE = {'CAR_AR24_1': 'CAR_AR24[-1,1]', 'CAR_AR24_5': 'CAR_AR24[-1,5]', 'CAR_AR25_1': 'CAR_AR25[-1,1]', 'CAR_AR25_5': 'CAR_AR25[-1,5]'}
FE_NAME = {'csrc': '证监会门类', 'sw': '申万一级'}
STAR_NOTE = '* p<0.1，** p<0.05，*** p<0.01'


def lab(v):
    """变量的“中文名（代码名）”；上年值（以 24 结尾、不在 NAME 里）写成“上年·中文名（代码名）”。"""
    if v in NAME:
        return f'{NAME[v]}（{CODE.get(v, v)}）'
    if v.endswith('24') and v[:-2] in NAME:
        return f'上年·{NAME[v[:-2]]}（{v}）'
    return v


def sig3(v):
    """3 位有效数字、不用科学计数法。"""
    if v is None or not np.isfinite(v):
        return ''
    v = float(f'{v:.2e}')  # 先舍入到 3 位有效数字，再按数量级定小数位（保留末尾的 0）
    return f'{v:.{max(0, 2 - int(np.floor(np.log10(abs(v)))))}f}' if v else '0.00'


def f2(v):
    return '' if v is None or not np.isfinite(v) else f'{v:.2f}'


def f3(v):
    return '' if v is None or not np.isfinite(v) else f'{v:.3f}'


def md(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join('' if c is None else str(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def write(path, *blocks):
    Path(path).write_text('\n\n'.join(blocks) + '\n', encoding='utf-8')


class Res:
    """reg_results 的按键查询：(model, spec, Y, reg, X) -> 行（dict）。"""

    def __init__(self, res):
        self.d = res.set_index(['model', 'spec', 'Y', 'reg', 'X']).to_dict('index')

    def get(self, model, spec, y, x, reg='单独'):
        return self.d.get((model, spec, y, reg, x))


def cell(r, t2=None, hc=False, perm=False):
    """系数星号 (主 t) [t2 或 HC1 t] {置换 p}；r 为空时返回空串。"""
    if r is None:
        return ''
    s = f'{sig3(r["coef"])}{stars(r["p"])} ({f2(r["t"])})'
    if t2 is not None:
        s += f' [{f2(t2)}]'
    elif hc:
        s += f' [{f2(r["t_hc1"])}]'
    if perm:
        s += f' {{{f3(r["p_perm"])}}}'
    return s


def same_or_list(vals, fmt=str):
    vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if not vals:
        return ''
    return fmt(vals[0]) if len(set(vals)) == 1 else ' / '.join(fmt(v) for v in vals)


# ---------------------------------------------------------------- 各张表
def t_desc(D, desc, fw, rw, qw, with_y24=True):
    """描述统计；with_y24=False 时去掉“上年同一指标”一段（t01_desc_short）。"""
    dd = {(p, v): r for p, v, r in zip(desc.panel, desc['var'], desc.to_dict('records'))}
    head = ['变量', 'N', '均值', '标准差', '最小值', 'P25', '中位数', 'P75', '最大值']
    y24 = [v for v in dict.fromkeys(D.FIRM_Y.values()) if v and v not in D.CONTROLS]
    rows = []
    sections = [('公司层面：DAG 结构指标', '公司', D.DAG_X), ('公司层面：股价指标（2025）', '公司', D.PRICE_Y),
                ('公司层面：财务指标（2025 财年）', '公司', D.FIN_Y), ('公司层面：控制变量（2024）', '公司', D.CONTROLS),
                ('公司层面：上年同一指标（2024，作 Y24）', '公司', y24),
                ('研报层面：DAG 结构指标和研报级控制', '研报', D.DAG_X + ['LNCHARS_R', 'LNNODES_R']),
                ('季度面板：单季财务指标（2025Q1–Q4）', '季度', D.QUARTER_Y)]
    if not with_y24:
        sections = [s for s in sections if s[2] is not y24]
    for title, panel, vs in sections:
        rows.append([f'**{title}**'] + [''] * 8)
        for v in vs:
            r = dd[(panel, v)]
            rows.append([lab(v), int(r['n'])] + [sig3(r[k]) for k in ('mean', 'sd', 'min', 'p25', 'p50', 'p75', 'max')])
    note = (f'注：连续变量已按 1%、99% 分位缩尾。公司层面 {len(fw)} 家公司（2024 年沪深300 成分股中至少 1 篇研报抽出 DAG 的）；'
            f'研报层面 {len(rw)} 篇有进图边的研报，公司变量取所属公司的值；季度面板为公司 × 2025Q1–Q4（{len(qw)} 行）。'
            '源汇比 = 源点数 / 汇点数；路径数 = ln(1 + 源到汇的唯一有向路径条数)；平均路径长度 = 各条路径边数的均值；公司层面为各篇均值。'
            'GM、RD、ACC 只对非金融公司计算。')
    return md(head, rows) + '\n\n' + note


def t_corr(fw):
    vs = ['RATIO', 'LNPATH', 'PATHLEN', 'LNNODES', 'LNCHARS', 'LNNDAG', 'SIZE', 'LNCOV']
    d = fw[vs].astype(float)
    pe, sp = d.corr(), d.rank().corr()
    rows = []
    for i, a in enumerate(vs):
        rows.append([f'({i + 1}) {lab(a)}'] + ['1' if i == j else f2(pe.loc[a, b] if j < i else sp.loc[a, b]) for j, b in enumerate(vs)])
    note = (f'注：公司层面（N = {len(d)}，缩尾后），下三角为 Pearson 相关系数，上三角为 Spearman 秩相关系数。')
    return md(['变量'] + [f'({i + 1})' for i in range(len(vs))], rows) + '\n\n' + note


def ctrl_note(D, ys, ctrl):
    """控制变量和上年同一指标的说明。"""
    no = [y for y in ys if not D.FIRM_Y.get(y)]
    dup = [y for y in no if y + '24' in ctrl]
    car = [y for y in no if y not in dup]
    s = '控制变量：' + '、'.join(lab(c) for c in ctrl) + '；以及上年同一指标（Y24，被解释变量自己的 2024 年值）。'
    if no:
        s += '不含上年同一指标的 Y：'
        parts = []
        if dup:
            parts.append('、'.join(CODE.get(y, y) for y in dup) + '（上年值已在控制变量里）')
        if car:
            parts.append('、'.join(CODE.get(y, y) for y in car) + '（不加）')
        s += '；'.join(parts) + '。'
    return s


def t_m1(D, R_, groups, spec):
    fe = 'sw' if spec == 'sw' else 'csrc'
    head = ['被解释变量', 'Y 标准差'] + [lab(x) for x in D.DAG_X] + ['N', 'R²', '调整 R²', '聚类数']
    rows, ys = [], []
    for g, vs in groups:
        rows.append([f'**{g}**'] + [''] * (len(head) - 1))
        for y in vs:
            ys.append(y)
            rs = [R_.get('M1', spec, y, x) for x in D.DAG_X]
            ok = [r for r in rs if r]
            rows.append([lab(y), sig3(ok[0]['y_sd']) if ok else ''] + [cell(r, hc=True, perm=True) for r in rs] +
                        [same_or_list([r['n'] for r in ok]), ' / '.join(f3(r['r2']) for r in ok),
                         ' / '.join(f3(r['adj_r2']) for r in ok), same_or_list([r['n_clusters'] for r in ok])])
    note = ('注：M1 公司层面截面回归。每格为该 X 单独进入回归时的系数（X 在回归样本内标准化，表示 X 高 1 个标准差时 Y 的变化）、星号、'
            f'(按{FE_NAME[fe]}聚类的 t)、[HC1 t]、{{{FE_NAME[fe]}内 Freedman–Lane 置换检验的双侧 p 值，B = 2000}}。'
            f'星号按聚类标准误（CR1）的 p 值，p 值用 t(G−1) 分布：{STAR_NOTE}。'
            + ctrl_note(D, ys, D.CONTROLS) +
            f'固定效应：{FE_NAME[fe]}；标准误按{FE_NAME[fe]}聚类。R²、调整 R² 依次对应 RATIO、LNPATH、PATHLEN 三个单独回归；'
            'Y 标准差为回归样本内的标准差。')
    return md(head, rows) + '\n\n' + note


def t_steps(D, R_, groups, model, specs, footer, n_label='N'):
    """逐步加控制：specs 是 4 个口径名（对应 (1)–(4)）；footer：[(行名, [是/否 ×4])]。"""
    head = ['被解释变量', 'X', '(1)', '(2)', '(3)', '(4)'] + [f'{n_label} ({i})' for i in range(1, 5)]
    rows = []
    for g, vs in groups:
        rows.append([f'**{g}**'] + [''] * (len(head) - 1))
        for y in vs:
            for x in D.DAG_X:
                rs = [R_.get(model, s, y, x) for s in specs]
                rows.append([lab(y), lab(x)] + [cell(r) for r in rs] + [r['n'] if r else '' for r in rs])
    for name, flags in footer:
        rows.append([name, ''] + flags + [''] * 4)
    return md(head, rows)


def t_full(D, full, groups, model='M1', spec='main', fe_name='证监会门类', per=8):
    j = full[(full.model == model) & (full.spec == spec) & (full.reg == '联合')]
    idx = {(y, t): r for y, t, r in zip(j.Y, j.term, j.to_dict('records'))}
    ys = [y for _, vs in groups for y in vs]
    n_tab = -(-len(ys) // per)
    size = -(-len(ys) // n_tab)
    blocks = []
    for b in range(n_tab):
        cols = ys[b * size:(b + 1) * size]
        head = ['变量'] + [f'({b * size + i + 1}) {lab(y)}' for i, y in enumerate(cols)]
        rows = []
        for t in D.DAG_X + D.CONTROLS + ['Y24']:
            row = [lab(t) if t != 'Y24' else '上年同一指标']
            for y in cols:
                key = D.FIRM_Y.get(y) if t == 'Y24' else t
                r = idx.get((y, key)) if key else None
                row.append(f'{sig3(r["coef"])}{stars(r["p"])} ({f2(r["t"])})' if r else '')
            rows.append(row)
        first = lambda y: idx.get((y, D.DAG_X[0]))  # noqa: E731
        rows.append(['行业固定效应'] + ['是' for _ in cols])
        rows.append(['N'] + [first(y)['n'] if first(y) else '' for y in cols])
        rows.append(['R²'] + [f3(first(y)['r2']) if first(y) else '' for y in cols])
        rows.append(['调整 R²'] + [f3(first(y)['adj_r2']) if first(y) else '' for y in cols])
        rows.append(['聚类数'] + [first(y)['n_clusters'] if first(y) else '' for y in cols])
        blocks.append(md(head, rows))
    note = (f'注：M1 联合回归（三个 DAG 指标同时进入），每格为系数星号 (按{fe_name}聚类的 t)；三个 DAG 指标在回归样本内标准化，'
            f'控制变量为原始单位。“上年同一指标”是各列被解释变量自己的 2024 年值，空白表示不加（上年值已是控制变量，或为 CAR 类）。'
            f'固定效应：{fe_name}；星号按聚类标准误 p 值（t(G−1)）：{STAR_NOTE}。')
    return '\n\n'.join(blocks) + '\n\n' + note


def t_m2(D, R_, groups, info):
    head = ['被解释变量', 'Y 标准差'] + [lab(x) for x in D.DAG_X] + ['N（研报）', '公司数', 'R²', '调整 R²']
    rows, ys = [], []
    for g, vs in groups:
        rows.append([f'**{g}**'] + [''] * (len(head) - 1))
        for y in vs:
            ys.append(y)
            rs = [R_.get('M2', 'main', y, x) for x in D.DAG_X]
            tw = [R_.get('M2', 'twoway', y, x) for x in D.DAG_X]
            ok = [r for r in rs if r]
            rows.append([lab(y), sig3(ok[0]['y_sd']) if ok else ''] +
                        [cell(r, t2=w['t'] if w else np.nan) if r else '' for r, w in zip(rs, tw)] +
                        [same_or_list([r['n'] for r in ok]), same_or_list([r['n_clusters'] for r in ok]),
                         ' / '.join(f3(r['r2']) for r in ok), ' / '.join(f3(r['adj_r2']) for r in ok)])
    note = ('注：M2 研报层面回归，每篇研报一个观测，X 是这一篇的指标（在回归样本内标准化），Y 是所属公司的 2025 年指标。'
            '每格为 X 单独进入回归时的系数星号 (按公司聚类的 t) [按公司和券商双向聚类的 t]；双向聚类按 Cameron-Gelbach-Miller：'
            f'V = V_公司 + V_券商 − V_公司×券商，券商用原始券商名（{info["n_brokers"]} 家），p 值自由度取较小聚类数 − 1。'
            f'星号按公司聚类标准误的 p 值（t(G−1)）：{STAR_NOTE}。' + ctrl_note(D, ys, D.REPORT_CONTROLS) +
            f'固定效应：证监会门类 + 券商（样本里不足 {D.MIN_BROKER} 篇的并成“其他”，共 {info["n_broker_groups"]} 组）+ 发布月份。'
            'R²、调整 R² 依次对应三个单独回归。')
    return md(head, rows) + '\n\n' + note


def t_m3(D, R_):
    head = ['被解释变量', 'Y 标准差'] + [f'单独：{lab(x)}' for x in D.DAG_X] + [f'联合：{lab(x)}' for x in D.DAG_X] + ['N', '公司数', 'R²（单独）']
    rows = []
    for y in D.QUARTER_Y:
        rs = [R_.get('M3', 'main', y, x) for x in D.DAG_X]
        js = [R_.get('M3', 'main', y, x, '联合') for x in D.DAG_X]
        ok = [r for r in rs if r]
        rows.append([lab(y), sig3(ok[0]['y_sd']) if ok else ''] + [cell(r, hc=True) for r in rs] + [cell(r) for r in js] +
                    [same_or_list([r['n'] for r in ok]), same_or_list([r['n_clusters'] for r in ok]), ' / '.join(f3(r['r2']) for r in ok)])
    a = md(head, rows)
    note_a = ('注：M3 季度财务面板（公司 × 2025Q1–Q4），X 是公司层面指标（在回归样本内标准化）。单独回归的格子为系数星号 (按公司聚类的 t) [HC1 t]，'
              '联合回归（三个 X 一起进）的格子为系数星号 (按公司聚类的 t)。控制变量同 M1：' + '、'.join(lab(c) for c in D.CONTROLS) +
              '，以及上年同季度的同一指标；固定效应：证监会门类 × 季度；标准误按公司聚类，星号按聚类 p 值（t(G−1)）：' + STAR_NOTE + '。')
    b = t_steps(D, R_, [('季度财务指标', D.QUARTER_Y)], 'M3', ['c1', 'c2', 'c3', 'main'], [
        ('公司控制变量', ['否', '是', '是', '是']), ('DAG/文本控制', ['否', '否', '是', '是']),
        ('上年同季同一指标', ['否', '否', '否', '是']), ('行业 × 季度固定效应', ['是', '是', '是', '是'])])
    note_b = '注：M3 逐步加控制，每格为 X 单独进入回归时的系数星号 (按公司聚类的 t)；N 为公司-季度观测数。' + STAR_NOTE + '。'
    return a + '\n\n' + note_a + '\n\n' + b + '\n\n' + note_b


def t_robust(D, R_, groups, model, specs, labels, t_note=None):
    """稳健性表；t_note 为 None 时不写表下注释（同一文件里多张表时只在最后一张写）。"""
    head = ['被解释变量', 'X'] + [f'{s}' for s in specs]
    rows = []
    for g, vs in groups:
        rows.append([f'**{g}**'] + [''] * (len(head) - 1))
        for y in vs:
            for x in D.DAG_X:
                rows.append([lab(y), lab(x)] + [cell(R_.get(model, s, y, x)) for s in specs])
    ns = []
    for s in specs:
        v = [r['n'] for y in [y for _, vs in groups for y in vs] for r in [R_.get(model, s, y, D.DAG_X[0])] if r]
        ns.append(f'{min(v)}–{max(v)}' if v else '')
    rows.append(['N 范围', ''] + ns)
    if t_note is None:
        return md(head, rows)
    note = (f'注：{t_note}每格为 X 单独进入回归时的系数星号 (t)，t 和星号按该口径自己的主标准误（{STAR_NOTE}）。口径：'
            + '；'.join(f'{s} = {labels[s]}' for s in specs) + '。')
    return md(head, rows) + '\n\n' + note


def t_rank(D, ric, grp, groups):
    rk = ric.set_index(['Y', 'X']).to_dict('index')
    gp = grp.set_index(['Y', 'X']).to_dict('index')
    head = ['被解释变量', 'X', 'N', 'Spearman', '偏 Spearman', '偏 Spearman p', '低组', '中组', '高组', '高 − 低 (t)', '调整后 高 − 低 (t)']
    rows = []
    for g, vs in groups:
        rows.append([f'**{g}**'] + [''] * (len(head) - 1))
        for y in vs:
            for x in D.DAG_X:
                a, b = rk.get((y, x)), gp.get((y, x))
                if not a:
                    continue
                rows.append([lab(y), lab(x), a['n'], f3(a['spearman']) + stars(a['p_spearman']),
                             f3(a['partial_spearman']) + stars(a['p_partial']), f3(a['p_partial']),
                             sig3(b['low']), sig3(b['mid']), sig3(b['high']),
                             f'{sig3(b["hml"])}{stars(b["p"])} ({f2(b["t"])})', f'{sig3(b["adj_hml"])}{stars(b["adj_p"])} ({f2(b["adj_t"])})'])
    return md(head, rows)


def t_influence(D, R_, infl, ys):
    head = ['被解释变量', 'X', '全样本：系数 (聚类t) [HC1 t] {置换p}', '剔除单家公司：系数范围', '同号占比', '影响最大的公司：系数 (t)',
            '剔除单个门类：系数范围', 't 范围', '影响最大的门类：系数 (t)', '剔除制造业：系数 (t)', '剔除制造业 N']
    rows = []
    infl = infl[infl.Y.isin(ys)].assign(o=lambda x: x.Y.map({y: i for i, y in enumerate(ys)})).sort_values('o', kind='stable')
    for r in infl.to_dict('records'):
        m = R_.get('M1', 'main', r['Y'], r['X'])
        pp = m['p_perm'] if np.isfinite(m['p_perm']) else 1
        if not (m['p'] < 0.05 or pp < 0.10 or m['p_hc1'] < 0.05):
            continue
        rows.append([lab(r['Y']), lab(r['X']), cell(m, hc=True, perm=True), f'[{sig3(r["firm_coef_min"])}, {sig3(r["firm_coef_max"])}]',
                     f'{r["firm_same_sign"]:.1%}', f'{r["firm_top_name"]}（{r["firm_top_code"]}）：{sig3(r["firm_top_coef"])} ({f2(r["firm_top_t"])})',
                     f'[{sig3(r["ind_coef_min"])}, {sig3(r["ind_coef_max"])}]', f'[{f2(r["ind_t_min"])}, {f2(r["ind_t_max"])}]',
                     f'{r["ind_top"]}：{sig3(r["ind_top_coef"])} ({f2(r["ind_top_t"])})', f'{sig3(r["nomfg_coef"])} ({f2(r["nomfg_t"])})',
                     '' if pd.isna(r['nomfg_n']) else int(r['nomfg_n'])])
    key = md(head, rows) if rows else '（没有满足条件的组合）'
    note = ('注：只列 M1 主回归（单独回归）中按证监会门类聚类 p < 0.05、或置换检验 p < 0.10、或 HC1 p < 0.05 的组合。'
            '剔除单家公司：每次去掉一家公司重估（X 在剩下的样本里重新标准化；去掉后出现只有一个观测的行业一并剔除），'
            '“同号占比”是重估系数与全样本系数同号的比例，“影响最大”是系数变化绝对值最大的那家；剔除单个门类：每次去掉一个证监会门类重估，'
            't 为按证监会门类聚类的 t。剔除制造业即去掉证监会门类“制造业”的全部公司。')
    return key + '\n\n' + note


def t_sig(D, res):
    pr, fi = set(D.PRICE_Y), set(D.FIN_Y) | set(D.QUARTER_Y)
    s = res[res.reg == '单独']
    order = {'M1': ['main', 'sw', 'nonfin', 'unified', 'median', 'ndag2', 'ndag3', 'noNODES', 'c1', 'c2', 'noY24'],
             'M2': ['main', 'sw', 'nonfin', 'twoway', 'c1', 'c2', 'c3'], 'M3': ['main', 'sw', 'nonfin', 'c1', 'c2', 'c3']}
    head = ['模型', '口径', '股价：系数数', '10%', '5%', '1%', 'HC1 5%', '置换 5%', '期望 5%',
            '财务：系数数', '10%', '5%', '1%', 'HC1 5%', '置换 5%', '期望 5%']
    rows = []
    for m, specs in order.items():
        for sp in specs:
            g = s[(s.model == m) & (s.spec == sp)]
            label = g.spec_label.iloc[0] if len(g) else ''
            row = [m, f'{sp}：{label}']
            for ys in (pr, fi):
                h = g[g.Y.isin(ys) & g.p.notna()]
                if not len(h):
                    row += [''] * 7
                    continue
                perm = h.p_perm.notna().any()
                row += [len(h), int((h.p < .1).sum()), int((h.p < .05).sum()), int((h.p < .01).sum()), int((h.p_hc1 < .05).sum()),
                        int((h.p_perm < .05).sum()) if perm else '', f'{0.05 * len(h):.1f}']
            rows.append(row)
    note = ('注：只数单独回归（每个 Y 的三个 X 各一次）的 X 系数，不含 BHAR。10%、5%、1% 按该口径主标准误的 p 值（M1 按行业聚类，'
            'M2 按公司聚类、twoway 口径按公司和券商双向聚类，M3 按公司聚类）；“HC1 5%”按异方差稳健 p 值；“置换 5%”按行业内 Freedman–Lane '
            '置换检验 p 值（只有 M1 main、sw 做了）。“期望 5%” = 0.05 × 系数数，是所有真实系数为 0 且相互独立时 5% 水平下碰巧显著的期望个数；'
            '同一 Y 的三个 X 高度相关、各 Y 之间也相关，系数并不独立，期望数只作参照。M3 的被解释变量是单季财务指标，记在“财务”栏。')
    return md(head, rows) + '\n\n' + note


# ---------------------------------------------------------------- 每个 Y 一张表（伊志宏等 2019 表 2 的版式）
def _two_rows(label, vals):
    """vals：每列 (coef, t, p) 或 None -> 系数星号一行、(t) 一行。"""
    return [[label] + [f'{sig3(v[0])}{stars(v[2])}' if v else '' for v in vals],
            [''] + [f'({f2(v[1])})' if v else '' for v in vals]]


def by_y_table(D, R_, fidx, y, cols, quarter=False):
    """一个 Y 的表。cols：[(model, x)]，x 为 None 表示三个 X 一起进入。fidx：reg_coefs_full main 口径的 (model, Y, X_set, term) -> 行。"""
    ncol = len(cols)
    blank = lambda title: [f'**{title}**'] + [''] * ncol  # noqa: E731
    reg_row = lambda m, sp, x: R_.get(m, sp, y, x or D.DAG_X[0], '单独' if x else '联合')  # noqa: E731
    head = ['变量'] + [f'({i + 1})' for i in range(ncol)]
    level = {'M1': '公司', 'M2': '研报', 'M3': '公司×季度'}
    rows = [['回归层面'] + [level[m] for m, _ in cols]]

    rows.append(blank('Panel A：证监会门类' + ('×季度' if quarter else '') + '固定效应'))
    a = lambda m, x, term: (lambda r: (r['coef'], r['t'], r['p']) if r else None)(fidx.get((m, y, x or '联合', term)))  # noqa: E731
    for x in D.DAG_X:
        rows += _two_rows(lab(x), [a(m, cx, x) for m, cx in cols])
    y24 = y + '24' if quarter else D.FIRM_Y.get(y)
    rows.append(['控制变量'] + ['Y'] * ncol)  # 控制变量的系数不列，全在 reg_coefs_full.csv
    rows.append(['上年同季同一指标' if quarter else '上年同一指标'] + ['Y' if y24 else 'N'] * ncol)
    main = [reg_row(m, 'main', cx) for m, cx in cols]
    if quarter:
        rows.append(['行业×季度固定效应'] + ['Y'] * ncol)
        rows.append(['Cluster'] + ['公司'] * ncol)
    else:
        rows.append(['行业固定效应'] + ['Y'] * ncol)
        rows.append(['券商、月份固定效应'] + ['Y' if m == 'M2' else 'N' for m, _ in cols])
        rows.append(['Cluster'] + ['行业' if m == 'M1' else '公司' for m, _ in cols])
        rows.append(['置换检验 p 值'] + [f3(r['p_perm']) if r and m == 'M1' and cx else '' for r, (m, cx) in zip(main, cols)])
    rows.append(['样本量'] + [r['n'] if r else '' for r in main])
    rows.append(['调整后 R²'] + [f3(r['adj_r2']) if r else '' for r in main])

    rows.append(blank('Panel B：申万一级' + ('×季度' if quarter else '') + '固定效应（控制变量同 Panel A）'))
    for x in D.DAG_X:
        vals = []
        for m, cx in cols:
            r = R_.get(m, 'sw', y, x, '单独' if cx else '联合') if cx in (x, None) else None
            vals.append((r['coef'], r['t'], r['p']) if r else None)
        rows += _two_rows(lab(x), vals)
    sw = [reg_row(m, 'sw', cx) for m, cx in cols]
    if not quarter:
        rows.append(['置换检验 p 值'] + [f3(r['p_perm']) if r and m == 'M1' and cx else '' for r, (m, cx) in zip(sw, cols)])
    rows.append(['样本量'] + [r['n'] if r else '' for r in sw])
    rows.append(['调整后 R²'] + [f3(r['adj_r2']) if r else '' for r in sw])
    return md(head, rows)


def write_by_y(D, out, res, full):
    """by_y_price.md、by_y_fin.md、by_y_quarter.md：每个被解释变量一张表，表号从 3 开始连续编。"""
    R_ = Res(res)
    f = full[full.spec == 'main']
    fidx = {(m, y, xs, t): r for m, y, xs, t, r in zip(f.model, f.Y, f.X_set, f.term, f.to_dict('records'))}
    annual = [(m, x) for m in ('M1', 'M2') for x in D.DAG_X + [None]]
    quarter = [('M3', x) for x in D.DAG_X + [None]]
    no24 = lambda ys: ctrl_note(D, ys, D.CONTROLS).split('。', 1)[1].rstrip('。')  # noqa: E731  “不含上年同一指标的 Y：…”
    ctrl_names = '、'.join([lab(c) for c in D.FIRM_CTRL] + ['概念数', '研报字数', 'DAG 研报篇数'])
    note_annual = (
        '注：每张表一个被解释变量（2025 年值）。(1)–(3) 为公司层面截面回归（M1），源汇比、路径数、平均路径长度各自单独进入；(4) 为 M1 三个指标同时进入；'
        '(5)–(7) 为研报层面回归（M2，每篇研报一个观测，DAG 指标是该篇研报的值，被解释变量和公司控制变量取所属公司的值），三个指标各自单独进入；'
        '(8) 为 M2 三个指标同时进入。DAG 指标在各自回归样本内标准化，系数表示指标高 1 个标准差时被解释变量的变化。'
        f'控制变量（2024 年值，系数不列，见 reg_coefs_full.csv）：{ctrl_names}。'
        '括号内为聚类稳健（CR1）t 值：M1 按行业聚类（Panel A 证监会门类，Panel B 申万一级），M2 按公司聚类；'
        f'星号按该列聚类标准误的 p 值（t(G−1) 分布）：{STAR_NOTE}。'
        'Panel A 为证监会门类固定效应（主回归）；Panel B 把行业固定效应换成申万一级（M1 同时改按申万一级聚类，M2 仍按公司聚类），控制变量同 Panel A。'
        '概念数、研报字数在 M1 中是公司各篇研报均值的对数（LNNODES、LNCHARS），在 M2 中是该篇研报的对数（LNNODES_R、LNCHARS_R）；'
        'DAG 研报篇数为 ln(1 + 公司抽出 DAG 的研报篇数)。上年同一指标为被解释变量自己的 2024 年值（该行为 N 的表没有加）；{no24}。'
        '置换检验 p 值：在行业内（Panel A 证监会门类、Panel B 申万一级）置换约简模型（不含 DAG 指标）的残差 2000 次（Freedman–Lane），'
        '按聚类 t 的绝对值算双侧 p 值，只对 M1 单独回归做。Y 表示包含该固定效应，N 表示不包含；M2 的券商固定效应里样本不足 '
        f'{D.MIN_BROKER} 篇的券商并成“其他”，月份为研报发布月份。')
    n = 3
    for fname, groups, sec in (('by_y_price.md', D.PRICE_GROUPS, 4), ('by_y_fin.md', D.FIN_GROUPS, 5)):
        parts = []
        for i, (g, ys) in enumerate(groups, start=1):
            parts.append(f'### {sec}.{i} {g}')
            for y in ys:
                parts += [f'**表 {n}　{lab(y)}**', by_y_table(D, R_, fidx, y, annual)]
                n += 1
        parts.append(note_annual.format(no24=no24([y for _, ys in groups for y in ys])))
        write(out / fname, *parts)
    parts = ['### 5.5 季度财务指标']
    for y in D.QUARTER_Y:
        parts += [f'**表 {n}　{lab(y)}**', by_y_table(D, R_, fidx, y, quarter, quarter=True)]
        n += 1
    parts.append(
        '注：每张表一个被解释变量，季度财务面板（公司 × 2025Q1–Q4，每个公司-季度一个观测，M3）。(1)–(3) 为源汇比、路径数、平均路径长度各自单独进入，'
        '(4) 为三个指标同时进入；DAG 指标是公司层面的值（不随季度变化），在回归样本内标准化，控制变量同公司层面回归（2024 年值，系数不列），'
        '上年同季同一指标为被解释变量在 2024 年同一季度的值。括号内为按公司聚类的稳健（CR1）t 值，'
        f'星号按该列聚类标准误的 p 值（t(G−1) 分布）：{STAR_NOTE}。Panel A 为证监会门类 × 季度固定效应（主回归）；Panel B 换成申万一级 × 季度固定效应，'
        '仍按公司聚类，控制变量同 Panel A。季度面板不做置换检验。Y 表示包含该固定效应。')
    write(out / 'by_y_quarter.md', *parts)


def write_fe_r2(path, dec, fes, firm_note=''):
    """s_fe_r2.md / s_fe_r2_firm.md：研报层面三个 DAG 指标的调整后 R² 的 Shapley 分解（dec 是 RB.fe_r2_decomposition 的长表，
    取 target = adj_r2 的贡献行）。fes 含 'code'（公司）时表注里加公司固定效应的定义；firm_note 追加在表注末尾。"""
    from src.dag_outcomes import DAG_X, MIN_BROKER
    sh, sub = dec[(dec.kind == 'shapley') & (dec.target == 'adj_r2')], dec[dec.kind == 'subset']
    names = {'month': '发布月份', 'sw': '申万一级行业', 'broker_g': '券商', 'code': '公司'}
    m = len(fes)
    cn = {3: '三', 4: '四'}[m]
    head = ['固定效应', '组数'] + [h for x in DAG_X for h in (f'{NAME[x]}：调整后 R² 贡献', '占比')]
    rows = []
    for f in fes:
        g = sh[sh.fe == f].set_index('Y')
        rows.append([names[f], int(g.n_groups.iloc[0])] + [v for x in DAG_X for v in (f3(g.loc[x, 'contribution']), f'{g.loc[x, "share"]:.1%}')])
    full = sub[sub[list(fes)].astype(bool).all(axis=1)].set_index('Y')
    rows.append(['合计（调整后 R²）', ''] + [v for x in DAG_X for v in (f3(full.loc[x, 'adj_r2']), '100.0%')])
    rows.append(['样本量', ''] + [v for x in DAG_X for v in (int(full.loc[x, 'n']), '')])
    note = (f'注：样本为研报层面面板（{int(full.n.iloc[0])} 篇抽出 DAG 的研报，同 M2），被解释变量是该篇研报的源汇比（RATIO）、路径数（LNPATH）、'
            f'平均路径长度（PATHLEN），已在研报层面按 1%、99% 分位缩尾，三个指标用同一样本。{cn}组固定效应都用虚拟变量：发布月份为研报发布的年月；'
            f'申万一级行业为研报所属公司的申万一级行业；券商为发布研报的券商，样本里不足 {MIN_BROKER} 篇的券商并成“其他”'
            + ('；公司为研报所属公司' if 'code' in fes else '') + '。不剔除只有一个观测的组，全部子模型用同一样本。'
            f'表中分解的是调整后 R²：对{cn}组固定效应的全部 {2 ** m} 个子集 S 各估一次 OLS，调整后 R²(S) = 1 − (1 − R²(S))·(n − 1)/(n − 秩(S))，'
            '秩为设计矩阵（常数项加各组虚拟变量）的实际秩，空集为 0。各组的贡献按 Shapley 分解：'
            f'某一组的贡献是 {math.factorial(m)} 种加入顺序下这一组加入时调整后 R² 增量的平均值，{cn}组贡献之和等于{cn}组都放时的调整后 R²（合计行）；'
            '占比 = 贡献 / 合计。贡献可以为负：表示这组固定效应增加的解释力抵不上它消耗的自由度。' + firm_note)
    write(path, md(head, rows), note)


# ---------------------------------------------------------------- 入口
def write_tables(out, res, full, desc, fw, ric, grp, infl, info):
    from src import dag_outcomes as D  # 延迟导入，避免循环引用
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rw = pd.read_parquet(out.parent / 'report_panel.parquet')
    qw = pd.read_parquet(out.parent / 'quarter_panel.parquet')
    R_ = Res(res)
    P, F = D.PRICE_GROUPS, D.FIN_GROUPS
    write(out / 't01_desc.md', t_desc(D, desc, fw, rw, qw))
    write(out / 't02_corr.md', t_corr(fw))
    write(out / 't03_m1_price.md', t_m1(D, R_, P, 'main'))
    write(out / 't04_m1_fin.md', t_m1(D, R_, F, 'main'))
    write(out / 't05_m1_sw_price.md', t_m1(D, R_, P, 'sw'))
    write(out / 't06_m1_sw_fin.md', t_m1(D, R_, F, 'sw'))
    m1_footer = [('公司控制变量', ['否', '是', '是', '是']), ('DAG/文本控制', ['否', '否', '是', '是']),
                 ('上年同一指标', ['否', '否', '否', '是']), ('行业固定效应', ['是', '是', '是', '是'])]
    steps_note = lambda m, fe, cl: (  # noqa: E731
        f'注：{m} 逐步加控制，每格为 X 单独进入回归时的系数星号 ({cl}聚类的 t)，X 在回归样本内标准化。(1) 只有{fe}；(2) 加公司控制变量 '
        + '、'.join(lab(c) for c in D.FIRM_CTRL) + '；(3) 再加 DAG/文本控制 ' +
        '、'.join(lab(c) for c in (D.REPORT_CONTROLS if m == 'M2' else D.CONTROLS)[len(D.FIRM_CTRL):]) +
        '；(4) 再加上年同一指标（主回归）。不含上年同一指标的 Y，(3)、(4) 相同。各列 N 因控制变量缺失不同可能不同。'
        f'星号按聚类标准误 p 值（t(G−1)）：{STAR_NOTE}。')
    for name, g in (('t07_m1_steps_price.md', P), ('t08_m1_steps_fin.md', F)):
        write(out / name, t_steps(D, R_, g, 'M1', ['c1', 'c2', 'noY24', 'main'], m1_footer), steps_note('M1', '证监会门类固定效应', '按证监会门类'))
    write(out / 't09_m1_full_price.md', t_full(D, full, P))
    write(out / 't10_m1_full_fin.md', t_full(D, full, F))
    write(out / 't11_m2_price.md', t_m2(D, R_, P, info))
    write(out / 't12_m2_fin.md', t_m2(D, R_, F, info))
    m2_footer = [('公司控制变量', ['否', '是', '是', '是']), ('DAG/文本控制（研报层面）', ['否', '否', '是', '是']),
                 ('上年同一指标', ['否', '否', '否', '是']), ('行业、券商、月份固定效应', ['是', '是', '是', '是'])]
    write(out / 't13_m2_steps.md', t_steps(D, R_, P, 'M2', ['c1', 'c2', 'c3', 'main'], m2_footer),
          t_steps(D, R_, F, 'M2', ['c1', 'c2', 'c3', 'main'], m2_footer),
          steps_note('M2', '证监会门类、券商、发布月份固定效应', '按公司') + 'N 为研报数。第一张表为股价指标，第二张为财务指标。')
    write(out / 't14_m3.md', t_m3(D, R_))
    m1_lab = dict(D.M1_SPECS)
    m1_specs = ['main', 'sw', 'nonfin', 'unified', 'median', 'ndag2', 'ndag3', 'noNODES']
    m1_note = 'M1 稳健性（证监会门类固定效应和聚类，sw 口径为申万一级）。'
    write(out / 't15_robust_m1.md', t_robust(D, R_, P, 'M1', m1_specs, m1_lab),
          t_robust(D, R_, F, 'M1', m1_specs, m1_lab, m1_note) + '第一张表为股价指标，第二张为财务指标。')
    m2_lab, m3_lab = dict(D.M2_SPECS), dict(D.M3_SPECS)
    m2_note = 'M2 研报层面稳健性（主口径按公司聚类；twoway 按公司和券商双向聚类）。'
    write(out / 't16_robust_m2m3.md', t_robust(D, R_, P, 'M2', ['main', 'sw', 'nonfin', 'twoway'], m2_lab),
          t_robust(D, R_, F, 'M2', ['main', 'sw', 'nonfin', 'twoway'], m2_lab, m2_note) + '第一张表为股价指标，第二张为财务指标。',
          t_robust(D, R_, [('季度财务指标', D.QUARTER_Y)], 'M3', ['main', 'sw', 'nonfin'], m3_lab,
                   'M3 季度财务面板稳健性（按公司聚类）。'))
    rank_note = ('注：M1 主回归样本（每个 Y × X 为对应单独回归的样本）。Spearman 为原始秩相关；偏 Spearman：Y、X 先转成秩，各自对主回归的控制变量'
                 '（含能加的上年同一指标）和证监会门类固定效应回归取残差，再算残差的 Pearson 相关，p 值用 t 近似（df = n − 控制项数 − 1）。'
                 '低、中、高组为按 X 三等分后 Y 的均值，高 − 低用 Welch t；“调整后”是 Y、X 都先对上述控制变量和固定效应取残差，按 X 残差三等分，'
                 f'Y 残差的高组 − 低组。星号：Spearman、偏 Spearman 按各自 p 值，高 − 低按 Welch p 值，{STAR_NOTE}。第一张表为股价指标，第二张为财务指标。')
    write(out / 't17_rank_group.md', t_rank(D, ric, grp, P), t_rank(D, ric, grp, F), rank_note)
    write(out / 't18_influence.md', t_influence(D, R_, infl, D.PRICE_Y + D.FIN_Y))
    write(out / 't19_sig_count.md', t_sig(D, res))
    write(out / 't01_desc_short.md', t_desc(D, desc, fw, rw, qw, with_y24=False))
    write_by_y(D, out, res, full)
    print(f'表格片段：{len(list(out.glob("*.md")))} 个 -> {out}', flush=True)
