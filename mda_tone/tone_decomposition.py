"""MD&A 语调分解：变量构造（第一步，不读取任何 car_*、beta 和事件日之后的行情）。

规格来源：mda_tone/output/event_study/tone_decomposition/design.md（下称设计文档）。
输入：tone_by_sentence.csv（流式，同一 (year, code) 的句子连续）、cs_sentiment.csv、event_panel.csv（只读 ALLOWED_PANEL_COLS）、
      hypotheses/variables.csv（只读 unchanged_rate）、financial_data、daily_stock_data。
输出（out_dir）：
  - variables.csv：code, year + 设计文档第 3 节所有变量的未缩尾值，与 event_panel 逐行对齐，缺失写空
  - variables_summary.csv：每个变量的非缺失数、均值、标准差、极值
  - p3_first_stage.csv：P3 逐年一阶段的样本量和 R2
  - cache/sentence_vars.pkl：逐句扫描结果的缓存（1.9GB 文件只扫一次）

用法：在项目根目录运行 python -m mda_tone.tone_decomposition
"""
import csv
import itertools
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import DATA_DIR, OUTPUT_DIR
from .construct_hypothesis_vars import ALLOWED_PANEL_COLS, load_panel, load_turnover, market_vars
from .event_study import MIN_EST_OBS, EST_START, EST_END, load_fundamentals, load_returns, market_model

csv.field_size_limit(10 ** 9)

OUT_DIR = OUTPUT_DIR / 'event_study' / 'tone_decomposition'
SENTENCE_PATH = OUTPUT_DIR / 'sentiment_analysis/neutral_compare_glm_all/tone_by_sentence.csv'
CS_PATH = OUTPUT_DIR / 'sentiment_data/cs_sentiment.csv'
PANEL_PATH = OUTPUT_DIR / 'event_study/event_panel.csv'
HYP_VARS_PATH = OUTPUT_DIR / 'event_study/hypotheses/variables.csv'
FIN_PATH = DATA_DIR / 'market_data/financial_data.parquet'
DAILY_PATH = DATA_DIR / 'market_data/daily_stock_data.parquet'
SENTENCE_COLS = ['year', 'code', 'id', 'label_orig', 'label_neutral', 'pos_orig', 'neg_orig', 'pos_neutral',
                 'neg_neutral', 'original', 'neutral']

MIN_SENT = 20           # 逐句分布型变量的最少句数
MIN_GROUP = 5           # 各类“至少 5 句”的门槛
MIN_STD = 0.1           # P4：标准差下限
MIN_CELL = 30           # P2：年度 × 组别的最少样本
MIN_PEERS_2D = 50       # P2-d：同年事件日更早的同行下限
MIN_FS_OBS = 200        # P3：每年一阶段最少观测
MIN_RET_DAYS = 120      # P3：ret_y、std_ret 的最少有效日

# ---------------------------------------------------------------- 前瞻句、风险句规则（设计文档“共用逐句变量”，逐字照抄）
EXC = re.compile(r'预计(?:可使用|使用寿命|净残值|负债|效益|投资|未来现金流|现金流)|《[^》]{0,40}展望[^》]{0,40}》'
                 r'|未来年度.{0,12}(?:减值|摊销|测试)|为.{0,12}(?:未来|今后).{0,12}(?:奠定|打下|基础)|竞争取|以来年|未来年度内')
RULE_A = re.compile(r'展望|未来|今后|明年|下一年|下年度|(?<![未以])来年|下一步')
RULE_FY = re.compile(r'(?<!\d)(20\d\d)年(?!\d{1,2}月)')
RULE_B = re.compile(r'将(?:继续|进一步|持续|不断|加大|加强|积极|努力|着力|重点|坚持|全面|大力|稳步|力争|争取|逐步'
                    r'|会|要|可能|有望|迎来|面临|成为|保持|实现|达到|带来|导致|产生|受到|得到|进入|呈现'
                    r'|提升|提高|推动|促进|加快|推进|通过|更加|千方百计|在|对|于20|于本|于明|于下)|即将|有望')
RULE_C = re.compile(r'力争|争取|预计|计划(?:于|在|将|实现|完成|投资|新增|开工|建设|推出|扩大)|预期(?:将|未来|到|明年|20)|可以预期')
RISK = re.compile(r'风险|如果|倘若|一旦|(?<![倘])若(?=[^，。；]{0,30}[，,])')
SELF_REF = re.compile(r'公司|本行|本集团|集团|我们|我行|管理层|本企业')  # “本公司”包含在“公司”内
DIGIT = re.compile(r'\d')

SENT_VARS = (
    ['skew_orig', 'neg_sandwich_exc', 'neg_sandwich_exc_norisk', 'neg_sandwich_exc_s10',
     'hard_neg_share', 'soft_neg_share', 'hard_neg_o', 'soft_neg_o', 'hard_neg_nr', 'soft_neg_nr',
     'boiler_share', 'neu_digit_share',
     'pos_neg_fact', 'pos_pos_all', 'neg_fact_share', 'pos_neg_all', 'pos_neg_fact_m3', 'pos_neg_fact_m10',
     'pos_change_mean', 'neg_change_mean', 'one_sided', 'two_sided', 'frac_neg_orig',
     'one_sided_rw', 'one_sided_m10', 'two_sided_m10']
    + [f'{v}{s}' for s in ('', '_a', '_b', '_c', '_d') for v in ('sc_fwd', 'sc_nonfwd', 'mn_fwd', 'share_fwd')])


def clean(v):
    return None if v is None or (isinstance(v, float) and math.isnan(v)) else v


def nan_if_none(v):
    return math.nan if v is None else float(v)


def masked_mean(a, mask, min_n):
    return float(a[mask].mean()) if mask.sum() >= min_n else None


# ---------------------------------------------------------------- 逐句扫描
def doc_features(rows, col, year):
    """一篇年报的全部逐句变量（n >= MIN_SENT 才计算）。"""
    rows = sorted(rows, key=lambda r: int(r[col['id']]))
    n = len(rows)
    if n < MIN_SENT:
        return None
    o = np.array([float(r[col['pos_orig']]) - float(r[col['neg_orig']]) for r in rows])
    x = np.array([float(r[col['pos_neutral']]) - float(r[col['neg_neutral']]) for r in rows])
    c = x - o
    lab_o = np.array([r[col['label_orig']] for r in rows])
    lab_n = np.array([r[col['label_neutral']] for r in rows])
    pos_o, neu_o, neg_o = lab_o == 'positive', lab_o == 'neutral', lab_o == 'negative'
    pos_n, neg_n = lab_n == 'positive', lab_n == 'negative'
    txt_o = [r[col['original']] for r in rows]
    txt_n = [r[col['neutral']] for r in rows]
    s_o = [''.join(t.split()) for t in txt_o]
    s_n = [''.join(t.split()) for t in txt_n]
    dig_o = np.array([bool(DIGIT.search(t)) for t in txt_o])
    dig_n = np.array([bool(DIGIT.search(t)) for t in txt_n])
    rewritten = np.array([a != b for a, b in zip(s_o, s_n)])
    pos = np.arange(n) / (n - 1)

    # 前瞻句 / 风险句：在原文（去空白）上匹配，见 deviations.md
    exc = np.array([bool(EXC.search(s)) for s in s_o])
    a_hit = np.array([bool(RULE_A.search(s)) for s in s_o])
    fy_hit = np.array([any(int(m.group(1)) > year for m in RULE_FY.finditer(s)) for s in s_o])
    b_hit = np.array([bool(RULE_B.search(s)) for s in s_o])
    c_hit = np.array([bool(RULE_C.search(s)) for s in s_o])
    risk = np.array([bool(RISK.search(s)) for s in s_o])
    selfref = np.array([bool(SELF_REF.search(s)) for s in s_o])
    fwd = (a_hit | fy_hit | b_hit | c_hit) & ~exc
    fwd_pure = fwd & ~risk

    f = {}
    std = o.std()
    f['skew_orig'] = float(((o - o.mean()) ** 3).mean() / std ** 3) if std > 0 else None

    # P5a 包夹：S = 非首尾句且原文为负面
    inner = np.zeros(n, dtype=bool)
    inner[1:-1] = True
    prev_pos = np.zeros(n, dtype=bool)
    next_pos = np.zeros(n, dtype=bool)
    prev_pos[1:] = pos_o[:-1]
    next_pos[:-1] = pos_o[1:]
    sandwiched = (prev_pos & next_pos).astype(float)
    n_pos = pos_o.sum()
    expected = n_pos * (n_pos - 1) / ((n - 1) * (n - 2))
    s_set = inner & neg_o
    f['neg_sandwich_exc'] = float(sandwiched[s_set].mean() - expected) if s_set.sum() >= MIN_GROUP else None
    s_nr = s_set & ~risk
    f['neg_sandwich_exc_norisk'] = float(sandwiched[s_nr].mean() - expected) if s_nr.sum() >= MIN_GROUP else None
    f['neg_sandwich_exc_s10'] = float(sandwiched[s_set].mean() - expected) if s_set.sum() >= 10 else None

    # P5b 硬/软负面
    f['hard_neg_share'] = float((neg_n & dig_n).mean())
    f['soft_neg_share'] = float((neg_n & ~dig_n).mean())
    f['hard_neg_o'] = float((neg_o & dig_o).mean())
    f['soft_neg_o'] = float((neg_o & ~dig_o).mean())
    nr = ~risk
    f['hard_neg_nr'] = float((neg_n & dig_n)[nr].mean()) if nr.any() else None
    f['soft_neg_nr'] = float((neg_n & ~dig_n)[nr].mean()) if nr.any() else None

    # P5c 套话
    f['boiler_share'] = float((neu_o & ~dig_o).mean())
    f['neu_digit_share'] = float((neu_o & dig_o).mean())

    # P6 负面事实的位置
    neg_fact = neg_n & ~fwd & ~risk
    f['pos_neg_fact'] = masked_mean(pos, neg_fact, 5)
    f['pos_neg_fact_m3'] = masked_mean(pos, neg_fact, 3)
    f['pos_neg_fact_m10'] = masked_mean(pos, neg_fact, 10)
    f['pos_neg_all'] = masked_mean(pos, neg_n, 5)
    f['pos_pos_all'] = masked_mean(pos, pos_n, 1)
    f['neg_fact_share'] = float(neg_fact.mean())

    # P7 单边修辞
    def one_two(thr, extra=None):
        pm = pos_o if extra is None else pos_o & extra
        nm = neg_o if extra is None else neg_o & extra
        p, q = masked_mean(c, pm, thr), masked_mean(c, nm, thr)
        return p, q

    p5, q5 = one_two(5)
    f['pos_change_mean'], f['neg_change_mean'] = p5, q5
    f['one_sided'] = -(p5 + q5) if p5 is not None and q5 is not None else None
    f['two_sided'] = q5 - p5 if p5 is not None and q5 is not None else None
    f['frac_neg_orig'] = float(neg_o.mean())
    prw, qrw = one_two(5, rewritten)
    f['one_sided_rw'] = -(prw + qrw) if prw is not None and qrw is not None else None
    p10, q10 = one_two(10)
    f['one_sided_m10'] = -(p10 + q10) if p10 is not None and q10 is not None else None
    f['two_sided_m10'] = q10 - p10 if p10 is not None and q10 is not None else None

    # P8 前瞻句：主规格 fwd_pure；_a 放回风险句；_b 公司自身；_c 高精度（A 或 FY）；_d 门槛 10 句
    hp = (a_hit | fy_hit) & ~exc & ~risk
    for suffix, mask, thr in (('', fwd_pure, 5), ('_a', fwd, 5), ('_b', fwd_pure & selfref, 5), ('_c', hp, 5),
                              ('_d', fwd_pure, 10)):
        f[f'sc_fwd{suffix}'] = masked_mean(c, mask, thr)
        f[f'sc_nonfwd{suffix}'] = masked_mean(c, ~mask, 1)
        f[f'mn_fwd{suffix}'] = masked_mean(x, mask, thr)
        f[f'share_fwd{suffix}'] = float(mask.mean())
    return f


def scan_sentences(keys, cache_path):
    """流式扫描逐句文件；结果缓存到 cache_path，已存在则直接读取。"""
    if cache_path.exists():
        with cache_path.open('rb') as fh:
            return pickle.load(fh)
    result, seen = {}, set()
    with SENTENCE_PATH.open(encoding='utf-8-sig', newline='') as fh:
        reader = csv.reader(fh)
        header = next(reader)
        col = {c: header.index(c) for c in SENTENCE_COLS}
        groups = itertools.groupby(reader, key=lambda r: (r[col['code']], r[col['year']]))
        for n_docs, (key, group) in enumerate(groups, 1):
            if key in seen:
                raise SystemExit(f'年报 {key} 的句子不连续，请检查结果文件')
            seen.add(key)
            if key in keys:
                result[key] = doc_features(list(group), col, int(key[1]))
            if n_docs % 2000 == 0:
                print(f'已处理 {n_docs} 篇年报', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open('wb') as fh:
        pickle.dump(result, fh)
    return result


# ---------------------------------------------------------------- 财务和行情
def load_fin_extra(path):
    """返回 {(code, 年份): 年报当期财务字段}，NaN 视同缺失。"""
    columns = ['code', 'report_date', 'company_type', 'fixed_assets', 'total_assets', 'total_liability',
               'operating_revenue_ttm', 'operating_cost_ttm', 'total_operating_revenue_ttm', 'net_profit_ttm',
               'np_parent_company_owners_ttm']
    fin = {}
    for r in pq.read_table(path, columns=columns).to_pylist():
        if r['report_date'].endswith('1231'):
            fin[r['code'], int(r['report_date'][:4])] = {k: clean(v) for k, v in r.items()}
    return fin


def load_cap(daily_path, days, code_col):
    """[日, 股] 总市值矩阵，轴与 load_returns 一致。"""
    t = pq.read_table(daily_path, columns=['date', 'code', 'mkcap'])
    day_idx = np.searchsorted(days, pc.cast(t['date'], 'int32').to_numpy())
    encoded = t['code'].combine_chunks().dictionary_encode()
    mapping = np.array([code_col[c] for c in encoded.dictionary.to_pylist()])
    stock_idx = mapping[encoded.indices.to_numpy(zero_copy_only=False)]
    cap = np.full((len(days), len(code_col)), np.nan)
    cap[day_idx, stock_idx] = t['mkcap'].to_numpy(zero_copy_only=False).astype(float)
    return cap


def avg_rank(x):
    """平均秩（并列取平均），从 1 起。"""
    uniq, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    cum = np.cumsum(counts)
    return ((cum - counts + 1 + cum) / 2.0)[inv]


def rank10_cells(values, valid, cells, min_cell=MIN_CELL):
    """在 cells 内把 values 分十等：ceil(10 * 平均秩 / N_cell)；N_cell < min_cell 或 valid 为假时为 NaN。"""
    out = np.full(len(values), np.nan)
    order = {}
    for i in np.flatnonzero(valid):
        order.setdefault(cells[i], []).append(i)
    for idx in order.values():
        idx = np.array(idx)
        if len(idx) < min_cell:
            continue
        out[idx] = np.ceil(10 * avg_rank(values[idx]) / len(idx))
    return out


def rank10_earlier(values, valid, cells, event_date, min_peers=MIN_PEERS_2D):
    """P2-d：只在同 cell、事件日严格更早的同行（加自己）中定位十分位；同行少于 min_peers 时为 NaN。"""
    out = np.full(len(values), np.nan)
    by_cell = {}
    for i in np.flatnonzero(valid):
        by_cell.setdefault(cells[i], []).append(i)
    for idx in by_cell.values():
        idx = np.array(idx)
        v, d = values[idx], event_date[idx]
        for k, i in enumerate(idx):
            peers = v[d < d[k]]
            if len(peers) < min_peers:
                continue
            less = (peers < v[k]).sum()
            equal = (peers == v[k]).sum() + 1
            rank = less + (equal + 1) / 2.0
            out[i] = math.ceil(10 * rank / (len(peers) + 1))
    return out


def ols_resid(y, X):
    """带截距的 OLS 残差和 R2。"""
    A = np.column_stack([np.ones(len(y)), X])
    coef = np.linalg.lstsq(A, y, rcond=None)[0]
    resid = y - A @ coef
    return resid, 1 - (resid @ resid) / (((y - y.mean()) ** 2).sum()), coef


X_COLS = ['roe', 'd_roe', 'loss', 'ln_assets', 'leverage', 'sue', 'btm', 'ret_y', 'std_ret', 'std_roe']


def first_stage_by_year(tone, X, year, extra=None, min_n=MIN_FS_OBS):
    """P3 逐年截面 OLS 残差。extra：(n,k) 附加回归量（如行业哑变量），其缺失行不进入。返回 (残差数组, {年: (n, r2)})。"""
    resid = np.full(len(tone), np.nan)
    info = {}
    for y in np.unique(year):
        sel = year == y
        mats = [X] + ([extra] if extra is not None else [])
        ok = sel & ~np.isnan(tone) & ~np.isnan(np.column_stack(mats)).any(axis=1)
        if ok.sum() < min_n:
            info[int(y)] = (int(ok.sum()), math.nan)
            continue
        cols = np.column_stack(mats)[ok]
        if extra is not None:  # 行业哑变量：去掉全零列
            keep = np.ones(cols.shape[1], dtype=bool)
            keep[X.shape[1]:] = cols[:, X.shape[1]:].std(axis=0) > 0
            cols = cols[:, keep]
        r, r2, _ = ols_resid(tone[ok], cols)
        resid[ok] = r
        info[int(y)] = (int(ok.sum()), float(r2))
    return resid, info


def first_stage_expanding(tone, X, year, first_year=2014):
    """P3-a：只用 Y 之前各年合并估计一阶段（Y >= first_year），用当年 X 算残差，无前视。"""
    resid = np.full(len(tone), np.nan)
    ok_all = ~np.isnan(tone) & ~np.isnan(X).any(axis=1)
    for y in np.unique(year):
        if y < first_year:
            continue
        train, test = ok_all & (year < y), ok_all & (year == y)
        if train.sum() < MIN_FS_OBS or test.sum() < MIN_FS_OBS:
            continue
        A = np.column_stack([np.ones(train.sum()), X[train]])
        coef = np.linalg.lstsq(A, tone[train], rcond=None)[0]
        resid[test] = tone[test] - np.column_stack([np.ones(test.sum()), X[test]]) @ coef
    return resid


def main(max_docs=None):
    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(PANEL_PATH)  # 只含 ALLOWED_PANEL_COLS，不含 car_* 和 beta
    keys = {(r['code'], r['year']) for r in panel}
    assert len(keys) == len(panel)
    n_ev = len(panel)
    code = np.array([r['code'] for r in panel])
    year = np.array([int(r['year']) for r in panel])
    event_date = np.array([int(r['event_date']) for r in panel])

    def col(name):
        return np.array([float(r[name]) if r[name] != '' else math.nan for r in panel])

    # ---- cs_sentiment：n_sentences、sent_std_neutral、上一年语调
    with CS_PATH.open(encoding='utf-8-sig', newline='') as fh:
        cs = {(r['code'], int(r['year'])): r for r in csv.DictReader(fh)}

    def cs_get(c, y, name):
        r = cs.get((c, y))
        return math.nan if r is None or r[name] == '' else float(r[name])

    n_sent = np.array([cs_get(c, y, 'n_sentences') for c, y in zip(code, year)])
    std_o, std_n = col('sent_std_orig'), np.array([cs_get(c, y, 'sent_std_neutral') for c, y in zip(code, year)])
    mean_o, mean_n, sc = col('sent_mean_orig'), col('sent_mean_neutral'), col('sent_change_mean')
    roe, sue, ln_assets, leverage = col('roe'), col('sue'), col('ln_assets'), col('leverage')

    # ---- 逐句变量
    docs = scan_sentences(keys, out_dir / 'cache' / 'sentence_vars.pkl')
    print(f'逐句变量完成：{len(docs)} 篇年报', flush=True)
    V = {name: np.full(n_ev, math.nan) for name in SENT_VARS}
    for i, (c, y) in enumerate(zip(code, year)):
        f = docs.get((c, str(y)))
        if f:
            for name in SENT_VARS:
                V[name][i] = nan_if_none(f[name])
    V['n_sentences'] = n_sent

    # ---- P4
    ok4 = (n_sent >= MIN_SENT) & (std_o >= MIN_STD) & (std_n >= MIN_STD)
    with np.errstate(invalid='ignore', divide='ignore'):
        V['ln_std_orig'] = np.where(ok4, np.log(std_o), np.nan)
        V['ln_std_neutral'] = np.where(ok4, np.log(std_n), np.nan)
        V['ln_ret'] = V['ln_std_neutral'] - V['ln_std_orig']
        V['ret_ratio'] = np.where(ok4, std_n / std_o, np.nan)
    with HYP_VARS_PATH.open(encoding='utf-8-sig', newline='') as fh:
        unchanged = {(r['code'], int(r['year'])): r['unchanged_rate'] for r in csv.DictReader(fh)}
    V['unchanged_rate'] = np.array([float(unchanged[c, y]) if unchanged.get((c, y), '') != '' else math.nan
                                    for c, y in zip(code, year)])

    # ---- 财务
    print('读取财务数据', flush=True)
    fund = load_fundamentals(str(FIN_PATH))
    fin = load_fin_extra(FIN_PATH)
    ctype = np.array([nan_if_none(fin.get((c, y), {}).get('company_type')) for c, y in zip(code, year)])
    V['company_type'] = ctype
    V['abs_sue'] = np.abs(sue)

    # ---- P1
    prev_mean_n = np.array([cs_get(c, y - 1, 'sent_mean_neutral') for c, y in zip(code, year)])
    prev_sc = np.array([cs_get(c, y - 1, 'sent_change_mean') for c, y in zip(code, year)])
    prev_n = np.array([cs_get(c, y - 1, 'n_sentences') for c, y in zip(code, year)])
    prev_roe = np.array([nan_if_none(fund.get((c, y - 1), {}).get('roe')) for c, y in zip(code, year)])
    V['d_neutral'] = mean_n - prev_mean_n
    V['d_rhet'] = -(sc - prev_sc)
    V['d_orig'] = V['d_neutral'] + V['d_rhet']
    V['d_roe'] = roe - prev_roe
    with np.errstate(invalid='ignore', divide='ignore'):
        V['d_ln_n'] = np.log(n_sent) - np.log(prev_n)
    V['sent_mean_neutral_l1'] = prev_mean_n
    V['sc_l1'] = prev_sc
    assert np.nanmax(np.abs(V['d_orig'] - ((mean_o - np.array([cs_get(c, y - 1, 'sent_mean_orig') for c, y in zip(code, year)]))))) < 5e-6  # 三个量各自四舍五入到 6 位小数

    # ---- P2
    nonfin = ctype == 99
    group = np.where(np.isnan(ctype), '', np.where(nonfin, 'nonfin', 'fin'))
    np_ttm = np.array([nan_if_none(fin.get((c, y), {}).get('net_profit_ttm')) for c, y in zip(code, year)])
    assets = np.array([nan_if_none(fin.get((c, y), {}).get('total_assets')) for c, y in zip(code, year)])
    rev = np.array([nan_if_none(fin.get((c, y), {}).get('total_operating_revenue_ttm')) for c, y in zip(code, year)])
    with np.errstate(invalid='ignore', divide='ignore'):
        perfs = {'': roe, '_roa': np.where(assets > 0, np_ttm / assets, np.nan),
                 '_nm': np.where(rev > 0, np_ttm / rev, np.nan)}
    # 伪行业（P2-a）：Y-1 年毛利率三分位 × 固定资产/总资产三分位，非金融，年内分位
    gm, fa = np.full(n_ev, math.nan), np.full(n_ev, math.nan)
    for i, (c, y) in enumerate(zip(code, year)):
        f = fin.get((c, y - 1))
        if f is None:
            continue
        r_, cost = f['operating_revenue_ttm'], f['operating_cost_ttm']
        if r_ and r_ > 0 and cost is not None:
            gm[i] = (r_ - cost) / r_
        if f['fixed_assets'] is not None and f['total_assets'] and f['total_assets'] > 0:
            fa[i] = f['fixed_assets'] / f['total_assets']
    pseudo = np.full(n_ev, math.nan)
    for y in np.unique(year):
        sel = nonfin & (year == y) & ~np.isnan(gm) & ~np.isnan(fa)
        if sel.sum() < MIN_CELL:
            continue
        t_gm = np.digitize(gm[sel], np.quantile(gm[sel], [1 / 3, 2 / 3]), right=True)
        t_fa = np.digitize(fa[sel], np.quantile(fa[sel], [1 / 3, 2 / 3]), right=True)
        pseudo[sel] = 3 * t_gm + t_fa
    V['pseudo_ind'] = pseudo

    def gaps(perf, cells, valid_extra=None, earlier=False):
        valid = ~np.isnan(mean_o) & ~np.isnan(mean_n) & ~np.isnan(perf) & (cells != '')
        if valid_extra is not None:
            valid &= valid_extra
        if earlier:
            ro = rank10_earlier(mean_o, valid, cells, event_date)
            rn = rank10_earlier(mean_n, valid, cells, event_date)
            rp = rank10_earlier(perf, valid, cells, event_date)
        else:
            ro, rn, rp = (rank10_cells(v, valid, cells) for v in (mean_o, mean_n, perf))
        ad_o, ad_n = np.abs(ro - rp), np.abs(rn - rp)
        return ro, rn, rp, ad_n, ad_o - ad_n

    cells_yg = np.array([f'{y}|{g}' if g else '' for y, g in zip(year, group)])
    ro, rn, rp, ad_n, gap = gaps(perfs[''], cells_yg)
    V['R_o'], V['R_p'], V['abs_delta_neutral'], V['gap_delta'] = ro, rp, ad_n, gap
    for suffix in ('_roa', '_nm'):
        *_, V[f'abs_delta_neutral{suffix}'], V[f'gap_delta{suffix}'] = gaps(perfs[suffix], cells_yg)
    cells_ind = np.array([f'{y}|{int(p)}' if not np.isnan(p) else '' for y, p in zip(year, pseudo)])
    *_, V['abs_delta_neutral_ind'], V['gap_delta_ind'] = gaps(perfs[''], cells_ind)
    *_, V['abs_delta_neutral_d'], V['gap_delta_d'] = gaps(perfs[''], cells_yg, earlier=True)

    # ---- 行情变量（只用事件日之前）
    print('读取行情数据', flush=True)
    days, code_col, ret, market = load_returns(str(DAILY_PATH))
    turnover = load_turnover(str(DAILY_PATH), days, code_col)
    cap = load_cap(str(DAILY_PATH), days, code_col)
    ln_turn, runup = np.full(n_ev, math.nan), np.full(n_ev, math.nan)
    sigma = np.full(n_ev, math.nan)
    for i, (c, y, ed) in enumerate(zip(code, year, event_date)):
        a, b = market_vars(c, int(ed), days, code_col, ret, market, turnover)
        ln_turn[i], runup[i] = nan_if_none(a), nan_if_none(b)
        t0 = int(np.searchsorted(days, ed))
        if c in code_col and t0 < len(days) and days[t0] == ed:
            fit = market_model(ret[:, code_col[c]], market, t0)
            if fit is not None:
                alpha, beta = fit
                er, em = ret[t0 + EST_START:t0 + EST_END + 1, code_col[c]], market[t0 + EST_START:t0 + EST_END + 1]
                ok = ~np.isnan(er) & ~np.isnan(em)
                sigma[i] = float((er[ok] - alpha - beta * em[ok]).std(ddof=1))
    V['ln_turnover_60'], V['runup_60'] = ln_turn, runup
    V['sigma_est'] = sigma
    with np.errstate(invalid='ignore', divide='ignore'):
        V['ln_sigma_est'] = np.log(sigma)

    # ---- P3 一阶段回归量（年度内：当年 Y0101-Y1231 的行情，事件日之前）
    loss = np.array([nan_if_none(None if fin.get((c, y), {}).get('np_parent_company_owners_ttm') is None
                                 else float(fin[c, y]['np_parent_company_owners_ttm'] < 0)) for c, y in zip(code, year)])
    btm, ret_y, std_ret, std_roe = (np.full(n_ev, math.nan) for _ in range(4))
    for i, (c, y) in enumerate(zip(code, year)):
        if c not in code_col:
            continue
        j = code_col[c]
        lo, hi = int(np.searchsorted(days, y * 10000 + 101)), int(np.searchsorted(days, (y + 1) * 10000 + 101))
        r_ = ret[lo:hi, j]
        ex = r_ - market[lo:hi]
        ok = ~np.isnan(ex)
        if ok.sum() >= MIN_RET_DAYS:
            ret_y[i] = float(ex[ok].sum())
        ok = ~np.isnan(r_)
        if ok.sum() >= MIN_RET_DAYS:
            std_ret[i] = float(r_[ok].std(ddof=1))
        caps = cap[lo:hi, j]
        good = np.flatnonzero(caps > 0)
        f = fin.get((c, y))
        if len(good) and f and f['total_assets'] is not None and f['total_liability'] is not None:
            btm[i] = (f['total_assets'] - f['total_liability']) / caps[good[-1]]
        hist = [fund[c, k]['roe'] for k in range(y - 4, y + 1) if (c, k) in fund and fund[c, k]['roe'] is not None]
        if len(hist) >= 3:
            std_roe[i] = float(np.std(hist, ddof=1))
    V.update({'loss': loss, 'btm': btm, 'ret_y': ret_y, 'std_ret': std_ret, 'std_roe': std_roe})
    Xmat = np.column_stack([{'roe': roe, 'd_roe': V['d_roe'], 'ln_assets': ln_assets, 'leverage': leverage,
                             'sue': sue}.get(n, V.get(n)) for n in X_COLS])
    fs_rows = []
    for tone_name, tone in (('orig', mean_o), ('neutral', mean_n)):
        V[f'at_{tone_name}'], info = first_stage_by_year(tone, Xmat, year)
        fs_rows += [{'tone': tone_name, 'spec': 'main', 'year': y, 'n': n, 'r2': r2} for y, (n, r2) in info.items()]
        V[f'at_{tone_name}_a'] = first_stage_expanding(tone, Xmat, year)
        dummies = np.full((n_ev, 8), math.nan)
        has = ~np.isnan(pseudo)
        for g in range(8):  # 9 组去掉最后一组
            dummies[has, g] = (pseudo[has] == g).astype(float)
        V[f'at_{tone_name}_d'], info = first_stage_by_year(tone, Xmat, year, extra=dummies)
        fs_rows += [{'tone': tone_name, 'spec': 'pseudo_ind_FE', 'year': y, 'n': n, 'r2': r2} for y, (n, r2) in info.items()]
    V['at_rhet'] = V['at_orig'] - V['at_neutral']
    with (out_dir / 'p3_first_stage.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['tone', 'spec', 'year', 'n', 'r2'])
        w.writeheader()
        w.writerows(fs_rows)

    # ---- 写出
    names = ['n_sentences', 'company_type', 'unchanged_rate'] + [n for n in V if n not in
                                                                  ('n_sentences', 'company_type', 'unchanged_rate')]
    names = list(dict.fromkeys(names))
    with (out_dir / 'variables.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['code', 'year'] + names)
        for i in range(n_ev):
            w.writerow([code[i], year[i]] + ['' if np.isnan(V[n][i]) else round(float(V[n][i]) + 0.0, 10) for n in names])
    with (out_dir / 'variables_summary.csv').open('w', encoding='utf-8-sig', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(['variable', 'n_nonmissing', 'mean', 'sd', 'min', 'max'])
        for n in names:
            v = V[n][~np.isnan(V[n])]
            w.writerow([n, len(v)] + ([float(v.mean()), float(v.std(ddof=1)), float(v.min()), float(v.max())] if len(v) > 1 else ['', '', '', '']))
    g = V['gap_delta'][~np.isnan(V['gap_delta'])]
    print(f'共 {n_ev} 行，{len(names)} 个变量；gap_delta==0 占比 {(g == 0).mean():.4f}（N={len(g)}）')
    for n in names:
        print(f'{n:<28}{int((~np.isnan(V[n])).sum()):>7}')


if __name__ == '__main__':
    main()
