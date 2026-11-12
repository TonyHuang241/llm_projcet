"""MD&A 语调分解：回归（第二步）。读取 variables.csv 和 event_panel.csv，按设计文档运行 11 个主检验及稳健性。

规格来源：mda_tone/output/event_study/tone_decomposition/design.md。复用 run_hypothesis_regressions 的 fe_ols2（公司聚类 + 公司/事件日双向聚类）、
run_one、bh_q、vif_max，以及 event_study 的 winsorize、CONTROLS、HORIZONS；缩尾用显式列清单（WINSORIZE_COLS），不缩尾交互项的乘积。
第一步是校准：必须复现 car_15 对 sent_change_mean / sent_std_orig 的系数和 t，否则立即中止、不写任何结果文件。

输出（out_dir）：primary_tests.csv、regression_all.csv、robustness.csv

用法：在项目根目录运行 python -m mda_tone.tone_decomposition_reg
"""
import math
import warnings
from pathlib import Path

import numpy as np

from .event_study import CONTROLS, HORIZONS, winsorize, load_returns
from .run_hypothesis_regressions import bh_q, fe_ols2, pvalue, read_csv, run_one, vif_max, write_csv
from .tone_decomposition import (DAILY_PATH, OUT_DIR, PANEL_PATH, X_COLS, first_stage_by_year)

SEED = 20261108
B_BOOT = 499

# 缩尾清单（显式）：事件面板和 variables.csv 的全部列都必须出现在 WINSORIZE_COLS 或 NO_WINSOR_COLS，否则报错
WINSORIZE_COLS = (
    [f'car_{k}' for k in HORIZONS] + [f'abs_car_{k}' for k in HORIZONS]
    + ['sent_mean_orig', 'sent_mean_neutral', 'sent_change_mean', 'sent_std_orig', 'pos_change_mean_panel', 'sent_std_neutral']
    + ['ln_assets', 'roe', 'leverage', 'sue', 'ln_nsent', 'abs_sue']
    + ['unchanged_rate', 'ln_std_orig', 'ln_std_neutral', 'ln_ret', 'ret_ratio']
    + ['d_neutral', 'd_rhet', 'd_orig', 'd_roe', 'd_ln_n', 'sent_mean_neutral_l1', 'sc_l1']
    + ['gap_delta', 'gap_delta_roa', 'gap_delta_nm', 'gap_delta_ind', 'gap_delta_d']
    + ['btm', 'ret_y', 'std_ret', 'std_roe', 'at_orig', 'at_neutral', 'at_rhet', 'at_orig_a', 'at_neutral_a',
       'at_orig_d', 'at_neutral_d']
    + ['ln_turnover_60', 'runup_60', 'sigma_est', 'ln_sigma_est']
    + ['skew_orig', 'neg_sandwich_exc', 'neg_sandwich_exc_norisk', 'neg_sandwich_exc_s10',
       'hard_neg_share', 'soft_neg_share', 'hard_neg_o', 'soft_neg_o', 'hard_neg_nr', 'soft_neg_nr',
       'boiler_share', 'neu_digit_share',
       'pos_neg_fact', 'pos_pos_all', 'neg_fact_share', 'pos_neg_all', 'pos_neg_fact_m3', 'pos_neg_fact_m10',
       'pos_change_mean', 'neg_change_mean', 'one_sided', 'two_sided', 'frac_neg_orig',
       'one_sided_rw', 'one_sided_m10', 'two_sided_m10']
    + [f'{v}{s}' for s in ('', '_a', '_b', '_c', '_d') for v in ('sc_fwd', 'sc_nonfwd', 'mn_fwd', 'share_fwd')]
    + ['ab_turn_3', 'abs_car3_scaled'])
NO_WINSOR_COLS = (['year', 'announce_date', 'event_date', 'beta', 'n_sentences', 'company_type', 'loss', 'pseudo_ind',
                   'R_o', 'R_p', 'abs_delta_neutral', 'abs_delta_neutral_roa', 'abs_delta_neutral_nm',
                   'abs_delta_neutral_ind', 'abs_delta_neutral_d'])


def load_data2(with_extra=True):
    """读取事件面板 + variables.csv，只对 WINSORIZE_COLS 缩尾（pooled 全样本 1%/99%）。返回 (data, raw, meta)。
    乘积（交互项）不在此处构造；需要时必须在成分缩尾、去均值后相乘，乘积不缩尾。"""
    panel = read_csv(PANEL_PATH)
    var_rows = read_csv(OUT_DIR / 'variables.csv')
    assert len(panel) == len(var_rows), '行数不一致'
    for p, v in zip(panel, var_rows):
        assert p['code'] == v['code'] and p['year'] == v['year'], 'variables 与 event_panel 不一一对应'

    def arr(rows, name):
        return np.array([float(r[name]) if r[name] != '' else np.nan for r in rows])

    raw = {}
    for name in panel[0]:
        if name not in ('code', 'company'):
            raw['pos_change_mean_panel' if name == 'pos_change_mean' else name] = arr(panel, name)
    for name in var_rows[0]:
        if name not in ('code', 'year'):
            assert name not in raw, f'列重复 {name}'
            raw[name] = arr(var_rows, name)
    with np.errstate(invalid='ignore', divide='ignore'):
        raw['ln_nsent'] = np.log(raw['n_sentences'])
    for k in HORIZONS:
        raw[f'abs_car_{k}'] = np.abs(raw[f'car_{k}'])  # 先取绝对值再缩尾
    if with_extra:
        for name, arr_ in extra_columns(raw).items():
            raw[name] = arr_
    allowed = set(WINSORIZE_COLS) | set(NO_WINSOR_COLS)
    unknown = set(raw) - allowed
    assert not unknown, f'列未在缩尾清单中归类：{sorted(unknown)}'
    data = {name: (winsorize(x) if name in WINSORIZE_COLS else x) for name, x in raw.items()}
    code = np.array([r['code'] for r in panel])
    meta = {'firm': np.unique(code, return_inverse=True)[1], 'date': np.unique(raw['event_date'], return_inverse=True)[1],
            'year': raw['year'].astype(int), 'code': code, 'n_events': len(panel)}
    return data, raw, meta


def extra_columns(raw):
    """P5c 的 abs_car3_scaled 和次要因变量 ab_turn_3（后者读取事件日之后的换手率，只在回归阶段构造）。"""
    out = {}
    with np.errstate(invalid='ignore', divide='ignore'):
        out['abs_car3_scaled'] = np.abs(raw['car_3']) / (raw['sigma_est'] * 2)
    out['ab_turn_3'] = ab_turnover(raw)
    return out


def ab_turnover(raw):
    """ab_turn_3 = [0,3] 日均 ln 换手率 - [-60,-1] 日均 ln 换手率，再减去同一事件日全市场横截面均值。缓存到 cache/ab_turn.npy。"""
    cache = OUT_DIR / 'cache' / 'ab_turn.npy'
    if cache.exists():
        return np.load(cache)
    from .construct_hypothesis_vars import load_turnover
    days, code_col, ret, market = load_returns(str(DAILY_PATH))
    ln_turn = np.log(load_turnover(str(DAILY_PATH), days, code_col).astype(float))
    panel = read_csv(PANEL_PATH)
    t0s = np.searchsorted(days, raw['event_date'].astype(int))
    out = np.full(len(panel), np.nan)
    market_mean = {}
    for t0 in np.unique(t0s):
        if t0 < 60 or t0 + 3 >= len(days):
            continue
        post, pre = ln_turn[t0:t0 + 4], ln_turn[t0 - 60:t0]
        with np.errstate(invalid='ignore'), warnings.catch_warnings():
            warnings.simplefilter('ignore')
            diff = np.where((~np.isnan(post)).sum(0) >= 3, np.nanmean(post, axis=0), np.nan) \
                - np.where((~np.isnan(pre)).sum(0) >= 20, np.nanmean(pre, axis=0), np.nan)
        market_mean[int(t0)] = (diff, np.nanmean(diff))
    for i, (r, t0) in enumerate(zip(panel, t0s)):
        if int(t0) in market_mean and r['code'] in code_col:
            diff, mm = market_mean[int(t0)]
            out[i] = diff[code_col[r['code']]] - mm
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def calibrate(data, meta):
    """规格校准：用新的读取函数复现 car_15 的 sent_change_mean / sent_std_orig 结果；不通过则抛出异常。"""
    result = {}
    for key, sent, coef0, t0 in [('sc', 'sent_change_mean', 0.011494, 0.73), ('std', 'sent_std_orig', 0.047406, 3.53)]:
        r = run_one(data, meta, 'car_15', [sent] + CONTROLS)
        t = r['coef'][0] / r['se_firm'][0]
        print(f'校准 car_15 ~ {sent}: coef={r["coef"][0]:.6f} (目标 {coef0}), t_firm={t:.4f} (目标 {t0}), n={r["n"]}')
        result[f'cal_{key}_coef'], result[f'cal_{key}_t'] = float(r['coef'][0]), float(t)
        if not (abs(r['coef'][0] - coef0) < 5e-7 and abs(t - t0) < 0.006):
            raise SystemExit(f'规格校准失败：{sent} coef={r["coef"][0]:.6f} t={t:.4f}，已中止，未写任何结果文件')
    return result


# ---------------------------------------------------------------- 规格
def _mask_year_not_2021(d, m):
    return m['year'] != 2021


SPECS = {
    'P1': dict(focal=['d_neutral', 'd_rhet'], extra=['d_roe', 'd_ln_n'], controls=CONTROLS, k=5, kind='car',
               mask=_mask_year_not_2021, tests={'P1a': ('d_neutral', '+'), 'P1b': ('d_rhet', '-')}),
    'P2': dict(focal=['gap_delta'], extra=['abs_delta_neutral'], controls=CONTROLS, k=5, kind='car', mask=None,
               tests={'P2': ('gap_delta', '-')}),
    'P3': dict(focal=['at_neutral'], extra=['at_orig'], controls=CONTROLS, k=20, kind='car', mask=None,
               tests={'P3': ('at_neutral', '+')}),
    'P4': dict(focal=['ln_ret'], extra=['ln_std_orig', 'sent_mean_orig', 'ln_nsent'], controls=CONTROLS, k=15, kind='car',
               mask=None, tests={'P4': ('ln_ret', '+')}),
    'P5a': dict(focal=['neg_sandwich_exc'], extra=['sent_mean_orig', 'sent_std_orig', 'ln_nsent'], controls=CONTROLS, k=20,
                kind='car', mask=None, tests={'P5a': ('neg_sandwich_exc', '-')}),
    'P5b': dict(focal=['hard_neg_share'], extra=['soft_neg_share', 'sent_mean_neutral', 'ln_nsent'], controls=CONTROLS, k=3,
                kind='car', mask=None, tests={'P5b': ('hard_neg_share', '-')}),
    'P5c': dict(focal=['boiler_share'], extra=['neu_digit_share', 'ln_sigma_est', 'abs_sue'],
                controls=['ln_assets', 'roe', 'leverage', 'ln_turnover_60', 'ln_nsent'], k=3, kind='abs_car', mask=None,
                tests={'P5c': ('boiler_share', '-')}),
    'P6': dict(focal=['pos_neg_fact'], extra=['pos_pos_all', 'neg_fact_share', 'sent_mean_neutral', 'ln_nsent'],
               controls=CONTROLS, k=20, kind='car', mask=None, tests={'P6': ('pos_neg_fact', '-')}),
    'P7': dict(focal=['one_sided'], extra=['two_sided', 'sent_mean_neutral', 'frac_neg_orig', 'ln_nsent'],
               controls=CONTROLS, k=20, kind='car', mask=None, tests={'P7': ('one_sided', '-')}),
    'P8': dict(focal=['sc_fwd'], extra=['sc_nonfwd', 'mn_fwd', 'share_fwd', 'sent_mean_neutral', 'ln_nsent'],
               controls=CONTROLS, k=20, kind='car', mask=None, tests={'P8': ('sc_fwd', '+')}),
}


def spec_terms(spec):
    terms = []
    for t in spec['focal'] + spec['extra'] + spec['controls']:
        if t not in terms:
            terms.append(t)
    return terms


def outcome_name(spec, k):
    return f"{'abs_car' if spec['kind'] == 'abs_car' else 'car'}_{k}"


def r4_mask(raw, meta):
    code, year = meta['code'], meta['year']
    with np.errstate(invalid='ignore'):
        return (raw['n_sentences'] >= 20) & ~((code == '600030') & (year == 2012)) \
            & ~((code == '601628') & np.isin(year, [2013, 2014]))


def fit(data, raw, meta, outcome, terms, masks):
    """回归并返回结果；masks 为额外的样本限制（布尔数组列表）。"""
    mask = None
    for m in masks:
        if m is not None:
            mask = m if mask is None else mask & m
    return run_one(data, meta, outcome, terms, mask)


def result_dict(res, terms, term, n_extra=None):
    j = terms.index(term)
    b, se_f, se_t = res['coef'][j], res['se_firm'][j], res['se_two'][j]
    return {'coef': float(b), 'se_firm': float(se_f), 't_firm': float(b / se_f), 'p_firm': pvalue(b / se_f),
            'se_twoway': float(se_t), 't_twoway': float(b / se_t), 'p_twoway': pvalue(b / se_t)}


def variants(sid, spec):
    """各命题专属稳健性。每项：(检验名, 说明, 参数字典)。参数：replace/add/remove/terms/outcome/focal/mask/base_mask。"""
    f = spec['focal'][0]
    if sid == 'P1':
        return [('P1-a', '加入 sent_mean_neutral(Y-1) 和 sc(Y-1)', dict(add=['sent_mean_neutral_l1', 'sc_l1'])),
                ('P1-b', '剔除 year=2015', dict(mask=lambda d, r, m: m['year'] != 2015)),
                ('P1-c', '把 2021 放回（跨口径，仅供参考）', dict(base_mask=False))]
    if sid == 'P2':
        return [('P2-a', '伪行业：Y-1 毛利率三分位×固定资产占比三分位（非金融）×年度排序',
                 dict(replace={'gap_delta': 'gap_delta_ind', 'abs_delta_neutral': 'abs_delta_neutral_ind'})),
                ('P2-c1', 'perf 改为 ROA', dict(replace={'gap_delta': 'gap_delta_roa', 'abs_delta_neutral': 'abs_delta_neutral_roa'})),
                ('P2-c2', 'perf 改为净利率', dict(replace={'gap_delta': 'gap_delta_nm', 'abs_delta_neutral': 'abs_delta_neutral_nm'})),
                ('P2-d', '只在同年事件日更早的公司中定位分位',
                 dict(replace={'gap_delta': 'gap_delta_d', 'abs_delta_neutral': 'abs_delta_neutral_d'})),
                ('P2-e', '只保留 R_o > R_p', dict(mask=lambda d, r, m: d['R_o'] > d['R_p']))]
    if sid == 'P3':
        return [('P3-a', '一阶段只用 Y 之前各年合并估计（Y>=2014），无前视',
                 dict(replace={'at_neutral': 'at_neutral_a', 'at_orig': 'at_orig_a'})),
                ('P3-d', '一阶段加伪行业固定效应', dict(replace={'at_neutral': 'at_neutral_d', 'at_orig': 'at_orig_d'})),
                ('P3-b', '一步法：neutral + orig + X×year + FE（不含 CONTROLS）', 'special')]
    if sid == 'P4':
        return [('P4-a', 'ret = std_neutral/std_orig 与 sent_std_orig 水平值',
                 dict(replace={'ln_ret': 'ret_ratio', 'ln_std_orig': 'sent_std_orig'})),
                ('P4-b', '加入 unchanged_rate', dict(add=['unchanged_rate'])),
                ('P4-sec', '次要：ln_std_neutral 与 ln_std_orig-ln_std_neutral（β1, β2）',
                 dict(terms=['ln_std_neutral', 'diff_orig_neutral', 'sent_mean_orig', 'ln_nsent'] + CONTROLS,
                      focal=['ln_std_neutral', 'diff_orig_neutral']))]
    if sid == 'P5a':
        return [('P5a-sec', '次要：skew_orig 作为焦点（预期 +）', dict(replace={'neg_sandwich_exc': 'skew_orig'})),
                ('P5-a', '计算 sand 时把风险句移出 S', dict(replace={'neg_sandwich_exc': 'neg_sandwich_exc_norisk'})),
                ('P5-b', '|S| >= 10', dict(replace={'neg_sandwich_exc': 'neg_sandwich_exc_s10'}))]
    if sid == 'P5b':
        return [('P5-c', '原文标签 + 原文数字', dict(replace={'hard_neg_share': 'hard_neg_o', 'soft_neg_share': 'soft_neg_o'})),
                ('P5-d', '剔除风险句后重算 hard/soft', dict(replace={'hard_neg_share': 'hard_neg_nr', 'soft_neg_share': 'soft_neg_nr'}))]
    if sid == 'P5c':
        return [('P5-e', '因变量改为 abs_car_5', dict(outcome='abs_car_5')),
                ('P5-f', '因变量 abs_car_3/(σ_est·2)，去掉 ln_sigma_est', dict(outcome='abs_car3_scaled', remove=['ln_sigma_est'])),
                ('P5c-sec', '次要因变量 ab_turn_3', dict(outcome='ab_turn_3'))]
    if sid == 'P6':
        return [('P6-a', '全部负面句版本 pos_neg_all', dict(replace={'pos_neg_fact': 'pos_neg_all'})),
                ('P6-b3', '门槛 3 句', dict(replace={'pos_neg_fact': 'pos_neg_fact_m3'})),
                ('P6-b10', '门槛 10 句', dict(replace={'pos_neg_fact': 'pos_neg_fact_m10'})),
                ('P6-c', '安慰剂：焦点换成 pos_pos_all', dict(terms=['pos_pos_all', 'neg_fact_share', 'sent_mean_neutral', 'ln_nsent'] + CONTROLS,
                                                         focal=['pos_pos_all']))]
    if sid == 'P7':
        return [('P7-a', 'one_sided 只用被改写的句子', dict(replace={'one_sided': 'one_sided_rw'})),
                ('P7-b', '加入 unchanged_rate', dict(add=['unchanged_rate'])),
                ('P7-c', '门槛 10 句', dict(replace={'one_sided': 'one_sided_m10', 'two_sided': 'two_sided_m10'})),
                ('P7-sec', '次要：pos_change_mean 与 neg_change_mean 分开放',
                 dict(terms=['pos_change_mean', 'neg_change_mean', 'sent_mean_neutral', 'frac_neg_orig', 'ln_nsent'] + CONTROLS,
                      focal=['pos_change_mean', 'neg_change_mean']))]
    if sid == 'P8':
        rep = lambda s: {v: f'{v}{s}' for v in ('sc_fwd', 'sc_nonfwd', 'mn_fwd', 'share_fwd')}
        return [('P8-a', '把风险句放回（fwd_i）', dict(replace=rep('_a'))),
                ('P8-b', '只用含公司自身主语的前瞻句', dict(replace=rep('_b'))),
                ('P8-c', '高精度子集（A 或 FY）', dict(replace=rep('_c'))),
                ('P8-d', '门槛 10 句', dict(replace=rep('_d')))]
    return []


def build_variant(spec, opts):
    """返回 (terms, focal 列表, 结果变量 k 之外的 outcome 覆盖)。"""
    if 'terms' in opts:
        return list(opts['terms']), list(opts['focal'])
    terms = [opts.get('replace', {}).get(t, t) for t in spec_terms(spec)]
    focal = [opts.get('replace', {}).get(t, t) for t in spec['focal']]
    for t in opts.get('add', []):
        if t not in terms:
            terms.append(t)
    terms = [t for t in terms if t not in opts.get('remove', [])]
    return terms, focal


def run_checks(sid, spec, data, raw, meta, extra_state):
    """对一个主检验的主窗口跑 R1-R6 和专属项，返回 [(check, note, terms, focal, res 或 None, error)]。"""
    k = spec['k']
    outcome = outcome_name(spec, k)
    base_terms = spec_terms(spec)
    year = meta['year']
    base_mask = spec['mask'](data, meta) if spec['mask'] else None
    out = []

    def go(check, note, terms, focal, outcome_=outcome, masks=(), use_base=True):
        masks = list(masks) + ([base_mask] if use_base else [])
        try:
            res = fit(data, raw, meta, outcome_, terms, masks)
            out.append((check, note, terms, focal, res, None))
        except Exception as e:  # 记录失败而不静默
            out.append((check, note, terms, focal, None, repr(e)))

    focal = spec['focal']
    # R1：含 ln_nsent 则去掉，否则加入
    t_r1 = [t for t in base_terms if t != 'ln_nsent'] if 'ln_nsent' in base_terms else base_terms + ['ln_nsent']
    go('R1', '去掉 ln_nsent' if 'ln_nsent' in base_terms else '加入 ln_nsent', t_r1, focal)
    go('R2', 'year<=2020', base_terms, focal, masks=[year <= 2020])
    go('R3', 'year>=2021' + ('（P1 仅 2022-2023）' if sid == 'P1' else ''), base_terms, focal, masks=[year >= 2021])
    go('R4', '剔除 n<20、600030/2012、601628/2013-2014', base_terms, focal, masks=[r4_mask(raw, meta)])
    go('R5', '非金融（company_type==99）', base_terms, focal, masks=[raw['company_type'] == 99])
    t_r6 = base_terms + [t for t in ('runup_60', 'ln_turnover_60') if t not in base_terms]
    go('R6', '加入 runup_60 和 ln_turnover_60（已含的不重复）', t_r6, focal)
    for item in variants(sid, spec):
        if item[2] == 'special':
            continue
        check, note, opts = item
        terms, foc = build_variant(spec, opts)
        masks = [opts['mask'](data, raw, meta)] if 'mask' in opts else []
        go(check, note, terms, foc, outcome_=opts.get('outcome', outcome), masks=masks, use_base=opts.get('base_mask', True))
    return out


def p3_one_step(data, raw, meta, k=20):
    """P3-b：car_k ~ sent_mean_neutral + sent_mean_orig + X×year + 公司 FE + 年份 FE。X 为缩尾后的成分，去均值后与年份哑变量相乘，乘积不缩尾。"""
    ok = ~np.isnan(data['at_neutral']) & ~np.isnan(data['at_orig'])
    for n in X_COLS:
        ok &= ~np.isnan(data[n])
    names, local = [], dict(data)
    years = np.unique(meta['year'][ok])
    for n in X_COLS:
        comp = data[n] if n == 'loss' else data[n] - np.nanmean(np.where(ok, data[n], np.nan))
        for y in years:
            col = comp * (meta['year'] == y)
            if col[ok].std() > 0:
                name = f'{n}_x_{y}'
                local[name] = col
                names.append(name)
    terms = ['sent_mean_neutral', 'sent_mean_orig'] + names
    res = run_one(local, meta, f'car_{k}', terms, ok)
    return res, terms


def p3_bootstrap(data, raw, meta, k=20, B=B_BOOT, seed=SEED):
    """P3-c：一阶段 + 二阶段整体的公司块 bootstrap。一阶段用未缩尾的 X 和语调，残差缩尾点沿用原样本，CONTROLS、car 用已缩尾列。"""
    rng = np.random.default_rng(seed)
    X = np.column_stack([raw[n] for n in X_COLS])
    firm, year, date = meta['firm'], meta['year'], meta['date']
    n_firm = firm.max() + 1
    members = [np.flatnonzero(firm == f) for f in range(n_firm)]
    cut = {n: np.nanquantile(raw[n], [0.01, 0.99]) for n in ('at_orig', 'at_neutral')}
    coefs = []
    for b in range(B):
        draw = rng.integers(0, n_firm, n_firm)
        idx = np.concatenate([members[f] for f in draw])
        new_firm = np.repeat(np.arange(n_firm), [len(members[f]) for f in draw])
        yr = year[idx]
        at = {}
        for tone in ('orig', 'neutral'):
            resid, _ = first_stage_by_year(raw[f'sent_mean_{tone}'][idx], X[idx], yr)
            at[tone] = np.clip(resid, *cut[f'at_{tone}'])
        cols = np.column_stack([data[f'car_{k}'][idx], at['neutral'], at['orig']] + [data[c][idx] for c in CONTROLS])
        ok = ~np.isnan(cols).any(axis=1)
        try:
            r = fe_ols2(cols[ok, 0], cols[ok, 1:], new_firm[ok], yr[ok], date[idx][ok])
            coefs.append(float(r['coef'][0]))
        except Exception as e:
            print(f'bootstrap {b} 失败：{e!r}', flush=True)
        if (b + 1) % 50 == 0:
            print(f'bootstrap {b + 1}/{B}', flush=True)
    return np.array(coefs)


def main():
    out_dir = OUT_DIR
    data0, _, meta0 = load_data2(with_extra=False)
    print(f'事件数 {meta0["n_events"]}', flush=True)
    cal = calibrate(data0, meta0)  # 失败则抛出 SystemExit，不写任何结果文件；通过后才构造需要事件日之后数据的列
    data, raw, meta = load_data2()
    data['diff_orig_neutral'] = -data['ln_ret']  # = ln_std_orig - ln_std_neutral（对称缩尾下与先差后缩尾等价）
    data['d_sum'] = data['d_neutral'] + data['d_rhet']

    all_rows, primary, robust = [], [], []
    for sid, spec in SPECS.items():
        terms = spec_terms(spec)
        base_mask = spec['mask'](data, meta) if spec['mask'] else None
        for k in HORIZONS:
            outcome = outcome_name(spec, k)
            res = fit(data, raw, meta, outcome, terms, [base_mask])
            for j, term in enumerate(terms):
                is_focal = term in spec['focal']
                tid = next((t for t, (ft, _) in spec['tests'].items() if ft == term), sid)
                b, se_f, se_t = res['coef'][j], res['se_firm'][j], res['se_two'][j]
                all_rows.append({'id': tid, 'spec': 'main', 'outcome': outcome, 'k': k, 'term': term,
                                 'is_focal': int(is_focal), 'is_primary': int(is_focal and k == spec['k']),
                                 'coef': b, 'se_firm': se_f, 't_firm': b / se_f, 'se_twoway': se_t, 't_twoway': b / se_t,
                                 'p_twoway': pvalue(b / se_t), 'n': res['n'], 'n_firms': res['n_firms'], 'r2_within': res['r2']})
            if k != spec['k']:
                continue
            sample = res['idx']
            for tid, (term, sign) in spec['tests'].items():
                j = terms.index(term)
                f_vals = data[term][sample]
                f_firm = f_vals - (np.bincount(res['firm_sample'], weights=f_vals) / np.bincount(res['firm_sample']))[res['firm_sample']]
                rd = result_dict(res, terms, term)
                sd_focal = float(f_vals.std(ddof=1))
                prow = {'id': tid, 'prop': sid, 'focal_term': term, 'outcome': outcome, 'k': k, 'expected_sign': sign, **rd,
                        'sign_matches': int((rd['coef'] > 0) == (sign == '+')), 'n': res['n'], 'n_firms': res['n_firms'],
                        'r2_within': float(res['r2']), 'sd_focal': sd_focal, 'within_sd_focal': float(f_firm.std(ddof=1)),
                        'effect_1sd': rd['coef'] * sd_focal, 'max_vif': vif_max(res['x_d']), 'n_terms': len(terms), **cal}
                primary.append(prow)
            # 稳健性
            checks = run_checks(sid, spec, data, raw, meta, None)
            for check, note, t_terms, focal, rres, err in checks:
                if rres is None:
                    print(f'稳健性 {sid} {check} 失败：{err}', flush=True)
                    robust.append({'id': sid, 'check': check, 'spec_note': f'{note}（失败：{err}）', 'term': '', 'coef': '',
                                   'se_twoway': '', 't_twoway': '', 'p_twoway': '', 'n': '', 'n_firms': ''})
                    continue
                for term in focal:
                    rd = result_dict(rres, t_terms, term)
                    tid = next((t for t, (ft, _) in spec['tests'].items() if ft == term), sid)
                    row = {'id': tid if check.startswith('R') else sid, 'check': check, 'spec_note': note, 'term': term,
                           'coef': rd['coef'], 'se_twoway': rd['se_twoway'], 't_twoway': rd['t_twoway'],
                           'p_twoway': rd['p_twoway'], 'n': rres['n'], 'n_firms': rres['n_firms']}
                    robust.append(row)
                    if check.startswith('R') and check[1:].isdigit():
                        for p in primary:
                            if p['id'] == tid:
                                p[f'{check.lower()}_coef'], p[f'{check.lower()}_t'], p[f'{check.lower()}_n'] = \
                                    rd['coef'], rd['t_twoway'], rres['n']
            if sid == 'P1':  # 分解的两部分相关性、Wald 差值检验（重参数化：d_sum 的系数 = b_neutral，d_rhet 的系数 = b_rhet - b_neutral）
                x_d = res['x_d']
                corr_fe = float(np.corrcoef(x_d[:, 0], x_d[:, 1])[0, 1])
                corr_raw = float(np.corrcoef(data['d_neutral'][sample], data['d_rhet'][sample])[0, 1])
                robust.append({'id': 'P1', 'check': 'P1-corr', 'spec_note': f'corr(d_neutral,d_rhet) 样本内 {corr_raw:.4f}；去 FE 后 {corr_fe:.4f}；max_vif {vif_max(x_d):.3f}',
                               'term': 'corr', 'coef': corr_raw, 'se_twoway': '', 't_twoway': '', 'p_twoway': '', 'n': res['n'], 'n_firms': res['n_firms']})
                wterms = ['d_sum', 'd_rhet', 'd_roe', 'd_ln_n'] + CONTROLS
                wres = fit(data, raw, meta, outcome, wterms, [base_mask])
                rd = result_dict(wres, wterms, 'd_rhet')
                robust.append({'id': 'P1', 'check': 'P1-wald', 'spec_note': 'b(d_rhet) - b(d_neutral) 的 Wald 检验（次要结果，不计入检验族）',
                               'term': 'd_rhet_minus_d_neutral', 'coef': rd['coef'], 'se_twoway': rd['se_twoway'],
                               't_twoway': rd['t_twoway'], 'p_twoway': rd['p_twoway'], 'n': wres['n'], 'n_firms': wres['n_firms']})
            if sid == 'P3':
                bres, bterms = p3_one_step(data, raw, meta)
                rd = result_dict(bres, bterms, 'sent_mean_neutral')
                robust.append({'id': 'P3', 'check': 'P3-b', 'spec_note': '一步法：neutral + orig + X×year + FE（不含 CONTROLS）；焦点 sent_mean_neutral（对应 at_neutral 的增量）',
                               'term': 'sent_mean_neutral', 'coef': rd['coef'], 'se_twoway': rd['se_twoway'], 't_twoway': rd['t_twoway'],
                               'p_twoway': rd['p_twoway'], 'n': bres['n'], 'n_firms': bres['n_firms']})
                coefs = p3_bootstrap(data, raw, meta)
                lo, hi = np.percentile(coefs, [2.5, 97.5])
                robust.append({'id': 'P3', 'check': 'P3-c', 'spec_note': f'公司块 bootstrap B={len(coefs)}/{B_BOOT} seed={SEED}；95% 区间 [{lo:.6f}, {hi:.6f}]；coef 为原样本点估计，se 为 bootstrap 标准差',
                               'term': 'at_neutral', 'coef': primary[-1]['coef'], 'se_twoway': float(coefs.std(ddof=1)),
                               't_twoway': primary[-1]['coef'] / float(coefs.std(ddof=1)), 'p_twoway': '', 'n': res['n'], 'n_firms': res['n_firms']})

    # 多重检验
    p = np.array([r['p_twoway'] for r in primary])
    q = bh_q(p)
    for r, qq, pp in zip(primary, q, p):
        r['bh_q_primary'] = qq
        r['bonf_p_primary'] = min(1.0, (1.0 if np.isnan(pp) else pp) * len(primary))
        if qq < 0.05 and r['sign_matches']:
            r['verdict'] = '强支持' if r['bonf_p_primary'] < 0.05 else '支持'
        elif qq < 0.05:
            r['verdict'] = '与预期相反'
        else:
            r['verdict'] = '不显著'
    foc = [r for r in all_rows if r['is_focal']]
    q66 = bh_q(np.array([r['p_twoway'] for r in foc]))
    for r in all_rows:
        r['bh_q_all66'] = ''
    for r, qq in zip(foc, q66):
        r['bh_q_all66'] = qq

    fields_p = ['id', 'prop', 'focal_term', 'outcome', 'k', 'expected_sign', 'coef', 'se_firm', 't_firm', 'p_firm',
                'se_twoway', 't_twoway', 'p_twoway', 'bh_q_primary', 'bonf_p_primary', 'sign_matches', 'verdict', 'n',
                'n_firms', 'r2_within', 'sd_focal', 'within_sd_focal', 'effect_1sd', 'max_vif', 'n_terms',
                'cal_sc_coef', 'cal_sc_t', 'cal_std_coef', 'cal_std_t'] + [f'r{i}_{s}' for i in range(1, 7) for s in ('coef', 't', 'n')]
    write_csv(out_dir / 'primary_tests.csv', primary, fields_p)
    write_csv(out_dir / 'regression_all.csv', all_rows,
              ['id', 'spec', 'outcome', 'k', 'term', 'is_focal', 'is_primary', 'coef', 'se_firm', 't_firm', 'se_twoway',
               't_twoway', 'p_twoway', 'bh_q_all66', 'n', 'n_firms', 'r2_within'])
    write_csv(out_dir / 'robustness.csv', robust,
              ['id', 'check', 'spec_note', 'term', 'coef', 'se_twoway', 't_twoway', 'p_twoway', 'n', 'n_firms'])
    print(f'结果保存在 {out_dir}')


if __name__ == '__main__':
    main()
