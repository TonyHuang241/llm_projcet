"""对 hypotheses.json 中的全部事前假说跑公告后 CAR 的固定效应回归。

回归（与 event_study.run_regressions 一致）：
  car_k = b * focal_term + extra_terms + ln_assets + roe + leverage + sue + 公司 FE + 年份 FE
对每条假说、每个窗口 k in {3,5,7,10,15,20} 各跑一次；只跑回归，不改假说、不加变量、不按结果换窗口。
  - 缩尾：连续变量在全样本（event_panel 全部事件的非缺失值）pooled 1%/99% 缩尾；只取值 {0,1} 的哑变量不缩尾。
  - 标准误：(a) 按公司聚类（公式与 fe_ols 相同，含小样本调整）；
            (b) 公司 + 事件日双向聚类（CGM）：V = V_firm + V_date - V_HC，三项都用未调整的 meat，再乘同一调整系数；
                方差为负时记空。
  - p 值用正态近似（math.erfc），双侧。
主检验：每条假说用 expected_window 解析出的唯一窗口（不随结果更改），24 个主检验用双向聚类 p 值做
Benjamini-Hochberg FDR 和 Bonferroni 校正；144 个检验的 BH 只作参考。
稳健性（只对主检验、主窗口）：R1 额外控制 ln(n_sentences)；R2 year<=2020；R3 year>=2021；
R4 剔除 n_sentences<20 的文档和 600030/2012。缩尾分位点沿用全样本，不在子样本内重新缩尾。
开头先做规格校准：必须复现 regression_results.csv 里的 sent_change_mean / sent_std_orig 的 car_15 系数和 t 值。

输出（都在 out_dir 中，utf-8-sig）：
  - regression_all.csv：24 条假说 * 6 个窗口的焦点变量结果
  - primary_tests.csv：24 个主检验及多重检验校正、稳健性
  - regression_controls.csv：主检验回归中所有项的系数（长表）

用法：在项目根目录运行 python -m mda_tone.run_hypothesis_regressions
"""
import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from .config import OUTPUT_DIR
from .event_study import CONTROLS, HORIZONS, drop_singletons, absorb, winsorize

EVENT_DIR = OUTPUT_DIR / 'event_study'
ROBUST = {'r1': '额外控制 ln(n_sentences)', 'r2': 'year<=2020', 'r3': 'year>=2021',
          'r4': '剔除 n_sentences<20 及 600030/2012'}


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def pvalue(t):
    return math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else float('nan')


def fe_ols2(y, x, firm, year, date):
    """双向固定效应 OLS。返回字典：系数、公司聚类标准误、公司+事件日双向聚类标准误、n、公司数、within R2、去 FE 后的回归量等。"""
    keep = drop_singletons(firm, year)
    y, x, date = y[keep], x[keep], date[keep]
    firm0 = firm[keep]
    firm = np.unique(firm0, return_inverse=True)[1]
    year = np.unique(year[keep], return_inverse=True)[1]
    date = np.unique(date, return_inverse=True)[1]
    demeaned = absorb(np.column_stack([y, x]), [(firm, np.bincount(firm)), (year, np.bincount(year))])
    y_d, x_d = demeaned[:, 0], demeaned[:, 1:]
    xtx_inv = np.linalg.inv(x_d.T @ x_d)
    coef = xtx_inv @ x_d.T @ y_d
    resid = y_d - x_d @ coef
    n, n_firm, k = len(y_d), firm.max() + 1, x_d.shape[1]
    n_date = date.max() + 1
    xe = x_d * resid[:, None]
    s_firm = np.column_stack([np.bincount(firm, weights=xe[:, j], minlength=n_firm) for j in range(k)])
    s_date = np.column_stack([np.bincount(date, weights=xe[:, j], minlength=n_date) for j in range(k)])
    adj = n_firm / (n_firm - 1) * (n - 1) / (n - k - (year.max() + 1))
    v_firm = xtx_inv @ (s_firm.T @ s_firm) @ xtx_inv * adj
    v_two = xtx_inv @ (s_firm.T @ s_firm + s_date.T @ s_date - xe.T @ xe) @ xtx_inv * adj
    diag_two = np.diag(v_two)
    se_two = np.where(diag_two > 0, np.sqrt(np.abs(diag_two)), np.nan)
    return {'coef': coef, 'se_firm': np.sqrt(np.diag(v_firm)), 'se_two': se_two, 'n': n, 'n_firms': int(n_firm),
            'r2': 1 - (resid @ resid) / (y_d @ y_d), 'x_d': x_d, 'keep': keep, 'firm_sample': firm}


def vif_max(x_d):
    """去 FE 后各回归量对其余回归量的 VIF 最大值；只有一列时为 1。"""
    if x_d.shape[1] < 2:
        return 1.0
    corr = np.corrcoef(x_d, rowvar=False)
    return float(np.max(np.diag(np.linalg.inv(corr))))


def load_data():
    """合并 event_panel、variables、n_sentences；返回 (字典{列名: 全样本缩尾后的数组}, 原始列, 元信息)。"""
    panel = read_csv(EVENT_DIR / 'event_panel.csv')
    var_rows = read_csv(EVENT_DIR / 'hypotheses' / 'variables.csv')
    assert len(panel) == len(var_rows), '行数不一致'
    for p, v in zip(panel, var_rows):
        assert p['code'] == v['code'] and p['year'] == v['year'], 'variables 与 event_panel 不一一对应'
    sent = {(r['code'], r['year']): int(r['n_sentences']) for r in read_csv(ROOT / 'output' / 'sentiment_data' / 'cs_sentiment.csv')}
    raw = {}
    for name in panel[0]:
        if name not in ('code', 'company'):
            raw[name] = np.array([float(r[name]) if r[name] != '' else np.nan for r in panel])
    for name in var_rows[0]:
        if name not in ('code', 'year'):
            raw[name] = np.array([float(r[name]) if r[name] != '' else np.nan for r in var_rows])
    n_sent = np.array([sent.get((r['code'], r['year']), np.nan) for r in panel], dtype=float)
    raw['n_sentences'] = n_sent
    with np.errstate(invalid='ignore', divide='ignore'):
        raw['ln_nsent'] = np.log(n_sent)
    data, dummies = {}, []
    for name, x in raw.items():
        if name in ('code', 'year', 'announce_date', 'event_date', 'beta', 'n_sentences'):
            data[name] = x
            continue
        vals = np.unique(x[~np.isnan(x)])
        if set(vals) <= {0.0, 1.0}:  # 哑变量不缩尾
            data[name] = x
            dummies.append(name)
        else:
            data[name] = winsorize(x)
    code = np.array([r['code'] for r in panel])
    meta = {'firm': np.unique(code, return_inverse=True)[1], 'date': np.unique(raw['event_date'], return_inverse=True)[1],
            'year': raw['year'].astype(int), 'code': code, 'dummies': dummies, 'n_events': len(panel)}
    return data, raw, meta


def run_one(data, meta, outcome, terms, mask=None):
    """回归 outcome ~ terms + FE；mask 为额外的样本限制（布尔数组）。返回 fe_ols2 的结果和样本信息。"""
    matrix = np.column_stack([data[outcome]] + [data[t] for t in terms])
    ok = ~np.isnan(matrix).any(axis=1)
    if mask is not None:
        ok &= mask
    res = fe_ols2(matrix[ok, 0], matrix[ok, 1:], meta['firm'][ok], meta['year'][ok], meta['date'][ok])
    res['idx'] = np.flatnonzero(ok)[res['keep']]
    res['terms'] = terms
    return res


def build_terms(h):
    terms = []
    for t in [h['focal_term']] + h['extra_terms'] + CONTROLS:
        if t not in terms:
            terms.append(t)
    return terms


def primary_window(h):
    found = re.findall(r'\[0,\s*(\d+)\]', h['expected_window'])
    return int(found[0]), len(found) > 1 or int(found[0]) not in HORIZONS


def bh_q(p):
    """Benjamini-Hochberg q 值；缺失 p 值按 1 计入检验数。"""
    p = np.where(np.isnan(p), 1.0, p)
    m = len(p)
    order = np.argsort(p)
    q = np.empty(m)
    running = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        running = min(running, p[i] * m / rank)
        q[i] = running
    return q


def calibrate(data, meta):
    """规格校准：复现 regression_results.csv 里 car_15 的 sent_change_mean 和 sent_std_orig 结果。"""
    for sent, coef0, t0 in [('sent_change_mean', 0.011494, 0.73), ('sent_std_orig', 0.047406, 3.53)]:
        r = run_one(data, meta, 'car_15', [sent] + CONTROLS)
        t = r['coef'][0] / r['se_firm'][0]
        print(f'校准 car_15 ~ {sent}: coef={r["coef"][0]:.6f} (目标 {coef0}), t_firm={t:.3f} (目标 {t0}), n={r["n"]}')
        assert abs(r['coef'][0] - coef0) < 5e-7 and abs(t - t0) < 0.006, '规格校准失败'


def main(out_dir=EVENT_DIR / 'hypotheses'):
    out_dir = Path(out_dir)
    hyps = json.loads((out_dir / 'hypotheses.json').read_text(encoding='utf-8'))
    data, raw, meta = load_data()
    print(f'事件数 {meta["n_events"]}；哑变量（不缩尾）：{meta["dummies"]}')
    calibrate(data, meta)

    all_rows, controls_rows, primary = [], [], []
    for h in hyps:
        terms = build_terms(h)
        main_k, ambiguous = primary_window(h)
        if ambiguous:
            print(f'注意：{h["id"]} 的 expected_window 无法唯一解析：{h["expected_window"]}，取 k={main_k}')
        for k in HORIZONS:
            r = run_one(data, meta, f'car_{k}', terms)
            focal = h['focal_term']
            sample = r['idx']
            f_vals = data[focal][sample]
            sd_focal = float(f_vals.std(ddof=1))
            f_firm = f_vals - (np.bincount(r['firm_sample'], weights=f_vals) / np.bincount(r['firm_sample']))[r['firm_sample']]
            within_sd = float(f_firm.std(ddof=1))
            b, se_f, se_t = r['coef'][0], r['se_firm'][0], r['se_two'][0]
            sign = h['expected_sign']
            row = {'id': h['id'], 'name': h['name'], 'name_cn': h['name_cn'], 'family': h['family'], 'focal_term': focal,
                   'expected_sign': sign, 'outcome': f'car_{k}', 'k': k, 'is_primary': int(k == main_k),
                   'coef': b, 'se_firm': se_f, 't_firm': b / se_f, 'p_firm': pvalue(b / se_f),
                   'se_twoway': se_t, 't_twoway': b / se_t, 'p_twoway': pvalue(b / se_t),
                   'n': r['n'], 'n_firms': r['n_firms'], 'r2_within': r['r2'], 'sd_focal': sd_focal,
                   'within_sd_focal': within_sd, 'effect_1sd': b * sd_focal,
                   'sign_matches': '' if sign not in ('+', '-') else int((b > 0) == (sign == '+')),
                   'max_vif': vif_max(r['x_d']), 'n_terms': len(terms)}
            all_rows.append(row)
            if k == main_k:
                for j, term in enumerate(terms):
                    t2 = r['coef'][j] / r['se_two'][j]
                    controls_rows.append({'id': h['id'], 'k': k, 'term': term, 'is_focal': int(j == 0), 'coef': r['coef'][j],
                                          'se_firm': r['se_firm'][j], 't_firm': r['coef'][j] / r['se_firm'][j],
                                          'se_twoway': r['se_two'][j], 't_twoway': t2, 'p_twoway': pvalue(t2), 'n': r['n']})
                prow = dict(row)
                # 稳健性：只对主窗口
                year = meta['year']
                n_sent = raw['n_sentences']
                with np.errstate(invalid='ignore'):
                    r4 = (n_sent >= 20) & ~((meta['code'] == '600030') & (year == 2012))
                specs = {'r1': (terms + ['ln_nsent'], None), 'r2': (terms, year <= 2020), 'r3': (terms, year >= 2021),
                         'r4': (terms, r4)}
                for key, (tt, mask) in specs.items():
                    rr = run_one(data, meta, f'car_{k}', tt, mask)
                    prow[f'{key}_coef'] = rr['coef'][0]
                    prow[f'{key}_t'] = rr['coef'][0] / rr['se_two'][0]
                    prow[f'{key}_n'] = rr['n']
                primary.append(prow)

    q144 = bh_q(np.array([r['p_twoway'] for r in all_rows]))
    for r, q in zip(all_rows, q144):
        r['bh_q_all144'] = q
    prim_p = np.array([r['p_twoway'] for r in primary])
    q24 = bh_q(prim_p)
    for r, q, p in zip(primary, q24, prim_p):
        r['bh_q_primary'] = q
        r['bonf_p_primary'] = min(1.0, (1.0 if np.isnan(p) else p) * len(primary))
        r['bh_q_all144'] = next(a['bh_q_all144'] for a in all_rows if a['id'] == r['id'] and a['k'] == r['k'])

    fields = ['id', 'name', 'name_cn', 'family', 'focal_term', 'expected_sign', 'outcome', 'k', 'is_primary', 'coef',
              'se_firm', 't_firm', 'p_firm', 'se_twoway', 't_twoway', 'p_twoway', 'bh_q_all144', 'n', 'n_firms',
              'r2_within', 'sd_focal', 'within_sd_focal', 'effect_1sd', 'sign_matches', 'max_vif', 'n_terms']
    write_csv(out_dir / 'regression_all.csv', all_rows, fields)
    write_csv(out_dir / 'primary_tests.csv', primary,
              fields + ['bh_q_primary', 'bonf_p_primary'] + [f'{k}_{s}' for k in ROBUST for s in ('coef', 't', 'n')])
    write_csv(out_dir / 'regression_controls.csv', controls_rows,
              ['id', 'k', 'term', 'is_focal', 'coef', 'se_firm', 't_firm', 'se_twoway', 't_twoway', 'p_twoway', 'n'])
    print(f'结果保存在 {out_dir}')


if __name__ == '__main__':
    main()
