"""年报事件研究：计算年报发布后的 CAR，并回归到年报文本情感指标上。

样本：cs_sentiment.csv 中有情感评分的公司*年份（year 为年报所属年份，对应 report_date = year1231）。
事件日 t0：financial_data 中公告日 date 当天或之后的第一个交易日。
收益率：daily_stock_data 的 close 已是后复权价，r_t = close_t / close_{t-1} - 1；
        当天或前一天停牌（或无行情）时 r_t 记为缺失，所以复牌首日的多日累计收益不会混入。
市场回报：全市场股票按前一日总市值加权的日收益率（数据中没有指数）。
异常回报：市场模型 r = alpha + beta * rm，在 [t0-250, t0-21] 上估计，至少 120 个有效交易日；AR = r - alpha - beta * rm。
CAR_k：AR 在 [t0, t0+k] 上的累计和，窗口内有任一天收益率缺失则该窗口缺失。
控制变量（都用年报当期 year1231 的数据）：
  - ln_assets = ln(资产总额)；leverage = 负债总额 / 资产总额
  - roe = 净利润 / (资产总额 - 负债总额)，净资产不为正时缺失
  - sue = (Q4 归母净利润 - 上年 Q4) / 此前 8 个季度同比差值的标准差（季节性随机游走，至少 4 个季度）
回归：CAR_k = b * 情感指标 + 控制变量 + 公司固定效应 + 年份固定效应，每个情感指标、每个窗口单独回归；
      连续变量在全样本 1%/99% 缩尾，标准误按公司聚类，p 值用正态近似。
输出（都在 out_dir 中）：
  - event_panel.csv：每个事件一行，含未缩尾的 CAR、情感指标、控制变量
  - regression_results.csv：所有回归的系数（长表）

用法：在 main.py 中设置 TASK = 'event_study'
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

HORIZONS = [3, 5, 7, 10, 15, 20]
EST_START, EST_END = -250, -21  # 市场模型估计窗口（相对事件日，含两端）
MIN_EST_OBS = 120
SUE_WINDOW, SUE_MIN_OBS = 8, 4
WINSOR = 0.01
SENTIMENTS = ['sent_mean_orig', 'sent_mean_neutral', 'sent_change_mean', 'sent_std_orig', 'pos_change_mean',
              'sent_std_neutral']
CONTROLS = ['ln_assets', 'roe', 'leverage', 'sue']
QUARTER_END = {'0331': 0, '0630': 1, '0930': 2, '1231': 3}


def load_returns(daily_path):
    """返回 (交易日数组, {股票代码: 列号}, 个股收益率矩阵[日, 股], 市值加权市场收益率)。"""
    t = pq.read_table(daily_path, columns=['date', 'code', 'close', 'is_suspended', 'mkcap'])
    days, day_idx = np.unique(pc.cast(t['date'], 'int32').to_numpy(), return_inverse=True)
    codes = t['code'].combine_chunks().dictionary_encode()
    stock_idx = codes.indices.to_numpy(zero_copy_only=False)
    code_col = {c: i for i, c in enumerate(codes.dictionary.to_pylist())}

    close = t['close'].to_numpy(zero_copy_only=False).astype(float)
    trading = ~t['is_suspended'].to_numpy(zero_copy_only=False).astype(bool) & (close > 0)
    price = np.full((len(days), len(code_col)), np.nan)
    price[day_idx[trading], stock_idx[trading]] = close[trading]
    cap = np.full_like(price, np.nan)
    cap[day_idx, stock_idx] = t['mkcap'].to_numpy(zero_copy_only=False).astype(float)

    ret = np.full_like(price, np.nan)
    ret[1:] = price[1:] / price[:-1] - 1
    weight = np.full_like(price, np.nan)
    weight[1:] = cap[:-1]
    weight[np.isnan(ret)] = np.nan
    with np.errstate(invalid='ignore', divide='ignore'):
        market = np.nansum(weight * ret, axis=1) / np.nansum(weight, axis=1)
    return days, code_col, ret, market


def load_fundamentals(fin_path):
    """返回 {(股票代码, 年份): 公告日和控制变量}，只含年报（Q4）。"""
    columns = ['code', 'report_date', 'date', 'total_assets', 'total_liability', 'net_profit_ttm',
               'np_parent_company_owners_q']
    rows = pq.read_table(fin_path, columns=columns).to_pylist()
    profit = {(r['code'], int(r['report_date'][:4]) * 4 + QUARTER_END[r['report_date'][4:]]):
              r['np_parent_company_owners_q'] for r in rows if r['report_date'][4:] in QUARTER_END}

    def seasonal_diff(code, q):
        now, last = profit.get((code, q)), profit.get((code, q - 4))
        return None if now is None or last is None else now - last

    result = {}
    for r in rows:
        if r['report_date'][4:] != '1231':
            continue
        code, year, assets, debt = r['code'], int(r['report_date'][:4]), r['total_assets'], r['total_liability']
        equity = None if assets is None or debt is None else assets - debt
        q = year * 4 + 3
        ue = seasonal_diff(code, q)
        past = [d for d in (seasonal_diff(code, q - k) for k in range(1, SUE_WINDOW + 1)) if d is not None]
        sd = float(np.std(past, ddof=1)) if len(past) >= SUE_MIN_OBS else 0.0
        result[code, year] = {
            'announce_date': int(r['date']),
            'ln_assets': math.log(assets) if assets and assets > 0 else None,
            'leverage': debt / assets if equity is not None and assets > 0 else None,
            'roe': r['net_profit_ttm'] / equity if equity and equity > 0 and r['net_profit_ttm'] is not None else None,
            'sue': ue / sd if ue is not None and sd > 0 else None,
        }
    return result


def market_model(r, market, t0):
    """在估计窗口上拟合市场模型，返回 (alpha, beta)；数据不足时返回 None。"""
    if t0 + EST_START < 0:
        return None
    est_r, est_m = r[t0 + EST_START:t0 + EST_END + 1], market[t0 + EST_START:t0 + EST_END + 1]
    ok = ~np.isnan(est_r) & ~np.isnan(est_m)
    if ok.sum() < MIN_EST_OBS or est_m[ok].var() == 0:
        return None
    beta = np.cov(est_m[ok], est_r[ok], ddof=0)[0, 1] / est_m[ok].var()
    return est_r[ok].mean() - beta * est_m[ok].mean(), beta


def event_cars(r, market, t0):
    """返回一个事件的市场模型 CAR 字典和 beta；估计窗口数据不足时返回 None。"""
    fit = market_model(r, market, t0)
    if fit is None:
        return None
    alpha, beta = fit
    cars = {}
    for k in HORIZONS:
        window = r[t0:t0 + k + 1] - alpha - beta * market[t0:t0 + k + 1]
        cars[f'car_{k}'] = float(window.sum()) if len(window) == k + 1 and not np.isnan(window).any() else None
    return {'beta': float(beta), **cars}


def build_events(panel_path, fundamentals, days, code_col, ret, market):
    """把情感面板、公告日、控制变量和 CAR 合成事件面板，并统计各环节损失的样本。"""
    with Path(panel_path).open(encoding='utf-8-sig', newline='') as f:
        panel = list(csv.DictReader(f))
    events, lost = [], defaultdict(int)
    for row in panel:
        code, year = row['code'], int(row['year'])
        fund = fundamentals.get((code, year))
        if fund is None:
            lost['财务数据中没有该年报'] += 1
            continue
        if code not in code_col:
            lost['行情数据中没有该股票'] += 1
            continue
        t0 = int(np.searchsorted(days, fund['announce_date']))
        if t0 >= len(days):
            lost['公告日晚于行情数据'] += 1
            continue
        cars = event_cars(ret[:, code_col[code]], market, t0)
        if cars is None:
            lost['估计窗口有效交易日不足'] += 1
            continue
        events.append({'code': code, 'year': year, 'company': row['company'], 'announce_date': fund['announce_date'],
                       'event_date': int(days[t0]), **cars, **{s: float(row[s]) if row[s] else None for s in SENTIMENTS},  # 只有一句的年报标准差为空
                       **{c: fund[c] for c in CONTROLS}})
    return events, dict(lost)


def winsorize(x):
    """按非缺失值的 1%/99% 分位数缩尾。"""
    low, high = np.nanquantile(x, [WINSOR, 1 - WINSOR])
    return np.clip(x, low, high)


def drop_singletons(*groups):
    """反复剔除只出现一次的固定效应组（它们被完全吸收，不提供信息），返回保留样本的布尔掩码。"""
    keep = np.ones(len(groups[0]), dtype=bool)
    while True:
        before = keep.sum()
        for g in groups:
            keep &= np.bincount(g, weights=keep.astype(float))[g] > 1
        if keep.sum() == before:
            return keep


def absorb(m, groups, tol=1e-10, max_iter=5000):
    """交替去均值，去掉 m 各列在多组固定效应上的部分。"""
    m = m.copy()
    for _ in range(max_iter):
        prev = m.copy()
        for g, counts in groups:
            for j in range(m.shape[1]):
                m[:, j] -= (np.bincount(g, weights=m[:, j]) / counts)[g]
        if np.abs(m - prev).max() < tol:
            return m
    raise RuntimeError('固定效应去均值未收敛')


def fe_ols(y, x, firm, year):
    """双向固定效应 OLS，标准误按公司聚类。返回 (系数, 标准误, n, 公司数, within R2)。"""
    keep = drop_singletons(firm, year)
    y, x = y[keep], x[keep]
    firm = np.unique(firm[keep], return_inverse=True)[1]
    year = np.unique(year[keep], return_inverse=True)[1]
    groups = [(firm, np.bincount(firm)), (year, np.bincount(year))]
    demeaned = absorb(np.column_stack([y, x]), groups)
    y_d, x_d = demeaned[:, 0], demeaned[:, 1:]
    xtx_inv = np.linalg.inv(x_d.T @ x_d)
    coef = xtx_inv @ x_d.T @ y_d
    resid = y_d - x_d @ coef
    n, n_firm, k = len(y_d), firm.max() + 1, x_d.shape[1]
    scores = np.column_stack([np.bincount(firm, weights=x_d[:, j] * resid, minlength=n_firm) for j in range(k)])
    # 公司固定效应嵌套在聚类内，不计入自由度；年份固定效应计入（含截距）
    adj = n_firm / (n_firm - 1) * (n - 1) / (n - k - (year.max() + 1))
    vcov = xtx_inv @ (scores.T @ scores) @ xtx_inv * adj
    return coef, np.sqrt(np.diag(vcov)), n, n_firm, 1 - (resid @ resid) / (y_d @ y_d)


def run_regressions(events):
    """跑全部 情感指标*窗口 的回归，返回长表。"""
    def column(name):
        return np.array([np.nan if e[name] is None else e[name] for e in events], dtype=float)

    data = {name: winsorize(column(name)) for name in SENTIMENTS + CONTROLS + [f'car_{k}' for k in HORIZONS]}
    firm = np.unique([e['code'] for e in events], return_inverse=True)[1]
    year = np.array([e['year'] for e in events])
    rows = []
    for k in HORIZONS:
        for sentiment in SENTIMENTS:
            terms = [sentiment] + CONTROLS
            matrix = np.column_stack([data[f'car_{k}']] + [data[t] for t in terms])
            ok = ~np.isnan(matrix).any(axis=1)
            coef, se, n, n_firm, r2 = fe_ols(matrix[ok, 0], matrix[ok, 1:], firm[ok], year[ok])
            for term, b, s in zip(terms, coef, se):
                t = b / s
                rows.append({'outcome': f'car_{k}', 'sentiment': sentiment, 'term': term, 'coef': round(float(b), 6),
                             'se': round(float(s), 6), 't': round(float(t), 3),
                             'p': round(math.erfc(abs(t) / math.sqrt(2)), 4), 'n': n, 'n_firms': int(n_firm),
                             'r2_within': round(float(r2), 4)})
    return rows


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig：Excel 打开不乱码
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main(panel_path, fin_path, daily_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    days, code_col, ret, market = load_returns(daily_path)
    events, lost = build_events(panel_path, load_fundamentals(fin_path), days, code_col, ret, market)
    print(f'事件数 {len(events)}；样本损失：{lost}')
    event_fields = ['code', 'year', 'company', 'announce_date', 'event_date', 'beta'] + [f'car_{k}' for k in HORIZONS] \
        + SENTIMENTS + CONTROLS
    write_csv(out_dir / 'event_panel.csv', events, event_fields)
    results = run_regressions(events)
    write_csv(out_dir / 'regression_results.csv', results,
              ['outcome', 'sentiment', 'term', 'coef', 'se', 't', 'p', 'n', 'n_firms', 'r2_within'])
    for r in results:
        if r['term'] == r['sentiment']:
            stars = '***' if r['p'] < 0.01 else '**' if r['p'] < 0.05 else '*' if r['p'] < 0.1 else ''
            print(f"{r['outcome']:<7}{r['sentiment']:<19}{r['coef']:>10.4f}{stars:<4}t={r['t']:>7.2f}  n={r['n']}")
    print(f'结果保存在 {out_dir}')
