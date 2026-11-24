"""实验：券商研报的概念 DAG 结构指标与研报发布后的市场反应（事件研究）。由根目录 experiment_report_car.py 调用。
复用 src/dag_outcomes_vars.py 的行情、财务读取函数和 src/dag_outcomes_reg.py 的 fit（OLS + 固定效应 + 聚类）。

口径要点（规格没写到的细节在这里定）：
  事件日 0     发布日当天，非交易日取之后第一个交易日；交易日序列 = 日行情里沪深 A 股出现过的全部日期
  停牌         某日在行情表里没有这只股票的未停牌记录（is_suspended、无收盘价或无市值）即算停牌；第 0 天停牌的研报剔除
  个股日收益   相对上一个未停牌收盘（daily_returns）；窗口内停牌日个股收益按 0，超额收益 = 0 − 基准收益；
               复牌日的收益包含停牌期间的累计涨跌。上市前的日子（早于该股在行情表里第一个未停牌日）记缺失，窗口内有缺失则 CAR 缺失
  市场收益     沪深 A 股、前一交易日也有交易的股票按前一日总市值加权（daily_returns）
  行业收益     申万一级行业同样加权，剔除本股（与已有流程 ex_industry 的做法一致）；本股当天不在组合里时就是整个行业的收益
  换手率       成交量 / 总股本 × 100（%），总股本 = 总市值 / 未复权收盘价（同 stock_day_measures）；只用未停牌日
  PREVOL、PRETURN  第 −60 到 −11 天未停牌日的日收益标准差、日均换手率，不足 MIN_PRE_DAYS 个未停牌日记缺失
  ATURN        第 0–2 天未停牌日的日均换手率 / PRETURN − 1
  EA           事件日与某次定期报告公告日（financial_data.date，非交易日顺延）之间相差 0–5 个交易日（事件日在后）
  NREP         同一只股票、发布日（顺延到交易日）落在事件日 −1 到 +1 个交易日内的其他个股研报（is_stock_report）篇数，取 ln(1+n)
  公司控制     2023 年值：SIZE23 = ln(2023 年末总资产)，LEV23，BM23 = 2023 年末净资产 / 2023 年最后一个未停牌日总市值，ROA23；公司层面缩尾
  缩尾         全部 Y、PREVOL、PRETURN 在研报层面按 1%/99% 缩尾；PRECAR 就是缩尾后的 CAR[-10,-1]；NREP、EA 不缩尾
"""
import time

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src import dag_outcomes_vars as V

# ---------------------------------------------------------------- 参数
DAILY_START, DAILY_END = '20230601', '20250430'  # 读日行情的区间（覆盖 2024 年研报的 −60 到 +20 天）
CAR_WIN = {'CAR_0_1': (0, 1), 'CAR_0_2': (0, 2), 'CAR_0_5': (0, 5), 'CAR_0_10': (0, 10), 'CAR_0_20': (0, 20),
           'CAR_m10_m1': (-10, -1), 'CAR_m5_m1': (-5, -1)}
IND_WIN = {'CARI_0_2': (0, 2), 'CARI_0_5': (0, 5)}  # 行业调整
ABS_OF = {'ABSCAR_0_1': 'CAR_0_1', 'ABSCAR_0_2': 'CAR_0_2', 'ABSCAR_0_5': 'CAR_0_5', 'ABSCARI_0_2': 'CARI_0_2'}
PRE_WIN = (-60, -11)  # PREVOL、PRETURN 和 ATURN 分母的窗口
TURN_WIN = (0, 2)  # ATURN 的窗口
MIN_PRE_DAYS = 20  # 事件前窗口至少这么多个未停牌日
EA_DAYS = 5  # 定期报告公告后 0–EA_DAYS 个交易日
NREP_DAYS = 1  # 同股其他研报：事件日 ±NREP_DAYS 个交易日

DAG_X = ['RATIO', 'LNPATH', 'PATHLEN']
Y_MAIN = ['CAR_0_1', 'CAR_0_2', 'CAR_0_5', 'CAR_0_10', 'CAR_0_20', 'ABSCAR_0_1', 'ABSCAR_0_2', 'ABSCAR_0_5', 'ATURN_0_2']
Y_PLACEBO = ['CAR_m10_m1', 'CAR_m5_m1']
Y_IND = ['CARI_0_2', 'CARI_0_5', 'ABSCARI_0_2']
Y_ALL = Y_MAIN + Y_PLACEBO + Y_IND
Y_LABEL = {'CAR_0_1': 'CAR[0,1]', 'CAR_0_2': 'CAR[0,2]', 'CAR_0_5': 'CAR[0,5]', 'CAR_0_10': 'CAR[0,10]', 'CAR_0_20': 'CAR[0,20]',
           'ABSCAR_0_1': '|CAR[0,1]|', 'ABSCAR_0_2': '|CAR[0,2]|', 'ABSCAR_0_5': '|CAR[0,5]|', 'ATURN_0_2': 'ATURN[0,2]',
           'CAR_m10_m1': 'CAR[-10,-1]', 'CAR_m5_m1': 'CAR[-5,-1]',
           'CARI_0_2': 'CAR_IND[0,2]', 'CARI_0_5': 'CAR_IND[0,5]', 'ABSCARI_0_2': '|CAR_IND[0,2]|'}
CTRL_REPORT = ['LNCHARS_R', 'LNNODES_R']
CTRL_PRE = ['PRECAR', 'PREVOL', 'PRETURN']
CTRL_INFO = ['EA', 'NREP']
CTRL_FIRM = ['SIZE23', 'LEV23', 'BM23', 'ROA23']
SPECS = [  # (口径, 说明, 固定效应, 聚类, 控制变量, 样本筛选)
    ('A', '申万一级+券商+月份固定效应，全部控制', ['sw', 'broker_g', 'month'], 'code', CTRL_REPORT + CTRL_PRE + CTRL_INFO + CTRL_FIRM, None),
    ('B', '公司+券商+月份固定效应（重点）', ['code', 'broker_g', 'month'], 'code', CTRL_REPORT + CTRL_PRE + CTRL_INFO, None),
    ('B_twoway', '口径 B，按公司和事件日双向聚类', ['code', 'broker_g', 'month'], ['code', 'day0'], CTRL_REPORT + CTRL_PRE + CTRL_INFO, None),
    ('B_noEA', '口径 B，剔除 EA=1', ['code', 'broker_g', 'month'], 'code', CTRL_REPORT + CTRL_PRE + ['NREP'], 'noEA'),
    ('B_single', '口径 B，只留该公司该事件日只有 1 篇样本研报的', ['code', 'broker_g', 'month'], 'code', CTRL_REPORT + CTRL_PRE + CTRL_INFO,
     'single'),
]


# ---------------------------------------------------------------- 行情矩阵
def market_matrices(daily_path, codes, sw_map):
    """返回 cal（交易日列表）、市场日收益、样本股票 × 交易日的矩阵（收益、是否交易、是否已上市、换手率、行业剔除本股的收益）。"""
    d = V.load_daily(daily_path, DAILY_START, DAILY_END)
    t, mkt = V.daily_returns(d)
    cal = list(mkt.index)
    # 申万一级行业日收益：连续两天有交易的股票按前一日总市值加权，剔除本股
    c = t[t.consec].copy()
    c['g'] = c.code.map(sw_map)
    c = c[c.g.notna()]
    c['wr'] = c.w * c.ret
    ind_sum = c.groupby(['g', 'date'])[['wr', 'w']].sum()
    s = t[t.code.isin(codes)].copy()
    s['turn'] = s.volume * s.close / (s.adj_factor * s.mkcap) * 100
    piv = lambda col: s.pivot(index='code', columns='date', values=col).reindex(index=codes, columns=cal)  # noqa: E731
    ret, turn = piv('ret'), piv('turn')
    traded = s.assign(one=1.0).pivot(index='code', columns='date', values='one').reindex(index=codes, columns=cal).notna()
    first = s.groupby('code').date.min().reindex(codes)
    listed = pd.DataFrame(np.array(cal)[None, :] >= first.to_numpy()[:, None], index=codes, columns=cal)
    own_c = s[s.consec].assign(wr=lambda x: x.w * x.ret)
    own_wr = own_c.pivot(index='code', columns='date', values='wr').reindex(index=codes, columns=cal).fillna(0.0)
    own_w = own_c.pivot(index='code', columns='date', values='w').reindex(index=codes, columns=cal).fillna(0.0)
    ind = pd.DataFrame(np.nan, index=codes, columns=cal)
    for code in codes:
        g = sw_map.get(code)
        if g is None or g not in ind_sum.index.get_level_values(0):
            continue
        x = ind_sum.loc[g].reindex(cal).fillna(0.0)
        den = x.w - own_w.loc[code]
        ind.loc[code] = ((x.wr - own_wr.loc[code]) / den).where(den > 0)
    return d, t, cal, mkt, {'ret': ret, 'traded': traded, 'listed': listed, 'turn': turn, 'ind': ind}


def window_car(code, p0, lo, hi, M, bench):
    """CAR[lo,hi]：停牌日个股收益按 0；返回 (CAR, 窗口内停牌天数)，有缺失时 CAR 为 nan。bench：基准日收益（Series[date] 或 行业矩阵的一行）。"""
    if p0 + lo < 0 or p0 + hi >= M['ret'].shape[1]:
        return np.nan, 0
    sl = slice(p0 + lo, p0 + hi + 1)
    tr = M['traded'].loc[code].to_numpy()[sl]
    if not M['listed'].loc[code].to_numpy()[sl].all():
        return np.nan, 0
    r = np.where(tr, M['ret'].loc[code].to_numpy()[sl], 0.0)
    b = bench[sl]
    if np.isnan(r).any() or np.isnan(b).any():
        return np.nan, int((~tr).sum())
    return float((r - b).sum()), int((~tr).sum())


def build_panel(rep, daily_path, fin_path, meta_path, industry_path):
    """研报面板 -> 事件研究面板（未缩尾）和筛选信息。"""
    t0 = time.time()
    info = {'n0': len(rep), 'firms0': rep.code.nunique()}
    codes = sorted(rep.code.unique())
    ind = V.load_industry(industry_path)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in rep.groupby('code').sw.first().items() if c not in sw_map})
    d, t, cal, mkt, M = market_matrices(daily_path, codes, sw_map)
    print(f'行情：{len(d)} 行、{len(cal)} 个交易日（{time.time() - t0:.0f} 秒）', flush=True)
    cal_arr = np.array(cal)
    pos_of = lambda dates: np.searchsorted(cal_arr, np.asarray(dates, dtype=str), side='left')  # noqa: E731  第一个 ≥ 该日的交易日

    p = rep[['report_id', 'code', 'company', 'date', 'month', 'broker', 'broker_g', 'sw', 'csrc', 'fin'] + DAG_X + CTRL_REPORT].copy()
    p['p0'] = pos_of(p.date)
    p['day0'] = [cal[i] if i < len(cal) else None for i in p.p0]
    info['n_shifted'] = int((p.day0 != p.date).sum())
    p['susp0'] = [not M['traded'].loc[c].iat[i] for c, i in zip(p.code, p.p0)]
    info['n_susp0'] = int(p.susp0.sum())
    mk = mkt.to_numpy()
    rows = []
    for r in p.itertuples():
        rec = {'report_id': r.report_id}
        susp = {}
        for name, (lo, hi) in CAR_WIN.items():
            rec[name], susp[name] = window_car(r.code, r.p0, lo, hi, M, mk)
        for name, (lo, hi) in IND_WIN.items():
            rec[name], susp[name] = window_car(r.code, r.p0, lo, hi, M, M['ind'].loc[r.code].to_numpy())
        rec.update({f'susp_{k}': v for k, v in susp.items()})
        pre = slice(r.p0 + PRE_WIN[0], r.p0 + PRE_WIN[1] + 1)
        tr = M['traded'].loc[r.code].to_numpy()[pre]
        rr, tu = M['ret'].loc[r.code].to_numpy()[pre][tr], M['turn'].loc[r.code].to_numpy()[pre][tr]
        rr = rr[~np.isnan(rr)]
        rec['n_pre_days'] = int(tr.sum())
        rec['PREVOL'] = float(np.std(rr, ddof=1)) if len(rr) >= MIN_PRE_DAYS else np.nan
        rec['PRETURN'] = float(np.mean(tu)) if tr.sum() >= MIN_PRE_DAYS else np.nan
        w = slice(r.p0 + TURN_WIN[0], r.p0 + TURN_WIN[1] + 1)
        tw = M['traded'].loc[r.code].to_numpy()[w]
        rec['susp_ATURN'] = int((~tw).sum())
        rec['ATURN_0_2'] = float(np.mean(M['turn'].loc[r.code].to_numpy()[w][tw])) / rec['PRETURN'] - 1 if tw.any() else np.nan
        rows.append(rec)
    p = p.merge(pd.DataFrame(rows), on='report_id')
    for a, c in ABS_OF.items():
        p[a] = p[c].abs()
    print(f'CAR 和事件前变量（{time.time() - t0:.0f} 秒）', flush=True)

    # EA：定期报告公告日（四个季度）
    fin = pd.read_parquet(fin_path, columns=['code', 'report_date', 'date'])
    fin = fin[fin.code.isin(codes) & fin.date.notna()].drop_duplicates(['code', 'report_date'], keep='last')
    fin['pa'] = pos_of(fin.date)
    ann = fin.groupby('code').pa.apply(np.array).to_dict()
    p['EA'] = [float(np.any((p0 - ann.get(c, np.array([10 ** 9])) >= 0) & (p0 - ann.get(c, np.array([10 ** 9])) <= EA_DAYS)))
               for c, p0 in zip(p.code, p.p0)]
    # NREP：同股 ±1 个交易日内的其他个股研报
    meta = pd.read_parquet(meta_path, columns=['report_id', 'stock_code', 'date', 'is_stock_report'])
    meta = meta[meta.is_stock_report & meta.stock_code.notna()].copy()
    meta['code'] = meta.stock_code.str[:6]
    meta = meta[meta.code.isin(codes)]
    meta['pm'] = pos_of(meta.date)
    by = {c: g for c, g in meta.groupby('code')}
    n_other = []
    for r in p.itertuples():
        g = by.get(r.code)
        n_other.append(0 if g is None else int(((g.pm - r.p0).abs() <= NREP_DAYS).sum() - (g.report_id == r.report_id).any()))
    p['N_OTHER'] = n_other
    p['NREP'] = np.log1p(p.N_OTHER)
    p['n_same_day0'] = p.groupby(['code', 'day0']).report_id.transform('size')  # 同一公司同一事件日的样本研报篇数

    # 公司层面 2023 年控制变量（公司层面缩尾后并入）
    f, _ = V.load_fin(fin_path, codes)
    fin_codes = set(rep.code[rep.fin.astype(bool)])
    a = V.annual_fin(f, codes, 2023, fin_codes).set_index('code')
    mk23 = V.last_traded_value(t, codes, '20231231', 'mkcap')
    firm = pd.DataFrame({'SIZE23': a.SIZE, 'LEV23': a.LEV, 'BM23': a.equity / mk23.reindex(a.index), 'ROA23': a.ROA})
    firm_w = firm.apply(R.winsor)
    p = p.merge(firm.add_suffix('_raw'), left_on='code', right_index=True, how='left')
    p = p.merge(firm_w, left_on='code', right_index=True, how='left')
    return p, info, (cal, mkt, M)


def winsorize(p):
    """全部 Y、PREVOL、PRETURN 在研报层面（剔除第 0 天停牌之后的样本）缩尾；PRECAR = 缩尾后的 CAR[-10,-1]。"""
    w = p.copy()
    for c in Y_ALL + ['PREVOL', 'PRETURN']:
        w[c] = R.winsor(w[c])
    w['PRECAR'] = w['CAR_m10_m1']
    return w


# ---------------------------------------------------------------- 回归
def spec_data(w, filt):
    if filt == 'noEA':
        return w[w.EA == 0]
    if filt == 'single':
        return w[w.n_same_day0 == 1]
    return w


def run_all(w):
    rows = []
    for spec, label, fe, cl, ctrl, filt in SPECS:
        d = spec_data(w, filt)
        for y in Y_ALL:
            c = [x for x in ctrl if not (y in Y_PLACEBO and x == 'PRECAR')]  # 安慰剂回归不加 PRECAR
            for xs, kind in [([x], '单独') for x in DAG_X] + [(DAG_X, '联合')]:
                res, s = R.fit(d, y, xs, c, fe, cl)
                for row in res:
                    rows.append({'spec': spec, 'spec_label': label, 'Y': y, 'Y_label': Y_LABEL[y], 'reg': kind, 'X': row['X'],
                                 'coef': row['coef'], 'se': row['se'], 't': row['t'], 'p': row['p'], 'n': row['n'],
                                 'n_firms': s.code.nunique(), 'r2': row['r2'], 'adj_r2': row['adj_r2'], 'n_clusters': row['n_clusters'],
                                 'n_singletons': row['n_singletons'], 'fe': '+'.join(fe),
                                 'cluster': cl if isinstance(cl, str) else '+'.join(cl), 'controls': '、'.join(c),
                                 'y_mean': row['y_mean'], 'y_sd': row['y_sd']})
    return pd.DataFrame(rows)


def descriptives(w):
    cols = Y_ALL + DAG_X + CTRL_REPORT + CTRL_PRE + CTRL_INFO + ['N_OTHER'] + CTRL_FIRM
    out = []
    for c in cols:
        s = pd.to_numeric(w[c], errors='coerce').dropna()
        q = s.quantile([0, .01, .25, .5, .75, .99, 1]).to_numpy()
        out.append({'var': c, 'label': Y_LABEL.get(c, c), 'n': len(s), 'mean': s.mean(), 'sd': s.std(), 'min': q[0], 'p1': q[1],
                    'p25': q[2], 'p50': q[3], 'p75': q[4], 'p99': q[5], 'max': q[6]})
    return pd.DataFrame(out)


def main(out_dir, report_panel, daily_path, fin_path, meta_path, industry_path):
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    rep = pd.read_parquet(report_panel)
    p, info, _ = build_panel(rep, daily_path, fin_path, meta_path, industry_path)
    print(f'样本：{info["n0"]} 篇、{info["firms0"]} 家；发布日不是交易日顺延的 {info["n_shifted"]} 篇；'
          f'第 0 天停牌剔除 {info["n_susp0"]} 篇', flush=True)
    p = p[~p.susp0].reset_index(drop=True)
    print(f'剔除后：{len(p)} 篇、{p.code.nunique()} 家', flush=True)
    for c in list(CAR_WIN) + list(IND_WIN) + ['ATURN']:
        k = (p[f'susp_{c}'] > 0).sum()
        print(f'  {c}：窗口内有停牌日的研报 {k} 篇，缺失 {p[c if c != "ATURN" else "ATURN_0_2"].isna().sum()} 篇', flush=True)
    for c in ['PREVOL', 'PRETURN'] + CTRL_FIRM:
        print(f'  {c} 缺失 {p[c].isna().sum()} 篇', flush=True)
    print(f'  EA=1：{int(p.EA.sum())} 篇；同一公司同一事件日有多篇样本研报：{int((p.n_same_day0 > 1).sum())} 篇、'
          f'{p[p.n_same_day0 > 1].groupby(["code", "day0"]).ngroups} 组', flush=True)
    w = winsorize(p)
    p.to_parquet(out_dir / 'report_car_panel_raw.parquet', index=False)
    w.to_parquet(out_dir / 'report_car_panel.parquet', index=False)
    descriptives(w).to_csv(out_dir / 'report_car_descriptives.csv', index=False, encoding='utf-8-sig')
    res = run_all(w)
    res.to_csv(out_dir / 'report_car_results.csv', index=False, encoding='utf-8-sig')
    single = res[res.reg == '单独']
    for spec, g in single.groupby('spec', sort=False):
        print(f'  口径 {spec}：单独回归 {len(g)} 个系数，p<0.05 {int((g.p < 0.05).sum())} 个（期望 {len(g) * 0.05:.1f}），'
              f'p<0.10 {int((g.p < 0.10).sum())} 个（期望 {len(g) * 0.10:.1f}）', flush=True)
    print(f'完成：{out_dir}（{time.time() - t0:.0f} 秒）', flush=True)
    return w, res
