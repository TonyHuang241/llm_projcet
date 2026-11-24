"""DAG 结构指标与 2025 年股价、财务指标回归（docs/DAG结构指标与2025年股价财务回归方案.md）的变量构造部分：
读样本、行业、DAG 结构指标、日行情、财务数据，算出公司层面、研报层面、季度面板的全部变量（未缩尾、未标准化）。
回归在 src/dag_outcomes_reg.py，流程和输出在 src/dag_outcomes.py。

口径要点（方案没写到的细节在这里定，汇报里逐条列出）：
  周          按自然周（周一到周日）划分；一周归属的年份 = 这一周最后一个市场交易日所在的年份
              （2024-12-30 那周算 2025 年；2025-12-29 那周 2026 年元旦休市，最后交易日是 12-31，算 2025 年，所以 2025 年有 53 周）
  个股周收益  本周最后一个未停牌交易日收盘 / 上一个有交易的周的最后收盘 − 1（中间隔了停牌周也算一个收益）
  组合周收益  只用沪深 A 股（代码 0、3、6 开头），只用上一周也有交易的股票（连续两周有交易），按上一周最后一个交易日的总市值加权
  组合日收益  同上：只用前一个交易日也有交易的股票，按前一日总市值加权
  行业收益    同行业剔除本股：(行业 Σw·r − 本股 w·r) / (行业 Σw − 本股 w)；本股这一周不在组合里时就是整个行业的收益
  RET         年末最后一个未停牌收盘 / 上年末最后一个未停牌收盘 − 1；要求年内最后一个交易日这只股票还在行情表里（退市的记缺失）
  TURN、ILLIQ 年内未停牌交易日的均值，不足 100 个交易日记缺失；TURN 用百分比
  CAR         窗口内任何一天停牌记缺失；股票日收益相对上一个未停牌收盘
"""
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.viz.plot_dag_structure import measure

FIN_INDUSTRY = '金融业'  # 证监会门类里的金融业（剔除金融业的稳健性、GM 和 RD 只算非金融都按这个）
MANUAL_INDUSTRY = {'600837': ('金融业', '非银金融'), '601989': ('制造业', '国防军工')}  # 行业表里没有的两只（2025 年退市），按方案手工补
MIN_WEEKS = 30  # 一年不足这么多周收益，周收益类指标记缺失（同伊志宏等 2019）
MIN_DAYS = 100  # 一年不足这么多个未停牌交易日，TURN、ILLIQ 记缺失
CAR_WINDOWS = {'1': (-1, 1), '5': (-1, 5)}  # 年报公告窗口（交易日）


# ---------------------------------------------------------------- 行业
def read_xlsx_first_sheet(path):
    """不依赖 openpyxl 读 xlsx 的第一张表：返回 [{列字母: 文本}]，一行一个字典。"""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    tag = lambda t: '{%s}%s' % (ns['m'], t)  # noqa: E731
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns):
            shared.append(''.join(t.text or '' for t in si.iter(tag('t'))))
    sheets = sorted((n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$', n)),
                    key=lambda n: int(re.search(r'(\d+)\.xml$', n).group(1)))
    rows = []
    for row in ET.fromstring(z.read(sheets[0])).iter(tag('row')):
        r = {}
        for c in row.findall('m:c', ns):
            col = re.match(r'[A-Z]+', c.get('r')).group(0)
            v, t = c.find('m:v', ns), c.get('t')
            if t == 's' and v is not None:
                r[col] = shared[int(v.text)]
            elif t == 'inlineStr':
                r[col] = ''.join(x.text or '' for x in c.iter(tag('t')))
            else:
                r[col] = v.text if v is not None else None
        rows.append(r)
    return rows


def load_industry(path):
    """industry.xlsx（Wind 导出：证券代码、证券简称、申万一级行业（2021）、证监会门类行业）-> code、name、sw、csrc。
    只留证券代码形如 000001.SZ 的行（去掉表头和最后的“数据来源”），空的行业记缺失；补上 MANUAL_INDUSTRY。"""
    out = []
    for r in read_xlsx_first_sheet(path):
        code = (r.get('A') or '').strip()
        if re.fullmatch(r'\d{6}\.[A-Z]{2}', code):
            clean = lambda v: (v or '').strip() or None  # noqa: E731
            out.append({'code': code[:6], 'name': clean(r.get('B')), 'sw': clean(r.get('C')), 'csrc': clean(r.get('D'))})
    ind = pd.DataFrame(out).drop_duplicates('code')
    add = [{'code': c, 'name': None, 'csrc': v[0], 'sw': v[1]} for c, v in MANUAL_INDUSTRY.items() if c not in set(ind.code)]
    return pd.concat([ind, pd.DataFrame(add)], ignore_index=True)


# ---------------------------------------------------------------- DAG 结构指标
def load_dag(dag_dirs, codes):
    """各 DAG 输出目录的 reports / edges / unified_edges（按 dag_dirs 的顺序，report_id 重复时用前面的）-> 每篇研报一行：
    原始边（in_dag 为真）上的 ratio、lnpath（ln(1+路径数)）、pathlen、n_nodes，统一边（没有统一结果的用原始边）上的同名 *_u。
    没有进图边的研报指标记缺失。codes 是 6 位代码集合。"""
    frames = []
    for i, d in enumerate(dag_dirs):
        r = pq.read_table(d / 'reports.parquet').to_pandas()
        frames.append(r.assign(src=i))
    reports = pd.concat(frames, ignore_index=True).drop_duplicates('report_id', keep='first')
    reports['code'] = reports.stock_code.str[:6]
    reports = reports[reports.code.isin(codes)].reset_index(drop=True)
    raw, uni = defaultdict(list), defaultdict(list)
    for i, d in enumerate(dag_dirs):
        mine = set(reports.report_id[reports.src == i])
        for e in pq.read_table(d / 'edges.parquet', columns=['report_id', 'cause', 'effect', 'support', 'in_dag']).to_pylist():
            if e['in_dag'] and e['report_id'] in mine:
                raw[e['report_id']].append((e['cause'], e['effect'], e['support']))
        for e in pq.read_table(d / 'unified_edges.parquet', columns=['report_id', 'cause', 'effect', 'support']).to_pylist():
            if e['report_id'] in mine:
                uni[e['report_id']].append((e['cause'], e['effect'], e['support']))
    rows = []
    for rid in reports.report_id:
        rec = {'report_id': rid, 'unified': rid in uni}
        for suf, edges in (('', raw[rid]), ('_u', uni[rid] if rid in uni else raw[rid])):
            _, m = measure(edges)
            ok = m['n_edges'] > 0
            rec.update({'ratio' + suf: m['ratio'] if ok else None,
                        'lnpath' + suf: math.log1p(m['n_paths']) if ok and m['n_paths'] else None,
                        'pathlen' + suf: m['len_mean'] if ok else None,
                        'n_paths' + suf: m['n_paths'], 'n_nodes' + suf: m['n_nodes'] if ok else None,
                        'n_edges' + suf: m['n_edges'], 'n_cycle_removed' + suf: m['n_cycle_removed']})
        rows.append(rec)
    keep = ['report_id', 'code', 'company', 'stock_code', 'date', 'broker', 'n_chars', 'src']
    out = reports[keep].merge(pd.DataFrame(rows), on='report_id')
    for c in ('ratio', 'lnpath', 'pathlen', 'n_nodes', 'ratio_u', 'lnpath_u', 'pathlen_u', 'n_nodes_u'):
        out[c] = pd.to_numeric(out[c], errors='coerce').astype(float)
    return out


def firm_dag(rep):
    """研报层面指标 -> 公司层面：三个指标各篇的均值（主口径，RATIO、LNPATH、PATHLEN）和中位数（*_MED）、统一边口径（*_U），
    LNNODES = ln(各篇进图概念数的均值)、LNNODES_U（统一边的图）、LNCHARS = ln(各篇正文字数的均值)、
    N_DAG = 抽出 DAG（至少一条进图边）的研报篇数、LNNDAG = ln(1 + N_DAG)。"""
    dag = rep[rep.n_edges > 0]
    g = dag.groupby('code')
    out = pd.DataFrame({'RATIO': g.ratio.mean(), 'LNPATH': g.lnpath.mean(), 'PATHLEN': g.pathlen.mean(),
                        'RATIO_MED': g.ratio.median(), 'LNPATH_MED': g.lnpath.median(), 'PATHLEN_MED': g.pathlen.median(),
                        'RATIO_U': g.ratio_u.mean(), 'LNPATH_U': g.lnpath_u.mean(), 'PATHLEN_U': g.pathlen_u.mean(),
                        'LNNODES': np.log(g.n_nodes.mean()), 'LNNODES_U': np.log(g.n_nodes_u.mean()),
                        'LNCHARS': np.log(g.n_chars.mean()), 'N_DAG': g.size()})
    out['LNNDAG'] = np.log1p(out.N_DAG)
    return out.reset_index()


# ---------------------------------------------------------------- 行情：日收益、周收益、组合收益
def load_daily(path, start, end):
    """日行情按日期裁剪，只留沪深 A 股（代码 0、3、6 开头）。traded = 未停牌且有收盘价和总市值。"""
    cols = ['date', 'code', 'close', 'volume', 'money', 'is_suspended', 'trade_day', 'mkcap', 'adj_factor']
    d = pq.read_table(path, columns=cols, filters=[('date', '>=', start), ('date', '<=', end)]).to_pandas()
    d = d[d.code.str[0].isin(['0', '3', '6'])].sort_values(['code', 'date']).reset_index(drop=True)
    d['traded'] = (~d.is_suspended) & (d.close > 0) & d.mkcap.notna() & (d.mkcap > 0)
    return d


def daily_returns(d):
    """未停牌交易日的日收益（相对上一个未停牌收盘）和市值加权市场日收益。
    返回 (个股日收益表 code、date、trade_day、ret、consec、w、close、volume、money、mkcap、adj_factor, 市场日收益 Series[date])。"""
    t = d[d.traded].copy()
    same = t.code.eq(t.code.shift())
    t['ret'] = np.where(same, t.close / t.close.shift() - 1, np.nan)
    t['consec'] = same & (t.trade_day - t.trade_day.shift() == 1)
    t['w'] = np.where(same, t.mkcap.shift(), np.nan)
    c = t[t.consec]
    mkt = (c.w * c.ret).groupby(c.date).sum() / c.w.groupby(c.date).sum()
    cal = pd.Index(sorted(d.date.unique()), name='date')
    return t, mkt.reindex(cal)


def make_weeks(dates):
    """交易日 -> 周序号（自然周，周一为起点，按时间编号）和每周所属的年份（这一周最后一个交易日的年份）。"""
    dt = pd.to_datetime(pd.Series(sorted(set(dates))), format='%Y%m%d')
    start = dt - pd.to_timedelta(dt.dt.weekday, unit='D')
    wk = start.rank(method='dense').astype(int).to_numpy()
    cal = pd.DataFrame({'date': dt.dt.strftime('%Y%m%d').to_numpy(), 'week': wk})
    year = cal.groupby('week').date.max().str[:4].astype(int)
    return dict(zip(cal.date, cal.week)), year


def weekly_returns(t, week_of):
    """个股周收益：每周最后一个未停牌交易日的收盘、总市值；ret = 收盘 / 上一个有交易的周的收盘 − 1；
    consec 表示上一个有交易的周就是上一周；w = 上一个有交易的周最后一天的总市值（组合权重）。"""
    t = t.assign(week=t.date.map(week_of))
    last = t.groupby(['code', 'week'], sort=True).agg(close=('close', 'last'), mkcap=('mkcap', 'last')).reset_index()
    same = last.code.eq(last.code.shift())
    last['ret'] = np.where(same, last.close / last.close.shift() - 1, np.nan)
    last['consec'] = same & (last.week - last.week.shift() == 1)
    last['w'] = np.where(same, last.mkcap.shift(), np.nan)
    return last[same].reset_index(drop=True)


def portfolio_sums(wk, ind_map=None):
    """连续两周有交易的股票按 w 加权：返回 {组: (Σw·r Series[week], Σw Series[week])}；ind_map 为 None 时只有一个组 'ALL'。"""
    c = wk[wk.consec].copy()
    c['g'] = 'ALL' if ind_map is None else c.code.map(ind_map)
    c = c[c.g.notna()]
    c['wr'] = c.w * c.ret
    s = c.groupby(['g', 'week'])[['wr', 'w']].sum()
    return {g: (x.wr.droplevel(0), x.w.droplevel(0)) for g, x in s.groupby(level=0)}


# ---------------------------------------------------------------- 个股的周收益类指标
def _ols_resid(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ b
    r2 = 1 - (e @ e) / ((y - y.mean()) @ (y - y.mean()))
    return e, r2


def crash_measures(e):
    """NCSKEW、DUVOL：W = ln(1 + ε) 去均值后按方案的公式。"""
    w = np.log1p(e)
    if not np.all(np.isfinite(w)):
        return np.nan, np.nan
    w = w - w.mean()
    n = len(w)
    ncskew = -(n * (n - 1) ** 1.5 * (w ** 3).sum()) / ((n - 1) * (n - 2) * ((w ** 2).sum()) ** 1.5)
    down, up = w[w < 0], w[w >= 0]
    nd, nu = len(down), len(up)
    duvol = math.log(((nu - 1) * (down ** 2).sum()) / ((nd - 1) * (up ** 2).sum())) if nd > 1 and nu > 1 else np.nan
    return ncskew, duvol


def ex_industry(own, sums, weeks):
    """本股所在行业剔除本股后的市值加权收益（按市场周序号 weeks 对齐）。own：本股的周收益表（含 consec、w、ret）。"""
    if sums is None:
        return pd.Series(np.nan, index=weeks)
    swr, sw = sums[0].reindex(weeks).fillna(0.0), sums[1].reindex(weeks).fillna(0.0)
    o = own[own.consec].set_index('week')
    owr, ow = (o.w * o.ret).reindex(weeks).fillna(0.0), o.w.reindex(weeks).fillna(0.0)
    den = sw - ow
    return ((swr - owr) / den).where(den > 0)


def stock_week_measures(wk, codes, mkt_w, ind_sums, ind_maps, week_year, years):
    """样本股票每年的 SYNCH、SYNCH_SW、SYNCH_MKT、IVOL、NCSKEW、DUVOL、SIGMA、N_WEEKS。
    ind_sums / ind_maps：{'csrc': ..., 'sw': ...}。不足 MIN_WEEKS 个周收益（回归变量齐全的）记缺失。"""
    weeks = mkt_w.index
    rm = pd.DataFrame({f'rm{k}': mkt_w.shift(-k) for k in (-2, -1, 0, 1, 2)})  # rm-1 = 上一周，rm1 = 下一周
    rows = []
    by_code = dict(tuple(wk[wk.code.isin(codes)].groupby('code')))
    for code in codes:
        own = by_code.get(code)
        ri = {}
        for k in ('csrc', 'sw'):
            g = ind_maps[k].get(code)
            ri[k] = ex_industry(own if own is not None else wk.iloc[:0], ind_sums[k].get(g) if g else None, weeks)
        for y in years:
            rec = {'code': code, 'year': y}
            if own is not None:
                r = own[own.week.map(week_year) == y].set_index('week').ret
                f = rm.loc[r.index].assign(r=r, ri=ri['csrc'].loc[r.index], ri1=ri['csrc'].shift(1).loc[r.index],
                                           rs=ri['sw'].loc[r.index], rs1=ri['sw'].shift(1).loc[r.index])
                rec['N_WEEKS'] = len(r)
                if len(r) >= MIN_WEEKS:
                    rec['SIGMA'] = r.std(ddof=1)
                    specs = {'SYNCH': ['rm0', 'rm-1', 'ri', 'ri1'], 'SYNCH_SW': ['rm0', 'rm-1', 'rs', 'rs1'],
                             'SYNCH_MKT': ['rm0', 'rm-1'], 'CRASH': ['rm-2', 'rm-1', 'rm0', 'rm1', 'rm2']}
                    for name, xs in specs.items():
                        s = f.dropna(subset=['r'] + xs)
                        if len(s) < MIN_WEEKS:
                            continue
                        X = np.column_stack([np.ones(len(s)), s[xs].to_numpy(float)])
                        e, r2 = _ols_resid(s.r.to_numpy(float), X)
                        if name == 'CRASH':
                            rec['NCSKEW'], rec['DUVOL'] = crash_measures(e)
                            rec['N_WEEKS_CRASH'] = len(s)
                        else:
                            rec[name] = math.log(r2 / (1 - r2))
                            if name == 'SYNCH':
                                rec['IVOL'] = e.std(ddof=1)
                                rec['R2_SYNCH'] = r2
                                rec['N_WEEKS_SYNCH'] = len(s)
            rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 个股的日度指标
def year_end_value(t, d, codes, year, col='close'):
    """年末值：每只股票 date ≤ year1231 的最后一个未停牌交易日的 col；要求年内最后一个交易日它还在行情表里（否则缺失）。"""
    end = f'{year}1231'
    last_day = d.date[d.date <= end].max()
    alive = set(d.code[d.date == last_day])
    v = t[(t.date <= end) & t.code.isin(codes)].groupby('code')[col].last()
    return v[v.index.isin(alive)]


def last_traded_value(t, codes, day, col):
    """date ≤ day 的最后一个未停牌交易日的 col（不要求之后还上市；作分母的市值用）。"""
    return t[(t.date <= day) & t.code.isin(codes)].groupby('code')[col].last()


def stock_day_measures(t, d, mkt_d, codes, years):
    """样本股票每年的 RET、BHAR、TURN（%）、ILLIQ、N_DAYS，以及市场组合年度买入持有收益（返回第二个值，{年: 收益}）。"""
    rows, mkt_bh = [], {}
    for y in years:
        p1, p0 = year_end_value(t, d, codes, y), last_traded_value(t, codes, f'{y - 1}1231', 'close')
        m = mkt_d[(mkt_d.index >= f'{y}0101') & (mkt_d.index <= f'{y}1231')]
        mkt_bh[y] = float((1 + m).prod() - 1)
        s = t[(t.date >= f'{y}0101') & (t.date <= f'{y}1231') & t.code.isin(codes)].copy()
        s['turn'] = s.volume * s.close / (s.adj_factor * s.mkcap) * 100  # 成交量 / 总股本，总股本 = 总市值 / 未复权收盘价
        s['illiq'] = np.where(s.money > 0, s.ret.abs() / (s.money / 1e8), np.nan)
        g = s.groupby('code')
        n = g.size()
        f = pd.DataFrame({'N_DAYS': n, 'TURN': g.turn.mean().where(n >= MIN_DAYS),
                          'ILLIQ': np.log(g.illiq.mean()).where(n >= MIN_DAYS)})
        f = f.reindex(codes)
        f['RET'] = (p1 / p0 - 1).reindex(codes)
        f['BHAR'] = f.RET - mkt_bh[y]
        rows.append(f.assign(year=y).rename_axis('code').reset_index())
    return pd.concat(rows, ignore_index=True), mkt_bh


def event_car(t, mkt_d, codes, ann):
    """年报公告窗口的市场调整 CAR。ann：{code: 公告日}；事件日 0 = 公告日当天或之后第一个交易日。
    返回 DataFrame[code, ann_date, day0, CAR_1, CAR_5]（窗口内有停牌或数据不全的记缺失）。"""
    cal = list(mkt_d.index)
    pos = {dt: i for i, dt in enumerate(cal)}
    ret = t[t.code.isin(codes)].set_index(['code', 'date']).ret
    rows = []
    for code in codes:
        a = ann.get(code)
        rec = {'code': code, 'ann_date': a}
        if a:
            i0 = next((i for i, dt in enumerate(cal) if dt >= a), None)
            if i0 is not None:
                rec['day0'] = cal[i0]
                for key, (lo, hi) in CAR_WINDOWS.items():
                    days = cal[i0 + lo:i0 + hi + 1] if i0 + lo >= 0 else []
                    if len(days) == hi - lo + 1:
                        r = [ret.get((code, dt), np.nan) for dt in days]
                        if not np.any(np.isnan(r)):
                            rec[f'CAR_{key}'] = float(np.sum(np.array(r) - mkt_d.loc[days].to_numpy()))
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 财务
FIN_COLS = ['report_date', 'code', 'date', 'company_type', 'total_assets', 'total_liability', 'total_operating_revenue_ttm',
            'total_operating_revenue_q', 'operating_revenue_ttm', 'operating_revenue_q', 'operating_cost_ttm', 'operating_cost_q',
            'operating_profit_ttm', 'net_profit_ttm', 'net_profit_q', 'np_parent_company_owners_ttm', 'np_parent_company_owners_q', 'r_and_d_ttm',
            'net_operate_cash_flow_ttm', 'net_operate_cash_flow_q']


def load_fin(path, codes):
    f = pq.read_table(path, columns=FIN_COLS).to_pandas()
    f = f[f.code.isin(codes)]
    dup = int(f.duplicated(['code', 'report_date']).sum())
    f = f.drop_duplicates(['code', 'report_date'], keep='last').set_index(['code', 'report_date']).sort_index()
    return f, dup


def fin_get(f, codes, rd, col):
    """codes 每只在报告期 rd 的 col（Series[code]，没有的记缺失）。"""
    idx = pd.MultiIndex.from_arrays([list(codes), [rd] * len(codes)])
    return pd.Series(f[col].reindex(idx).to_numpy(), index=list(codes))


def annual_fin(f, codes, y, fin_codes):
    """财年 y 的年度财务指标（ROA、ROE、REVG、NPG、GM、CFO、ASSETG、DLEV、RD、EARNVOL、DROA、NPM、ATO、OPG、ACC），
    以及 SIZE、LEV、equity（年末净资产）。fin_codes：金融业公司（GM、RD、ACC 记缺失）。
    NPM = 净利润 TTM / 营业总收入 TTM；ATO = 营业总收入 TTM / 年初年末总资产均值；OPG = 营业利润 TTM 同比 (y − y−1) / |y−1|；
    ACC = (净利润 TTM − 经营现金流净额 TTM) / 年初年末总资产均值。"""
    g = lambda yy, col: fin_get(f, codes, f'{yy}1231', col)  # noqa: E731
    ta, ta0, ta00 = g(y, 'total_assets'), g(y - 1, 'total_assets'), g(y - 2, 'total_assets')
    tl, tl0, tl00 = g(y, 'total_liability'), g(y - 1, 'total_liability'), g(y - 2, 'total_liability')
    eq, eq0 = ta - tl, ta0 - tl0
    ta_avg = (ta + ta0) / 2
    op, op0 = g(y, 'operating_profit_ttm'), g(y - 1, 'operating_profit_ttm')
    roa = g(y, 'net_profit_ttm') / ((ta + ta0) / 2)
    roa_prev = g(y - 1, 'net_profit_ttm') / ((ta0 + ta00) / 2)
    npar, npar0 = g(y, 'np_parent_company_owners_ttm'), g(y - 1, 'np_parent_company_owners_ttm')
    rev, rev0 = g(y, 'total_operating_revenue_ttm'), g(y - 1, 'total_operating_revenue_ttm')
    orev = g(y, 'operating_revenue_ttm')
    nonfin = ~pd.Series(list(codes), index=list(codes)).isin(fin_codes)
    qroa = pd.concat([fin_get(f, codes, f'{y}{md}', 'net_profit_q') / fin_get(f, codes, f'{y}{md}', 'total_assets')
                      for md in ('0331', '0630', '0930', '1231')], axis=1)
    out = pd.DataFrame({
        'ROA': roa, 'ROE': npar / ((eq + eq0) / 2), 'DROA': roa - roa_prev, 'REVG': rev / rev0 - 1,
        'NPG': (npar - npar0) / npar0.abs(), 'GM': (1 - g(y, 'operating_cost_ttm') / orev).where(nonfin),
        'CFO': g(y, 'net_operate_cash_flow_ttm') / ((ta + ta0) / 2), 'ASSETG': ta / ta0 - 1,
        'DLEV': tl / ta - tl0 / ta0, 'RD': (g(y, 'r_and_d_ttm') / orev).where(nonfin),
        'EARNVOL': qroa.std(axis=1, ddof=1).where(qroa.notna().all(axis=1)),
        'NPM': g(y, 'net_profit_ttm') / rev, 'ATO': rev / ta_avg, 'OPG': (op - op0) / op0.abs(),
        'ACC': ((g(y, 'net_profit_ttm') - g(y, 'net_operate_cash_flow_ttm')) / ta_avg).where(nonfin),
        'SIZE': np.log(ta), 'LEV': tl / ta, 'equity': eq})
    return out.replace([np.inf, -np.inf], np.nan).rename_axis('code').reset_index()


def quarter_fin(f, codes, y, fin_codes, mkcap_prev_end):
    """财年 y 四个季度的单季指标：ROA_Q、REVG_Q、NPG_Q、GM_Q、CFO_Q、SUE_Q（分母 = 上年末总市值 mkcap_prev_end）。"""
    nonfin = ~pd.Series(list(codes), index=list(codes)).isin(fin_codes)
    rows = []
    for q, md in enumerate(('0331', '0630', '0930', '1231'), start=1):
        g = lambda col, yy=y: fin_get(f, codes, f'{yy}{md}', col)  # noqa: E731
        ta = g('total_assets')
        np_, np0 = g('np_parent_company_owners_q'), g('np_parent_company_owners_q', y - 1)
        rows.append(pd.DataFrame({
            'code': list(codes), 'quarter': q,
            'ROA_Q': g('net_profit_q') / ta, 'REVG_Q': g('total_operating_revenue_q') / g('total_operating_revenue_q', y - 1) - 1,
            'NPG_Q': (np_ - np0) / np0.abs(), 'GM_Q': (1 - g('operating_cost_q') / g('operating_revenue_q')).where(nonfin),
            'CFO_Q': g('net_operate_cash_flow_q') / ta, 'SUE_Q': (np_ - np0) / mkcap_prev_end.reindex(list(codes))}).reset_index(drop=True))
    return pd.concat(rows, ignore_index=True).replace([np.inf, -np.inf], np.nan)
