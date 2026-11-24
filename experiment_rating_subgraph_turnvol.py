"""评级子图五个指标的市场反应回归，因变量换成窗口内换手率和窗口内分钟价格波动率（不调用大模型，可反复运行）。

因变量（九个窗口 [−10,−1]、[−5,−1]、[0,1]、[0,2]、[0,5]、[0,10]、[0,20]、[0,40]、[0,60]，交易日；事件日 0 = 市场反应面板的 day0 / p0）
  TURN_w   窗口内未停牌日的日均换手率（%）。日换手率沿用 src/report_car.py：report_car.market_matrices 的 M['turn']、M['traded']
           （成交量 / 总股本 × 100，只用未停牌日）；窗口内没有未停牌日记缺失
  ATURN_w  TURN_w ÷ PRETURN − 1，PRETURN 取 report_car_panel_raw.parquet（未缩尾，第 −60 到 −11 天日均换手率）；原单位（比例）
  RV1_w    1 分钟已实现波动率：股票-交易日内相邻两根 1 分钟 K 线 r = ln(close_i / close_{i−1})（不含隔夜），
           日 RV = sqrt(Σ r²) × 100（百分点）；窗口值 = 窗口内各交易日日 RV 的均值（没有日 RV 的日子跳过，全部没有则缺失）
  RV5_w    同上，用 5 分钟采样收盘价（09:30、09:35、…、11:30、13:05、…、15:00，共 49 个点、48 个收益）
分钟数据  data/market_data/2024.zip（每个交易日一个 YYYYMMDD.parquet），多进程各自打开 zip 读，只留样本公司
           （6 开头 .SH，0/3 开头 .SZ）。K 线放到标准 241 个时点（09:30、09:31–11:30、13:01–15:00）上：
           close 缺失或 ≤ 0 的 K 线和不在标准时点上的 K 线丢掉；缺的时点用当天前一根有效 K 线的收盘价前向填充
           （等价于在相邻有效 K 线之间算收益），开头缺的时点不算收益。全天成交量为 0 或没有 K 线的日子（停牌）不算日 RV。
           窗口里只要有一个交易日不在 20240102–20241231，RV 记缺失。
缩尾      四类因变量 × 九个窗口在市场反应面板 790 篇里按 1%/99% 缩尾，再限定样本。
回归      样本 = “投资评级”是严格汇点的 606 篇 ∩ 市场反应面板（605 篇）；X = rating_subgraph/measures.parquet 里缩尾好的五个指标，
          估计样本内标准化，每次一个；Panel N、A、B、C 和控制变量同 experiment_rating_subgraph.py 的分析三，按公司聚类；
          发布前窗口不放 PRECAR。

在仓库根目录运行：
    C:\\Users\\hung\\projects\\.venv\\Scripts\\python.exe -X utf8 experiment_rating_subgraph_turnvol.py
结果写到 output_dag/forecast_newdata/rating_subgraph/turn_vol/：
    rv_daily.parquet       样本公司每个交易日的 RV1、RV5、K 线根数、有效 K 线根数、成交量
    turnvol_panel.parquet  每篇研报四类因变量 × 九个窗口（<Y>_RAW 原始值，<Y> 缩尾值）
    reg_turnvol.csv        长表（ytype、Y、panel、X、coef、se、t、p、n、n_clusters、fe、controls …）
    missing.csv            各窗口缺失篇数（790 篇和 605 篇样本内）
    bar_check.csv          分钟数据完整性（K 线根数分布、坏 K 线、非标准时点）
    check_rv_naive.csv     随机 20 个股票-交易日朴素重算日 RV 的核对
"""
import io
import math
import random
import sys
import time
import zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
OUT_RS = ROOT_DIR / 'output_dag' / 'forecast_newdata' / 'rating_subgraph'
OUT = OUT_RS / 'turn_vol'
CAR_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
REPORT_PANEL = ROOT_DIR / 'output_dag' / 'regression' / 'report_panel.parquet'
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'
MINUTE_ZIP = DATA_DIR / 'market_data' / '2024.zip'
MIN_FIRST, MIN_LAST = '20240102', '20241231'
WINS = {'m10_m1': (-10, -1), 'm5_m1': (-5, -1), '0_1': (0, 1), '0_2': (0, 2), '0_5': (0, 5), '0_10': (0, 10), '0_20': (0, 20),
        '0_40': (0, 40), '0_60': (0, 60)}
PRE_WINS = ('m10_m1', 'm5_m1')
YTYPES = ['TURN', 'ATURN', 'RV1', 'RV5']
N_PROC = 24
N_NAIVE = 20
SEED = 20261122
# 标准 241 个时点和 5 分钟采样点
GRID = ['09:30:00'] + [f'{h:02d}:{m:02d}:00' for h, m in
                       [(9, x) for x in range(31, 60)] + [(10, x) for x in range(60)] + [(11, x) for x in range(31)] +
                       [(13, x) for x in range(1, 60)] + [(14, x) for x in range(60)] + [(15, 0)]]
GRID5 = [t for t in GRID if int(t[3:5]) % 5 == 0 and t != '13:00:00']
# ==================================================
assert len(GRID) == 241 and len(GRID5) == 49


def full_code(c):
    return c + ('.SH' if c.startswith('6') else '.SZ')


# ---------------------------------------------------------------- 分钟数据（多进程）
def read_day(name, codes_full):
    with zipfile.ZipFile(MINUTE_ZIP) as z:
        b = z.read(name)
    tb = pq.read_table(io.BytesIO(b), columns=['code', 'trade_time', 'close', 'vol', 'date'])
    tb = tb.filter(pc.is_in(tb['code'], value_set=pa.array(sorted(codes_full))))
    return tb.to_pandas()


def day_rv(args):
    """一个交易日文件 -> (每个股票一行的日 RV 表, 完整性统计 dict)。"""
    name, codes_full = args
    df = read_day(name, codes_full)
    date = name[:8]
    info = {'date': date, 'n_rows': len(df), 'n_codes': df.code.nunique(), 'bad_date_col': int((df.date != date).sum())}
    df['t'] = df.trade_time.str[11:]
    info['bad_time_prefix'] = int((df.trade_time.str[:10].str.replace('-', '') != date).sum())
    info['dup_bars'] = int(df.duplicated(['code', 't']).sum())
    bad = df.close.isna() | (df.close <= 0)
    info['bad_close'] = int(bad.sum())
    offgrid = ~df.t.isin(GRID)
    info['offgrid'] = int(offgrid.sum())
    nb = df.groupby('code').size().rename('n_bars')
    vol = df.groupby('code').vol.sum(min_count=1).rename('vol')
    g = df[~bad & ~offgrid].drop_duplicates(['code', 't'], keep='last')
    nv = g.groupby('code').size().rename('n_valid')
    mat = g.pivot(index='code', columns='t', values='close').reindex(columns=GRID)
    filled = mat.ffill(axis=1).to_numpy(float)
    lr = np.diff(np.log(filled), axis=1)
    rv1 = np.sqrt(np.nansum(lr ** 2, axis=1)) * 100
    idx5 = [GRID.index(t) for t in GRID5]
    lr5 = np.diff(np.log(filled[:, idx5]), axis=1)
    rv5 = np.sqrt(np.nansum(lr5 ** 2, axis=1)) * 100
    out = pd.DataFrame({'RV1': rv1, 'RV5': rv5}, index=mat.index)
    out = pd.concat([nb, nv, vol, out], axis=1)
    out['n_valid'] = out.n_valid.fillna(0).astype(int)
    out.loc[~(out.vol > 0), ['RV1', 'RV5']] = np.nan  # 全天成交量 0：停牌，不算
    out.loc[out.n_valid < 2, ['RV1', 'RV5']] = np.nan
    out.index.name = 'code_full'
    out = out.reset_index()
    out['date'] = date
    return out, info


def minute_rv(codes):
    codes_full = sorted(full_code(c) for c in codes)
    with zipfile.ZipFile(MINUTE_ZIP) as z:
        names = sorted(n for n in z.namelist() if n.endswith('.parquet'))
    t0 = time.time()
    with Pool(N_PROC) as pool:
        res = pool.map(day_rv, [(n, codes_full) for n in names], chunksize=2)
    rv = pd.concat([r[0] for r in res], ignore_index=True)
    info = pd.DataFrame([r[1] for r in res])
    rv['code'] = rv.code_full.str[:6]
    print(f'分钟数据：{len(names)} 个交易日、{N_PROC} 进程，{time.time() - t0:.0f} 秒', flush=True)
    return rv, info, names


def naive_rv(name, code_full):
    """最朴素的写法：单进程逐行，按时间排序，相邻两根有效 K 线之间算 1 分钟收益；5 分钟取每个采样点及之前最后一根有效 K 线。"""
    df = read_day(name, [code_full])
    bars = sorted((t[11:], c, v) for t, c, v in zip(df.trade_time, df.close, df.vol))
    if sum(v for _, _, v in bars if v == v) <= 0:
        return float('nan'), float('nan')
    good = [(t, c) for t, c, _ in bars if t in GRID and c == c and c > 0]
    s = 0.0
    for i in range(1, len(good)):
        s += math.log(good[i][1] / good[i - 1][1]) ** 2
    pts = []
    for g in GRID5:
        last = None
        for t, c in good:
            if t <= g:
                last = c
        if last is not None:
            pts.append(last)
    s5 = sum(math.log(pts[i] / pts[i - 1]) ** 2 for i in range(1, len(pts)))
    return math.sqrt(s) * 100, math.sqrt(s5) * 100


# ---------------------------------------------------------------- 窗口变量
def window_vars(raw, rv):
    """每篇研报的 TURN、ATURN、RV1、RV5 × 九个窗口（未缩尾），并核对换手率与已有文件一致。"""
    from src import report_car as RC
    from src import dag_outcomes_vars as V
    rep = pd.read_parquet(REPORT_PANEL)
    codes = sorted(rep.code.unique())
    ind = V.load_industry(INDUSTRY)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in rep.groupby('code').sw.first().items() if c not in sw_map})
    _, _, cal, _, M = RC.market_matrices(DAILY, codes, sw_map)
    cal_arr = np.array(cal)
    p0 = np.searchsorted(cal_arr, raw.date.to_numpy(dtype=str), side='left')
    assert (p0 == raw.p0.to_numpy()).all() and (cal_arr[p0] == raw.day0.to_numpy()).all(), '事件日与面板不一致'
    cal24 = [d for d in cal if MIN_FIRST <= d <= MIN_LAST]
    rvp = {k: rv.pivot(index='code', columns='date', values=k).reindex(index=codes, columns=cal24) for k in ('RV1', 'RV5')}
    rows = []
    for rid, code, i, pret in zip(raw.report_id, raw.code, p0, raw.PRETURN):
        turn, tr = M['turn'].loc[code].to_numpy(), M['traded'].loc[code].to_numpy()
        rec = {'report_id': rid}
        for s, (lo, hi) in WINS.items():
            assert i + lo >= 0 and i + hi < len(cal)
            w = slice(i + lo, i + hi + 1)
            tw = tr[w]
            rec[f'TURN_{s}'] = float(np.mean(turn[w][tw])) if tw.any() else np.nan
            rec[f'ATURN_{s}'] = rec[f'TURN_{s}'] / pret - 1
            days = cal[i + lo:i + hi + 1]
            inside = days[0] >= MIN_FIRST and days[-1] <= MIN_LAST
            rec[f'out24_{s}'] = not inside
            for k in ('RV1', 'RV5'):
                if not inside:
                    rec[f'{k}_{s}'] = np.nan
                    continue
                v = rvp[k].loc[code, days].to_numpy(float)
                rec[f'{k}_{s}'] = float(np.nanmean(v)) if np.isfinite(v).any() else np.nan
        rows.append(rec)
    p = pd.DataFrame(rows)
    # 核对：TURN 与 car_metrics、ATURN_0_2 与 report_car_panel_raw
    cm = pd.read_parquet(CAR_DIR / 'car_metrics' / 'report_car_metrics.parquet').set_index('report_id').loc[p.report_id]
    msgs = []
    for s in ('m5_m1', '0_1', '0_2', '0_5', '0_10'):
        assert np.allclose(p[f'TURN_{s}'], cm[f'TURN_{s}'], rtol=1e-12, atol=0, equal_nan=True), s
    msgs.append(f'TURN_m5_m1、0_1、0_2、0_5、0_10 与 car_metrics/report_car_metrics.parquet 逐篇一致（{len(p)} 篇）')
    assert np.allclose(p.ATURN_0_2, raw.ATURN_0_2, rtol=1e-12, atol=0, equal_nan=True)
    msgs.append(f'ATURN_0_2 与 report_car_panel_raw.parquet 逐篇一致（{len(p)} 篇）')
    # 交易日历与分钟数据文件一致
    return p, msgs, cal24


# ---------------------------------------------------------------- 主流程
def main():
    from experiment_rating_subgraph import CAR_PANELS, CTRL_CAR, CTRL_FIRM, MEAS, SG
    from src import dag_outcomes_reg as R
    t_start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(CAR_DIR / 'report_car_panel_raw.parquet')
    w = pd.read_parquet(CAR_DIR / 'report_car_panel.parquet')
    assert list(raw.report_id) == list(w.report_id)
    codes = sorted(raw.code.unique())

    rv, info, names = minute_rv(codes)
    # 完整性
    present = set(rv.code_full)
    miss_codes = sorted(set(full_code(c) for c in codes) - present)
    bc = rv.groupby('n_bars').size().rename('n_stock_days').reset_index()
    bc.insert(0, 'item', 'K 线根数分布（有记录的股票-交易日）')
    tot = pd.DataFrame([{'item': k, 'n_bars': np.nan, 'n_stock_days': int(info[k].sum())}
                        for k in ('bad_close', 'offgrid', 'dup_bars', 'bad_date_col', 'bad_time_prefix')])
    extra = pd.DataFrame([
        {'item': '交易日文件数', 'n_bars': np.nan, 'n_stock_days': len(names)},
        {'item': '样本股票数（790 篇的公司）', 'n_bars': np.nan, 'n_stock_days': len(codes)},
        {'item': '分钟数据里一次也没出现的样本股票', 'n_bars': np.nan, 'n_stock_days': len(miss_codes)},
        {'item': '有记录的股票-交易日', 'n_bars': np.nan, 'n_stock_days': len(rv)},
        {'item': '其中全天成交量为 0（不算 RV）', 'n_bars': np.nan, 'n_stock_days': int((~(rv.vol > 0)).sum())},
        {'item': '其中有效 K 线 < 241（成交量 > 0）', 'n_bars': np.nan, 'n_stock_days': int(((rv.n_valid < 241) & (rv.vol > 0)).sum())},
        {'item': '股票-交易日总数（样本股票 × 242）', 'n_bars': np.nan, 'n_stock_days': len(codes) * len(names)},
    ])
    bar_check = pd.concat([extra, tot, bc], ignore_index=True)
    bar_check.to_csv(OUT / 'bar_check.csv', index=False, encoding='utf-8-sig')
    print(bar_check.to_string(index=False), flush=True)
    if miss_codes:
        print('  分钟数据里没有的样本股票：', miss_codes, flush=True)
    rv[['code', 'code_full', 'date', 'RV1', 'RV5', 'n_bars', 'n_valid', 'vol']].to_parquet(OUT / 'rv_daily.parquet', index=False)

    # 朴素重算抽查
    ok_rv = rv[rv.RV1.notna()].reset_index(drop=True)
    pick = random.Random(SEED).sample(range(len(ok_rv)), N_NAIVE)
    chk = []
    for j in pick:
        r = ok_rv.iloc[j]
        a1, a5 = naive_rv(r.date + '.parquet', r.code_full)
        chk.append({'code_full': r.code_full, 'date': r.date, 'n_bars': r.n_bars, 'RV1': r.RV1, 'RV1_naive': a1, 'RV5': r.RV5, 'RV5_naive': a5})
    chk = pd.DataFrame(chk)
    chk['ok'] = np.isclose(chk.RV1, chk.RV1_naive, rtol=1e-9) & np.isclose(chk.RV5, chk.RV5_naive, rtol=1e-9)
    chk.to_csv(OUT / 'check_rv_naive.csv', index=False, encoding='utf-8-sig')
    assert chk.ok.all(), chk[~chk.ok]
    print(f'  核对：随机 {N_NAIVE} 个股票-交易日朴素重算 RV1、RV5 全部一致', flush=True)

    p, msgs, cal24 = window_vars(raw, rv)
    assert sorted(n[:8] for n in names) == cal24, '分钟数据交易日与日行情 2024 年交易日历不一致'
    msgs.append(f'分钟数据 {len(names)} 个交易日与日行情 2024 年交易日历完全一致')
    for s in msgs:
        print('  核对：' + s, flush=True)
    ys = [f'{k}_{s}' for k in YTYPES for s in WINS]
    panel = p[['report_id']].copy()
    for y in ys:
        panel[y + '_RAW'] = p[y]
        panel[y] = R.winsor(p[y])
    panel = panel.merge(raw[['report_id', 'code', 'date', 'day0']], on='report_id')
    m = pd.read_parquet(OUT_RS / 'measures.parquet')
    in_s = panel.report_id.isin(m.report_id)
    miss = []
    for s in WINS:
        miss.append({'window': s, 'out_of_2024_790': int(p[f'out24_{s}'].sum()),
                     **{f'{k}_missing_790': int(p[f'{k}_{s}'].isna().sum()) for k in YTYPES},
                     **{f'{k}_missing_605': int(p.loc[in_s.to_numpy(), f'{k}_{s}'].isna().sum()) for k in YTYPES}})
    miss = pd.DataFrame(miss)
    miss.to_csv(OUT / 'missing.csv', index=False, encoding='utf-8-sig')
    print(miss.to_string(index=False), flush=True)
    panel.to_parquet(OUT / 'turnvol_panel.parquet', index=False)

    # 回归
    d = w.drop(columns=[c for c in w.columns if c.startswith(('TURN_', 'ATURN_', 'RV1_', 'RV5_'))])
    d = d.merge(panel[['report_id'] + ys], on='report_id').merge(m[['report_id'] + MEAS].rename(columns=SG), on='report_id')
    assert len(d) == 605
    rows = []
    for k in YTYPES:
        for s in WINS:
            y = f'{k}_{s}'
            for pan, fe in CAR_PANELS.items():
                ctrl = [c for c in CTRL_CAR if not (pan in ('B', 'C') and c in CTRL_FIRM) and not (s in PRE_WINS and c == 'PRECAR')]
                for x in MEAS:
                    res, smp = R.fit(d, y, [SG[x]], ctrl, fe, 'code')
                    r = res[0]
                    rows.append({'ytype': k, 'Y': y, 'window': s, 'panel': pan, 'X': x, 'coef': r['coef'], 'se': r['se'], 't': r['t'],
                                 'p': r['p'], 'n': r['n'], 'n_clusters': r['n_clusters'], 'n_singletons': r['n_singletons'],
                                 'y_mean': r['y_mean'], 'y_sd': r['y_sd'], 'fe': '+'.join(fe), 'controls': '+'.join(ctrl)})
    res = pd.DataFrame(rows)
    res.to_csv(OUT / 'reg_turnvol.csv', index=False, encoding='utf-8-sig')
    print(f'回归 {len(res)} 个系数；完成 -> {OUT}（总耗时 {time.time() - t_start:.0f} 秒）', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
