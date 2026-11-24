"""790 篇深度研报逐篇整理：发布前后 6 个窗口的 CAR、换手率和异常换手率 + 概念 DAG 结构指标（含“到投资评级”的四个指标），写数据文件。
页面在 src/viz/plot_car_metrics.py，入口 experiment_car_metrics.py。实验性分析，只读已有输出。

口径：
  CAR       直接取事件面板（pathlen_event_panel_raw.parquet，未缩尾）里的 6 个窗口，个股收益减全市场市值加权收益的累计值，百分点
  DAG       output_dag/deep/data 与 output_dag/data 的 edges.parquet（report_id 重复时用 deep 的），只用 in_dag 为真的边；
            建图和整图指标用 src/viz/plot_dag_structure.py 的 build_graph / path_stats / measure（同 dag_outcomes_vars.load_dag）
  整图      RATIO = 源点数 / 汇点数；NPATH = 源到汇路径数；LNPATH = ln(1 + NPATH)；PATHLEN = 平均路径边数（每条路径权重相同）
  投资评级  T = 名称恰好等于 src/valuation_sinks.TARGET（“投资评级”）的概念，精确相等，不要求是汇点
            V_HAS = 图里有没有 T（1/0）；
            V_NPATH = 从源点（入度为 0 且不是 T 本身）沿有向边走到 T 的不同路径条数，路径在 T 处结束；V_LNPATH = ln(1 + V_NPATH)；
            V_PATHLEN = 这些路径的平均边数（V_NPATH = 0 时缺失）；V_RATIO = 能走到 T 的源点数 / 1（没有 T 时缺失）
            没有 T 或没有源点能走到 T 时 V_NPATH = 0。
  换手率    沿用 src/report_car.py：日换手率 = 成交量 / 总股本 × 100（%）= volume * close / (adj_factor * mkcap) * 100，
            直接用 report_car.market_matrices 的 M['turn']、M['traded']，只用未停牌日；第 0 天 = 事件面板的 day0 / p0（与 CAR 相同）。
            TURN_<窗口>  = 窗口内未停牌日的日均换手率（%），窗口内没有未停牌日则缺失；窗口同 CAR 的 6 个
            PRETURN     = 第 −60 到 −11 天未停牌日的日均换手率（%），不足 20 个未停牌日记缺失（同 report_car.PRE_WIN / MIN_PRE_DAYS）
            ATURN_<窗口> = TURN_<窗口> / PRETURN − 1（与 report_car 的 ATURN_0_2 同一定义）
            自检：ATURN_0_2、PRETURN 须与 report_car_panel_raw.parquet（未缩尾）逐篇相等，否则不写文件。
  全部不缩尾。（上一版用的是“估值相关汇点”宽口径 src/valuation_sinks.group()，现在不用。）
"""
import math
import time
from collections import Counter, defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import dag_outcomes_vars as V
from src import report_car as RC
from src.valuation_sinks import TARGET
from src.viz.plot_dag_structure import measure

CARS = ['CAR_m5_m1', 'CAR_0_1', 'CAR_0_2', 'CAR_0_3', 'CAR_0_5', 'CAR_0_10']
TURN_WIN = {'m5_m1': (-5, -1), '0_1': (0, 1), '0_2': (0, 2), '0_3': (0, 3), '0_5': (0, 5), '0_10': (0, 10)}  # 后缀同 CARS
TURNS = [f'TURN_{s}' for s in TURN_WIN]
ATURNS = [f'ATURN_{s}' for s in TURN_WIN]
TURN_COLS = TURNS + ATURNS + ['PRETURN']
METRICS = ['RATIO', 'NPATH', 'LNPATH', 'PATHLEN', 'V_HAS', 'V_NPATH', 'V_LNPATH', 'V_PATHLEN', 'V_RATIO']


def load_edges(dag_dirs, report_ids):
    """照 dag_outcomes_vars.load_dag：各目录 reports.parquet 按 dag_dirs 顺序合并、report_id 重复时用前面的；
    每篇只取所属目录 edges.parquet 里 in_dag 为真的边 -> {report_id: [(cause, effect, support)]}。"""
    frames = [pq.read_table(d / 'reports.parquet', columns=['report_id']).to_pandas().assign(src=i)
              for i, d in enumerate(dag_dirs)]
    reports = pd.concat(frames, ignore_index=True).drop_duplicates('report_id', keep='first')
    reports = reports[reports.report_id.isin(report_ids)]
    lack = set(report_ids) - set(reports.report_id)
    if lack:
        raise SystemExit(f'{len(lack)} 篇研报在 DAG 目录里找不到：{sorted(lack)[:5]}')
    raw = defaultdict(list)
    for i, d in enumerate(dag_dirs):
        mine = set(reports.report_id[reports.src == i])
        for e in pq.read_table(d / 'edges.parquet', columns=['report_id', 'cause', 'effect', 'support', 'in_dag']).to_pylist():
            if e['in_dag'] and e['report_id'] in mine:
                raw[e['report_id']].append((e['cause'], e['effect'], e['support']))
    return raw


def target_stats(g, t):
    """动态规划（同 path_stats，从 T 往回递推，只在 T 的祖先子图上算）：
    cnt[v] = 从 v 出发、在 T 处结束的路径条数，tot[v] = 这些路径的边数之和；cnt[T] = 1、tot[T] = 0（T 的出边不管）。
    返回 (V_NPATH, 边数总和, 能走到 T 的源点数)；源点 = 入度为 0 且不是 T 本身。图里没有 T 时返回 (0, 0, 0)。"""
    if t not in g:
        return 0, 0, 0
    anc = nx.ancestors(g, t)
    sub = g.subgraph(anc | {t})
    cnt, tot = {}, {}
    for v in reversed(list(nx.topological_sort(sub))):
        if v == t:
            cnt[v], tot[v] = 1, 0
        else:
            succ = list(sub.successors(v))
            cnt[v] = sum(cnt[s] for s in succ)
            tot[v] = sum(tot[s] + cnt[s] for s in succ)
    sources = [v for v in anc if g.in_degree(v) == 0]  # T 的祖先里入度为 0 的就是能走到 T 的源点（祖先都至少有一条到 T 的路）
    return sum(cnt[v] for v in sources), sum(tot[v] for v in sources), len(sources)


def report_metrics(edges):
    g, m = measure(edges)
    sinks = [v for v in g if g.out_degree(v) == 0]
    has = TARGET in g
    n_v, len_v, src_v = target_stats(g, TARGET)
    rec = {'RATIO': m['ratio'], 'NPATH': m['n_paths'], 'LNPATH': math.log1p(m['n_paths']), 'PATHLEN': m['len_mean'],
           'V_HAS': int(has), 'V_NPATH': n_v, 'V_LNPATH': math.log1p(n_v),
           'V_PATHLEN': len_v / n_v if n_v else None, 'V_RATIO': src_v / 1 if has else None}
    return rec, sinks, m, g


def turnover(ev, daily_path, industry_path):
    """事件面板（含 code、date、day0、p0、sw）-> 每篇一行：report_id + TURN_* + ATURN_* + PRETURN（未缩尾），和自检信息。
    行情矩阵用 report_car.market_matrices（sw_map 的构造同 report_car_event.load_market，只影响行业收益，换手率用不到）；
    事件日用同样的 searchsorted 重新定位，并核对与事件面板的 p0、day0 一致。"""
    t0 = time.time()
    codes = sorted(ev.code.unique())
    ind = V.load_industry(industry_path)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in ev.groupby('code').sw.first().items() if c not in sw_map})
    _, _, cal, _, M = RC.market_matrices(daily_path, codes, sw_map)
    t_load = time.time() - t0
    cal_arr = np.array(cal)
    p0 = np.searchsorted(cal_arr, ev.date.to_numpy(dtype=str), side='left')
    if not ((p0 == ev.p0.to_numpy()).all() and (cal_arr[np.minimum(p0, len(cal) - 1)] == ev.day0.to_numpy()).all()):
        raise SystemExit('事件日与事件面板的 p0 / day0 不一致；未写出任何文件')
    rows, n_out, n_nan_turn = [], 0, 0
    for rid, code, i in zip(ev.report_id, ev.code, p0):
        turn, tr = M['turn'].loc[code].to_numpy(), M['traded'].loc[code].to_numpy()
        n_nan_turn += int(np.isnan(turn[tr]).sum())
        rec = {'report_id': rid}
        pre = slice(i + RC.PRE_WIN[0], i + RC.PRE_WIN[1] + 1)
        tp = tr[pre]
        rec['PRETURN'] = float(np.mean(turn[pre][tp])) if tp.sum() >= RC.MIN_PRE_DAYS else np.nan  # 同 report_car.build_panel
        for s, (lo, hi) in TURN_WIN.items():
            if i + lo < 0 or i + hi >= len(cal):  # 窗口超出行情区间（本样本不会发生）
                n_out += 1
                rec[f'TURN_{s}'] = rec[f'ATURN_{s}'] = np.nan
                continue
            w = slice(i + lo, i + hi + 1)
            tw = tr[w]
            rec[f'TURN_{s}'] = float(np.mean(turn[w][tw])) if tw.any() else np.nan
            rec[f'ATURN_{s}'] = rec[f'TURN_{s}'] / rec['PRETURN'] - 1  # PRETURN 缺失则缺失
        rows.append(rec)
    info = {'n_cal': len(cal), 'cal_first': cal[0], 'cal_last': cal[-1], 'n_window_out_of_range': n_out,
            'n_nan_turn_on_traded_days': n_nan_turn, 'sec_load_market': round(t_load, 1), 'sec_total': round(time.time() - t0, 1)}
    return pd.DataFrame(rows), info


def gate(df, car_raw_panel):
    """新算的 ATURN_0_2、PRETURN 与 report_car_panel_raw.parquet（未缩尾）逐篇比较：缺失位置一致且数值完全相等才算通过。"""
    ref = pd.read_parquet(car_raw_panel, columns=['report_id', 'PRETURN', 'ATURN_0_2'])
    m = df[['report_id', 'PRETURN', 'ATURN_0_2']].merge(ref, on='report_id', how='left', suffixes=('', '_ref'), indicator=True)
    res = {'n_reports': len(df), 'n_matched': int((m._merge == 'both').sum())}
    ok = res['n_matched'] == len(df)
    for c in ('PRETURN', 'ATURN_0_2'):
        a, b = m[c].to_numpy(float), m[c + '_ref'].to_numpy(float)
        both = ~np.isnan(a) & ~np.isnan(b)
        res[c] = {'n_nonmissing': int(both.sum()), 'n_missing_mismatch': int((np.isnan(a) != np.isnan(b)).sum()),
                  'n_value_unequal': int((a[both] != b[both]).sum()),
                  'max_absdiff': float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0}
        ok &= res[c]['n_missing_mismatch'] == 0 and res[c]['n_value_unequal'] == 0
    return ok, res


def build(event_panel, reports_meta, dag_dirs):
    """-> (每篇一行的表, 自检信息)。"""
    ev = pd.read_parquet(event_panel, columns=['report_id', 'code', 'company', 'date', 'day0'] + CARS)
    meta = pd.read_parquet(reports_meta, columns=['report_id', 'broker']).drop_duplicates('report_id')
    ev = ev.merge(meta, on='report_id', how='left')
    raw = load_edges(dag_dirs, set(ev.report_id))
    rows, sink_names = [], Counter()
    n_sinks_total, n_edges0, n_t_sink, n_t_out = 0, 0, 0, 0
    for rid in ev.report_id:
        rec, sinks, m, g = report_metrics(raw[rid])
        n_edges0 += m['n_edges'] == 0
        n_sinks_total += len(sinks)
        sink_names.update(set(sinks))
        if rec['V_HAS']:
            if g.out_degree(TARGET) == 0:
                n_t_sink += 1
            else:
                n_t_out += 1
        rows.append({'report_id': rid, **rec})
    out = ev[['report_id', 'code', 'company', 'broker', 'date', 'day0'] + CARS].merge(pd.DataFrame(rows), on='report_id')
    for c in ('RATIO', 'LNPATH', 'PATHLEN', 'V_LNPATH', 'V_PATHLEN', 'V_RATIO'):
        out[c] = pd.to_numeric(out[c], errors='coerce').astype(float)
    for c in ('NPATH', 'V_NPATH'):
        if out[c].max() < 2 ** 63:
            out[c] = out[c].astype('int64')
        else:
            out[c] = out[c].astype(float)
    out['V_HAS'] = out.V_HAS.astype('int64')
    n_has = int(out.V_HAS.sum())
    check = {'n_reports': len(out), 'n_sinks': n_sinks_total, 'n_sink_names': len(sink_names),
             'n_reports_with_t': n_has, 'n_t_sink': n_t_sink, 'n_t_with_out': n_t_out, 'n_reports_no_t': len(out) - n_has,
             'n_reports_no_edges': int(n_edges0), 'n_broker_missing': int(out.broker.isna().sum())}
    return out, check


EXPECT = {'n_reports': 790, 'n_sinks': 17164, 'n_sink_names': 11660,
          'n_reports_with_t': 607, 'n_t_sink': 605, 'n_t_with_out': 2, 'n_reports_no_t': 183}


def describe(df):
    rows = []
    for c in CARS + TURN_COLS + METRICS:
        x = df[c].dropna().astype(float)
        rows.append({'var': c, 'N': len(x), 'n_missing': int(df[c].isna().sum()), 'mean': x.mean(), 'median': x.median(),
                     'min': x.min(), 'max': x.max()})
    return pd.DataFrame(rows)


def compare_regression(df, report_panel):
    """与 output_dag/regression/report_panel.parquet（1%/99% 缩尾）比：把本表的值按那边各变量的最小、最大值（即缩尾界）截断后逐篇比较。"""
    rp = pd.read_parquet(report_panel, columns=['report_id', 'RATIO', 'LNPATH', 'PATHLEN'])
    m = df[['report_id', 'RATIO', 'LNPATH', 'PATHLEN']].merge(rp, on='report_id', suffixes=('', '_reg'))
    res = []
    for c in ('RATIO', 'LNPATH', 'PATHLEN'):
        a, b = m[c], m[c + '_reg']
        both = a.notna() & b.notna()
        lo, hi = b.min(), b.max()
        inner = both & (a > lo) & (a < hi)
        diff_inner = (a[inner] - b[inner]).abs().max() if inner.any() else 0.0
        diff_clip = (a[both].clip(lo, hi) - b[both]).abs().max()
        res.append({'var': c, 'n_matched': len(m), 'n_both': int(both.sum()), 'n_missing_mismatch': int((a.isna() != b.isna()).sum()),
                    'lo': lo, 'hi': hi, 'n_inner': int(inner.sum()), 'max_absdiff_inner': diff_inner,
                    'max_absdiff_after_clip': diff_clip})
    return pd.DataFrame(res)


def main(out_dir, event_panel, reports_meta, dag_dirs, daily_path, industry_path, car_raw_panel, report_panel=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    df, check = build(event_panel, reports_meta, dag_dirs)
    print('自检：', check)
    bad = {k: (check[k], v) for k, v in EXPECT.items() if check[k] != v}
    if bad:
        raise SystemExit(f'自检不符（实际, 预期）：{bad}；未写出任何文件')
    ev = pd.read_parquet(event_panel, columns=['report_id', 'code', 'date', 'day0', 'p0', 'sw'])
    tv, tinfo = turnover(ev, daily_path, industry_path)
    print('换手率：', tinfo, flush=True)
    ok, gres = gate(tv, car_raw_panel)
    print(f'闸门（对 {car_raw_panel.name} 逐篇比较 PRETURN、ATURN_0_2）：', gres, flush=True)
    if not ok:
        raise SystemExit('闸门未通过：新算的 PRETURN / ATURN_0_2 与未缩尾面板不一致；未写出任何文件')
    base = [c for c in df.columns if c not in METRICS]
    df = df[base].merge(tv[['report_id'] + TURN_COLS], on='report_id', how='left').merge(df[['report_id'] + METRICS], on='report_id')
    df.to_parquet(out_dir / 'report_car_metrics.parquet', index=False)
    df.to_csv(out_dir / 'report_car_metrics.csv', index=False, encoding='utf-8-sig')
    for old in ('valuation_sinks.csv', 'sinks_all.csv'):  # 上一版“估值相关汇点”口径的产物，留着会误导
        if (out_dir / old).exists():
            (out_dir / old).unlink()
            print(f'已删除旧文件 {out_dir / old}')
    with pd.option_context('display.width', 200, 'display.max_columns', 20, 'display.float_format', '{:.4f}'.format):
        print(describe(df).to_string(index=False))
        print('V_HAS 分布：', df.V_HAS.value_counts().sort_index().to_dict())
        if report_panel is not None and report_panel.exists():
            print(compare_regression(df, report_panel).to_string(index=False))
    return df, check
