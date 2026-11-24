"""“投资评级”上游子图的五个结构指标，与盈利预测准确率、研报发布后市场回报的关系（不调用大模型，可反复运行）。

样本  791 篇 DAG 研报里，图中有名称恰好为“投资评级”的节点、且它是严格汇点（入度 > 0、出度 = 0）的研报（应为 606 篇）。
子图  “投资评级”节点 + 它的全部祖先，以及这些节点之间的边（祖先的前驱也是祖先，所以子图内入度 = 原图入度）。
指标  src/dag_measures.five_measures(graphs, '投资评级', strict=True)（定义见该模块；整张图口径用 target=None）：
      NSRC     子图里入度为 0 的节点数（= 面板 R_RATIO_RAW）
      LNPATH   ln(1 + 从各源点到“投资评级”的不同路径条数)（= R_LNPATH_RAW）
      PATHLEN  这些路径的平均边数（= R_PATHLEN_RAW）
      NNODE    子图节点数，含“投资评级”本身（= R_N_ANC + 1）
      DEPTH    从源点到“投资评级”的最长路径边数
      五个指标在 606 篇里按 1%/99% 缩尾（列名不带后缀为缩尾后，_RAW 为原始值）；回归时在估计样本内标准化，每次只放一个。

分析二（预测准确率）  读 <输入目录>/rating_strict_n/（及 main_only/）的 forecast_panel.parquet（缩尾后），
      Y = PMAFE_FY24、PMAFE_FY25、AFE_FY24、AFE_FY25；Panel N、A、B、C 的固定效应 / 控制变量同 src/dag_forecast.py（= reg_results.csv），
      按公司聚类（dag_forecast.regress_each）。先用同一估计程序复现 reg_results.csv 里 R_RATIO、R_LNPATH、R_PATHLEN 的单独回归（系数、t、N）。
分析三（市场回报）  读 output_dag/experiments/report_car/report_car_panel.parquet（缩尾后）；新算 CAR_0_40、CAR_0_60
      （src/report_car.py 的 market_matrices + window_car，同一套口径；在 790 篇市场反应面板里按 1%/99% 缩尾，与已有 Y 一致），
      同时重算已有 7 个窗口并核对与 report_car_panel_raw.parquet 完全一致。CAR × 100（百分点）。
      Panel N 无固定效应；A 申万一级 + 发布月份；B 公司 + 发布月份；C 公司 + 券商 + 发布月份；按公司聚类。
      控制变量 N、A：LNCHARS_R、LNNODES_R、PRECAR、PREVOL、PRETURN、EA、NREP、SIZE23、LEV23、BM23、ROA23；B、C 去掉四个公司变量；
      发布前窗口（CAR_m10_m1、CAR_m5_m1，安慰剂）去掉 PRECAR。

在仓库根目录运行：
    C:\\Users\\hung\\projects\\.venv\\Scripts\\python.exe -X utf8 experiment_rating_subgraph.py [--input-dir 目录] [--output 目录]
--input-dir 默认 output_dag/forecast_newdata（读其下 rating_strict_n/），--output 默认 output_dag/forecast_newdata/rating_subgraph/：
    measures.parquet      每篇五个指标（原始值、缩尾值）
    descriptives.csv      分布（原始值、缩尾值）
    corr.csv              五个指标与 LNNODES_R、LNCHARS_R 的相关系数矩阵（Pearson、Spearman；缩尾值）
    reg_forecast.csv      分析二长表（scope = all / main）
    reg_car.csv           分析三长表
    car_long.parquet      新算的 CAR_0_40、CAR_0_60（长表：原始值、缩尾值、窗口内停牌天数）
    check_baseline.csv    分析二对 reg_results.csv 的复现核对
    check_depth.csv       DEPTH 的抽查（穷举简单路径）
"""
import argparse
import random
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import dag_forecast as F
from src import dag_measures as D
from src import dag_outcomes_reg as R
from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
ND = ROOT_DIR / 'output_dag' / 'forecast_newdata'
OUT = ND / 'rating_subgraph'
DAG_DIRS = [ROOT_DIR / 'output_dag' / 'deep' / 'data', ROOT_DIR / 'output_dag' / 'data']  # 同 regress_dag_forecast.py
CAR_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
REPORT_PANEL = ROOT_DIR / 'output_dag' / 'regression' / 'report_panel.parquet'  # 同 experiment_report_car.py
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'
MEAS = D.MEAS  # NSRC、LNPATH、PATHLEN、NNODE、DEPTH
SG = {x: 'SG_' + x for x in MEAS}  # 回归时的列名（市场反应面板和预测面板里已有整张图的 LNPATH、PATHLEN）
PANEL_COL = {'NSRC': 'R_RATIO', 'LNPATH': 'R_LNPATH', 'PATHLEN': 'R_PATHLEN'}  # 面板已有的对应列
YS_FC = ['PMAFE_FY24', 'PMAFE_FY25', 'AFE_FY24', 'AFE_FY25']
SCOPES = {'all': '', 'main': 'main_only'}
NEW_WIN = {'CAR_0_40': (0, 40), 'CAR_0_60': (0, 60)}
Y_CAR_POST = ['CAR_0_1', 'CAR_0_2', 'CAR_0_5', 'CAR_0_10', 'CAR_0_20', 'CAR_0_40', 'CAR_0_60']
Y_CAR_PRE = ['CAR_m10_m1', 'CAR_m5_m1']
CTRL_FIRM = ['SIZE23', 'LEV23', 'BM23', 'ROA23']
CTRL_CAR = ['LNCHARS_R', 'LNNODES_R', 'PRECAR', 'PREVOL', 'PRETURN', 'EA', 'NREP'] + CTRL_FIRM
CAR_PANELS = {'N': [], 'A': ['sw', 'month'], 'B': ['code', 'month'], 'C': ['code', 'broker_g', 'month']}
N_DEPTH_CHECK = 60  # 穷举简单路径抽查的篇数
MAX_ENUM_PATHS = 50000  # 抽查只在路径条数不超过这个数的研报里抽
SEED = 20261122
# ==================================================


# ---------------------------------------------------------------- 指标与核对
def check_measures(m, graphs, raw):
    """前四个指标与面板一致；DEPTH 全部用 nx.dag_longest_path_length 复核，再穷举简单路径抽查。"""
    p = raw.set_index('report_id')
    have = set(p.index[p.R_LNPATH_RAW.notna()])
    assert set(m.report_id) == have, f'样本与面板 R_LNPATH_RAW 非缺失的研报不一致：{len(set(m.report_id) ^ have)} 篇'
    q = p.loc[m.report_id]
    msgs = []
    for x, col in PANEL_COL.items():
        assert np.allclose(m[x + '_RAW'], q[col + '_RAW'], rtol=1e-12, atol=1e-12), x
        assert np.allclose(m[x], q[col], rtol=1e-12, atol=1e-12), x + '（缩尾）'
        msgs.append(f'{x} 与 {col}_RAW / {col} 逐篇一致（{len(m)} 篇）')
    assert (m.NNODE_RAW.to_numpy() == q.R_N_ANC.to_numpy() + 1).all(), 'NNODE'
    assert (m.N_PATHS.to_numpy() == q.R_N_PATHS.to_numpy()).all(), 'N_PATHS'
    msgs.append(f'NNODE = R_N_ANC + 1、路径条数 = R_N_PATHS 逐篇一致（{len(m)} 篇）')
    t = D.RATING_NODE
    subs = {r: graphs[r].subgraph(nx.ancestors(graphs[r], t) | {t}) for r in m.report_id}
    lp = np.array([nx.dag_longest_path_length(subs[r]) for r in m.report_id], float)
    assert (lp == m.DEPTH_RAW.to_numpy()).all(), 'DEPTH vs dag_longest_path_length'
    msgs.append(f'DEPTH 与 networkx dag_longest_path_length（子图）全部 {len(m)} 篇一致')
    pool = sorted(m.report_id[m.N_PATHS <= MAX_ENUM_PATHS])
    pick = random.Random(SEED).sample(pool, min(N_DEPTH_CHECK, len(pool)))
    rows = []
    for rid in pick:
        sub, r = subs[rid], m.set_index('report_id').loc[rid]
        lens = [len(pth) - 1 for s in sub if sub.in_degree(s) == 0 for pth in nx.all_simple_paths(sub, s, t)]
        rows.append({'report_id': rid, 'n_paths_enum': len(lens), 'n_paths': int(r.N_PATHS), 'depth_enum': max(lens),
                     'depth': r.DEPTH_RAW, 'pathlen_enum': float(np.mean(lens)), 'pathlen': r.PATHLEN_RAW})
    ck = pd.DataFrame(rows)
    ck['ok'] = (ck.n_paths_enum == ck.n_paths) & (ck.depth_enum == ck.depth) & np.isclose(ck.pathlen_enum, ck.pathlen)
    assert ck.ok.all(), ck[~ck.ok]
    msgs.append(f'穷举简单路径抽查 {len(ck)} 篇（路径条数 ≤ {MAX_ENUM_PATHS} 的 {len(pool)} 篇中随机抽）：路径条数、平均长度、DEPTH 全部一致')
    return msgs, ck


# ---------------------------------------------------------------- 分析一
def describe(m):
    rows = []
    for ver, suf in (('原始值', '_RAW'), ('缩尾值', '')):
        for x in MEAS:
            s = m[x + suf]
            q = s.quantile([.05, .25, .5, .75, .95])
            rows.append({'version': ver, 'var': x, 'n': int(s.notna().sum()), 'mean': s.mean(), 'sd': s.std(), 'min': s.min(),
                         'p5': q.iloc[0], 'p25': q.iloc[1], 'p50': q.iloc[2], 'p75': q.iloc[3], 'p95': q.iloc[4], 'max': s.max()})
    return pd.DataFrame(rows)


def corr(m, raw):
    d = m[['report_id'] + MEAS].merge(raw[['report_id', 'LNNODES_R', 'LNCHARS_R']], on='report_id')
    cols = MEAS + ['LNNODES_R', 'LNCHARS_R']
    out = []
    for meth in ('pearson', 'spearman'):
        c = d[cols].corr(method=meth) if meth == 'pearson' else d[cols].rank().corr()
        c.insert(0, 'var', c.index)
        c.insert(0, 'method', meth)
        out.append(c)
    return pd.concat(out, ignore_index=True), len(d)


# ---------------------------------------------------------------- 分析二
def run_forecast(m, nd):
    res, chk, used = [], [], {}
    for scope, sub in SCOPES.items():
        base_dir = nd / 'rating_strict_n' / sub if sub else nd / 'rating_strict_n'
        w = pd.read_parquet(base_dir / 'forecast_panel.parquet')
        w = w.merge(m[['report_id'] + MEAS].rename(columns=SG), on='report_id', how='left')  # 面板里已有整张图的 LNPATH、PATHLEN，加前缀避免重名
        orig = pd.read_csv(base_dir / 'reg_results.csv')
        for y in YS_FC:  # 固定效应和控制变量与 reg_results.csv 一致
            for pan, (fe, ctrl) in F.panels(with_nofe=True).items():
                o0 = orig[(orig.Y == y) & (orig.panel == pan)].iloc[0]
                assert o0.controls == '+'.join(ctrl) and (str(o0.fe) if pd.notna(o0.fe) else '') == '+'.join(fe), (y, pan)
        r, used[scope] = F.regress_each(w, SG, YS_FC, return_ids=True)
        res.append(r.assign(scope=scope))
        c = F.regress_each(w, list(PANEL_COL.values()), YS_FC).assign(scope=scope)  # 复现核对
        o = orig[orig.reg == '单独'].set_index(['Y', 'panel', 'X'])[['coef', 't', 'n', 'n_clusters']].add_suffix('_orig')
        chk.append(c.join(o, on=['Y', 'panel', 'X']))
    cols = ['scope', 'Y', 'panel', 'X', 'coef', 'se', 't', 'p', 'n', 'n_clusters', 'fe', 'controls']
    res = pd.concat(res, ignore_index=True)[cols]
    chk = pd.concat(chk, ignore_index=True)[['scope', 'Y', 'panel', 'X', 'coef', 'coef_orig', 't', 't_orig', 'n', 'n_orig', 'n_clusters',
                                             'n_clusters_orig']]
    ok = (np.allclose(chk.coef, chk.coef_orig, rtol=1e-8, atol=1e-12) and np.allclose(chk.t, chk.t_orig, rtol=1e-8)
          and (chk.n == chk.n_orig).all() and (chk.n_clusters == chk.n_clusters_orig).all())
    assert ok, '复现 reg_results.csv 失败'
    # NSRC 缩尾值就是 R_RATIO，两者回归应完全相同
    a, b = res[res.X == 'NSRC'].reset_index(drop=True), chk[chk.X == 'R_RATIO'].reset_index(drop=True)
    assert np.allclose(a.coef, b.coef) and np.allclose(a.t, b.t)
    return res, chk, used


# ---------------------------------------------------------------- 分析三：新窗口
def new_car_windows():
    """用 src/report_car.py 的 market_matrices + window_car 算 CAR_0_40、CAR_0_60，并重算已有 7 个窗口核对。"""
    from src import report_car as RC
    from src import dag_outcomes_vars as V
    raw = pd.read_parquet(CAR_DIR / 'report_car_panel_raw.parquet')
    rep = pd.read_parquet(REPORT_PANEL)
    # 日行情区间是否覆盖 +60：读交易日历，看最晚事件日 + 60 是否 ≤ DAILY_END
    dates = np.array(sorted(set(pq.read_table(DAILY, columns=['date'], filters=[('date', '>=', RC.DAILY_START)]).column(0).to_pylist())))
    last_day0 = raw.day0.max()
    need = dates[np.searchsorted(dates, last_day0) + max(h for _, h in NEW_WIN.values())]
    msg = [f'日行情最晚日期 {dates[-1]}；最晚事件日 {last_day0}，其 +60 交易日为 {need}；report_car.DAILY_END = {RC.DAILY_END}']
    if need > RC.DAILY_END:
        RC.DAILY_END = dates[-1]
        msg.append(f'DAILY_END 不够，改为 {RC.DAILY_END}（只在本脚本内）')
    else:
        msg.append('DAILY_END 已覆盖 +60，沿用原区间（保证已有窗口逐篇可复现）')
    codes = sorted(rep.code.unique())
    ind = V.load_industry(INDUSTRY)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in rep.groupby('code').sw.first().items() if c not in sw_map})
    _, _, cal, mkt, M = RC.market_matrices(DAILY, codes, sw_map)
    cal_arr, mk = np.array(cal), mkt.to_numpy()
    p0 = np.searchsorted(cal_arr, raw.date.to_numpy(dtype=str), side='left')
    assert (p0 == raw.p0.to_numpy()).all(), '事件日位置与面板不一致'
    wins = {**RC.CAR_WIN, **NEW_WIN}
    recs = []
    for rid, code, i in zip(raw.report_id, raw.code, p0):
        rec = {'report_id': rid, 'code': code}
        for name, (lo, hi) in wins.items():
            rec[name], rec['susp_' + name] = RC.window_car(code, i, lo, hi, M, mk)
            if name in NEW_WIN:
                rec['incomplete_' + name] = bool(i + hi >= len(cal))
        recs.append(rec)
    c = pd.DataFrame(recs)
    for name in RC.CAR_WIN:
        assert np.allclose(c[name], raw[name], rtol=1e-12, atol=1e-14, equal_nan=True), name
    msg.append(f'已有 7 个窗口重算与 report_car_panel_raw 逐篇一致（{len(c)} 篇）')
    longs = []
    for name in NEW_WIN:
        miss = c[name].isna()
        msg.append(f'{name}：缺失 {int(miss.sum())} 篇（其中窗口超出日行情 {int(c["incomplete_" + name].sum())} 篇），'
                   f'窗口内有停牌日 {int((c["susp_" + name] > 0).sum())} 篇')
        longs.append(pd.DataFrame({'report_id': c.report_id, 'window': name, 'car_raw': c[name], 'car_w': R.winsor(c[name]),
                                   'n_susp': c['susp_' + name], 'incomplete': c['incomplete_' + name]}))
    return pd.concat(longs, ignore_index=True), msg


def run_car(m, car_long):
    w = pd.read_parquet(CAR_DIR / 'report_car_panel.parquet')
    for name in NEW_WIN:
        w = w.merge(car_long.loc[car_long.window == name, ['report_id', 'car_w']].rename(columns={'car_w': name}), on='report_id', how='left')
    d = w.merge(m[['report_id'] + MEAS].rename(columns=SG), on='report_id', how='inner')
    for y in Y_CAR_POST + Y_CAR_PRE:
        d[y] = d[y] * 100
    rows, used = [], set()
    for y in Y_CAR_POST + Y_CAR_PRE:
        for pan, fe in CAR_PANELS.items():
            ctrl = [c for c in CTRL_CAR if not (pan in ('B', 'C') and c in CTRL_FIRM) and not (y in Y_CAR_PRE and c == 'PRECAR')]
            for x in MEAS:
                res, s = R.fit(d, y, [SG[x]], ctrl, fe, 'code')
                r = res[0]
                used |= set(d.loc[s.index, 'report_id'])
                rows.append({'scope': 'all', 'Y': y, 'panel': pan, 'X': x, 'coef': r['coef'], 'se': r['se'], 't': r['t'], 'p': r['p'],
                             'n': r['n'], 'n_clusters': r['n_clusters'], 'n_singletons': r['n_singletons'], 'fe': '+'.join(fe),
                             'controls': '+'.join(ctrl)})
    return pd.DataFrame(rows), d, used


# ---------------------------------------------------------------- 主流程
def main(nd=ND, out=OUT):
    nd, out = Path(nd), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(nd / 'rating_strict_n' / 'forecast_panel_raw.parquet')
    graphs = D.load_graphs(list(raw.report_id), DAG_DIRS)
    m = D.five_measures(graphs, D.RATING_NODE, strict=True)
    print(f'样本：“投资评级”为严格汇点的研报 {len(m)} 篇、{raw.set_index("report_id").loc[m.report_id].code.nunique()} 家', flush=True)
    msgs, ck = check_measures(m, graphs, raw)
    for s in msgs:
        print('  核对：' + s, flush=True)
    m = m.merge(raw[['report_id', 'code']], on='report_id')
    m[['report_id', 'code', 'N_PATHS', 'N_EDGES'] + [x + '_RAW' for x in MEAS] + MEAS].to_parquet(out / 'measures.parquet', index=False)
    ck.to_csv(out / 'check_depth.csv', index=False, encoding='utf-8-sig')

    desc = describe(m)
    desc.to_csv(out / 'descriptives.csv', index=False, encoding='utf-8-sig')
    cr, n_cr = corr(m, raw)
    cr.to_csv(out / 'corr.csv', index=False, encoding='utf-8-sig')
    print(f'描述和相关系数：{n_cr} 篇', flush=True)

    res_fc, chk, used_fc = run_forecast(m, nd)
    res_fc.to_csv(out / 'reg_forecast.csv', index=False, encoding='utf-8-sig')
    chk.to_csv(out / 'check_baseline.csv', index=False, encoding='utf-8-sig')
    print(f'分析二：复现 reg_results.csv 的 {len(chk)} 个系数（系数、t、N、聚类数）全部一致', flush=True)
    for sc, u in used_fc.items():
        print(f'  [{sc}] 进入任一预测准确率回归：{len(u)} 篇、{m[m.report_id.isin(u)].code.nunique()} 家', flush=True)

    car_long, cmsg = new_car_windows()
    for s in cmsg:
        print('  CAR：' + s, flush=True)
    car_long.to_parquet(out / 'car_long.parquet', index=False)
    res_car, d_car, used_car = run_car(m, car_long)
    res_car.to_csv(out / 'reg_car.csv', index=False, encoding='utf-8-sig')
    print(f'分析三：606 篇里在市场反应面板中的 {len(d_car)} 篇、{d_car.code.nunique()} 家；进入任一回归 {len(used_car)} 篇、'
          f'{d_car[d_car.report_id.isin(used_car)].code.nunique()} 家', flush=True)
    print(f'完成 -> {out}', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='“投资评级”上游子图的五个结构指标')
    ap.add_argument('--input-dir', default=ND, help='读 <目录>/rating_strict_n/（及 main_only/）的预测面板和 reg_results.csv')
    ap.add_argument('--output', default=OUT, help='输出目录')
    a = ap.parse_args()
    main(a.input_dir, a.output)
