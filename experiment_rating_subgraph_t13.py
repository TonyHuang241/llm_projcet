"""评级子图五个指标与带方向的买入持有超额收益：照 Liu, Liu & Sui (2026) Table 13（Market Impact）的算法（不调用大模型，可反复运行）。

因变量  SignedCAR_i[a,b] = Sign_i × [ Π_{t=a..b}(1 + R_i,t) − Π_{t=a..b}(1 + R_bench,t) ] × 100（百分点）
        窗口 [0,1]、[0,2]、[0,3]（原文）及 [0,5]、[0,10]、[0,20]、[0,40]、[0,60]；安慰剂 [−10,−1]、[−5,−1]
        个股日收益口径同 src/report_car.py：停牌日个股收益按 0，上市前（窗口内任一天）记缺失
基准    ind_vw  主口径：申万一级、剔除本股、市值加权（report_car.market_matrices 的 M['ind']）
        mkt     全市场市值加权（原文口径）
        ind_ew  申万一级、剔除本股、等权（成分同 M['ind']：t[t.consec] 且有行业；行业 × 日收益和与只数各减本股后相除）
方向    fdir  主口径：report_fdir.DIR（FY2024 净利润预测高于 / 低于其他券商中位数，+1 / −1）
        frev  第二口径：report_frev.DIR（相对同一券商上一次预测上调 / 下调，+1 / −1）
样本    “投资评级”严格汇点的 606 篇 ∩ 市场反应面板（605 篇）∩ 方向非缺失；Y 在该样本内 1%/99% 缩尾
X       measures.parquet 里缩尾好的五个指标，估计样本内标准化，每次一个
设定    T13   公司 + 券商 + 月份固定效应；控制 SIZE_M、BM、MOM、VOL、ROA_T、LNAGE（rating_strict_n/forecast_panel.parquet）
        T13b  同上，不加月份固定效应
        PREV  固定效应同 T13；控制 LNCHARS_R、LNNODES_R、方向 × PRECAR（发布前窗口不放）、PREVOL、PRETURN、EA、NREP
        S0    无固定效应、无控制
        都按公司聚类

在仓库根目录运行：
    C:\\Users\\hung\\projects\\.venv\\Scripts\\python.exe -X utf8 experiment_rating_subgraph_t13.py
结果写到 output_dag/forecast_newdata/rating_subgraph/t13/：
    reg_t13.csv      长表（dir_version、bench、window、spec、X、coef、se、t、p、n、n_clusters …）
    t13_panel.parquet  每篇研报各基准 × 窗口的未乘方向买入持有超额收益（原始值，百分点）和两个方向
    check_bhar.csv   市场基准连乘版与相加版 CAR 的对照
"""
import hashlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
OUT_RS = ROOT_DIR / 'output_dag' / 'forecast_newdata' / 'rating_subgraph'
OUT = OUT_RS / 't13'
CAR_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
FC_PANEL = ROOT_DIR / 'output_dag' / 'forecast_newdata' / 'rating_strict_n' / 'forecast_panel.parquet'
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'
MEAS = ['NSRC', 'LNPATH', 'PATHLEN', 'NNODE', 'DEPTH']
SG = {x: 'SG_' + x for x in MEAS}
WINS = {'m10_m1': (-10, -1), 'm5_m1': (-5, -1), '0_1': (0, 1), '0_2': (0, 2), '0_3': (0, 3), '0_5': (0, 5), '0_10': (0, 10),
        '0_20': (0, 20), '0_40': (0, 40), '0_60': (0, 60)}
PRE = ('m10_m1', 'm5_m1')
BENCH = ['ind_vw', 'mkt', 'ind_ew']
DIRS = {'fdir': ('fdir', 'report_fdir.parquet'), 'frev': ('frev', 'report_frev.parquet')}
CTRL_T13 = ['SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']
CTRL_PREV = ['LNCHARS_R', 'LNNODES_R', 'PRECAR_S', 'PREVOL', 'PRETURN', 'EA', 'NREP']
FE3 = ['code', 'broker_g', 'month']
N_PROC = 32
# ==================================================


def md5_tree():
    root = ROOT_DIR / 'output_dag'
    skip = OUT.resolve()
    return {p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and skip not in p.resolve().parents}


def specs(win):
    prev = [c for c in CTRL_PREV if not (win in PRE and c == 'PRECAR_S')]
    return {'T13': (FE3, CTRL_T13), 'T13b': (['code', 'broker_g'], CTRL_T13), 'PREV': (FE3, prev), 'S0': ([], [])}


def bhar(raw):
    """790 篇 × 窗口 × 基准的买入持有超额收益（小数），以及相加版市场 CAR（核对用）。"""
    from src import report_car as RC
    from src import dag_outcomes_vars as V
    codes = sorted(raw.code.unique())
    ind = V.load_industry(INDUSTRY)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in raw.groupby('code').sw.first().items() if c not in sw_map})
    _, t, cal, mkt, M = RC.market_matrices(DAILY, codes, sw_map)
    # 行业等权（剔除本股）
    c = t[t.consec].copy()
    c['g'] = c.code.map(sw_map)
    c = c[c.g.notna()]
    gs = c.groupby(['g', 'date']).ret.agg(['sum', 'count'])
    own_r = c[c.code.isin(codes)].pivot(index='code', columns='date', values='ret').reindex(index=codes, columns=cal)
    ew = pd.DataFrame(np.nan, index=codes, columns=cal)
    for code in codes:
        g = sw_map.get(code)
        if g is None or g not in gs.index.get_level_values(0):
            continue
        x = gs.loc[g].reindex(cal).fillna(0.0)
        o = own_r.loc[code]
        den = x['count'] - o.notna().astype(float)
        ew.loc[code] = ((x['sum'] - o.fillna(0.0)) / den).where(den > 0)
    cal_arr = np.array(cal)
    p0 = np.searchsorted(cal_arr, raw.date.to_numpy(dtype=str), side='left')
    assert (p0 == raw.p0.to_numpy()).all() and (cal_arr[p0] == raw.day0.to_numpy()).all()
    assert (p0 - 10 >= 0).all() and (p0 + 60 < len(cal)).all(), '日行情不覆盖 [−10, +60]'
    taus = np.arange(-10, 61)
    TI = {int(v): j for j, v in enumerate(taus)}
    row = pd.Index(codes).get_indexer(raw.code)
    idx = p0[:, None] + taus[None, :]
    rr = row[:, None]
    tr = M['traded'].to_numpy(bool)[rr, idx]
    r = np.where(tr, M['ret'].to_numpy(float)[rr, idx], 0.0)
    r[~M['listed'].to_numpy(bool)[rr, idx]] = np.nan
    bm = {'ind_vw': M['ind'].to_numpy(float)[rr, idx], 'mkt': np.broadcast_to(mkt.to_numpy(float)[idx], r.shape),
          'ind_ew': ew.to_numpy(float)[rr, idx]}
    out = {}
    for w, (lo, hi) in WINS.items():
        sl = slice(TI[lo], TI[hi] + 1)
        pi = np.prod(1 + r[:, sl], axis=1)  # nan 传播
        for b, B in bm.items():
            out[(b, w)] = pi - np.prod(1 + B[:, sl], axis=1)
    add = {w: np.sum(r[:, slice(TI[lo], TI[hi] + 1)] - bm['mkt'][:, slice(TI[lo], TI[hi] + 1)], axis=1) for w, (lo, hi) in WINS.items()}
    # 自检：等权基准朴素重算
    rng = np.random.default_rng(20261122)
    for _ in range(5):
        code = codes[rng.integers(len(codes))]
        day = cal[rng.integers(len(cal) // 3, len(cal) * 2 // 3)]
        peers = c[(c.g == sw_map.get(code)) & (c.date == day) & (c.code != code)].ret
        assert np.isclose(peers.mean(), ew.loc[code, day], rtol=1e-12), (code, day)
    return out, add


def run_task(task):
    meta, df = task
    rows = []
    for spec, (fe, ctrl) in specs(meta['window']).items():
        for x in MEAS:
            res, s = R.fit(df, 'Y', [SG[x]], ctrl, fe, 'code')
            r = res[0] if res else {k: np.nan for k in ('coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons')}
            rows.append({**meta, 'spec': spec, 'X': x, 'coef': r['coef'], 'se': r['se'], 't': r['t'], 'p': r['p'], 'n': r['n'],
                         'n_clusters': r['n_clusters'], 'n_singletons': r['n_singletons'], 'n_sample': int(df.Y.notna().sum()),
                         'fe': '+'.join(fe), 'controls': '+'.join(ctrl)})
    return rows


def main():
    t0 = time.time()
    md5_before = md5_tree()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(CAR_DIR / 'report_car_panel_raw.parquet')
    w = pd.read_parquet(CAR_DIR / 'report_car_panel.parquet')
    assert list(raw.report_id) == list(w.report_id)
    bh, add = bhar(raw)
    print('  核对：行业等权基准朴素重算 5 个股票-日一致', flush=True)
    # 连乘 vs 相加（市场基准）
    ck = []
    for wname, col in (('0_1', 'CAR_0_1'), ('0_2', 'CAR_0_2'), ('0_5', 'CAR_0_5'), ('m5_m1', 'CAR_m5_m1'), ('m10_m1', 'CAR_m10_m1')):
        a, b = bh[('mkt', wname)], raw[col].to_numpy(float)
        assert np.allclose(add[wname], b, rtol=1e-10, atol=1e-13, equal_nan=True), col
        ok = np.isfinite(a) & np.isfinite(b)
        ck.append({'window': wname, 'car_col': col, 'n': int(ok.sum()), 'n_nan_mismatch': int((np.isfinite(a) != np.isfinite(b)).sum()),
                   'corr': float(np.corrcoef(a[ok], b[ok])[0, 1]), 'max_abs_diff_pp': float(np.max(np.abs(a[ok] - b[ok])) * 100),
                   'mean_abs_diff_pp': float(np.mean(np.abs(a[ok] - b[ok])) * 100), 'n_identical': int((a[ok] == b[ok]).sum())})
    ck = pd.DataFrame(ck)
    ck.to_csv(OUT / 'check_bhar.csv', index=False, encoding='utf-8-sig')
    print(ck.to_string(index=False), flush=True)

    m = pd.read_parquet(OUT_RS / 'measures.parquet')[['report_id'] + MEAS].rename(columns=SG)
    fc = pd.read_parquet(FC_PANEL)[['report_id'] + CTRL_T13]
    keep = w.report_id.isin(m.report_id).to_numpy()
    base = w.loc[keep, ['report_id', 'code', 'broker_g', 'month', 'LNCHARS_R', 'LNNODES_R', 'PRECAR', 'PREVOL', 'PRETURN', 'EA', 'NREP']]
    base = base.reset_index(drop=True).merge(m, on='report_id', how='left').merge(fc, on='report_id', how='left')
    assert len(base) == 605
    panel = base[['report_id', 'code']].copy()
    for dv, (sub, fn) in DIRS.items():
        panel[dv] = panel.report_id.map(pd.read_parquet(CAR_DIR / sub / fn).set_index('report_id').DIR).to_numpy(float)
    for (b, wn), v in bh.items():
        panel[f'BHAR_{b}_{wn}'] = v[keep] * 100
    panel.to_parquet(OUT / 't13_panel.parquet', index=False)

    tasks, samp = [], []
    for dv in DIRS:
        sgn = panel[dv].to_numpy(float)
        smp = np.isfinite(sgn)
        cnt = pd.Series(sgn).value_counts(dropna=False)
        samp.append(f'{dv}：+1 {int(cnt.get(1.0, 0))} 篇、−1 {int(cnt.get(-1.0, 0))} 篇、缺失 {int((~smp).sum())} 篇；'
                    f'样本 {int(smp.sum())} 篇、{base.code[smp].nunique()} 家')
        for b in BENCH:
            for wn in WINS:
                df = base.loc[smp].copy()
                df['PRECAR_S'] = sgn[smp] * df.PRECAR
                df['Y'] = R.winsor(pd.Series(sgn[smp] * panel.loc[smp, f'BHAR_{b}_{wn}'].to_numpy(), index=df.index))
                tasks.append(({'dir_version': dv, 'bench': b, 'window': wn}, df))
    for s in samp:
        print('样本 ' + s, flush=True)
    with Pool(min(N_PROC, len(tasks))) as pool:
        out = pool.map(run_task, tasks, chunksize=1)
    res = pd.DataFrame([r for rows in out for r in rows])
    res.to_csv(OUT / 'reg_t13.csv', index=False, encoding='utf-8-sig')
    print(f'回归 {len(res)} 个系数', flush=True)

    lab = {'m10_m1': '[−10,−1]', 'm5_m1': '[−5,−1]', **{k: f'[0,{k[2:]}]' for k in WINS if k.startswith('0_')}}
    for dv in DIRS:
        for b in BENCH:
            for spec in ('T13', 'T13b', 'PREV', 'S0'):
                q = res[(res.dir_version == dv) & (res.bench == b) & (res.spec == spec)]
                print(f'\n### {dv} {b} {spec}')
                print('| 窗口 | ' + ' | '.join(MEAS) + ' | N | 公司 |')
                for wn in WINS:
                    g = q[q.window == wn].set_index('X')
                    print(f'| {lab[wn]} | ' + ' | '.join(f'{g.loc[x].coef:.3f}{R.stars(g.loc[x].p)} ({g.loc[x].t:.2f})' for x in MEAS)
                          + f' | {int(g.n.iloc[0])} | {int(g.n_clusters.iloc[0])} |')
    md5_after = md5_tree()
    changed = [k for k in md5_before if md5_after.get(k) != md5_before[k]]
    added = sorted(set(md5_after) - set(md5_before))
    print(f'\nmd5 核对：output_dag/ 已有 {len(md5_before)} 个文件（不含 t13/）变化 {len(changed)} 个、新增 {len(added)} 个', flush=True)
    assert not changed and not added
    print(f'完成 -> {OUT}（{time.time() - t0:.0f} 秒）', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
