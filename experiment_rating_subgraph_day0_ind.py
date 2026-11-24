"""experiment_rating_subgraph_day0.py 的补充：不带方向、行业基准的累计超额收益（在第 0 天归零）（不调用大模型，可反复运行）。

基准（个股日收益、停牌日按 0、上市前缺失，口径同 src/report_car.py / src/report_car_event.py::daily_ar）
  ind_vw  主口径：申万一级行业、剔除本股的总市值加权收益 = report_car.market_matrices 的 M['ind']（CARI_* 用的就是它）
  ind_ew  第二口径：同一批股票（market_matrices 里连续两天有交易、有申万一级行业的股票，即 t[t.consec]）按行业 × 日期
          求日收益之和与只数，减去本股当天的收益和 1（本股当天不在组合里则不减），相除；剔除本股后只数为 0 记缺失
  mkt     对照：全市场市值加权收益（同上一轮，不带方向）
因变量  cum：τ > 0 为 Σ_{s=1..τ} AR(s)；τ < 0 为 −Σ_{s=τ+1..0} AR(s)；daily：AR(τ) − AR(0)（只存文件）。× 100（百分点）
τ      −10 … +20（不含 0）及 40、60
回归    同 experiment_rating_subgraph_day0.py：605 篇；Y 在该 τ 可用样本内 1%/99% 缩尾；X 估计样本内标准化；S0 / S1；按公司聚类

在仓库根目录运行：
    C:\\Users\\hung\\projects\\.venv\\Scripts\\python.exe -X utf8 experiment_rating_subgraph_day0_ind.py
结果写到 output_dag/forecast_newdata/rating_subgraph/day0_ind/：
    reg_day0_ind.csv    长表（bench、form、spec、tau、X、coef、se、t、p、n、n_clusters …）
    panel_long.parquet  605 篇 × τ（−10 … +60）的三种 AR 和累计值
"""
import hashlib
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from experiment_rating_subgraph_day0 import CAR_DIR, CTRL, DAILY, INDUSTRY, MEAS, OUT_RS, SG, TAUS, TI, T_LO, T_HI, run_task
from src import dag_outcomes_reg as R
from src.config import ROOT_DIR

# ==================== 运行参数 ====================
OUT = OUT_RS / 'day0_ind'
REG_TAUS = [t for t in range(-10, 21) if t != 0] + [40, 60]
BENCH = ['ind_vw', 'ind_ew', 'mkt']
N_PROC = 32
# ==================================================


def md5_tree():
    root = ROOT_DIR / 'output_dag'
    skip = OUT.resolve()
    return {p.relative_to(root).as_posix(): hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob('*')) if p.is_file() and skip not in p.resolve().parents}


def ar_matrices(raw):
    from src import report_car as RC
    from src import dag_outcomes_vars as V
    codes = sorted(raw.code.unique())
    ind = V.load_industry(INDUSTRY)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in raw.groupby('code').sw.first().items() if c not in sw_map})
    _, t, cal, mkt, M = RC.market_matrices(DAILY, codes, sw_map)
    # 等权行业收益（剔除本股）：同 market_matrices 的成分（t.consec、有行业），权重改为 1
    c = t[t.consec].copy()
    c['g'] = c.code.map(sw_map)
    c = c[c.g.notna()]
    gs = c.groupby(['g', 'date']).ret.agg(['sum', 'count'])
    own = c[c.code.isin(codes)]
    own_r = own.pivot(index='code', columns='date', values='ret').reindex(index=codes, columns=cal)
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
    assert (p0 == raw.p0.to_numpy()).all() and (p0 + T_LO >= 0).all() and (p0 + T_HI < len(cal)).all()
    row = pd.Index(codes).get_indexer(raw.code)
    idx = p0[:, None] + np.array(TAUS)[None, :]
    rr = row[:, None]
    tr = M['traded'].to_numpy(bool)[rr, idx]
    r = np.where(tr, M['ret'].to_numpy(float)[rr, idx], 0.0)
    lst = M['listed'].to_numpy(bool)[rr, idx]
    out = {}
    for name, b in (('ind_vw', M['ind'].to_numpy(float)[rr, idx]), ('ind_ew', ew.to_numpy(float)[rr, idx]),
                    ('mkt', mkt.to_numpy(float)[idx])):
        a = (r - b) * 100
        a[~lst] = np.nan
        out[name] = a
    # 自检：等权行业收益的朴素重算（抽 5 个股票-日）
    rng = np.random.default_rng(20261122)
    for _ in range(5):
        code = codes[rng.integers(len(codes))]
        day = cal[rng.integers(len(cal) // 3, len(cal) * 2 // 3)]
        peers = c[(c.g == sw_map.get(code)) & (c.date == day) & (c.code != code)].ret
        assert np.isclose(peers.mean(), ew.loc[code, day], rtol=1e-12), (code, day)
    return out


def cum(ar):
    Y = np.full_like(ar, np.nan)
    Y[:, TI[0]] = 0.0
    Y[:, TI[1]:] = np.cumsum(ar[:, TI[1]:], axis=1)
    pre = np.cumsum(ar[:, TI[T_LO + 1]:TI[0] + 1][:, ::-1], axis=1)
    for k in range(pre.shape[1]):
        Y[:, TI[-k - 1]] = -pre[:, k]
    return Y


def main():
    t0 = time.time()
    md5_before = md5_tree()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(CAR_DIR / 'report_car_panel_raw.parquet')
    w = pd.read_parquet(CAR_DIR / 'report_car_panel.parquet')
    assert list(raw.report_id) == list(w.report_id)
    A = ar_matrices(raw)
    print('  核对：等权行业基准朴素重算 5 个股票-日一致', flush=True)
    for name, (lo, hi) in {'CARI_0_2': (0, 2), 'CARI_0_5': (0, 5)}.items():
        s = A['ind_vw'][:, TI[lo]:TI[hi] + 1].sum(axis=1) / 100
        assert np.allclose(s, raw[name].to_numpy(float), rtol=1e-10, atol=1e-13, equal_nan=True), name
        print(f'  核对：主口径 AR_ind 的 [{lo},{hi}] 累计与 report_car_panel_raw.{name} 逐篇一致（{len(raw)} 篇，'
              f'缺失 {int(np.isnan(s).sum())} 篇双方一致）', flush=True)
    for name, (lo, hi) in {'CAR_0_2': (0, 2), 'CAR_0_5': (0, 5)}.items():
        s = A['mkt'][:, TI[lo]:TI[hi] + 1].sum(axis=1) / 100
        assert np.allclose(s, raw[name].to_numpy(float), rtol=1e-10, atol=1e-13, equal_nan=True), name
    print('  核对：全市场基准 AR 的 [0,2]、[0,5] 累计与 CAR_0_2、CAR_0_5 一致', flush=True)

    m = pd.read_parquet(OUT_RS / 'measures.parquet')[['report_id'] + MEAS].rename(columns=SG)
    keep = w.report_id.isin(m.report_id).to_numpy()
    base = w.loc[keep, ['report_id', 'code', 'broker_g', 'month'] + CTRL].reset_index(drop=True).merge(m, on='report_id', how='left')
    assert len(base) == 605
    print(f'样本：{len(base)} 篇、{base.code.nunique()} 家', flush=True)
    long = pd.DataFrame({'report_id': np.repeat(base.report_id.to_numpy(), len(TAUS)), 'tau': np.tile(TAUS, len(base))})
    tasks = []
    for b in BENCH:
        ar = A[b][keep]
        Yc = cum(ar)
        long[f'AR_{b}'] = ar.ravel()
        long[f'CUM_{b}'] = Yc.ravel()
        for form in ('cum', 'daily'):
            for t in REG_TAUS:
                D = Yc[:, TI[t]] if form == 'cum' else ar[:, TI[t]] - ar[:, TI[0]]
                df = base[['report_id', 'code', 'broker_g', 'month'] + CTRL + list(SG.values())].copy()
                df['D'] = R.winsor(pd.Series(D, index=df.index))
                tasks.append(({'bench': b, 'form': form, 'tau': t}, df))
    long.to_parquet(OUT / 'panel_long.parquet', index=False)
    with Pool(min(N_PROC, len(tasks))) as pool:
        out = pool.map(run_task, tasks, chunksize=2)
    res = pd.DataFrame([r for rows in out for r in rows])
    res = res[['bench', 'form', 'spec', 'tau', 'X', 'coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons', 'n_avail', 'firms_avail']]
    res.to_csv(OUT / 'reg_day0_ind.csv', index=False, encoding='utf-8-sig')
    print(f'回归 {len(res)} 个系数', flush=True)

    for b in BENCH:
        for spec in ('S1', 'S0'):
            q = res[(res.bench == b) & (res.form == 'cum') & (res.spec == spec)]
            print(f'\n### {b} {spec} cum')
            print('| τ | ' + ' | '.join(MEAS) + ' | N | 公司 | 可用 N/公司 |')
            for t in REG_TAUS:
                g = q[q.tau == t].set_index('X')
                print(f'| {t} | ' + ' | '.join(f'{g.loc[x].coef:.3f}{R.stars(g.loc[x].p)} ({g.loc[x].t:.2f})' for x in MEAS)
                      + f' | {int(g.n.iloc[0])} | {int(g.n_clusters.iloc[0])} | {int(g.n_avail.iloc[0])}/{int(g.firms_avail.iloc[0])} |')
    md5_after = md5_tree()
    changed = [k for k in md5_before if md5_after.get(k) != md5_before[k]]
    added = sorted(set(md5_after) - set(md5_before))
    print(f'\nmd5 核对：output_dag/ 已有 {len(md5_before)} 个文件（不含 day0_ind/）变化 {len(changed)} 个、新增 {len(added)} 个', flush=True)
    assert not changed and not added
    print(f'完成 -> {OUT}（{time.time() - t0:.0f} 秒）', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
