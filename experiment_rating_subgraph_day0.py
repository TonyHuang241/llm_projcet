"""评级子图五个指标的事件研究重跑：以发布当天（τ = 0）为基准的事件时间哑变量设定（不调用大模型，可反复运行）。

模型  Y(i,τ) = 研报固定效应 + Σ_{τ≠0} [ β_τ · X_i × 1(τ) + 固定效应和控制变量 × 1(τ) ] + ε，τ = 0 为基准。
      全交互模型等价于对每个 τ ≠ 0 把 D(i,τ) = Y(i,τ) − Y(i,0) 对 X_i 做截面回归（平衡样本下系数完全相同），本脚本按差分写法估计，
      并用堆叠数据 + 日哑变量 + 研报固定效应的写法直接估一遍核对（check_stack.csv）。
样本  “投资评级”是严格汇点的 606 篇 ∩ 市场反应面板（605 篇）；带方向的收益再要求方向非缺失。
因变量（τ = 相对事件日 day0 / p0 的交易日序号）
  SAR   (1a) 方向 × AR(i,τ)，AR = 个股日收益 − 全市场市值加权日收益（口径同 src/report_car_event.py::daily_ar：停牌日个股收益按 0，
        上市前记缺失），× 100（百分点）。D = Y(τ) − Y(0)
  SCAR  (1b) 累计、在第 0 天归零：τ > 0 为 方向 × Σ_{s=1..τ} AR(s)；τ < 0 为 −方向 × Σ_{s=τ+1..0} AR(s)；直接对 X 回归
  方向  dir24 = report_fdir.DIR（主口径），dir25 = 2·HIGH_25 − 1
  TURN  第 τ 天换手率（%），停牌日缺失（report_car.market_matrices 的 M['turn']、M['traded']）。D = Y(τ) − Y(0)
  RV1 / RV5  第 τ 天 1 / 5 分钟已实现波动率（百分点，turn_vol/rv_daily.parquet），τ 那天不在 2024 年或无 RV 则缺失。D = Y(τ) − Y(0)
形式  daily：τ = −10 … +20（不含 0）
      bin：[−10,−6]、[−5,−1]、[1,2]、[3,5]、[6,10]、[11,20]、[21,40]、[41,60]；逐日型取段内均值减第 0 天（段内每一天和第 0 天都要可用），
           累计型取段末那天的值（发布前两段取段首 τ = −10、−5）
回归  每个 τ / 段：D 在可用样本内 1%/99% 缩尾；X = rating_subgraph/measures.parquet 里缩尾好的五个指标之一，估计样本内标准化；
      S0 无固定效应无控制；S1 公司 + 券商 + 发布月份固定效应，控制 LNCHARS_R、LNNODES_R、PREVOL、PRETURN、EA、NREP；按公司聚类。

在仓库根目录运行：
    C:\\Users\\hung\\projects\\.venv\\Scripts\\python.exe -X utf8 experiment_rating_subgraph_day0.py
结果写到 output_dag/forecast_newdata/rating_subgraph/day0/：
    reg_day0.csv        长表（ytype、dir_version、spec、form、tau、bin、X、coef、se、t、p、n、n_clusters …）
    panel_long.parquet  605 篇 × τ（−10 … +60）的各因变量原始值
    check_stack.csv     堆叠写法与差分写法的逐 τ 对照
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
OUT = OUT_RS / 'day0'
CAR_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
FDIR = CAR_DIR / 'fdir' / 'report_fdir.parquet'
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'
MIN_FIRST, MIN_LAST = '20240102', '20241231'
MEAS = ['NSRC', 'LNPATH', 'PATHLEN', 'NNODE', 'DEPTH']
SG = {x: 'SG_' + x for x in MEAS}
CTRL = ['LNCHARS_R', 'LNNODES_R', 'PREVOL', 'PRETURN', 'EA', 'NREP']
SPECS = {'S0': ([], []), 'S1': (['code', 'broker_g', 'month'], CTRL)}
T_LO, T_HI = -10, 60  # 面板的 τ 范围
DAILY_TAUS = [t for t in range(-10, 21) if t != 0]
BINS = [(-10, -6), (-5, -1), (1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 60)]
SERIES = [('SAR', 'dir24'), ('SAR', 'dir25'), ('SCAR', 'dir24'), ('SCAR', 'dir25'), ('TURN', 'none'), ('RV1', 'none'), ('RV5', 'none')]
CUM_TYPES = {'SCAR'}
N_PROC = 32
TV_WINS = {'m10_m1': (-10, -1), 'm5_m1': (-5, -1), '0_1': (0, 1), '0_2': (0, 2), '0_5': (0, 5), '0_10': (0, 10), '0_20': (0, 20),
           '0_40': (0, 40), '0_60': (0, 60)}
CAR_CHECK = {'CAR_0_1': (0, 1), 'CAR_0_2': (0, 2), 'CAR_0_5': (0, 5), 'CAR_0_10': (0, 10), 'CAR_0_20': (0, 20),
             'CAR_m10_m1': (-10, -1), 'CAR_m5_m1': (-5, -1)}
CHECK_SERIES = [('SAR', 'dir24'), ('TURN', 'none')]  # 堆叠核对用的因变量
# ==================================================
TAUS = list(range(T_LO, T_HI + 1))
TI = {t: j for j, t in enumerate(TAUS)}


def bin_name(b):
    return f'[{b[0]},{b[1]}]'


def md5_tree():
    root = ROOT_DIR / 'output_dag'
    skip = OUT.resolve()
    out = {}
    for p in sorted(root.rglob('*')):
        if p.is_file() and skip not in p.resolve().parents:
            out[p.relative_to(root).as_posix()] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


# ---------------------------------------------------------------- 研报 × τ 的原始值
def build_matrices(raw):
    """790 篇 × τ（−10 … +60）：AR（百分点）、换手率、RV1、RV5、该日是否在 2024 年。"""
    from src import report_car as RC
    from src import dag_outcomes_vars as V
    codes = sorted(raw.code.unique())
    ind = V.load_industry(INDUSTRY)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in raw.groupby('code').sw.first().items() if c not in sw_map})
    _, _, cal, mkt, M = RC.market_matrices(DAILY, codes, sw_map)
    cal_arr = np.array(cal)
    p0 = np.searchsorted(cal_arr, raw.date.to_numpy(dtype=str), side='left')
    assert (p0 == raw.p0.to_numpy()).all() and (cal_arr[p0] == raw.day0.to_numpy()).all(), '事件日与面板不一致'
    assert (p0 + T_LO >= 0).all() and (p0 + T_HI < len(cal)).all(), f'日行情不覆盖 τ = {T_LO} … {T_HI}（DAILY_END = {RC.DAILY_END}）'
    row = pd.Index(codes).get_indexer(raw.code)
    idx = p0[:, None] + np.array(TAUS)[None, :]
    rr = row[:, None]
    ret, traded = M['ret'].to_numpy(float), M['traded'].to_numpy(bool)
    listed, turn = M['listed'].to_numpy(bool), M['turn'].to_numpy(float)
    mk = mkt.to_numpy(float)
    tr = traded[rr, idx]
    ar = np.where(tr, ret[rr, idx], 0.0) - mk[idx]
    ar[~listed[rr, idx]] = np.nan
    ar *= 100
    tv = np.where(tr, turn[rr, idx], np.nan)
    rv = pd.read_parquet(OUT_RS / 'turn_vol' / 'rv_daily.parquet')
    in24 = (cal_arr[idx] >= MIN_FIRST) & (cal_arr[idx] <= MIN_LAST)
    rvm = {}
    for k in ('RV1', 'RV5'):
        pv = rv.pivot(index='code', columns='date', values=k).reindex(index=codes, columns=cal).to_numpy(float)
        v = pv[rr, idx]
        v[~in24] = np.nan
        rvm[k] = v
    return {'AR': ar, 'TURN': tv, 'RV1': rvm['RV1'], 'RV5': rvm['RV5'], 'in24': in24, 'traded': tr}


def check_raw(raw, A):
    msgs = []
    # 逐日 AR vs pathlen_event_panel_raw
    pe = pd.read_parquet(CAR_DIR / 'pathlen_event' / 'pathlen_event_panel_raw.parquet').set_index('report_id').loc[raw.report_id]
    tags = {t: f'AR_{"m" + str(-t) if t < 0 else t}' for t in range(-10, 11)}
    mine = A['AR'][:, [TI[t] for t in range(-10, 11)]]
    theirs = pe[[tags[t] for t in range(-10, 11)]].to_numpy(float)
    assert np.allclose(mine, theirs, rtol=1e-10, atol=1e-12, equal_nan=True), '逐日 AR 与 pathlen_event_panel_raw 不一致'
    msgs.append(f'逐日 AR（τ = −10 … +10）与 pathlen_event_panel_raw.parquet 的 AR_m10 … AR_10 逐篇一致（{len(raw)} 篇，'
                f'非缺失 {int(np.isfinite(theirs).sum())} 个、双方同为缺失 {int(np.isnan(theirs).sum())} 个，最大差 {np.nanmax(np.abs(mine - theirs)):.1e}）')
    # 窗口累计 vs report_car_panel_raw
    for name, (lo, hi) in CAR_CHECK.items():
        s = A['AR'][:, TI[lo]:TI[hi] + 1].sum(axis=1) / 100  # 有缺失则 nan
        assert np.allclose(s, raw[name].to_numpy(float), rtol=1e-10, atol=1e-13, equal_nan=True), name
    msgs.append(f'AR 的窗口和与 report_car_panel_raw 的 {"、".join(CAR_CHECK)} 逐篇一致（{len(raw)} 篇）')
    # 窗口换手率、RV vs turnvol_panel
    tvp = pd.read_parquet(OUT_RS / 'turn_vol' / 'turnvol_panel.parquet').set_index('report_id').loc[raw.report_id]
    for s, (lo, hi) in TV_WINS.items():
        sl = slice(TI[lo], TI[hi] + 1)
        t = A['TURN'][:, sl]
        with np.errstate(all='ignore'):
            m = np.where(np.isfinite(t).any(axis=1), np.nanmean(np.where(np.isfinite(t), t, np.nan), axis=1), np.nan)
        assert np.allclose(m, tvp[f'TURN_{s}_RAW'].to_numpy(float), rtol=1e-12, atol=0, equal_nan=True), s
        inside = A['in24'][:, sl].all(axis=1)
        for k in ('RV1', 'RV5'):
            v = A[k][:, sl]
            with np.errstate(all='ignore'):
                mv = np.where(inside & np.isfinite(v).any(axis=1), np.nanmean(v, axis=1), np.nan)
            assert np.allclose(mv, tvp[f'{k}_{s}_RAW'].to_numpy(float), rtol=1e-12, atol=0, equal_nan=True), (k, s)
    msgs.append(f'窗口内日均换手率与 turnvol_panel 的 TURN_*_RAW（9 个窗口）逐篇一致；窗口 RV1、RV5 均值也与 RV1_*_RAW、RV5_*_RAW 一致（{len(raw)} 篇）')
    return msgs


# ---------------------------------------------------------------- 因变量
def series_matrix(ytype, ver, A, dirv):
    """一个因变量在 605 篇上的 研报 × τ 矩阵（方向缺失的整行 nan）。"""
    if ytype == 'SAR':
        return dirv[:, None] * A['AR']
    if ytype == 'SCAR':
        ar = A['AR']
        Y = np.full_like(ar, np.nan)
        Y[:, TI[0]] = 0.0
        post = np.cumsum(ar[:, TI[1]:], axis=1)  # nan 向后传播
        Y[:, TI[1]:] = post
        pre = np.cumsum(ar[:, TI[T_LO + 1]:TI[0] + 1][:, ::-1], axis=1)  # 列 = AR(0), AR(0)+AR(−1), …
        for k in range(pre.shape[1]):  # pre[:, k] = Σ_{s=−k..0} AR(s)，对应 τ = −k−1
            Y[:, TI[-k - 1]] = -pre[:, k]
        Y[:, TI[0]] = 0.0
        return dirv[:, None] * Y
    return A[ytype].copy()


def dep_vars(ytype, Y):
    """返回 [(form, tau, bin, D 向量（未缩尾）)]。"""
    out = []
    cum = ytype in CUM_TYPES
    y0 = Y[:, TI[0]]
    for t in DAILY_TAUS:
        out.append(('daily', t, '', Y[:, TI[t]] if cum else Y[:, TI[t]] - y0))
    for b in BINS:
        if cum:
            v = Y[:, TI[b[0]]] if b[1] < 0 else Y[:, TI[b[1]]]
        else:
            v = Y[:, TI[b[0]]:TI[b[1]] + 1].mean(axis=1) - y0  # 段内任一天缺失即 nan
        out.append(('bin', np.nan, bin_name(b), v))
    return out


def run_task(task):
    meta, df = task
    rows = []
    for spec, (fe, ctrl) in SPECS.items():
        for x in MEAS:
            res, s = R.fit(df, 'D', [SG[x]], ctrl, fe, 'code')
            r = res[0] if res else {k: np.nan for k in ('coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons')}
            rows.append({**meta, 'spec': spec, 'X': x, 'coef': r['coef'], 'se': r['se'], 't': r['t'], 'p': r['p'], 'n': r['n'],
                         'n_clusters': r['n_clusters'], 'n_singletons': r['n_singletons'], 'n_avail': int(df.D.notna().sum()),
                         'firms_avail': int(df.loc[df.D.notna(), 'code'].nunique())})
    return rows


# ---------------------------------------------------------------- 堆叠写法核对
def stack_check(base, Y, label, balanced):
    """S0、未缩尾：堆叠（研报虚拟变量 + τ 虚拟变量 + Z×1(τ)，按公司聚类）vs 逐 τ 差分回归。"""
    taus = DAILY_TAUS
    yy = Y[:, [TI[t] for t in [0] + taus]]
    ok0 = np.isfinite(yy[:, 0])
    if balanced:
        keep = np.isfinite(yy).all(axis=1)
    else:
        keep = ok0 & np.isfinite(yy[:, 1:]).any(axis=1)
    b = base[keep].reset_index(drop=True)
    yy = yy[keep]
    n = len(b)
    rows = []
    for x in MEAS:
        xr = b[SG[x]].to_numpy(float)
        recs, Zs = [], {}
        for j, t in enumerate(taus, start=1):
            av = np.isfinite(yy[:, j])  # 第 0 天都可用
            z = np.full(n, np.nan)
            z[av] = (xr[av] - xr[av].mean()) / xr[av].std(ddof=1)
            Zs[t] = z
            df = pd.DataFrame({'D': yy[:, j] - yy[:, 0], SG[x]: xr, 'code': b.code})
            r = R.fit(df, 'D', [SG[x]], [], [], 'code')[0][0]
            recs.append(r)
        # 堆叠
        ii, tt, yv = [], [], []
        for k, t in enumerate([0] + taus):
            av = np.isfinite(yy[:, k])
            ii.append(np.nonzero(av)[0]); tt.append(np.full(av.sum(), t)); yv.append(yy[av, k])
        ii, tt, yv = np.concatenate(ii), np.concatenate(tt), np.concatenate(yv)
        N = len(yv)
        Xr = np.zeros((N, n)); Xr[np.arange(N), ii] = 1.0
        Xt = np.zeros((N, len(taus))); Xz = np.zeros((N, len(taus)))
        for j, t in enumerate(taus):
            m = tt == t
            Xt[m, j] = 1.0
            Xz[m, j] = Zs[t][ii[m]]
        X = np.column_stack([Xr, Xt, Xz])
        o = R.ols(yv, X, b.code.to_numpy()[ii])
        for j, t in enumerate(taus):
            bs, ses = o['b'][n + len(taus) + j], o['se_cl'][n + len(taus) + j]
            rows.append({'series': label, 'sample': 'balanced' if balanced else 'unbalanced', 'X': x, 'tau': t,
                         'coef_stack': bs, 'coef_diff': recs[j]['coef'], 'abs_diff': abs(bs - recs[j]['coef']),
                         'se_stack': ses, 'se_diff': recs[j]['se'], 't_stack': bs / ses, 't_diff': recs[j]['t'],
                         'n_reports': n, 'n_obs_stack': N, 'n_diff': recs[j]['n'], 'n_clusters': o['G']})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 打印表格
def cell(r):
    return f'{r.coef:.3f}{R.stars(r.p)} ({r.t:.2f})'


def print_tables(res):
    lab = {'SAR': '1a 带方向逐日超额收益（dir24）', 'SCAR': '1b 带方向累计超额收益、第 0 天归零（dir24）', 'TURN': '2 日换手率（%）',
           'RV1': '3 1 分钟已实现波动率 RV1'}
    for form in ('bin', 'daily'):
        for yt in ('SAR', 'SCAR', 'TURN', 'RV1'):
            ver = 'dir24' if yt in ('SAR', 'SCAR') else 'none'
            q = res[(res.ytype == yt) & (res.dir_version == ver) & (res.spec == 'S1') & (res.form == form)]
            key = 'bin' if form == 'bin' else 'tau'
            order = [bin_name(b) for b in BINS] if form == 'bin' else DAILY_TAUS
            print(f'\n### S1 {form}：{lab[yt]}')
            print('| ' + ('段' if form == 'bin' else 'τ') + ' | ' + ' | '.join(MEAS) + ' | N | 公司 |')
            print('|' + '---|' * (len(MEAS) + 3))
            for k in order:
                g = q[q[key] == k].set_index('X')
                print(f'| {k} | ' + ' | '.join(cell(g.loc[x]) for x in MEAS) + f' | {int(g.n.iloc[0])} | {int(g.n_clusters.iloc[0])} |')


def print_samples(res):
    print('\n### 样本量（n_avail/firms_avail = 可用样本 = S0 的 N；S1 = 加控制并剔除 singleton 后）')
    q = res[(res.X == 'NSRC')]
    pts = [('daily', t) for t in (-10, -1, 1, 10, 20)] + [('bin', bin_name(b)) for b in BINS]
    print('| 因变量 | 方向 | ' + ' | '.join(str(p[1]) for p in pts) + ' |')
    print('|' + '---|' * (len(pts) + 2))
    for yt, ver in SERIES:
        cells = []
        for form, k in pts:
            g = q[(q.ytype == yt) & (q.dir_version == ver) & (q.form == form) & ((q.tau == k) if form == 'daily' else (q.bin == k))]
            s0, s1 = g[g.spec == 'S0'].iloc[0], g[g.spec == 'S1'].iloc[0]
            cells.append(f'{int(s0.n_avail)}/{int(s0.firms_avail)}; S1 {int(s1.n)}/{int(s1.n_clusters)}')
        print(f'| {yt} | {ver} | ' + ' | '.join(cells) + ' |')


# ---------------------------------------------------------------- 主流程
def main():
    t0 = time.time()
    md5_before = md5_tree()
    print(f'output_dag/ 已有文件 {len(md5_before)} 个（不含 day0/），已记 md5', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_parquet(CAR_DIR / 'report_car_panel_raw.parquet')
    w = pd.read_parquet(CAR_DIR / 'report_car_panel.parquet')
    assert list(raw.report_id) == list(w.report_id)
    A = build_matrices(raw)
    print(f'行情矩阵：{time.time() - t0:.0f} 秒', flush=True)
    for s in check_raw(raw, A):
        print('  核对：' + s, flush=True)

    m = pd.read_parquet(OUT_RS / 'measures.parquet')[['report_id'] + MEAS].rename(columns=SG)
    f = pd.read_parquet(FDIR).set_index('report_id')
    keep = w.report_id.isin(m.report_id).to_numpy()
    base = w.loc[keep, ['report_id', 'code', 'broker_g', 'month'] + CTRL].reset_index(drop=True).merge(m, on='report_id', how='left')
    assert len(base) == 605 and base[list(SG.values())].notna().all().all()
    base['dir24'] = base.report_id.map(f.DIR).to_numpy(float)
    base['dir25'] = base.report_id.map(2 * f.HIGH_25 - 1).to_numpy(float)
    A5 = {k: v[keep] for k, v in A.items()}
    print(f'样本：{len(base)} 篇、{base.code.nunique()} 家；dir24 非缺失 {int(base.dir24.notna().sum())} 篇，'
          f'dir25 非缺失 {int(base.dir25.notna().sum())} 篇', flush=True)

    # 面板长表
    mats = {}
    for yt, ver in SERIES:
        dv = base[ver].to_numpy(float) if ver != 'none' else None
        mats[(yt, ver)] = series_matrix(yt, ver, A5, dv)
    long = pd.DataFrame({'report_id': np.repeat(base.report_id.to_numpy(), len(TAUS)), 'code': np.repeat(base.code.to_numpy(), len(TAUS)),
                         'tau': np.tile(TAUS, len(base)), 'dir24': np.repeat(base.dir24.to_numpy(), len(TAUS)),
                         'dir25': np.repeat(base.dir25.to_numpy(), len(TAUS)), 'AR': A5['AR'].ravel(), 'traded': A5['traded'].ravel(),
                         'in2024': A5['in24'].ravel()})
    for (yt, ver), Y in mats.items():
        long[yt if ver == 'none' else f'{yt}_{ver}'] = Y.ravel()
    long.to_parquet(OUT / 'panel_long.parquet', index=False)

    # 回归任务
    tasks = []
    for (yt, ver), Y in mats.items():
        smp = base[ver].notna().to_numpy() if ver != 'none' else np.ones(len(base), bool)
        for form, tau, bn, D in dep_vars(yt, Y):
            df = base.loc[smp, ['report_id', 'code', 'broker_g', 'month'] + CTRL + list(SG.values())].copy()
            df['D'] = R.winsor(pd.Series(D[smp], index=df.index))
            tasks.append(({'ytype': yt, 'dir_version': ver, 'form': form, 'tau': tau, 'bin': bn}, df))
    t1 = time.time()
    with Pool(min(N_PROC, len(tasks))) as pool:
        out = pool.map(run_task, tasks, chunksize=2)
    res = pd.DataFrame([r for rows in out for r in rows])
    cols = ['ytype', 'dir_version', 'spec', 'form', 'tau', 'bin', 'X', 'coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons',
            'n_avail', 'firms_avail']
    res = res[cols]
    res.to_csv(OUT / 'reg_day0.csv', index=False, encoding='utf-8-sig')
    print(f'回归：{len(tasks)} 个 (因变量, τ/段) × 2 设定 × 5 指标 = {len(res)} 个系数，{time.time() - t1:.0f} 秒', flush=True)
    assert res.coef.notna().all()

    # 堆叠核对
    t2 = time.time()
    cks = []
    for yt, ver in CHECK_SERIES:
        smp = base[ver].notna().to_numpy() if ver != 'none' else np.ones(len(base), bool)
        for bal in (True, False):
            cks.append(stack_check(base[smp].reset_index(drop=True), mats[(yt, ver)][smp], f'{yt}_{ver}', bal))
    ck = pd.concat(cks, ignore_index=True)
    ck.to_csv(OUT / 'check_stack.csv', index=False, encoding='utf-8-sig')
    for (se, sa), g in ck.groupby(['series', 'sample'], sort=False):
        print(f'  堆叠核对 [{se}, {sa}]：{g.n_reports.iloc[0]} 篇、堆叠 {g.n_obs_stack.iloc[0]} 个观测；'
              f'系数最大绝对差 {g.abs_diff.max():.2e}（系数绝对值中位数 {g.coef_diff.abs().median():.3f}）；'
              f'聚类 se 比值（堆叠/差分）{(g.se_stack / g.se_diff).min():.3f}–{(g.se_stack / g.se_diff).max():.3f}', flush=True)
    bal = ck[ck['sample'] == 'balanced']
    assert (bal.abs_diff < 1e-8).all(), '平衡样本下堆叠与差分系数不一致'
    print(f'  堆叠核对耗时 {time.time() - t2:.0f} 秒', flush=True)

    print_samples(res)
    print_tables(res)

    md5_after = md5_tree()
    changed = [k for k in md5_before if md5_after.get(k) != md5_before[k]]
    added = sorted(set(md5_after) - set(md5_before))
    print(f'\nmd5 核对：output_dag/ 已有 {len(md5_before)} 个文件（不含 day0/）变化 {len(changed)} 个、新增 {len(added)} 个', flush=True)
    assert not changed, changed
    print(f'完成 -> {OUT}（总耗时 {time.time() - t0:.0f} 秒）', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
