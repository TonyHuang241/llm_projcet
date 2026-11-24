"""实验：研报概念 DAG 的结构指标（PATHLEN、LNPATH、RATIO）与研报前后分日超额收益（事件研究的分日展开）。由根目录 experiment_dag_event.py 调用。
沿用 src/report_car.py 的全部口径（事件日 0、停牌、个股日收益、市场收益、行业收益、1%/99% 缩尾），只是把窗口 CAR 拆成逐日 AR：
  基准        market：沪深 A 股市值加权市场收益（同上一轮 CAR_*）；
              industry：所在申万一级行业收益，剔除本股（同上一轮 CARI_*，即 RC.market_matrices 的 M['ind']）
  AR_τ        第 τ 天个股收益 − 基准收益（停牌日个股收益按 0；上市前或无法计算记缺失），τ = −10..+10，乘以 100（百分点）
  窗口 CAR    CAR[−10,−1]、CAR[0,10]、CAR[1,10]、CAR[0,1]（行业版另加 CAR[0,2]、CAR[0,5]）都是对应 AR 之和；CUM_τ = Σ_{s=−10..τ} AR_s
  平衡样本    上一轮 790 篇（已剔除第 0 天停牌）里 21 个 AR 都非缺失的研报；窗口和分日回归都用它
  缩尾        每个因变量在平衡样本内按 1%/99% 缩尾（R.winsor）；控制变量直接用上一轮缩尾后的值（上一轮在 790 篇上缩尾）；
              解释变量取上一轮面板的值（来自 output_dag/regression/report_panel.parquet，已在 791 篇研报层面按 1%/99% 缩尾），不再缩尾
  回归        y ~ X（只放一个，估计样本内标准化）+ 固定效应 + 控制变量，按公司聚类；另算公司 × 事件日双向聚类的 t 值
  规格        none/ind/firm（表 1、2）、同样本对照 none_s/ind_s、全部控制 ind_c/firm_c，以及只控制研报字数和概念数的
              none_k/ind_k/firm_k（表 3、4；控制变量不标准化，系数另存在结果 csv 的 b_/t_/p_LNCHARS_R、LNNODES_R 列），
              以及 none_sk（无固定效应、样本同 firm、控制字数和概念数）；表 5（none_sk）、表 6（firm_k）是六个窗口各一列
"""
import time

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src import dag_outcomes_vars as V
from src import report_car as RC

# ---------------------------------------------------------------- 参数
TAUS = list(range(-10, 11))
WIN = {'CAR_m10_m1': (-10, -1), 'CAR_0_10': (0, 10), 'CAR_1_10': (1, 10), 'AR_0': (0, 0),
       'CAR_0_1': (0, 1)}  # 窗口因变量（后三个只进 csv）
WIN_EXTRA = {'market': {'CAR_m5_m1': (-5, -1), 'CAR_0_2': (0, 2), 'CAR_0_3': (0, 3), 'CAR_0_5': (0, 5)},  # 只进 csv 和表 5、6
             'industry': {'CAR_0_2': (0, 2), 'CAR_0_5': (0, 5),  # 和上一轮 CARI_0_2、CARI_0_5 对照
                          'CAR_m5_m1': (-5, -1), 'CAR_0_3': (0, 3)}}
WIN6 = ['CAR_m5_m1', 'CAR_0_1', 'CAR_0_2', 'CAR_0_3', 'CAR_0_5', 'CAR_0_10']  # 表 5、6 的六列
BENCHES = list(WIN_EXTRA)
BENCH_NOTE = {'market': '个股收益减市场收益',  # 市场版沿用原表注
               'industry': '个股收益减所在申万一级行业不含本股的市值加权收益'}
PREV_COL = {'market': {'CAR_0_10': 'CAR_0_10', 'CAR_m10_m1': 'CAR_m10_m1', 'CAR_0_1': 'CAR_0_1',  # 与上一轮面板逐篇对比的列
                       'CAR_m5_m1': 'CAR_m5_m1', 'CAR_0_2': 'CAR_0_2', 'CAR_0_5': 'CAR_0_5'},
            'industry': {'CAR_0_2': 'CARI_0_2', 'CAR_0_5': 'CARI_0_5'}}
CTRL = ['LNCHARS_R', 'LNNODES_R', 'PREVOL', 'PRETURN', 'EA', 'NREP']  # 不放 PRECAR：发布前的日子本身是因变量
CTRL_K = ['LNCHARS_R', 'LNNODES_R']  # 只控制研报字数和概念数（*_k 规格），控制变量不标准化
SPECS = [  # (规格, 固定效应, 控制变量, 样本)
    ('none', [], [], 'all'),
    ('ind', ['sw'], [], 'all'),
    ('firm', ['code'], [], 'all'),
    ('none_s', [], [], 'firm'),
    ('ind_s', ['sw'], [], 'firm'),
    ('ind_c', ['sw', 'broker_g', 'month'], CTRL, 'all'),
    ('firm_c', ['code', 'broker_g', 'month'], CTRL, 'all'),
    ('none_k', [], CTRL_K, 'all'),
    ('ind_k', ['sw'], CTRL_K, 'all'),
    ('firm_k', ['code'], CTRL_K, 'all'),
    ('none_sk', [], CTRL_K, 'firm'),  # 无固定效应，样本同 firm（有 2 篇及以上研报的公司），控制字数和概念数（表 5）
]
MAIN3, CTRL3 = ['none', 'ind', 'firm'], ['none_k', 'ind_k', 'firm_k']  # 表 1、2 和表 3、4 的三列
KEEP_CTRL = CTRL3 + ['none_sk']  # 这些规格另存两个控制变量的系数
CTRL_LABEL = {'LNCHARS_R': '研报字数', 'LNNODES_R': '研报概念数'}
CTRL_NOTE = ('研报字数 LNCHARS_R = ln(研报清理后正文段落的总字符数)，研报概念数 LNNODES_R = ln(该研报概念 DAG 的节点数)，'
             '两者都已在 791 篇研报层面按 1%/99% 缩尾，未标准化（系数是原始单位下的系数）')
DAG_X = ['PATHLEN', 'LNPATH', 'RATIO']
X_LABEL = {'PATHLEN': ('平均路径长度', '平均路径长度 PATHLEN'), 'LNPATH': ('路径数', '路径数 LNPATH（ln(1+路径数)）'),
           'RATIO': ('源汇比', '源汇比 RATIO')}  # (表格行名, 表注里的写法)
KEEP = ['report_id', 'code', 'company', 'date', 'day0', 'p0', 'month', 'broker_g', 'sw'] + DAG_X


def tag(tau):
    return f'm{-tau}' if tau < 0 else str(tau)


AR = {t: f'AR_{tag(t)}' for t in TAUS}
CUM = {t: f'CUM_{tag(t)}' for t in TAUS}
Y_LABEL = {'CAR_m10_m1': 'CAR[−10,−1]', 'CAR_0_10': 'CAR[0,10]', 'CAR_1_10': 'CAR[1,10]', 'AR_0': 'AR[0]', 'CAR_0_1': 'CAR[0,1]',
           'CAR_0_2': 'CAR[0,2]', 'CAR_0_5': 'CAR[0,5]', 'CAR_m5_m1': 'CAR[−5,−1]', 'CAR_0_3': 'CAR[0,3]', **{AR[t]: f'AR[{t}]' for t in TAUS}, **{CUM[t]: f'CUM[−10,{t}]' for t in TAUS}}


def wins_of(bench):
    return {**WIN, **WIN_EXTRA[bench]}


def out_name(x, bench):
    return f'{x.lower()}_event' + ('' if bench == 'market' else '_ind')


# ---------------------------------------------------------------- 分日超额收益
def daily_ar(code, p0, M, bench):
    """第 −10..+10 天的超额收益（小数）：口径同 RC.window_car，窗口内停牌日个股收益按 0，上市前或收益、基准缺失的日子记 nan。"""
    lo, hi = TAUS[0], TAUS[-1]
    out = np.full(len(TAUS), np.nan)
    if p0 + lo < 0 or p0 + hi >= M['ret'].shape[1]:
        return out
    sl = slice(p0 + lo, p0 + hi + 1)
    tr = M['traded'].loc[code].to_numpy()[sl]
    ar = np.where(tr, M['ret'].loc[code].to_numpy()[sl], 0.0) - bench[sl]
    ar[~M['listed'].loc[code].to_numpy()[sl]] = np.nan
    return ar


def load_market(prev_raw, daily_path, industry_path):
    """行情矩阵（同上一轮 build_panel 的 sw_map 和 market_matrices）；核对事件日与上一轮一致。"""
    t0 = time.time()
    codes = sorted(prev_raw.code.unique())
    ind = V.load_industry(industry_path)
    sw_map = ind.dropna(subset=['sw']).set_index('code').sw.to_dict()
    sw_map.update({c: s for c, s in prev_raw.groupby('code').sw.first().items() if c not in sw_map})
    _, _, cal, mkt, M = RC.market_matrices(daily_path, codes, sw_map)
    print(f'行情：{len(cal)} 个交易日（{time.time() - t0:.0f} 秒）', flush=True)
    p0 = np.searchsorted(np.array(cal), prev_raw.date.to_numpy(dtype=str), side='left')
    assert (p0 == prev_raw.p0.to_numpy()).all() and (np.array(cal)[p0] == prev_raw.day0.to_numpy()).all(), '事件日与上一轮不一致'
    return mkt, M


def build_panel(prev_raw, prev_w, mkt, M, bench):
    """上一轮 790 篇 -> 平衡样本面板（未缩尾，单位百分点）和筛选信息。"""
    mk = mkt.to_numpy()
    bench_of = (lambda c: mk) if bench == 'market' else (lambda c: M['ind'].loc[c].to_numpy())  # noqa: E731
    A = np.vstack([daily_ar(c, i, M, bench_of(c)) for c, i in zip(prev_raw.code, prev_raw.p0)]) * 100
    traded = np.vstack([M['traded'].loc[c].to_numpy()[i + TAUS[0]:i + TAUS[-1] + 1] for c, i in zip(prev_raw.code, prev_raw.p0)])
    p = prev_raw[KEEP].reset_index(drop=True)
    p[[AR[t] for t in TAUS]] = A
    p['n_susp'] = (~traded).sum(axis=1)
    ok = ~np.isnan(A).any(axis=1)
    info = {'n0': len(p), 'firms0': p.code.nunique(), 'dropped': p.loc[~ok, ['report_id', 'code', 'company', 'date']],
            'n_susp_reports': int((p.n_susp > 0).sum())}
    p = p[ok].reset_index(drop=True)
    for name, (lo, hi) in wins_of(bench).items():
        p[name] = p[[AR[t] for t in range(lo, hi + 1)]].sum(axis=1)
    p[[CUM[t] for t in TAUS]] = np.cumsum(p[[AR[t] for t in TAUS]].to_numpy(), axis=1)
    p = p.merge(prev_w[['report_id'] + CTRL + ['PRECAR']], on='report_id', how='left')  # 控制变量：上一轮缩尾后的值；PRECAR 只用于自检
    return p, info


def y_list(bench):
    """(Y, tau, 类型)：窗口回归 tau 留空。"""
    return ([(y, np.nan, 'window') for y in wins_of(bench)] + [(AR[t], t, 'daily') for t in TAUS] + [(CUM[t], t, 'cum') for t in TAUS])


def winsorize(p, bench):
    w = p.copy()
    for c in dict.fromkeys(y for y, _, _ in y_list(bench)):
        w[c] = R.winsor(w[c])
    return w


# ---------------------------------------------------------------- 回归
def run_all(p, x, winsor_flag, bench):
    rows = []
    for y, tau, kind in y_list(bench):
        _, s_firm = R.fit(p, y, [x], [], ['code'], 'code')
        for spec, fe, ctrl, samp in SPECS:
            d = p.loc[s_firm.index] if samp == 'firm' else p
            keep_ctrl = spec in KEEP_CTRL
            res, s = R.fit(d, y, [x], ctrl, fe, 'code', terms=keep_ctrl)
            res2, _ = R.fit(d, y, [x], ctrl, fe, ['code', 'day0'])
            r = res[0]
            extra = {f'{k}_{c["X"]}': c[v] for c in res[1:] for k, v in (('b', 'coef'), ('t', 't'), ('p', 'p'))} if keep_ctrl else {}
            rows.append({'X': x, 'spec': spec, 'Y': y, 'Y_label': Y_LABEL[y], 'kind': kind, 'tau': tau, 'winsor': winsor_flag,
                         'coef': r['coef'], 'se': r['se'], 't': r['t'], 'p': r['p'], 't_twoway': res2[0]['t'], 'p_twoway': res2[0]['p'],
                         'n': r['n'], 'n_firms': s.code.nunique(), 'r2': r['r2'], 'adj_r2': r['adj_r2'],
                         'n_clusters': r['n_clusters'], 'n_singletons': r['n_singletons'], 'fe': '+'.join(fe), 'controls': '、'.join(ctrl),
                         'y_mean': r['y_mean'], 'y_sd': r['y_sd'], **extra})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 表格
def num(x, k):
    return f'{x:.{k}f}'.replace('-', '−')


def cell(r):
    return num(r.coef, 3) + R.stars(r.p), f'({num(r.t, 2)})'


def ctrl_cell(r, c):
    return num(r[f'b_{c}'], 3) + R.stars(r[f'p_{c}']), f'({num(r[f"t_{c}"], 2)})'


def table_window(res, x, bench='market', specs=MAIN3):
    """表 1（specs=MAIN3）；表 3（specs=CTRL3：多列两个控制变量的系数，去掉“控制变量”行）。"""
    ctrl = specs == CTRL3
    fe_ind, fe_firm = specs[1], specs[2]
    w = res[(res.winsor == 1) & (res.kind == 'window')].set_index(['Y', 'spec'])
    cols = [(y, s) for y in ['CAR_m10_m1', 'CAR_0_10'] for s in specs]
    rs = [w.loc[c] for c in cols]
    yn = lambda cond: [('Y' if cond(s) else 'N') for _, s in cols]  # noqa: E731
    lines = ['| | CAR[−10,−1] | | | CAR[0,10] | | |', '|---|:-:|:-:|:-:|:-:|:-:|:-:|', '| | (1) | (2) | (3) | (4) | (5) | (6) |']
    row = lambda name, vals: lines.append(f'| {name} | ' + ' | '.join(vals) + ' |')  # noqa: E731
    row(X_LABEL[x][0], [cell(r)[0] for r in rs])
    row('', [cell(r)[1] for r in rs])
    if ctrl:
        for c in CTRL_K:
            row(CTRL_LABEL[c], [ctrl_cell(r, c)[0] for r in rs])
            row('', [ctrl_cell(r, c)[1] for r in rs])
    else:
        row('控制变量', ['N'] * 6)
    row('行业固定效应', yn(lambda s: s == fe_ind))
    row('公司固定效应', yn(lambda s: s == fe_firm))
    row('样本量', [str(int(r.n)) for r in rs])
    row('公司数', [str(int(r.n_firms)) for r in rs])
    row('调整后 R²', [num(r.adj_r2, 3) for r in rs])
    lines += ['', f'注：因变量为研报发布前后窗口的累积超额收益（{BENCH_NOTE[bench]}，百分点），已按 1%/99% 缩尾；解释变量为{X_LABEL[x][1]}，'
                  '已在各回归样本内标准化；' + (CTRL_NOTE + '；' if ctrl else '') +
                  '括号内为按公司聚类的 t 值；***、**、* 分别表示在 1%、5%、10% 水平上显著（按公司聚类的 p 值）。'
                  '公司固定效应列剔除了只有 1 篇研报的公司。']
    return '\n'.join(lines) + '\n'


def table_windows(res, x, bench, spec):
    """表 5（spec='none_sk'）、表 6（spec='firm_k'）：一个规格，六列各一个窗口（WIN6）。R² 是 csv 的 r2：
    1 − SSE/TSS，SSE 来自含常数项、控制变量和全部固定效应虚拟变量的回归，即含公司虚拟变量的整体 R²，不是组内 R²。"""
    ctrl = spec in KEEP_CTRL
    w = res[(res.winsor == 1) & (res.kind == 'window')].set_index(['Y', 'spec'])
    rs = [w.loc[(y, spec)] for y in WIN6]
    lines = ['| | ' + ' | '.join(Y_LABEL[y] for y in WIN6) + ' |', '|---|' + ':-:|' * 6,
             '| | ' + ' | '.join(f'({i})' for i in range(1, 7)) + ' |']
    row = lambda name, vals: lines.append(f'| {name} | ' + ' | '.join(vals) + ' |')  # noqa: E731
    row(X_LABEL[x][0], [cell(r)[0] for r in rs])
    row('', [cell(r)[1] for r in rs])
    if ctrl:
        for c in CTRL_K:
            row(CTRL_LABEL[c], [ctrl_cell(r, c)[0] for r in rs])
            row('', [ctrl_cell(r, c)[1] for r in rs])
    firm_fe = 'code' in (rs[0].fe if isinstance(rs[0].fe, str) else '').split('+')
    row('公司固定效应', ['Y' if firm_fe else 'N'] * 6)
    if not ctrl:
        row('控制变量', ['N'] * 6)
    row('样本量', [str(int(r.n)) for r in rs])
    row('公司数', [str(int(r.n_firms)) for r in rs])
    row('R²', [num(r.r2, 3) for r in rs])
    n, nf = int(rs[0].n), int(rs[0].n_firms)
    sample = (f'样本限定在有 2 篇及以上研报的公司（{n} 篇、{nf} 家），与公司固定效应回归的样本相同；'
              if spec in ('none_sk', 'firm_k') else '')
    if firm_fe:
        design = '加入公司固定效应；R² 是包含公司虚拟变量的整体 R²（不是组内 R²）；'
    else:
        design = '未加固定效应；R² 是普通 OLS 的 R²；'
    lines += ['', f'注：因变量为各窗口的累积超额收益（{BENCH_NOTE[bench]}，百分点），窗口以研报发布日（非交易日顺延到下一个交易日）为第 0 天，'
                  f'各窗口分别按 1%/99% 缩尾；解释变量为{X_LABEL[x][1]}，已在各回归样本内标准化；' + sample +
                  ('控制研报字数和研报概念数：' + CTRL_NOTE + '；' if ctrl else '') + design +
                  '括号内为按公司聚类的 t 值；***、**、* 分别表示在 1%、5%、10% 水平上显著（按公司聚类的 p 值）。']
    return '\n'.join(lines) + '\n'


def table_daily(res, x, bench='market', specs=MAIN3):
    """表 2（specs=MAIN3）；表 4（specs=CTRL3：只列 X 的系数，“控制变量”行写 Y）。"""
    ctrl = specs == CTRL3
    d = res[(res.winsor == 1) & (res.kind == 'daily')].set_index(['tau', 'spec'])
    lines = ['| 事件日 | (1) | (2) | (3) |', '|---|:-:|:-:|:-:|', '| | 无固定效应 | 行业固定效应 | 公司固定效应 |']
    for t in TAUS:
        cs = [cell(d.loc[(t, s)]) for s in specs]
        lines.append(f'| {f"+{t}" if t > 0 else num(t, 0)} | ' + ' | '.join(c[0] for c in cs) + ' |')
        lines.append('| | ' + ' | '.join(c[1] for c in cs) + ' |')
    r0 = [d.loc[(0, s)] for s in specs]
    lines += [f'| 控制变量 | {" | ".join(["Y" if ctrl else "N"] * 3)} |', '| 行业固定效应 | N | Y | N |', '| 公司固定效应 | N | N | Y |',
              '| 样本量 | ' + ' | '.join(str(int(r.n)) for r in r0) + ' |',
              '| 公司数 | ' + ' | '.join(str(int(r.n_firms)) for r in r0) + ' |',
              '', f'注：因变量为研报事件日当日的超额收益（{BENCH_NOTE[bench]}，百分点），已按 1%/99% 缩尾；每格是一个单独的回归，'
                  f'解释变量为{X_LABEL[x][1]}，已在回归样本内标准化；' +
                  ('各列另控制研报字数和研报概念数（' + CTRL_NOTE + '），表中未列出其系数；' if ctrl else '') +
                  '括号内为按公司聚类的 t 值；***、**、* 分别表示在 1%、5%、10% 水平上显著。'
                  '公司固定效应列剔除了只有 1 篇研报的公司。']
    return '\n'.join(lines) + '\n'


# ---------------------------------------------------------------- 自检
def selfcheck_sum(res, x, bench):
    """未缩尾、同一样本下，分日系数之和 = 窗口系数（三个主规格和三个 *_k 规格）。"""
    out = []
    raw = res[res.winsor == 0]
    for spec in MAIN3 + KEEP_CTRL:
        g = raw[raw.spec == spec]
        c = g[g.kind != 'daily'].set_index('Y').coef
        daily = g[g.kind == 'daily'].set_index('tau').coef
        diffs = [f'Σ全部AR − CAR[−10,10] = {daily.sum() - c[CUM[10]]:.2e}']
        for name, (lo, hi) in wins_of(bench).items():
            if name != 'AR_0':
                diffs.append(f'Σ({lo}..{hi}) − {Y_LABEL[name]} = {daily.loc[lo:hi].sum() - c[name]:.2e}')
        out.append(f'  [自检1 {x}] {spec}：' + '；'.join(diffs) + f'；各回归样本量相同：{g.n.nunique() == 1}（n={g.n.iloc[0]}）')
    return out


def selfcheck_prev(res, w, x, prev_res, bench):
    """firm_c 与上一轮口径 B（多一个 PRECAR，PRECAR 是上一轮的市场调整 CAR[−10,−1]）对比；把 PRECAR 加回去，应当复现上一轮系数。"""
    out = []
    ctrl_b = ['LNCHARS_R', 'LNNODES_R', 'PRECAR', 'PREVOL', 'PRETURN', 'EA', 'NREP']  # 上一轮口径 B 的控制变量顺序
    pairs = {'market': [('CAR_0_10', 'CAR_0_10')], 'industry': [('CAR_0_2', 'CARI_0_2'), ('CAR_0_5', 'CARI_0_5')]}[bench]
    for y, y_prev in pairs:
        b = prev_res[(prev_res.spec == 'B') & (prev_res.Y == y_prev) & (prev_res.reg == '单独') & (prev_res.X == x)].iloc[0]
        f = res[(res.winsor == 1) & (res.Y == y) & (res.spec == 'firm_c')].iloc[0]
        rb, _ = R.fit(w.assign(**{y: w[y] / 100}), y, [x], ctrl_b, ['code', 'broker_g', 'month'], 'code')
        out.append(f'  [自检3 {x}] firm_c {Y_LABEL[y]}：本轮 {f.coef / 100:.6f}（t {f.t:.2f}，n {f.n}）；上一轮口径 B {y_prev} {b.coef:.6f}'
                   f'（t {b.t:.2f}，n {b.n}）；差 {f.coef / 100 - b.coef:+.6f}；本轮面板加回 PRECAR 后 {rb[0]["coef"]:.6f}（t {rb[0]["t"]:.2f}），'
                   f'与上一轮差 {abs(rb[0]["coef"] - b.coef):.2e}、t 差 {abs(rb[0]["t"] - b.t):.2e}')
    return out


def selfcheck_same_sample(w, x, y='CAR_0_1'):
    """none_sk 与 firm_k 的估计样本是否是同一批研报（比对 report_id 集合）。"""
    _, s_firm = R.fit(w, y, [x], [], ['code'], 'code')
    _, s_sk = R.fit(w.loc[s_firm.index], y, [x], CTRL_K, [], 'code')
    _, s_k = R.fit(w, y, [x], CTRL_K, ['code'], 'code')
    a, b = set(w.report_id[s_sk.index]), set(w.report_id[s_k.index])
    return [f'  [同样本 {x}] none_sk {len(a)} 篇、{s_sk.code.nunique()} 家；firm_k {len(b)} 篇、{s_k.code.nunique()} 家；'
            f'report_id 集合相同：{a == b}（只在 none_sk {len(a - b)} 篇，只在 firm_k {len(b - a)} 篇）']


def selfcheck_collinear(w, x):
    """X 与两个控制变量的两两相关系数，以及（无 / 行业 / 公司固定效应下）X 对两个控制变量回归的 R²：
    组内 R² 是先去掉固定效应后的 R²，整体 R² 含固定效应虚拟变量的解释力。"""
    cols = [x] + CTRL_K
    c = w[cols].corr()
    out = [f'  [共线性 {x}] 相关系数：{x}–LNCHARS_R {c.iloc[0, 1]:.3f}，{x}–LNNODES_R {c.iloc[0, 2]:.3f}，'
           f'LNCHARS_R–LNNODES_R {c.iloc[1, 2]:.3f}（{len(w)} 篇）']
    for name, fe in [('无固定效应', []), ('行业固定效应', ['sw']), ('公司固定效应', ['code'])]:
        s, _ = R.drop_singletons(w[cols + fe].dropna(), fe)
        z = R.within(s, cols, fe) if fe else s[cols].to_numpy(float) - s[cols].to_numpy(float).mean(0)
        e = z[:, 0] - z[:, 1:] @ np.linalg.lstsq(z[:, 1:], z[:, 0], rcond=None)[0]
        r_full, _ = R.fit(s, x, CTRL_K, [], fe, None, standardize=False)
        out.append(f'  [共线性 {x}] {name}（n={len(s)}）：{x} 对两个控制变量的组内 R² {1 - (e @ e) / (z[:, 0] @ z[:, 0]):.3f}；'
                   f'含固定效应的整体 R² {r_full[0]["r2"]:.3f}')
    return out


def selfcheck_car(p, prev_raw, bench):
    cols = PREV_COL[bench]
    m = p.merge(prev_raw[['report_id'] + list(cols.values())].rename(columns={v: v + '_prev' for v in cols.values()}), on='report_id')
    return [f'  [自检2] {c}：新面板/100 与上一轮 {v} 逐篇最大绝对差 {np.abs(m[c] / 100 - m[v + "_prev"]).max():.2e}（{len(m)} 篇）'
            for c, v in cols.items()]


def main(out_base, prev_dir, daily_path, industry_path, xs=DAG_X, benches=('market',)):
    t0 = time.time()
    prev_raw = pd.read_parquet(prev_dir / 'report_car_panel_raw.parquet')
    prev_w = pd.read_parquet(prev_dir / 'report_car_panel.parquet')
    prev_res = pd.read_csv(prev_dir / 'report_car_results.csv')
    mkt, M = load_market(prev_raw, daily_path, industry_path)
    out = {}
    for bench in benches:
        p, info = build_panel(prev_raw, prev_w, mkt, M, bench)
        print(f'[{bench}] 上一轮样本 {info["n0"]} 篇、{info["firms0"]} 家；窗口内有停牌日的 {info["n_susp_reports"]} 篇；'
              f'21 天 AR 有缺失而剔除 {len(info["dropped"])} 篇；平衡样本 {len(p)} 篇、{p.code.nunique()} 家', flush=True)
        if len(info['dropped']):
            print(info['dropped'].to_string(index=False), flush=True)
        for c in CTRL + DAG_X + ['sw']:
            k = p[c].isna().sum()
            if k:
                print(f'  {c} 缺失 {k} 篇', flush=True)
        w = winsorize(p, bench)
        for line in selfcheck_car(p, prev_raw, bench):
            print(line, flush=True)
        for x in xs:
            name = out_name(x, bench)
            od = out_base / name
            (od / 'tables').mkdir(parents=True, exist_ok=True)
            p.to_parquet(od / f'{name}_panel_raw.parquet', index=False)  # 同一基准下三个变量的面板内容相同，各目录存一份方便单独使用
            w.to_parquet(od / f'{name}_panel.parquet', index=False)
            res = pd.concat([run_all(w, x, 1, bench), run_all(p, x, 0, bench)], ignore_index=True)
            res.to_csv(od / f'{name}_results.csv', index=False, encoding='utf-8-sig')
            (od / 'tables' / 't1_window.md').write_text(table_window(res, x, bench), encoding='utf-8')
            (od / 'tables' / 't2_daily.md').write_text(table_daily(res, x, bench), encoding='utf-8')
            (od / 'tables' / 't3_window_ctrl.md').write_text(table_window(res, x, bench, CTRL3), encoding='utf-8')
            (od / 'tables' / 't4_daily_ctrl.md').write_text(table_daily(res, x, bench, CTRL3), encoding='utf-8')
            (od / 'tables' / 't5_windows_none.md').unlink(missing_ok=True)  # 旧版表 5（规格 none），已由下一行替代
            (od / 'tables' / 't5_windows_pooled_ctrl.md').write_text(table_windows(res, x, bench, 'none_sk'), encoding='utf-8')
            for line in selfcheck_same_sample(w, x):
                print(line, flush=True)
            (od / 'tables' / 't6_windows_firm_ctrl.md').write_text(table_windows(res, x, bench, 'firm_k'), encoding='utf-8')
            for line in selfcheck_collinear(w, x):
                print(line, flush=True)
            for line in selfcheck_sum(res, x, bench) + selfcheck_prev(res, w, x, prev_res, bench):
                print(line, flush=True)
            print(f'{x} × {bench} 完成：{od}（{len(res)} 个回归，{time.time() - t0:.0f} 秒）', flush=True)
            out[(x, bench)] = res
    return out
