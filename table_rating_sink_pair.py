"""v6–v8：分析单位改为“公司-券商”（年度只有 2024），复用 table_rating_sink.py 的样本、X、回归函数。

观测    (公司 code, 券商 ORG_CL)：该券商 2024 年对该公司至少有一篇样本研报（见 --x）。
X       --x rt11（v6/v7）：样本研报 = 606 篇（“投资评级”是严格汇点），11 个研报层面原始值（*_RAW：RT_ 五个、LNCHARS、WG_ 五个）。
        --x val3（v8）：样本研报 = 791 篇 DAG 研报里含至少一个估值类汇点的（三个指标有定义），3 个指标为全部估值类严格汇点的子图
                 （F.sink_frame(…, 'sink_val') = D.sink_measures(…, 'VAL', count=False)）：VAL_RATIO = 源点数 ÷ 估值类汇点个数、
                 VAL_LNPATH = ln(1 + 源点到这些汇点的路径条数)、VAL_PATHLEN = 这些路径的平均边数。
        --agg mean：该对名下各篇研报原始值的平均（v6）；--agg last：2024 年内最新一篇（发布日最大，同日取 report_id 较大的；v7/v8）。
        在进入样本的对内 1%/99% 缩尾，回归时在估计样本内标准化（R.fit）。
Y       被评价的预测 F：该券商 2024 年内（RPT_DT 在 2024 年）对该公司目标年度 t 归母净利润的最后一条预测（IS_VLD = 1、预测值非缺失，
        按 RPT_DT、CREAT_TM、SEQ 取最后），不一定出自深度研报。AFE = |F − A_t|。
        比较组：2024 年内对该公司目标年度 t 做过预测的全部券商，每家取 2024 年内最后一条（含本券商），≥ 4 家。
        PMAFE = (AFE − MAFE)/MAFE（MAFE 含本券商），PMAFEX（MAFE 不含本券商），SCORE（Hong-Kubik 分数，越大越准，不缩尾），
        RA = (median − AFE)/SD（含本券商，ddof=1）。t = 2025（主）和 2024（PMAFE_FY24、SCORE_FY24）。
DAGE    Age = 被评价预测的发布日到目标年度 12 月 31 日的天数；DAGE = 本券商 Age − 比较组 Age 均值（正 = 更陈旧）。不缩尾。
控制    SIZE_M、BM、MOM、VOL、ROA_T、LNAGE 在被评价预测的发布日上度量（F.market_vars + 面板的公式，T = 该预测行的 RPT_YR_T），
        MOM / VOL / ROA_T × 100。FY2025 的回归用 FY2025 被评价预测日上的值；FY2024 的回归用 FY2024 被评价预测日上的值（列名 *_24）。
缩尾    Y（SCORE 除外）和控制变量：在“786 篇基础样本研报涉及的对”内求 1%/99% 分位，用同一界限截断全部对。
固定效应 N 无；A 申万行业（+ 常数的目标年度）；B 公司；C 公司 + 券商（broker_g：各 ORG_CL 在面板里最常见的 broker_g）。按公司聚类。
提速    中间结果（面板与研报样本、2018–2025 年预测表与长表、新日期上的控制变量、估值类汇点指标）缓存在 output_dag/_cache/table10_pair/，
        缓存键含参数、源代码文件内容的哈希和输入数据文件的大小 / 修改时间；回归按（块，Y，面板）拆成任务多进程并行，
        工作进程的 BLAS 线程数限为 1。--no-cache 不读写缓存；--jobs 1 不开进程池。
用法    C:/Users/hung/projects/.venv/Scripts/python.exe -X utf8 table_rating_sink_pair.py [--x rt11|val3] [--agg mean|last] [--out 目录]
        rt11 + mean → rating_table10_v6/；rt11 + last → rating_table10_v7/；val3 + last → rating_table10_v8/（val3 + mean 需给 --out）。
"""
import hashlib
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table_rating_sink as T  # noqa: E402
from src import dag_forecast as F  # noqa: E402
from src import dag_measures as D  # noqa: E402
from src import dag_outcomes_reg as R  # noqa: E402

ND = T.ROOT / 'output_dag' / 'forecast_newdata'
OUT_DEFAULT = {('rt11', 'mean'): ND / 'rating_table10_v6', ('rt11', 'last'): ND / 'rating_table10_v7', ('val3', 'last'): ND / 'rating_table10_v8',
               ('new11', 'last'): ND / 'rating_table10_v9', ('new13', 'last'): ND / 'rating_table10_v10',
               ('rt6', 'last'): ND / 'rating_table10_v12' / 'pair', ('rt6a', 'last'): ND / 'rating_table10_v13' / 'pair'}
CACHE_DIR = T.ROOT / 'output_dag' / '_cache' / 'table10_pair'
CTRL, PANS, XS2, MEAS2 = T.CTRL, T.PANS, T.XS2, T.MEAS2
XS_VAL = ['VAL_RATIO', 'VAL_LNPATH', 'VAL_PATHLEN']
XS_NEW = ['EARN_SHARE', 'EARN_LNSRC', 'MULTI_SUP', 'NEG_SHARE', 'LCC_SHARE', 'ISO_SHARE', 'MULTI_CAUSE', 'MID_SHARE', 'METRIC_SHARE',
          'EXT_SRC', 'DENSITY']
XS_13 = ['RATE_NSRC', 'ALL_NSRC', 'ALL_LNNODE', 'EARN_SHARE', 'MULTI_SUP', 'NEG_SHARE', 'METRIC_SHARE', 'DENSITY', 'LCC_SHARE', 'ISO_SHARE',
         'MULTI_CAUSE', 'MID_SHARE', 'EXT_SRC']
XS_13V = ['RATE_LNSRC', 'ALL_LNSRC']  # 只作对照（logsrc 块）
# v12（--x rt6）：评级子图与整张图的源点数、ln(1+路径数)、平均路径长度（D.five_measures 的 NSRC/LNPATH/PATHLEN 原始值）
XS_RT6 = ['RT_NSRC', 'RT_LNPATH', 'RT_PATHLEN', 'WG_NSRC', 'WG_LNPATH', 'WG_PATHLEN']
XS_RT6V = ['RT_LNSRC', 'WG_LNSRC']  # ln(源点数)，只作对照
RT6_PAIRS = [['RT_NSRC', 'WG_NSRC'], ['RT_LNPATH', 'WG_LNPATH'], ['RT_PATHLEN', 'WG_PATHLEN']]
# v13（--x rt6a）：券商固定效应换成分析师（analyst = broker_g|第一作者，见 analyst_keys）；D = 分析师，C = 公司 + 分析师
PANS13 = {'N': PANS['N'], 'A': PANS['A'], 'B': PANS['B'], 'D': (['analyst', 'ty'], PANS['B'][1]), 'C': (['code_ty', 'analyst'], PANS['C'][1])}
REF_COMBOS = [['RATE_NSRC', 'ALL_NSRC'], ['RATE_NSRC', 'ALL_LNNODE'], ['RATE_NSRC', 'ALL_NSRC', 'ALL_LNNODE']]
EARN_RE = r'净利润|归母|利润|盈利|业绩|每股收益|EPS|营业收入|营收|收入|毛利|净利率|ROE|ROA|ROIC'
VALX_RE = r'估值|评级|目标价|股价|市值'
EXT_RE = (r'宏观|经济|GDP|利率|汇率|货币政策|财政|政策|通胀|CPI|PPI|大宗商品|原油|油价|煤价|金价|铜价|原材料价格|行业|市场需求|下游需求|景气|周期|'
          r'出口|关税|房地产|地产|人口|消费升级')
PCT = ['MOM', 'VOL', 'ROA_T']
Y_W = ['PMAFE_FY25', 'PMAFE_FY24', 'PMAFEX_FY25', 'RA_FY25']  # 缩尾
YS_LADDER = ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25', 'SCORE_FY24', 'PMAFEX_FY25', 'RA_FY25']
YS_HORSE = ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25']
MIN_GROUP = 4
SAME_DAYS = 30
log = T.log
USE_CACHE = True
TIMES = {}


def dage_col(y):
    return 'DAGE' + y[-2:]


# ---------------------------------------------------------------- 计时和缓存
@contextmanager
def stage(name):
    t0 = time.perf_counter()
    yield
    TIMES[name] = TIMES.get(name, 0) + time.perf_counter() - t0


_FP = None


def fingerprint():
    """源代码文件内容的哈希 + 输入数据文件的（路径, 大小, 修改时间）。任何一项变了缓存就失效。"""
    global _FP
    if _FP is None:
        src = [Path(T.__file__), T.ROOT / 'src' / 'dag_forecast.py', T.ROOT / 'src' / 'dag_measures.py', T.ROOT / 'src' / 'dag_outcomes_vars.py',
               T.ROOT / 'src' / 'dag_outcomes_reg.py', T.ROOT / 'src' / 'viz' / 'plot_dag_structure.py', T.ROOT / 'src' / 'config.py']
        data = [F.PANEL, F.FINANCIAL, F.DAILY, T.REPORT_PANEL, T.EXRIGHTS] + sorted(Path(F.EST_DIR).rglob('*.parquet')) \
            + [p for d in D.DAG_DIRS for p in sorted(Path(d).glob('*.parquet'))]
        h = hashlib.sha1()
        for p in src:
            h.update(p.name.encode() + hashlib.sha1(p.read_bytes()).digest())
        for p in data:
            st = p.stat()
            h.update(f'{p}|{st.st_size}|{st.st_mtime_ns}'.encode())
        _FP = h.hexdigest()
    return _FP


def cached(name, params, compute, n):
    """compute() 返回 n 个 DataFrame 的元组；按 (name, params, fingerprint) 缓存为 parquet。"""
    key = hashlib.sha1(json.dumps({'name': name, 'params': params, 'fp': fingerprint()}, sort_keys=True, default=str).encode()).hexdigest()[:16]
    paths = [CACHE_DIR / f'{name}_{key}_{i}.parquet' for i in range(n)]
    if USE_CACHE and all(p.exists() for p in paths):
        return tuple(pd.read_parquet(p) for p in paths), True
    out = compute()
    if USE_CACHE:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        for df, p in zip(out, paths):
            df.to_parquet(p, index=False)
    return out, False


def frame_hash(df):
    return hashlib.sha1(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()


# ---------------------------------------------------------------- 比较组和因变量
def groups(lg, ty, codes):
    """2024 年内对目标年度 ty 的预测，每个（公司，券商）取最后一条（lg 已按公司、目标年度、券商、时间排序）。"""
    g = lg[(lg.ty == ty) & (lg.RPT_DT.str[:4] == '2024') & lg.SEC_CD.isin(codes)]
    return g.groupby(['SEC_CD', 'ORG_CL']).tail(1).reset_index(drop=True)


def pair_ys(pairs, grp, ann, ty, tag):
    """每个对的被评价预测和因变量；grp = groups(...)。返回 (每对一行, 比较组明细)。"""
    fyend = pd.Timestamp(f'{ty}1231')
    grp = grp.copy()
    a_ty = ann.A.xs(ty, level='fy').to_dict()  # 同 ann.A.get((code, ty))，查字典代替逐行查 MultiIndex（提速，取值不变）
    grp['A'] = [a_ty.get(c, np.nan) for c in grp.SEC_CD]
    grp['AFE'] = (grp.F - grp.A).abs()
    grp['AGE'] = (fyend - pd.to_datetime(grp.RPT_DT, format='%Y%m%d')).dt.days
    # 每家公司的比较组拆成 numpy 数组（提速；顺序同 groupby，算法与原来逐项相同）
    by = {c: (g.ORG_CL.to_numpy(), g.AFE.to_numpy(float), g.AGE.to_numpy(), g.RPT_ID.to_numpy(), g.RPT_DT.to_numpy(), g.F.to_numpy(),
              g.RPT_YR_T.to_numpy(), g.A.to_numpy()) for c, g in grp.groupby('SEC_CD')}
    rows = []
    for code, org in zip(pairs.code, pairs.ORG_CL):
        rec = {'code': code, 'ORG_CL': org}
        if code not in by or not (by[code][0] == org).any():
            rows.append(rec)
            continue
        orgs, e, ages, rids, dts, fs, ts, aa = by[code]
        i = int(np.flatnonzero(orgs == org)[0])
        a, n = e[i], len(e)
        rec.update({f'EVAL_RPT_ID{tag}': rids[i], f'EVAL_DT{tag}': dts[i], f'EVAL_F{tag}': fs[i], f'EVAL_T{tag}': ts[i],
                    f'A{tag}': aa[i], f'AFE{tag}': a, f'N_GROUP{tag}': n, f'AGE{tag}': ages[i]})
        others = e[orgs != org]
        if n >= MIN_GROUP and np.isfinite(e).all():
            mafe, mafex = e.mean(), others.mean()
            less, eq = (others < a).sum(), (others == a).sum()
            r = 1 + less + eq / 2
            sd = e.std(ddof=1)
            rec.update({f'MAFE{tag}': mafe, f'MAFEX{tag}': mafex,
                        f'PMAFE_FY{tag}': (a - mafe) / mafe if mafe > 0 else np.nan,
                        f'PMAFEX_FY{tag}': (a - mafex) / mafex if mafex > 0 else np.nan,
                        f'SCORE_FY{tag}': 100 - (r - 1) / (n - 1) * 100,
                        f'RA_FY{tag}': (np.median(e) - a) / sd if sd > 0 else np.nan,
                        f'DAGE{tag}': ages[i] - ages.mean()})
        rows.append(rec)
    det = grp[['SEC_CD', 'ORG_CL', 'RPT_ID', 'RPT_DT', 'RPT_YR_T', 'F', 'A', 'AFE', 'AGE']].rename(columns={'SEC_CD': 'code'})
    return pd.DataFrame(rows), det


def controls_at(ev, ann):
    """六个控制变量在被评价预测的发布日上（同 F._build_panel 的公式）。ev：key、code、date、T。"""
    mv = F.market_vars(ev.rename(columns={'key': 'report_id'})[['report_id', 'code', 'date']], F.DAILY)
    o = ev.merge(mv.rename(columns={'report_id': 'key'}), on='key', how='left')
    look = lambda col: [ann[col].get((c, int(t)), np.nan) if pd.notna(t) else np.nan for c, t in zip(o.code, o['T'])]  # noqa: E731
    o['SIZE_M'] = np.log(o.MKCAP)
    o['BM'] = np.array(look('EQ'), float) / o.MKCAP
    o['ROA_T'] = np.array(look('A'), float) / np.array(look('TA'), float)
    age = (-(pd.to_datetime(o.FIRST_DAY, format='%Y%m%d', errors='coerce') - pd.to_datetime(o.date, format='%Y%m%d')).dt.days) / 30.4375
    o['LNAGE'] = np.log(age.where(age > 0))
    return o[['key', 'SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']]


def winsor_by(df, cols, universe):
    """在 universe（布尔）内求 1%/99% 分位，截断全部行；返回界限。"""
    bounds = {}
    for c in cols:
        lo, hi = df.loc[universe, c].quantile([.01, .99])
        bounds[c] = (lo, hi)
        df[c + '_RAW'] = df[c]
        df[c] = df[c].clip(lo, hi)
    return bounds


# ---------------------------------------------------------------- 估值类汇点（v8）
def val_report_measures(report_ids):
    """每篇研报：全部估值类严格汇点子图的三个指标（F.sink_frame 'sink_val' = D.sink_measures(…, 'VAL', count=False)）、汇点名单。"""
    sf = F.sink_frame(report_ids, 'sink_val')
    graphs = D.load_graphs(list(report_ids))
    names = {rid: [v for v in D.strict_sinks(g) if D.classify_sink(v) == 'VAL'] for rid, g in graphs.items()}
    out = pd.DataFrame({'report_id': sf.report_id, 'N_VAL_SINKS': sf.R_N_SINKS, 'N_SOURCES': sf.R_N_SOURCES, 'N_PATHS': sf.R_N_PATHS,
                        'N_ANC': sf.R_N_ANC, 'VAL_RATIO_RAW': sf.R_RATIO_RAW, 'VAL_LNPATH_RAW': sf.R_LNPATH_RAW,
                        'VAL_PATHLEN_RAW': sf.R_PATHLEN_RAW})
    out['VAL_SINKS'] = ['；'.join(names[r]) for r in out.report_id]
    assert (out.N_VAL_SINKS.fillna(0).astype(int) == [len(names[r]) for r in out.report_id]).all()
    return out


# ---------------------------------------------------------------- 回归（多进程）
def _reg_task(args):
    data, xs_, y, pans, ctrl, joint = args
    return F.regress_each(data, xs_, [y], pans=pans, controls=ctrl, joint=joint, terms=True)


BLAS_THREADS = 24  # 工作进程的 BLAS 线程数；见 run_tasks 的说明


def run_tasks(tasks, jobs):
    """tasks：[(块, spec_name, regress_each 的参数元组)]；按顺序返回 reshape 后的结果列表（与串行完全同序）。
    工作进程的 BLAS 线程数由 BLAS_THREADS 决定（启动前写进环境变量，工作进程 import numpy 时读到）。本机 numpy 自带的 OpenBLAS
    MAX_THREADS = 24、默认就开 24 个线程；OpenBLAS 的结果随线程数变化（求和分块不同），设计矩阵病态时（如面板 B/C 里几乎被公司固定效应
    吸收的 LNAGE）系数会差到相对 1e-5。所以默认 BLAS_THREADS = 24，与串行（默认线程数）逐位一致；--blas-threads 1 更快但不保证逐位一致。"""
    if jobs <= 1:
        outs = [_reg_task(a) for _, _, a in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        for v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
            os.environ[v] = str(BLAS_THREADS)
        with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as ex:
            outs = list(ex.map(_reg_task, [a for _, _, a in tasks], chunksize=1))
    return [T.reshape(o, blk, spec) for (blk, spec, _), o in zip(tasks, outs)]


# ---------------------------------------------------------------- 主流程
def main(agg_mode='mean', out=None, xset='rt11', jobs=16):
    """agg_mode：'mean' 或 'last'；xset：'rt11'（606 篇、11 个 X）或 'val3'（含估值类汇点的研报、3 个 X）。"""
    t_all = time.perf_counter()
    TIMES.clear()
    if out is None and (xset, agg_mode) not in OUT_DEFAULT:
        raise SystemExit(f'--x {xset} --agg {agg_mode} 没有默认输出目录，请给 --out')
    OUT = Path(out) if out else OUT_DEFAULT[(xset, agg_mode)]
    OUT.mkdir(parents=True, exist_ok=True)
    XS = {'rt11': XS2, 'val3': XS_VAL, 'new11': XS_NEW, 'new13': XS_13, 'rt6': XS_RT6, 'rt6a': XS_RT6}[xset]
    XW = XS + (XS_13V if xset == 'new13' else [])  # 要缩尾的全部 X（含只作对照的变体）
    ver = {('rt11', 'mean'): 'v6', ('rt11', 'last'): 'v7', ('val3', 'last'): 'v8', ('new11', 'last'): 'v9',
           ('new13', 'last'): 'v10', ('rt6', 'last'): 'v12', ('rt6a', 'last'): 'v13'}.get((xset, agg_mode), f'{xset}/{agg_mode}')
    PAN_ = PANS13 if xset == 'rt6a' else PANS  # 自检用的固定效应

    with stage('面板、研报样本、图指标（build_sample_v2 + 未缩尾面板）'):
        def _sample():
            d2_, s_, _ = T.build_sample_v2()
            return d2_, s_, F.build_panel(main_only=False)[0]
        (d2, s, raw), hit = cached('sample_v2', {}, _sample, 3)
    T.log_lines.clear()
    log(f'（{ver}）分析单位 = 公司-券商（2024 年），X = {xset}，聚合方式 = {agg_mode}；以下为构造和核对')
    log(f'缓存：面板与研报样本 {"命中" if hit else "重算"}')
    codes = sorted(set(d2.code))
    with stage('读预测表（2018–2025）并建长表'):
        (est, lg), hit = cached('est_long', {'years': [2018, 2025], 'codes': codes}, lambda: (lambda e: (e, T.long_v5(e)))(
            T.load_est_eps(codes, years=range(2018, 2026))), 2)
    log(f'缓存：预测表与长表 {"命中" if hit else "重算"}')
    ann = F.load_annual(F.FINANCIAL, codes)

    # broker_g：每个 ORG_CL 在面板里最常见的 broker_g
    bg = d2.dropna(subset=['ORG_CL']).groupby('ORG_CL').broker_g.agg(lambda x: x.value_counts().index[0])
    multi = d2.dropna(subset=['ORG_CL']).groupby('ORG_CL').broker_g.nunique()
    log(f'ORG_CL {len(bg)} 个；对应多个 broker_g 的 ORG_CL {int((multi > 1).sum())} 个（取最常见的）；broker_g 组数 {bg.nunique()}')
    sw = d2.groupby('code').sw.first()

    # 研报样本
    if xset == 'rt11':
        s_ok = s[s.ORG_CL.notna()]
        log(f'606 篇里没有 ORG_CL（未匹配到本篇预测行）的 {int(s.ORG_CL.isna().sum())} 篇，不能归到券商，剔除')
    elif xset == 'new11':
        with stage('新指标（11 个）'):
            (nm,), hit = cached('new_measures', {'n': len(d2), 'res': [EARN_RE, VALX_RE, EXT_RE]}, lambda: (new_report_measures(list(d2.report_id)),), 1)
        nm.to_parquet(OUT / 'new_report_measures.parquet', index=False)
        new_report_stats(nm)
        s_ok = d2[['report_id', 'code', 'ORG_CL', 'date', 'base', 'RPT_YR_T']].merge(
            nm[['report_id'] + XS_NEW].rename(columns={x: x + '_RAW' for x in XS_NEW}), on='report_id', how='left')
        log(f'791 篇里没有 ORG_CL 的 {int(s_ok.ORG_CL.isna().sum())} 篇，剔除；11 个指标缺失的篇数 {int(s_ok[[x + "_RAW" for x in XS_NEW]].isna().any(axis=1).sum())}')
        s_ok = s_ok[s_ok.ORG_CL.notna()].copy()
    elif xset == 'new13':
        with stage('新指标（13 个）'):
            (nm,), hit = cached('new_measures', {'n': len(d2), 'res': [EARN_RE, VALX_RE, EXT_RE]}, lambda: (new_report_measures(list(d2.report_id)),), 1)
            nm = new13_measures(nm)
        nm.to_parquet(OUT / 'new_report_measures.parquet', index=False)
        new13_stats(nm)
        s_ok = d2[['report_id', 'code', 'ORG_CL', 'date', 'base', 'RPT_YR_T']].merge(
            nm[['report_id'] + XW].rename(columns={x: x + '_RAW' for x in XW}), on='report_id', how='left')
        log(f'791 篇里没有 ORG_CL 的 {int(s_ok.ORG_CL.isna().sum())} 篇，剔除')
        s_ok = s_ok[s_ok.ORG_CL.notna()].copy()
    elif xset in ('rt6', 'rt6a'):
        rm = rt6_measures(list(d2.report_id))
        rm.to_parquet(OUT / 'rt6_measures.parquet', index=False)
        log(f'v12 X：RT_* = five_measures(graphs, "投资评级", strict=True) 的 NSRC/LNPATH/PATHLEN 原始值（非严格汇点缺失），WG_* = five_measures(graphs, None)；'
            f'RT_NSRC 非缺失 {int(rm.RT_NSRC.notna().sum())} 篇，WG_* 非缺失 {int(rm.WG_NSRC.notna().sum())} 篇')
        s_ok = d2[['report_id', 'code', 'ORG_CL', 'date', 'base', 'RPT_YR_T']].merge(
            rm[['report_id'] + XW].rename(columns={x: x + '_RAW' for x in XW}), on='report_id', how='left')
        log(f'791 篇里没有 ORG_CL 的 {int(s_ok.ORG_CL.isna().sum())} 篇，剔除')
        s_ok = s_ok[s_ok.ORG_CL.notna()].copy()
    else:
        with stage('估值类汇点指标'):
            (vm,), hit = cached('val_measures', {'n': len(d2)}, lambda: (val_report_measures(list(d2.report_id)),), 1)
        vm.to_parquet(OUT / 'val_report_measures.parquet', index=False)
        val_report_stats(vm, d2)
        has = d2[['report_id', 'code', 'ORG_CL', 'date', 'base', 'RPT_YR_T']].merge(vm, on='report_id', how='left')
        has = has[has.VAL_RATIO_RAW.notna()]
        log(f'791 篇里有估值类汇点（三个指标有定义）的 {len(has)} 篇；其中没有 ORG_CL 的 {int(has.ORG_CL.isna().sum())} 篇，剔除；'
            f'这些研报都在 2024 年：{bool(has.date.str[:4].eq("2024").all())}')
        s_ok = has[has.ORG_CL.notna()].copy()
        log(f'与 606 篇（投资评级是严格汇点）的重合：{int(s_ok.report_id.isin(s.report_id).sum())} 篇')
    agg = {x + '_RAW': 'mean' for x in XW}
    P = (s_ok.groupby(['code', 'ORG_CL']).agg(n_reports=('report_id', 'size'), report_ids=('report_id', lambda x: ';'.join(x)),
                                             last_deep_dt=('date', 'max'), **{k: (k, v) for k, v in agg.items()}).reset_index())
    P_mean_x = P[['code', 'ORG_CL'] + list(agg)].copy()
    extra = ['N_VAL_SINKS', 'N_SOURCES', 'N_PATHS'] if xset == 'val3' else []
    if agg_mode == 'last':  # 换成该对 2024 年内最新一篇的原始值（发布日最大；同日取 report_id 字典序较大的）
        srt = s_ok.sort_values(['code', 'ORG_CL', 'date', 'report_id'])
        last = srt.groupby(['code', 'ORG_CL']).tail(1)
        ties = srt.groupby(['code', 'ORG_CL']).apply(lambda g: int((g.date == g.date.max()).sum() > 1), include_groups=False).sum()
        log(f'X 取最新一篇：同一对内最新发布日有两篇及以上的对 {int(ties)} 个；所用研报都在 2024 年内：{bool(last.date.str[:4].eq("2024").all())}')
        last = last[['code', 'ORG_CL', 'report_id', 'date'] + list(agg) + extra].rename(columns={'report_id': 'X_report_id', 'date': 'X_report_dt'})
        P = P.drop(columns=list(agg)).merge(last, on=['code', 'ORG_CL'], how='left', validate='1:1')
    U = d2[d2.base & d2.ORG_CL.notna()][['code', 'ORG_CL']].drop_duplicates()
    allp = pd.concat([P[['code', 'ORG_CL']], U]).drop_duplicates().reset_index(drop=True)
    in_u = allp.merge(U.assign(_u=True), on=['code', 'ORG_CL'], how='left')._u.fillna(False).astype(bool).to_numpy()
    log(f'样本对 {len(P)} 个（公司 {P.code.nunique()} 家、券商 {P.ORG_CL.nunique()} 家）；每对研报篇数：1 篇 {int((P.n_reports == 1).sum())}、'
        f'2 篇 {int((P.n_reports == 2).sum())}、3 篇及以上 {int((P.n_reports >= 3).sum())}（最多 {P.n_reports.max()}）；'
        f'缩尾全集（786 篇基础样本涉及的对）{int(in_u.sum())} 个；两者并集 {len(allp)} 个')

    # 因变量
    with stage('比较组与因变量'):
        y25, det25 = pair_ys(allp, groups(lg, 2025, codes), ann, 2025, '25')
        y24, det24 = pair_ys(allp, groups(lg, 2024, codes), ann, 2024, '24')
        Y = y25.merge(y24, on=['code', 'ORG_CL'], how='left')
    det25.to_parquet(OUT / 'group_detail_fy25.parquet', index=False)
    det24.to_parquet(OUT / 'group_detail_fy24.parquet', index=False)

    # 控制变量：FY2025 回归用 FY2025 被评价预测发布日上的值；FY2024 回归用 FY2024 被评价预测发布日上的值（列名加 _24）
    Y['key'] = [f'{c}|{o}' for c, o in zip(Y.code, Y.ORG_CL)]  # ORG_CL 是数值
    with stage('新日期上的控制变量'):
        for tag, suf in (('25', ''), ('24', '_24')):
            ev = Y[['key', 'code', f'EVAL_DT{tag}', f'EVAL_T{tag}']].dropna().rename(columns={f'EVAL_DT{tag}': 'date', f'EVAL_T{tag}': 'T'})
            (ctl,), hit = cached('controls', {'ev': frame_hash(ev)}, lambda: (controls_at(ev, ann),), 1)
            log(f'缓存：FY{tag} 控制变量 {"命中" if hit else "重算"}')
            Y = Y.merge(ctl.rename(columns={c: c + suf for c in CTRL}), on='key', how='left')

    # 控制变量核对：被评价预测就是某篇面板研报本身时，应等于面板里该研报的缩尾前值
    rid = raw[['report_id', 'code'] + CTRL].assign(RID=lambda t: t.report_id.str.upper())
    j = Y.dropna(subset=['EVAL_RPT_ID25']).assign(RID=lambda t: t.EVAL_RPT_ID25.str.upper()).merge(
        rid, on=['RID', 'code'], suffixes=('', '_panel'))
    diffs = {c: float(np.nanmax(np.abs(j[c] - j[c + '_panel']))) if len(j) else np.nan for c in CTRL}
    nmis = {c: int((j[c].isna() != j[c + '_panel'].isna()).sum()) for c in CTRL}
    log(f'控制变量核对：被评价的 FY2025 预测正好是面板研报的 {len(j)} 对，六个变量与面板缩尾前值的最大绝对差 '
        + '，'.join(f'{c} {v:.1e}' for c, v in diffs.items()) + f'；缺失不一致 {nmis}')

    # 缩尾和换单位
    for c in PCT:
        Y[c] = Y[c] * 100
        Y[c + '_24'] = Y[c + '_24'] * 100
    bnd = winsor_by(Y, Y_W + CTRL + [c + '_24' for c in CTRL], in_u)
    log('缩尾界（缩尾全集内 1%/99%）：' + '；'.join(f'{c} [{lo:.4g}, {hi:.4g}]' for c, (lo, hi) in bnd.items()))

    # 合并到样本对
    def finish(P_):
        P_ = P_.merge(Y, on=['code', 'ORG_CL'], how='left', validate='1:1')
        P_['broker_g'] = P_.ORG_CL.map(bg)
        P_['sw'] = P_.code.map(sw)
        for x in XW:  # 每个变量在各自非缺失的样本对内缩尾（R.winsor 只用非缺失值求分位数）
            P_[x] = R.winsor(P_[x + '_RAW'])
        P_['report_id'] = P_.key  # regress_each 用 report_id 记进入回归的观测
        P_['base'] = True
        P_['TY25'], P_['TY24'] = 2025, 2024
        P_['LNHORIZON25'] = P_['LNHORIZON24'] = np.nan  # set_target_year 需要这两列；不进回归
        return P_
    P = finish(P)
    if xset == 'rt6a':  # 分析师 = 提供 X 的那篇（该对 2024 年最新一篇深度研报）的第一作者
        ak = analyst_keys(d2[['report_id', 'code', 'broker_g']]).set_index('report_id')
        P['analyst'] = P.X_report_id.map(ak.analyst)
        log(f'分析师（EST_COMP_TN AUTHOR 第一作者，键 broker_g|名字，取 X_report_id 那篇）：791 篇中有 AUTHOR 的 {int(ak.AUTHOR.notna().sum())} 篇；'
            f'样本对 {len(P)} 个中 analyst 缺失 {int(P.analyst.isna().sum())} 个，分析师 {P.analyst.nunique()} 位')
    v5 = pd.read_parquet(T.OUT_V5 / 'sample.parquet', columns=['report_id', 'PMAFE_FY25', 'PMAFE_FY24'])
    m5 = s_ok[['report_id', 'code', 'ORG_CL']].merge(v5, on='report_id').groupby(['code', 'ORG_CL'])[['PMAFE_FY25', 'PMAFE_FY24']].mean()
    P = P.merge(m5.add_prefix('V5_').reset_index(), on=['code', 'ORG_CL'], how='left')

    # 被评价预测与深度研报的关系
    ids = P.report_ids.str.upper().str.split(';')
    P['EVAL_IS_DEEP'] = [isinstance(e, str) and e.upper() in set(i) for e, i in zip(P.EVAL_RPT_ID25, ids)]
    if 'X_report_id' in P:
        P['EVAL_IS_XREPORT'] = [isinstance(e, str) and isinstance(x, str) and e.upper() == x.upper() for e, x in zip(P.EVAL_RPT_ID25, P.X_report_id)]
    P['EVAL_GAP'] = (pd.to_datetime(P.EVAL_DT25, format='%Y%m%d') - pd.to_datetime(P.last_deep_dt, format='%Y%m%d')).dt.days
    g_ = P.EVAL_GAP.dropna()
    log(f'被评价的 FY2025 预测：有 {int(P.EVAL_DT25.notna().sum())} 对；正好出自该对名下某篇深度研报的 {int(P.EVAL_IS_DEEP.sum())} 对'
        f'（{P.EVAL_IS_DEEP.mean():.3f}）' + (f'，正好出自所用那篇的 {int(P.EVAL_IS_XREPORT.sum())} 对' if 'EVAL_IS_XREPORT' in P else '')
        + f'；发布日 − 该对最后一篇深度研报日（天）：均值 {g_.mean():.1f}、p10/p50/p90 = '
        f'{g_.quantile(.1):.0f}/{g_.median():.0f}/{g_.quantile(.9):.0f}、最大 {g_.max():.0f}、为 0 的 {int((g_ == 0).sum())} 对、为负的 {int((g_ < 0).sum())} 对')
    neg = P[P.EVAL_GAP < 0]
    if len(neg):
        rr = s_ok[s_ok.report_id.isin([r for x in neg.report_ids for r in x.split(';')])][['report_id', 'code', 'ORG_CL', 'date', 'RPT_YR_T']]
        rr = rr.merge(d2[['report_id', 'F25']], on='report_id')
        log('  被评价预测早于深度研报的对：\n' + neg[['code', 'ORG_CL', 'last_deep_dt', 'EVAL_DT25', 'EVAL_RPT_ID25']].to_string(index=False)
            + '\n  这些对名下的深度研报（F25 = 本篇对 FY2025 的预测）：\n' + rr.to_string(index=False))
    log(f'没有被评价 FY2025 预测的对 {int(P.EVAL_DT25.isna().sum())}；比较组 < {MIN_GROUP} 家的 {int((P.N_GROUP25 < MIN_GROUP).sum())}；'
        f'A25 缺失 {int((P.EVAL_DT25.notna() & P.A25.isna()).sum())}')
    for tag in ('25', '24'):
        n = P[f'N_GROUP{tag}'].dropna()
        log(f'FY{tag} 比较组家数（样本对）：均值 {n.mean():.2f}、p10/p50/p90 = {n.quantile(.1):.0f}/{n.median():.0f}/{n.quantile(.9):.0f}、最小 {n.min():.0f}、最大 {n.max():.0f}')
    for y in ['PMAFE_FY25', 'PMAFE_FY24', 'PMAFEX_FY25', 'RA_FY25', 'SCORE_FY25', 'SCORE_FY24', 'DAGE25', 'DAGE24']:
        col = y + '_RAW' if y + '_RAW' in P else y
        x = P[col].dropna()
        q = x.quantile([.01, .1, .5, .9, .99])
        t_, tcl, G = T.cluster_mean_t(x, P.loc[x.index, 'code'])
        log(f'{y}{"（缩尾前）" if col != y else ""}：{len(x)} 对，均值 {x.mean():.4f}（t {t_:.2f}，按公司聚类 t {tcl:.2f}，{G} 家），标准差 {x.std():.4f}，'
            f'p1/p10/p50/p90/p99 = ' + '/'.join(f'{v:.3f}' for v in q) + f'，最小 {x.min():.3f}，最大 {x.max():.3f}')
    cols = ['PMAFE_FY25', 'SCORE_FY25', 'RA_FY25', 'PMAFEX_FY25', 'V5_PMAFE_FY25', 'PMAFE_FY24']
    pc, sc = P[cols].corr(), P[cols].rank().corr()
    log('相关（样本对，Pearson/Spearman）：' + '；'.join(f'PMAFE_FY25–{c} {pc.loc[cols[0], c]:.3f}/{sc.loc[cols[0], c]:.3f}' for c in cols[1:]))
    log('非缺失对数：' + '，'.join(f'{y} {int(P[y].notna().sum())}' for y in YS_LADDER) + f'，DAGE25 {int(P.DAGE25.notna().sum())}')
    if xset == 'val3':
        d_ = P[[x + '_RAW' for x in XS]]
        log('样本对的三个 X（原始值）：' + '；'.join(f'{x} 均值 {d_[x + "_RAW"].mean():.3f}、标准差 {d_[x + "_RAW"].std():.3f}、p10/p50/p90 '
                                                f'{d_[x + "_RAW"].quantile(.1):.3f}/{d_[x + "_RAW"].median():.3f}/{d_[x + "_RAW"].quantile(.9):.3f}' for x in XS))
        c3 = P[XS].corr()
        log(f'样本对的三个 X（缩尾后）相关：RATIO–LNPATH {c3.iloc[0, 1]:.3f}，RATIO–PATHLEN {c3.iloc[0, 2]:.3f}，LNPATH–PATHLEN {c3.iloc[1, 2]:.3f}')

    # ---------------------------------------------------------------- 回归
    xs = {x: x for x in XS}
    pbc = {p: PANS[p] for p in ('B', 'C')}
    tasks = []

    def add(block, data, ys, pans, xs_, ctrl_extra=('DAGE',), joint=False, spec_name=None, ctrl24=True):
        for y in ys:
            ctrl = CTRL + [dage_col(y) if c == 'DAGE' else c for c in ctrl_extra]
            # FY24：控制变量在 FY24 被评价预测日上（公司-券商层面）；研报层面（ctrl24=False）一律用研报日上的
            dd = data.assign(**{c: data[c + '_24'] for c in CTRL}) if (ctrl24 and y.endswith('24')) else data
            cols_ = list(dict.fromkeys(['report_id', 'code', 'base', 'broker_g', 'sw', 'TY25', 'TY24', 'LNHORIZON25', 'LNHORIZON24', y]
                                       + [c for fe_, _ in pans.values() for c in fe_ if c in data.columns]  # 如 analyst（原有各版这里不增列）
                                       + list(xs_.values()) + ctrl))
            for p, fe_ctrl in pans.items():  # 每个（块，Y，面板）一个任务；结果按这个顺序拼回，与串行一致
                tasks.append((block, spec_name, (dd[cols_], xs_, y, {p: fe_ctrl}, ctrl, joint)))

    same = P.EVAL_GAP <= SAME_DAYS
    log(f'same（被评价预测距最后一篇深度研报 ≤ {SAME_DAYS} 天）{int(same.sum())} 对，later {int((P.EVAL_GAP > SAME_DAYS).sum())} 对')
    if xset not in ('rt6', 'rt6a'):
        add('main', P, ['PMAFE_FY25'], {'C': PANS['C']}, xs)
    if xset == 'rt6a':
        add('ladder', P, ['PMAFE_FY25', 'SCORE_FY25'], PANS13, xs)
        for combo in RT6_PAIRS:
            add('ref', P, ['PMAFE_FY25'], {p: PANS13[p] for p in ('B', 'D', 'C')}, {c: c for c in combo}, joint=True, spec_name='+'.join(combo))
    elif xset == 'rt6':
        add('ladder', P, ['PMAFE_FY25', 'SCORE_FY25'], PANS, xs)
        for combo in RT6_PAIRS:
            add('ref', P, ['PMAFE_FY25'], pbc, {c: c for c in combo}, joint=True, spec_name='+'.join(combo))
        add('joint', P, ['PMAFE_FY25'], pbc, xs, joint=True, spec_name='joint6')
    elif xset == 'new11':
        add('ladder', P, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25', 'SCORE_FY24'], PANS, xs)
        add('nodage', P, ['PMAFE_FY25'], PANS, xs, ctrl_extra=())
        add('joint', P, ['PMAFE_FY25', 'SCORE_FY25'], pbc, xs, joint=True, spec_name='joint11')
        add('same', P[same], ['PMAFE_FY25'], pbc, xs)
        add('later', P[P.EVAL_GAP > SAME_DAYS], ['PMAFE_FY25'], pbc, xs)
        with stage('研报层面因变量（v5 口径）'):
            Rr = report_level(d2, lg, nm, s)
        add('rmain', Rr, ['PMAFE_FY25'], {'C': PANS['C']}, xs, ctrl_extra=(), ctrl24=False)
        add('rladder', Rr, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'], PANS, xs, ctrl_extra=(), ctrl24=False)
        add('rjoint', Rr, ['PMAFE_FY25'], pbc, xs, ctrl_extra=(), joint=True, spec_name='joint11', ctrl24=False)
    elif xset == 'new13':
        y4 = ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25', 'SCORE_FY24']
        lg2 = {x: x for x in XS_13V}
        add('ladder', P, y4, PANS, xs)
        add('nodage', P, ['PMAFE_FY25'], PANS, xs, ctrl_extra=())
        for combo in REF_COMBOS:
            add('ref', P, ['PMAFE_FY25', 'SCORE_FY25'], pbc, {c: c for c in combo}, joint=True, spec_name='+'.join(combo))
        add('joint', P, ['PMAFE_FY25', 'SCORE_FY25'], pbc, xs, joint=True, spec_name='joint13')
        add('same', P[same], ['PMAFE_FY25'], pbc, xs)
        add('later', P[P.EVAL_GAP > SAME_DAYS], ['PMAFE_FY25'], pbc, xs)
        add('logsrc', P, ['PMAFE_FY25'], PANS, lg2)
        log(f'公司-券商层面 RATE_NSRC 非缺失 {int(P.RATE_NSRC.notna().sum())} 对（共 {len(P)} 对）；其余 12 个 X 非缺失：'
            + '，'.join(f'{x} {int(P[x].notna().sum())}' for x in XS[1:]))
        with stage('研报层面因变量（[d−90, d] 与 [d−90, d+30]）'):
            Rr, Rr30 = report_level13(d2, lg, nm)
        add('rmain', Rr, ['PMAFE_FY25'], {'C': PANS['C']}, xs, ctrl_extra=(), ctrl24=False)
        add('rladder', Rr, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'], PANS, xs, ctrl_extra=(), ctrl24=False)
        for combo in REF_COMBOS:
            add('rref', Rr, ['PMAFE_FY25', 'SCORE_FY25'], pbc, {c: c for c in combo}, ctrl_extra=(), joint=True, spec_name='+'.join(combo), ctrl24=False)
        add('rjoint', Rr, ['PMAFE_FY25'], pbc, xs, ctrl_extra=(), joint=True, spec_name='joint13', ctrl24=False)
        add('rlogsrc', Rr, ['PMAFE_FY25'], PANS, lg2, ctrl_extra=(), ctrl24=False)
        add('rwin30', Rr30, ['PMAFE_FY25'], pbc, xs, ctrl_extra=(), ctrl24=False)
    else:
        add('ladder', P, YS_LADDER, PANS, xs)
        add('nodage', P, ['PMAFE_FY25', 'SCORE_FY25'], PANS, xs, ctrl_extra=())
    if xset == 'rt11':
        for m in MEAS2:
            for block, cols_ in (('horse', ['RT_' + m, 'WG_' + m, 'LNCHARS']), ('horse2', ['RT_' + m, 'LNCHARS'])):
                add(block, P, YS_HORSE, pbc, {c: c for c in cols_}, joint=True, spec_name='+'.join(cols_))
    if xset not in ('new11', 'new13', 'rt6', 'rt6a'):
        add('same', P[same], ['PMAFE_FY25', 'SCORE_FY25'], pbc, xs)
        add('later', P[P.EVAL_GAP > SAME_DAYS], ['PMAFE_FY25', 'SCORE_FY25'], pbc, xs)
    if xset == 'val3':
        add('joint', P, ['PMAFE_FY25', 'SCORE_FY25'], pbc, xs, joint=True, spec_name='+'.join(XS))
        # mean：同一批对，X 换成该对名下各篇研报的平均（在样本对内缩尾）
        Pm = P.drop(columns=XS + [x + '_RAW' for x in XS]).merge(P_mean_x, on=['code', 'ORG_CL'], how='left', validate='1:1')
        for x in XS:
            Pm[x] = R.winsor(Pm[x + '_RAW'])
        nchg = int(((Pm[[x + '_RAW' for x in XS]].to_numpy() - P[[x + '_RAW' for x in XS]].to_numpy()) != 0).any(axis=1).sum())
        log(f'mean 对照：X 取平均与取最新一篇不同的对 {nchg} 个')
        add('mean', Pm, ['PMAFE_FY25'], pbc, xs)
    with stage('回归'):
        res = run_tasks(tasks, jobs)
    log(f'回归任务 {len(tasks)} 个，并行进程数 {min(jobs, len(tasks)) if jobs > 1 else 1}')
    res = pd.concat(res, ignore_index=True)[T.COLS]
    if xset in ('new11', 'new13'):
        res.insert(0, 'level', np.where(res.block.isin(['rmain', 'rladder', 'rref', 'rjoint', 'rlogsrc', 'rwin30']), 'report', 'pair'))
    assert not res.duplicated(['spec', 'term']).any()
    res.to_csv(OUT / 'reg_all.csv', index=False, encoding='utf-8-sig')
    got = res.groupby('block').spec.nunique().to_dict()
    k = len(XS)
    exp = {'main': k, 'ladder': 6 * 4 * k, 'nodage': 2 * 4 * k, 'same': 2 * 2 * k, 'later': 2 * 2 * k}
    exp.update({'horse': 3 * 2 * 5, 'horse2': 3 * 2 * 5} if xset == 'rt11' else {'joint': 2 * 2, 'mean': 2 * k})
    if xset == 'new11':
        exp = {'main': k, 'ladder': 4 * 4 * k, 'nodage': 4 * k, 'joint': 2 * 2, 'same': 2 * k, 'later': 2 * k,
               'rmain': k, 'rladder': 3 * 4 * k, 'rjoint': 2}
    if xset == 'new13':
        exp = {'main': k, 'ladder': 4 * 4 * k, 'nodage': 4 * k, 'ref': 3 * 2 * 2, 'joint': 2 * 2, 'same': 2 * k, 'later': 2 * k, 'logsrc': 4 * 2,
               'rmain': k, 'rladder': 3 * 4 * k, 'rref': 3 * 2 * 2, 'rjoint': 2, 'rlogsrc': 4 * 2, 'rwin30': 2 * k}
    if xset == 'rt6':
        exp = {'ladder': 2 * 4 * k, 'ref': 3 * 2, 'joint': 2}
    if xset == 'rt6a':
        exp = {'ladder': 2 * 5 * k, 'ref': 3 * 3}
    log(f'reg_all.csv：{len(res)} 行；回归数 {got}' + ('，与设计相符' if got == exp else f'（期望 {exp}）'))
    if xset in ('new11', 'new13'):
        sig_summary(res)
    for blk, ys_ in (('ladder', YS_LADDER), ('same', ['PMAFE_FY25']), ('later', ['PMAFE_FY25']), ('rladder', ['PMAFE_FY25'])):
        for y in ys_:
            for p in 'NABDC':  # D 只在 rt6a 里有
                q = res[(res.block == blk) & (res.Y == y) & (res.panel == p)]
                if len(q):
                    log(f'  N/公司数 {blk} {y} {p}：{int(q.n.iloc[0])} / {int(q.n_clusters.iloc[0])}，单例 {int(q.n_singletons.iloc[0])}')

    # 自检：独立 OLS
    with stage('自检'):
        P24 = P.assign(**{c: P[c + '_24'] for c in CTRL})
        checks = [('main', P, 'PMAFE_FY25', 'C', [XS[0]], CTRL + ['DAGE25']),
                  ('ladder', P24, 'SCORE_FY24', 'B', [XS[-1]], CTRL + ['DAGE24']),
                  ('nodage', P, 'PMAFE_FY25', 'A', [XS[1]], CTRL)]
        if xset == 'rt11':
            checks += [('horse', P, 'PMAFE_FY25', 'C', ['RT_LNPATH', 'WG_LNPATH', 'LNCHARS'], CTRL + ['DAGE25']),
                       ('same', P[same], 'SCORE_FY25', 'C', ['RT_DEPTH'], CTRL + ['DAGE25'])]
        elif xset == 'new11':
            checks = [('main', P, 'PMAFE_FY25', 'C', ['NEG_SHARE'], CTRL + ['DAGE25']),
                      ('ladder', P24, 'SCORE_FY24', 'B', ['DENSITY'], CTRL + ['DAGE24']),
                      ('nodage', P, 'PMAFE_FY25', 'A', ['EARN_SHARE'], CTRL),
                      ('joint', P, 'SCORE_FY25', 'C', XS, CTRL + ['DAGE25']),
                      ('later', P[P.EVAL_GAP > SAME_DAYS], 'PMAFE_FY25', 'B', ['MID_SHARE'], CTRL + ['DAGE25']),
                      ('rmain', Rr, 'PMAFE_FY25', 'C', ['ISO_SHARE'], CTRL),
                      ('rladder', Rr, 'PMAFE_FY24', 'N', ['EXT_SRC'], CTRL),
                      ('rjoint', Rr, 'PMAFE_FY25', 'B', XS, CTRL)]
        elif xset == 'new13':
            checks = [('main', P, 'PMAFE_FY25', 'C', ['RATE_NSRC'], CTRL + ['DAGE25']),
                      ('ladder', P24, 'PMAFE_FY24', 'B', ['ALL_LNNODE'], CTRL + ['DAGE24']),
                      ('nodage', P, 'PMAFE_FY25', 'A', ['ALL_NSRC'], CTRL),
                      ('ref', P, 'SCORE_FY25', 'C', ['RATE_NSRC', 'ALL_NSRC', 'ALL_LNNODE'], CTRL + ['DAGE25']),
                      ('joint', P, 'PMAFE_FY25', 'B', XS, CTRL + ['DAGE25']),
                      ('logsrc', P, 'PMAFE_FY25', 'N', ['RATE_LNSRC'], CTRL + ['DAGE25']),
                      ('rmain', Rr, 'PMAFE_FY25', 'C', ['RATE_NSRC'], CTRL),
                      ('rladder', Rr, 'SCORE_FY25', 'B', ['NEG_SHARE'], CTRL),
                      ('rref', Rr, 'PMAFE_FY25', 'C', ['RATE_NSRC', 'ALL_LNNODE'], CTRL),
                      ('rjoint', Rr, 'PMAFE_FY25', 'C', XS, CTRL),
                      ('rwin30', Rr30, 'PMAFE_FY25', 'B', ['RATE_NSRC'], CTRL)]
        elif xset == 'rt6a':
            checks = [('ladder', P, 'PMAFE_FY25', 'C', ['RT_NSRC'], CTRL + ['DAGE25']),
                      ('ladder', P, 'SCORE_FY25', 'D', ['WG_PATHLEN'], CTRL + ['DAGE25']),
                      ('ref', P, 'PMAFE_FY25', 'C', ['RT_LNPATH', 'WG_LNPATH'], CTRL + ['DAGE25'])]
        elif xset == 'rt6':
            checks = [('ladder', P, 'PMAFE_FY25', 'C', ['RT_NSRC'], CTRL + ['DAGE25']),
                      ('ladder', P, 'SCORE_FY25', 'A', ['WG_PATHLEN'], CTRL + ['DAGE25']),
                      ('ref', P, 'PMAFE_FY25', 'B', ['RT_LNPATH', 'WG_LNPATH'], CTRL + ['DAGE25']),
                      ('joint', P, 'PMAFE_FY25', 'C', XS, CTRL + ['DAGE25'])]
        else:
            checks += [('joint', P, 'SCORE_FY25', 'C', XS, CTRL + ['DAGE25']), ('same', P[same], 'PMAFE_FY25', 'B', [XS[1]], CTRL + ['DAGE25']),
                       ('mean', Pm, 'PMAFE_FY25', 'C', [XS[2]], CTRL + ['DAGE25'])]
        for block, data, y, pan, xx, ctrl in checks:
            o_ = T.own_ols(F.y_sample(data, y), y, xx, ctrl, PAN_[pan][0])
            r_ = res[(res.block == block) & (res.Y == y) & (res.panel == pan) & (res.X == ({'new11': 'joint11', 'new13': 'joint13', 'rt6': 'joint6'}[xset] if (xset in ('new11', 'new13', 'rt6') and xx == XS) else '+'.join(xx)))].set_index('term').loc[xx + ctrl]
            dc = np.max(np.abs(r_.coef.to_numpy() - o_['coef']) / np.maximum(np.abs(o_['coef']), 1e-8))
            ds = np.max(np.abs(r_.se.to_numpy() - o_['se']) / o_['se'])
            meta = (r_.n.iloc[0] == o_['n']) and (r_.n_clusters.iloc[0] == o_['G']) and (r_.n_singletons.iloc[0] == o_['n_single'])
            log(f'  独立 OLS [{block} {y} {pan} {"+".join(xx)}]：系数相对差 {dc:.1e}，se 相对差 {ds:.1e}，N/G/单例 {"一致" if meta else "不一致"}')
            # rt6a 的 C 档样本小、虚拟变量多（225 对、公司 + 分析师），两种解法（pinv / lstsq）的系数相对差可到 1e-6 量级，放宽到 1e-5
            tol_ = 1e-5 if xset == 'rt6a' else 1e-6
            assert meta and dc < tol_ and ds < tol_
        # 从预测表直接手算一对的 PMAFE_FY25（另一条代码路径）
        kk_ = P.dropna(subset=['PMAFE_FY25']).iloc[len(P) // 3]
        e = est[(est.SEC_CD == kk_.code)].copy()
        rows = []
        for kk in (1, 2, 3):
            x = e[(e.RPT_YR_T + kk == 2025) & (e.RPT_DT.str[:4] == '2024') & e[f'PROFIT_T{kk}'].notna()]
            rows.append(x.assign(F=x[f'PROFIT_T{kk}'])[['ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ', 'F']])
        x = pd.concat(rows).sort_values(['ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ'], kind='mergesort').groupby('ORG_CL').tail(1)
        afe = (x.F - ann.A[(kk_.code, 2025)]).abs()
        own = afe[x.ORG_CL == kk_.ORG_CL].iloc[0]
        log(f'  手算一对（{kk_.code}，{kk_.ORG_CL}）：比较组 {len(x)} 家（表中 {int(kk_.N_GROUP25)}），PMAFE = {(own - afe.mean()) / afe.mean():.6f}'
            f'（表中缩尾前 {kk_.PMAFE_FY25_RAW:.6f}）')

    # ---------------------------------------------------------------- 输出
    keep = (['code', 'ORG_CL', 'broker_g'] + [c for c in ('analyst',) if c in P] + ['sw', 'n_reports', 'report_ids', 'last_deep_dt']
            + [c for c in ('X_report_id', 'X_report_dt') if c in P]
            + extra + ['EVAL_IS_DEEP', 'EVAL_GAP'] + [c for c in ('EVAL_IS_XREPORT',) if c in P]
            + [c for c in Y.columns if c not in ('code', 'ORG_CL', 'key')] + XS + [x + '_RAW' for x in XS] + ['V5_PMAFE_FY25', 'V5_PMAFE_FY24'])
    if xset == 'rt11':  # 保持 v6 / v7 的列
        keep = [c for c in keep if c != 'EVAL_IS_XREPORT']
    P[list(dict.fromkeys(keep))].to_parquet(OUT / 'pair_sample.parquet', index=False)
    rows = []
    for v, ver_ in ([(x + '_RAW', 'raw_mean' if agg_mode == 'mean' or xset == 'rt11' else 'raw_last') for x in XS] + [(x, 'winsorized') for x in XS]
                    + [(y + '_RAW', 'raw') for y in Y_W]
                    + [(y, 'regression') for y in Y_W + ['SCORE_FY25', 'SCORE_FY24']] + [(c, 'regression(% 单位)') for c in CTRL]
                    + [(c, 'raw') for c in ['DAGE25', 'DAGE24', 'N_GROUP25', 'N_GROUP24', 'n_reports', 'EVAL_GAP', 'MAFE25', 'AFE25'] + extra]):
        z = P[v].dropna().astype(float)
        q = z.quantile([.01, .1, .25, .5, .75, .9, .99])
        rows.append({'version': ver_, 'var': v.replace('_RAW', ''), 'n': len(z), 'mean': z.mean(), 'sd': z.std(), 'min': z.min(),
                     'p1': q.iloc[0], 'p10': q.iloc[1], 'p25': q.iloc[2], 'p50': q.iloc[3], 'p75': q.iloc[4], 'p90': q.iloc[5],
                     'p99': q.iloc[6], 'max': z.max()})
    pd.DataFrame(rows).to_csv(OUT / 'descriptives.csv', index=False, encoding='utf-8-sig')
    c_ = P[XS].corr()
    c_.insert(0, 'var', c_.index)
    c_.to_csv(OUT / 'corr.csv', index=False, encoding='utf-8-sig')
    yy = YS_LADDER + ['DAGE25', 'DAGE24', 'V5_PMAFE_FY25', 'V5_PMAFE_FY24']
    yc = []
    for meth in ('pearson', 'spearman'):
        m_ = P[yy].corr() if meth == 'pearson' else P[yy].rank().corr()
        m_.insert(0, 'var', m_.index)
        m_.insert(0, 'method', meth)
        yc.append(m_)
    pd.concat(yc, ignore_index=True).to_csv(OUT / 'y_corr.csv', index=False, encoding='utf-8-sig')
    TIMES['总计'] = time.perf_counter() - t_all
    log('耗时（秒）：' + '；'.join(f'{k_} {v:.1f}' for k_, v in TIMES.items()))
    (OUT / 'checks.txt').write_text('\n'.join(T.log_lines) + '\n', encoding='utf-8')


# ---------------------------------------------------------------- 新指标（v9）
def dag_tables(report_ids):
    """与 D.load_graphs 同一来源的概念表、边表（只 in_dag）和研报表：report_id 重复时用 D.DAG_DIRS 里靠前的目录。"""
    want, owner = set(report_ids), {}
    for i, d in enumerate(D.DAG_DIRS):
        for rid in pd.read_parquet(Path(d) / 'reports.parquet', columns=['report_id']).report_id:
            if rid in want and rid not in owner:
                owner[rid] = i
    pick = lambda t, i: t[[owner.get(r) == i for r in t.report_id]]  # noqa: E731
    C = pd.concat([pick(pd.read_parquet(Path(d) / 'concepts.parquet', columns=['report_id', 'name', 'kind', 'in_dag']), i)
                   for i, d in enumerate(D.DAG_DIRS)], ignore_index=True)
    E = pd.concat([pick(pd.read_parquet(Path(d) / 'edges.parquet', columns=['report_id', 'cause', 'effect', 'direction', 'support', 'in_dag']), i)
                   for i, d in enumerate(D.DAG_DIRS)], ignore_index=True)
    Rp = pd.concat([pick(pd.read_parquet(Path(d) / 'reports.parquet', columns=['report_id', 'n_chars', 'n_nodes', 'n_edges']), i)
                    for i, d in enumerate(D.DAG_DIRS)], ignore_index=True)
    return C, E[E.in_dag], Rp


def new_report_measures(report_ids):
    """每篇研报的 11 个新指标（原始值）和中间量；并逐篇核对图与概念表、边表、研报表一一对应。"""
    import re
    import networkx as nx
    graphs = D.load_graphs(list(report_ids))
    C, E, Rp = dag_tables(report_ids)
    gc, ge, rp = dict(tuple(C.groupby('report_id'))), dict(tuple(E.groupby('report_id'))), Rp.set_index('report_id')
    earn = lambda v: bool(re.search(EARN_RE, v)) and not re.search(VALX_RE, v)  # noqa: E731
    rows, bad = [], {'edges': 0, 'nodes': 0, 'support': 0, 'counts': 0}
    for rid, g in graphs.items():
        c, e = gc[rid], ge[rid]
        bad['edges'] += set(zip(e.cause, e.effect)) != set(g.edges())
        bad['nodes'] += set(c.name[c.in_dag]) != set(g.nodes())
        bad['support'] += any(g.edges[u, v]['support'] != s_ for u, v, s_ in zip(e.cause, e.effect, e.support))
        bad['counts'] += (rp.loc[rid, 'n_nodes'] != g.number_of_nodes()) or (rp.loc[rid, 'n_edges'] != g.number_of_edges())
        n, m = g.number_of_nodes(), g.number_of_edges()
        ind, outd = dict(g.in_degree()), dict(g.out_degree())
        srcs = [v for v in g if ind[v] == 0 and outd[v] > 0]
        Es = [v for v in g if earn(v)]
        U = set(Es)
        for v in Es:
            U |= nx.ancestors(g, v)
        src_e = [v for v in srcs if v in U]  # 能到达 E 的源点（源点本身在 E 里也算）
        ext = [v for v in srcs if re.search(EXT_RE, v)]
        kind = dict(zip(c.name, c.kind))
        n_iso, n_conc = int((~c.in_dag).sum()), len(c)
        lcc = max(len(x) for x in nx.weakly_connected_components(g))
        n_chars = float(rp.loc[rid, 'n_chars'])
        rows.append({'report_id': rid, 'n_nodes': n, 'n_edges': m, 'n_E': len(Es), 'n_U': len(U) if Es else 0, 'n_src': len(srcs),
                     'n_src_E': len(src_e), 'n_src_ext': len(ext), 'n_iso': n_iso, 'n_concepts': n_conc, 'n_lcc': lcc, 'n_chars': n_chars,
                     'EARN_SHARE': len(U) / n if Es else 0.0,
                     'EARN_LNSRC': float(np.log1p(len(src_e))),
                     'MULTI_SUP': float((e.support >= 2).mean()),
                     'NEG_SHARE': float((e.direction == '负向').mean()),
                     'LCC_SHARE': lcc / n,
                     'ISO_SHARE': n_iso / n_conc,
                     'MULTI_CAUSE': sum(ind[v] >= 2 for v in g) / sum(ind[v] >= 1 for v in g),
                     'MID_SHARE': sum(ind[v] > 0 and outd[v] > 0 for v in g) / n,
                     'METRIC_SHARE': sum(kind.get(v) == '指标' for v in g) / n,
                     'EXT_SRC': len(ext) / len(srcs) if srcs else np.nan,
                     'DENSITY': m / n_chars * 1000,
                     'EARN_NODES': '；'.join(Es), 'EXT_SOURCES': '；'.join(ext)})
    out = pd.DataFrame(rows)
    assert not any(bad.values()), bad
    return out


def new_report_stats(nm):
    log(f'新指标：{len(nm)} 篇；图与概念表（in_dag 概念 = 图节点）、边表（in_dag 边 = 图的边，support 相同）、研报表（n_nodes、n_edges）逐篇一致（不一致即报错）；'
        f'n_chars 取自同一来源的 reports.parquet')
    rows = []
    for x in XS_NEW:
        z = nm[x]
        log(f'  {x}：{int(z.notna().sum())} 篇，均值 {z.mean():.4f}、标准差 {z.std():.4f}、p10/p50/p90 = {z.quantile(.1):.4f}/{z.median():.4f}/'
            f'{z.quantile(.9):.4f}、最小 {z.min():.4f}、最大 {z.max():.4f}、取 0 的 {int((z == 0).sum())} 篇')
    c = nm[XS_NEW].corr()
    log('  两两相关（Pearson）：\n' + c.round(3).to_string())
    ln = pd.DataFrame({'ln_n_nodes': np.log(nm.n_nodes), 'ln_n_chars': np.log(nm.n_chars)})
    log('  与 ln(n_nodes)、ln(n_chars) 的相关：' + '；'.join(f'{x} {nm[x].corr(ln.ln_n_nodes):.3f}/{nm[x].corr(ln.ln_n_chars):.3f}' for x in XS_NEW))
    for col, lab in (('EARN_NODES', '匹配 EARN_RE（且不匹配 VALX_RE）的节点名'), ('EXT_SOURCES', '匹配 EXT_RE 的源点名')):
        names = pd.Series([v for s_ in nm[col] for v in set(s_.split('；')) if v])
        vc = names.value_counts()
        log(f'  {lab}：共 {vc.size} 个不同名称；出现篇数前 20：' + '，'.join(f'{k}（{v}）' for k, v in vc.head(20).items()))
    log(f'  中间量：没有盈利类节点的 {int((nm.n_E == 0).sum())} 篇；没有源点的 {int((nm.n_src == 0).sum())} 篇；源点里没有匹配 EXT_RE 的 {int((nm.n_src_ext == 0).sum())} 篇')


def _report_y(d2, lg):
    """研报层面 v5 口径的 PMAFE、SCORE（T.pmafe_v5），PMAFE 在全部基础样本内 1%/99% 缩尾（同 v5）。"""
    d5, _, chk = T.pmafe_v5(d2, lg)
    assert all(v[0] == 0 for v in chk.values()), chk
    out = d5[['report_id', 'base', 'PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25', 'SCORE_FY24', 'N_GRP_FY25', 'N_GRP_FY24']].copy()
    for y in ('PMAFE_FY25', 'PMAFE_FY24'):
        out[y + '_RAW'] = out[y]
        out.loc[out.base, y] = R.winsor(out.loc[out.base, y])
    return out.drop(columns='base')


def report_level(d2, lg, nm, s):
    """研报层面的回归数据：全部 791 篇（回归只用基础样本），Y = v5 口径，X 在基础样本研报内缩尾，控制 = 研报日上的六个（MOM/VOL/ROA_T × 100）。"""
    (yv,), hit = cached('report_y_v5', {'n': len(d2)}, lambda: (_report_y(d2, lg),), 1)
    log(f'缓存：研报层面因变量 {"命中" if hit else "重算"}')
    Rr = d2.drop(columns=[c for c in ('PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25', 'SCORE_FY24') if c in d2]).merge(yv, on='report_id', how='left')
    v5 = pd.read_parquet(T.OUT_V5 / 'sample.parquet', columns=['report_id', 'PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'])
    j = v5.merge(Rr[['report_id', 'PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25']], on='report_id', suffixes=('_v5', ''))
    log('研报层面因变量与 v5 sample.parquet（606 篇）的最大差：' + '，'.join(
        f'{y} {float(np.nanmax(np.abs(j[y] - j[y + "_v5"]))):.1e}' for y in ('PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25')))
    Rr = Rr.merge(nm[['report_id'] + XS_NEW].rename(columns={x: x + '_RAW' for x in XS_NEW}), on='report_id', how='left')
    b = Rr.base.astype(bool)
    for x in XS_NEW:
        Rr[x] = np.nan
        Rr.loc[b, x] = R.winsor(Rr.loc[b, x + '_RAW'])
    for c in PCT:
        Rr[c] = Rr[c] * 100
    log(f'研报层面：基础样本 {int(b.sum())} 篇、{Rr.code[b].nunique()} 家；非缺失 PMAFE_FY25 {int(Rr.PMAFE_FY25[b].notna().sum())}、'
        f'PMAFE_FY24 {int(Rr.PMAFE_FY24[b].notna().sum())}、SCORE_FY25 {int(Rr.SCORE_FY25[b].notna().sum())}')
    return Rr


def new13_measures(nm):
    """v10：在 v9 的指标表上加 RATE_NSRC（“投资评级”是严格汇点时其上游的源点数，= D.five_measures(…, '投资评级', strict=True) 的 NSRC_RAW）、
    ALL_NSRC（整张图入度 0 且出度 > 0 的节点数）、ALL_LNNODE（ln 整张图节点数），以及对照用的 RATE_LNSRC、ALL_LNSRC。"""
    out = nm.copy()
    graphs = D.load_graphs(list(out.report_id))
    rt = D.five_measures(graphs, D.RATING_NODE, strict=True).set_index('report_id')
    wg = D.five_measures(graphs, None).set_index('report_id')
    out['RATE_NSRC'] = out.report_id.map(rt.NSRC_RAW)
    out['ALL_NSRC'] = out.n_src.astype(float)
    out['ALL_LNNODE'] = np.log(out.n_nodes.astype(float))
    out['RATE_LNSRC'] = np.log1p(out.RATE_NSRC)
    out['ALL_LNSRC'] = np.log(out.ALL_NSRC)
    # 核对：ALL_NSRC、ALL_LNNODE 与之前的 WG_NSRC、WG_LNNODE（five_measures 整张图）逐篇相同
    assert np.array_equal(out.ALL_NSRC.to_numpy(), out.report_id.map(wg.NSRC_RAW).to_numpy())
    assert np.allclose(out.ALL_LNNODE.to_numpy(), np.log(out.report_id.map(wg.NNODE_RAW).to_numpy()), rtol=0, atol=1e-12)
    return out


def rt6_measures(report_ids):
    """v12：每篇研报的 RT_*（D.five_measures(graphs, '投资评级', strict=True)，非严格汇点记缺失）和 WG_*（five_measures(graphs, None)）
    的 NSRC、LNPATH、PATHLEN 原始值，以及 RT_LNSRC = ln(RT_NSRC)、WG_LNSRC = ln(WG_NSRC)。"""
    graphs = D.load_graphs(list(report_ids))
    rt = D.five_measures(graphs, D.RATING_NODE, strict=True).set_index('report_id')
    wg = D.five_measures(graphs, None).set_index('report_id')
    out = pd.DataFrame({'report_id': list(report_ids)})
    for pre, m in (('RT', rt), ('WG', wg)):
        for x in ('NSRC', 'LNPATH', 'PATHLEN'):
            out[f'{pre}_{x}'] = out.report_id.map(m[x + '_RAW']).astype(float)
        out[f'{pre}_LNSRC'] = np.log(out[f'{pre}_NSRC'])
    return out


def analyst_keys(df):
    """df：report_id、code、broker_g。EST_COMP_TN 2024 年文件（不限 IS_VLD）按 RPT_ID（= report_id 大写）+ SEC_CD（= code）取 AUTHOR；
    analyst = broker_g|第一作者（AUTHOR 逗号前第一个名字），team = broker_g|完整作者名单（名字排序后逗号拼接）；n_author = 不同 AUTHOR 串个数。"""
    import pyarrow.parquet as pq
    fs = sorted((T.ROOT / 'data' / 'EST_COMP_TN' / '2024').glob('2024-[0-9][0-9].parquet'))
    ids = set(df.report_id.str.upper())
    e = pd.concat([pq.read_table(f, columns=['RPT_ID', 'SEC_CD', 'AUTHOR']).to_pandas() for f in fs], ignore_index=True)
    e = e[e.RPT_ID.str.upper().isin(ids)].assign(RID=lambda t: t.RPT_ID.str.upper())
    m = df[['report_id', 'code']].assign(RID=df.report_id.str.upper()).merge(e, left_on=['RID', 'code'], right_on=['RID', 'SEC_CD'], how='left')
    g = m.groupby('report_id').AUTHOR.agg(lambda x: sorted(set(x.dropna())))
    out = df[['report_id']].copy()
    out['n_author'] = out.report_id.map(g.str.len()).fillna(0).astype(int)
    out['AUTHOR'] = out.report_id.map(g.map(lambda v: v[0] if v else np.nan))
    first = out.AUTHOR.map(lambda v: v.split(',')[0].strip() if isinstance(v, str) else np.nan)
    team = out.AUTHOR.map(lambda v: ','.join(sorted(n.strip() for n in v.split(','))) if isinstance(v, str) else np.nan)
    bg_ = df.broker_g.astype(str).to_numpy()
    out['analyst'] = np.where(out.AUTHOR.notna(), bg_ + '|' + first.astype(str), None)
    out['team'] = np.where(out.AUTHOR.notna(), bg_ + '|' + team.astype(str), None)
    return out


def new13_stats(nm):
    log('v10 新增 X：RATE_NSRC = D.five_measures(graphs, "投资评级", strict=True).NSRC_RAW（非严格汇点记缺失）；ALL_NSRC = 入度 0 且出度 > 0 的节点数'
        '（与 five_measures(graphs, None).NSRC_RAW 逐篇相同）；ALL_LNNODE = ln(节点数)（与 ln NNODE_RAW 相同）')
    for x in ['RATE_NSRC', 'ALL_NSRC', 'ALL_LNNODE', 'RATE_LNSRC', 'ALL_LNSRC']:
        z = nm[x]
        log(f'  {x}：非缺失 {int(z.notna().sum())} 篇，均值 {z.mean():.3f}、标准差 {z.std():.3f}、p10/p50/p90 = {z.quantile(.1):.3f}/{z.median():.3f}/'
            f'{z.quantile(.9):.3f}、最小 {z.min():.3f}、最大 {z.max():.3f}')
    c = nm[XS_13].corr()
    log('  13 个 X 的相关（791 篇原始值，Pearson，两两非缺失）：\n' + c.round(3).to_string())
    log('  RATE_NSRC / ALL_NSRC / ALL_LNNODE 与其余 10 个的相关：' + '；'.join(
        f'{x} {c.loc["RATE_NSRC", x]:.3f}/{c.loc["ALL_NSRC", x]:.3f}/{c.loc["ALL_LNNODE", x]:.3f}' for x in XS_13[3:]))


def _report_y13(d2, lg):
    """研报层面：T.pmafe_v5 的 PMAFE0（[d−90, d]、MAFE 含本篇）和 PMAFE（[d−90, d+30]），PMAFE 在全部基础样本内 1%/99% 缩尾；
    SCORE0 = 100 × (1 − RANK)（RANK 的同行 = [d−90, d] 的 PMAFE 同行，组含本篇时 Hong-Kubik 分数恰为此式）。"""
    d5, _, chk = T.pmafe_v5(d2, lg)
    assert all(v[0] == 0 for v in chk.values()), chk
    out = d5[['report_id', 'base', 'PMAFE0_FY25', 'PMAFE0_FY24', 'PMAFE_FY25', 'N_GRP0_FY25', 'N_GRP0_FY24']].rename(
        columns={'PMAFE_FY25': 'PMAFEW30_FY25'}).copy()
    for y in ('PMAFE0_FY25', 'PMAFE0_FY24', 'PMAFEW30_FY25'):
        out[y + '_RAW'] = out[y]
        out.loc[out.base, y] = R.winsor(out.loc[out.base, y])
    return out.drop(columns='base')


def report_level13(d2, lg, nm):
    """研报层面回归数据（全部 791 篇，回归只用基础样本）：Rr 的 PMAFE_FY25 / PMAFE_FY24 = PMAFE0 口径、SCORE_FY25 = SCORE0；
    Rr30 的 PMAFE_FY25 = [d−90, d+30] 口径（rwin30 用）。X 在基础样本研报内各自缩尾；控制 = 研报日上的六个（MOM/VOL/ROA_T × 100）。"""
    (yv,), hit = cached('report_y13', {'n': len(d2)}, lambda: (_report_y13(d2, lg),), 1)
    log(f'缓存：研报层面因变量 {"命中" if hit else "重算"}')
    base = d2[['report_id', 'base', 'code', 'broker_g', 'sw', 'TY25', 'TY24', 'LNHORIZON25', 'LNHORIZON24', 'RANK_FY25'] + CTRL].merge(
        yv, on='report_id', how='left', validate='1:1')
    base = base.merge(nm[['report_id'] + XS_13 + XS_13V].rename(columns={x: x + '_RAW' for x in XS_13 + XS_13V}), on='report_id', how='left')
    b = base.base.astype(bool)
    for x in XS_13 + XS_13V:
        base[x] = np.nan
        base.loc[b, x] = R.winsor(base.loc[b, x + '_RAW'])
    for c in PCT:
        base[c] = base[c] * 100
    base['SCORE_FY25'] = (100 * (1 - base.RANK_FY25)).where(b)
    v5 = pd.read_parquet(T.OUT_V5 / 'sample.parquet', columns=['report_id', 'PMAFE0_FY25', 'PMAFE0_FY24', 'PMAFE_FY25', 'HK_FY25'])
    j = v5.merge(base[['report_id', 'PMAFE0_FY25', 'PMAFE0_FY24', 'PMAFEW30_FY25', 'SCORE_FY25']], on='report_id', suffixes=('_v5', ''))
    log('研报层面因变量与 v5 sample.parquet（606 篇）的最大差：PMAFE0_FY25 {:.1e}，PMAFE0_FY24 {:.1e}，[d−90,d+30] 的 PMAFE_FY25 {:.1e}，SCORE0 与 HK_FY25 {:.1e}'.format(
        *[float(np.nanmax(np.abs(j[a] - j[b_]))) for a, b_ in (('PMAFE0_FY25', 'PMAFE0_FY25_v5'), ('PMAFE0_FY24', 'PMAFE0_FY24_v5'),
                                                              ('PMAFEW30_FY25', 'PMAFE_FY25'), ('SCORE_FY25', 'HK_FY25'))]))
    Rr = base.assign(PMAFE_FY25=base.PMAFE0_FY25, PMAFE_FY24=base.PMAFE0_FY24)
    Rr30 = base.assign(PMAFE_FY25=base.PMAFEW30_FY25)
    log(f'研报层面：基础样本 {int(b.sum())} 篇、{base.code[b].nunique()} 家；非缺失 PMAFE_FY25（[d−90,d]）{int(Rr.PMAFE_FY25[b].notna().sum())}、'
        f'PMAFE_FY24 {int(Rr.PMAFE_FY24[b].notna().sum())}、SCORE_FY25 {int(Rr.SCORE_FY25[b].notna().sum())}、PMAFE_FY25（[d−90,d+30]）'
        f'{int(Rr30.PMAFE_FY25[b].notna().sum())}；X 非缺失（基础样本）：' + '，'.join(f'{x} {int(Rr[x][b].notna().sum())}' for x in XS_13))
    return Rr, Rr30


def sig_summary(res):
    """各层面、各因变量在面板 B、C 下显著（p < 0.10）的单独进入格子，以及检验总数。"""
    # 核心单独进入块：ladder、nodage、same、later、rladder（main/rmain 与 ladder/rladder 的 PMAFE_FY25 C 格重复，不计）
    single = res[(res.role == 'x') & res.block.isin(['ladder', 'nodage', 'same', 'later', 'rladder'])]
    for lev in ('pair', 'report'):
        q = single[single.level == lev]
        bc = q[q.panel.isin(['B', 'C'])]
        log(f'[{lev}] 单独进入的 X 系数：共 {len(q)} 格（面板 B、C {len(bc)} 格），p < 0.10 的 {int((q.p < .1).sum())} 格（B、C {int((bc.p < .1).sum())} 格），'
            f'p < 0.05 的 {int((q.p < .05).sum())} 格（B、C {int((bc.p < .05).sum())} 格）')
        for (blk, y), g in bc.groupby(['block', 'Y'], sort=False):
            sig = g[g.p < .1]
            log(f'  {blk} {y}（B、C 共 {len(g)} 格）：' + ('；'.join(f'{r.panel} {r.X} {r.coef:.4f}(t {r.t:.2f})' for r in sig.itertuples()) if len(sig) else '无'))
        for blks in (['joint', 'rjoint'], ['ref', 'rref'], ['logsrc', 'rlogsrc'], ['rwin30']):
            jq = res[(res.level == lev) & res.block.isin(blks) & (res.role == 'x') & res.panel.isin(['B', 'C'])]
            if len(jq):
                log(f'  {"/".join(blks)}（面板 B、C 共 {len(jq)} 格）：' + ('；'.join(f'{r.Y} {r.panel} [{r.X}] {r.term} {r.coef:.4f}(t {r.t:.2f})'
                                                                                 for r in jq[jq.p < .1].itertuples()) or '无'))


def val_report_stats(vm, d2):
    """估值类汇点的口径说明和分布（写进 checks）。"""
    log('估值类汇点口径：F.sink_frame(…, "sink_val") → D.sink_measures(targets="VAL", count=False, strict=False)；按类别取汇点时只取严格汇点'
        '（D.strict_sinks：入度 > 0、出度 = 0）；类别 = D.classify_sink：名称匹配 VAL_RE 且不匹配 VAL_EXCLUDE（“市值管理”）为估值类；'
        f'VAL_RE = {D.VAL_RE}；“投资评级”的类别 = {D.classify_sink("投资评级")}。子图 = 这些汇点 + 全部祖先；源点 = 子图里入度为 0 的祖先；'
        '路径 = 源点到任一估值类汇点的不同有向路径（一个汇点在另一个上游时经过它的路径也算），VAL_RATIO = 源点数 ÷ 估值类汇点个数，'
        'VAL_LNPATH = ln(1 + 路径条数)，VAL_PATHLEN = 路径边数的平均。')
    v = vm[vm.VAL_RATIO_RAW.notna()]
    log(f'791 篇里有估值类汇点的 {len(v)} 篇（基础样本内 {int(v.report_id.isin(d2.report_id[d2.base]).sum())} 篇）')
    names = pd.Series([n for s_ in v.VAL_SINKS for n in s_.split('；') if n])
    cnt = names.value_counts()
    log(f'估值类汇点名 {cnt.size} 个；出现篇数最多的前 15 个：' + '，'.join(f'{k}（{n}）' for k, n in cnt.head(15).items()))
    ns = v.N_VAL_SINKS.astype(int)
    log('每篇估值类汇点个数：' + '，'.join(f'{k} 个 {n} 篇' for k, n in ns.value_counts().sort_index().items())
        + f'；均值 {ns.mean():.2f}、中位数 {ns.median():.0f}')
    for x in XS_VAL:
        z = v[x + '_RAW']
        log(f'  {x}（原始值，{len(z)} 篇）：均值 {z.mean():.3f}、标准差 {z.std():.3f}、最小 {z.min():.3f}、p10/p50/p90 = '
            f'{z.quantile(.1):.3f}/{z.median():.3f}/{z.quantile(.9):.3f}、最大 {z.max():.3f}')
    c = v[[x + '_RAW' for x in XS_VAL]].corr()
    sc = v[[x + '_RAW' for x in XS_VAL]].rank().corr()
    log(f'  两两相关（原始值，Pearson/Spearman）：RATIO–LNPATH {c.iloc[0, 1]:.3f}/{sc.iloc[0, 1]:.3f}，RATIO–PATHLEN {c.iloc[0, 2]:.3f}/{sc.iloc[0, 2]:.3f}，'
        f'LNPATH–PATHLEN {c.iloc[1, 2]:.3f}/{sc.iloc[1, 2]:.3f}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--x', choices=['rt11', 'val3', 'new11', 'new13', 'rt6', 'rt6a'], default='rt11',
                    help='rt11：v6/v7 的 11 个 X；val3：v8 的估值类汇点 3 个 X；new11：v9 的 11 个新指标；new13：v10 的 13 个指标（两者都另含研报层面）；'
                         'rt6：v12 公司-券商层面的 6 个 X（研报层面见 table_rating_sink_report.py --ver v12）；'
                         'rt6a：v13，同 rt6 但券商固定效应换成分析师（面板 N/A/B/D/C）')
    ap.add_argument('--agg', choices=['mean', 'last'], default='mean', help='mean：取平均；last：取 2024 年内最新一篇')
    ap.add_argument('--out', default=None, help='另指定输出目录（核对用）')
    ap.add_argument('--jobs', type=int, default=16, help='回归并行进程数（1 = 不开进程池；本机实测 16 最快）')
    ap.add_argument('--no-cache', action='store_true', help='不读写 output_dag/_cache/table10_pair/')
    ap.add_argument('--blas-threads', type=int, default=24, help='工作进程的 BLAS 线程数（24 = 本机 OpenBLAS 默认，逐位复现串行结果）')
    a = ap.parse_args()
    USE_CACHE = not a.no_cache
    BLAS_THREADS = a.blas_threads
    main(a.agg, a.out, a.x, a.jobs)
