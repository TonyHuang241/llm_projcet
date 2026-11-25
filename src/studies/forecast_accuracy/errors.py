"""预测误差变量：读 data/forecast_errors.parquet（外部提供的预测误差文件），整理成本研究的被解释变量。

数据文件（约 380 万行，450MB；只按 report_id 读需要的行，不整表读入）
  每篇研报 6 行 = horizon（1、2、3）× metric（'PROFIT' 归母净利润、'EPS'）。用到的列：
    report_id    研报 ID（与本研究研报的 report_id 比较时两边都转大写）；sec 6 位公司代码；d 研报日期；horizon；target_year
    F、A         预测值、实际值；signed_error = A − F；error_status 非空表示算不了（如 announcement_missing）
    D            缩放基数（PROFIT 为市值，EPS 为股价）；raw_fe_abs = 100 × |A − F| / D
    n75、n180    同行组的预测条数；C75、C180 同行一致预期（均值，不含本券商）；C_all75、C_all180 含本券商的一致预期
    M75、M180    同行绝对误差的中位数；S75、S180 同行绝对误差的离散度
    relative_fe_w     = (|A − F| − M_w) / S_w
    incremental_fe_w  = (|F − A| − |C_w − A|) / D；incremental_all_w 同理但用 C_all_w
  同行窗口 75 / 180 的具体取法（往前多少天、是否每家券商取最新一条）以数据文件为准，文件未附说明。

研报层面的被解释变量（wide，每篇研报一行）
  只取 report_id（大写）相同且 sec == 本研报公司代码 code 的行（一篇研报覆盖多家公司时文件可能取了另一家，此时记缺失）；
  error_status 非空的行各变量记缺失。列名 {定义}_{口径}_H{h}，口径 NP = PROFIT、EPS = EPS，h = horizon：
    RFE75、RFE180        relative_fe_75 / 180（无量纲）
    IFE75、IFE180        incremental_fe_75 / 180（占 D 的比例，文件原始单位）
    IFEALL75、IFEALL180  incremental_all_75 / 180（同上）
    RAWFE                raw_fe_abs / 100（换回占 D 的比例）
    PMED75、PMED180      (|signed_error| − M_w) / M_w：用文件的同行中位数自己派生的比例口径（文件没有直接给），M_w = 0 或缺失时缺失
  另有 N75_{口径}_H{h}、N180_{口径}_H{h}（n75、n180，浮点数）和 LNN75_{口径}_H{h} = ln(n75)、LNN180_{口径}_H{h} = ln(n180)
  （条数为 0 时缺失）。
  这些变量都是“越大越不准”。缩尾、并入回归样本在 sample.py 里做。
  主变量由 sample.MAIN_DEF 决定（现为 PMED180，即 Y_MAIN = PMED180_NP_H2：净利润、第 2 个预测年度、用 180 天窗口同行绝对误差
  中位数派生的比例误差，形式上最接近原来自算的 PMAFE）。

公司-券商层面（pair_y）：每对的被评价预测 EVAL_RPT_ID_H2 所在研报，取 report_id 大写相同、sec == code、metric = PROFIT、
  target_year == TY_H2 的那一行的主变量定义（sample.MAIN_DEF，算法同上；error_status 非空时缺失），记 {MAIN_DEF}_NP_H2；
  另给匹配到的行的 horizon。

一致性（consistency、consistency_fields）和中位数核对（median_check，另读 data/EST_COMP_TN）见各函数说明。
"""
import numpy as np
import pandas as pd

from src.config import DATA_DIR

FILE = DATA_DIR / 'forecast_errors.parquet'
METRICS = {'NP': 'PROFIT', 'EPS': 'EPS'}  # 口径 -> 文件的 metric
HS = [1, 2, 3]
# 定义 -> (文件列或派生方式, 说明)
DEFS = {
    'RFE75': ('relative_fe_75', '(|A − F| − M75) / S75'),
    'RFE180': ('relative_fe_180', '(|A − F| − M180) / S180'),
    'IFE75': ('incremental_fe_75', '(|F − A| − |C75 − A|) / D'),
    'IFE180': ('incremental_fe_180', '(|F − A| − |C180 − A|) / D'),
    'IFEALL75': ('incremental_all_75', '(|F − A| − |C_all75 − A|) / D'),
    'IFEALL180': ('incremental_all_180', '(|F − A| − |C_all180 − A|) / D'),
    'RAWFE': ('raw_fe_abs', '|A − F| / D'),
    'PMED75': ('M75', '(|A − F| − M75) / M75（派生）'),
    'PMED180': ('M180', '(|A − F| − M180) / M180（派生）'),
}
NO_SIGN = {'RAWFE'}  # 没有符号含义（恒为正）的定义：一致性表的符号一致率记 NA
READ_COLS = ['report_id', 'sec', 'd', 'horizon', 'target_year', 'metric', 'F', 'A', 'D', 'error_status', 'signed_error', 'raw_fe_abs',
             'n75', 'n180', 'M75', 'M180', 'S75', 'S180', 'relative_fe_75', 'relative_fe_180', 'incremental_fe_75',
             'incremental_fe_180', 'incremental_all_75', 'incremental_all_180']


def yname(defn, metric, h):
    """被解释变量的列名：yname('RFE75', 'NP', 2) -> 'RFE75_NP_H2'。"""
    return f'{defn}_{metric}_H{h}'


ALL_Y = [yname(k, m, h) for h in HS for m in METRICS for k in DEFS]  # 54 列，按 h、口径、定义排列
NCOLS = [f'{v}_{m}_H{h}' for h in HS for m in METRICS for v in ('N75', 'N180')]
LNN = [f'LNN{w}_{m}_H{h}' for h in HS for m in METRICS for w in (75, 180)]


def read_rows(report_ids, columns=READ_COLS):
    """数据文件里 report_id（大写）属于 report_ids 的全部行（不管 sec），report_id 转大写。"""
    import pyarrow.dataset as ds
    ids = sorted({str(x).upper() for x in report_ids if pd.notna(x)})
    r = ds.dataset(FILE).to_table(columns=columns, filter=ds.field('report_id').isin(ids)).to_pandas()
    r['report_id'] = r.report_id.str.upper()
    assert not r.duplicated(['report_id', 'metric', 'horizon']).any(), '数据文件里同一研报、口径、horizon 有重复行'
    return r


def values(r):
    """文件行 -> 各定义的值（DataFrame，列为 DEFS 的键）；error_status 非空的行全部缺失。"""
    out = pd.DataFrame(index=r.index)
    for k, (col, _) in DEFS.items():
        if k.startswith('PMED'):
            m = r[col].where(r[col] != 0)
            out[k] = (r.signed_error.abs() - m) / m
        elif k == 'RAWFE':
            out[k] = r[col] / 100
        else:
            out[k] = r[col]
    out.loc[r.error_status.notna().to_numpy()] = np.nan
    return out


def matched(rows, keys):
    """rows 中与 keys（列 report_id、code）匹配、且 sec == code 的行（带上 code）。"""
    k = keys[['report_id', 'code']].assign(report_id=keys.report_id.str.upper())
    m = rows.merge(k, on='report_id', how='inner')
    return m[m.sec == m.code]


def wide(rows, base):
    """研报层面的宽表：base（列 report_id、code）每篇一行，列 ALL_Y、NCOLS、LNN（未缩尾）；行序同 base。"""
    m = matched(rows, base)
    v = pd.concat([m[['report_id', 'metric', 'horizon', 'n75', 'n180']], values(m)], axis=1)
    out = base[['report_id']].copy()
    key = out.report_id.str.upper()
    for mk, metric in METRICS.items():
        for h in HS:
            s = v[(v.metric == metric) & (v.horizon == h)].set_index('report_id')
            for k in DEFS:
                out[yname(k, mk, h)] = key.map(s[k]).astype(float).to_numpy()
            for w in (75, 180):
                n = key.map(s[f'n{w}']).astype(float)
                out[f'N{w}_{mk}_H{h}'] = n.to_numpy()
                out[f'LNN{w}_{mk}_H{h}'] = np.log(n.where(n > 0)).to_numpy()
    return out[['report_id'] + ALL_Y + NCOLS + LNN]


def pair_y(rows, P, h, defn):
    """公司-券商层面：P（列 code、EVAL_RPT_ID_H{h}、TY_H{h}）每对一行，返回 {defn}_NP_H{h}（defn 为 DEFS 的键，未缩尾）、EVAL_FILE_FOUND
    （被评价研报在文件里找到、sec == code、PROFIT、target_year == TY 的行存在）、EVAL_FILE_HORIZON（那一行的 horizon）。"""
    k = P[['code', f'EVAL_RPT_ID_H{h}', f'TY_H{h}']].rename(columns={f'EVAL_RPT_ID_H{h}': 'report_id', f'TY_H{h}': 'ty'})
    k = k.assign(_i=np.arange(len(k)), report_id=k.report_id.str.upper())
    r = rows[rows.metric == 'PROFIT']
    r = pd.concat([r[['report_id', 'sec', 'target_year', 'horizon']], values(r)[[defn]]], axis=1)
    m = k.dropna(subset=['report_id']).merge(r, on='report_id', how='inner')
    m = m[(m.sec == m.code) & (m.target_year == m.ty.astype('float'))]
    assert not m._i.duplicated().any()
    m = m.set_index('_i')
    idx = pd.Series(np.arange(len(k)))
    return pd.DataFrame({yname(defn, 'NP', h): idx.map(m[defn]).astype(float).to_numpy(),
                         'EVAL_FILE_FOUND': idx.isin(m.index).to_numpy(),
                         'EVAL_FILE_HORIZON': idx.map(m.horizon).astype(float).to_numpy()}, index=P.index)


# ---------------------------------------------------------------- 一致性
def spearman(x, y):
    """Spearman 相关：各自取秩（并列取平均秩）后的 Pearson 相关（venv 里没有 scipy）。"""
    return float(np.corrcoef(x.rank(), y.rank())[0, 1])


def consistency(D):
    """文件的每个定义 × 口径 × 窗口与自算 PMAFE_Hh 的一致性（D = 研报层面回归样本，786 篇；两边都用缩尾后的列）：
    文件变量非缺失篇数、两者都非缺失的篇数、Pearson、Spearman、符号一致率（sign 相同的比例；RAWFE 没有符号含义记 NA）。"""
    rows = []
    for h in HS:
        own = D[f'PMAFE_H{h}']
        for mk in METRICS:
            for k, (_, formula) in DEFS.items():
                y = D[yname(k, mk, h)]
                ok = y.notna() & own.notna()
                a, b = y[ok], own[ok]
                rows.append({'定义': k, '口径': mk, '窗口': f'H{h}', '变量': yname(k, mk, h), '公式': formula,
                             '文件变量非缺失篇数': int(y.notna().sum()), '与自算PMAFE同时非缺失篇数': int(ok.sum()),
                             'Pearson': float(np.corrcoef(a, b)[0, 1]) if ok.sum() > 2 else np.nan,
                             'Spearman': spearman(a, b) if ok.sum() > 2 else np.nan,
                             '符号一致率': np.nan if k in NO_SIGN or not ok.sum() else float((np.sign(a) == np.sign(b)).mean())})
    return pd.DataFrame(rows)


def consistency_fields(rows, S):
    """字段层面的一致性（S = 研究范围，取其中 base 的 786 篇；比较 PROFIT 行与 accuracy 的 TY_Hh、F_Hh、A_Hh）。
    返回 [项目, 窗口, 数值, 说明] 的长表。"""
    b = S[S.base][['report_id', 'code'] + [f'{v}_H{h}' for h in HS for v in ('TY', 'F', 'A')]].copy()
    b['report_id'] = b.report_id.str.upper()
    allm = rows.merge(b[['report_id', 'code']], on='report_id', how='inner')
    out = [{'项目': '基础样本篇数', '窗口': '', '数值': len(b), '说明': '研报层面基础样本'},
           {'项目': '在文件里找到的篇数', '窗口': '', '数值': allm.report_id.nunique(), '说明': 'report_id 大写相同（不管 sec）'},
           {'项目': 'sec ≠ code 的篇数', '窗口': '', '数值': allm.loc[allm.sec != allm.code, 'report_id'].nunique(),
            '说明': '文件取了研报覆盖的另一家公司，这些研报的文件变量记缺失'}]
    m = allm[(allm.sec == allm.code) & (allm.metric == 'PROFIT')].merge(b.drop(columns='code'), on='report_id')
    for h in HS:
        g = m[m.horizon == h]
        ty = g[f'TY_H{h}'].astype(float)
        a_ok = g.A.notna() & g[f'A_H{h}'].notna()
        a, ao = g.A[a_ok], g[f'A_H{h}'][a_ok]
        rel = ((a - ao).abs() / ao.abs()).where(a != ao)
        f_ok = g.F.notna() & g[f'F_H{h}'].notna()
        out += [{'项目': 'sec == code 的 PROFIT 行数', '窗口': f'H{h}', '数值': len(g), '说明': ''},
                {'项目': 'target_year 与 TY_Hh 相同的比例', '窗口': f'H{h}', '数值': float((g.target_year == ty).mean()), '说明': ''},
                {'项目': 'F 与 F_Hh 都非缺失的行数', '窗口': f'H{h}', '数值': int(f_ok.sum()), '说明': ''},
                {'项目': 'F 与 F_Hh 完全相同的比例', '窗口': f'H{h}', '数值': float((g.F[f_ok] == g[f'F_H{h}'][f_ok]).mean()), '说明': '浮点数逐位相等'},
                {'项目': 'A 与 A_Hh 都非缺失的行数', '窗口': f'H{h}', '数值': int(a_ok.sum()), '说明': ''},
                {'项目': 'A 与 A_Hh 完全相同的比例', '窗口': f'H{h}', '数值': float((a == ao).mean()) if len(a) else np.nan, '说明': '浮点数逐位相等'},
                {'项目': 'A 与 A_Hh 不同的篇数', '窗口': f'H{h}', '数值': int((a != ao).sum()), '说明': ''},
                {'项目': 'A 不同时相对差异的中位数', '窗口': f'H{h}', '数值': float(rel.median()) if rel.notna().any() else np.nan,
                 '说明': '|A − A_Hh| / |A_Hh|'},
                {'项目': 'A 不同时相对差异的最大值', '窗口': f'H{h}', '数值': float(rel.max()) if rel.notna().any() else np.nan, '说明': ''}]
    return pd.DataFrame(out)


def median_check(rows, S, window=75):
    """核对 M75 是同行绝对误差的中位数。样本行 = 研报层面基础样本里 sec == code、error_status 为空、M75 非缺失的文件行。
    两个来源找“同行预测”，看是否存在同一公司、同一 target_year 的另一条预测，其 |A − F| 与 M75 相等（相对误差 1e-9 以内，A 用样本行的）：
      数据文件    文件里同一 sec、target_year、metric 的其他研报的行（PROFIT 和 EPS）。文件只收了一部分预测（同行窗口里文件自己的行
                  通常少于 n75 条），所以这一口径找不全；
      预测表      data/EST_COMP_TN（IS_VLD == 1）同一 SEC_CD、目标年度的其他 RPT_ID 的净利润预测（只有 PROFIT）。
    各分 n75 为奇数 / 偶数（奇数时中位数等于某一家的绝对误差，偶数时一般是中间两家的均值）、不限日期 / 对方日期在 [d − window, d] 内。
    返回 [项目, 口径, 数值] 的长表。"""
    import pyarrow.dataset as ds

    from src.accuracy.forecasts import load_est, to_long
    m = matched(rows, S[S.base][['report_id', 'code']])
    m = m[m.error_status.isna() & m.M75.notna()].copy()
    m['odd'] = m.n75 % 2 == 1
    secs, tys = sorted(set(m.sec)), sorted(int(t) for t in set(m.target_year))
    fr = ds.dataset(FILE).to_table(columns=['report_id', 'sec', 'd', 'target_year', 'metric', 'F'],
                                   filter=ds.field('sec').isin(secs) & ds.field('target_year').isin(tys)).to_pandas()
    fr = fr.assign(report_id=fr.report_id.str.upper(), dt=pd.to_datetime(fr.d)).dropna(subset=['F'])
    est = to_long(load_est(secs, years=range(min(tys) - 3, 2026)))
    est = est[est.ty.isin(tys)].assign(report_id=lambda e: e.RPT_ID.str.upper(), dt=lambda e: pd.to_datetime(e.RPT_DT),
                                      metric='PROFIT', sec=lambda e: e.SEC_CD, target_year=lambda e: e.ty)
    out = []
    for src, peers, mm in (('数据文件', fr, m), ('预测表 EST_COMP_TN', est, m[m.metric == 'PROFIT'])):
        grp = {k: g for k, g in peers.groupby(['sec', 'target_year', 'metric'])}
        hit_any, hit_win = [], []
        for r in mm.itertuples():
            g = grp.get((r.sec, r.target_year, r.metric))
            if g is None:
                hit_any.append(False), hit_win.append(False)
                continue
            g = g[g.report_id != r.report_id]
            eq = ((r.A - g.F).abs() - r.M75).abs() <= 1e-9 * max(1.0, abs(r.M75))
            dd = (pd.Timestamp(r.d) - g.dt).dt.days
            hit_any.append(bool(eq.any()))
            hit_win.append(bool((eq & dd.between(0, window)).any()))
        mm = mm.assign(hit_any=hit_any, hit_win=hit_win)
        for metric, g0 in [('全部', mm)] + (list(mm.groupby('metric')) if mm.metric.nunique() > 1 else []):
            for odd, g in ((True, g0[g0.odd]), (False, g0[~g0.odd])):
                lab = f'{src}，{metric}，n75 为{"奇数" if odd else "偶数"}'
                out += [{'项目': '样本行数', '口径': lab, '数值': len(g)},
                        {'项目': '存在 |A − F| = M75 的同行预测的比例（不限日期）', '口径': lab, '数值': float(g.hit_any.mean())},
                        {'项目': f'存在 |A − F| = M75 的同行预测的比例（对方日期在 [d − {window}, d] 内）', '口径': lab,
                         '数值': float(g.hit_win.mean())}]
    return pd.DataFrame(out)
