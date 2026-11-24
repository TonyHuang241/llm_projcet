"""期限假说检验：为什么 DAG 图结构指标对 FY2024 的相对预测误差基本没影响、对 FY2025 却有。
假说：FY2024 是近期预测（发布时当年业绩已部分公布、各家预测接近），FY2025 是远期预测（分歧大），分析深度只在远期体现。

只读 output_dag/forecast_newdata/ 下已有面板（rating_strict_n/、sink_rev_strict_n/、sink_final_n/，各含 main_only/），不重建面板：
  检验一  描述统计：FY2024 对 FY2025 的误差规模、同行误差和离散度、PMAFE 分布、预测期限、发布月份、发布时 2024 年定期报告披露进度
  检验二  按发布时间看系数：(a) X × LATE（2024-07-01 及以后）；(b) 三个时段（1–4、5–8、9–12 月）各自的 X 效果和相等性检验；
          (c) X × LNHORIZON（去均值）
  检验三  被解释变量换成排名口径 RANK_FY24 / RANK_FY25（本篇绝对误差在“本篇 + 同行”里的分位排名，0–1，越小越准）
Panel N、A、B、C 的固定效应和控制变量同 src/dag_forecast.py，按公司聚类（CR1，p 值用 t(G−1)；联合检验用 F(q, G−1)）。
X 在回归样本内标准化（和原回归一样，全样本标准化，不分时段）。

在仓库根目录运行：
    python experiment_fy_horizon.py
结果写到 output_dag/forecast_newdata/horizon_check/（全部预测行）和 horizon_check/main_only/（主体公司行）：
    t1_compare.csv、t1_pmafe.csv、t1_corr.csv、t1_month.csv、t1_disclosure.csv     检验一
    t2_interact.csv                                                                 检验二（a、b、c 长表）
    t3_rank.csv、t3_rank_corr.csv、rank_panel.parquet                               检验三
    check_baseline.csv                                                              自写估计与原 reg_results.csv 的核对
"""
import sys
from pathlib import Path

import mpmath
import numpy as np
import pandas as pd

from src import dag_forecast as F
from src import dag_outcomes_reg as R
from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
ND = ROOT_DIR / 'output_dag' / 'forecast_newdata'
OUT = ND / 'horizon_check'
SETS = {'rating': 'rating_strict_n', 'rev': 'sink_rev_strict_n', 'final': 'sink_final_n'}
SET_NAME = {'rating': '评级终点', 'rev': '营业收入终点', 'final': '最终指标清单'}
SCOPES = {'all': '', 'main': 'main_only'}
XS = ['R_RATIO', 'R_LNPATH', 'R_PATHLEN']
YS = ['PMAFE_FY24', 'PMAFE_FY25']
LATE_DAY = '20240701'
PERIODS = [('P1', '01', '04'), ('P2', '05', '08'), ('P3', '09', '12')]  # 1–4 月、5–8 月、9–12 月
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'
QDATES = {'Q1': ('20240331', '20240430'), 'H1': ('20240630', '20240831'), 'Q3': ('20240930', '20241031')}  # 报告期、法定截止日
# ==================================================


# ---------------------------------------------------------------- 估计（返回完整方差矩阵，便于线性组合和联合检验）
def f_pvalue(f, q, df2):
    """F(q, df2) 的右尾概率：I_{df2/(df2 + q f)}(df2/2, q/2)。"""
    if not np.isfinite(f) or df2 < 1:
        return np.nan
    return float(mpmath.betainc(df2 / 2, q / 2, 0, df2 / (df2 + q * f), regularized=True))


def fitv(d, y, x, ctrl, fe, build=None, extra=()):
    """同 R.fit 的样本处理（去缺失 → 剔除 singleton → x 在回归样本内标准化），回归元 = build(s)（DataFrame，默认只有 x），
    y ~ 常数 + 回归元 + ctrl + fe 虚拟变量，按公司聚类 CR1。返回 dict(names, b, V, n, G)；只含回归元部分。"""
    cols = list(dict.fromkeys([y, x] + list(extra) + ctrl + fe + ['code']))
    s = d[cols].dropna()
    s, _ = R.drop_singletons(s, fe)
    s = s.copy()
    s[x] = (s[x] - s[x].mean()) / s[x].std()
    regs = build(s) if build else s[[x]]
    X = np.column_stack([np.ones(len(s)), regs.to_numpy(float), s[ctrl].to_numpy(float), R.fe_dummies(s, fe)])
    yv = s[y].to_numpy(float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ yv)
    e = yv - X @ b
    n, k = len(yv), int(np.linalg.matrix_rank(X))
    V, G = R._cr1(XtX_inv, X * e[:, None], s.code.to_numpy(), n, k)
    m = regs.shape[1]
    return {'names': list(regs.columns), 'b': b[1:m + 1], 'V': V[1:m + 1, 1:m + 1], 'n': n, 'G': G}


def lincom(r, w):
    """线性组合 w'b 的估计、标准误、t、p（t(G−1)）。"""
    w = np.asarray(w, float)
    est, se = float(w @ r['b']), float(np.sqrt(w @ r['V'] @ w))
    t = est / se if se > 0 else np.nan
    return est, se, t, R.t_pvalue(t, r['G'] - 1)


def wald(r, Rm):
    """H0: Rm b = 0 的 F 检验 p 值（F = W/q，df = (q, G−1)）。"""
    Rm = np.asarray(Rm, float)
    rb = Rm @ r['b']
    W = float(rb @ np.linalg.pinv(Rm @ r['V'] @ Rm.T) @ rb)
    q = Rm.shape[0]
    return f_pvalue(W / q, q, r['G'] - 1)


def prep(w, y):
    """同 run_regressions：base 样本，LNHORIZON、ty、code_ty 按 y 取；另加 LATE、时段哑变量。"""
    yr = y[-2:]
    d = w[w.base].copy()
    d['LNHORIZON'] = d[F.Y_HORIZON.get(y, f'LNHORIZON{yr}')]
    d['ty'] = d[F.Y_TY.get(y, f'TY{yr}')].astype(str)
    d['code_ty'] = d.code + '_' + d.ty
    d['LATE'] = (d.date >= LATE_DAY).astype(float)
    mon = d.date.str[4:6]
    for p, a, b in PERIODS:
        d[p] = ((mon >= a) & (mon <= b)).astype(float)
    return d


# ---------------------------------------------------------------- 检验一
def qstats(s, qs=(.25, .5, .75)):
    s = pd.Series(s, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    out = {'n': len(s), 'mean': s.mean()}
    out.update({f'p{int(q * 100)}': s.quantile(q) for q in qs})
    return out


def pearson_spearman(a, b):
    m = a.notna() & b.notna()
    a, b = a[m].astype(float), b[m].astype(float)
    return int(m.sum()), float(np.corrcoef(a, b)[0, 1]), float(np.corrcoef(a.rank(), b.rank())[0, 1])


def disclosure_stage(raw):
    """发布时该公司 2024 年已披露到哪一期（公告日 <= 研报日算已披露）。公告日取 financial_data 同一报告期最早的 date，取不到用法定截止日。"""
    f = pd.read_parquet(FINANCIAL, columns=['report_date', 'code', 'date'])
    f = f[f.report_date.isin([v[0] for v in QDATES.values()]) & f.code.isin(set(raw.code))]
    ann = f.groupby(['code', 'report_date']).date.min()
    stage = pd.Series('无 2024 定期报告', index=raw.index)
    n_imp = {}
    for q, (rd, deadline) in QDATES.items():
        a = pd.Series([ann.get((c, rd), np.nan) for c in raw.code], index=raw.index)
        n_imp[q] = int(a.isna().sum())
        a = a.fillna(deadline)
        stage[raw.date >= a] = {'Q1': '已有一季报', 'H1': '已有中报', 'Q3': '已有三季报'}[q]
    return stage, n_imp


def peer_dispersion(detail, raw):
    """每篇研报同行预测的标准差（ddof=1）÷ |同行预测均值|。"""
    g = detail.groupby('report_id').F
    disp = (g.std(ddof=1) / g.mean().abs())
    return raw.report_id.map(disp)


def test1(raw, w, det, out):
    s = raw[raw.base & raw.PMAFE_FY24.notna() & raw.PMAFE_FY25.notna()].copy()
    ws = w.set_index('report_id').loc[s.report_id]
    rows = []
    for t in ('24', '25'):
        A = s[f'A{t}'].abs()
        items = {
            '本篇绝对误差 ÷ |实际|': s[f'ABSERR{t}'] / A,
            '本篇绝对误差 ÷ 发布时市值 ×100': s[f'ABSERR{t}'] / s.MKCAP * 100,
            '同行平均绝对误差 ÷ |实际|': s[f'PEER_ERR{t}'] / A,
            '同行家数': s[f'N_PEER{t}'],
            '同行预测离散度（标准差 ÷ |均值|）': peer_dispersion(det[t], s),
            '预测期限（天）': s[f'HORIZON{t}'],
        }
        for k, v in items.items():
            rows.append({'item': k, 'FY': f'FY20{t}', **qstats(v)})
    pd.DataFrame(rows).to_csv(out / 't1_compare.csv', index=False, encoding='utf-8-sig')
    pm = []
    for y in YS:
        for lab, v in (('缩尾前', s[y]), ('缩尾后', ws[y])):
            v = pd.Series(v.to_numpy(float))
            pm.append({'Y': y, 'version': lab, 'n': v.notna().sum(), 'mean': v.mean(), 'sd': v.std(),
                       **{f'p{q}': v.quantile(q / 100) for q in (1, 5, 25, 50, 75, 95, 99)}})
    pd.DataFrame(pm).to_csv(out / 't1_pmafe.csv', index=False, encoding='utf-8-sig')
    cr = []
    for lab, a, b in (('缩尾前', s.PMAFE_FY24, s.PMAFE_FY25), ('缩尾后', ws.PMAFE_FY24.reset_index(drop=True), ws.PMAFE_FY25.reset_index(drop=True))):
        n, pr, sp = pearson_spearman(a.reset_index(drop=True), b.reset_index(drop=True))
        cr.append({'version': lab, 'n': n, 'pearson': pr, 'spearman': sp})
    pd.DataFrame(cr).to_csv(out / 't1_corr.csv', index=False, encoding='utf-8-sig')
    s['month'] = s.date.str[4:6]
    s.groupby('month').size().rename('n_reports').reset_index().to_csv(out / 't1_month.csv', index=False, encoding='utf-8-sig')
    stage, n_imp = disclosure_stage(s)
    order = ['无 2024 定期报告', '已有一季报', '已有中报', '已有三季报']
    dc = stage.value_counts().reindex(order, fill_value=0).rename('n_reports').reset_index()
    dc.columns = ['stage', 'n_reports']
    dc['imputed_deadline_firm_reports'] = str(n_imp)
    dc.to_csv(out / 't1_disclosure.csv', index=False, encoding='utf-8-sig')
    return len(s), s.code.nunique()


# ---------------------------------------------------------------- 检验三：排名口径
def rank_panel(raw, det):
    """RANK_FY{t}：本篇绝对误差在“本篇 + 同行”里的平均名次，(名次 − 1) ÷ 同行家数，0–1，越小越准；只在 PMAFE_FY{t} 非缺失时有值。
    同时核对同行家数、同行误差均值与面板里的 N_PEER{t}、PEER_ERR{t} 一致。"""
    out = raw[['report_id']].copy()
    for t in ('24', '25'):
        g = {rid: x.F.to_numpy(float) for rid, x in det[t].groupby('report_id')}
        vals, bad = [], 0
        for rid, own, a, npr, pe, pm in zip(raw.report_id, raw[f'ABSERR{t}'], raw[f'A{t}'], raw[f'N_PEER{t}'], raw[f'PEER_ERR{t}'],
                                            raw[f'PMAFE_FY{t}']):
            if pd.isna(pm):
                vals.append(np.nan)
                continue
            pe_abs = np.abs(g[rid] - a)
            bad += (len(pe_abs) != npr) or not np.isclose(pe_abs.mean(), pe)
            less, eq = (pe_abs < own).sum(), np.isclose(pe_abs, own, rtol=1e-12, atol=0).sum()
            less -= ((pe_abs < own) & np.isclose(pe_abs, own, rtol=1e-12, atol=0)).sum()
            rank = less + (eq + 2) / 2  # 1..n+1 的平均名次（本篇和 eq 个同行并列）
            vals.append((rank - 1) / len(pe_abs))
        assert bad == 0, f'FY20{t} 同行集合与面板不一致 {bad} 篇'
        out[f'RANK_FY{t}'] = vals
    return out


# ---------------------------------------------------------------- 主流程
def run_scope(scope):
    sub = SCOPES[scope]
    out = OUT / sub if sub else OUT
    out.mkdir(parents=True, exist_ok=True)
    pans = F.panels(with_nofe=True)
    ref = None
    res2, res3, chk = [], [], []
    rank_corr = []
    for key, dname in SETS.items():
        base_dir = ND / dname / sub if sub else ND / dname
        raw = pd.read_parquet(base_dir / 'forecast_panel_raw.parquet')
        w = pd.read_parquet(base_dir / 'forecast_panel.parquet')
        det = {t: pd.read_parquet(base_dir / f'peer_detail_fy{t}.parquet') for t in ('24', '25')}
        if ref is None:  # 检验一只做一次（Y 和同行与 X 口径无关）
            ref = raw
            n1, f1 = test1(raw, w, det, out)
            print(f'[{scope}] 检验一样本 {n1} 篇 / {f1} 家', flush=True)
            rk = rank_panel(raw, det)
            rk.to_parquet(out / 'rank_panel.parquet', index=False)
            for t in ('24', '25'):
                for lab, pm in (('缩尾前', raw[f'PMAFE_FY{t}']), ('缩尾后', w[f'PMAFE_FY{t}'])):
                    m = raw.base
                    n, pr, sp = pearson_spearman(rk[f'RANK_FY{t}'][m].reset_index(drop=True), pm[m].reset_index(drop=True))
                    rank_corr.append({'FY': f'FY20{t}', 'PMAFE': lab, 'n': n, 'pearson': pr, 'spearman': sp})
        else:  # 三个口径的 Y、同行必须一样
            for c in ['report_id', 'base', 'PMAFE_FY24', 'PMAFE_FY25', 'ABSERR24', 'ABSERR25']:
                assert ref[c].equals(raw[c]), c
        w = w.merge(rk, on='report_id', how='left')
        orig = pd.read_csv(base_dir / 'reg_results.csv')
        for y in YS + ['RANK_FY24', 'RANK_FY25']:
            d = prep(w, y)
            for pan, (fe, ctrl) in pans.items():
                for x in XS:
                    meta = {'set': key, 'set_name': SET_NAME[key], 'scope': scope, 'Y': y, 'panel': pan, 'X': x}
                    r0 = fitv(d, y, x, ctrl, fe)
                    if y.startswith('RANK'):
                        est, se, t, p = lincom(r0, [1])
                        res3.append({**meta, 'coef': est, 'se': se, 't': t, 'p': p, 'n': r0['n'], 'n_clusters': r0['G']})
                        continue
                    # 核对：基础回归与原 reg_results.csv 一致
                    o = orig[(orig.Y == y) & (orig.panel == pan) & (orig.reg == '单独') & (orig.X == x)].iloc[0]
                    chk.append({**meta, 'coef': r0['b'][0], 'coef_orig': o.coef, 't': lincom(r0, [1])[2], 't_orig': o.t,
                                'n': r0['n'], 'n_orig': o.n})
                    # (a) X、LATE、X×LATE
                    ra = fitv(d, y, x, ctrl, fe, lambda s, x=x: pd.DataFrame({x: s[x], 'LATE': s.LATE, 'XxLATE': s[x] * s.LATE}),
                              extra=['LATE'])
                    for term, wv in (('X（上半年）', [1, 0, 0]), ('X×LATE', [0, 0, 1]), ('X+X×LATE（下半年）', [1, 0, 1])):
                        est, se, t, p = lincom(ra, wv)
                        res2.append({**meta, 'test': 'a', 'term': term, 'coef': est, 'se': se, 't': t, 'p': p, 'n': ra['n'],
                                     'n_clusters': ra['G']})
                    # (b) 三个时段各自的 X 效果（X×P1、X×P2、X×P3，时段主效应 P2、P3）
                    rb = fitv(d, y, x, ctrl, fe, lambda s, x=x: pd.DataFrame({'XxP1': s[x] * s.P1, 'XxP2': s[x] * s.P2, 'XxP3': s[x] * s.P3,
                                                                             'P2': s.P2, 'P3': s.P3}), extra=['P1', 'P2', 'P3'])
                    pj = wald(rb, [[1, -1, 0, 0, 0], [0, 1, -1, 0, 0]])
                    for j, (p_, a_, b_) in enumerate(PERIODS):
                        wv = np.zeros(5)
                        wv[j] = 1
                        est, se, t, p = lincom(rb, wv)
                        res2.append({**meta, 'test': 'b', 'term': f'X｜{int(a_)}–{int(b_)} 月', 'coef': est, 'se': se, 't': t, 'p': p,
                                     'n': rb['n'], 'n_clusters': rb['G'], 'p_equal': pj})
                    # (c) X × LNHORIZON（在回归样本内去均值）
                    rc = fitv(d, y, x, ctrl, fe, lambda s, x=x: pd.DataFrame({x: s[x], 'XxLNH': s[x] * (s.LNHORIZON - s.LNHORIZON.mean())}))
                    for term, wv in (('X（期限均值处）', [1, 0]), ('X×LNHORIZON', [0, 1])):
                        est, se, t, p = lincom(rc, wv)
                        res2.append({**meta, 'test': 'c', 'term': term, 'coef': est, 'se': se, 't': t, 'p': p, 'n': rc['n'],
                                     'n_clusters': rc['G']})
    chk = pd.DataFrame(chk)
    chk.to_csv(out / 'check_baseline.csv', index=False, encoding='utf-8-sig')
    assert np.allclose(chk.coef, chk.coef_orig, rtol=1e-8, atol=1e-12) and np.allclose(chk.t, chk.t_orig, rtol=1e-8) and (chk.n == chk.n_orig).all(), \
        '自写估计与原 reg_results.csv 不一致'
    pd.DataFrame(res2).to_csv(out / 't2_interact.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(res3).to_csv(out / 't3_rank.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(rank_corr).to_csv(out / 't3_rank_corr.csv', index=False, encoding='utf-8-sig')
    print(f'[{scope}] 基础回归核对通过（{len(chk)} 个），结果 -> {out}', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    for sc in SCOPES:
        run_scope(sc)
