"""DAG 结构指标回归的估计部分：手写 OLS + 固定效应（虚拟变量）+ 聚类稳健标准误（CR1）和异方差稳健标准误（HC1），
不依赖 statsmodels / scipy。t 分布的 p 值用 mpmath.betainc（正则化不完全贝塔函数）。

  CR1：V = (X'X)⁻¹ (Σ_g X_g'e_g e_g'X_g) (X'X)⁻¹ · G/(G−1) · (N−1)/(N−K)，p 值用 t(G−1)（同 Stata reg ..., vce(cluster)）
  HC1：V = (X'X)⁻¹ (Σ_i e_i² x_i x_i') (X'X)⁻¹ · N/(N−K)，p 值用 t(N−K)
  K 是设计矩阵（含常数项和全部固定效应虚拟变量）的秩。
  固定效应里只有一个观测的组（singleton）先反复剔除到没有为止，再估计。
"""
import math

import mpmath
import numpy as np
import pandas as pd


def t_pvalue(t, df):
    """t(df) 分布的双侧 p 值：P(|T| > |t|) = I_{df/(df+t²)}(df/2, 1/2)。"""
    if t is None or not np.isfinite(t) or df is None or df < 1:
        return np.nan
    return float(mpmath.betainc(df / 2, 0.5, 0, df / (df + t * t), regularized=True))


def winsor(s, p=0.01):
    """按 p、1−p 分位缩尾（只用非缺失值算分位数）。"""
    s = pd.to_numeric(s, errors='coerce').astype(float)
    if s.notna().sum() < 3:
        return s
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def drop_singletons(d, fe_cols):
    """反复剔除各固定效应里只有一个观测的组，返回 (剩下的数据, 剔除的观测数)。"""
    n0 = len(d)
    while fe_cols and len(d):
        bad = np.zeros(len(d), dtype=bool)
        for c in fe_cols:
            code = pd.factorize(d[c])[0]  # 缺失值为 −1，不算 singleton（同 value_counts 不计缺失）
            cnt = np.bincount(code[code >= 0], minlength=max(code.max() + 1, 1))
            bad |= (code >= 0) & (cnt[np.maximum(code, 0)] == 1)
        if not bad.any():
            break
        d = d[~bad]
    return d, n0 - len(d)


def fe_dummies(d, fe_cols):
    """各固定效应的虚拟变量（每个固定效应去掉第一组，常数项另加）。"""
    mats = [pd.get_dummies(d[c].astype(str), drop_first=True, dtype=float).to_numpy() for c in fe_cols]
    return np.column_stack(mats) if mats else np.empty((len(d), 0))


def _cr1(XtX_inv, xe, cluster, n, k):
    """单向聚类 CR1 方差矩阵和聚类数。"""
    _, idx = np.unique(np.asarray(cluster), return_inverse=True)
    G = int(idx.max()) + 1
    S = np.zeros((G, xe.shape[1]))
    np.add.at(S, idx, xe)
    return XtX_inv @ (S.T @ S) @ XtX_inv * G / (G - 1) * (n - 1) / (n - k), G


def ols(y, X, cluster=None):
    """OLS。返回 dict：b、se_cl（CR1，cluster 为 None 时缺失）、se_hc1、r2、n、k（秩）、G。
    cluster 是 n×2 的数组时做双向聚类（Cameron-Gelbach-Miller）：V = V_1 + V_2 − V_12，各项用各自的 CR1 小样本修正，
    G 取两个维度里较小的聚类数（p 值用 t(G−1)）。"""
    n = len(y)
    XtX_inv = np.linalg.pinv(X.T @ X)
    b = XtX_inv @ (X.T @ y)
    e = y - X @ b
    k = int(np.linalg.matrix_rank(X))
    tss = ((y - y.mean()) ** 2).sum()
    out = {'b': b, 'r2': 1 - (e @ e) / tss if tss > 0 else np.nan, 'n': n, 'k': k, 'e': e}
    xe = X * e[:, None]
    out['se_hc1'] = np.sqrt(np.clip(np.diag(XtX_inv @ (xe.T @ xe) @ XtX_inv) * n / (n - k), 0, None))
    if cluster is not None:
        c = np.asarray(cluster)
        if c.ndim == 1:
            V, G = _cr1(XtX_inv, xe, c, n, k)
        else:
            V1, G1 = _cr1(XtX_inv, xe, c[:, 0], n, k)
            V2, G2 = _cr1(XtX_inv, xe, c[:, 1], n, k)
            V12, _ = _cr1(XtX_inv, xe, pd.factorize(pd.MultiIndex.from_arrays([c[:, 0], c[:, 1]]))[0], n, k)
            V, G = V1 + V2 - V12, min(G1, G2)
        out['se_cl'], out['G'] = np.sqrt(np.clip(np.diag(V), 0, None)), G
    else:
        out['se_cl'], out['G'] = np.full(len(b), np.nan), None
    return out


def fit(d, y, xs, controls, fe, cluster, standardize=True, terms=False):
    """一次回归：去掉缺失、剔除 singleton、xs 在回归样本内标准化，y ~ 常数 + xs + controls + fe 虚拟变量，
    标准误按 cluster 聚类（列名；两个列名的列表表示双向聚类）。返回 (每个 x 一行的结果列表, 回归样本)；
    样本不足（去缺失和 singleton 后少于 len(xs) + len(controls) + 5 行，或 n ≤ 设计矩阵的秩 k，或聚类数 < 2）时返回 ([], None)。
    xs 与 controls 不要有重名的列（同一列会进设计矩阵两次）。
    terms=True 时结果列表里还有每个控制变量一行（X 列是变量名），常数项和固定效应虚拟变量不输出。"""
    cl_cols = [cluster] if isinstance(cluster, str) else list(cluster or [])
    cols = list(dict.fromkeys([y] + xs + controls + fe + cl_cols))
    s = d[cols].dropna()
    s, n_single = drop_singletons(s, fe)
    if len(s) < len(xs) + len(controls) + 5:
        return [], None
    s = s.copy()
    for x in xs:
        if standardize:
            sd = s[x].std()
            s[x] = (s[x] - s[x].mean()) / sd if sd > 0 else np.nan
    if s[xs].isna().any().any():
        return [], None
    X = np.column_stack([np.ones(len(s)), s[xs + controls].to_numpy(float), fe_dummies(s, fe)])
    # 上面的样本量下限没算固定效应虚拟变量：自由度 n − k ≤ 0（k 为设计矩阵的秩，含全部虚拟变量）或只有一个聚类时标准误无定义，按样本不足处理
    if len(s) <= X.shape[1] and len(s) <= np.linalg.matrix_rank(X):
        return [], None
    if any(s[c].nunique() < 2 for c in cl_cols):
        return [], None
    cl = None if not cl_cols else s[cl_cols[0]].to_numpy() if len(cl_cols) == 1 else s[cl_cols].astype(str).to_numpy()
    r = ols(s[y].to_numpy(float), X, cl)
    adj = 1 - (1 - r['r2']) * (r['n'] - 1) / (r['n'] - r['k'])
    y_mean, y_sd = s[y].mean(), s[y].std()
    rows = []
    for i, x in enumerate(xs + (controls if terms else []), start=1):
        t_cl = r['b'][i] / r['se_cl'][i] if r['se_cl'][i] > 0 else np.nan
        t_hc = r['b'][i] / r['se_hc1'][i] if r['se_hc1'][i] > 0 else np.nan
        rows.append({'X': x, 'coef': r['b'][i], 'se': r['se_cl'][i], 't': t_cl,
                     'p': t_pvalue(t_cl, r['G'] - 1) if r['G'] else np.nan,
                     'se_hc1': r['se_hc1'][i], 't_hc1': t_hc, 'p_hc1': t_pvalue(t_hc, r['n'] - r['k']),
                     'n': r['n'], 'k': r['k'], 'r2': r['r2'], 'adj_r2': adj, 'n_clusters': r['G'], 'n_singletons': n_single,
                     'y_mean': y_mean, 'y_sd': y_sd})
    return rows, s


def stars(p):
    if p is None or not np.isfinite(p):
        return ''
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''


# ---------------------------------------------------------------- 自检
def within(d, cols, fe_cols, iters=500, tol=1e-12):
    """组内去均值（多个固定效应时交替投影）。"""
    z = d[cols].to_numpy(float).copy()
    groups = [pd.factorize(d[c])[0] for c in fe_cols]
    for _ in range(iters):
        old = z.copy()
        for g in groups:
            sums = np.zeros((g.max() + 1, z.shape[1]))
            np.add.at(sums, g, z)
            z -= (sums / np.bincount(g)[:, None])[g]
        if np.abs(z - old).max() < tol:
            break
    return z


def selfcheck():
    """①虚拟变量法与组内去均值法的系数一致（一个、两个固定效应）；②每个观测自成一类时 CR1 = HC1；
    ③手算的小数据核对 OLS 系数、经典标准误、HC1 标准误；④t 分布 p 值对照已知值。全部通过返回 True。"""
    rng = np.random.default_rng(20261121)
    ok = True
    lines = []

    def check(name, a, b, tol=1e-8):
        nonlocal ok
        a, b = np.atleast_1d(np.asarray(a, float)), np.atleast_1d(np.asarray(b, float))
        diff = float(np.max(np.abs(a - b)))
        passed = diff < tol
        ok &= passed
        lines.append(f'  [{"通过" if passed else "失败"}] {name}：{np.round(a, 10).tolist()} vs {np.round(b, 10).tolist()}（最大差 {diff:.2e}）')

    # ① 固定效应：虚拟变量 vs 组内去均值
    n = 400
    d = pd.DataFrame({'g': rng.integers(0, 15, n), 'h': rng.integers(0, 8, n), 'x1': rng.normal(size=n), 'x2': rng.normal(size=n)})
    d['y'] = 1 + 0.5 * d.x1 - 0.3 * d.x2 + d.g * 0.2 + np.sin(d.h) + rng.normal(size=n)
    for fe in (['g'], ['g', 'h']):
        Xd = np.column_stack([np.ones(n), d[['x1', 'x2']].to_numpy(), fe_dummies(d, fe)])
        b_dummy = ols(d.y.to_numpy(), Xd)['b'][1:3]
        z = within(d, ['y', 'x1', 'x2'], fe)
        b_within = np.linalg.lstsq(z[:, 1:], z[:, 0], rcond=None)[0]
        check(f'固定效应 {"+".join(fe)}：虚拟变量法 vs 组内去均值法的系数', b_dummy, b_within)
    # ② 每个观测自成一类：CR1 = HC1
    X = np.column_stack([np.ones(n), d[['x1', 'x2']].to_numpy()])
    r = ols(d.y.to_numpy(), X, cluster=np.arange(n))
    check('每个观测自成一类：CR1 标准误 vs HC1 标准误', r['se_cl'], r['se_hc1'])
    # ③ 手算：x = 0..4，y = 1,2,4,4,6 -> 斜率 12/10 = 1.2，截距 3.4 − 2.4 = 1.0；残差 0,−.2,.6,−.6,.2，SSE = 0.8
    #    经典 se(斜率) = sqrt(0.8/3/10)；HC0 方差 = Σ(x−x̄)²e² / Sxx² = 0.56/100，HC1 再乘 5/3
    x = np.array([0, 1, 2, 3, 4.]); y = np.array([1, 2, 4, 4, 6.])
    r = ols(y, np.column_stack([np.ones(5), x]))
    check('手算 OLS 系数（截距、斜率）', r['b'], [1.0, 1.2])
    resid_var = (r['e'] @ r['e']) / 3
    check('手算经典标准误（斜率）', math.sqrt(resid_var / 10), math.sqrt(0.8 / 3 / 10))
    check('手算 HC1 标准误（斜率）', r['se_hc1'][1], math.sqrt(0.0056 * 5 / 3))
    # ④ p 值：t=2, df=10 双侧 0.0733880…；t=1.959964, df=∞ 近似 0.05；t=2.228139, df=10 为 0.05
    check('t 分布 p 值（t=2.228139, df=10 -> 0.05）', t_pvalue(2.228139, 10), 0.05, tol=1e-6)
    check('t 分布 p 值（t=2, df=10 -> 0.07338803）', t_pvalue(2.0, 10), 0.0733880347, tol=1e-8)
    check('t 分布 p 值（t=1.959964, df=1e6 -> 0.05）', t_pvalue(1.959964, 1e6), 0.05, tol=1e-5)
    # 聚类和 singleton 的小检查：同一聚类结构下 fit 的结果与 ols 直接调用一致
    d['c'] = d.g
    rows, s = fit(d, 'y', ['x1'], ['x2'], ['g'], 'c', standardize=False)
    Xd = np.column_stack([np.ones(n), d[['x1', 'x2']].to_numpy(), fe_dummies(d, ['g'])])
    r = ols(d.y.to_numpy(), Xd, cluster=d.c.to_numpy())
    check('fit() 与直接 ols() 的系数和聚类标准误', [rows[0]['coef'], rows[0]['se']], [r['b'][1], r['se_cl'][1]])
    # 双向聚类：第二个维度每个观测自成一类时，V_2 = V_12，双向聚类退化为按第一个维度的单向聚类
    d['obs'] = np.arange(n)
    rows2, _ = fit(d, 'y', ['x1'], ['x2'], ['g'], ['c', 'obs'], standardize=False)
    check('双向聚类（第二维每个观测一类）vs 单向聚类的标准误', rows2[0]['se'], rows[0]['se'])
    check('调整 R² = 1 − (1 − R²)(n − 1)/(n − k)', rows[0]['adj_r2'], 1 - (1 - rows[0]['r2']) * (n - 1) / (n - rows[0]['k']))
    return ok, lines
