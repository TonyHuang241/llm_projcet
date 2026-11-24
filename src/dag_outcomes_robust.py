"""DAG 结构指标回归的补充推断（都针对 M1 公司截面）：Freedman–Lane 行业内置换检验、秩相关、按 X 三等分、逐一剔除。
由 src/dag_outcomes.py 调用；估计函数在 src/dag_outcomes_reg.py。

  置换检验  M1 按证监会门类聚类只有约 12 类，聚类 t 的分布不可靠。Freedman–Lane：约简模型（不含 X）得拟合值和残差 e，
            在行业（该口径的固定效应行业）内置换 e，Y* = 拟合值 + 置换后的 e，估完整模型取 X 的聚类 t*；
            双侧 p_perm = (1 + #{|t*| ≥ |t|}) / (B + 1)。用 FWL 先把常数项、控制变量、固定效应从 y、x 里消掉：
            约简模型残差 = M_Z y，置换后 M_Z Y* = M_Z(Π e)，X 的系数 = x̃'ỹ / x̃'x̃，聚类方差 = Σ_g (x̃_g'e_g)² / (x̃'x̃)² · CR1 修正，
            K 用完整模型的秩，所以 t 与完整回归完全一致（运行时核对）。
  秩相关    原始 Spearman；偏 Spearman：Y、X 先在回归样本内转成秩，各自对 main 口径的控制变量（含 Y24）和行业固定效应回归取残差，
            算残差相关，p 用 t = r·sqrt(df / (1 − r²))，df = n − rank(Z) − 1
  三等分    按 X 三等分看 Y 均值，高组 − 低组用 Welch t；调整后：Y、X 都先对控制变量和固定效应取残差，按 X 残差三等分看 Y 残差均值
  逐一剔除  逐一剔除一家公司（剔除后若出现 singleton 一并剔除，X 在剩下的样本里重新标准化）、逐一剔除一个证监会门类、剔除制造业
  R² 分解   研报层面的三个 DAG 指标对发布月份、申万一级、券商三组固定效应回归，总 R² 按 Shapley 分解成三组各自的贡献
"""
import math
import zlib

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R

SEED = 20261121  # 置换检验的随机种子（每个口径 × Y × X 再用名字的 crc32 区分）
B_PERM = 2000  # 置换次数


# ---------------------------------------------------------------- FWL 和置换检验
def _basis(Z):
    """Z 列空间的正交基（SVD，去掉数值上为 0 的方向）。"""
    U, S, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, S > S.max() * max(Z.shape) * np.finfo(float).eps]


def design(s, controls, fe):
    """回归样本 s 上除 X 以外的设计矩阵 Z：常数项 + 控制变量 + 固定效应虚拟变量（与 R.fit 的列一致）。"""
    return np.column_stack([np.ones(len(s)), s[controls].to_numpy(float), R.fe_dummies(s, fe)])


def perm_core(y, x, Z, groups, cluster, B, rng):
    """Freedman–Lane 置换检验的核心（数组版）。groups：在哪些组内置换；cluster：聚类编号。返回 (观测的聚类 t, p_perm)。"""
    n = len(y)
    Q = _basis(Z)
    k = Q.shape[1] + 1
    resid = lambda v: v - Q @ (Q.T @ v)  # noqa: E731
    yt, xt = resid(y), resid(x)
    sxx = xt @ xt
    _, cl = np.unique(cluster, return_inverse=True)
    G = cl.max() + 1
    H = np.zeros((G, n))
    H[cl, np.arange(n)] = 1.0
    c = G / (G - 1) * (n - 1) / (n - k)

    def tstat(E):  # E：n×m 的 M_Z y，返回 m 个聚类 t
        b = xt @ E / sxx
        u = xt[:, None] * (E - np.outer(xt, b))
        return b / np.sqrt(((H @ u) ** 2).sum(axis=0) * c / sxx ** 2)

    t0 = float(tstat(yt[:, None])[0])
    P = np.empty((n, B), dtype=int)
    for g in np.unique(groups):
        ix = np.flatnonzero(groups == g)
        P[ix] = ix[np.argsort(rng.random((len(ix), B)), axis=0)]
    ts = tstat(resid(yt[P]))
    return t0, (1 + int((np.abs(ts) >= abs(t0) - 1e-12).sum())) / (B + 1)


def perm_test(d, y, x, controls, fe, cl, tag, B=B_PERM):
    """一次 M1 单独回归的置换检验：返回 (fit 的聚类 t, FWL 算的聚类 t, p_perm)；样本不足时返回 None。"""
    rows, s = R.fit(d, y, [x], controls, [fe], cl)
    if not rows:
        return None
    rng = np.random.default_rng([SEED, zlib.crc32(tag.encode('utf-8'))])
    t0, p = perm_core(s[y].to_numpy(float), s[x].to_numpy(float), design(s, controls, [fe]), s[fe].to_numpy(), s[cl].to_numpy(), B, rng)
    return rows[0]['t'], t0, p


def selfcheck_perm(n_sim=200, B=199):
    """①FWL 算的聚类 t 与 R.fit 的完整回归一致；②原假设下（y 与 x 条件独立）p_perm 大致均匀：名义 10% 的拒绝率在 5%–16%。"""
    rng = np.random.default_rng(SEED)
    lines = []

    def sim():
        g = np.repeat(np.arange(12), rng.integers(8, 35, 12))
        m = len(g)
        z1, z2 = rng.normal(size=m), rng.normal(size=m)
        ge = rng.normal(size=12)[g]
        x = 0.6 * z1 + ge + rng.normal(size=m)
        y = z1 - 0.5 * z2 + 2 * ge + rng.normal(size=m) * (0.5 + np.abs(z2))  # 异方差，x 的真实系数为 0
        return pd.DataFrame({'y': y, 'x': x, 'z1': z1, 'z2': z2, 'g': g})

    d = sim()
    rows, s = R.fit(d, 'y', ['x'], ['z1', 'z2'], ['g'], 'g')
    t0, _ = perm_core(s.y.to_numpy(), s.x.to_numpy(), design(s, ['z1', 'z2'], ['g']), s.g.to_numpy(), s.g.to_numpy(), 9, rng)
    ok1 = abs(t0 - rows[0]['t']) < 1e-8
    lines.append(f'  [{"通过" if ok1 else "失败"}] FWL 聚类 t vs 完整回归聚类 t：{t0:.10f} vs {rows[0]["t"]:.10f}')
    ps = []
    for _ in range(n_sim):
        d = sim()
        Z = np.column_stack([np.ones(len(d)), d[['z1', 'z2']].to_numpy(), R.fe_dummies(d, ['g'])])
        ps.append(perm_core(d.y.to_numpy(), d.x.to_numpy(), Z, d.g.to_numpy(), d.g.to_numpy(), B, rng)[1])
    ps = np.array(ps)
    rej10, rej5 = (ps < 0.10).mean(), (ps < 0.05).mean()
    ok2 = 0.05 <= rej10 <= 0.16
    lines.append(f'  [{"通过" if ok2 else "失败"}] 原假设下置换检验：{n_sim} 次模拟、B = {B}，名义 10% 拒绝率 {rej10:.3f}'
                 f'（要求 0.05–0.16），名义 5% 拒绝率 {rej5:.3f}，p_perm 均值 {ps.mean():.3f}')
    return ok1 and ok2, lines


# ---------------------------------------------------------------- 秩相关和三等分
def _rcorr(a, b, df):
    r = float(np.corrcoef(a, b)[0, 1])
    t = r * np.sqrt(df / (1 - r * r)) if abs(r) < 1 else np.nan
    return r, R.t_pvalue(t, df)


def _welch(a, b):
    """均值差 a − b 的 Welch t 和 p。"""
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    t = (a.mean() - b.mean()) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return a.mean() - b.mean(), t, R.t_pvalue(t, df)


def _tertile(v):
    return pd.qcut(pd.Series(v).rank(method='first'), 3, labels=False).to_numpy()


def rank_group(d, ys, y24_of, ctrl, fe, cl, xs):
    """每个 Y × X（M1 main 单独回归的样本）：秩相关表和三等分表。"""
    ric, grp = [], []
    for y in ys:
        c = list(ctrl) + ([y24_of[y]] if y24_of.get(y) else [])
        for x in xs:
            rows, s = R.fit(d, y, [x], c, [fe], cl)
            if not rows:
                continue
            n = len(s)
            Z = design(s, c, [fe])
            Q = _basis(Z)
            resid = lambda v: v - Q @ (Q.T @ v)  # noqa: E731
            yv, xv = s[y].to_numpy(float), s[x].to_numpy(float)
            ry, rx = pd.Series(yv).rank().to_numpy(), pd.Series(xv).rank().to_numpy()
            rho, p_rho = _rcorr(ry, rx, n - 2)
            prho, p_prho = _rcorr(resid(ry), resid(rx), n - Q.shape[1] - 1)
            ric.append({'Y': y, 'X': x, 'n': n, 'spearman': rho, 'p_spearman': p_rho, 'partial_spearman': prho, 'p_partial': p_prho})
            rec = {'Y': y, 'X': x, 'n': n}
            for tag, yy, xx in (('', yv, xv), ('adj_', resid(yv), resid(xv))):
                g = _tertile(xx)
                for k, name in enumerate(('low', 'mid', 'high')):
                    rec[f'{tag}{name}'] = yy[g == k].mean()
                rec[f'{tag}hml'], rec[f'{tag}t'], rec[f'{tag}p'] = _welch(yy[g == 2], yy[g == 0])
            grp.append(rec)
    return pd.DataFrame(ric), pd.DataFrame(grp)


# ---------------------------------------------------------------- 逐一剔除
def _drop_fit(yv, X, xcol, fe_codes, cl_codes, keep):
    """在 keep 的观测上重估（剔除后出现的 singleton 一并剔除），返回 X 在剩余样本里重新标准化后的系数、聚类 t。"""
    keep = keep.copy()
    while True:
        cnt = np.bincount(fe_codes[keep], minlength=fe_codes.max() + 1)
        bad = keep & (cnt[fe_codes] == 1)
        if not bad.any():
            break
        keep &= ~bad
    r = R.ols(yv[keep], X[keep], cl_codes[keep])
    return r['b'][1] * X[keep, xcol].std(ddof=1), r['b'][1] / r['se_cl'][1]


def influence(d, ys, y24_of, ctrl, fe, cl, xs, big='制造业'):
    """M1 main 单独回归的逐一剔除：公司、证监会门类、剔除制造业。返回 (结果表, 核对信息：numpy 剔除法 vs R.fit 的最大差)。"""
    out, maxdiff = [], 0.0
    for y in ys:
        c = list(ctrl) + ([y24_of[y]] if y24_of.get(y) else [])
        for x in xs:
            rows, s = R.fit(d, y, [x], c, [fe], cl)
            if not rows:
                continue
            b0, t0 = rows[0]['coef'], rows[0]['t']
            yv = s[y].to_numpy(float)
            X = np.column_stack([np.ones(len(s)), s[[x] + c].to_numpy(float), R.fe_dummies(s, [fe])])
            fe_codes = pd.factorize(s[fe])[0]
            cl_codes = s[cl].to_numpy()
            firm = []
            for i in range(len(s)):
                keep = np.ones(len(s), dtype=bool)
                keep[i] = False
                firm.append(_drop_fit(yv, X, 1, fe_codes, cl_codes, keep))
            firm = np.array(firm)
            if not out:  # 第一组：抽 3 家公司用 R.fit 重估核对
                for i in (0, len(s) // 2, len(s) - 1):
                    rr, _ = R.fit(d.drop(index=s.index[i]), y, [x], c, [fe], cl)
                    maxdiff = max(maxdiff, abs(rr[0]['coef'] - firm[i, 0]), abs(rr[0]['t'] - firm[i, 1]))
            top = int(np.argmax(np.abs(firm[:, 0] - b0)))
            ind = []
            for g in sorted(s[fe].unique()):
                rr, _ = R.fit(d[d[fe] != g], y, [x], c, [fe], cl)
                if rr:
                    ind.append((g, rr[0]['coef'], rr[0]['t'], rr[0]['n']))
            ind = pd.DataFrame(ind, columns=['g', 'coef', 't', 'n'])
            ti = ind.loc[(ind.coef - b0).abs().idxmax()]
            mfg = ind[ind.g == big]
            out.append({
                'Y': y, 'X': x, 'n': len(s), 'coef': b0, 't': t0,
                'firm_coef_min': firm[:, 0].min(), 'firm_coef_max': firm[:, 0].max(),
                'firm_same_sign': float((np.sign(firm[:, 0]) == np.sign(b0)).mean()),
                'firm_top_code': d.loc[s.index[top], 'code'], 'firm_top_name': d.loc[s.index[top], 'company'],
                'firm_top_coef': firm[top, 0], 'firm_top_t': firm[top, 1],
                'ind_coef_min': ind.coef.min(), 'ind_coef_max': ind.coef.max(), 'ind_t_min': ind.t.min(), 'ind_t_max': ind.t.max(),
                'ind_top': ti.g, 'ind_top_coef': ti.coef, 'ind_top_t': ti.t,
                'nomfg_coef': mfg.coef.iloc[0] if len(mfg) else np.nan, 'nomfg_t': mfg.t.iloc[0] if len(mfg) else np.nan,
                'nomfg_n': int(mfg.n.iloc[0]) if len(mfg) else np.nan})
    return pd.DataFrame(out), maxdiff


# ---------------------------------------------------------------- 研报层面 DAG 指标的固定效应 R² 分解
def fe_r2_decomposition(rw, ys, fes):
    """研报层面的 DAG 指标对几组固定效应（虚拟变量，不剔除 singleton，所有子模型用同一样本）回归，全部子集各估一次，
    对 R²(S) 和调整后 R²(S) = 1 − (1 − R²(S))(n − 1)/(n − 秩(S)) 分别做 Shapley 分解（空集都为 0）：
    φ_k = Σ_{S∌k} |S|!(m−|S|−1)!/m! · [v(S∪{k}) − v(S)]，Shapley 行的 target 列标明分解的是 r2 还是 adj_r2。
    固定效应之间可以嵌套（如公司嵌套在行业里），设计矩阵秩亏：这时用 SVD 的 lstsq 求残差，秩取设计矩阵的实际秩。
    返回 (长表：每个 Y 的每个子集一行 + 每个分解对象每组一行 Shapley 贡献,
          核对信息 {Y: (R² 贡献和 − 总 R², 总 R² − 组内去均值法的 R², 调整后 R² 贡献和 − 总调整后 R²)})。"""
    from itertools import combinations
    d = rw.dropna(subset=list(ys) + list(fes))
    m, n = len(fes), len(d)
    subsets = [S for k in range(m + 1) for S in combinations(fes, k)]
    weight = {S: math.factorial(len(S)) * math.factorial(m - len(S) - 1) / math.factorial(m) for S in subsets if len(S) < m}
    rows, check = [], {}
    for y in ys:
        yv = d[y].to_numpy(float)
        tss = (yv - yv.mean()) @ (yv - yv.mean())
        val = {'r2': {}, 'adj_r2': {}}
        for S in subsets:
            X = np.column_stack([np.ones(n), R.fe_dummies(d, list(S))])
            r = R.ols(yv, X)
            rank = r['k']
            if rank < X.shape[1]:  # 秩亏（嵌套的固定效应）：不用 X'X 的伪逆，改用 SVD 的 lstsq；满秩时沿用 R.ols，三组分解的数值和原来逐位相同
                e = yv - X @ np.linalg.lstsq(X, yv, rcond=None)[0]
                r['r2'] = 1 - (e @ e) / tss
            r2 = r['r2'] if S else 0.0
            val['r2'][S], val['adj_r2'][S] = r2, 1 - (1 - r2) * (n - 1) / (n - rank)
            rows.append({'kind': 'subset', 'Y': y, 'subset': '+'.join(S) or '（空）', **{f: f in S for f in fes}, 'r2': r2,
                         'adj_r2': val['adj_r2'][S], 'rank': int(rank), 'n': n})
        gap = {}
        for target, v in val.items():
            total = v[tuple(fes)]
            phis = {f: sum(w * (v[tuple(g for g in fes if g in S or g == f)] - v[S]) for S, w in weight.items() if f not in S) for f in fes}
            for f in fes:
                rows.append({'kind': 'shapley', 'target': target, 'Y': y, 'fe': f, 'n_groups': int(d[f].nunique()),
                             'contribution': phis[f], 'share': phis[f] / total, 'n': n})
            gap[target] = sum(phis.values()) - total
        z = R.within(d, [y], list(fes), iters=5000, tol=1e-14)[:, 0]
        check[y] = (gap['r2'], val['r2'][tuple(fes)] - (1 - (z @ z) / tss), gap['adj_r2'])
    return pd.DataFrame(rows), check


def nested_gap(dec, fes, inner, outer, col='r2'):
    """嵌套核对：凡是含 outer 的子集，再加 inner 后 col（r2 或 adj_r2）的增量（应为 0）的最大绝对值。dec 是 fe_r2_decomposition 的长表。"""
    s = dec[dec.kind == 'subset']
    key = lambda r, extra: (r.Y,) + tuple(bool(r[f]) or f == extra for f in fes)  # noqa: E731
    v = {key(r, None): r[col] for _, r in s.iterrows()}
    return max(abs(v[key(r, inner)] - r[col]) for _, r in s.iterrows() if r[outer] and not r[inner])
