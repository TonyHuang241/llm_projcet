"""structure_distribution 的统计和画图：每列的分布统计、整数变量的频数表、按正文字数分组、两张直方图（口径见 __init__.py）。
统计哪几列由目标决定（columns(target)：目标的三个变量 + WG 三个，rating 是 RT_*、revenue 是 RV_*）；下面以 c 为参数的函数
默认用 rating 的列（模块常量 X6 等）。"""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.measure.variables import DEFAULT_TARGET, TARGETS


def columns(target=DEFAULT_TARGET):
    """目标 target 要统计的列：X6 回归用的六个变量、NPATHS 路径条数、LOG1P_X / LOG1P 对数变换前后的四列（同
    forecast_accuracy/sample.py 的 XVars.LOG1P_X，--log1p 换成 ln(1 + x)）、INTS 出频数表的整数变量。"""
    p = TARGETS[target]['prefix']
    log1p_x = [f'{p}_NSRC', f'{p}_PATHLEN', 'WG_NSRC', 'WG_PATHLEN']
    return SimpleNamespace(target=target, X6=[f'{p}_NSRC', f'{p}_LNPATH', f'{p}_PATHLEN', 'WG_NSRC', 'WG_LNPATH', 'WG_PATHLEN'],
                           NPATHS=[f'{p}_N_PATHS', 'WG_N_PATHS'], LOG1P_X=log1p_x, LOG1P=[f'{c}_LOG1P' for c in log1p_x],
                           INTS=[f'{p}_NSRC', 'WG_NSRC', f'{p}_N_PATHS', 'WG_N_PATHS'])


C = columns()  # 默认目标（rating）
X6, NPATHS, LOG1P_X, LOG1P, INTS = C.X6, C.NPATHS, C.LOG1P_X, C.LOG1P, C.INTS
NAMES = {'NSRC': '源点数', 'LNPATH': 'ln(1+路径数)', 'PATHLEN': '平均路径长度', 'N_PATHS': '路径条数'}
TAG = {**{t['prefix']: t['tag'] for t in TARGETS.values()}, 'WG': '全文'}  # 列名前缀 -> 标签里的两字称呼
DOC = {f'{pre}_{m}': f'{sym}{suf}' for pre, sym in [(t['prefix'], t['symbol']) for t in TARGETS.values()] + [('WG', 'All')]
       for m, suf in (('NSRC', 'Src'), ('LNPATH', 'Path'), ('PATHLEN', 'Len'))}
PCTS = [1, 5, 10, 25, 50, 75, 90, 95, 99]
FREQ_COVER, FREQ_MIN_CAP = 0.99, 21  # 频数表合并行的上限 k：前面各行累计占比 ≥ 99% 的最小整数，且 ≥ 21（0–20 逐个列出）
LENGTH_BINS = [0, 1000, 2000, 5000, 10000, np.inf]  # 同 concept_coverage
LENGTH_LABELS = ['<1000', '1000–2000', '2000–5000', '5000–10000', '≥10000']


def label(col):
    """列名 -> 中文标签，如 RT_NSRC -> “评级：源点数 RateSrc”，RV_NSRC -> “营收：源点数 RevSrc”，
    WG_PATHLEN_LOG1P -> “全文：ln(1+平均路径长度) AllLen”。"""
    pre, rest = col[:2], col[3:]
    log = rest.endswith('_LOG1P')
    base = rest[:-6] if log else rest
    name = f'ln(1+{NAMES[base]})' if log else NAMES[base]
    doc = DOC.get(f'{pre}_{base}', '')
    return f'{TAG[pre]}：{name}' + (f' {doc}' if doc else '')


def add_log1p(x, cols=C):
    """加上 <原名>_LOG1P 四列（ln(1 + 原始值)，与回归的 --log1p 相同）。"""
    for c, n in zip(cols.LOG1P_X, cols.LOG1P):
        x[n] = np.log1p(x[c].astype(float))
    return x


def moments(v):
    """(偏度, 超额峰度)：偏差校正的样本偏度 G1 和超额峰度 G2（与 pandas 的 skew()、kurt() 相同）。"""
    n = len(v)
    d = v - v.mean()
    m2, m3, m4 = (d ** 2).mean(), (d ** 3).mean(), (d ** 4).mean()
    if n < 4 or m2 == 0:
        return np.nan, np.nan
    g1, g2 = m3 / m2 ** 1.5, m4 / m2 ** 2 - 3
    return g1 * np.sqrt(n * (n - 1)) / (n - 2), ((n + 1) * g2 + 6) * (n - 1) / ((n - 2) * (n - 3))


def describe(s):
    """一列（缺失为 NaN）-> 一行统计的 dict。分位数用 numpy 的线性插值（同 pandas 的 quantile）。"""
    v = s.dropna().to_numpy(float)
    row = {'n': len(v), 'n_missing': int(s.isna().sum())}
    if not len(v):
        return row
    skew, kurt = moments(v)
    row.update(mean=v.mean(), sd=v.std(ddof=1) if len(v) > 1 else np.nan, min=v.min())
    row.update({f'p{p}': q for p, q in zip(PCTS, np.percentile(v, PCTS))})
    row.update(max=v.max(), skewness=skew, excess_kurtosis=kurt, share_min=(v == v.min()).mean(), n_unique=len(np.unique(v)))
    return row


def summary(x, cols=C):
    """统计表：六个变量、两个路径条数、四个对数变换列，一列一行。"""
    t = pd.DataFrame([{'variable': c, 'label': label(c), **describe(x[c].astype(float))} for c in cols.X6 + cols.NPATHS + cols.LOG1P])
    return t.astype({'n': int, 'n_missing': int, 'n_unique': 'Int64'})


def frequency(x, cols=C):
    """四个整数变量的取值频数表；≥ k 的取值合并成一行（k 见 FREQ_COVER、FREQ_MIN_CAP）。"""
    rows = []
    for c in cols.INTS:
        v = x[c].dropna().astype(int)
        n = len(v)
        vc = v.value_counts().sort_index()
        hit = vc.cumsum().to_numpy() * 100 >= FREQ_COVER * 100 * n  # 整数比较，避免浮点误差
        k = max(int(vc.index[hit.argmax()]) + 1, FREQ_MIN_CAP)
        part = [(str(val), int(cnt)) for val, cnt in vc[vc.index < k].items()]
        tail = int(vc[vc.index >= k].sum())
        if tail:
            part.append((f'≥{k}', tail))
        cum = 0
        for val, cnt in part:
            cum += cnt
            rows.append({'variable': c, 'value': val, 'n_reports': cnt, 'share': cnt / n, 'cum_share': cum / n})
    return pd.DataFrame(rows)


def by_length(x, cols=C):
    """按正文字数分组（左闭右开）：六个变量各自的 n、mean、p50；n_reports 是该组研报数（含该变量缺失的）。"""
    g = pd.cut(x.n_chars, LENGTH_BINS, right=False, labels=LENGTH_LABELS)
    rows = []
    for c in cols.X6:
        for grp in LENGTH_LABELS:
            s = x.loc[g == grp, c].dropna()
            rows.append({'variable': c, 'group': grp, 'n_reports': int((g == grp).sum()), 'n': len(s),
                         'mean': s.mean() if len(s) else np.nan, 'p50': s.median() if len(s) else np.nan})
    return pd.DataFrame(rows)


def histograms(x, path, log1p=False, cols=C):
    """2 行 3 列直方图：上排目标（RT 或 RV）、下排 WG 的三个变量；log1p=True 时源点数和平均路径长度换成 ln(1 + x)。
    原始源点数用整数柱、横轴截到 p99（截掉的篇数写在子图脚注）；图上的 n、中位数、偏度都按全部非缺失值算。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 中文字体，同 forecast_accuracy/figures.py
    plt.rcParams['axes.unicode_minus'] = False
    surf, ink, ink2, grid, blue = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#2a78d6'
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.6), facecolor=surf)
    for ax, c in zip(axes.ravel(), cols.X6):
        col = f'{c}_LOG1P' if log1p and c in cols.LOG1P_X else c
        v = x[col].dropna().to_numpy(float)
        med, skew = float(np.median(v)), moments(v)[0]
        note = ''
        if c.endswith('NSRC') and not log1p:
            p99 = np.percentile(v, 99)
            cap = int(np.ceil(p99))
            shown = v[v <= cap]
            ax.hist(shown, bins=np.arange(shown.min() - 0.5, cap + 1.5), color=blue, edgecolor=surf, linewidth=0.8, zorder=2)
            note = f'横轴截到 {cap}（p99 = {p99:.2f}），截掉 {len(v) - len(shown)} 篇'
        else:
            ax.hist(v, bins=40, color=blue, edgecolor=surf, linewidth=0.8, zorder=2)
        ax.axvline(med, color=ink, linestyle='--', linewidth=1.1, zorder=3)
        ax.text(0.97, 0.95, f'n = {len(v)}\n中位数 = {med:.2f}\n偏度 = {skew:.2f}', transform=ax.transAxes,
                ha='right', va='top', fontsize=9.5, color=ink, linespacing=1.4)
        if note:
            ax.text(0.97, 0.62, note, transform=ax.transAxes, ha='right', va='top', fontsize=8.5, color=ink2)
        ax.set_title(label(col), color=ink, fontsize=11, loc='left')
        ax.set_xlabel(col, color=ink2, fontsize=9)
        ax.set_facecolor(surf)
        ax.grid(axis='y', color=grid, linewidth=0.6, zorder=0)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            ax.spines[s_].set_color(ink2)
        ax.tick_params(colors=ink2, labelsize=9)
    axes[0, 0].set_ylabel('研报篇数', color=ink2)
    axes[1, 0].set_ylabel('研报篇数', color=ink2)
    title = '逻辑结构变量的分布' + ('（源点数、平均路径长度取 ln(1 + x)）' if log1p else '（原始值）')
    fig.suptitle(title + '；虚线为中位数', color=ink, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, facecolor=surf)
    plt.close(fig)
