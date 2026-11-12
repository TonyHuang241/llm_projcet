"""按年报内句子情感标准差分成低、高两组，画公告日前后累积异常回报（CAR）中位数及其 95% 置信区间。

分组：每个年报年份内按 column 指定的标准差（原文 sent_std_orig 或中性化 sent_std_neutral）的中位数分组，
      避开 2021 年口径断点造成的水平差异。
路径：异常回报与 event_study 相同（市场模型，估计窗口 [t0-250, t0-21]），在 [t0-PRE, t0+POST] 上逐日累加，
      再减去 t0-1 的累积值，所以公告前一交易日为 0，t0+k 的值等于 CAR[0,k]；只保留整段窗口都有收益率的事件。
置信区间：按事件日聚类的自助法，有放回地抽取事件日（同一天公告的事件一起抽），重复 BOOTSTRAP 次，
         取每个事件日中位数的 2.5% 和 97.5% 分位数。两组差值用同一批抽样计算。
输出（都在 out_dir 中）：
  - <stem>.csv：每组及两组之差（high_minus_low）每个事件日的样本数、中位数和置信区间
  - <stem>.png：两组 CAR 中位数之差（高 - 低）及其 95% 置信区间

用法：先运行 event_study，再在 main.py 中设置 TASK = 'plot_std_car'
"""
import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from . import event_study as es

PRE, POST = 20, 20
BOOTSTRAP, SEED = 1000, 0
GROUPS = {'high': ('高标准差组', '#eb6834'), 'low': ('低标准差组', '#2a78d6')}  # 分类色板第 2、1 槽
INK, MUTED, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'


def assign_groups(events, column):
    """每个年份内按 column 的中位数分组，返回与 events 等长的列表（'low'、'high' 或 None）。"""
    values = np.array([float(e[column]) if e[column] else np.nan for e in events])
    years = np.array([int(e['year']) for e in events])
    groups = [None] * len(events)
    for year in np.unique(years):
        idx = np.where((years == year) & ~np.isnan(values))[0]
        cut = np.median(values[idx])
        for i in idx:
            groups[i] = 'high' if values[i] > cut else 'low' if values[i] < cut else None
    return groups


def car_paths(events, groups, days, code_col, ret, market):
    """返回 ({分组: 路径矩阵[事件, 事件日]}, {分组: 事件日数组}, 因窗口内缺失收益率剔除的事件数)。"""
    paths, dates, dropped = {g: [] for g in GROUPS}, {g: [] for g in GROUPS}, 0
    for e, group in zip(events, groups):
        if group is None:
            continue
        t0 = int(np.searchsorted(days, int(e['event_date'])))
        r = ret[:, code_col[e['code']]]
        fit = es.market_model(r, market, t0)
        if t0 < PRE or t0 + POST >= len(days) or fit is None:
            dropped += 1
            continue
        window = r[t0 - PRE:t0 + POST + 1] - fit[0] - fit[1] * market[t0 - PRE:t0 + POST + 1]
        if np.isnan(window).any():
            dropped += 1
            continue
        cumulative = np.cumsum(window)
        paths[group].append(cumulative - cumulative[PRE - 1])  # 公告前一交易日归零
        dates[group].append(int(e['event_date']))
    return ({g: np.array(p) for g, p in paths.items()},
            {g: np.array(d) for g, d in dates.items()}, dropped)


def bootstrap_medians(paths, dates, n_boot=BOOTSTRAP, seed=SEED):
    """按事件日聚类抽样，返回 {分组: 中位数路径的自助样本[抽样次数, 事件日]}。"""
    rng = np.random.default_rng(seed)
    all_dates = np.unique(np.concatenate(list(dates.values())))
    members = {g: [np.where(dates[g] == d)[0] for d in all_dates] for g in GROUPS}
    samples = {g: np.empty((n_boot, paths[g].shape[1])) for g in GROUPS}
    for b in range(n_boot):
        draw = rng.integers(len(all_dates), size=len(all_dates))
        for g in GROUPS:
            idx = np.concatenate([members[g][d] for d in draw])
            samples[g][b] = np.median(paths[g][idx], axis=0)
    return samples


def summarize(paths, samples):
    """返回 {序列: (中位数, 下限, 上限)}，序列包括两组及其差值 high_minus_low。"""
    point = {g: np.median(p, axis=0) for g, p in paths.items()}
    point['high_minus_low'] = point['high'] - point['low']
    samples = {**samples, 'high_minus_low': samples['high'] - samples['low']}
    return {s: (point[s], np.quantile(samples[s], 0.025, axis=0), np.quantile(samples[s], 0.975, axis=0))
            for s in point}


def draw(stats, n_events, path, label):
    plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False, 'font.size': 11})
    taus = np.arange(-PRE, POST + 1)
    fig, ax = plt.subplots(figsize=(10.5, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.spines['bottom'].set_color(GRID)
    ax.grid(axis='y', color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.axvline(0, color=MUTED, linewidth=1)
    ax.axhline(0, color=MUTED, linewidth=1)
    ax.set_xlim(-PRE, POST)
    ax.set_xticks(range(-20, 21, 5))
    ax.set_xlabel('相对公告日的交易日', color=MUTED)
    ax.set_ylabel('CAR 中位数之差（高 − 低，百分点）', color=MUTED)
    median, low, high = (v * 100 for v in stats['high_minus_low'])
    ax.fill_between(taus, low, high, color='#2a78d6', alpha=0.14, linewidth=0)
    ax.plot(taus, median, color='#2a78d6', linewidth=2)
    ax.set_title(f'{label}高、低组 CAR 中位数之差及其 95% 置信区间', color=INK, loc='left', fontsize=14, pad=14)
    fig.text(0.01, 0.01, f'CAR 在公告前一交易日归零；每年按标准差中位数分组；事件数：高 {n_events["high"]:,}、低 {n_events["low"]:,}，'
             f'2011–2023 年报，未控制其他变量；阴影为按事件日聚类的自助法 95% 置信区间（{BOOTSTRAP} 次）', color=MUTED, fontsize=9)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main(event_panel_path, daily_path, out_dir, column, label, stem):
    """column：用于分组的标准差列；label：图标题中的变量名；stem：输出文件名（不含扩展名）。"""
    out_dir = Path(out_dir)
    with Path(event_panel_path).open(encoding='utf-8-sig', newline='') as f:
        events = list(csv.DictReader(f))
    days, code_col, ret, market = es.load_returns(daily_path)
    groups = assign_groups(events, column)
    paths, dates, dropped = car_paths(events, groups, days, code_col, ret, market)
    n_events = {g: len(p) for g, p in paths.items()}
    print(f'事件数：{n_events}；窗口内缺失收益率剔除 {dropped} 个；开始自助法（{BOOTSTRAP} 次）')
    stats = summarize(paths, bootstrap_medians(paths, dates))

    with (out_dir / f'{stem}.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['tau', 'series', 'n', 'median', 'ci_low', 'ci_high'])
        for series, values in stats.items():
            n = n_events.get(series, '')
            for i, tau in enumerate(range(-PRE, POST + 1)):
                writer.writerow([tau, series, n, *(round(float(v[i]), 6) for v in values)])
    draw(stats, n_events, out_dir / f'{stem}.png', label)
    print(f'结果保存在 {out_dir}')
    return paths, dates, stats, events, groups
