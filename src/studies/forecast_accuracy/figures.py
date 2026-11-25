"""文档的两张图（写到 docs/figures/，文件名固定）。

图 1  logic_structure_schematic.png：逻辑结构变量的构建。用一个虚构的小图说明源点数、路径数、平均路径长度怎么数，并对比全文
      逻辑结构与投资评级概念对应的逻辑结构。数值用 src.measure.variables.graph_measures 计算（target=None 为全文，
      target=RATING_NODE 为投资评级子图），并与手算结果 assert 相等；各条路径及其边数由 networkx 穷举简单路径得到，再与之核对。
      图上：源点排在最左一列，列标题写源点数；每条源点到汇点的路径一种颜色、编号 ①–⑤，共用的边画成平行的彩色线；每条路径在源点
      左侧有同色标签写边数；面板内三行汇总；(b) 中不属于投资评级逻辑结构的概念、路径和标签画成浅灰虚线。
图 2  peer_lag_distribution.png：研报发布前 90 天内同行预测的时滞分布。样本 = 基准回归（表 5，主变量 sample.Y_MAIN）的估计样本，同行 =
      accuracy 步骤 report_peers 里这些研报在主窗口（h = sample.MAIN_H，目标年度 TY_Hh 逐篇不同）、[d − 90, d] 比较组的同行
      （自算的主窗口 PMAFE 和各回归控制的 PeerLag 用的就是它们）；
      (a) 全部同行预测的时滞（研报发布日 − 同行预测发布日），(b) 每篇研报同行时滞的均值（PeerLag 缩尾前的值），虚线为中位数。
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ROOT_DIR

FIG_DIR = ROOT_DIR / 'docs' / 'figures'
FIG1 = FIG_DIR / 'logic_structure_schematic.png'
FIG2 = FIG_DIR / 'peer_lag_distribution.png'

# ==================== 图 1：示意图 ====================
CHAIN_Y = 2.5
NODES = {  # 概念 -> (x, y)；源点一列在 x = 0
    '品牌力': (0, 3.85), '行业需求': (0, 3.0), '新品上市': (0, 2.0), '原材料价格': (0, 0.8),
    '销量': (1.6, CHAIN_Y), '营业收入': (3.0, CHAIN_Y), '盈利预测': (4.4, CHAIN_Y), '投资评级': (5.8, CHAIN_Y),
    '毛利率': (3.0, 1.2),
}
EDGES = [('行业需求', '销量'), ('新品上市', '销量'), ('销量', '营业收入'), ('营业收入', '盈利预测'),
         ('盈利预测', '投资评级'), ('品牌力', '投资评级'), ('新品上市', '毛利率'), ('原材料价格', '毛利率')]
ROUTE = {('品牌力', '投资评级'): [(5.8, 3.85)]}  # 折线拐点：品牌力向右走到投资评级正上方再向下，不压主链
# 路径编号顺序（源点, 汇点）与颜色（Okabe-Ito；共用主链的 ① ② 用灰度差别最大的蓝、橙）
PATH_ORDER = [('行业需求', '投资评级'), ('新品上市', '投资评级'), ('品牌力', '投资评级'),
              ('新品上市', '毛利率'), ('原材料价格', '毛利率')]
PATH_COLORS = ['#0072B2', '#E69F00', '#009E73', '#D55E00', '#CC79A7']
CIRCLED = '①②③④⑤⑥⑦⑧⑨'

INK, INK2, GREY, GREY_TXT = '#111111', '#444444', '#bdbdbd', '#a3a3a3'
DASH = (0, (4, 2.5))
STYLE = {  # 节点类别 -> 填充、边框色、边框宽、线型、文字色
    'src': dict(fc='#fff2cc', ec='#333333', lw=1.4, ls='-', tc=INK),
    'sink': dict(fc='white', ec='#111111', lw=2.8, ls='-', tc=INK),
    'mid': dict(fc='white', ec='#666666', lw=1.0, ls='-', tc=INK),
    'out': dict(fc='white', ec=GREY, lw=1.1, ls=DASH, tc=GREY_TXT),
}
LEGEND = [('src', '源点'), ('sink', '汇点'), ('mid', '其他概念'), ('out', '不属于该逻辑结构的概念')]
FS_NODE, FS_LABEL, FS_NOTE, FS_TITLE = 11, 10, 10.5, 12
BOX_H = 0.42
LW_PATH, OFF_PT = 1.9, 4.6        # 路径线宽；平行线之间半间距（磅）
MS, HEAD_LEN = 11, 0.6             # 箭头 mutation_scale 与 head_length（箭头长 = HEAD_LEN * MS 磅）
HEAD = f'-|>,head_length={HEAD_LEN},head_width=0.3'
XLIM, YLIM = (-1.5, 7.35), (0.45, 4.6)
FIG_W, PANEL_H, LEG_H = 8.2, 3.15, 0.45
SX = FIG_W / (XLIM[1] - XLIM[0])   # 每数据单位多少英寸（坐标轴占满图宽）
SY = PANEL_H / (YLIM[1] - YLIM[0])


def box_w(name):
    return 0.155 * len(name) + 0.22


def example_graph():
    import networkx as nx
    g = nx.DiGraph()
    g.add_nodes_from(NODES)
    g.add_edges_from(EDGES)
    assert nx.is_directed_acyclic_graph(g)
    return g


def structure(g, target):
    """target=None：全文；否则 target 及其全部上游构成的子图。
    返回 dict：sub 子图、srcs 源点、sinks 汇点、paths 按 PATH_ORDER 编号的全部路径（全文口径）、
    counted 每条路径是否属于本结构、lens 本结构各路径边数、m 指标（src.measure.variables.graph_measures）。"""
    import networkx as nx

    from src.measure.paths import strict_sinks
    from src.measure.variables import graph_measures
    m = graph_measures(g, target)
    if target is None:
        sub, sinks = g, strict_sinks(g)
    else:
        assert g.in_degree(target) > 0 and g.out_degree(target) == 0  # 与 measure 步骤一致：目标须为严格汇点
        sub, sinks = g.subgraph(nx.ancestors(g, target) | {target}).copy(), [target]
    srcs = [v for v in sub if sub.in_degree(v) == 0 and sub.out_degree(v) > 0]
    paths = []
    for s, t in PATH_ORDER:
        ps = list(nx.all_simple_paths(g, s, t))
        assert len(ps) == 1
        paths.append(ps[0])
    all_paths = {tuple(p) for s in [v for v in g if g.in_degree(v) == 0] for t in strict_sinks(g)
                 for p in nx.all_simple_paths(g, s, t)}
    assert all_paths == {tuple(p) for p in paths}  # 编号覆盖全文的全部路径
    own = {tuple(p) for s in srcs for t in sinks for p in nx.all_simple_paths(sub, s, t)}
    counted = [tuple(p) in own for p in paths]
    assert sum(counted) == len(own)
    lens = [len(p) - 1 for p, c in zip(paths, counted) if c]
    assert m['NSRC'] == len(srcs) and m['N_PATHS'] == len(lens) and math.isclose(m['PATHLEN'], sum(lens) / len(lens))
    return dict(sub=sub, srcs=srcs, sinks=sinks, paths=paths, counted=counted, lens=lens, m=m)


def schematic_structures():
    """示意图两个面板的 structure，并与手算结果核对。返回 (图, 全文, 投资评级)。"""
    from src.measure.variables import RATING_NODE
    g = example_graph()
    full, rate = structure(g, None), structure(g, RATING_NODE)
    assert sorted(full['srcs']) == sorted(['行业需求', '新品上市', '品牌力', '原材料价格'])
    assert sorted(full['sinks']) == sorted(['投资评级', '毛利率'])
    assert full['m']['NSRC'] == 4 and full['m']['N_PATHS'] == 5
    assert full['lens'] == [4, 4, 1, 1, 1] and math.isclose(full['m']['PATHLEN'], 2.2)
    assert set(rate['sub']) == {'投资评级', '盈利预测', '营业收入', '销量', '行业需求', '新品上市', '品牌力'}
    assert rate['m']['NSRC'] == 3 and rate['m']['N_PATHS'] == 3
    assert rate['lens'] == [4, 4, 1] and math.isclose(rate['m']['PATHLEN'], 3.0)
    return g, full, rate


def schematic_numbers():
    """图 1 例子的指标（正文 1.2 节引用）：{'全文' / '投资评级': graph_measures 的结果 + N_SINKS（汇点个数）}。"""
    _, full, rate = schematic_structures()
    return {name: {**st['m'], 'N_SINKS': len(st['sinks'])} for name, st in (('全文', full), ('投资评级', rate))}


def offset_xy(p, q, k):
    """p→q 线段向左侧（前进方向的左边）平移 k 个半间距后的端点（数据坐标）。"""
    dx, dy = (q[0] - p[0]) * SX, (q[1] - p[1]) * SY
    L = math.hypot(dx, dy)
    ox, oy = -dy / L * k * OFF_PT / 72 / SX, dx / L * k * OFF_PT / 72 / SY
    return (p[0] + ox, p[1] + oy), (q[0] + ox, q[1] + oy)


def draw_panel(ax, g, st, title, pre):
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    sub, srcs, sinks, paths, counted, m = st['sub'], st['srcs'], st['sinks'], st['paths'], st['counted'], st['m']
    patches = {}
    for n, (x, y) in NODES.items():
        kind = 'out' if n not in sub else 'src' if n in srcs else 'sink' if n in sinks else 'mid'
        s = STYLE[kind]
        w = box_w(n)
        p = FancyBboxPatch((x - w / 2, y - BOX_H / 2), w, BOX_H, boxstyle='round,pad=0,rounding_size=0.07',
                           mutation_aspect=SX / SY, fc=s['fc'], ec=s['ec'], lw=s['lw'], ls=s['ls'], zorder=3)
        ax.add_patch(p)
        patches[n] = p
        ax.text(x, y, n, ha='center', va='center', fontsize=FS_NODE, color=s['tc'], zorder=4)

    # 每条边上经过的路径（编号），共用的边画成平行线；同一边上编号小的在前进方向左侧
    on_edge = {e: [i for i, p in enumerate(paths) if e in zip(p, p[1:])] for e in g.edges}
    for (u, v), idx in on_edge.items():
        pts = [NODES[u]] + ROUTE.get((u, v), []) + [NODES[v]]
        for j, i in enumerate(idx):
            k = (len(idx) - 1) / 2 - j
            c, cnt = PATH_COLORS[i], counted[i]
            col, ls = (c, '-') if cnt else (GREY, DASH)
            kw = dict(lw=LW_PATH if cnt else 1.3, color=col, zorder=2)
            if len(pts) > 2:  # 折线：先画到最后一个拐点的线段（端点从源节点框外开始），再画带箭头的最后一段
                x0, y0 = pts[0]
                start = (x0 + box_w(u) / 2 + 2 / 72 / SX, y0)
                xs, ys = zip(start, *pts[1:-1])
                ax.add_line(Line2D(xs, ys, ls=ls, solid_capstyle='projecting', **kw))
                a, b, pa = pts[-2], pts[-1], None
            else:
                (a, b), pa = offset_xy(pts[0], pts[1], k), patches[u]
            akw = dict(patchA=pa, patchB=patches[v], mutation_scale=MS, shrinkA=2, shrinkB=2)
            if cnt:
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle=HEAD, **akw, **kw))
            else:  # 虚线箭杆 + 实心箭头（箭头若也用虚线描边会发虚）
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-', ls=DASH, **{**akw, 'shrinkB': 2 + HEAD_LEN * MS}, **kw))
                ax.add_patch(FancyArrowPatch(a, b, arrowstyle=HEAD, **akw, **{**kw, 'lw': 0.01}))

    # 每条路径在源点左侧的标签：编号 + 边数；同一源点有多条路径时上下排开
    by_src = {}
    for i, p in enumerate(paths):
        by_src.setdefault(p[0], []).append(i)
    for s, idx in by_src.items():
        x, y = NODES[s]
        for j, i in enumerate(idx):
            dy = ((len(idx) - 1) / 2 - j) * 0.36
            cnt = counted[i]
            ax.text(x - box_w(s) / 2 - 0.13, y + dy, f'{CIRCLED[i]} {len(paths[i]) - 1} 条边', ha='right', va='center',
                    fontsize=FS_LABEL, color=INK if cnt else GREY_TXT, zorder=4,
                    bbox=dict(boxstyle='round,pad=0.22', fc='white', ec=PATH_COLORS[i] if cnt else GREY,
                              lw=1.6 if cnt else 1.0, ls='-' if cnt else DASH))

    # 列标题
    top = YLIM[1] - 0.22
    ax.text(0, top, f'源点数 = {m["NSRC"]}', ha='center', va='center', fontsize=FS_NOTE, color=INK, fontweight='bold')
    ax.text(-1.05, top, f'路径数 = {m["N_PATHS"]}', ha='center', va='center', fontsize=FS_NOTE, color=INK, fontweight='bold')

    # 三行汇总（主链下方、毛利率右侧的空白处）
    n = m['N_PATHS']
    lines = [rf'源点数 $\mathit{{{pre}Src}}$ = {m["NSRC"]}',
             rf'路径数 {n} 条，$\mathit{{{pre}Path}}$ = ln(1 + {n})',
             rf'平均路径长度 $\mathit{{{pre}Len}}$ = ({" + ".join(map(str, st["lens"]))}) / {n} = {m["PATHLEN"]:.1f}']
    for i, t in enumerate(lines):
        ax.text(3.62, 1.62 - 0.36 * i, t, ha='left', va='center', fontsize=FS_NOTE, color=INK)

    ax.set_title(title, color=INK, fontsize=FS_TITLE, loc='left', pad=2)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis('off')


def schematic(out=FIG1):
    """画图 1。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    g, full, rate = schematic_structures()
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    T = 0.35  # 每个面板标题占的高度（英寸）
    H = LEG_H + 2 * (PANEL_H + T)
    fig = plt.figure(figsize=(FIG_W, H), facecolor='white')
    ax_a = fig.add_axes((0, (LEG_H + PANEL_H + T) / H, 1, PANEL_H / H))
    ax_b = fig.add_axes((0, LEG_H / H, 1, PANEL_H / H))
    draw_panel(ax_a, g, full, '(a) 全文逻辑结构', 'All')
    draw_panel(ax_b, g, rate, '(b) 投资评级概念对应的逻辑结构', 'Rate')
    handles = [Patch(fc=STYLE[k]['fc'], ec=STYLE[k]['ec'], lw=STYLE[k]['lw'], ls=STYLE[k]['ls'], label=lab)
               for k, lab in LEGEND]
    fig.legend(handles=handles, loc='center', ncol=4, frameon=False, fontsize=FS_NOTE,
               bbox_to_anchor=(0.5, LEG_H / 2 / H), handlelength=2.2, handleheight=1.1, columnspacing=2.2)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor='white', bbox_inches='tight', pad_inches=0.22)  # 四周留白，汇总文字不贴右边缘
    plt.close(fig)


# ==================== 图 2：同行时滞分布 ====================
def peer_lag_frames(est):
    """est：基准回归的估计样本（研报层面样本的行）。返回 (同行明细, 每篇研报)：
    同行明细（主窗口 MAIN_H 的比较组）：report_id、code、report_dt、report_ORG_CL、ORG_CL（同行券商）、peer_RPT_ID、peer_dt、LAG、F；
    每篇研报：report_id、code、ORG_CL、report_dt、month、n_peers、LAG_MEAN（= PeerLag 缩尾前）、LAG_MEDIAN、LAG_MIN、LAG_MAX、
    SHARE_LE7、SHARE_LE30、RT_NSRC、WG_NSRC（缩尾前），按 report_dt、report_id 排序。"""
    from src.accuracy.load import report_peers
    from src.studies.forecast_accuracy.sample import MAIN_H
    rep = est[['report_id', 'code', 'ORG_CL', 'date', 'RT_NSRC_RAW', 'WG_NSRC_RAW']]
    pr = report_peers()
    pr = pr[pr.h == MAIN_H].drop(columns=['h', 'ty'])
    pl = rep.merge(pr, on='report_id', how='inner', validate='1:m')
    pl = pl.rename(columns={'date': 'report_dt', 'ORG_CL': 'report_ORG_CL', 'peer_ORG_CL': 'ORG_CL'})
    pl = pl[['report_id', 'code', 'report_dt', 'report_ORG_CL', 'ORG_CL', 'peer_RPT_ID', 'peer_dt', 'LAG', 'F']].reset_index(drop=True)
    g = pl.groupby('report_id').LAG
    rl = rep.rename(columns={'date': 'report_dt', 'RT_NSRC_RAW': 'RT_NSRC', 'WG_NSRC_RAW': 'WG_NSRC'})
    rl = rl.assign(month=rl.report_dt.str[:4] + '-' + rl.report_dt.str[4:6])
    agg = pd.DataFrame({'n_peers': g.size(), 'LAG_MEAN': g.mean(), 'LAG_MEDIAN': g.median(), 'LAG_MIN': g.min(), 'LAG_MAX': g.max(),
                        'SHARE_LE7': g.apply(lambda z: (z <= 7).mean()), 'SHARE_LE30': g.apply(lambda z: (z <= 30).mean())})
    rl = rl.merge(agg.reset_index(), on='report_id', how='left', validate='1:1')
    rl = rl[['report_id', 'code', 'ORG_CL', 'report_dt', 'month', 'n_peers', 'LAG_MEAN', 'LAG_MEDIAN', 'LAG_MIN', 'LAG_MAX',
             'SHARE_LE7', 'SHARE_LE30', 'RT_NSRC', 'WG_NSRC']]
    return pl, rl.sort_values(['report_dt', 'report_id']).reset_index(drop=True)


def peer_lag_plot(pl, rl, out=FIG2):
    """画图 2。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    surf, ink, ink2, grid, blue = '#fcfcfb', '#0b0b0b', '#52514e', '#e1e0d9', '#2a78d6'
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor=surf)
    bins = np.arange(0, 95, 5)
    titles = (f'(a) 全部同行预测（{len(pl)} 条）', f'(b) 每篇研报的同行平均时滞（{len(rl)} 篇）')
    suptitle = '研报发布前 90 天内同行预测的时滞分布'
    for ax, z, title, ylab in ((axes[0], pl.LAG, titles[0], '同行预测条数'),
                               (axes[1], rl.LAG_MEAN, titles[1], '研报篇数')):
        ax.set_facecolor(surf)
        ax.hist(z, bins=bins, color=blue, edgecolor=surf, linewidth=1.5, zorder=2)
        med = float(np.median(z))
        ax.axvline(med, color=ink, linestyle='--', linewidth=1.2, zorder=3)
        ax.text(med + 1.5, ax.get_ylim()[1] * 0.95, f'中位数 {med:.0f} 天' if med == int(med) else f'中位数 {med:.1f} 天',
                color=ink, fontsize=10, va='top', ha='left')
        ax.set_xlim(0, 90)
        ax.set_xticks(range(0, 91, 15))
        ax.set_xlabel('研报发布日 − 同行预测发布日（天）', color=ink2)
        ax.set_ylabel(ylab, color=ink2)
        ax.set_title(title, color=ink, fontsize=12, loc='left')
        ax.grid(axis='y', color=grid, linewidth=0.6, zorder=0)
        for s_ in ('top', 'right'):
            ax.spines[s_].set_visible(False)
        for s_ in ('left', 'bottom'):
            ax.spines[s_].set_color(ink2)
        ax.tick_params(colors=ink2)
    fig.suptitle(suptitle, color=ink, fontsize=13, x=0.01, ha='left')
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=200, facecolor=surf)
    plt.close(fig)
