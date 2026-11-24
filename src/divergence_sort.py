"""分歧度与股票数据的分组统计（不调用大模型）。三件事，结果都写到 <输出目录>/data/divergence/：

1. 公司-季度分歧度（company_quarter.parquet）。同一家公司、同一个自然季度（按研报发布日期）里的研报放在一起：
   每篇研报先压成一张流量分布 {(起点主题, 终点): 占比}（定义见 src/viz/plot_intro.py 的 graph_payload，主题来自框架对比的
   themes.parquet），再和这些研报的平均分布比较。没有概念连到估值的研报不参与；不足 MIN_REPORTS 篇的公司-季度不算。
       total / theme / rel   整体口径的总分歧、主题项、关系项（total = theme + rel），已除以 log2(研报数) 归一到 0-1
       shared                不止一篇研报谈到的主题在平均分布里的分量
       pair_total / pair_theme / pair_rel   季度内研报两两比较后取平均（两篇研报时和整体口径相同）
2. 匹配股票数据（panel.parquet）。股票代码去掉交易所后缀后和行情、财务数据对上：
       mkcap        当季最后一个有市值的交易日的总市值（元）
       next_ret     下一个季度的股票回报：下季度最后一个交易日的后复权收盘价 / 当季最后一个交易日的后复权收盘价 - 1
       fin_*        当季末之前已经公告的最新一期财务数据（按公告日取，不用未来数据），fin_report_date 是它的报告期
3. 分组（sort_*.csv）。先按总分歧、主题项、关系项各自做一次单变量分组（sort_single.csv，每个指标一行；
   sort_single_detail.csv 是每组的明细），再做双重分组：每个季度内把公司按两个分歧指标各分成 N_GROUPS 组，统计每个格子的回报、
   当季末市值均值和样本数。回报按 Ang, Hodrick, Xing & Zhang (2006, The Cross-Section of Volatility and Expected
   Returns) 的组合做法：每个格子是一个组合，在季末建仓、持有下一个季度，下个季末按新的分组换仓；组合每天的回报是成分股
   当天回报的平均（默认等权，可以按建仓时的季末市值加权），表里是持有期组合日回报的平均（基点）。“高-低”是最高组和
   最低组两个组合每日回报之差的平均，t 值用 Newey-West (1987) 标准误（滞后阶数默认按 Newey-West 1994 的经验规则）。
4. 逻辑链大小和深度（report_sizes.parquet、company_quarter_size.parquet、size_sort_*.csv）。每篇研报的逻辑图用几种算法量
   大小和深度（graph_measures），按研报类型分开，在公司-季度内取平均，再每个季度内分 N 组看下季度回报。逻辑图默认从各篇研报的
   存档（cache/reports/）读，抽取还没跑完、data/graphs/ 还没更新时也能用上新做完的研报（archive_reports）。
   “链条”指从链条起点（没有别的概念指向它、能连到估值的概念）沿边走到公司估值的一条路径；只看连到估值的概念。
       大小  n_concepts    概念数：第 3-5 层的概念个数
             n_edges       边数：图里全部的边（含补全的边和预置的会计关系）
             n_text_edges  原文边数：有原文证据的边（extracted、bridged），不含补全和预置的
             chains        链条数：不同链条的条数，取 log10(1 + 条数)
             density       每千字概念数：概念数 / 正文字数 × 1000，扣掉篇幅的影响
       深度  depth         最长链步数：从概念走到公司估值最多要几步（几条边）
             depth_mean    平均链长：全部链条的平均步数
             depth_flow    流量平均步数：每个概念发出 1 份流量，每到一个节点平均分给它通往估值的下游，平均几步到公司估值
             mean_layer    平均层级：概念平均在第几层（3 经营指标、4 公司经营与行业、5 外部环境），越大越远离财务科目
   研报里没有概念连到估值时，深度的后三个算法为空。新数据里没有标题，研报类型按正文判断（report_kind）：
       深度报告   正文不少于 DEEP_CHARS 字，或者写明“首次覆盖”且不少于 DEEP_CHARS 的一半
       财报点评   不是深度报告，开头 400 字里在讲一期财报：“发布/披露……年报、季报、业绩快报”或者“营收/净利润……同比”
       其他       其余（事件点评、跟踪报告等）
   这一部分不要求公司-季度有两篇以上研报，一篇也算；没有概念连到估值的研报也算（图本身还在）。
5. 回归（panel_reg.parquet、reg_returns.csv、reg_financial.csv）。股票收益（stock_chars）和财务特征（fin_chars）两组因变量，
   分别对总分歧、主题项、关系项各回归一次，再把主题项和关系项放在一起回归；两个模型：只有季度固定效应，再加控制变量（CONTROLS）。
   分歧度标准化，因变量和控制变量 1%/99% 缩尾，标准误按公司聚类（div_regressions）。company_quarter_reports.parquet 记下
   参与分歧度的研报，算研报发布窗口的超额收益用。
"""
import itertools
import math
import re
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import OutDir
from src.fin_items import ROOT
from src.logic.base import TEXT_ORIGINS
from src.logic.output import read_graphs
from src.viz.plot_intro import graph_payload, kl, split

QUARTER_END = {1: '0331', 2: '0630', 3: '0930', 4: '1231'}


def log(msg):
    print(msg, flush=True)


def quarter_of(date):
    return f'{date[:4]}Q{(int(date[4:6]) - 1) // 3 + 1}'


def next_quarter(q):
    y, k = int(q[:4]), int(q[5])
    return f'{y + k // 4}Q{k % 4 + 1}'


def quarter_end(q):
    return q[:4] + QUARTER_END[int(q[5])]


# ==================== 1. 公司-季度分歧度 ====================

def report_flows(out_dir, recs=None):
    """每篇研报的流量分布。返回 ({report_id: 元数据}, {report_id: 分布})；没有概念连到估值的研报不在分布里。
    recs 是已经读好的逻辑图（read_graphs 的 reports），不给就现读。"""
    out = OutDir(out_dir)
    themes = defaultdict(dict)
    for t in pq.read_table(out.compare / 'themes.parquet').to_pylist():
        themes[t['stock_code']][t['concept']] = t['theme']
    info, flows = {}, {}
    for r in recs if recs is not None else read_graphs(out_dir)['reports']:
        info[r['report_id']] ={k: r[k] for k in ('stock_code', 'company', 'broker', 'date')}
        if r['stats']['reach'] <= 0:
            continue
        try:
            flows[r['report_id']] = graph_payload(r, themes[r['stock_code']])[1]
        except (ZeroDivisionError, KeyError):  # 图里没有流量，或者有概念没归到主题（框架对比之后又新抽了研报）
            pass
    return info, flows


def group_split(ds):
    """多篇研报一起比较：M 是各篇分布的平均，总分歧 = 各篇 KL(P_i‖M) 的平均，主题项只看起点主题的占比，关系项 = 总分歧 - 主题项。
    返回未归一的 (总分歧, 主题项, 关系项, 共同主题分量)。"""
    n = len(ds)
    M = defaultdict(float)
    for d in ds:
        for k, v in d.items():
            M[k] += v / n
    row = lambda X: {r: sum(v for (a, _), v in X.items() if a == r) for r in {a for a, _ in X}}
    m, rows = row(M), [row(d) for d in ds]
    total = sum(kl(d, M) for d in ds) / n
    theme = sum(kl(p, m) for p in rows) / n
    shared = sum(v for r, v in m.items() if sum(r in p for p in rows) >= 2)
    return total, theme, max(total - theme, 0.0), shared


def company_quarter(out_dir, min_reports=2, recs=None):
    """公司-季度分歧度表，写到 data/divergence/company_quarter.parquet。"""
    info, flows = report_flows(out_dir, recs)
    cells = defaultdict(list)
    for rid in flows:
        cells[(info[rid]['stock_code'], quarter_of(info[rid]['date']))].append(rid)
    rows = []
    for (code, q), rids in sorted(cells.items()):
        n = len(rids)
        if n < max(min_reports, 2):
            continue
        total, theme, rel, shared = group_split([flows[r] for r in rids])
        z = math.log2(n)
        pairs = [split(flows[a], flows[b]) for a, b in itertools.combinations(rids, 2)]
        mean = lambda k: sum(p[k] for p in pairs) / len(pairs)
        rows.append({'stock_code': code, 'company': info[rids[0]]['company'], 'quarter': q, 'n_reports': n,
                     'n_brokers': len({info[r]['broker'] for r in rids}), 'total': total / z, 'theme': theme / z, 'rel': rel / z,
                     'shared': shared, 'pair_total': mean('total'), 'pair_theme': mean('theme'), 'pair_rel': mean('rel')})
    cq = pd.DataFrame(rows)
    path = OutDir(out_dir).divergence / 'company_quarter.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    cq.to_parquet(path)
    keep = {(r.stock_code, r.quarter) for r in cq.itertuples()}
    pd.DataFrame([{'report_id': rid, 'stock_code': code, 'quarter': q, 'date': info[rid]['date']}
                  for (code, q), rids in sorted(cells.items()) if (code, q) in keep for rid in rids]
                 ).to_parquet(path.with_name('company_quarter_reports.parquet'))
    single = sum(len(v) == 1 for v in cells.values())
    log(f'公司-季度分歧度：研报 {len(info)} 篇，参与 {len(flows)} 篇；公司-季度 {len(cells)} 个，其中只有 1 篇的 {single} 个；'
        f'算出分歧度的 {len(cq)} 个（{cq.stock_code.nunique()} 家公司、{int(cq.n_reports.sum())} 篇研报）-> {path}')
    return cq


# ==================== 2. 匹配股票数据 ====================

def data_last_date(daily_path):
    """行情数据的最后一天（读 parquet 的统计信息，不读数据）。"""
    f = pq.ParquetFile(daily_path)
    i = f.schema_arrow.names.index('date')
    stats = [f.metadata.row_group(g).column(i).statistics for g in range(f.metadata.num_row_groups)]
    if all(s is not None and s.has_min_max for s in stats):
        return max(s.max for s in stats)
    return pq.read_table(daily_path, columns=['date']).column('date').to_pandas().max()


def market_quarters(daily_path, codes, quarters):
    """每只股票每个季度：季末后复权收盘价、季末总市值、下一个季度的回报。行情表的 close 是后复权价（除权除息日连续），
    停牌日也有行（收盘价沿用、市值为空），所以收盘价取季度内最后一行、市值取季度内最后一个非空值。
    下一个季度的行情还没走完（数据最后一天离季末超过 3 天）时，回报记为空。"""
    need = sorted(set(quarters) | {next_quarter(q) for q in quarters})
    d = pq.read_table(daily_path, columns=['date', 'code', 'close', 'mkcap'],
                      filters=[('date', '>=', need[0][:4] + '0101'), ('date', '<=', quarter_end(need[-1])), ('code', 'in', sorted(codes))]).to_pandas()
    d['quarter'] = d.date.str[:4] + 'Q' + ((d.date.str[4:6].astype(int) - 1) // 3 + 1).astype(str)
    m = d.sort_values(['code', 'date']).groupby(['code', 'quarter']).agg(close=('close', 'last'), mkcap=('mkcap', 'last'), last_trade=('date', 'last'))
    last = data_last_date(daily_path)
    done = {q: last >= quarter_end(q) or (pd.Timestamp(quarter_end(q)) - pd.Timestamp(last)).days <= 3 for q in need}
    nxt = m.close.reset_index()
    nxt = nxt[nxt.quarter.map(done)].rename(columns={'close': 'next_close', 'quarter': 'next_quarter'})
    m = m.reset_index()
    m['next_quarter'] = m.quarter.map(next_quarter)
    m = m.merge(nxt, on=['code', 'next_quarter'], how='left')
    m['next_ret'] = m.next_close / m.close - 1
    undone = [q for q in need if not done[q]]
    if undone:
        log(f'行情数据到 {last} 为止，{"、".join(undone)} 还没走完，以它们为下一个季度的回报记为空')
    return m[m.quarter.isin(quarters)][['code', 'quarter', 'mkcap', 'next_ret', 'last_trade']]


def quarter_start(q):
    return f'{q[:4]}{(int(q[5]) - 1) * 3 + 1:02d}01'


def daily_returns(daily_path, codes, quarters):
    """持有期（各形成季度的下一个季度）里每只股票每个交易日的回报：后复权收盘价 / 前一个交易日的后复权收盘价 - 1。
    停牌日行情表里也有行（收盘价沿用），回报为 0。返回 date、code、quarter（交易日所在的季度）、ret。"""
    hold = sorted({next_quarter(q) for q in quarters})
    d = pq.read_table(daily_path, columns=['date', 'code', 'close'],
                      filters=[('date', '>=', quarter_start(min(quarters))), ('date', '<=', quarter_end(hold[-1])),
                               ('code', 'in', sorted(codes))]).to_pandas().sort_values(['code', 'date'])
    d['ret'] = d.close / d.groupby('code').close.shift() - 1
    d['quarter'] = d.date.str[:4] + 'Q' + ((d.date.str[4:6].astype(int) - 1) // 3 + 1).astype(str)
    return d[d.quarter.isin(hold)][['date', 'code', 'quarter', 'ret']].reset_index(drop=True)


def fundamentals(fin_path, codes, quarters, cols):
    """每只股票每个季度末已经公告的最新一期财务数据（date 是公告日，report_date 是报告期）。"""
    fin = pd.read_parquet(fin_path, columns=['code', 'report_date', 'date'] + list(cols))
    fin = fin[fin.code.isin(codes)].sort_values(['code', 'report_date'])
    out = []
    for q in sorted(set(quarters)):
        known = fin[fin.date <= quarter_end(q)].groupby('code').tail(1)
        out.append(known.drop(columns='date').assign(quarter=q))
    out = pd.concat(out) if out else fin.drop(columns='date').assign(quarter='').head(0)
    return out.rename(columns={c: 'fin_' + c for c in ['report_date'] + list(cols)})


def build_panel(cq, daily_path, fin_path=None, fin_cols=()):
    """分歧度表接上行情（和财务）数据。股票代码去掉交易所后缀后匹配；行情里没有的（港股等）整行去掉。"""
    panel = cq.copy()
    panel['code'] = panel.stock_code.str.split('.').str[0]
    mkt = market_quarters(daily_path, set(panel.code), set(panel.quarter))
    panel = panel.merge(mkt, on=['code', 'quarter'], how='left')
    missing = panel[panel.last_trade.isna()]
    if len(missing):
        log(f'行情里没有的公司-季度 {len(missing)} 个（{missing.stock_code.nunique()} 家公司，如 {"、".join(missing.stock_code.unique()[:5])}），已去掉')
    panel = panel[panel.last_trade.notna()].reset_index(drop=True)
    if fin_path and Path(fin_path).exists() and fin_cols:
        panel = panel.merge(fundamentals(fin_path, set(panel.code), set(panel.quarter), fin_cols), on=['code', 'quarter'], how='left')
        log(f'财务数据：{panel.fin_report_date.notna().sum()}/{len(panel)} 个公司-季度匹配上')
    log(f'匹配行情后：公司-季度 {len(panel)} 个、公司 {panel.stock_code.nunique()} 家；有季末市值的 {panel.mkcap.notna().sum()} 个，'
        f'有下季度回报的 {panel.next_ret.notna().sum()} 个；各季度个数 {panel.quarter.value_counts().sort_index().to_dict()}')
    return panel


# ==================== 3. 双重分组 ====================

def assign_groups(panel, row_var, col_var, n=3, how='independent'):
    """每个季度内分组（1 最低，n 最高）。independent：两个指标各自在全部公司里分组；dependent：先按行指标分组，
    再在每个行组内部按列指标分组。按名次等分，数值相同的按出现顺序排。公司数不足 n 的季度不要。"""
    cut = lambda s: np.ceil(s.rank(method='first') * n / len(s)).astype(int)
    d = panel.dropna(subset=[row_var, col_var]).copy()
    d = d[d.groupby('quarter')[row_var].transform('size') >= n]
    d['g_row'] = d.groupby('quarter')[row_var].transform(cut)
    d['g_col'] = d.groupby(['quarter', 'g_row'] if how == 'dependent' else 'quarter')[col_var].transform(cut)
    return d


def group_label(g, n, name=''):
    """分组的名字：0 是“全部”，1 是低，n 是高，三分组时中间一组叫中，其余用序号。"""
    if g == 0:
        return '全部'
    return name + {1: '低', n: '高'}.get(g, '中' if n == 3 else str(g))


BP = 1e4  # 日回报的表里用基点（1 基点 = 0.01%）


def port_daily(members, rets, weight='equal'):
    """一个组合在持有期每个交易日的回报（小数）。members 是组合里的公司-季度（要有 code、quarter、mkcap）：在 quarter
    的季末建仓，持有下一个季度，季度末按新的分组换仓。每天对当天有回报的股票加权平均：equal 等权；value 按建仓时的
    季末总市值加权（没有市值的股票不进）。rets 是 daily_returns 的结果。返回以日期为索引的 Series。"""
    m = members[['code', 'quarter', 'mkcap']].assign(quarter=members.quarter.map(next_quarter))
    x = rets.merge(m, on=['code', 'quarter']).dropna(subset=['ret'])
    if weight == 'value':
        x = x.dropna(subset=['mkcap'])
        return (x.ret * x.mkcap).groupby(x.date).sum() / x.mkcap.groupby(x.date).sum()
    return x.groupby('date').ret.mean()


def nw_t(x, lags=None):
    """一列日回报（差）的均值和 Newey-West (1987) t 值（Bartlett 核）。lags 是滞后阶数，None 时按 Newey-West (1994)
    的经验规则取 floor(4·(T/100)^(2/9))，T 是天数。返回 (均值, t 值, 滞后阶数, 天数)。"""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    T = len(x)
    if T < 2:
        return (x.mean() if T else np.nan), np.nan, 0, T
    lags = int(4 * (T / 100) ** (2 / 9)) if lags is None else int(lags)
    u = x - x.mean()
    s = u @ u / T
    for k in range(1, min(lags, T - 1) + 1):
        s += 2 * (1 - k / (lags + 1)) * (u[k:] @ u[:-k]) / T
    se = math.sqrt(s / T) if s > 0 else np.nan
    return x.mean(), x.mean() / se if se else np.nan, lags, T


def hl_daily(high, low, lags=None):
    """最高组减最低组：两个组合每日回报之差（两边都有回报的交易日），返回 nw_t 的结果。"""
    return nw_t((high - low).dropna().to_numpy(), lags)


def method_note(weight, lags):
    return (f'组合在季末建仓、持有下一个季度，{"等权" if weight == "equal" else "按季末市值加权"}；表里是组合日回报的平均（基点）；'
            f'高-低 = 两个组合每日回报之差的平均，t 值用 Newey-West 标准误'
            + ('（滞后阶数按经验规则）' if lags is None else f'（滞后 {lags} 阶）'))


def mean_table(d, value, n=3, scale=1.0, names=('行', '列')):
    """分组均值表（市值等）：每个格子是落在里面的全部公司-季度的等权均值，行、列各多一个“全部”。"""
    d = d.dropna(subset=[value])
    order = list(range(1, n + 1)) + [0]
    avg = lambda r, c: d[((d.g_row == r) | (r == 0)) & ((d.g_col == c) | (c == 0))][value].mean() * scale
    return pd.DataFrame({group_label(c, n, names[1]): [avg(r, c) for r in order] for c in order},
                        index=[group_label(r, n, names[0]) for r in order])


def return_table(d, rets, n=3, names=('行', '列'), weight='equal', lags=None):
    """双重分组的回报表：每个格子是一个组合（行、列各多一个“全部”，即只按另一个指标分组），值是组合在持有期的日回报平均
    （基点）；再加“高-低”一行一列和 t 值（hl_daily）。返回 (DataFrame, 高-低序列的天数和滞后阶数)。"""
    order = list(range(1, n + 1)) + [0]
    pick = lambda r, c: d[((d.g_row == r) | (r == 0)) & ((d.g_col == c) | (c == 0))]
    series = {(r, c): port_daily(pick(r, c), rets, weight) for r in order for c in order}
    out = pd.DataFrame({group_label(c, n, names[1]): [series[r, c].mean() * BP for r in order] for c in order},
                       index=[group_label(r, n, names[0]) for r in order])
    col = [hl_daily(series[r, n], series[r, 1], lags) for r in order]
    out['高-低'] = [x[0] * BP for x in col]
    out['t值'] = [x[1] for x in col]
    rowd = [hl_daily(series[n, c], series[1, c], lags) for c in order]
    out.loc['高-低'] = [x[0] * BP for x in rowd] + [np.nan, np.nan]
    out.loc['t值'] = [x[1] for x in rowd] + [np.nan, np.nan]
    return out, (rowd[-1][3], rowd[-1][2])


def count_table(d, value, n=3, names=('行', '列')):
    """每个格子里有 value 的公司-季度个数。"""
    x = d.dropna(subset=[value])
    order = list(range(1, n + 1)) + [0]
    t = pd.crosstab(x.g_row, x.g_col, margins=True, margins_name='all').rename(index={'all': 0}, columns={'all': 0})
    t = t.reindex(index=order, columns=order, fill_value=0)
    t.index = [group_label(g, n, names[0]) for g in order]
    t.columns = [group_label(g, n, names[1]) for g in order]
    return t


# ==================== 4. 逻辑链大小和深度 ====================

KINDS = {'deep': '深度报告', 'earn': '财报点评', 'other': '其他'}
MEASURES = {'n_concepts': '概念数', 'n_edges': '边数', 'n_text_edges': '原文边数', 'chains': '链条数(log10)',
            'density': '每千字概念数', 'depth': '最长链步数', 'depth_mean': '平均链长', 'depth_flow': '流量平均步数',
            'mean_layer': '平均层级'}  # 列名: 表头用的名字；前五个量大小，后四个量深度，定义见本文件开头
BOILERPLATE = re.compile(r'有关分析师的申明.{0,90}?免责申明。?')  # 个别券商每页开头的声明，清理时没有去掉
PERIOD = r'(?:年报|年度报告|半年报|半年度报告|中报|中期报告|季报|季度报告|业绩快报|业绩预告|业绩预增|业绩公告|经营数据|财报|业绩)'
ANNOUNCE = re.compile(r'(?:发布|披露|公布|公告)[^。；]{0,25}' + PERIOD + '|' + PERIOD + r'[^。；]{0,4}点评')
RESULT = re.compile(r'(?:营业总?收入|总?营收|归母净利润?|净利润)[^。；]{0,40}(?:同比|yoy|YOY|YoY|增速|增长|下降|增幅|降幅)')


def report_kind(text, deep_chars=6000):
    """研报类型：'deep' 深度报告、'earn' 财报点评、'other' 其他。text 是一篇研报的全部正文（各段拼起来）。规则见本文件开头。"""
    text = BOILERPLATE.sub('', text)
    if len(text) >= deep_chars or ('首次覆盖' in text and len(text) >= deep_chars / 2):
        return 'deep'
    head = text[:400]
    return 'earn' if ANNOUNCE.search(head) or RESULT.search(head) else 'other'


def archive_reports(out_dir, paras_path, reports_path=None):
    """从各篇研报的存档（cache/reports/）重建逻辑图，不必等抽取全部跑完、写出 data/graphs/ 再统计。只用元数据表里有、
    指纹对得上的存档（规则和抽取脚本读回存档时相同）；抽取正在后台运行时也能读（存档整篇写完才改名生效）。
    reports_path 是研报元数据表，不给时用和段落表配套的那张。返回精简的逻辑图记录：研报信息、stats、
    nodes（id、layer）、edges（source、target、origin），够 graph_measures 用（分歧度那部分仍然读 data/graphs/）。"""
    from src.logic.output import graph_record
    from src.report_logic import archive_base, archive_path, archive_sig, load_reports, read_archive
    from src.report_text import reports_table_path
    out = OutDir(out_dir)
    reports_path = Path(reports_path) if reports_path else reports_table_path(paras_path)
    rows = pq.read_table(reports_path).to_pylist()
    info = {r['report_id']: r for r in rows if r.get('status', 'ok') == 'ok' and archive_path(out, r['report_id']).exists()}
    base = archive_base()
    recs, stale = [], 0
    for r in load_reports(paras_path, None, info)[0] if info else []:
        done = read_archive(out, r, archive_sig(r, base))
        if done is None:
            stale += 1
            continue
        rec = graph_record(done, {})
        recs.append({**{k: rec[k] for k in ('report_id', 'stock_code', 'company', 'broker', 'date')}, 'stats': rec['stats'],
                     'nodes': [{'id': n['id'], 'layer': n['layer']} for n in rec['nodes']],
                     'edges': [{k: e[k] for k in ('source', 'target', 'origin')} for e in rec['edges']]})
    files = sum(1 for _ in out.reports.rglob('*.pkl')) if out.reports.exists() else 0
    log(f'从存档读逻辑图：{out.reports} 下 {files} 个存档，元数据表里有、指纹对得上的 {len(recs)} 篇'
        f'（{len({r["stock_code"] for r in recs})} 家公司）' + (f'，{stale} 篇指纹对不上没有用' if stale else ''))
    return recs


def graph_measures(rec):
    """一篇研报逻辑图的大小和深度（每千字概念数要正文字数，在 report_sizes 里算）。rec 要有 nodes（id、layer）和
    edges（source、target、origin）。抽取时已经去掉了环，所以能按拓扑序从公司估值往回推。"""
    layer = {n['id']: n['layer'] for n in rec['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(layer)
    g.add_edges_from((e['source'], e['target']) for e in rec['edges'])
    concepts = [n for n in g if layer[n] >= 3]
    h = g.subgraph(nx.ancestors(g, ROOT) | {ROOT} if ROOT in g else set())  # 能走到公司估值的部分
    linked = [n for n in concepts if n in h]
    paths, steps, longest, flow = {}, {}, {}, {}  # 从各节点到公司估值：路径条数、这些路径的步数之和、最长步数、流量的期望步数
    for n in reversed(list(nx.topological_sort(h))):
        if n == ROOT:
            paths[n], steps[n], longest[n], flow[n] = 1, 0, 0, 0.0
            continue
        succ = list(h.successors(n))
        paths[n] = sum(paths[s] for s in succ)
        steps[n] = sum(steps[s] + paths[s] for s in succ)
        longest[n] = 1 + max(longest[s] for s in succ)
        flow[n] = 1 + sum(flow[s] for s in succ) / len(succ)
    starts = [n for n in linked if not any(layer[p] >= 3 for p in h.predecessors(n))]  # 链条起点
    n_paths = sum(paths[n] for n in starts)
    nan = float('nan')
    return {'n_concepts': len(concepts), 'n_edges': g.number_of_edges(),
            'n_text_edges': sum(e['origin'] in TEXT_ORIGINS for e in rec['edges']), 'chains': math.log10(1 + n_paths),
            'depth': max((longest[n] for n in linked), default=0),
            'depth_mean': sum(steps[n] for n in starts) / n_paths if n_paths else nan,
            'depth_flow': float(np.mean([flow[n] for n in linked])) if linked else nan,
            'mean_layer': float(np.mean([layer[n] for n in linked])) if linked else nan}


def report_sizes(out_dir, paras_path, recs=None, deep_chars=6000):
    """每篇研报一行：类型、正文字数和逻辑图的各种大小、深度，写到 data/divergence/report_sizes.parquet。"""
    recs = recs if recs is not None else read_graphs(out_dir)['reports']
    ids = sorted({r['report_id'] for r in recs})
    paras = pq.read_table(paras_path, columns=['report_id', 'para_id', 'text'], filters=[('report_id', 'in', ids)]).to_pandas()
    text = paras.sort_values(['report_id', 'para_id']).groupby('report_id').text.apply(' '.join)
    rows = []
    for r in recs:
        t = text.get(r['report_id'], '')
        m = graph_measures(r)
        m['density'] = m['n_concepts'] / len(t) * 1000 if t else float('nan')
        rows.append({'report_id': r['report_id'], 'stock_code': r['stock_code'], 'company': r['company'], 'broker': r['broker'],
                     'date': r['date'], 'quarter': quarter_of(r['date']), 'kind': report_kind(t, deep_chars), 'n_chars': len(t),
                     **{k: m[k] for k in MEASURES}, 'reach': r['stats']['reach']})
    sizes = pd.DataFrame(rows)
    path = OutDir(out_dir).divergence / 'report_sizes.parquet'
    path.parent.mkdir(parents=True, exist_ok=True)
    sizes.to_parquet(path)
    by = sizes.groupby('kind').agg(n=('report_id', 'size'), chars=('n_chars', 'median'), concepts=('n_concepts', 'mean'),
                                   depth=('depth', 'mean'))
    log(f'研报类型和逻辑图大小：{len(sizes)} 篇、{sizes.stock_code.nunique()} 家公司；'
        + '；'.join(f'{KINDS[k]} {int(v.n)} 篇（正文中位 {v.chars:.0f} 字，平均 {v.concepts:.1f} 个概念，最长链平均 {v.depth:.1f} 步）'
                   for k, v in by.iterrows()) + f' -> {path}')
    return sizes


def company_quarter_size(sizes, size_vars=tuple(MEASURES)):
    """公司-季度内按研报类型取平均：每个算法 v 一列 {v}_deep、{v}_earn、{v}_other，以及各类研报的篇数 n_*。没有这类研报的为空。"""
    key = ['stock_code', 'quarter']
    out = sizes.groupby(key).agg(company=('company', 'first'), n_all=('report_id', 'size'))
    for k in KINDS:
        g = sizes[sizes.kind == k].groupby(key)
        out[f'n_{k}'] = g.size().reindex(out.index).fillna(0).astype(int)
        for v in size_vars:
            out[f'{v}_{k}'] = g[v].mean()
    return out.reset_index()


def single_sort(panel, var, n=5):
    """每个季度内按一个指标分 n 组（1 最低），公司数不足 n 的季度不要。结果在 g 列。"""
    d = panel.dropna(subset=[var]).copy()
    d = d[d.groupby('quarter')[var].transform('size') >= n]
    d['g'] = d.groupby('quarter')[var].transform(lambda s: np.ceil(s.rank(method='first') * n / len(s)).astype(int))
    return d


def single_sort_table(d, rets, values, n=5, weight='equal', lags=None):
    """单变量分组表：每组一行。“日均回报（基点）”是这一组的组合在持有期的日回报平均，最后两行“高-低”“t值”是最高组减
    最低组每日回报差的平均和 Newey-West t 值（hl_daily）；values 是 {表头: (列名, 乘数)}，给出组内公司-季度的等权均值
    （指标本身、市值等，不算高-低）。另有一列公司-季度个数。返回 (DataFrame, hl_daily 的结果)。"""
    groups = list(range(1, n + 1))
    series = {g: port_daily(d[d.g == g], rets, weight) for g in groups}
    hl = hl_daily(series[n], series[1], lags)
    names = [f'第 {g} 组' + {1: '（低）', n: '（高）'}.get(g, '') for g in groups]
    table = pd.DataFrame({'日均回报（基点）': [series[g].mean() * BP for g in groups] + [hl[0] * BP, hl[1]]},
                         index=names + ['高-低', 't值'])
    for title, (col, scale) in values.items():
        table[title] = [d.loc[d.g == g, col].mean() * scale for g in groups] + [np.nan, np.nan]
    table['公司-季度个数'] = [int((d.g == g).sum()) for g in groups] + [np.nan, np.nan]
    return table, hl


def single_sorts(panel, rets, cols, n=5, weight='equal', lags=None):
    """对 cols（{列名: 表头}）里的每个指标各做一次单变量分组（每个季度内分 n 组，组合持有下一个季度）。返回
    (汇总表：每个指标一行，各组日均回报（基点）、高-低和 t 值、最低和最高组的指标均值、个数；
     明细表：每个指标每组的日均回报、指标均值、下季度回报、季末市值和个数；(交易日数, 滞后阶数))。"""
    rows, detail, hl = {}, [], (np.nan, np.nan, 0, 0)
    for col, title in cols.items():
        d = single_sort(panel, col, n)
        t, hl = single_sort_table(d, rets, {'指标均值': (col, 1.0), '下季度回报（%）': ('next_ret', 100.0),
                                            '季末市值（亿元）': ('mkcap', 1e-8)}, n, weight, lags)
        detail.append(t.rename_axis('组别').reset_index().assign(指标=title))
        ret = t['日均回报（基点）']
        rows[title] = {**{f'第{g}组': ret.iloc[g - 1] for g in range(1, n + 1)}, '高-低': ret['高-低'], 't值': ret['t值'],
                       '第1组指标均值': t['指标均值'].iloc[0], f'第{n}组指标均值': t['指标均值'].iloc[n - 1],
                       '公司-季度个数': len(d), '公司数': d.stock_code.nunique(), '交易日数': hl[3]}
    summary = pd.DataFrame(rows).T.astype({'公司-季度个数': int, '公司数': int, '交易日数': int})
    detail = pd.concat(detail)
    return summary, detail[['指标'] + [c for c in detail if c != '指标']], (hl[3], hl[2])


def size_sorts(out_dir, paras_path, daily_path, recs=None, size_vars=tuple(MEASURES), n=5, weight='equal', lags=None,
               deep_chars=6000):
    """逻辑链大小和深度的全流程：研报类型和各算法的值 -> 公司-季度平均 -> 接上行情 -> 深度报告、财报点评分别按每个算法
    做一次单变量分组，各组组合持有下一个季度。size_sort_{类型}.csv 每个算法一行（各组日均回报、高-低和 t 值、最低和
    最高组的指标均值）；size_sort_detail.csv 是每个算法每组的日均回报、指标均值、下季度回报、季末市值和个数。"""
    out = OutDir(out_dir).divergence
    sizes = report_sizes(out_dir, paras_path, recs, deep_chars)
    panel = build_panel(company_quarter_size(sizes, size_vars), daily_path)
    panel.to_parquet(out / 'company_quarter_size.parquet')
    rets = daily_returns(daily_path, set(panel.code), set(panel.quarter))
    name = lambda v: MEASURES.get(v, v)
    detail, summary = [], {}
    for k in ('deep', 'earn'):
        x = sizes.loc[sizes.kind == k, list(size_vars)]
        log(f'\n【{KINDS[k]}：各算法之间的秩相关（研报层面，{len(x)} 篇）】\n'
            + x.rank().corr().rename(index=name, columns=name).round(2).to_string())
        summary[k], det, (days, lag) = single_sorts(panel, rets, {f'{v}_{k}': name(v) for v in size_vars}, n, weight, lags)
        detail.append(det.assign(类型=KINDS[k]))
        summary[k].to_csv(out / f'size_sort_{k}.csv', encoding='utf-8-sig', float_format='%.4f')
        log(f'\n【{KINDS[k]}：公司-季度内取平均后，每个季度内按各算法分 {n} 组，组合持有下一个季度的日均回报（基点）】\n'
            f'{method_note(weight, lags)}（{days} 个交易日、滞后 {lag} 阶）\n{summary[k].round(2).to_string(na_rep="")}')
    detail = pd.concat(detail)
    detail[['类型'] + [c for c in detail if c != '类型']].to_csv(
        out / 'size_sort_detail.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    log(f'\n结果在 {out}：report_sizes.parquet、company_quarter_size.parquet、size_sort_deep.csv、size_sort_earn.csv、size_sort_detail.csv')
    return panel, summary


# ==================== 5. 分歧度对股票收益和财务特征的回归 ====================

MIN_DAYS = 30  # 一只股票一个季度至少要有这么多个交易日（不含停牌）才算特质波动和崩盘指标
RET_VARS = {'next_ret': '下季收益（%）', 'car': 'CAR（%）', 'mom': '动量 12-2 月（%）', 'rev': '反转 近 1 月（%）',
            'ivol': '特质波动 当季（%/日）', 'ivol_next': '特质波动 下季（%/日）', 'ncskew': 'NCSKEW 当季',
            'ncskew_next': 'NCSKEW 下季', 'duvol': 'DUVOL 当季', 'duvol_next': 'DUVOL 下季'}
FIN_VARS = {'size': '规模 ln(市值/亿元)', 'bm': 'BM', 'ep': 'EP（%）', 'sp': 'SP', 'lev': '杠杆（%）', 'roa': 'ROA（%）',
            'rev_g': '营收增长（%）', 'np_g': '利润增长（%）', 'd_roa_next': '下季 ROA 变化（百分点）',
            'rev_g_next': '下季营收增长（%）', 'np_g_next': '下季利润增长（%）'}
CONTROLS = {'股票收益': ['size', 'bm', 'ln_n'], '财务特征': ['size', 'ln_n']}  # 加控制变量的模型用哪些（因变量本身不放）


def month_back(date, k):
    """date（YYYYMMDD）往前 k 个月的那个月的月末。"""
    return (pd.Period(date[:6], 'M') - k).end_time.strftime('%Y%m%d')


def load_daily(daily_path, start, end, codes=None):
    """start 到 end 的日行情，按股票排好，ret 是相对前一行的后复权收盘价回报（每只股票第一行为空）。"""
    filters = [('date', '>=', start), ('date', '<=', end)] + ([('code', 'in', sorted(codes))] if codes is not None else [])
    d = pq.read_table(daily_path, columns=['date', 'code', 'close', 'mkcap', 'is_suspended'], filters=filters).to_pandas()
    d = d.sort_values(['code', 'date']).reset_index(drop=True)
    d['ret'] = d.close / d.groupby('code').close.shift() - 1
    return d


def market_index(d):
    """全部 A 股的市值加权日回报，权重是前一个交易日的总市值。返回以 date 为索引的 Series。"""
    x = d.assign(w=d.groupby('code').mkcap.shift()).dropna(subset=['ret', 'w'])
    return (x.ret * x.w).groupby(x.date).sum() / x.w.groupby(x.date).sum()


def crash_stats(r, M):
    """一只股票一个季度的日回报 r（只含交易日）和对齐的市场回报矩阵 M（第一列为当天，其后为前后各两天）。
    特质波动：r 对当天市场回报回归（市场模型）的残差标准差（%/日）。崩盘指标按 Kim, Li & Zhang (2011)：r 对当天和前后各两天的
    市场回报回归（扩展市场模型），W = ln(1 + 残差) 去均值后，
        NCSKEW = -[n(n-1)^1.5 ΣW³] / [(n-1)(n-2)(ΣW²)^1.5]
        DUVOL  = ln[(n_u - 1) Σ_{W<0} W² / ((n_d - 1) Σ_{W>0} W²)]，n_d、n_u 是 W 低于、高于均值的天数。"""
    n = len(r)
    if n < MIN_DAYS:
        return np.nan, np.nan, np.nan
    resid = lambda X: r - X @ np.linalg.lstsq(X, r, rcond=None)[0]
    one = np.ones((n, 1))
    ivol = resid(np.hstack([one, M[:, :1]])).std(ddof=2) * 100
    w = np.log1p(resid(np.hstack([one, M])))
    w = w - w.mean()
    s2, s3 = (w ** 2).sum(), (w ** 3).sum()
    ncskew = -(n * (n - 1) ** 1.5 * s3) / ((n - 1) * (n - 2) * s2 ** 1.5)
    down, up = w < 0, w > 0
    duvol = math.log((up.sum() - 1) * (w[down] ** 2).sum() / ((down.sum() - 1) * (w[up] ** 2).sum())) if down.sum() > 1 and up.sum() > 1 else np.nan
    return ivol, ncskew, duvol


def stock_chars(daily_path, panel, cq_reports=None, car_window=(-3, 1)):
    """股票收益类变量，每个公司-季度一行（code、quarter）：
        next_ret      下季收益（已在面板里）
        car           当季参与分歧度的研报，发布日前后 car_window 个交易日的累计超额收益（个股减市场），在公司-季度内取平均；
                      发布日不是交易日时取之后的第一个交易日
        mom、rev      季末的动量（季末前 12 个月到前 1 个月的回报）、反转（季末前 1 个月的回报）
        ivol、ncskew、duvol             当季的特质波动、崩盘指标（crash_stats，只用交易日，停牌日不算）
        ivol_next、ncskew_next、duvol_next   下一个季度的
    市场回报是全部 A 股的市值加权日回报（market_index）。"""
    quarters = sorted(set(panel.quarter))
    hold = sorted(set(quarters) | {next_quarter(q) for q in quarters})
    start, end = month_back(quarter_end(quarters[0]), 13), quarter_end(hold[-1])
    log(f'回归用的股票变量：读全部 A 股 {start}-{end} 的日行情算市场回报 ...')
    mkt = market_index(load_daily(daily_path, month_back(quarter_start(hold[0]), 2), end))
    cal = mkt.index.to_numpy()
    M = pd.DataFrame({'m0': mkt, 'm_1': mkt.shift(1), 'm_2': mkt.shift(2), 'm1': mkt.shift(-1), 'm2': mkt.shift(-2)})
    d = load_daily(daily_path, start, end, set(panel.code))
    d['quarter'] = d.date.map(quarter_of)
    rows = []
    for (code, q), x in d[d.quarter.isin(hold) & ~d.is_suspended.astype(bool)].dropna(subset=['ret']).groupby(['code', 'quarter']):
        m = M.reindex(x.date)
        ok = m.notna().all(axis=1).to_numpy()
        ivol, ncskew, duvol = crash_stats(x.ret.to_numpy()[ok], m.to_numpy()[ok])
        rows.append({'code': code, 'quarter': q, 'ivol': ivol, 'ncskew': ncskew, 'duvol': duvol})
    risk = pd.DataFrame(rows)
    prev = {next_quarter(p): p for p in quarters}
    nxt = risk[risk.quarter.isin(prev)].assign(quarter=lambda x: x.quarter.map(prev)).rename(
        columns={'ivol': 'ivol_next', 'ncskew': 'ncskew_next', 'duvol': 'duvol_next'})
    price = d.set_index('date').groupby('code').close
    at = lambda code, date: price.get_group(code).loc[:date].iloc[-1] if code in price.groups and len(price.get_group(code).loc[:date]) else np.nan
    out = panel[['code', 'quarter', 'next_ret']].copy()
    out['next_ret'] *= 100
    p0, p1, p12 = ([at(c, month_back(quarter_end(q), k)) for c, q in zip(out.code, out.quarter)] for k in (0, 1, 12))
    out['mom'] = (np.array(p1) / np.array(p12) - 1) * 100
    out['rev'] = (np.array(p0) / np.array(p1) - 1) * 100
    out = out.merge(risk, on=['code', 'quarter'], how='left').merge(nxt, on=['code', 'quarter'], how='left')
    if cq_reports is not None and len(cq_reports):
        ar = d[['date', 'code', 'ret']].merge(mkt.rename('m').reset_index(), on='date')
        ar = ar.assign(ar=ar.ret - ar.m).set_index(['code', 'date']).ar
        lo, hi = car_window
        cars = []
        for r in cq_reports.itertuples():
            i = int(np.searchsorted(cal, r.date))
            days = cal[max(i + lo, 0): i + hi + 1]
            v = ar.reindex(pd.MultiIndex.from_product([[r.stock_code.split('.')[0]], days]))
            cars.append(v.sum() * 100 if i + hi < len(cal) and v.notna().all() else np.nan)
        car = cq_reports.assign(code=cq_reports.stock_code.str.split('.').str[0], car=cars).groupby(['code', 'quarter']).car.mean()
        out = out.merge(car.reset_index(), on=['code', 'quarter'], how='left')
    return out


def fin_chars(fin_path, panel):
    """财务特征，每个公司-季度一行。当季末的按季末之前已经公告的最新一期财务报表（同 fundamentals）和季末总市值：
        size = ln(总市值/亿元)；bm = (总资产-总负债)/总市值；ep = TTM 归母净利润/总市值；sp = TTM 营收/总市值；
        lev = 总负债/总资产；roa = TTM 归母净利润/总资产；rev_g、np_g = TTM 营收、TTM 归母净利润相对上年同期报告期的增长
        （利润增长的分母取绝对值）
    下季的用报告期是下一个季度末的那期报表（季末时还没公告）：d_roa_next = 那期 ROA 减当季末已知的 ROA，rev_g_next、
    np_g_next 是那期的同比增长。百分比变量都乘 100。"""
    cols = ['total_assets', 'total_liability', 'operating_revenue_ttm', 'np_parent_company_owners_ttm']
    fin = pd.read_parquet(fin_path, columns=['code', 'report_date', 'date'] + cols)
    fin = fin[fin.code.isin(set(panel.code))].astype({'report_date': str})
    fin = fin.sort_values(['code', 'report_date', 'date']).groupby(['code', 'report_date']).tail(1)
    by = fin.set_index(['code', 'report_date'])[cols]
    get = lambda codes, dates, col: by[col].reindex(pd.MultiIndex.from_arrays([codes, dates])).to_numpy()
    year_ago = lambda dates: [f'{int(x[:4]) - 1}{x[4:]}' if isinstance(x, str) else None for x in dates]
    known = fundamentals(fin_path, set(panel.code), set(panel.quarter), cols)
    out = panel[['code', 'quarter', 'mkcap']].merge(known, on=['code', 'quarter'], how='left')
    rd = [str(x) if pd.notna(x) else None for x in out.fin_report_date]
    ta, tl = out.fin_total_assets.to_numpy(), out.fin_total_liability.to_numpy()
    rev, np_ = out.fin_operating_revenue_ttm.to_numpy(), out.fin_np_parent_company_owners_ttm.to_numpy()
    rev0, np0 = get(out.code, year_ago(rd), 'operating_revenue_ttm'), get(out.code, year_ago(rd), 'np_parent_company_owners_ttm')
    res = pd.DataFrame({'code': out.code, 'quarter': out.quarter, 'size': np.log(out.mkcap / 1e8), 'bm': (ta - tl) / out.mkcap,
                        'ep': np_ / out.mkcap * 100, 'sp': rev / out.mkcap, 'lev': tl / ta * 100, 'roa': np_ / ta * 100,
                        'rev_g': (rev / rev0 - 1) * 100, 'np_g': (np_ - np0) / np.abs(np0) * 100})
    nd = [quarter_end(next_quarter(q)) for q in out.quarter]
    ta1, rev1, np1 = (get(out.code, nd, c) for c in ('total_assets', 'operating_revenue_ttm', 'np_parent_company_owners_ttm'))
    rev10, np10 = get(out.code, year_ago(nd), 'operating_revenue_ttm'), get(out.code, year_ago(nd), 'np_parent_company_owners_ttm')
    res['d_roa_next'] = np1 / ta1 * 100 - res.roa
    res['rev_g_next'] = (rev1 / rev10 - 1) * 100
    res['np_g_next'] = (np1 - np10) / np.abs(np10) * 100
    return res.replace([np.inf, -np.inf], np.nan)


def t_pvalue(t, df):
    """t 分布的双侧 p 值（数值积分，不依赖 scipy）。"""
    if not np.isfinite(t) or df < 1:
        return np.nan
    x = np.linspace(0, abs(t), 20001)
    f = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi) * (1 + x ** 2 / df) ** (-(df + 1) / 2)
    return max(0.0, 1 - 2 * float(((f[1:] + f[:-1]) / 2).sum() * (x[1] - x[0])))


def ols_cluster(y, X, groups):
    """OLS，标准误按 groups 聚类（CR1：乘 G/(G-1)·(N-1)/(N-K)）。返回 (系数, 标准误, 聚类数)。"""
    n, k = X.shape
    inv = np.linalg.pinv(X.T @ X)
    b = inv @ X.T @ y
    e = y - X @ b
    _, idx = np.unique(groups, return_inverse=True)
    G = idx.max() + 1
    S = np.zeros((G, k))
    np.add.at(S, idx, X * e[:, None])
    V = inv @ (S.T @ S) @ inv * G / (G - 1) * (n - 1) / (n - k)
    return b, np.sqrt(np.diag(V)), G


def winsor(s, p=0.01):
    lo, hi = s.quantile([p, 1 - p])
    return s.clip(lo, hi)


def stars(p):
    return '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''


def div_regressions(d, ys, div_vars, controls, labels):
    """每个因变量 y：分别对每个分歧指标回归，再把除总分歧外的指标放在一起回归（主题项 + 关系项）；两个模型：只有季度固定效应、
    再加 controls（因变量本身不作控制）。分歧指标先在样本里标准化（系数 = 分歧度高 1 个标准差时 y 的变化），因变量和
    控制变量按 1%/99% 缩尾；标准误按公司聚类，p 值用自由度为公司数 - 1 的 t 分布。返回长表。"""
    d = d.copy()
    for v in div_vars:
        d[v] = (d[v] - d[v].mean()) / d[v].std()
    for c in set(ys) | set(controls):
        d[c] = winsor(d[c])
    specs = [[v] for v in div_vars] + [[v for v in div_vars if v != 'total']]
    rows = []
    for y in ys:
        for model, ctrl in (('只有季度固定效应', []), ('加控制变量', [c for c in controls if c != y])):
            for xs in specs:
                x = d.dropna(subset=[y] + xs + ctrl)
                fe = pd.get_dummies(x.quarter, drop_first=True, dtype=float).to_numpy()
                X = np.column_stack([np.ones(len(x)), x[xs + ctrl].to_numpy(float), fe])
                b, se, G = ols_cluster(x[y].to_numpy(float), X, x.code.to_numpy())
                for i, v in enumerate(xs, start=1):
                    t = b[i] / se[i]
                    rows.append({'因变量': labels[y], '模型': model, '回归': '单独' if len(xs) == 1 else '联合',
                                 '分歧指标': v, '系数': b[i], 't值': t, 'p值': t_pvalue(t, G - 1), '样本数': len(x), '公司数': G,
                                 '因变量均值': x[y].mean(), '因变量标准差': x[y].std(), '控制变量': '、'.join(ctrl)})
    return pd.DataFrame(rows)


def reg_table(res, model, names):
    """长表转成打印用的宽表：每个因变量一行，每个回归一列“系数星号 (t)”。"""
    r = res[res.模型 == model]
    cell = lambda x: f'{x.系数:.3f}{stars(x.p值)} ({x.t值:.2f})'
    cols = {}
    for (kind, v), g in r.groupby(['回归', '分歧指标'], sort=False):
        cols[names.get(v, v) + ('（联合）' if kind == '联合' else '')] = g.set_index('因变量').apply(cell, axis=1)
    t = pd.DataFrame(cols)
    first = r[r.回归 == '单独'].groupby('因变量', sort=False).first()
    t['样本数'] = first.样本数.astype(int)
    t['均值'] = first.因变量均值.map('{:.3f}'.format)
    t['标准差'] = first.因变量标准差.map('{:.3f}'.format)
    return t.loc[r.因变量.unique()]


def regressions(out_dir, panel, daily_path, fin_path, div_vars=('total', 'theme', 'rel'), labels=None, car_window=(-3, 1)):
    """三种分歧度对股票收益、财务特征两组变量的回归（div_regressions）。变量表写到 panel_reg.parquet，
    结果写到 reg_returns.csv、reg_financial.csv（长表：每个因变量、模型、回归、分歧指标一行）。"""
    out = OutDir(out_dir).divergence
    path = out / 'company_quarter_reports.parquet'
    cq_reports = pd.read_parquet(path) if path.exists() else None
    if cq_reports is None:
        log(f'没有 {path.name}（重跑一次 --task divergence 会写出），CAR 不算')
    s = stock_chars(daily_path, panel, cq_reports, car_window)
    f = fin_chars(fin_path, panel)
    d = panel[['stock_code', 'company', 'code', 'quarter', 'n_reports'] + list(div_vars)].merge(
        s, on=['code', 'quarter'], how='left').merge(f, on=['code', 'quarter'], how='left')
    d['ln_n'] = np.log(d.n_reports)
    d.to_parquet(out / 'panel_reg.parquet')
    names = {v: (labels or {}).get(v, v) for v in div_vars}
    ret_vars = dict(RET_VARS, car=f'CAR[{car_window[0]},+{car_window[1]}]（%）')
    log(f'\n回归样本：{len(d)} 个公司-季度、{d.code.nunique()} 家公司、{d.quarter.nunique()} 个季度。分歧指标已标准化，'
        f'系数是分歧度高 1 个标准差时因变量的变化；括号里是按公司聚类的 t 值；*** 1%、** 5%、* 10% 显著；因变量和控制变量 1%/99% 缩尾。\n'
        f'“联合”列是主题项和关系项放进同一个回归。控制变量：股票收益组 ln 市值、BM、ln 研报数；财务特征组 ln 市值、ln 研报数（因变量本身不放）。')
    results = {}
    for group, ys, file in (('股票收益', ret_vars, 'reg_returns.csv'), ('财务特征', FIN_VARS, 'reg_financial.csv')):
        res = div_regressions(d, list(ys), list(div_vars), CONTROLS[group], ys)
        res['分歧指标'] = res.分歧指标.map(names)
        res.to_csv(out / file, index=False, encoding='utf-8-sig', float_format='%.6f')
        results[group] = res
        for model in ('只有季度固定效应', '加控制变量'):
            log(f'\n【{group}：{model}】\n{reg_table(res, model, {}).to_string()}')
    log(f'\n结果在 {out}：panel_reg.parquet（回归用的变量）、reg_returns.csv、reg_financial.csv')
    return d, results


# ==================== 入口 ====================

def main(out_dir, daily_path, fin_path=None, fin_cols=(), task='all', min_reports=2, row_var='theme', col_var='rel',
         n_groups=3, how='independent', weight='equal', nw_lags=None, labels=None, paras_path=None, size_vars=None,
         size_groups=5, deep_chars=6000, size_source='archive', reports_path=None, single_vars=('total', 'theme', 'rel'),
         single_groups=3, reg_vars=('total', 'theme', 'rel'), car_window=(-3, 1)):
    """task：'divergence' 只算公司-季度分歧度；'sort' 用已有的分歧度表匹配股票数据并做双重分组；'size' 逻辑链大小和深度的分组；
    'reg' 用已有的分歧度表，对股票收益和财务特征两组变量回归（regressions）；'all' 全部。reg_vars：回归用哪些分歧指标；
    car_window：研报发布窗口的累计超额收益取发布日前后哪几个交易日。size_vars：逻辑链大小和深度用哪些算法（MEASURES 的列名），None 为全部。size_source：逻辑链大小和深度
    的逻辑图从哪里读，'archive' 各篇研报的存档（最新），'graphs' data/graphs/（上一次完整运行抽取时写出的）；
    分歧度总是读 data/graphs/，因为它要用框架对比给概念归的主题。weight：组合等权（'equal'）还是按季末市值加权
    （'value'）；nw_lags：高-低 t 值的 Newey-West 滞后阶数，None 按经验规则（见 nw_t）。single_vars、single_groups：
    双重分组之前，先按这些分歧指标各自做一次单变量分组（分几组），空表示不做。"""
    if weight not in ('equal', 'value'):
        raise SystemExit(f"weight 只能是 'equal' 或 'value'，收到 {weight!r}")
    out = OutDir(out_dir).divergence
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 250)
    pd.set_option('display.unicode.east_asian_width', True)
    graphs_needed = task in ('all', 'divergence') or (task == 'size' and size_source == 'graphs')
    recs = read_graphs(out_dir)['reports'] if graphs_needed else None
    if task in ('all', 'size'):
        if size_source not in ('archive', 'graphs'):
            raise SystemExit(f"size_source 只能是 'archive' 或 'graphs'，收到 {size_source!r}")
        size_recs = archive_reports(out_dir, paras_path, reports_path) if size_source == 'archive' else recs
        unknown = [v for v in size_vars or () if v not in MEASURES]
        if unknown:
            raise SystemExit(f'没有这些逻辑链算法：{unknown}，可选 {list(MEASURES)}')
        size_sorts(out_dir, paras_path, daily_path, size_recs, list(size_vars or MEASURES), size_groups, weight, nw_lags,
                   deep_chars)
        log('')
    if task == 'size':
        return
    if task in ('all', 'divergence'):
        cq = company_quarter(out_dir, min_reports, recs)
    if task == 'divergence':
        return
    if task in ('sort', 'reg'):
        cq = pd.read_parquet(out / 'company_quarter.parquet')
        cq = cq[cq.n_reports >= min_reports]
    panel = build_panel(cq, daily_path, fin_path, fin_cols)
    panel.to_parquet(out / 'panel.parquet')
    label = lambda v: (labels or {}).get(v, v)
    if task == 'reg':
        return regressions(out_dir, panel, daily_path, fin_path, reg_vars, labels, car_window)
    rets = daily_returns(daily_path, set(panel.code), set(panel.quarter))
    if single_vars:
        s, det, (days, lags) = single_sorts(panel, rets, {v: label(v) for v in single_vars}, single_groups, weight, nw_lags)
        s.to_csv(out / 'sort_single.csv', encoding='utf-8-sig', float_format='%.4f')
        det.to_csv(out / 'sort_single_detail.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
        log(f'\n【单变量分组：每个季度内按各分歧指标分 {single_groups} 组，组合持有下一个季度的日均回报（基点）】\n'
            f'{method_note(weight, nw_lags)}（{days} 个交易日、滞后 {lags} 阶）\n{s.round(2).to_string(na_rep="")}')
    names = (label(row_var), label(col_var))
    d = assign_groups(panel, row_var, col_var, n_groups, how)
    ret_table, (days, lags) = return_table(d, rets, n_groups, names, weight, nw_lags)
    log(f'\n双重分组：每个季度内按 {names[0]}（行）和 {names[1]}（列）各分 {n_groups} 组（{how}），{d.quarter.nunique()} 个季度、'
        f'{len(d)} 个公司-季度\n{method_note(weight, nw_lags)}（{days} 个交易日、滞后 {lags} 阶）')
    tables = {'持有下一个季度的组合日均回报（基点）': ('sort_next_ret.csv', ret_table, 2),
              '季末市值均值（亿元）': ('sort_mkcap.csv', mean_table(d, 'mkcap', n_groups, 1e-8, names), 1),
              '有下季度回报的公司-季度个数': ('sort_count.csv', count_table(d, 'next_ret', n_groups, names), 0)}
    for title, (name, table, digits) in tables.items():
        table.to_csv(out / name, encoding='utf-8-sig', float_format=f'%.{digits + 2}f')
        log(f'\n【{title}】\n{table.round(digits).to_string(na_rep="")}')
    d.to_parquet(out / 'panel_sorted.parquet')
    log(f'\n结果在 {out}：company_quarter.parquet、panel.parquet、panel_sorted.parquet（带分组）、sort_*.csv'
        + ('（sort_single*.csv 是单变量分组）' if single_vars else ''))
    if task == 'all' and reg_vars:
        regressions(out_dir, panel, daily_path, fin_path, reg_vars, labels, car_window)
    return panel, tables
