"""只用原文的逻辑图：图结构指标，以及它们和股票收益、财务特征的关系（不调用大模型，可以反复运行）。

“原文图”是逻辑图去掉大模型自己补全的逻辑边（origin = logic）之后剩下的部分，包括：
- 有原文证据的边：extracted 是原文抽取的；bridged 是跨段补全时附了原文引文的。
- 预置的会计关系 preset：会计科目之间、科目到公司估值的固定关系。
口径和框架对比 edge_set='text' 相同（src/report_compare.py 的 TEXT_EDGES）。节点不变：概念都是从原文抽出来的，
有些概念只靠补全边才连到估值，去掉补全边后就悬空了。

每篇研报的指标（text_measures）：
    n_concepts   概念数：第 3-5 层的概念个数
    n_text       原文边数：extracted、bridged 边的条数（不含预置的会计关系）
    degree       每个概念的原文边数：n_text / n_concepts，概念之间连得多密
    modules      论点模块数：只看概念层（第 3-5 层）和原文边时，弱连通分量的个数（单个概念也算一个）。一个模块是
                 一组互相有原文因果关系的概念，模块多说明研报的论述分成几条互不相干的线
    top_module   最大模块占比：最大的论点模块里的概念数 / 概念数
    subgraphs    走不到估值的子图数：去掉补全边后，走不到公司估值的概念按弱连通分组的个数。和抽取流程第 3 步统计的
                 “走不到估值的子图”是同一个定义（src/logic/graph.py 的 subgraphs）
    reach        连到估值的概念占比：只沿原文边和预置关系能走到公司估值的概念 / 概念数
    entries      会计入口数：原文边从概念直接指向的不同会计科目（第 2 层）和公司估值的个数，即论述落到几个财务科目上
    chains       原文链条数：log10(1 + 从链条起点沿原文图走到公司估值的路径条数)
    depth        原文最长链步数：从概念沿原文图走到公司估值最多几步
    mean_layer   平均层级：能连到估值的概念平均在第几层

分析分两层：
1. 公司-季度：同一家公司、同一个季度的研报取平均（一篇也算）。因变量是 divergence_sort 第 5 部分的股票收益变量
   （stock_chars）和财务特征变量（fin_chars）。每个指标单独回归，两个模型：
   - 只有季度固定效应；
   - 再加控制变量：ln 市值、BM（财务特征组不放）、ln 研报数、ln 平均正文字数、深度报告占比。
   另外按每个指标在季度内分 5 组，看组合持有下一个季度的日均回报（AHXZ 做法，同 divergence_sort.single_sorts）。
2. 研报：每篇研报发布前后 CAR_WINDOW 个交易日的累计超额收益，以及发布后 DRIFT_WINDOW 个交易日的漂移，对这篇研报
   的指标回归。控制变量：ln 正文字数、研报类型、季末 ln 市值和 BM，加季度固定效应。
指标标准化（系数 = 指标高 1 个标准差时因变量的变化），因变量和控制变量 1%/99% 缩尾，标准误按公司聚类。
结果写到 <输出目录>/data/divergence/ 下的 text_*.parquet 和 text_*.csv。
"""
import math

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import divergence_sort as ds
from src.config import OutDir
from src.fin_items import ROOT
from src.logic.graph import subgraphs

DROP_ORIGINS = ('logic', 'part_of')  # 原文图去掉的边：大模型补全的逻辑边（part_of 只在缩减前的阶段图里，这里不会出现）
TEXT_ONLY = ('extracted', 'bridged')  # 计入“原文边数”的边
MEASURES = {'n_concepts': '概念数', 'n_text': '原文边数', 'degree': '每个概念的原文边数', 'modules': '论点模块数',
            'top_module': '最大模块占比', 'subgraphs': '走不到估值的子图数', 'reach': '连到估值的概念占比',
            'entries': '会计入口数', 'chains': '原文链条数(log10)', 'depth': '原文最长链步数', 'mean_layer': '平均层级'}
CONTROLS = {'股票收益': ['size', 'bm', 'ln_n', 'ln_chars', 'deep'], '财务特征': ['size', 'ln_n', 'ln_chars', 'deep']}
CONTROL_NAMES = {'size': 'ln 市值', 'bm': 'BM', 'ln_n': 'ln 研报数', 'ln_chars': 'ln 正文字数', 'deep': '深度报告占比',
                 'is_deep': '深度报告', 'is_earn': '财报点评'}
CAR_WINDOW = (-3, 1)  # 研报发布窗口：发布日前 3 个交易日到后 1 个交易日（和分歧度回归相同）
DRIFT_WINDOW = (2, 60)  # 发布后的漂移：发布后第 2 到第 60 个交易日
SORT_GROUPS = 5


def text_graph(rec):
    """一篇研报的原文图：去掉补全的逻辑边。返回 (DiGraph, {节点: 层})。"""
    layer = {n['id']: n['layer'] for n in rec['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(layer)
    g.add_edges_from((e['source'], e['target']) for e in rec['edges'] if e['origin'] not in DROP_ORIGINS)
    return g, layer


def text_measures(rec):
    """一篇研报原文图的各项指标，定义见本文件开头。"""
    g, layer = text_graph(rec)
    concepts = [n for n in g if layer[n] >= 3]
    nan = float('nan')
    n_text = sum(e['origin'] in TEXT_ONLY for e in rec['edges'])
    mods = [len(c) for c in nx.weakly_connected_components(g.subgraph(concepts))]
    conn = nx.ancestors(g, ROOT) | {ROOT} if ROOT in g else set()
    linked = [n for n in concepts if n in conn]
    h = g.subgraph(conn)
    paths, longest = {}, {}
    for n in reversed(list(nx.topological_sort(h))):
        succ = list(h.successors(n))
        paths[n] = 1 if n == ROOT else sum(paths[s] for s in succ)
        longest[n] = 0 if n == ROOT else 1 + max(longest[s] for s in succ)
    starts = [n for n in linked if not any(layer[p] >= 3 for p in h.predecessors(n))]
    entries = {v for u, v in g.edges if layer[u] >= 3 and layer[v] <= 2}
    k = len(concepts)
    return {'n_concepts': k, 'n_text': n_text, 'degree': n_text / k if k else nan, 'modules': len(mods),
            'top_module': max(mods) / k if k else nan,
            'subgraphs': sum(any(layer.get(n, 0) >= 3 for n in s) for s in subgraphs(g)),
            'reach': len(linked) / k if k else nan, 'entries': len(entries),
            'chains': math.log10(1 + sum(paths[n] for n in starts)), 'depth': max((longest[n] for n in linked), default=0),
            'mean_layer': float(np.mean([layer[n] for n in linked])) if linked else nan}


def report_table(recs, paras_path, deep_chars=6000):
    """每篇研报一行：研报信息、类型、正文字数、原文图的各项指标，以及完整图（含补全边）的逻辑边条数和连到估值的概念占比。"""
    ids = sorted({r['report_id'] for r in recs})
    paras = pq.read_table(paras_path, columns=['report_id', 'para_id', 'text'], filters=[('report_id', 'in', ids)]).to_pandas()
    text = paras.sort_values(['report_id', 'para_id']).groupby('report_id').text.apply(' '.join)
    rows = []
    for r in recs:
        t = text.get(r['report_id'], '')
        rows.append({'report_id': r['report_id'], 'stock_code': r['stock_code'], 'company': r['company'], 'broker': r['broker'],
                     'date': r['date'], 'quarter': ds.quarter_of(r['date']), 'kind': ds.report_kind(t, deep_chars),
                     'n_chars': len(t), **text_measures(r), 'n_logic': sum(e['origin'] == 'logic' for e in r['edges']),
                     'reach_full': r['stats']['reach']})
    rep = pd.DataFrame(rows)
    rep['code'] = rep.stock_code.str.split('.').str[0]
    rep['ln_chars'] = np.log(rep.n_chars.clip(lower=1))
    return rep


def company_quarter_text(rep):
    """公司-季度内取平均：各项指标、ln 正文字数；另有研报数 n_reports、ln 研报数 ln_n、深度报告占比 deep。"""
    key = ['stock_code', 'quarter']
    cq = rep.groupby(key)[list(MEASURES) + ['ln_chars']].mean()
    cq['n_reports'] = rep.groupby(key).size()
    cq['deep'] = rep.assign(x=rep.kind.eq('deep')).groupby(key).x.mean()
    cq['company'] = rep.groupby(key).company.first()
    cq = cq.reset_index()
    cq['ln_n'] = np.log(cq.n_reports)
    return cq


def report_cars(daily_path, rep, windows):
    """每篇研报在 windows（{列名: (lo, hi)}，相对发布日的交易日）里的累计超额收益（%），超额收益 = 个股日回报 - 全部 A 股
    市值加权日回报。发布日不是交易日时以之后的第一个交易日为第 0 天；窗口里缺行情超过一成的记为空。"""
    lo = min(w[0] for w in windows.values())
    hi = max(w[1] for w in windows.values())
    start = ds.month_back(min(rep.date), 2)
    end = (pd.Timestamp(max(rep.date)) + pd.Timedelta(days=int(hi * 1.6) + 30)).strftime('%Y%m%d')
    ds.log(f'研报窗口收益：读全部 A 股 {start}-{end} 的日行情算市场回报 ...')
    mkt = ds.market_index(ds.load_daily(daily_path, start, end))
    cal = mkt.index.to_numpy()
    d = ds.load_daily(daily_path, start, end, set(rep.code))
    ar = d[['date', 'code', 'ret']].merge(mkt.rename('m').reset_index(), on='date')
    ar = ar.assign(ar=ar.ret - ar.m).set_index(['code', 'date']).ar.sort_index()
    out = {c: [] for c in windows}
    for code, date in zip(rep.code, rep.date):
        i = int(np.searchsorted(cal, date))
        for c, (a, b) in windows.items():
            if i + a < 0 or i + b >= len(cal):
                out[c].append(np.nan)
                continue
            v = ar.reindex(pd.MultiIndex.from_product([[code], cal[i + a: i + b + 1]]))
            out[c].append(v.sum() * 100 if v.notna().mean() >= 0.9 else np.nan)
    return rep.assign(**out)


def regress(d, ys, xs, controls, cluster='code', fe='quarter'):
    """每个因变量 y 对每个指标 x 单独回归（x 标准化），可加 controls（y 本身不作控制）和 fe 固定效应，标准误按 cluster 聚类，
    p 值用自由度为聚类数 - 1 的 t 分布。因变量和控制变量 1%/99% 缩尾。返回长表。"""
    d = d.copy()
    for c in set(ys) | {c for c in controls if d[c].nunique() > 2}:
        d[c] = ds.winsor(d[c])
    rows = []
    for y in ys:
        ctrl = [c for c in controls if c != y]
        for x in xs:
            z = d.dropna(subset=[y, x] + ctrl)
            if len(z) < 30 or z[x].std() == 0:
                continue
            fx = pd.get_dummies(z[fe], drop_first=True, dtype=float).to_numpy()
            X = np.column_stack([np.ones(len(z)), (z[x] - z[x].mean()) / z[x].std(), z[ctrl].to_numpy(float), fx])
            b, se, G = ds.ols_cluster(z[y].to_numpy(float), X, z[cluster].to_numpy())
            t = b[1] / se[1]
            rows.append({'因变量': y, '指标': x, '系数': b[1], 't值': t, 'p值': ds.t_pvalue(t, G - 1), '样本数': len(z),
                         '公司数': G, '因变量均值': z[y].mean(), '因变量标准差': z[y].std(), '控制变量': '、'.join(ctrl)})
    return pd.DataFrame(rows)


def wide(res, ylabels):
    """长表转成打印用的宽表：每个因变量一行，每个指标一列“系数星号 (t)”。"""
    if res.empty:
        return pd.DataFrame()
    cell = lambda r: f'{r.系数:.3f}{ds.stars(r.p值)} ({r.t值:.2f})'
    t = res.assign(cell=res.apply(cell, axis=1)).pivot(index='因变量', columns='指标', values='cell')
    t = t.loc[[y for y in ylabels if y in t.index], [x for x in MEASURES if x in t.columns]]
    first = res.groupby('因变量').first()
    t.insert(0, '样本数', first.样本数.reindex(t.index).astype(int))
    t.index = [ylabels[y] for y in t.index]
    return t.rename(columns=MEASURES)


def tally(res):
    """显著个数：5% 水平下显著的回归有几个，随机情况下大约是总数的 5%。"""
    n, k = len(res), int((res.p值 < 0.05).sum())
    return f'5% 水平显著 {k}/{n} 个（随机情况下约 {n * 0.05:.0f} 个）'


def main(out_dir, paras_path, daily_path, fin_path, reports_path=None, deep_chars=6000, car_window=CAR_WINDOW,
         drift_window=DRIFT_WINDOW, sort_groups=SORT_GROUPS, weight='equal', nw_lags=None):
    out = OutDir(out_dir).divergence
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option('display.width', 300)
    pd.set_option('display.max_columns', 30)
    pd.set_option('display.unicode.east_asian_width', True)
    recs = ds.archive_reports(out_dir, paras_path, reports_path)
    rep = report_table(recs, paras_path, deep_chars)
    ds.log(f'\n原文图（去掉补全的逻辑边）：{len(rep)} 篇研报、{rep.stock_code.nunique()} 家公司；'
           f'完整图里连到估值的概念平均 {rep.reach_full.mean():.0%}，只用原文图 {rep.reach.mean():.0%}；'
           f'平均每篇补全边 {rep.n_logic.mean():.1f} 条、原文边 {rep.n_text.mean():.1f} 条')
    name = lambda v: MEASURES.get(v, v)
    desc = rep.groupby('kind')[list(MEASURES)].mean().T.rename(index=name, columns=ds.KINDS)
    desc['全部'] = rep[list(MEASURES)].mean().rename(index=name)
    ds.log(f'\n【各类研报的指标均值】（篇数 {rep.kind.value_counts().rename(ds.KINDS).to_dict()}）\n{desc.round(2).to_string()}')
    corr = rep[list(MEASURES) + ['ln_chars']].rank().corr().rename(index=name, columns=name).rename(
        index={'ln_chars': 'ln 正文字数'}, columns={'ln_chars': 'ln 正文字数'})
    ds.log(f'\n【指标之间的秩相关（研报层面）】\n{corr.round(2).to_string()}')

    # 1. 公司-季度
    cq = company_quarter_text(rep)
    panel = ds.build_panel(cq, daily_path)
    s = ds.stock_chars(daily_path, panel, rep[['stock_code', 'quarter', 'date']], car_window)
    f = ds.fin_chars(fin_path, panel)
    d = panel.merge(s.drop(columns='next_ret'), on=['code', 'quarter'], how='left').merge(f, on=['code', 'quarter'], how='left')
    d['next_ret'] = d.next_ret * 100
    d.to_parquet(out / 'text_company_quarter.parquet')
    ret_vars = dict(ds.RET_VARS, car=f'CAR[{car_window[0]},+{car_window[1]}]（%）')
    ds.log(f'\n公司-季度样本：{len(d)} 个公司-季度、{d.code.nunique()} 家公司、{d.quarter.nunique()} 个季度（一篇研报也算）。'
           f'指标已标准化，系数是指标高 1 个标准差时因变量的变化；括号里是按公司聚类的 t 值；*** 1%、** 5%、* 10%；'
           f'因变量和控制变量 1%/99% 缩尾；都有季度固定效应。')
    results = {}
    for group, ys, file in (('股票收益', ret_vars, 'text_reg_returns.csv'), ('财务特征', ds.FIN_VARS, 'text_reg_financial.csv')):
        res = []
        for model, ctrl in (('只有季度固定效应', []), ('加控制变量', CONTROLS[group])):
            r = regress(d, list(ys), list(MEASURES), ctrl).assign(模型=model)
            res.append(r)
            ds.log(f'\n【{group}：{model}' + (f'（{"、".join(CONTROL_NAMES[c] for c in ctrl)}）' if ctrl else '')
                   + f'】{tally(r)}\n{wide(r, ys).to_string()}')
        res = pd.concat(res)
        res.assign(因变量=res.因变量.map(ys), 指标=res.指标.map(name)).to_csv(
            out / file, index=False, encoding='utf-8-sig', float_format='%.6f')
        results[group] = res

    rets = ds.daily_returns(daily_path, set(d.code), set(d.quarter))
    summary, detail, (days, lags) = ds.single_sorts(d.assign(next_ret=d.next_ret / 100), rets,
                                                   {v: name(v) for v in MEASURES}, sort_groups, weight, nw_lags)
    summary.to_csv(out / 'text_sort.csv', encoding='utf-8-sig', float_format='%.4f')
    detail.to_csv(out / 'text_sort_detail.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
    ds.log(f'\n【每个季度内按各指标分 {sort_groups} 组，组合持有下一个季度的日均回报（基点）】\n'
           f'{ds.method_note(weight, nw_lags)}（{days} 个交易日、滞后 {lags} 阶）\n{summary.round(2).to_string(na_rep="")}')

    # 2. 研报
    windows = {'car': car_window, 'drift': drift_window}
    rep = report_cars(daily_path, rep, windows)
    rep = rep.merge(f[['code', 'quarter', 'size', 'bm']], on=['code', 'quarter'], how='left')
    rep['is_deep'], rep['is_earn'] = rep.kind.eq('deep').astype(float), rep.kind.eq('earn').astype(float)
    rep.to_parquet(out / 'text_report.parquet')
    ylab = {'car': f'CAR[{car_window[0]},+{car_window[1]}]（%）', 'drift': f'漂移[+{drift_window[0]},+{drift_window[1]}]（%）'}
    res = []
    for model, ctrl in (('只有季度固定效应', []), ('加控制变量', ['ln_chars', 'is_deep', 'is_earn', 'size', 'bm'])):
        r = regress(rep, list(ylab), list(MEASURES), ctrl).assign(模型=model)
        res.append(r)
        ds.log(f'\n【研报层面：{model}' + (f'（{"、".join(CONTROL_NAMES[c] for c in ctrl)}）' if ctrl else '')
               + f'】{len(rep)} 篇研报；{tally(r)}\n{wide(r, ylab).to_string()}')
    res = pd.concat(res)
    res.assign(因变量=res.因变量.map(ylab), 指标=res.指标.map(name)).to_csv(
        out / 'text_reg_report.csv', index=False, encoding='utf-8-sig', float_format='%.6f')
    results['研报'] = res
    ds.log(f'\n结果在 {out}：text_report.parquet（每篇研报的指标和窗口收益）、text_company_quarter.parquet（公司-季度）、'
           f'text_reg_returns.csv、text_reg_financial.csv、text_reg_report.csv（回归长表）、text_sort.csv、text_sort_detail.csv（分组）')
    return rep, d, results
