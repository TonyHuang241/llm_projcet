"""汇点频次的纯描述统计（2024 年沪深300 深度研报样本，口径同 regress_dag_forecast.py）。

口径：
  样本      output_dag/regression/report_panel.parquet 的 report_id（791 篇 / 241 家），公司 = code，券商 = broker。
  图        src.dag_forecast.load_graphs（edges.parquet 里 in_dag 为真的原始边，节点为原始概念名、未做跨研报统一；
            report_id 在两个 DAG 目录都有时用 output_dag/deep/data）。
  汇点      src.dag_forecast.strict_sinks：图里入度 > 0、出度 = 0 的节点（孤立点不算）。
  任意角色  图里的节点（源点/中间/汇点/图内孤立点）加上同一 DAG 目录 concepts.parquet 里该篇不在图里的概念（孤立概念）。
  祖先数    networkx.ancestors（能到达该节点的节点数，不含自己），同 sink_measures 的 R_N_ANC。
输出到 output_dag/sink_stats/：sink_frequency.csv、sink_top15.csv、summary.csv、report_sink_counts.csv。
"""
from collections import defaultdict
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from src.dag_forecast import load_graphs, strict_sinks

ROOT = Path(__file__).resolve().parent
PANEL = ROOT / 'output_dag' / 'regression' / 'report_panel.parquet'
DAG_DIRS = [ROOT / 'output_dag' / 'deep' / 'data', ROOT / 'output_dag' / 'data']  # 同 regress_dag_forecast.DAG_DIRS
OUT = ROOT / 'output_dag' / 'sink_stats'
SINKS_ALL = ROOT / 'output_dag' / 'forecast' / 'final_indicators' / 'sinks_all.tsv'


def owner_concepts(report_ids):
    """每篇研报所属 DAG 目录（规则同 load_graphs：前面的目录优先）里 concepts.parquet 的概念名集合。"""
    want, owner = set(report_ids), {}
    for i, d in enumerate(DAG_DIRS):
        for rid in pd.read_parquet(d / 'reports.parquet', columns=['report_id']).report_id:
            if rid in want and rid not in owner:
                owner[rid] = i
    names = defaultdict(set)
    for i, d in enumerate(DAG_DIRS):
        c = pd.read_parquet(d / 'concepts.parquet', columns=['report_id', 'name'])
        for rid, n in zip(c.report_id, c.name):
            if owner.get(rid) == i:
                names[rid].add(n)
    return names


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(PANEL, columns=['report_id', 'code', 'broker'])
    assert panel.report_id.is_unique
    rids = list(panel.report_id)
    firm, broker = dict(zip(panel.report_id, panel.code)), dict(zip(panel.report_id, panel.broker))
    graphs = load_graphs(rids, DAG_DIRS)
    concepts = owner_concepts(rids)

    occ = []  # 每条“汇点-研报”记录
    any_rows = []  # 每条“概念-研报”记录（任意角色）
    per_report = []
    for rid in rids:
        g = graphs[rid]
        sinks = strict_sinks(g)
        per_report.append({'report_id': rid, 'n_nodes_graph': g.number_of_nodes(), 'n_sinks': len(sinks),
                           'n_sinks_unique': len(set(sinks))})
        for v in sinks:
            occ.append({'report_id': rid, 'name': v, 'indeg': g.in_degree(v), 'n_anc': len(nx.ancestors(g, v)),
                        'firm': firm[rid], 'broker': broker[rid]})
        graph_nodes = set(g.nodes)
        for v in graph_nodes:
            any_rows.append((rid, v, True))
        for v in concepts.get(rid, set()) - graph_nodes:
            any_rows.append((rid, v, False))
    occ = pd.DataFrame(occ)
    per_report = pd.DataFrame(per_report)
    anyd = pd.DataFrame(any_rows, columns=['report_id', 'name', 'in_graph'])

    n_rep = len(rids)
    f = occ.groupby('name').agg(n_reports_sink=('report_id', 'nunique'), n_occ_sink=('report_id', 'size'),
                                n_firms_sink=('firm', 'nunique'), n_brokers_sink=('broker', 'nunique'),
                                mean_indeg=('indeg', 'mean'), mean_n_anc=('n_anc', 'mean')).reset_index()
    f['share_reports'] = f.n_reports_sink / n_rep
    na = anyd.groupby('name').report_id.nunique().rename('n_reports_any')
    ng = anyd[anyd.in_graph].groupby('name').report_id.nunique().rename('n_reports_in_graph')
    f = f.merge(na, on='name', how='left').merge(ng, on='name', how='left')
    f['sink_rate'] = f.n_reports_sink / f.n_reports_any
    f = f.sort_values(['n_reports_sink', 'n_occ_sink', 'name'], ascending=[False, False, True]).reset_index(drop=True)
    f.insert(0, 'rank', np.arange(1, len(f) + 1))
    cols = ['rank', 'name', 'n_reports_sink', 'share_reports', 'n_occ_sink', 'n_firms_sink', 'n_brokers_sink',
            'n_reports_any', 'n_reports_in_graph', 'sink_rate', 'mean_indeg', 'mean_n_anc']
    f = f[cols]
    f.to_csv(OUT / 'sink_frequency.csv', index=False, encoding='utf-8-sig')
    f.head(15).to_csv(OUT / 'sink_top15.csv', index=False, encoding='utf-8-sig')
    per_report.to_csv(OUT / 'report_sink_counts.csv', index=False, encoding='utf-8-sig')

    # ---------------- 整体概况
    total = len(occ)
    ns = per_report.n_sinks
    s = [('研报总数', n_rep), ('公司数', panel.code.nunique()), ('券商数', panel.broker.nunique()),
         ('有至少一个汇点的研报数', int((ns > 0).sum())), ('汇点“节点-研报”总条数', total),
         ('不同汇点名个数', len(f)), ('每篇汇点数_均值', ns.mean()), ('每篇汇点数_标准差', ns.std())]
    for q in (0, 0.1, 0.25, 0.5, 0.75, 0.9, 1):
        s.append((f'每篇汇点数_p{int(q * 100)}', float(ns.quantile(q))))
    s += [('只在1篇研报作汇点的名字数', int((f.n_reports_sink == 1).sum())),
          ('只在1篇研报作汇点的名字占比', float((f.n_reports_sink == 1).mean())),
          ('至少5篇研报作汇点的名字数', int((f.n_reports_sink >= 5).sum())),
          ('至少10篇研报作汇点的名字数', int((f.n_reports_sink >= 10).sum()))]
    for k in (15, 50, 100):
        s.append((f'前{k}名累计节点-研报条数', int(f.n_occ_sink.head(k).sum())))
        s.append((f'前{k}名累计节点-研报条数占比', float(f.n_occ_sink.head(k).sum() / total)))
    s += [('n_occ_sink与n_reports_sink恒等', bool((f.n_occ_sink == f.n_reports_sink).all())),
          ('同篇重复汇点名的研报数', int((per_report.n_sinks != per_report.n_sinks_unique).sum()))]
    summary = pd.DataFrame(s, columns=['item', 'value'])
    summary.to_csv(OUT / 'summary.csv', index=False, encoding='utf-8-sig')

    # ---------------- 自检
    chk = {}
    chk['sum n_reports_sink == 总条数'] = int(f.n_reports_sink.sum()) == total
    chk['n_reports_any >= n_reports_sink'] = bool((f.n_reports_any >= f.n_reports_sink).all())
    chk['n_reports_in_graph >= n_reports_sink'] = bool((f.n_reports_in_graph >= f.n_reports_sink).all())
    chk['mean_indeg >= 1'] = bool((f.mean_indeg >= 1).all())
    chk['mean_n_anc >= mean_indeg'] = bool((f.mean_n_anc >= f.mean_indeg - 1e-12).all())
    chk['graphs 是 DAG'] = all(nx.is_directed_acyclic_graph(g) for g in graphs.values())
    if SINKS_ALL.exists():
        ref = pd.read_csv(SINKS_ALL, sep='\t', dtype={'name': str}, keep_default_na=False)
        m = f[['name', 'n_reports_sink']].merge(ref[['name', 'n_reports']], on='name', how='outer', indicator=True)
        chk['sinks_all 名字数'] = len(ref)
        chk['本次名字数'] = len(f)
        chk['仅本次有'] = int((m._merge == 'left_only').sum())
        chk['仅 sinks_all 有'] = int((m._merge == 'right_only').sum())
        both = m[m._merge == 'both']
        chk['共有名字中篇数不等'] = int((both.n_reports_sink != both.n_reports).sum())
        m.to_csv(OUT / '_check_vs_sinks_all.csv', index=False, encoding='utf-8-sig')
    for k, v in chk.items():
        print(f'[check] {k}: {v}')
    print(summary.to_string(index=False))
    pd.set_option('display.width', 250)
    print(f.head(40).to_string(index=False))


if __name__ == '__main__':
    main()
