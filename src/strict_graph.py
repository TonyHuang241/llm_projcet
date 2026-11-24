"""严格原文逻辑图：只保留原文里写出的概念和因果关系（不调用大模型，从抽取存档重建，可以反复运行）。

和逻辑图（data/graphs/、存档里的最终图）的区别：
- 边只用抽取第 1 步从原文抽出的关系（Report.rels），每条都带一句原文引文。不加预置的会计关系，不加补全的边
  （logic、bridged 都不要），不加悬挂补全（Report.extra：缺的一端是大模型推断的），只写了一端的悬挂关系也不要。
- 节点是这些关系的两端，另加孤立节点：抽取变量表里有（原文提到的概念）、但没有出现在任何一条原文因果关系里的概念
  （isolated = True，入度出度都是 0）。不做包含关系缩减（“消费贷增速”不并入“贷款增速”），不预置会计科目和公司估值：
  会计科目、估值只有原文关系里写到时才出现（变量表里没有会计科目，所以孤立节点都是概念）。层级方向不一致的关系
  （如“分红 → 配置价值”）照样保留，所以图里可能有环。
- 连不到估值就是连不到，不补。

节点的 ID 用抽取时的变量名（同一篇里同一个概念在不同句子里用同一个名字，关系才连得起来），另给一个原文字句作标签：
    exact   变量名在正文里原样出现
    alias   变量名没有原样出现，但它的某个别名原样出现（变量表的别名；会计科目和估值用 fin_items 的别名），取最长的
    clause  都没有出现：从这个节点所在关系的原文引文里，按标点切成小句，取和变量名共同汉字最多的一句（至少 2 个字）
    name    以上都找不到，标签就用变量名

节点类型：
    valuation          本公司的估值：fin_items 认作“公司估值”或“估值倍数”的名称（估值、目标价、评级、股价、市值、PE、PB ……，
                       抽取时已经统一成这两个名字）。“连到估值”只认这一类
    accounting         会计科目：fin_items 的会计科目（按名称和别名匹配）
    valuation_related  名称里有估值、目标价、评级、投资价值、配置价值、上涨空间、上行空间、市盈率、市净率，但不是本公司估值本身的
                       （可比公司估值、历史估值中枢、板块估值、债券估值收益、配置价值等），不算连到估值，单列出来供取舍
    concept            其余，层级用抽取变量表里的层（3 宏观行业 / 4 公司经营 / 5 …，见 src/logic/base.py）

输出到 <输出目录>/data/strict_graph/：
    strict_nodes.parquet    每篇研报每个节点一行：report_id、node、label、label_type、node_type、layer、kind（指标/事件/状态）、
                            definition、in_deg、out_deg、n_evidence、isolated
    strict_edges.parquet    每篇研报每条边一行：report_id、source、target、sign（+1 正向 / -1 负向）、n（同一对概念被抽到几次）、
                            n_conflict（方向相反的次数）、para_ids、evidence（原文引文，多条用 ‖ 分隔）
    strict_reports.parquet  每篇研报一行：研报信息、研报类型（deep/earn/other）、正文字数，以及有连边的节点数 n_nodes、
                            孤立节点数 n_isolated、边、估值节点个数、能走到估值的概念占比、是否有环、弱连通分量数（不含孤立节点）等

运行：python -m src.strict_graph [--output 输出目录]
"""
import argparse
import re
import sys
import unicodedata
from collections import defaultdict

import networkx as nx
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import divergence_sort as ds
from src import fin_items
from src.config import DATA_DIR, OUTPUT_DIR, OutDir

PARAS = DATA_DIR / 'hs300_2024_paragraphs.parquet'
DEEP_CHARS = 6000
VALUATION_RE = re.compile(r'估值|目标价|评级|投资价值|配置价值|上涨空间|上行空间|市盈率|市净率')
CLAUSE_RE = re.compile(r'[，,。；;：:、！？!?\s]+|\.{2,}|…+')


def norm(s):
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(s or ''))).casefold()


def node_type(name):
    m = fin_items.match(name)
    if m in (fin_items.ROOT, fin_items.MULTIPLE):
        return 'valuation'
    if m:
        return 'accounting'
    return 'valuation_related' if VALUATION_RE.search(name) else 'concept'


def text_label(name, aliases, evidence, text_n):
    """节点的原文字句：(标签, 类型)，规则见文件开头。text_n 是规范化后的正文。"""
    if norm(name) and norm(name) in text_n:
        return name, 'exact'
    hits = [a for a in aliases if len(norm(a)) >= 2 and norm(a) in text_n]
    if hits:
        return max(hits, key=lambda a: len(norm(a))), 'alias'
    chars = set(norm(name))
    best, score = None, 1
    for ev in evidence:
        for c in CLAUSE_RE.split(unicodedata.normalize('NFKC', ev)):
            s = len(chars & set(norm(c)))
            if s > score or (s == score and best is not None and len(c) < len(best)):
                best, score = c, s
    return (best, 'clause') if best else (name, 'name')


def aliases_of(name, table):
    al = list((table.get(name) or {}).get('aliases') or [])
    m = fin_items.match(name)
    if m:
        al += [m, *fin_items.ITEMS[m][0].split()]
    return al


def build(r, text):
    """一篇研报 -> (节点行, 边行, 研报统计)。r 是存档里的 Report。"""
    rid = r.rid
    agg = defaultdict(lambda: {'pos': 0, 'neg': 0, 'evidence': [], 'para_ids': set()})
    n_dangling = 0
    for x in r.rels:
        u, v = (x.get('cause') or '').strip(), (x.get('effect') or '').strip()
        if not u or not v:
            n_dangling += 1
            continue
        if u == v:
            continue
        a = agg[(u, v)]
        a['pos' if x['direction'] == '正向' else 'neg'] += 1
        ev = x['evidence'] if isinstance(x['evidence'], str) else '‖'.join(x['evidence'])
        if ev and ev not in a['evidence']:
            a['evidence'].append(ev)
        a['para_ids'].add(int(x['para_id']))
    g = nx.DiGraph()
    edges = []
    for (u, v), a in agg.items():
        sign = 1 if a['pos'] >= a['neg'] else -1
        g.add_edge(u, v)
        edges.append({'report_id': rid, 'source': u, 'target': v, 'sign': sign, 'n': a['pos'] + a['neg'],
                      'n_conflict': min(a['pos'], a['neg']), 'para_ids': sorted(a['para_ids']), 'evidence': '‖'.join(a['evidence'])})
    text_n = norm(text)
    ev_of = defaultdict(list)
    for e in edges:
        for n in (e['source'], e['target']):
            ev_of[n] += e['evidence'].split('‖')
    nodes = []
    for n in g:
        t = r.table.get(n) or {}
        typ = node_type(n)
        label, how = text_label(n, aliases_of(n, r.table), ev_of[n], text_n)
        nodes.append({'report_id': rid, 'node': n, 'label': label, 'label_type': how, 'node_type': typ,
                      'layer': {'valuation': 1, 'accounting': 2}.get(typ, t.get('layer') or 4), 'kind': t.get('kind', ''),
                      'definition': t.get('definition') or fin_items.DEFS.get(fin_items.match(n) or '', ''),
                      'in_deg': g.in_degree(n), 'out_deg': g.out_degree(n), 'n_evidence': len(set(ev_of[n]))})
    for d in nodes:
        d['isolated'] = False
    iso = sorted({n.strip() for n in r.table if n.strip() and n.strip() not in g})  # 原文抽到了、但没有出现在任何因果关系里的概念
    for n in iso:
        t = r.table.get(n) or {}
        typ = node_type(n)
        label, how = text_label(n, aliases_of(n, r.table), [], text_n)
        nodes.append({'report_id': rid, 'node': n, 'label': label, 'label_type': how, 'node_type': typ,
                      'layer': {'valuation': 1, 'accounting': 2}.get(typ, t.get('layer') or 4), 'kind': t.get('kind', ''),
                      'definition': t.get('definition') or '', 'in_deg': 0, 'out_deg': 0, 'n_evidence': 0, 'isolated': True})
    typ = {d['node']: d['node_type'] for d in nodes}
    vals = [n for n in g if typ[n] == 'valuation']
    concepts = [n for n in g if typ[n] == 'concept']
    reach = set().union(*(nx.ancestors(g, v) for v in vals)) if vals else set()
    stats = {'n_rels': len(r.rels), 'n_dangling': n_dangling, 'n_nodes': g.number_of_nodes(), 'n_isolated': len(iso),
             'n_edges': g.number_of_edges(),
             'n_concepts': len(concepts), 'n_accounting': sum(t == 'accounting' for t in typ.values()), 'n_valuation': len(vals),
             'reach_valuation': sum(n in reach for n in concepts) / len(concepts) if concepts else np.nan,
             'is_dag': nx.is_directed_acyclic_graph(g), 'n_components': nx.number_weakly_connected_components(g) if len(g) else 0}
    return nodes, edges, stats


def load_archived(out_dir, paras_path):
    """存档里的 Report（规则同 ds.archive_reports：元数据表里有、指纹对得上的）。"""
    from src.report_logic import archive_base, archive_path, archive_sig, load_reports, read_archive
    from src.report_text import reports_table_path
    out = OutDir(out_dir)
    rows = pq.read_table(reports_table_path(paras_path)).to_pylist()
    info = {r['report_id']: r for r in rows if r.get('status', 'ok') == 'ok' and archive_path(out, r['report_id']).exists()}
    base = archive_base()
    done = []
    for r in load_reports(paras_path, None, info)[0] if info else []:
        x = read_archive(out, r, archive_sig(r, base))
        if x is not None:
            done.append(x)
    return done


def main(out_dir=OUTPUT_DIR, paras_path=PARAS, deep_chars=DEEP_CHARS):
    reports = load_archived(out_dir, paras_path)
    ids = sorted(r.rid for r in reports)
    paras = pq.read_table(paras_path, columns=['report_id', 'para_id', 'text'], filters=[('report_id', 'in', ids)]).to_pandas()
    text = paras.sort_values(['report_id', 'para_id']).groupby('report_id').text.apply(' '.join)
    all_nodes, all_edges, rows = [], [], []
    for r in reports:
        t = text.get(r.rid, '')
        nodes, edges, stats = build(r, t)
        all_nodes += nodes
        all_edges += edges
        m = r.meta
        rows.append({'report_id': r.rid, 'stock_code': m.get('stock_code', ''), 'code': str(m.get('stock_code', '')).split('.')[0],
                     'company': m.get('company', ''), 'broker': m.get('broker', ''), 'date': m.get('date', ''),
                     'quarter': ds.quarter_of(m['date']) if m.get('date') else '', 'kind': ds.report_kind(t, deep_chars),
                     'n_chars': len(t), **stats})
    dst = OutDir(out_dir).data / 'strict_graph'
    dst.mkdir(parents=True, exist_ok=True)
    nodes, edges, reps = pd.DataFrame(all_nodes), pd.DataFrame(all_edges), pd.DataFrame(rows)
    nodes.to_parquet(dst / 'strict_nodes.parquet')
    edges.to_parquet(dst / 'strict_edges.parquet')
    reps.to_parquet(dst / 'strict_reports.parquet')
    ds.log(f'严格原文逻辑图：{len(reps)} 篇研报、{reps.code.nunique()} 家公司；平均每篇有连边的节点 {reps.n_nodes.mean():.1f} 个、'
           f'孤立节点 {reps.n_isolated.mean():.1f} 个、'
           f'边 {reps.n_edges.mean():.1f} 条（丢掉只写了一端的关系 {reps.n_dangling.mean():.2f} 条）；'
           f'有估值节点的 {(reps.n_valuation > 0).mean():.0%}，概念能走到估值的平均占比 {reps.reach_valuation.mean():.0%}；'
           f'有环的 {(~reps.is_dag).sum()} 篇')
    ds.log(f'节点类型 {nodes.node_type.value_counts().to_dict()}；标签来源 {nodes.label_type.value_counts(normalize=True).round(3).to_dict()}')
    ds.log(f'写到 {dst}')
    return nodes, edges, reps


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='严格原文逻辑图（从抽取存档重建）')
    ap.add_argument('--output', default=OUTPUT_DIR, help='抽取的输出目录（读存档、写结果）')
    ap.add_argument('--input', default=PARAS, help='正文段落 parquet')
    args = ap.parse_args()
    main(args.output, args.input)
