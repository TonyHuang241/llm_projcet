"""深度报告概念图结构页面：一个概念 DAG 输出目录（全样本深度报告是 output_dag/deep）里的全部研报，按公司内统一后的名称
（unify_concept_dag.py 写出的 unified_edges.parquet；没有统一结果的研报用原始边）算每篇的源汇比、唯一有向路径数、
平均路径长度，画三个指标的分布图（每篇一个值；另一排是每家公司各篇的平均值）和明细表，点一篇画出它的结构图（只有文中的概念和连边，没有补全）。
写到 <输出目录>/深度报告结构.html（模板 deep_structure_template.html，数据内嵌，双击打开）。
由 unify_concept_dag.py --task deep 调用；指标的算法同 plot_dag_structure.py（build_graph、measure）。

每篇研报的结构图数据：n 概念名列表（统一后的名称），k 各概念的类型（0 指标、1 事件、2 状态），
m 边 [原因序号, 结果序号, 引文数]（去环之后），al {序号: [合并到这个名称的原名...]}（只列名称有变化的）。
"""
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from src.viz.plot_dag_structure import measure

TEMPLATE = (Path(__file__).parent / 'deep_structure_template.html').read_text(encoding='utf-8')
KIND_CODE = {'指标': 0, '事件': 1, '状态': 2}


def main(out_dir, companies=None):
    out_dir = Path(out_dir)
    data_dir = out_dir / 'data'
    if not (data_dir / 'reports.parquet').exists():
        raise SystemExit(f'{data_dir} 下没有 reports.parquet：先运行 python extract_concept_dag.py')
    read = lambda name: pq.read_table(data_dir / f'{name}.parquet').to_pylist() if (data_dir / f'{name}.parquet').exists() else []  # noqa: E731
    reports = [r for r in read('reports') if not companies or r['stock_code'] in companies]
    if not reports:
        raise SystemExit('输出目录里没有这些股票的研报')
    keep = {r['report_id'] for r in reports}
    raw, unified = defaultdict(list), defaultdict(list)
    for e in read('edges'):
        if e['in_dag'] and e['report_id'] in keep:
            raw[e['report_id']].append((e['cause'], e['effect'], e['support']))
    for e in read('unified_edges'):
        if e['report_id'] in keep:
            unified[e['report_id']].append((e['cause'], e['effect'], e['support']))
    kind, merged = {}, defaultdict(lambda: defaultdict(set))  # (研报, 原名) -> 类型；研报 -> 统一名 -> {原名}
    for c in read('concepts'):
        if c['report_id'] in keep:
            kind[(c['report_id'], c['name'])] = c['kind']
    for c in read('concept_unify'):
        if c['report_id'] in keep and c['in_dag']:
            kind.setdefault((c['report_id'], c['unified']), kind.get((c['report_id'], c['name'])))
            if c['changed']:
                merged[c['report_id']][c['unified']].add(c['name'])

    rows = []
    for r in sorted(reports, key=lambda r: (r['stock_code'], r['date'], r['broker'])):
        rid = r['report_id']
        is_unified = rid in unified
        g, after = measure(unified[rid] if is_unified else raw[rid])
        _, before = measure(raw[rid])
        names = sorted(g.nodes, key=lambda v: (-g.out_degree(v), g.in_degree(v), v))
        idx = {v: i for i, v in enumerate(names)}
        graph = {'n': names, 'k': [KIND_CODE.get(kind.get((rid, v)), 0) for v in names],
                 'm': [[idx[a], idx[b], g.edges[a, b]['support']] for a, b in g.edges],
                 'al': {str(idx[u]): sorted(ns) for u, ns in merged[rid].items() if u in idx}}
        drop = ('lengths',)
        rows.append({'rid': rid, 'company': r['company'], 'code': r['stock_code'], 'date': r['date'], 'broker': r['broker'],
                     'chars': r['n_chars'], 'chunks': r['n_chunks'], 'unified': is_unified,
                     'a': {k: v for k, v in after.items() if k not in drop}, 'lengths': after['lengths'],
                     'b': {k: before[k] for k in ('n_nodes', 'n_edges', 'ratio', 'n_paths', 'len_mean')}, 'g': graph})

    unify_js = data_dir / 'unify.js'
    unify = {}
    if unify_js.exists():
        s = unify_js.read_text(encoding='utf-8')
        unify = json.loads(s[s.index('(') + 1:s.rindex(')')])
    st = unify.get('stats', {})
    data = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'unify_generated': unify.get('generated'),
            'unify': {k: st.get(k) for k in ('n_terms', 'n_terms_after', 'n_groups', 'n_multi_companies')},
            'reports': rows}
    page = out_dir / '深度报告结构.html'
    page.write_text(TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')),
                    encoding='utf-8')
    a = [x['a'] for x in rows]
    pick = lambda k: [v[k] for v in a if v[k] is not None]  # noqa: E731
    print(f'{len(rows)} 篇研报、{len({x["code"] for x in rows})} 家公司（{sum(x["unified"] for x in rows)} 篇按统一后的名称）：'
          f'源汇比中位数 {statistics.median(pick("ratio")):.2f}，唯一有向路径数中位数 {statistics.median(pick("n_paths")):g}，'
          f'平均路径长度中位数 {statistics.median(pick("len_mean")):.2f}；页面：{page}（{page.stat().st_size / 1024:.0f} KB）')
    return page
