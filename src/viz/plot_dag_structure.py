"""概念 DAG 结构统计页面：按跨研报统一后的名称（src/concept_unify.py 写出的 unified_edges.parquet），算每篇研报的
源汇比和唯一有向路径数，画成分布图，写到 <输出目录>/概念DAG结构.html（模板 dag_structure_template.html，数据内嵌，双击打开）。
由 unify_concept_dag.py --task structure 调用。

各指标（每篇研报一张图，只算进图的边，没有边的孤立概念不算）：
  源点        没有入边的概念（只作原因）；汇点：没有出边的概念（只作结果）
  源汇比      源点数 / 汇点数：大于 1 表示很多原因收敛到少数结论，小于 1 表示少数原因发散出很多结果
  唯一有向路径数   从源点到汇点的不同有向路径条数（经过的概念序列不同就算不同的路径），即这篇研报里完整的因果链条数；
              用动态规划计数（每个点的路径数 = 各直接结果的路径数之和），不必逐条展开
  路径长度    一条路径的边数；给出每篇的平均值（每条路径权重相同）、中位数和最大值
  共有路径    这篇的路径（统一名称后的概念序列）在同一公司的其他研报里也完整出现过的条数；路径太多时不逐条展开，记为空
对照“统一前”：同样的指标在原始 DAG（edges.parquet，各篇自己的概念名）上算一遍。
统一后同一篇里两个概念可能合成一个，图里可能出现环；这时按引文数从少到多删边去环（同 concept_dag.build_dag），删掉的条数记下。
"""
import json
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import networkx as nx
import pyarrow.parquet as pq

TEMPLATE = (Path(__file__).parent / 'dag_structure_template.html').read_text(encoding='utf-8')
MAX_ENUM_PATHS = 200_000  # 路径条数超过这个数的研报不逐条展开（不算共有路径）


def build_graph(edges):
    """[(原因, 结果, 引文数)] -> 无环有向图，返回 (图, 为去环删掉的边数)。"""
    g = nx.DiGraph()
    for a, b, s in edges:
        if a != b:
            w = g.edges[a, b]['support'] + s if g.has_edge(a, b) else s
            g.add_edge(a, b, support=w)
    removed = 0
    while not nx.is_directed_acyclic_graph(g):  # 先用快的判断（无环时 find_cycle 要遍历全图，慢约 10 倍），有环时同原来逐个删
        cycle = nx.find_cycle(g)
        a, b = min(((u, v) for u, v, *_ in cycle), key=lambda e: (g.edges[e]['support'], e))
        g.remove_edge(a, b)
        removed += 1
    return g, removed


def path_stats(g):
    """源点、汇点、源到汇的路径条数和长度分布（{边数: 条数}）。"""
    sources = [v for v in g if g.in_degree(v) == 0]
    sinks = [v for v in g if g.out_degree(v) == 0]
    dist = {}  # 点 -> Counter{从它出发到汇点的路径边数: 条数}
    for v in reversed(list(nx.topological_sort(g))):
        succ = list(g.successors(v))
        if not succ:
            dist[v] = Counter({0: 1})
        else:
            c = Counter()
            for s in succ:
                for k, n in dist[s].items():
                    c[k + 1] += n
            dist[v] = c
    lengths = Counter()
    for v in sources:
        lengths.update(dist[v])
    return sources, sinks, lengths


def enum_paths(g, limit):
    """逐条列出源到汇的路径（概念序列的元组）；超过 limit 条时返回 None。"""
    out, stack = [], [(v, (v,)) for v in g if g.in_degree(v) == 0]
    while stack:
        v, path = stack.pop()
        succ = list(g.successors(v))
        if not succ:
            out.append(path)
            if len(out) > limit:
                return None
        stack.extend((s, path + (s,)) for s in succ)
    return out


def median_len(lengths):
    total, acc = sum(lengths.values()), 0
    for k in sorted(lengths):
        acc += lengths[k]
        if acc * 2 >= total:
            return k
    return 0


def measure(edges):
    g, removed = build_graph(edges)
    sources, sinks, lengths = path_stats(g)
    n_paths = sum(lengths.values())
    return g, {'n_nodes': g.number_of_nodes(), 'n_edges': g.number_of_edges(), 'n_sources': len(sources),
               'n_sinks': len(sinks), 'ratio': round(len(sources) / len(sinks), 4) if sinks else None,
               'n_paths': n_paths, 'len_mean': round(sum(k * n for k, n in lengths.items()) / n_paths, 4) if n_paths else None,
               'len_median': median_len(lengths), 'len_max': max(lengths, default=0),
               'lengths': {str(k): v for k, v in sorted(lengths.items())}, 'n_cycle_removed': removed}


def main(out_dir, companies=None):
    out_dir = Path(out_dir)
    data_dir = out_dir / 'data'
    need = [data_dir / f for f in ('reports.parquet', 'edges.parquet', 'unified_edges.parquet', 'unify.js')]
    lack = [p.name for p in need if not p.exists()]
    if lack:
        raise SystemExit(f'{data_dir} 下缺少 {"、".join(lack)}：先运行 python extract_concept_dag.py 和 python unify_concept_dag.py')
    unify_js = (data_dir / 'unify.js').read_text(encoding='utf-8')
    unify = json.loads(unify_js[unify_js.index('(') + 1:unify_js.rindex(')')])
    reports = pq.read_table(data_dir / 'reports.parquet').to_pylist()
    unified = pq.read_table(data_dir / 'unified_edges.parquet').to_pylist()
    raw = [e for e in pq.read_table(data_dir / 'edges.parquet').to_pylist() if e['in_dag']]
    in_unify = {e['report_id'] for e in unified}
    keep = {r['report_id'] for r in reports if r['report_id'] in in_unify and (not companies or r['stock_code'] in companies)}
    if not keep:
        raise SystemExit('统一结果里没有这些股票的研报：先用 unify_concept_dag.py 统一它们')
    by_after, by_before = defaultdict(list), defaultdict(list)
    for e in unified:
        if e['report_id'] in keep:
            by_after[e['report_id']].append((e['cause'], e['effect'], e['support']))
    for e in raw:
        if e['report_id'] in keep:
            by_before[e['report_id']].append((e['cause'], e['effect'], e['support']))

    rows, paths = [], {}
    rep = {r['report_id']: r for r in reports}
    for rid in sorted(keep, key=lambda x: (rep[x]['stock_code'], rep[x]['date'], rep[x]['broker'])):
        r = rep[rid]
        g, after = measure(by_after[rid])
        _, before = measure(by_before[rid])
        listed = enum_paths(g, MAX_ENUM_PATHS)
        paths[rid] = None if listed is None else set(listed)
        rows.append({'rid': rid, 'company': r['company'], 'stock_code': r['stock_code'], 'date': r['date'],
                     'broker': r['broker'], 'n_chars': r['n_chars'], 'after': after, 'before': before})
    for x in rows:  # 共有路径：在同一公司其他研报的路径集合里也出现
        mine = paths[x['rid']]
        if mine is None:
            x['after']['n_shared'] = None
            continue
        others = set().union(*(p for rid, p in paths.items()
                               if rid != x['rid'] and p is not None and rep[rid]['stock_code'] == x['stock_code']))
        x['after']['n_shared'] = sum(p in others for p in mine)

    s = unify['stats']
    data = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'unify_generated': unify.get('generated'),
            'model': unify.get('model'), 'companies': sorted({x['company'] for x in rows}), 'reports': rows,
            'unify': {k: s.get(k) for k in ('n_terms', 'n_terms_after', 'n_groups', 'n_merged_terms')}}
    page = out_dir / '概念DAG结构.html'
    page.write_text(TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')),
                    encoding='utf-8')
    ratios = [x['after']['ratio'] for x in rows if x['after']['ratio'] is not None]
    counts = [x['after']['n_paths'] for x in rows]
    means = [x['after']['len_mean'] for x in rows if x['after']['len_mean'] is not None]
    print(f'{len(rows)} 篇研报：源汇比 {min(ratios):.2f}-{max(ratios):.2f}（中位数 {statistics.median(ratios):.2f}），'
          f'唯一有向路径数 {min(counts)}-{max(counts)}（中位数 {statistics.median(counts):g}），'
          f'平均路径长度 {min(means):.2f}-{max(means):.2f}（中位数 {statistics.median(means):.2f}）；页面：{page}')
    return page
