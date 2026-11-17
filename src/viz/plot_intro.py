"""生成介绍页：研报逻辑图的抽取流程、结果可信度，以及基于逻辑图的分歧度量（单个自包含 HTML，不依赖外部库）。

页面上的数字都从结果文件算出：
  report_graphs.json（report_logic 的输出）：图、边的引文、未进链概念、被删掉的关系、各篇统计；
  report_paragraphs.parquet：研报原文段落，用来和边的引文对照；
  themes.parquet（report_compare 的输出）：每家公司的概念 → 主题。
分歧度量：每个概念 1 单位流量沿出边平分流向公司估值，按（起点主题 → 终点主题或科目类别）汇总成分布；会计科目归为
“现金流”和“估值倍数”两类。两篇的 JS 散度 = 主题项 + 关系项，结构强度 = 关系项 / 共同主题分量。

用法：在 main.py 中设置 TASK = 'intro_page'
"""
import itertools
import json
import math
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pyarrow.parquet as pq

from src.fin_items import MULTIPLE, ROOT

SAMPLE_COMPANY = '海康威视'
SAMPLE_A = '20261018_国金证券_002415.SZ'  # 原文与图逐条对照：全部边都有引文，没有逻辑补全边
SAMPLE_B = '20261018_华泰证券_002415.SZ'  # 进链与未进链概念的对比：有未进链概念、并列删除、反向关系和一条逻辑补全边
# 未进链概念在原文里不是按概念名出现时，指明它对应的原文片段（生成时会核对片段确实在原文里）
MENTIONS = {SAMPLE_B: {'场景数字化业务收入': ['相关产品线收入年初以来保持较高增长']}}
PAIRS = [  # 分歧样例：(研报 a, 研报 b, 标题, 解读)
    ('20261018_国金证券_002415.SZ', '20261019_国元证券_002415.SZ', '样例一（海康威视）：逻辑相似',
     '两篇的主线相同：AI大模型技术推动场景数字化，场景数字化带来收入。国金多讲了收入结构和产品与技术，所以主题项不为 0；'
     '但在两篇都讨论的主题上，影响的去向完全一样，结构强度为 0。'),
    ('20261030_东吴证券_002594.SZ', '20261031_开源证券_002594.SZ', '样例二（比亚迪）：主线相同，中间环节不同',
     '两篇都认为海外产能和闪充技术带动销量，销量带来利润，讨论的主题高度重合，主题项很低。差别在中间环节：开源把闪充的作用直接写到销量上，'
     '东吴写的是闪充车型和新车型先改善本土需求和产品结构；开源还单独写了行业竞争压低利润。结构强度处于中等水平。'),
    ('20261018_华泰证券_002415.SZ', '20261018_国金证券_002415.SZ', '样例三（海康威视）：主题相近，推理结构不同',
     '两篇都把场景数字化放在核心位置。华泰把它写成“给予 24.3 倍 PE”的理由，连到估值倍数；国金把它写成未来收入增长的来源，'
     '连到现金流。点矩阵里“场景数字化”一行，可以在两张图上看到这条线的去向不同；结构分歧主要来自这里。'),
    ('20261031_华泰证券_600036.SH', '20261031_国信证券_600036.SH', '样例四（招商银行）：关注点不同',
     '同一天发布的两篇点评。华泰主要讲资产端收益、信贷和零售战略；国信主要讲资产质量，重点是信用卡不良。两篇只有负债成本一个共同主题，'
     '主题项很高；而负债成本在两篇里都通过净息差影响利润，去向一致，所以结构强度为 0。'),
]
CASH, MULT = '现金流', '估值倍数'
RERUN = {'node': '0.98–0.99', 'edge': '0.93'}  # 同一批研报重跑的逐篇 Jaccard 平均，见 docs/研报逻辑图抽取框架设计.md“版本记录”


# ==================== 原文定位 ====================

def squash_map(text):
    """与 logic/base.py 的 squash 相同的归一化（全角转半角，去空白和引号），同时返回每个字符在原文中的位置。"""
    chars, index = [], []
    for i, ch in enumerate(text):
        for c in unicodedata.normalize('NFKC', ch):
            if not re.match(r'[\s“”"\'‘’「」『』]', c):
                chars.append(c)
                index.append(i)
    return ''.join(chars), index


def locate(quote, text):
    """引文各段在原文中的 (起, 止) 位置；有一段找不到时返回 None。"""
    flat, index = squash_map(text)
    spans = []
    for part in re.split(r'…+|\.{3,}', squash_map(quote)[0]):
        if not part:
            continue
        k = flat.find(part)
        if k < 0:
            return None
        spans.append((index[k], index[k + len(part) - 1] + 1))
    return spans or None


def sentence_span(text, start, end):
    """把 (起, 止) 扩到所在的整句（以 。；为界）。"""
    a = max(text.rfind('。', 0, start), text.rfind('；', 0, start)) + 1
    ends = [k for k in (text.find('。', end - 1), text.find('；', end - 1)) if k >= 0]
    return a, (min(ends) + 1 if ends else len(text))


def segments(text, spans):
    """按高亮区间把一段原文切成 [文本, [键]] 片段；spans 为 [(起, 止, 键)]。"""
    cuts = sorted({0, len(text)} | {p for a, b, _ in spans for p in (a, b)})
    out = []
    for a, b in zip(cuts, cuts[1:]):
        keys = sorted({k for s, e, k in spans if s <= a and b <= e})
        out.append([text[a:b], keys])
    return out


# ==================== 图、流量与分歧 ====================

def graph_payload(rec, of):
    """一篇研报的图：节点用全局名称；每条边带引文和流量；另返回流量分布 {(起点主题, 终点主题或科目类别): 占比}。"""
    label = {n['id']: n['label'] for n in rec['nodes']}
    layer = {n['label']: n['layer'] for n in rec['nodes']}
    group = lambda n: of[n] if layer[n] >= 3 else None if n == ROOT else MULT if n == MULTIPLE else CASH
    g = nx.DiGraph()
    g.add_nodes_from(layer)
    edges = []
    for i, e in enumerate(rec['edges']):
        s, t = label[e['source']], label[e['target']]
        g.add_edge(s, t, i=i)
        edges.append({'i': i, 's': s, 't': t, 'neg': e['direction'] != '正向', 'origin': e['origin'],
                      'ev': e['evidence'], 'pids': e['para_ids'], 'ctx': e['context']})
    total, cell = {}, defaultdict(float)
    for n in nx.topological_sort(g):
        total[n] = (layer[n] >= 3) + sum(total[p] / g.out_degree(p) for p in g.predecessors(n))
        for s in g.successors(n):
            f = total[n] / g.out_degree(n)
            edges[g[n][s]['i']]['flow'] = round(f, 4)
            if layer[n] >= 3:
                cell[(group(n), group(s))] += f
    z = sum(cell.values())
    nodes = [{'id': n['label'], 'orig': n['id'] if n['id'] != n['label'] else '', 'layer': n['layer'], 'level': n['level'],
              'def': n.get('definition') or '', 'group': group(n['label'])} for n in rec['nodes']]
    return {'rid': rec['report_id'], 'company': rec['company'], 'broker': rec['broker'], 'date': rec['date'], 'title': rec['title'],
            'nodes': nodes, 'edges': edges}, {k: v / z for k, v in cell.items()}


def kl(p, q):
    return sum(v * math.log2(v / q[k]) for k, v in p.items() if v > 0)


def split(P, Q):
    """两张流量分布的 JS 散度及其拆分：总分歧 = 主题项 + 关系项；结构强度 = 关系项 / 共同主题分量。"""
    keys = set(P) | set(Q)
    M = {k: (P.get(k, 0) + Q.get(k, 0)) / 2 for k in keys}
    row = lambda X: {r: sum(v for (a, _), v in X.items() if a == r) for r in {a for a, _ in keys}}
    p, q, m = row(P), row(Q), row(M)
    total = (kl(P, M) + kl(Q, M)) / 2
    theme = (kl(p, m) + kl(q, m)) / 2
    shared = sorted(r for r in m if p[r] > 0 and q[r] > 0)
    by = {}
    for r in shared:
        cond = lambda X, x: {k: v / x[r] for k, v in X.items() if k[0] == r}
        mr = cond(M, m)
        by[r] = (p[r] * kl(cond(P, p), mr) + q[r] * kl(cond(Q, q), mr)) / 2
    mass = sum(m[r] for r in shared)
    rel = max(total - theme, 0.0)
    return {'total': total, 'theme': theme, 'rel': rel, 'shared': mass, 'strength': rel / mass if mass else None,
            'top': [r for r, v in sorted(by.items(), key=lambda x: -x[1]) if v > 0.02][:2]}


# ==================== 原文与图的对照 ====================

def text_payload(rec, graph, paras, extra=()):
    """原文段落切成片段，片段上标出它支撑的边（键 e边号）和额外的高亮（extra 为 [(键, [引文或片段], 是否扩到整句)]，每项至少要找到一个片段）。"""
    spans = defaultdict(list)

    def mark(key, quote, pids, whole=False):
        for pid in list(pids) + [p for p in paras if p not in pids]:
            found = locate(quote, paras[pid])
            if found:
                for a, b in found:
                    spans[pid].append((*(sentence_span(paras[pid], a, b) if whole else (a, b)), key))
                return True
        return False

    missing = [(e['s'], e['t'], q) for e in graph['edges'] for q in e['ev'] if not mark(f"e{e['i']}", q, e['pids'])]
    missing += [(key, quotes) for key, quotes, whole in extra if not sum(mark(key, q, [], whole) for q in quotes)]
    if missing:
        raise SystemExit(f'{rec["report_id"]}：这些引文在原文里找不到：{missing}')
    return [{'pid': pid, 'segs': segments(text, spans[pid])} for pid, text in sorted(paras.items())]


def sample_b_extra(rec):
    """样例 B：未进链概念的出处（按概念名、别名或 MENTIONS 找，扩到整句），以及流程没有采纳的关系。"""
    table = {v['name']: v for v in rec['variable_table']}
    iso, extra = [], []
    for k, name in enumerate(rec['isolated_concepts']):
        v = table[name]
        extra.append((f'c{k}', MENTIONS.get(rec['report_id'], {}).get(name) or [name] + v['aliases'], True))
        iso.append({'key': f'c{k}', 'name': name, 'layer': v['layer'], 'def': v['definition']})
    dropped = []
    for kind, items in (('并列复核删除', rec['parallel_relations']), ('反向关系，不进图', rec['reverse_relations'])):
        for x in items:
            key = f'x{len(dropped)}'
            extra += [(key, [q], False) for q in x['evidence']]
            dropped.append({'key': key, 'kind': kind, 's': x['cause'], 't': x['effect'], 'neg': x['direction'] != '正向',
                            'ev': x['evidence'], 'reason': x.get('reason') or ''})
    return iso, dropped, extra


# ==================== 主流程 ====================

def main(graphs_path, paras_path, themes_path, out_html):
    data = json.loads(Path(graphs_path).read_text(encoding='utf-8'))
    recs = {r['report_id']: r for r in data['reports']}
    theme_rows = pq.read_table(themes_path).to_pylist()
    of = defaultdict(dict)
    for t in theme_rows:
        of[t['stock_code']][t['concept']] = t['theme']
    paras = defaultdict(dict)
    for p in pq.read_table(paras_path, columns=['report_id', 'para_id', 'text']).to_pylist():
        if p['report_id'] in (SAMPLE_A, SAMPLE_B):
            paras[p['report_id']][int(p['para_id'])] = p['text']

    graphs, dist = {}, {}
    for rid, rec in recs.items():
        graphs[rid], dist[rid] = graph_payload(rec, of[rec['stock_code']])
    short = lambda rid: f"{recs[rid]['broker'].replace('证券', '').replace('公司', '')}{recs[rid]['date'][4:]}"
    all_pairs = []
    for a, b in itertools.combinations(sorted(recs, key=lambda r: (recs[r]['company'], recs[r]['date'], r)), 2):
        if recs[a]['stock_code'] == recs[b]['stock_code']:
            all_pairs.append({'company': recs[a]['company'], 'a': short(a), 'b': short(b), **split(dist[a], dist[b])})

    used = {SAMPLE_A, SAMPLE_B} | {r for a, b, *_ in PAIRS for r in (a, b)}
    themes, order = [], defaultdict(list)  # 样例公司的主题表；各公司主题的固定顺序
    for t in theme_rows:
        if t['theme'] not in order[t['company']]:
            order[t['company']].append(t['theme'])
        if t['company'] == SAMPLE_COMPANY:
            if not themes or themes[-1]['name'] != t['theme']:
                themes.append({'name': t['theme'], 'def': t['theme_definition'], 'concepts': []})
            themes[-1]['concepts'].append(t['concept'])
    # 主题配色：每家公司按固定顺序取色，只有 8 种颜色；主题更多的公司只给页面上出现的主题配色
    shown = defaultdict(set)
    for rid in used:
        shown[recs[rid]['company']] |= {n['group'] for n in graphs[rid]['nodes'] if n['layer'] >= 3}
    colors = {}
    for company, names in shown.items():
        keep = order[company] if len(order[company]) <= 8 else [t for t in order[company] if t in names]
        if len(keep) > 8:
            raise SystemExit(f'{company} 在页面上出现的主题有 {len(keep)} 个，超过 8 种颜色，请换样例')
        colors[company] = keep

    iso, dropped, extra = sample_b_extra(recs[SAMPLE_B])
    stat = lambda k: sum(r['stats'][k] for r in recs.values())
    origin = {o: sum(e['origin'] == o for r in recs.values() for e in r['edges']) for o in ('extracted', 'bridged', 'logic', 'preset')}
    quoted = sum(bool(e['evidence']) for r in recs.values() for e in r['edges'] if e['origin'] == 'extracted')
    companies = defaultdict(int)
    for r in recs.values():
        companies[r['company']] += 1
    payload = {
        'layers': data['layers'], 'model': data['model'], 'themes': themes, 'colors': colors, 'sampleCompany': SAMPLE_COMPANY, 'cash': CASH, 'mult': MULT,
        'graphs': {rid: graphs[rid] for rid in used},
        'dist': {rid: [[r, c, round(v, 4)] for (r, c), v in sorted(dist[rid].items())] for rid in used},
        'sampleA': {'rid': SAMPLE_A, 'text': text_payload(recs[SAMPLE_A], graphs[SAMPLE_A], paras[SAMPLE_A])},
        'sampleB': {'rid': SAMPLE_B, 'text': text_payload(recs[SAMPLE_B], graphs[SAMPLE_B], paras[SAMPLE_B], extra),
                    'iso': iso, 'dropped': dropped},
        'pairs': [{'a': a, 'b': b, 'title': title, 'note': note, **split(dist[a], dist[b])} for a, b, title, note in PAIRS],
        'allPairs': all_pairs,
        'stats': {'reports': len(recs), 'companies': dict(companies), 'tableVars': stat('n_table_vars'), 'relations': stat('n_relations'),
                  'parallel': stat('n_parallel'), 'reverse': stat('n_reverse'), 'isolated': stat('n_isolated'),
                  'concepts': stat('n_concepts'), 'edges': sum(origin.values()), 'origin': origin, 'quoted': quoted, 'rerun': RERUN},
    }
    html = TEMPLATE.replace('__DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    out_html = Path(out_html)
    out_html.write_text(html, encoding='utf-8')
    print(f'已写出 {out_html}（{len(html.encode("utf-8")) / 1024:.0f} KB），内联 {len(used)} 篇研报、{len(all_pairs)} 对分歧数值')


TEMPLATE = (Path(__file__).parent / 'intro_template.html').read_text(encoding='utf-8')
