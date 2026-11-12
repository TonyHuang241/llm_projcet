"""第 5、6 步：全局去重（跨研报统一概念名称）和整理输出（JSON、parquet、各阶段的图）。"""
import itertools
import json
from collections import Counter, defaultdict

import networkx as nx
import numpy as np

from common.llm_client import log
from research_report import fin_items
from research_report.logic.base import (LAYER_NAMES, ORIGIN_PRIORITY, ROOT, TEXT_ORIGINS, Step, as_bool, direction_of,
                                        name_key, parse_decisions, short, write_parquet)
from research_report.logic.extract import rel_row
from research_report.logic.graph import break_cycles, build_graph, connected, contain_items, layer_map

EMBED_MODEL = 'BAAI/bge-base-zh-v1.5'  # 中文句向量模型（BERT 结构），用 [CLS] 向量
GLOBAL_SIM = 0.80  # 全局去重：“名称：定义”的 BGE 余弦达到该值算候选
GLOBAL_TOPK = 5  # 每个名称最多取的近邻数
GLOBAL_BATCH = 12  # 每次交给大模型的候选对数
_embedder = {}


# ==================== 第 5 步：全局去重 ====================

def embed(texts, model_name):
    """用 BERT 类句向量模型编码，返回 L2 归一化后的矩阵。"""
    import torch  # 依赖 torch，只在用到时导入
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_name not in _embedder:
        log(f'加载句向量模型 {model_name}')
        _embedder[model_name] = (AutoTokenizer.from_pretrained(model_name),
                                 AutoModel.from_pretrained(model_name).to(device).eval())
    tok, model = _embedder[model_name]
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), 256):
            batch = tok(texts[i:i + 256], padding=True, truncation=True, max_length=96, return_tensors='pt').to(device)
            out.append(torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0], dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def global_dedup(ctx, reports, embed_model):
    """统一不同研报对同一种概念的叫法（只比较同一层的概念）。返回 ({名称: 全局名称}, 全局名称表)。
    字面相同的名称本来就统一；同一篇研报里的两个不同名称（填表时判定为不同概念）不会被并到一起；预置节点不参与。"""
    info = {}
    for r in reports:
        for n in r.g:
            t = r.table.get(n)
            if not t:  # 公司估值和会计层
                continue
            e = info.setdefault(n, {'reports': set(), 'definition': t['definition'], 'layer': t['layer']})
            e['reports'].add(r.rid)
    names = sorted(info)
    mapping = {n: n for n in names}
    if len(names) < 2:
        return mapping, []
    vec = embed([f'{n}：{info[n]["definition"]}' for n in names], embed_model)
    sim = vec @ vec.T
    pairs = {}
    for i in range(len(names)):
        for j in [j for j in np.argsort(-sim[i]) if j != i][:GLOBAL_TOPK]:
            if sim[i, j] >= GLOBAL_SIM:
                pairs[min(i, j), max(i, j)] = float(sim[i, j])
    keys = [name_key(n) for n in names]
    for i, j in itertools.combinations(range(len(names)), 2):
        if min(len(keys[i]), len(keys[j])) >= 2 and (keys[i] in keys[j] or keys[j] in keys[i]):
            pairs[i, j] = float(sim[i, j])
    pairs = sorted(((names[i], names[j], s) for (i, j), s in pairs.items()
                    if info[names[i]]['layer'] == info[names[j]]['layer']
                    and not info[names[i]]['reports'] & info[names[j]]['reports']), key=lambda x: (-x[2], x[0], x[1]))

    step = Step(ctx, 'global_cache.jsonl', '归一', fields=('same',))  # 按候选对缓存，一次调用判断一批
    pkey = lambda a, b: step.key(*sorted(f'{n}|{info[n]["layer"]}|{info[n]["definition"]}' for n in (a, b)))
    todo = [(a, b) for a, b, _ in pairs if not step.has(pkey(a, b))]
    batches = [todo[i:i + GLOBAL_BATCH] for i in range(0, len(todo), GLOBAL_BATCH)]
    log(f'全局去重：{len(names)} 个不同名称，候选对 {len(pairs)} 对，已缓存 {len(pairs) - len(todo)} 对，本次调用 {len(batches)} 次')
    line = lambda n: f'{n} ｜ 第{info[n]["layer"]}层 ｜ {info[n]["definition"] or "—"}'

    def work(batch):
        text = '\n'.join(f'【第{i}对】\n  A：{line(a)}\n  B：{line(b)}' for i, (a, b) in enumerate(batch, 1))
        res = step.call(lambda c: parse_decisions(c, len(batch), lambda it: (as_bool(it.get('same')), short(it.get('reason'))),
                                                  lambda lack: f'缺少这些编号的判断：{"、".join(lack)}'), pairs=text)
        for (a, b), (same, reason) in zip(batch, res):
            step.put(pkey(a, b), same, a=a, b=b, reason=reason)

    ctx.parallel(batches, work, '全局去重')

    parent, reps, blocked = {n: n for n in names}, {n: set(info[n]['reports']) for n in names}, 0

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for a, b, _ in pairs:
        if not step.get(pkey(a, b), False):
            continue
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        if reps[ra] & reps[rb]:  # 合并后会让同一篇研报的两个概念变成同一个名称，不合并
            blocked += 1
            continue
        parent[rb] = ra
        reps[ra] |= reps[rb]
    groups = defaultdict(list)
    for n in names:
        groups[find(n)].append(n)
    table = []
    for members in groups.values():
        canon = min(members, key=lambda n: (-len(info[n]['reports']), len(n), n))
        mapping.update({n: canon for n in members})
        if len(members) > 1:
            table.append({'global': canon, 'layer': info[canon]['layer'],
                          'n_reports': len(set().union(*(info[n]['reports'] for n in members))),
                          'members': [{'name': n, 'n_reports': len(info[n]['reports']), 'definition': info[n]['definition']}
                                      for n in sorted(members, key=lambda n: (-len(info[n]['reports']), n))]})
    table.sort(key=lambda x: (-x['n_reports'], x['global']))
    log(f'全局去重完成：合并成 {len(table)} 组，名称 {len(names)} -> {len(set(mapping.values()))} 个'
        f'（因同篇不合并而跳过 {blocked} 对）')
    return mapping, table


# ==================== 第 6 步：整理输出 ====================

def annotate(g, lay):
    """画图用的行号：按层从上到下排（公司估值在最上），层内有原文关系的，被影响的一方排在上面；
    净影响：沿通往“公司估值”的各条路径相乘得到的影响方向，到不了“公司估值”的节点为空。"""
    sub, effect = {}, {}
    for n in reversed(list(nx.topological_sort(g))):
        same = [s for s in g.successors(n) if lay[s] == lay[n]]
        sub[n] = 1 + max(sub[s] for s in same) if same else 0
        effect[n] = {1} if n == ROOT else {g.edges[n, s]['sign'] * e for s in g.successors(n) for e in effect[s]}
    offset, row = {}, 0
    for k in sorted(set(lay.values())):
        offset[k] = row
        row += 1 + max(sub[n] for n in g if lay[n] == k)
    label = lambda e: '+' if e == {1} else '-' if e == {-1} else '±' if e else ''
    return {n: offset[lay[n]] + sub[n] for n in g}, {n: label(e) for n, e in effect.items()}


def edge_rows(g, lay, level):
    return [{'source': u, 'target': v, 'direction': direction_of(d['sign']), 'origin': d['origin'],
             'gap': lay[u] - lay[v], 'weight': d['weight'], 'n_pos': d['n_pos'], 'n_neg': d['n_neg'],
             'evidence': d['evidence'], 'para_ids': d['para_ids'], 'context': d.get('context', []),
             'conflict_path': d.get('conflict_path', []), 'cut': d.get('cut', False)}
            for u, v, d in sorted(g.edges(data=True), key=lambda e: (level[e[0]], e[0], e[1]))]


def node_base(n, lay, level, table, mapping):
    """节点在最终图和阶段图里共有的字段。"""
    t = table.get(n, {})
    return {'id': n, 'label': mapping.get(n, n), 'layer': lay[n], 'layer_name': LAYER_NAMES[lay[n]], 'level': level[n],
            'is_root': n == ROOT, 'is_preset': n in fin_items.ITEMS, 'kind': t.get('kind', '')}


def up(d, tops):
    """{下位概念: [记录]} -> {并入的最上层概念: [记录]}。"""
    return {k: sum((v for x, v in d.items() if tops.get(x, x) == k), []) for k in {tops.get(x, x) for x in d}}


def graph_record(r, mapping):
    """一篇研报的最终图（包含关系缩减、补全、剪枝之后）及其统计。"""
    g, table, info = r.g, r.table, r.info
    lay = layer_map(g, table)
    level, effect = annotate(g, lay)
    conn = connected(g)
    mentions = Counter()
    for u, v, d in g.edges(data=True):
        if d['origin'] in TEXT_ORIGINS:
            mentions[u] += d['weight']
            mentions[v] += d['weight']
    unresolved, attached = up(r.unresolved, r.tops), up(r.attached, r.tops)
    nodes = []
    for n in sorted(g, key=lambda n: (level[n], n)):
        t = table.get(n, {})
        nodes.append({**node_base(n, lay, level, table, mapping), 'effect': effect[n], 'connected': n in conn,
                      'mentions': mentions[n], 'definition': t.get('definition') or fin_items.DEFS.get(n, ''),
                      'aliases': t.get('aliases', []), 'merged': sorted(r.merged.get(n, [])),
                      'unresolved': unresolved.get(n, []), 'attached': attached.get(n, []), 'hints': r.hints.get(n, []),
                      'cut_parent': table[n]['parent'] if n in r.cut else ''})
    edges = edge_rows(g, lay, level)
    concepts = [n for n in g if lay[n] >= 3]
    by_origin = Counter(e['origin'] for e in edges)
    reverse, accounting = [rel_row(x) for x in r.groups['reverse']], [rel_row(x) for x in r.acc_unmatched]
    stats = {'n_table_vars': len(table), 'n_relations': info['n_relations'], 'n_dangling': info['n_dangling'],
             'n_dangling_resolved': info['n_dangling_resolved'], 'n_rejected': info['n_rejected'],
             'n_reverse': len(reverse), 'n_accounting_text': info['n_accounting_text'],
             'n_nodes': len(nodes), 'n_edges': len(edges), 'n_concepts': len(concepts),
             'n_by_layer': {k: sum(lay[n] == k for n in g) for k in LAYER_NAMES},
             **{f'n_{o}': by_origin[o] for o in ORIGIN_PRIORITY},
             'n_logic_skip': sum(e['origin'] == 'logic' and e['gap'] >= 2 for e in edges),
             'reach_before_completion': info['reach_before'], 'reach': sum(n in conn for n in concepts) / max(len(concepts), 1),
             'n_unconnected': sum(n not in conn for n in concepts), 'n_pruned': len(r.pruned),
             'n_isolated': len(r.isolated), 'n_merged': sum(map(len, r.merged.values())),
             'n_parallel': len(r.parallel), 'n_subgraphs': info['n_subgraphs'],
             'n_cut': sum(n in r.cut for n in concepts),
             'n_replaced': len(r.replaced),
             'n_conflict': sum(bool(e['conflict_path']) for e in edges),
             'n_components': nx.number_weakly_connected_components(g),
             'self_loops': info['self_loops'], 'conflicts': info['conflicts'], 'removed_cycle_edges': len(r.removed),
             'depth': max(level.values()) if level else 0}
    return {**r.meta, 'stats': stats, 'nodes': nodes, 'edges': edges, 'replaced_edges': r.replaced,
            'removed_cycle_edges': r.removed, 'reverse_relations': reverse, 'accounting_relations': accounting,
            'parallel_relations': r.parallel, 'pruned_accounting': r.pruned, 'isolated_concepts': r.isolated,
            'variable_table': [{'name': n, **t} for n, t in table.items()]}


def stage_record(name, g, table, mapping, tops, ref=None, extra=None):
    """一个阶段的图。ref 为同一时刻包含关系缩减后的图：一个概念是否连到估值、净影响按它所在的包含关系树（最上层的概念）
    在 ref 中的情况标注（缩减前，上位概念自己常常没有出边，经下位概念的边连到估值）；ref 为 None 时（第 1 步，还没有边）不标。"""
    g = g.copy()
    break_cycles(g)
    lay = layer_map(g, table)
    level, _ = annotate(g, lay)
    conn, eff = (connected(ref), annotate(ref, layer_map(ref, table))[1]) if ref is not None else (set(), {})
    nodes = [{**node_base(n, lay, level, table, mapping),
              'effect': eff.get(tops.get(n, n), '') if ref is not None else '',
              'connected': tops.get(n, n) in conn if ref is not None else None,
              'definition': table.get(n, {}).get('definition') or fin_items.DEFS.get(n, ''),
              'aliases': table.get(n, {}).get('aliases', [])}
             for n in sorted(g, key=lambda n: (level[n], n))]
    edges = edge_rows(g, lay, level)
    concepts = [x for x in nodes if x['layer'] >= 3]
    stats = {'n_concepts': len(concepts), 'n_by_layer': {k: sum(x['layer'] == k for x in nodes) for k in LAYER_NAMES},
             'by_origin': Counter(e['origin'] for e in edges),
             'reach': sum(bool(x['connected']) for x in concepts) / max(len(concepts), 1), **(extra or {})}
    return {'name': name, 'nodes': nodes, 'edges': edges, 'stats': stats}


def stages(r, mapping):
    """包含关系缩减之前的三个阶段（第 4 阶段即最终图）：第 1 步的概念表和包含关系、第 2 步的原文边、第 3 步补全后。
    第 2、3 阶段的节点相同（会计层只有原文中出现过的科目），第 3 阶段只是多了补全的边。"""
    tb = r.table
    g1 = build_graph(contain_items(tb, tb))[0]
    g1.add_nodes_from(tb)
    g2, g3 = r.g0.copy(), r.g0.copy()
    g3.add_edges_from(r.added)
    return [stage_record('概念分层', g1, tb, mapping, r.tops, extra={'n_part_of': sum(bool(t['parent']) for t in tb.values())}),
            stage_record('原文连边', g2, tb, mapping, r.tops, r.ref_text, extra={'n_isolated': len(r.isolated)}),
            stage_record('补全', g3, tb, mapping, r.tops, r.ref_done,
                         extra={'n_subgraphs': r.info['n_subgraphs'], 'added': Counter(d['origin'] for *_, d in r.added)})]


def write_outputs(reports, mapping, global_table, out_dir, model, embed_model):
    """写出 report_graphs.json、edges.parquet、nodes.parquet，返回 (各篇的记录, 全局节点表, 边表)。"""
    records, edge_table = [], []
    for r in reports:
        rec = graph_record(r, mapping)
        rec['stages'] = stages(r, mapping)
        records.append(rec)
        node = {n['id']: n for n in rec['nodes']}
        edge_table += [{'report_id': r.rid, 'source': e['source'], 'target': e['target'], 'source_global': node[e['source']]['label'],
                       'target_global': node[e['target']]['label'], 'source_layer': node[e['source']]['layer'],
                       'target_layer': node[e['target']]['layer'], 'direction': e['direction'], 'origin': e['origin'],
                       'weight': e['weight'], 'evidence': ' | '.join(e['evidence']), 'para_ids': ','.join(map(str, e['para_ids']))}
                      for e in rec['edges']]
    in_reports, layer = defaultdict(set), {}
    for rec in records:
        for n in rec['nodes']:
            in_reports[n['label']].add(rec['report_id'])
            layer.setdefault(n['label'], n['layer'])
    node_rows = [{'node': c, 'layer': layer[c], 'n_reports': len(in_reports[c]),
                  'members': '、'.join(sorted({n for n, gl in mapping.items() if gl == c and n != c}))}
                 for c in sorted(in_reports, key=lambda c: (layer[c], -len(in_reports[c]), c))]
    (out_dir / 'report_graphs.json').write_text(
        json.dumps({'root': ROOT, 'layers': LAYER_NAMES, 'model': model, 'embed_model': embed_model, 'reports': records,
                    'global_names': global_table}, ensure_ascii=False, indent=1), encoding='utf-8')
    write_parquet(edge_table, out_dir / 'edges.parquet',
                  ['report_id', 'source', 'target', 'source_global', 'target_global', 'source_layer', 'target_layer',
                   'direction', 'origin', 'weight', 'evidence', 'para_ids'])
    write_parquet(node_rows, out_dir / 'nodes.parquet', ['node', 'layer', 'n_reports', 'members'])
    return records, node_rows, edge_table
