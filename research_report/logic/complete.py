"""第 3 步：子图补全。主图是沿边能走到公司估值的部分，已经接入主图的概念不再补全，只给走不到估值的子图补边。
每个子图调用一次大模型：原文写明了子图中的概念影响主图概念的，补原文边（逐字引用、程序核对）；原文没写的，给子图的终点
各补一条逻辑边（逻辑补全没有原文依据，终点接上后，子图里的概念都能沿原文关系走到估值）。"""
import itertools
from collections import Counter, defaultdict

from common.llm_client import log
from research_report import fin_items
from research_report.logic.base import (CONCEPT_LAYERS, LAYER_NAMES, ROOT, Step, cue_missing, direction_of, is_null,
                                        norm_direction, norm_name, para_block, parse_json, quote_in, resolve_name, short,
                                        squash, table_index, to_int)
from research_report.logic.graph import COMPLETION_ORDER, add_edge, connected, layer_map, subgraphs

COMPLETE_MAX_PARAS = 8  # 子图补全时最多附带的段落数
COMPLETE_MAX_TARGETS = 2  # 一个子图最多补几条原文边
FIN_ORDER = {n: i for i, n in enumerate(fin_items.ITEMS)}


def candidates_text(cands, lay):
    """候选概念按层列出：从下往上（离待接入的子图越近越靠前），会计科目按清单顺序排。"""
    by = defaultdict(list)
    for n in cands:
        by[lay[n]].append(n)
    return '\n'.join(f'第 {k} 层（{LAYER_NAMES[k]}）：{"、".join(sorted(by[k], key=lambda n: (FIN_ORDER.get(n, -1), n)))}'
                     for k in sorted(by, reverse=True))


def sinks_of(g, comp):
    """子图的终点：子图里不再影响其他概念的概念（子图里的概念只指向子图内部，否则就能走到估值了）。"""
    return [n for n in comp if g.out_degree(n) == 0]


def upper_cands(g, lay, sources):
    """每个影响因素可以连的主图概念：已经连到公司估值、层号比它小的概念。原文边只从下层指向上层或同层，比影响因素更上层的
    概念不可能是它的上游，所以补的边不会成环。"""
    conn = connected(g) - {ROOT}
    return {s: frozenset(m for m in conn if lay[m] < lay[s]) for s in sources}


def mention_paras(name, table, ps):
    names = [squash(x) for x in [name, *table.get(name, {}).get('aliases', [])] if len(squash(x)) >= 2]
    return [q for q in ps if any(x in squash(q['text']) for x in names)]


def comp_edges(g, comp):
    return sorted(g.subgraph(comp).edges(data=True), key=lambda e: e[2]['order'])


def subgraph_paras(comp, g, table, ps, hints):
    """子图补全时附带的段落：子图中概念的线索所在段落、子图内部原文关系所在段落、提到子图中概念（终点在前）的段落，
    各自再带上后一段；最多 COMPLETE_MAX_PARAS 段，按段落顺序排列。"""
    pos = {p['para_id']: i for i, p in enumerate(ps)}
    sinks = sinks_of(g, comp)
    first = ([x['para_id'] for n in comp for x in hints.get(n, [])] + [pid for *_, d in comp_edges(g, comp) for pid in d['para_ids']]
             + [q['para_id'] for n in sinks + [n for n in comp if n not in sinks] for q in mention_paras(n, table, ps)])
    out = []
    for pid in first:
        if pid in pos:
            out += [q['para_id'] for q in ps[pos[pid]:pos[pid] + 2] if q['para_id'] not in out]
    return [ps[pos[pid]] for pid in sorted(out[:COMPLETE_MAX_PARAS], key=pos.get)]


def subgraph_text(comp, g, table, lay, hints, sinks=None):
    """子图的文字说明：概念（层、定义）、内部的原文关系（附一句引文）、原文只写了它没写影响对象的说法；sinks 不为 None 时列出终点。"""
    nodes = '；'.join(f'{n}（第 {lay[n]} 层；{table.get(n, {}).get("definition") or "—"}）'
                     for n in sorted(comp, key=lambda n: (lay[n], n)))
    edges = '；'.join(f'{u} → {v}（{direction_of(d["sign"])}' + (f'，“{d["evidence"][0][:60]}”' if d['evidence'] else '')
                     + '）' for u, v, d in comp_edges(g, comp))
    notes = '；'.join(f'{n}：“{x["evidence"]}”' for n in comp for x in hints.get(n, []))
    lines = [f'  概念：{nodes}', f'  原文关系：{edges}']
    if sinks is not None:
        lines.append(f'  终点：{"、".join(sinks)}')
    if notes:
        lines.append(f'  原文中只写了它、没写影响对象的说法：{notes}')
    return '\n'.join(lines)


def parse_complete(content, cands, sinks, texts, index):
    """校验子图补全结果。cands 为 {子图中的概念: 它可以连的主图概念}，sinks 为要补边的终点（有候选可连的）。
    带引文的是原文边（最多 COMPLETE_MAX_TARGETS 条，引文逐字核对），不带引文的是逻辑边（只能从终点出发，每个终点一条，
    已有原文边的终点不再补）；每个终点都要有边。返回 [{'source', 'name', 'direction', 'para_id', 'quote', 'reason', 'origin'}]，
    原文边在前。补的边都从子图指向已经连到公司估值、更上层的主图概念，不会成环。"""
    links_in = parse_json(content, dict).get('links')
    if not isinstance(links_in, list):
        raise ValueError('缺少 links 数组')
    items = [it for it in links_in if isinstance(it, dict)]
    quoted = [it for it in items if not is_null(it.get('quote'))][:COMPLETE_MAX_TARGETS]
    links, errors = [], []

    def check(it):
        """名称、候选和方向的检查，合格时返回 (影响因素, 被影响的一端, 方向)。"""
        raw_s, raw, d = norm_name(it.get('source')), norm_name(it.get('name')), norm_direction(it.get('direction'))
        src, name = resolve_name(raw_s, index), resolve_name(raw, index)
        if src not in cands:
            errors.append(f'影响因素“{raw_s}”不是子图中的概念')
        elif name not in cands[src]:
            errors.append(f'“{raw}”不在候选概念中，或不比“{src}”更上层')
        elif d is None:
            errors.append(f'“{raw_s}”→“{raw}”的 direction 必须是正向或负向')
        else:
            return src, name, d
        return None

    for it in quoted:  # 原文边
        ok = check(it)
        if not ok:
            continue
        src, name, d = ok
        pid, quote = to_int(it.get('para_id')), it.get('quote')
        if pid not in texts or not quote_in(quote, texts[pid], min_len=4):
            errors.append(f'“{src}”→“{name}”的引文在第 {it.get("para_id")} 段中找不到，请从所给段落中逐字摘录；'
                          f'原文确实没有写明的，去掉这条原文边（终点改补逻辑边）')
        elif cue_missing(name, quote):
            errors.append(f'连到“{name}”时引文中必须有{cue_missing(name, quote)}字样')
        elif (src, name) not in [(x['source'], x['name']) for x in links]:
            links.append({'source': src, 'name': name, 'direction': d, 'para_id': pid, 'quote': short(quote),
                          'reason': short(it.get('reason')), 'origin': 'bridged'})
    covered = {x['source'] for x in links}
    for it in items:  # 逻辑边
        if not is_null(it.get('quote')):
            continue
        src = resolve_name(norm_name(it.get('source')), index)
        if src in covered:  # 已有原文边的终点、已经补过的终点，多给的不要
            continue
        if src not in sinks:
            errors.append(f'“{norm_name(it.get("source"))}”不是要补边的终点，不能补逻辑边（终点只有{"、".join(sinks)}）；'
                          f'原文写明了它的影响对象的，请附上引文作为原文边')
            continue
        ok = check(it)
        if ok:
            links.append({'source': src, 'name': ok[1], 'direction': ok[2], 'para_id': None, 'quote': '',
                          'reason': short(it.get('reason')), 'origin': 'logic'})
            covered.add(src)
    missing = [s for s in sinks if s not in covered]
    if missing and not errors:
        errors.append(f'这些终点还没有补边：{"、".join(missing)}')
    if errors:
        raise ValueError('；'.join(errors))
    return links


def complete_subgraphs(ctx, reports):
    """给走不到公司估值的子图补边，原地给 r.g 加边；已经能沿原文关系走到估值的概念不再补全。
    按子图最上面一层是第 3、4、5 层的顺序处理，先接入的子图可以作为后面子图的连接对象。每个子图调用一次：原文边最多
    COMPLETE_MAX_TARGETS 条；没有原文边的终点各补一条逻辑边，连到主图中更上层的概念，这是让子图全部走到估值所需的最少边数。
    r.hints 为原文只写了它、没写影响对象的记录，用来挑段落和说明子图。"""
    step = Step(ctx, 'complete_cache.jsonl', '补全')
    seq = itertools.count(COMPLETION_ORDER)  # 补出的边排在原文边之后

    def work(task):
        r, comp, cands, sinks = task
        lay = layer_map(r.g, r.table)
        paras = subgraph_paras(comp, r.g, r.table, r.paras, r.hints)
        texts = {p['para_id']: p['text'] for p in paras}
        sub = subgraph_text(comp, r.g, r.table, lay, r.hints, sinks=sinks)
        ctext, block = candidates_text(set().union(*cands.values()), lay), para_block(paras) or '（无）'
        index = table_index(r.table)
        try:
            return step.ask(step.key(r.rid, sub, ctext, block), lambda c: parse_complete(c, cands, sinks, texts, index),
                            extra={'report_id': r.rid, 'subgraph': list(comp)}, **ctx.about(r.rid), subgraph=sub,
                            candidates=ctext, paragraphs=block)
        except RuntimeError as e:
            log(f'!!! {r.rid} 子图（{"、".join(comp)}）补全失败，保持走不到估值：{e}')
            return None

    for layer in CONCEPT_LAYERS:
        tasks, stuck, n_comps, n_nodes = [], [], 0, 0  # tasks: (研报, 子图, {子图中的概念: 可以连的主图概念}, 要补边的终点)
        for r in reports:
            lay = layer_map(r.g, r.table)
            for comp in subgraphs(r.g):
                if min(lay[n] for n in comp) != layer:
                    continue
                n_comps, n_nodes = n_comps + 1, n_nodes + len(comp)
                cands = upper_cands(r.g, lay, comp)
                sinks = sinks_of(r.g, comp)
                stuck += [(r.rid, s) for s in sinks if not cands[s]]  # 主图里没有比它更上层的节点（原文没提到可连的科目）
                if any(cands.values()):
                    tasks.append((r, comp, cands, [s for s in sinks if cands[s]]))
        if stuck:
            log(f'第 {layer} 层有 {len(stuck)} 个终点在主图中没有更上层的节点可连，保持走不到估值：'
                f'{"、".join(f"{rid[:20]}…{s}" for rid, s in stuck[:10])}')
        results = ctx.parallel(tasks, work, f'第 {layer} 层子图补全')
        n = Counter()
        for origin in ('bridged', 'logic'):  # 同一层里原文边先加、逻辑边后加
            for (r, *_), res in zip(tasks, results):
                for x in res or []:
                    if x['origin'] != origin:
                        continue
                    if origin == 'bridged':
                        add_edge(r.g, x['source'], x['name'], x['direction'], origin, [x['quote']], [x['para_id']],
                                 ['子图补全（原文）' + (f'：{x["reason"]}' if x['reason'] else '')], next(seq))
                    else:
                        add_edge(r.g, x['source'], x['name'], x['direction'], origin, [], [], [x['reason']] if x['reason'] else [],
                                 next(seq))
                    n[origin] += 1
        log(f'第 {layer} 层子图补全：{n_comps} 个走不到估值的子图（{n_nodes} 个概念），调用 {len(tasks)} 次；'
            f'原文边 {n["bridged"]} 条，逻辑边 {n["logic"]} 条（共 {sum(len(x[3]) for x in tasks)} 个终点要补）')
