"""建图与收尾：包含关系缩减、按层归类、建图（含删除孤立概念、只保留原文提到的会计科目）、去环、剪枝、删除没有概念连入的会计科目。"""
from collections import defaultdict, deque

import networkx as nx
import numpy as np

from src.llm_client import log
from src import fin_items
from src.logic.base import DIRECTIONS, ORIGIN_PRIORITY, ROOT, direction_of, layer_of
from src.logic.extract import classify, note_of

CYCLE_RANK = {'logic': 0, 'bridged': 2, 'extracted': 3, 'part_of': 8, 'preset': 9}  # 去环时先删排在前面的
COMPLETION_ORDER = 10 ** 9  # 子图补全加的边，order 从这里开始编号（排在原文边之后，也用来认出补全边）


# ==================== 图的基本操作 ====================

def preset_items():
    return [{'cause': u, 'effect': v, 'direction': direction_of(s), 'origin': 'preset', 'evidence': [], 'para_ids': [],
             'context': []} for u, v, s in fin_items.PRESET_EDGES]


def build_graph(items):
    """一篇研报的各类关系 -> DAG。items 的 evidence、para_ids 为列表。返回 (图, 诊断计数, 为去环删掉的边)。"""
    agg, info = {}, defaultdict(int)
    for order, t in enumerate(items):
        u, v = t['cause'], t['effect']
        if u == v:
            info['self_loops'] += 1
            continue
        e = agg.setdefault((u, v), {'pos': 0, 'neg': 0, 'evidence': [], 'para_ids': [], 'context': [], 'origins': set(),
                                    'order': order})
        e['pos' if t['direction'] == '正向' else 'neg'] += 1
        e['origins'].add(t['origin'])
        for field in ('evidence', 'para_ids', 'context'):
            e[field] += [x for x in t[field] if x not in e[field]]
    g = nx.DiGraph()
    for (u, v), e in agg.items():
        info['conflicts'] += bool(e['pos'] and e['neg'])
        g.add_edge(u, v, sign=1 if e['pos'] >= e['neg'] else -1, origin=next(o for o in ORIGIN_PRIORITY if o in e['origins']),
                   weight=e['pos'] + e['neg'], n_pos=e['pos'], n_neg=e['neg'], evidence=e['evidence'],
                   para_ids=sorted(e['para_ids']), context=e['context'], order=e['order'])
    return g, info, break_cycles(g)


def add_edge(g, u, v, direction, origin, evidence, pids, context, order):
    """补全时加一条边（各类计数按一条原文支持记）。"""
    sign = DIRECTIONS[direction]
    g.add_edge(u, v, sign=sign, origin=origin, weight=1, n_pos=int(sign > 0), n_neg=int(sign < 0), evidence=evidence,
               para_ids=pids, context=context, order=order)


def break_cycles(g):
    """有环时反复删掉环上的一条边：先删逻辑补全边，再按支持次数从少到多（一样少时删后加入的），返回删掉的边。"""
    removed = []
    while True:
        try:
            cycle = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            return removed
        u, v = min(((a, b) for a, b in cycle),
                   key=lambda ab: (CYCLE_RANK[g.edges[ab]['origin']], g.edges[ab]['weight'], -g.edges[ab]['order']))
        removed.append({'source': u, 'target': v, 'direction': direction_of(g.edges[u, v]['sign']),
                        'origin': g.edges[u, v]['origin'], 'weight': g.edges[u, v]['weight']})
        g.remove_edge(u, v)


def attach_accounting(g, rels):
    """两端都是会计科目的原文关系：和预置关系同向时作为该关系的原文证据，返回对不上的那些。"""
    unmatched = []
    for r in rels:
        u, v = r['cause'], r['effect']
        if g.has_edge(u, v) and g.edges[u, v]['sign'] == DIRECTIONS[r['direction']]:
            d = g.edges[u, v]
            d['evidence'] += [x for x in r['evidence'] if x not in d['evidence']]
            d['para_ids'] = sorted(set(d['para_ids']) | set(r['para_ids']))
            d['n_text'] = d.get('n_text', 0) + 1
        else:
            unmatched.append(r)
    return unmatched


def layer_map(g, table):
    return {n: layer_of(n, table) or 4 for n in g}


def connected(g):
    """能走到公司估值的节点（含公司估值本身）。"""
    return nx.ancestors(g, ROOT) | {ROOT} if ROOT in g else set()


def subgraphs(g):
    """走不到公司估值的概念按弱连通分组，每个子图为按名称排序的元组。只看和主图是否弱连通不够：子图常和主图共用
    上游因素（如某项关税既影响主图里的毛利率，又影响子图里的海外销量），但子图本身没有一条边通往估值。"""
    conn = connected(g)
    return sorted(tuple(sorted(c)) for c in nx.weakly_connected_components(g.subgraph(n for n in g if n not in conn)))


def text_accounting(g, records):
    """第 2 步结束时保留的会计层节点：本篇原文抽取记录（含只写了一端的悬挂记录）中出现过的会计科目和估值倍数，
    以及从它们沿预置关系通往公司估值要经过的科目。g 为含全部预置科目的图。"""
    seen = {x for r in records for x in (r['cause'], r['effect']) if x in fin_items.ITEMS}
    return seen | {m for n in seen if n in g for m in nx.descendants(g, n) if m in fin_items.ITEMS}


def drop_accounting(g, keep):
    """删掉 keep 以外的会计科目：第 2 步之后节点就确定了，第 3 步补全只在已有的节点之间连边，不引入原文没有出现的科目。"""
    g.remove_nodes_from([n for n in list(g) if n in fin_items.ITEMS and n not in keep])


# ==================== 包含关系缩减 ====================

def cut_parents(table, rels):
    """包含关系缩减的规则：一个概念向上只连出一条包含关系时，并入上位概念；除了包含关系还有别的出边（进图的原文边）时，
    去掉它的包含关系、保留原文边，它自己留在图中。rels 为本篇进图的原文关系（缩减前的名称），返回去掉包含关系的概念。
    按第 2 步结束时的原文边判断：补全在缩减后的图上做，补全的边不影响这里的判断。"""
    out = {r['cause'] for r in rels if r['effect']}
    return {n for n, t in table.items() if t.get('parent') and n in out}


def top_parents(table, cut=frozenset()):
    """包含关系树：每个下位概念 -> 它并入的概念（沿上位概念往上走，走到最上层或走到去掉了包含关系的概念为止；不并入的不列）。"""
    tops = {}
    for n in table:
        cur = n
        while cur not in cut and table.get(cur, {}).get('parent'):
            cur = table[cur]['parent']
        if cur != n:
            tops[n] = cur
    return tops


def merge_up(r, tops):
    """把关系两端的下位概念换成最上层的概念（包含关系缩减），换过的在 context 里记下原来的说法。"""
    c, e = tops.get(r['cause'], r['cause']), tops.get(r['effect'], r['effect'])
    if (c, e) == (r['cause'], r['effect']):
        return r
    return {**r, 'cause': c, 'effect': e, 'context': [*r.get('context', []), f'合并前：{r["cause"]} → {r["effect"]}']}


def contain_items(table, names):
    """names 中各概念到上位概念的包含关系边（下位概念 → 上位概念，正向）。"""
    return [{'cause': n, 'effect': table[n]['parent'], 'direction': '正向', 'origin': 'part_of', 'evidence': [], 'para_ids': [],
             'context': [f'{table[n].get("parent_relation") or "包含"}：{table[n].get("parent_reason") or "—"}']}
            for n in sorted(names) if table[n].get('parent')]


def unmerged_graph(text_rels, acc_rels, table, cut):
    """包含关系缩减之前的图：预置的会计层 + 合并前的原文边 + 包含关系边。只保留自己或并入它的下位概念有原文边的概念，
    以及它们要并入的上位概念，其余的是孤立概念（缩减后同样被删掉）。cut 中的概念缩减时去掉包含关系：两端都在图中、
    且不会成环时也画出来，边上标 cut。"""
    keep = set()
    for n in {x for r in text_rels for x in (r['cause'], r['effect']) if (layer_of(x, table) or 0) >= 3}:
        while n and n not in keep:
            keep.add(n)
            n = None if n in cut else table.get(n, {}).get('parent')
    g = build_graph(preset_items() + text_rels + contain_items(table, keep - cut))[0]
    for n in sorted(keep & cut):
        p = table[n]['parent']
        if p in g and not g.has_edge(n, p) and not nx.has_path(g, p, n):
            g.add_edge(n, p, sign=1, origin='part_of', weight=1, n_pos=1, n_neg=0, evidence=[], para_ids=[],
                       context=contain_items(table, [n])[0]['context'], order=0, cut=True)
    attach_accounting(g, acc_rels)
    return g


def merge_parents(reports):
    """包含关系缩减和按层归类。每棵包含关系树只保留最上层的概念，下位概念的边都改连到它上面，树内部的关系去掉；
    向上只有包含关系的概念并入上位概念，除了包含关系还有别的出边的，去掉包含关系、保留原文边。
    补全在缩减后的图上做（同一棵树里只要有一个概念连到估值，整棵树都算连上），所以缩减在这里就算好，输出里另画缩减之前各阶段的图。
    之后按两端的层级归类：进图、会计科目之间、反向；第 3-5 层概念直接指向公司估值的转为线索（r.hints），供补全参考。"""
    n_inner = 0
    for r in reports:
        r.cut = cut_parents(r.table, r.kept)
        r.tops = top_parents(r.table, r.cut)
        r.groups, r.hints = {'edge': [], 'accounting': [], 'reverse': []}, defaultdict(list)
        for x in r.kept + [x for x in r.complete + r.extra if classify(x, r.table) != 'edge']:
            x = merge_up(x, r.tops)
            if x['cause'] == x['effect']:
                n_inner += 1
                continue
            kind = classify(x, r.table)
            if kind == 'to_root':  # 原文直接指向公司估值：作为第 3 步的线索
                r.hints[x['cause']].append({**note_of(x), 'note': '原文直接指向公司估值'})
            else:
                r.groups[kind].append(x)
        for x in r.rels:
            if x['cause'] and x['effect'] is None and (layer_of(x['cause'], r.table) or 0) >= 3:
                r.hints[r.tops.get(x['cause'], x['cause'])].append(note_of(x))
    log(f'包含关系缩减：{sum(len(r.tops) for r in reports)} 个下位概念并入上位概念，树内部的关系 {n_inner} 条随之去掉；'
        f'{sum(len(r.cut) for r in reports)} 个下位概念另有出边，去掉包含关系、保留原文边')
    log(f'按层归类：进图 {sum(len(r.groups["edge"]) for r in reports)} 条，会计科目之间 {sum(len(r.groups["accounting"]) for r in reports)} 条，'
        f'反向 {sum(len(r.groups["reverse"]) for r in reports)} 条，直接指向公司估值转为线索 '
        f'{sum(x.get("note") == "原文直接指向公司估值" for r in reports for v in r.hints.values() for x in v)} 条')


# ==================== 建图 ====================

def build_graphs(reports):
    """预置的会计层 + 缩减后的原文边。没有任何边的概念是孤立的单个节点，不做补全、直接删掉；只有原文线索而没有边的概念
    也算孤立，线索只供子图里的概念补全时参考。第 2 步结束时节点就确定了：会计层只留原文中出现过的科目（及其通往估值的预置路径），
    补全不能再引入别的科目。r.g 为缩减后的图（之后补全、剪枝都在它上面做），r.g0 为缩减前的图（画阶段图用）。"""
    reach = []
    for r in reports:
        g, info, r.removed = build_graph(preset_items() + r.groups['edge'])
        r.isolated = sorted(n for n in r.table if r.tops.get(n, n) not in g)  # 并入了孤立节点的下位概念也一并删掉
        r.merged = defaultdict(list)
        for child, top in r.tops.items():
            if top in g:
                r.merged[top].append(child)
        r.acc_unmatched = attach_accounting(g, r.groups['accounting'])
        keep_acc = text_accounting(g, r.rels + r.extra)
        drop_accounting(g, keep_acc)
        concepts = [n for n in g if (layer_of(n, r.table) or 0) >= 3]
        conn = connected(g)
        r.g0 = unmerged_graph(r.kept, r.groups['accounting'], r.table, r.cut)
        drop_accounting(r.g0, keep_acc)
        info.update(n_accounting=len(keep_acc), n_subgraphs=len(subgraphs(g)), n_relations=len(r.rels), n_rejected=r.rejected,
                    n_dangling=sum(not (x['cause'] and x['effect']) for x in r.rels), n_dangling_resolved=len(r.extra),
                    n_accounting_text=len(r.groups['accounting']) - len(r.acc_unmatched),
                    reach_before=sum(n in conn for n in concepts) / max(len(concepts), 1))
        r.g, r.info, r.ref_text = g, info, g.copy()  # ref_text：补全之前缩减后的图（阶段图用）
        reach.append((len(concepts), sum(n in conn for n in concepts)))
    n_concepts = sum(c for c, _ in reach)
    log(f'建图：删掉孤立概念 {sum(len(r.isolated) for r in reports)} 个，剩余概念 {n_concepts} 个，'
        f'保留原文中出现过的会计科目平均每篇 {np.mean([r.info["n_accounting"] for r in reports]):.1f} 个，'
        f'补全前能连到公司估值的 {sum(n for _, n in reach) / max(n_concepts, 1):.0%}，'
        f'走不到估值的子图 {sum(r.info["n_subgraphs"] for r in reports)} 个')


# ==================== 第 4 步：剪枝 ====================

def other_path(g, u, v, sign=None, first_ok=None):
    """DAG g 中不经过直连 u→v、从 u 到 v 的一条更长路径（广度优先，取最短的那条），返回节点列表或 None；
    sign 不为 None 时只找各段方向相乘等于 sign 的路径，first_ok 不为 None 时第一步只走 first_ok(节点) 为真的节点。
    按 (节点, 累计方向) 搜索，DAG 中找到的一定是简单路径。"""
    start = (u, 1)
    parent, queue = {start: None}, deque([start])
    while queue:
        n, s = queue.popleft()
        for m in g.successors(n):
            if n == u and (m == v or first_ok and not first_ok(m)):
                continue
            st = (m, s * g.edges[n, m]['sign'] if sign is not None else 1)
            if st in parent:
                continue
            parent[st] = (n, s)
            if m == v and (sign is None or st[1] == sign):
                path, cur = [], st
                while cur:
                    path.append(cur[0])
                    cur = parent[cur]
                return path[::-1]
            if m != v:
                queue.append(st)
    return None


def prune_shortcuts(g, lay):
    """剪枝：任意两个节点之间既有直连、又有方向一致的更长链路、且这条链路第一步就走到更上层时，删掉直连，
    其余链路都保留（不论各段是哪类边）。lay 为 {节点: 层号}。
    更长的链路第一步走的是同层节点时（如 海外销量 → 出口销量占比 → 营业收入），直连保留：下层的概念直接接入上层，
    同层的链路只是补充。更长的链路只有方向相反的，直连也保留，并在边上记下 conflict_path（可能是某条边的方向抽错了，
    也可能是两个不同的渠道，删掉直连会让净影响翻转）。
    删掉这样的直连不改变任何两点之间的可达性和可能的方向，所以逐条删即可。返回被删的直连，各附一条代替它的链路。"""
    removed = []
    for u, v in sorted(g.edges, key=lambda e: g.edges[e]['order']):
        d = g.edges[u, v]
        path = other_path(g, u, v, d['sign'], first_ok=lambda m: lay[m] < lay[u])
        if path:
            removed.append({'source': u, 'target': v, 'direction': direction_of(d['sign']), 'origin': d['origin'],
                            'evidence': d['evidence'], 'para_ids': d['para_ids'], 'context': d.get('context', []), 'path': path})
            g.remove_edge(u, v)
    for u, v, d in g.edges(data=True):  # 剩下的直连：没有方向一致的更长链路、只有方向相反的，才算冲突
        d['conflict_path'] = [] if other_path(g, u, v, d['sign']) else other_path(g, u, v) or []
    return removed


def prune_accounting(g):
    """反复删掉没有任何入边的会计科目（含估值倍数），直到不再变化；留下的科目都有下层概念直接或经由其他科目连入。
    返回被删的科目。"""
    removed = []
    while True:
        drop = [n for n in g if n != ROOT and n in fin_items.ITEMS and g.in_degree(n) == 0]
        if not drop:
            return removed
        g.remove_nodes_from(drop)
        removed += drop


def finish_graphs(reports):
    """补全之后：再查一次环（先删逻辑补全边），记下补全加的边；剪掉有方向一致的更长链路的直连，删掉没有下层概念连入的会计科目。"""
    for r in reports:
        r.removed += break_cycles(r.g)
        r.added = [(u, v, dict(d)) for u, v, d in r.g.edges(data=True) if d['order'] >= COMPLETION_ORDER]
        r.ref_done = r.g.copy()  # 补全之后、剪枝之前的图（阶段图用）
    for r in reports:
        r.replaced = prune_shortcuts(r.g, layer_map(r.g, r.table))
        r.pruned = prune_accounting(r.g)
    log(f'剪枝完成：删掉有方向一致、先往上走的更长链路的直连 {sum(len(r.replaced) for r in reports)} 条，'
        f'只有方向相反的链路而保留的直连 {sum(bool(d["conflict_path"]) for r in reports for *_, d in r.g.edges(data=True))} 条；'
        f'删掉没有下层概念连入的会计科目平均每篇 {np.mean([len(r.pruned) for r in reports]):.0f} 个')
