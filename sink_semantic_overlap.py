"""同一公司不同深度报告之间汇点的重合率：精确名称匹配 vs 语义相似度匹配（不调用大模型，用本地句向量模型）。
目的：从同一公司的多篇研报里找出同一个汇点概念，再对比各篇通往它的逻辑路径；这里先看能配上多少汇点、配上的汇点各有多少路径。

数据：概念 DAG 输出目录（默认 output_dag/deep）里按公司统一后的边 unified_edges.parquet，每篇建图同 endpoint_overlap.py
（源点 = 没有入边，汇点 = 没有出边）；只看有 2 篇以上研报的公司，同一公司研报两两配对，去掉疑似重复的对（同 endpoint_overlap.py）。
句向量：BAAI/bge-base-zh-v1.5（src/logic/output.embed，[CLS] 向量、L2 归一化，余弦即内积）。两种文本：
  名称      只编码概念名
  名称+定义  编码“名称：定义”（定义取这篇研报抽取时写的，见 concepts.parquet；统一后的名称取并进来的原名里同名的那个的定义）
名称完全相同算相似度 1。
匹配：两篇的汇点算两两相似度，从高到低一对一贪心配对（每个汇点最多配一个），相似度不低于阈值才算配上。
  配上数 m 之后：软 Jaccard = m / (|A| + |B| - m)，覆盖率 = m / min(|A|, |B|)（小的那篇的汇点有多少在另一篇找得到）。
另算“汇点在另一篇任意概念里找得到”的比例（不要求在另一篇也是汇点，也不要求一对一；两个方向平均）：
  另一篇里这个概念不是汇点时，仍可以取它的上游路径来对比。
路径数：通往某个概念的完整路径条数 = 从各源点走到它的不同路径条数（动态规划）。
输出例子：每个相似度区间随机抽一些“名称不同但配上了”的汇点对，用来判断阈值下的误配。
通用汇点逐个看（TARGETS，在最多研报对里出现的汇点名称）：每对研报里两篇是否都以它为汇点、是否都有这个概念（任意角色），
两篇都有时各有多少条路径通往它；另按 SYNONYMS 把同义叫法算作同一个概念再算一遍。SYNONYMS 是从句向量相似度 >= 0.80 的候选里
挑出的确实同义的叫法（毛利率 ⇄ 净利率、归母净利润 ⇄ 归母净利率、费用 ⇄ 费用率这类相似度也很高但不同义的没有收）。

在仓库根目录运行：
    python sink_semantic_overlap.py                   打印汇总，同时写 <输出目录>/data/sink_semantic_overlap.txt
    python sink_semantic_overlap.py --examples 30     每个区间抽 30 个例子
结果：<输出目录>/data/sink_matches.parquet（每对研报、每种文本、每个配上的汇点对一行，带相似度和两边的路径数）。
"""
import argparse
import random
import statistics
from collections import Counter, defaultdict
from datetime import date
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src.logic.output import EMBED_MODEL, embed
from src.viz.plot_dag_structure import build_graph

THRESHOLDS = [1.0, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]
BANDS = [(0.95, 1.0), (0.92, 0.95), (0.90, 0.92), (0.88, 0.90), (0.85, 0.88), (0.80, 0.85)]
TARGETS = ['投资评级', '营业收入', '归母净利润', '毛利率', '净利率', '盈利能力', '目标价', '销售费用率', '管理费用率',
           '期间费用率', '研发费用率', 'ROE']
SYNONYMS = {'投资评级': ['买入评级'], '营业收入': ['营业总收入', '主营营收', '总收入'],
            '毛利率': ['销售毛利率', '综合毛利率', '毛利润率', '公司整体毛利率'], '净利率': ['销售净利率', '净利润率'],
            '盈利能力': ['盈利水平', '公司盈利能力', '公司整体盈利能力'], '目标价': ['目标股价', '目标价格'],
            '期间费用率': ['三大期间费用率'], 'ROE': ['净资产收益率']}
DUP_DAYS, DUP_CHARS = 7, 0.02  # 疑似重复的研报对：同一券商、相隔天数不超过、字数相对差不超过（同 endpoint_overlap.py）


def day(d):
    return date(int(d[:4]), int(d[4:6]), int(d[6:8]))


def graph_info(edges):
    """-> (汇点列表, 全部概念列表, {概念: 从源点通往它的路径条数})。"""
    g, _ = build_graph(edges)
    up = {}
    for v in nx.topological_sort(g):
        preds = list(g.predecessors(v))
        up[v] = sum(up[p] for p in preds) if preds else 1
    return sorted(v for v in g if g.out_degree(v) == 0), sorted(g), up


def greedy(sim, t):
    """相似度矩阵上一对一贪心配对，返回 [(i, j, 相似度)]（相似度 >= t）。"""
    if sim.size == 0:
        return []
    idx = np.argwhere(sim >= t - 1e-9)
    order = sorted(idx.tolist(), key=lambda ij: -sim[ij[0], ij[1]])
    used_a, used_b, out = set(), set(), []
    for i, j in order:
        if i not in used_a and j not in used_b:
            used_a.add(i), used_b.add(j)
            out.append((i, j, float(sim[i, j])))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--output', default='output_dag/deep', help='概念 DAG 输出目录（默认 output_dag/deep）')
    ap.add_argument('--examples', type=int, default=20, help='每个相似度区间抽多少个配对例子')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    data_dir = Path(args.output) / 'data'
    reports = pq.read_table(data_dir / 'reports.parquet').to_pylist()
    rep = {r['report_id']: r for r in reports}
    by_code = defaultdict(list)
    for r in sorted(reports, key=lambda r: (r['date'], r['broker'], r['report_id'])):
        by_code[r['stock_code']].append(r['report_id'])
    multi = {c: rids for c, rids in by_code.items() if len(rids) > 1}
    keep = {rid for rids in multi.values() for rid in rids}

    edges = defaultdict(list)
    for e in pq.read_table(data_dir / 'unified_edges.parquet').to_pylist():
        if e['report_id'] in keep:
            edges[e['report_id']].append((e['cause'], e['effect'], e['support']))
    info = {rid: graph_info(edges[rid]) for rid in keep}

    # 每篇每个统一后名称的定义
    defn = {(c['report_id'], c['name']): c['definition'] or '' for c in pq.read_table(data_dir / 'concepts.parquet').to_pylist()
            if c['report_id'] in keep}
    members = defaultdict(list)
    for c in pq.read_table(data_dir / 'concept_unify.parquet').to_pylist():
        if c['report_id'] in keep and c['in_dag']:
            members[(c['report_id'], c['unified'])].append(c['name'])

    def definition(rid, name):
        names = members.get((rid, name), [name])
        own = name if name in names else names[0]
        return defn.get((rid, own), '')

    # 句向量：名称；名称+定义（每篇每个概念一条）
    names = sorted({v for rid in keep for v in info[rid][1]})
    texts = sorted({(rid, v) for rid in keep for v in info[rid][1]})
    print(f'编码 {len(names)} 个不同名称、{len(texts)} 条“名称：定义”（{EMBED_MODEL}）', flush=True)
    vec_name = dict(zip(names, embed(names, EMBED_MODEL)))
    vec_def = dict(zip(texts, embed([f'{v}：{definition(rid, v)}' if definition(rid, v) else v for rid, v in texts], EMBED_MODEL)))
    variants = {'名称': lambda rid, v: vec_name[v], '名称+定义': lambda rid, v: vec_def[(rid, v)]}

    def sim_matrix(ra, va, rb, vb, how):
        if not va or not vb:
            return np.zeros((len(va), len(vb)))
        A = np.stack([variants[how](ra, v) for v in va])
        B = np.stack([variants[how](rb, v) for v in vb])
        s = A @ B.T
        for i, x in enumerate(va):  # 名称相同算 1
            for j, y in enumerate(vb):
                if x == y:
                    s[i, j] = 1.0
        return s

    pairs = []
    for code, rids in multi.items():
        for a, b in combinations(rids, 2):
            ra_, rb_ = rep[a], rep[b]
            dup = (ra_['broker'] == rb_['broker'] and abs((day(ra_['date']) - day(rb_['date'])).days) <= DUP_DAYS
                   and abs(ra_['n_chars'] - rb_['n_chars']) <= DUP_CHARS * max(ra_['n_chars'], rb_['n_chars']))
            if not dup:
                pairs.append((code, a, b))

    res = {how: {t: [] for t in THRESHOLDS} for how in variants}  # 每对的 (m, |A|, |B|)
    anynode = {how: {t: [] for t in THRESHOLDS} for how in variants}  # 每对：汇点在另一篇任意概念里找得到的比例
    match_rows, name_stat = [], defaultdict(Counter)
    for code, a, b in pairs:
        (ka, na, upa), (kb, nb, upb) = info[a], info[b]
        for how in variants:
            s = sim_matrix(a, ka, b, kb, how)
            s_ab = sim_matrix(a, ka, b, nb, how)  # A 的汇点 vs B 的全部概念
            s_ba = sim_matrix(b, kb, a, na, how)
            for t in THRESHOLDS:
                m = greedy(s, t)
                res[how][t].append((len(m), len(ka), len(kb)))
                fa = (s_ab.max(axis=1) >= t - 1e-9).mean() if ka and nb else 0
                fb = (s_ba.max(axis=1) >= t - 1e-9).mean() if kb and na else 0
                anynode[how][t].append((fa + fb) / 2)
            for i, j, x in greedy(s, min(THRESHOLDS)):
                match_rows.append({'code': code, 'company': rep[a]['company'], 'rid_a': a, 'rid_b': b, 'how': how,
                                   'sink_a': ka[i], 'sink_b': kb[j], 'sim': x, 'exact': ka[i] == kb[j],
                                   'def_a': definition(a, ka[i]), 'def_b': definition(b, kb[j]),
                                   'paths_a': upa[ka[i]], 'paths_b': upb[kb[j]]})
        for v in set(ka) | set(kb):  # 每个名称：几对里两篇都是汇点 / 一篇是汇点另一篇也有这个概念 / 只有一篇是汇点
            in_a, in_b = v in ka, v in kb
            if in_a and in_b:
                name_stat[v]['都是汇点'] += 1
            elif (in_a and v in nb) or (in_b and v in na):
                name_stat[v]['另一篇有但不是汇点'] += 1
            else:
                name_stat[v]['另一篇没有'] += 1

    lines = [f'{len(multi)} 家公司、{len(keep)} 篇研报、{len(pairs)} 对（同一公司两两配对，不含疑似重复）；'
             f'每篇汇点中位数 {statistics.median(len(info[r][0]) for r in keep):g} 个。', '']
    lines.append('汇点匹配（一对一贪心）：软 Jaccard = m/(|A|+|B|-m)，覆盖率 = m/min(|A|,|B|)，每对平均；'
                 '“任意概念”= 汇点在另一篇任意概念里找得到的比例')
    lines.append(f'  {"文本":<8}{"阈值":>6}{"配上个数":>9}{"软Jaccard":>11}{"覆盖率":>9}{"至少配上1个":>11}{"任意概念":>9}')
    for how in variants:
        for t in THRESHOLDS:
            v = res[how][t]
            jac = [m / (x + y - m) if x + y - m else 0 for m, x, y in v]
            cov = [m / min(x, y) if min(x, y) else 0 for m, x, y in v]
            lines.append(f'  {how:<8}{"精确" if t == 1.0 else f"{t:.2f}":>7}{statistics.mean(m for m, _, _ in v):>11.2f}'
                         f'{statistics.mean(jac):>12.3f}{statistics.mean(cov):>10.3f}{sum(m > 0 for m, _, _ in v) / len(v):>12.1%}'
                         f'{statistics.mean(anynode[how][t]):>10.3f}')
        lines.append('')

    lines.append('各汇点名称（统一后、精确名称）：在多少对里两篇都是汇点（按“至少一篇是汇点”的对数排序，前 25 个）')
    lines.append(f'  {"名称":<12}{"至少一篇是汇点":>10}{"两篇都是汇点":>10}{"另一篇有但不是汇点":>14}{"另一篇没有":>8}')
    for v, c in sorted(name_stat.items(), key=lambda kv: -sum(kv[1].values()))[:25]:
        tot = sum(c.values())
        lines.append(f'  {v:<12}{tot:>12}{c["都是汇点"] / tot:>14.1%}{c["另一篇有但不是汇点"] / tot:>16.1%}{c["另一篇没有"] / tot:>12.1%}')

    rng = random.Random(args.seed)
    for how in variants:
        lines += ['', f'【{how}】名称不同但配上的汇点对，按相似度区间随机抽 {args.examples} 个（公司｜A ⇄ B｜相似度）：']
        rows = [r for r in match_rows if r['how'] == how and not r['exact']]
        for lo, hi in BANDS:
            band = [r for r in rows if lo <= r['sim'] < hi]
            lines.append(f'  相似度 {lo:.2f}–{hi:.2f}（共 {len(band)} 对）：')
            for r in rng.sample(band, min(args.examples, len(band))):
                lines.append(f'    {r["company"]}｜{r["sink_a"]} ⇄ {r["sink_b"]}｜{r["sim"]:.3f}')

    # 通用汇点逐个看
    def find(rid, t, syn):
        """这篇里代表目标 t 的概念：先找原名，syn 为真时再找同义叫法（取路径最多的）；返回 (名称, 是否汇点, 路径数) 或 None。"""
        sinks, nodes, up = info[rid]
        cands = [t] + (SYNONYMS.get(t, []) if syn else [])
        hit = [v for v in cands if v in up]
        if not hit:
            return None
        v = t if t in up else max(hit, key=lambda x: up[x])
        return v, v in set(sinks), up[v]

    lines += ['', '通用汇点：每对研报里两篇都以它为汇点 / 都有这个概念（任意角色）的比例（分母是至少一篇以它为汇点的对数）；',
              '“同义”把 SYNONYMS 里的叫法算作同一个概念；路径数是两篇都有这个概念时通往它的路径条数中位数（A 篇 / B 篇）']
    lines.append(f'  {"名称":<10}{"有它的篇数":>8}{"对数":>6}{"都是汇点":>8}{"同义":>7}{"都有":>8}{"同义":>7}{"路径数":>10}')
    for t in TARGETS:
        n_rep = sum(t in info[rid][2] for rid in keep)
        st = Counter()
        paths = []
        for code, a, b in pairs:
            fa, fb = find(a, t, False), find(b, t, False)
            ga, gb = find(a, t, True), find(b, t, True)
            if not ((ga and ga[1]) or (gb and gb[1])):
                continue
            st['n'] += 1
            st['sink'] += bool(fa and fb and fa[1] and fb[1])
            st['sink_syn'] += bool(ga and gb and ga[1] and gb[1])
            st['both'] += bool(fa and fb)
            st['both_syn'] += bool(ga and gb)
            if ga and gb:
                paths.append((ga[2], gb[2]))
        if not st['n']:
            continue
        med = f'{statistics.median(x for x, _ in paths):g} / {statistics.median(y for _, y in paths):g}' if paths else '—'
        lines.append(f'  {t:<10}{n_rep:>8}{st["n"]:>8}{st["sink"] / st["n"]:>10.1%}{st["sink_syn"] / st["n"]:>8.1%}'
                     f'{st["both"] / st["n"]:>9.1%}{st["both_syn"] / st["n"]:>8.1%}{med:>11}')

    exact = [r for r in match_rows if r['how'] == '名称' and r['exact']]
    lines += ['', f'名称相同的共有汇点 {len(exact)} 个：通往它的路径条数（A 篇、B 篇）中位数 '
              f'{statistics.median(r["paths_a"] for r in exact):g}、{statistics.median(r["paths_b"] for r in exact):g}，'
              f'两篇都只有 1 条的占 {sum(r["paths_a"] == 1 and r["paths_b"] == 1 for r in exact) / len(exact):.1%}']
    pq.write_table(pa.Table.from_pylist(match_rows), data_dir / 'sink_matches.parquet')
    text = '\n'.join(lines)
    (data_dir / 'sink_semantic_overlap.txt').write_text(text + '\n', encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
