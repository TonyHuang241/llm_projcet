"""同一公司不同深度报告之间源点、汇点的重合率（不调用大模型，几秒钟）。
读概念 DAG 输出目录（默认 output_dag/deep）里按公司统一后的边（unified_edges.parquet），每篇研报建一张图
（同 src/viz/plot_dag_structure.build_graph，只用进图的边），源点 = 没有入边的概念，汇点 = 没有出边的概念。
对有 2 篇以上研报的公司，同一公司的研报两两配对，每对算：
  jac       Jaccard = |A∩B| / |A∪B|
  ovl       重叠系数 = |A∩B| / min(|A|, |B|)：小的那篇的源点（汇点）有多少也是另一篇的源点（汇点）
分别对源点（src）、汇点（snk）、全部进图概念（node，参照）算；另用统一前的原名（edges.parquet）算一遍源点、汇点的 jac
（raw_ 开头），看统一带来多少；再随机配同样多对不同公司的研报，按原名算源点、汇点的 jac 作对照。
角色一致性：A 的源点在 B 的图里出现时，在 B 里也是源点、是中间概念、还是汇点（每对两个方向都算，按概念计数）。
去掉通用名称（_x 结尾）：在全样本 10% 以上公司的图里都出现过的名称（投资评级、营业收入、归母净利润、毛利率……，
按原样字符串数公司），从源点、汇点集合里去掉再算 jac，看重合是不是主要来自这些各家都有的财务指标。
疑似重复：同一券商、相隔不超过 7 天、正文字数相差不到 2% 的两篇（同一篇或勘误版），不进主要汇总，单独列出，
可以当作“同一篇文章抽两次”的重合率上限参照（同文重抽概念名也只有约四成相同）。
汇总分两种权重：每对一样重；先在公司内平均再在公司间平均（每家公司一样重，避免研报多的公司占比太大）。
另按同一券商 / 不同券商分开，并列出在最多公司里重合的源点、汇点名称。

在仓库根目录运行：
    python endpoint_overlap.py                         汇总打印到屏幕，同时写 <输出目录>/data/endpoint_overlap.txt
    python endpoint_overlap.py --output output_dag     换输出目录
结果：<输出目录>/data/endpoint_overlap_pairs.parquet（每对一行）、endpoint_overlap_companies.parquet（每家公司一行）。
"""
import argparse
import random
import statistics
from datetime import date
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.viz.plot_dag_structure import build_graph

GENERIC_SHARE = 0.10  # 在全样本这个比例以上公司的图里都出现的名称算通用名称
DUP_DAYS, DUP_CHARS = 7, 0.02  # 疑似重复：同一券商、相隔天数不超过、字数相对差不超过


def ends(edges):
    """[(原因, 结果, 引文数)] -> (源点集合, 汇点集合, 全部概念集合, {概念: 角色})。"""
    g, _ = build_graph(edges)
    src = {v for v in g if g.in_degree(v) == 0}
    snk = {v for v in g if g.out_degree(v) == 0}
    role = {v: '源点' if v in src else '汇点' if v in snk else '中间' for v in g}
    return src, snk, set(g), role


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else None


def ovl(a, b):
    return len(a & b) / min(len(a), len(b)) if a and b else None


def day(d):
    return date(int(d[:4]), int(d[4:6]), int(d[6:8]))


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def summarize(rows, by_company, keys, label):
    """每对一样重、每家公司一样重两种平均，外加中位数和“完全不重合”的比例。"""
    out = [f'{label}（{len(rows)} 对，{len({r["code"] for r in rows})} 家公司）']
    out.append(f'  {"指标":<14}{"每对平均":>8}{"中位数":>8}{"公司平均":>8}{"为 0 的对":>9}')
    for k, name in keys:
        v = [r[k] for r in rows if r[k] is not None]
        if not v:
            continue
        co = defaultdict(list)
        for r in rows:
            if r[k] is not None:
                co[r['code']].append(r[k])
        out.append(f'  {name:<14}{mean(v):>9.3f}{statistics.median(v):>10.3f}{mean([mean(x) for x in co.values()]):>10.3f}'
                   f'{sum(x == 0 for x in v) / len(v):>10.1%}')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--output', default='output_dag/deep', help='概念 DAG 输出目录（默认 output_dag/deep）')
    ap.add_argument('--seed', type=int, default=0, help='不同公司对照组随机配对的种子')
    args = ap.parse_args()
    data_dir = Path(args.output) / 'data'
    reports = pq.read_table(data_dir / 'reports.parquet').to_pylist()
    rep = {r['report_id']: r for r in reports}
    uni, raw = defaultdict(list), defaultdict(list)
    for e in pq.read_table(data_dir / 'unified_edges.parquet').to_pylist():
        uni[e['report_id']].append((e['cause'], e['effect'], e['support']))
    for e in pq.read_table(data_dir / 'edges.parquet').to_pylist():
        if e['in_dag']:
            raw[e['report_id']].append((e['cause'], e['effect'], e['support']))
    G = {rid: ends(uni[rid]) for rid in rep}
    seen_in = defaultdict(set)  # 名称 -> 出现在哪些公司的图里（统一后的名称，全样本）
    for rid, (_, _, nodes, _) in G.items():
        for v in nodes:
            seen_in[v].add(rep[rid]['stock_code'])
    n_co = len({r['stock_code'] for r in reports})
    generic = {v for v, cs in seen_in.items() if len(cs) >= GENERIC_SHARE * n_co}
    Graw = {rid: ends(raw[rid]) for rid in rep}

    by_code = defaultdict(list)
    for r in sorted(reports, key=lambda r: (r['date'], r['broker'], r['report_id'])):
        by_code[r['stock_code']].append(r['report_id'])
    multi = {c: rids for c, rids in by_code.items() if len(rids) > 1}

    rows, role_count = [], {'源点': Counter(), '汇点': Counter()}
    shared_src, shared_snk = defaultdict(set), defaultdict(set)  # 名称 -> 在哪些公司的研报对里重合
    for code, rids in multi.items():
        for a, b in combinations(rids, 2):
            (sa, ka, na, ra), (sb, kb, nb, rb) = G[a], G[b]
            (wsa, wka, _, _), (wsb, wkb, _, _) = Graw[a], Graw[b]
            ra_, rb_ = rep[a], rep[b]
            gap = abs((day(ra_['date']) - day(rb_['date'])).days)
            dup = (ra_['broker'] == rb_['broker'] and gap <= DUP_DAYS
                   and abs(ra_['n_chars'] - rb_['n_chars']) <= DUP_CHARS * max(ra_['n_chars'], rb_['n_chars']))
            for mine, other in ((ra, rb), (rb, ra)):
                if dup:
                    break
                for v, role in mine.items():
                    if role in role_count:
                        role_count[role][other.get(v, '不在图里')] += 1
            for v in (sa & sb) if not dup else ():
                shared_src[v].add(code)
            for v in (ka & kb) if not dup else ():
                shared_snk[v].add(code)
            rows.append({'code': code, 'company': ra_['company'], 'rid_a': a, 'rid_b': b,
                         'broker_a': ra_['broker'], 'broker_b': rb_['broker'], 'date_a': ra_['date'], 'date_b': rb_['date'],
                         'same_broker': ra_['broker'] == rb_['broker'], 'days': gap, 'dup': dup,
                         'n_src_a': len(sa), 'n_src_b': len(sb), 'n_snk_a': len(ka), 'n_snk_b': len(kb),
                         'src_common': len(sa & sb), 'snk_common': len(ka & kb),
                         'src_jac': jac(sa, sb), 'snk_jac': jac(ka, kb), 'node_jac': jac(na, nb),
                         'src_ovl': ovl(sa, sb), 'snk_ovl': ovl(ka, kb), 'node_ovl': ovl(na, nb),
                         'src_jac_x': jac(sa - generic, sb - generic), 'snk_jac_x': jac(ka - generic, kb - generic),
                         'raw_src_jac': jac(wsa, wsb), 'raw_snk_jac': jac(wka, wkb)})

    # 对照：同样多对不同公司的研报，按原名
    rng = random.Random(args.seed)
    pool = [rid for rids in multi.values() for rid in rids]
    cross = []
    while len(cross) < len(rows):
        a, b = rng.sample(pool, 2)
        if rep[a]['stock_code'] != rep[b]['stock_code']:
            cross.append({'code': f'{rep[a]["stock_code"]}|{rep[b]["stock_code"]}',
                          'src_jac': jac(Graw[a][0], Graw[b][0]), 'snk_jac': jac(Graw[a][1], Graw[b][1]),
                          'node_jac': jac(Graw[a][2], Graw[b][2]),
                          'src_jac_x': jac(Graw[a][0] - generic, Graw[b][0] - generic),
                          'snk_jac_x': jac(Graw[a][1] - generic, Graw[b][1] - generic)})

    keys = [('src_jac', '源点 Jaccard'), ('snk_jac', '汇点 Jaccard'), ('node_jac', '全部概念 Jaccard'),
            ('src_ovl', '源点 重叠系数'), ('snk_ovl', '汇点 重叠系数'), ('node_ovl', '全部概念 重叠系数'),
            ('src_jac_x', '去通用 源点 Jac'), ('snk_jac_x', '去通用 汇点 Jac'),
            ('raw_src_jac', '统一前 源点 Jac'), ('raw_snk_jac', '统一前 汇点 Jac')]
    dups = [r for r in rows if r['dup']]
    main_rows = [r for r in rows if not r['dup']]
    lines = [f'{len(multi)} 家公司有 2 篇以上深度报告，共 {sum(map(len, multi.values()))} 篇、{len(rows)} 对（同一公司内两两配对）。',
             f'每篇源点中位数 {statistics.median(len(G[r][0]) for r in pool):g} 个、汇点 {statistics.median(len(G[r][1]) for r in pool):g} 个、'
             f'进图概念 {statistics.median(len(G[r][2]) for r in pool):g} 个。', '']
    lines += [f'通用名称 {len(generic)} 个（在 {n_co} 家公司的 {GENERIC_SHARE:.0%} 以上出现）：'
              + '、'.join(sorted(generic, key=lambda v: (-len(seen_in[v]), v))), '']
    lines += summarize(main_rows, None, keys, f'同一公司的研报对（不含 {len(dups)} 对疑似重复）')
    lines += [''] + summarize([r for r in main_rows if r['same_broker']], None, keys[:8], '其中同一券商')
    lines += [''] + summarize([r for r in main_rows if not r['same_broker']], None, keys[:8], '其中不同券商')
    lines += [''] + summarize(cross, None, [('src_jac', '源点 Jaccard'), ('snk_jac', '汇点 Jaccard'), ('node_jac', '全部概念 Jaccard'),
                                            ('src_jac_x', '去通用 源点 Jac'), ('snk_jac_x', '去通用 汇点 Jac')],
                              '对照：随机配对的不同公司研报，按原名')
    lines += ['', '疑似重复的研报对（同一券商、相隔不超过 7 天、字数相差不到 2%；可当作同文重抽的上限参照）：']
    for r in dups:
        lines.append(f'  {r["company"]} {r["broker_a"]} {r["date_a"]}/{r["date_b"]}：源点 Jac {r["src_jac"]:.3f}、'
                     f'汇点 Jac {r["snk_jac"]:.3f}、全部概念 Jac {r["node_jac"]:.3f}')
    lines += ['', '角色一致性：一篇的源点（汇点）在同公司另一篇里是什么（每对两个方向，按概念计数）']
    for role, c in role_count.items():
        tot = sum(c.values())
        present = tot - c['不在图里']
        lines.append(f'  {role}（{tot} 次）：不在另一篇的图里 {c["不在图里"] / tot:.1%}；出现在另一篇的 {present} 次里，'
                     + '、'.join(f'是{k} {c[k] / present:.1%}' for k in ('源点', '中间', '汇点')))
    for name, d in (('源点', shared_src), ('汇点', shared_snk)):
        top = sorted(d.items(), key=lambda kv: (-len(kv[1]), kv[0]))[:20]
        lines += ['', f'在最多公司里重合的{name}（名称：公司数）：', '  ' + '、'.join(f'{k}：{len(v)}' for k, v in top)]

    co_rows = []
    for code, rids in multi.items():
        rs = [r for r in main_rows if r['code'] == code]
        if not rs:  # 这家公司只有一对，而且是疑似重复
            continue
        co_rows.append({'code': code, 'company': rs[0]['company'], 'n_reports': len(rids), 'n_pairs': len(rs),
                        **{k: mean([r[k] for r in rs]) for k, _ in keys}})
    pq.write_table(pa.Table.from_pylist(rows), data_dir / 'endpoint_overlap_pairs.parquet')
    pq.write_table(pa.Table.from_pylist(co_rows), data_dir / 'endpoint_overlap_companies.parquet')
    text = '\n'.join(lines)
    (data_dir / 'endpoint_overlap.txt').write_text(text + '\n', encoding='utf-8')
    print(text)
    print(f'\n每对明细：{data_dir / "endpoint_overlap_pairs.parquet"}；每家公司：{data_dir / "endpoint_overlap_companies.parquet"}')


if __name__ == '__main__':
    main()
