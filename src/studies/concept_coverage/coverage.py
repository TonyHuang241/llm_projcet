"""concept_coverage 的统计逻辑：每篇研报一行的标记表、度数核对、汇总表和分组表（口径见 __init__.py 的模块说明）。"""
import numpy as np
import pandas as pd

from src.graph.load import read_table

LENGTH_BINS = [0, 1000, 2000, 5000, 10000, np.inf]
LENGTH_LABELS = ['<1000', '1000–2000', '2000–5000', '5000–10000', '≥10000']
META = ['company', 'stock_code', 'date', 'broker', 'n_chars', 'n_chunks']
FLAGS = ['extracted', 'in_table', 'in_graph', 'strict_sink', 'has_out', 'source_only', 'n_in', 'n_out',
         'alias_only', 'has_variant', 'variant_names']
SHARES = {'share_in_table': 'in_table', 'share_in_graph': 'in_graph', 'share_strict_sink': 'strict_sink'}


def edge_degrees(E):
    """边表 -> 每个（研报, 概念）在图里的入度、出度：只用 in_dag 为真的边，同一对（原因, 结果）只算一次、自环不算，
    与 src/graph/load.py 的 build_graph 建出的图（measure 的口径）相同。"""
    e = E[E.in_dag & (E.cause != E.effect)].drop_duplicates(['report_id', 'cause', 'effect'])
    n_in = e.groupby(['report_id', 'effect']).size().rename_axis(['report_id', 'name']).rename('n_in_e')
    n_out = e.groupby(['report_id', 'cause']).size().rename_axis(['report_id', 'name']).rename('n_out_e')
    return pd.concat([n_in, n_out], axis=1).fillna(0).astype(int).reset_index()


def degree_check(C, D):
    """概念表的 n_in、n_out、in_dag 与按边表重算的逐个概念比对，返回 dict。"""
    m = C.merge(D, on=['report_id', 'name'], how='outer', indicator=True)
    only_edges = int((m._merge == 'right_only').sum())
    m = m[m._merge != 'right_only'].fillna({'n_in_e': 0, 'n_out_e': 0})
    bad = (m.n_in != m.n_in_e) | (m.n_out != m.n_out_e) | (m.in_dag != ((m.n_in_e + m.n_out_e) > 0))
    return {'n_concepts': len(m), 'n_mismatch_concepts': int(bad.sum()), 'n_mismatch_reports': int(m.report_id[bad].nunique()),
            'n_edge_nodes_not_in_table': only_edges}


def report_flags(sample, version, concept, contains):
    """(每篇研报一行的标记表（行序同样本清单）, 度数核对结果)。"""
    from src import samples
    plan = samples.assignment(sample, version)[0]
    R = read_table('reports', sample, columns=META, version=version)
    C = read_table('concepts', sample, columns=['name', 'aliases', 'in_dag', 'n_in', 'n_out'], version=version)
    E = read_table('edges', sample, columns=['cause', 'effect', 'in_dag'], version=version)
    D = edge_degrees(E)
    check = degree_check(C, D)

    t = plan.merge(R.drop(columns='version'), on='report_id', how='left')  # version：清单（或 --version）指定的版本
    t[['n_chars', 'n_chunks']] = t[['n_chars', 'n_chunks']].astype('Int64')  # 没有抽取结果的研报缺失
    t['extracted'] = t.report_id.isin(set(R.report_id))
    t['in_table'] = t.report_id.isin(set(C.report_id[C.name == concept]))
    deg = D[D.name == concept].set_index('report_id')
    t['n_in'] = deg.n_in_e.reindex(t.report_id).fillna(0).astype(int).values
    t['n_out'] = deg.n_out_e.reindex(t.report_id).fillna(0).astype(int).values
    t['in_graph'] = (t.n_in + t.n_out) > 0
    t['strict_sink'] = (t.n_in > 0) & (t.n_out == 0)
    t['has_out'] = t.n_out > 0
    t['source_only'] = (t.n_in == 0) & (t.n_out > 0)
    alias_hit = C.aliases.fillna('').map(lambda s: concept in s.split('、'))
    t['alias_only'] = t.report_id.isin(set(C.report_id[alias_hit])) & ~t.in_table
    if contains:
        v = C[C.name.str.contains(contains, regex=False) & (C.name != concept)]
        names = v.groupby('report_id', sort=False).name.agg('、'.join)
        t['variant_names'] = np.where(t.in_table, '', names.reindex(t.report_id).fillna('').values)
    else:
        t['variant_names'] = ''
    t['has_variant'] = t.variant_names != ''
    for c in ('n_in', 'n_out'):  # 没有抽取结果的研报度数缺失
        t[c] = t[c].astype('Int64').mask(~t.extracted)
    return t[['report_id', 'version', *META, *FLAGS]], check


def summary_table(t, contains):
    """汇总表：漏斗各层的篇数、占已抽取的比例、占上一层（parent 列）的比例。"""
    x = t[t.extracted]
    no_exact = ~x.in_table
    rows = [('样本清单', len(t), None), ('已抽取', len(x), '样本清单'),
            ('概念表里有同名概念', int(x.in_table.sum()), '已抽取'), ('在图里', int(x.in_graph.sum()), '概念表里有同名概念'),
            ('是严格汇点', int(x.strict_sink.sum()), '在图里'),
            ('在表里但不在图里（孤立）', int((x.in_table & ~x.in_graph).sum()), '概念表里有同名概念'),
            ('在图里但有出边', int(x.has_out.sum()), '在图里'), ('在图里但只有出边（入度 0）', int(x.source_only.sum()), '在图里'),
            ('没有同名概念', int(no_exact.sum()), '已抽取')]
    if contains:
        rows.append((f'没有同名概念但有近似名称（含“{contains}”）', int(x.has_variant.sum()), '没有同名概念'))
    rows += [('没有同名概念但别名里有', int(x.alias_only.sum()), '没有同名概念'),
             ('什么都没有', int((no_exact & ~x.alias_only & ~x.has_variant).sum()), '没有同名概念')]
    n = {k: v for k, v, _ in rows}
    return pd.DataFrame([{'item': k, 'n_reports': v, 'share_of_extracted': v / len(x) if k != '样本清单' else np.nan,
                          'parent': p or '', 'share_of_parent': v / n[p] if p and n[p] else np.nan} for k, v, p in rows])


def group_table(x, key, sort_by_n=False):
    """已抽取的研报按 key 分组：篇数和表里有 / 在图里 / 是严格汇点三个比例。"""
    g = x.groupby(key, observed=False, sort=not sort_by_n)
    out = g.size().rename('n_reports').to_frame()
    for s, f in SHARES.items():
        out[s] = g[f].mean()
    out = out.rename_axis('group').reset_index()
    if sort_by_n:
        out = out.sort_values(['n_reports', 'group'], ascending=[False, True], ignore_index=True)
    return out


def tables(t, contains):
    """{文件名: DataFrame}：summary、by_length、by_chunks、by_broker、variant_names。"""
    x = t[t.extracted].copy()
    x['length'] = pd.cut(x.n_chars, LENGTH_BINS, right=False, labels=LENGTH_LABELS)
    x['chunks'] = pd.Categorical(np.where(x.n_chunks >= 3, '3+', x.n_chunks.astype(int).astype(str)), ['1', '2', '3+'])
    names = x.variant_names[x.has_variant].str.split('、').explode()
    vn = names.groupby(names).size().rename('n_reports').rename_axis('name').reset_index()
    vn = vn.sort_values(['n_reports', 'name'], ascending=[False, True], ignore_index=True)
    return {'summary.csv': summary_table(t, contains), 'by_length.csv': group_table(x, 'length'),
            'by_chunks.csv': group_table(x, 'chunks'), 'by_broker.csv': group_table(x, 'broker', sort_by_n=True),
            'variant_names.csv': vn}


def md_table(df):
    """DataFrame -> Markdown 表格（share_ 开头的列显示为百分比，缺失值留空）。"""
    fmt = lambda v, pct: '' if isinstance(v, float) and np.isnan(v) else (f'{v:.2%}' if pct else str(v))  # noqa: E731
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(r[c], c.startswith('share_')) for c in cols) + ' |' for _, r in df.iterrows()]
    return '\n'.join(lines)
