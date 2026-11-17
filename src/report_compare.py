"""对比同一家公司不同研报的估值逻辑框架：把 report_logic 输出的逻辑图（report_graphs.json）转成邻接矩阵，算逐对、逐篇的
差异指标，供后续回归使用。

节点对齐分三个粒度：
  概念      全局名称（nodes[].label），只取第 3-5 层。同公司研报之间精确重合很少，只作参照。
  主题      每家公司调用一次大模型（prompt 的“主题”两节），把这家公司各篇研报的第 3-5 层概念归成主题，是主要的对比粒度。
  会计科目  第 2 层，表示框架最后落到哪些财务科目。

matrices.parquet：每篇研报的矩阵，长表（report_id、level、kind、row、col、value、sign），同一家公司内行列名称统一，
pivot 即可还原矩阵。value 为计数，sign 为净方向：+1 / -1，0 表示各条路径有正有负。
  concept / adj     概念级邻接矩阵（含会计科目和公司估值），value 为 1。
  concept / reach   概念级可达矩阵：沿边能走到就记一项，sign 合并所有路径的方向乘积。
  theme / adj       主题级邻接矩阵：两端都是第 3-5 层概念的边按主题累加，value 为边数；对角线是主题内部的传导。
  theme / reach     主题级可达矩阵：value 为能走到的概念对数。
  theme_acc / reach 主题→会计科目：value 为主题中能走到该科目的概念数。
  layer / adj       5×5 层间流量：value 为边数，行、列是层号，sign 为空。
默认用全部边；edge_set='text' 时只用有原文依据的边和预置边（去掉逻辑补全边），作稳健性检验。

pair_metrics.parquet：同公司每对研报一行，a 为较早发布的一篇。
  concept_jaccard           第 3-5 层全局名称的 Jaccard（参照）
  theme_jaccard             主题集合的 Jaccard
  theme_cont_ab / _ba       |Ta∩Tb|/|Ta|、|Ta∩Tb|/|Tb|：不对称覆盖，看点评的框架是否是深度报告的子集
  theme_cos                 主题权重（本篇落在该主题的概念数）向量的余弦；受大主题怎么拆分的影响大，主题重跑时
                            不稳定，优先用 theme_jaccard、theme_cont_ab/_ba
  acc_jaccard               会计科目集合的 Jaccard
  theme_edge_jaccard        主题级邻接矩阵非零项的 Jaccard（直接传导）
  theme_reach_jaccard       主题级可达矩阵非零项的 Jaccard（传导路径，不计写法）
  theme_acc_jaccard         主题→会计科目矩阵非零项的 Jaccard（驱动因素落到哪些科目）
  layer_flow_js             两篇层间流量占比的 JS 散度（以 2 为底，0-1）
  sign_agree                两篇都有、对估值的净影响都明确为正或负的同名概念中，方向一致的比例（概念级，见下）
  days_apart、same_broker   控制变量

report_metrics.parquet：每篇一行，与“同行”比较。同行默认是同公司的其他全部研报；peers='prior' 时只用更早发布的
（同一天的不算），window_days 限定往前多少天，用来避免回归时的前视偏差。
  体量  n_paragraphs、n_concepts、n_edges、n_themes、n_accounting、n_peers
  形态  share_l3 / l4 / l5   各层概念占比
        depth                最长链条的边数
        mean_path_len        概念到公司估值的平均最短路径长度
        theme_entropy        概念在主题上分布的熵，除以 log(公司主题数)，表示视角的广度
        acc_hhi              概念传导到各会计科目的集中度（每个概念权重 1，平分给它能走到的科目）
        logic_share          逻辑补全边的占比（没有原文依据，只作质量控制）
  关注点 theme_novelty       本篇主题中，没有任何同行提到的比例
        theme_coverage       半数以上同行提到的主题中，本篇覆盖的比例
        theme_js             本篇主题分布与同行平均分布的 JS 散度
  结构  link_novelty / link_coverage  同上，对象换成“连接”：主题→主题可达、主题→科目可达
        reach_agree          两端本篇都提到、且本篇或某个同行能走到的（主题，主题或科目）对中，两端都提到的同行与本篇
                             可达与否一致的平均比例
  方向  sign_conflict_n / sign_conflict  本篇对估值净影响明确的概念中，与同行（有同名概念且方向明确的）多数相反的个数、
        占可比概念的比例
  综合  dist_theme_mean / dist_link_mean  与各同行的平均 1-theme_jaccard、平均 1-（连接集合的 Jaccard）
没有同行或分母为 0 时为 NaN。

方向只在概念级比较：同一主题里常有极性相反的指标（行业竞争强度为负、本土需求为正；不良率为负、风控能力为正），
两篇各提一个时主题的净方向就会相反，并不是看法不同。主题级矩阵的 sign 也混有这种极性，只作参考。

themes.parquet：company、stock_code、theme、theme_definition、concept、layer、n_reports。

用法：先跑过 report_logic，再在 main.py 中设置 TASK = 'report_compare'。
"""
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import networkx as nx
import numpy as np

from src.llm_client import load_env, log, set_log_path
from src.logic.base import ROOT, Ctx, Step, norm_name, parse_json, short, write_parquet

TEXT_EDGES = ('extracted', 'bridged', 'preset')  # edge_set='text' 时保留的边
LAYERS = (1, 2, 3, 4, 5)
NAN = float('nan')


# ==================== 工具 ====================

def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if a | b else NAN


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else NAN


def js(p, q):
    """两个非负向量归一化后的 JS 散度（以 2 为底）；有一个全为 0 时为 NaN。"""
    p, q = np.asarray(p, float), np.asarray(q, float)
    if p.sum() == 0 or q.sum() == 0:
        return NAN
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda x: float(np.sum(x[x > 0] * np.log2(x[x > 0] / m[x > 0])))
    return (kl(p) + kl(q)) / 2


def cosine(a, b):
    keys = sorted(set(a) | set(b))
    u, v = np.array([a.get(k, 0) for k in keys], float), np.array([b.get(k, 0) for k in keys], float)
    d = np.linalg.norm(u) * np.linalg.norm(v)
    return float(u @ v / d) if d else NAN


def net(signs):
    return 1 if signs == {1} else -1 if signs == {-1} else 0


def to_date(s):
    return datetime.strptime(str(s), '%Y%m%d').date()


# ==================== 读图 ====================

def report_graph(rec, edge_set):
    """一篇研报的图，节点用全局名称；返回 (图, {节点: 层})。edge_set='text' 时去掉逻辑补全边。"""
    label = {n['id']: n['label'] for n in rec['nodes']}
    if len(set(label.values())) < len(label):
        raise SystemExit(f'{rec["report_id"]} 中有两个概念的全局名称相同，无法建矩阵')
    lay = {label[n['id']]: n['layer'] for n in rec['nodes']}
    g = nx.DiGraph()
    g.add_nodes_from(lay)
    for e in rec['edges']:
        if edge_set == 'text' and e['origin'] not in TEXT_EDGES:
            continue
        g.add_edge(label[e['source']], label[e['target']], sign=1 if e['direction'] == '正向' else -1)
    return g, lay


def path_signs(g):
    """{u: {v: u 沿边走到 v 的各条路径的方向乘积集合}}（图是 DAG）。"""
    out = {}
    for u in reversed(list(nx.topological_sort(g))):
        d = defaultdict(set)
        for s in g.successors(u):
            w = g.edges[u, s]['sign']
            d[s].add(w)
            for v, xs in out[s].items():
                d[v].update(w * x for x in xs)
        out[u] = d
    return out


# ==================== 主题 ====================

def concept_list(recs):
    """一家公司各篇研报的第 3-5 层概念：{全局名称: {'layer', 'definition', 'n_reports'}}，层取各篇中最常见的。"""
    info = defaultdict(lambda: {'layers': Counter(), 'definition': '', 'reports': set()})
    for rec in recs:
        for n in rec['nodes']:
            if n['layer'] >= 3:
                e = info[n['label']]
                e['layers'][n['layer']] += 1
                e['definition'] = e['definition'] or n['definition']
                e['reports'].add(rec['report_id'])
    return {k: {'layer': v['layers'].most_common(1)[0][0], 'definition': v['definition'], 'n_reports': len(v['reports'])}
            for k, v in info.items()}


def theme_range(n):
    """n 个概念时主题数的范围：大约 √n 到 1.6√n（20 个概念 4–7 个，117 个 11–17 个），用来稳住不同次运行的粒度。"""
    lo = min(n, max(2, round(math.sqrt(n))))
    return lo, min(n, max(lo + 2, round(1.6 * math.sqrt(n))))


def parse_themes(content, names, lo, hi, strict=True):
    """校验主题划分，返回 [{'name', 'definition', 'concepts'}]（概念按清单顺序）。strict=False 时（重试用完）只丢掉
    不合格的部分：assign 里出现、themes 里没列的主题照样建立，没有归入的概念各自单独成一个主题。"""
    obj = parse_json(content, dict)
    items, assign = obj.get('themes'), obj.get('assign')
    if not isinstance(items, list) or not isinstance(assign, dict):
        raise ValueError('需要 themes 数组和 assign 对象')
    index = {norm_name(n): n for n in names}
    defs, owner, errors = {}, {}, []
    for it in items:
        name = norm_name(it.get('name')) if isinstance(it, dict) else ''
        if name:
            if name in defs:
                errors.append(f'主题名“{name}”在 themes 中重复')
            defs.setdefault(name, short(it.get('definition')))
    for raw, t in assign.items():
        c, t = index.get(norm_name(raw)), norm_name(t)
        if c is None:
            errors.append(f'“{norm_name(raw)}”不在概念清单中：概念名只能从清单中原样照抄，规则里举的例子不一定在清单中')
        elif not t:
            errors.append(f'“{c}”没有写归入的主题')
        else:
            if t not in defs:
                errors.append(f'“{c}”归入的“{t}”不在 themes 中')
                defs[t] = ''
            owner[c] = t
    lack = [n for n in names if n not in owner]
    if lack:
        errors.append(f'assign 中缺少这些概念：{"、".join(lack)}')
        owner.update({n: n if n not in defs else f'{n}（单列）' for n in lack})
    members = defaultdict(list)
    for n in names:
        members[owner[n]].append(n)
    themes = [{'name': t, 'definition': defs.get(t, ''), 'concepts': cs} for t, cs in members.items()]
    if not lo <= len(themes) <= hi:
        errors.append(f'主题有 {len(themes)} 个，应在 {lo}–{hi} 个之间，请{"合并相近的主题" if len(themes) > hi else "把过粗的主题拆开"}')
    if errors and strict:
        raise ValueError('；'.join(errors))
    return themes


def assign_themes(ctx, companies):
    """每家公司调用一次大模型归主题。companies 为 {公司键: (公司名, 研报列表)}，
    返回 {公司键: {'themes': [...], 'of': {概念: 主题}, 'concepts': concept_list 的结果}}。"""
    step = Step(ctx, 'theme_cache.jsonl', '主题', fields=('themes',))

    def work(item):
        key, (company, recs) = item
        concepts = concept_list(recs)
        names = sorted(concepts, key=lambda n: (concepts[n]['layer'], -concepts[n]['n_reports'], n))
        if not names:
            return concepts, []
        text = '\n'.join(f'{n} ｜ 第{concepts[n]["layer"]}层 ｜ {concepts[n]["definition"] or "—"} ｜ {concepts[n]["n_reports"]} 篇'
                         for n in names)
        lo, hi = theme_range(len(names))
        themes = step.ask(step.key(company, text), lambda c: parse_themes(c, names, lo, hi),
                          lenient=lambda c: parse_themes(c, names, lo, hi, strict=False), extra={'company': company},
                          company=company, n=len(names), lo=lo, hi=hi, concepts=text)
        return concepts, themes

    items = sorted(companies.items())
    out = {}
    for (key, (company, recs)), res in zip(items, ctx.parallel(items, work, '主题归类')):
        if res is None:
            concepts = concept_list(recs)
            log(f'!!! {company} 主题归类失败，每个概念各自作为一个主题（主题级指标退化为概念级）')
            res = concepts, [{'name': n, 'definition': '', 'concepts': [n]} for n in sorted(concepts)]
        concepts, themes = res
        out[key] = {'themes': themes, 'of': {c: t['name'] for t in themes for c in t['concepts']}, 'concepts': concepts}
    return out


# ==================== 每篇研报的矩阵 ====================

def report_data(rec, edge_set, of, n_company_themes):
    """一篇研报的各级矩阵（{(行, 列): [计数, 方向集合]}）、集合与形态指标。of 为 {概念: 主题}。"""
    g, lay = report_graph(rec, edge_set)
    ps = path_signs(g)
    concepts = {n for n in g if lay[n] >= 3}
    acc = {n for n in g if lay[n] == 2}
    theme = lambda n: of.get(n, n)
    theme_w = Counter(theme(n) for n in sorted(concepts))  # 按名称顺序累加，浮点求和的结果每次相同

    def add(m, key, signs):
        cell = m.setdefault(key, [0, set()])
        cell[0] += 1
        cell[1] |= signs

    adj_t, reach_t, t_acc = {}, {}, {}
    for u, v, d in g.edges(data=True):
        if u in concepts and v in concepts:
            add(adj_t, (theme(u), theme(v)), {d['sign']})
    for u in concepts:
        for v, signs in ps[u].items():
            if v in concepts:
                add(reach_t, (theme(u), theme(v)), signs)
            elif v in acc:
                add(t_acc, (theme(u), v), signs)
    links = {('T', a, b) for a, b in reach_t} | {('A', a, m) for a, m in t_acc}
    c_effect = {u: net(ps[u][ROOT]) for u in concepts if ROOT in ps[u]}

    # 形态
    n_c = len(concepts)
    dist = nx.single_source_shortest_path_length(g.reverse(copy=False), ROOT) if ROOT in g else {}
    acc_w = Counter()
    for u in sorted(concepts):
        reach_acc = sorted(v for v in ps[u] if v in acc)
        for v in reach_acc:
            acc_w[v] += 1 / len(reach_acc)
    total = sum(theme_w.values())
    entropy = -sum(w / total * math.log(w / total) for w in theme_w.values()) if total else NAN
    shape = {**{f'share_l{k}': sum(lay[n] == k for n in concepts) / n_c if n_c else NAN for k in (3, 4, 5)},
             'depth': nx.dag_longest_path_length(g) if g.number_of_edges() else 0,
             'mean_path_len': mean(dist[n] for n in concepts if n in dist),
             'theme_entropy': entropy / math.log(n_company_themes) if total and n_company_themes >= 2 else NAN,
             'acc_hhi': sum((w / sum(acc_w.values())) ** 2 for w in acc_w.values()) if acc_w else NAN,
             'logic_share': mean(e['origin'] == 'logic' for e in rec['edges'])}
    return {'rec': rec, 'g': g, 'lay': lay, 'ps': ps, 'concepts': concepts, 'acc': acc, 'theme_w': theme_w,
            'themes': set(theme_w), 'adj_t': adj_t, 'reach_t': reach_t, 't_acc': t_acc,
            'c_effect': c_effect, 'links': links,
            'ends': {('T', t) for t in theme_w} | {('A', m) for m in acc},
            'layer_flow': Counter((lay[u], lay[v]) for u, v in g.edges()), 'shape': shape}


def matrix_rows(x):
    """一篇研报的矩阵长表行。"""
    rid, rows = x['rec']['report_id'], []
    row = lambda level, kind, r, c, value, sign: rows.append(
        {'report_id': rid, 'level': level, 'kind': kind, 'row': str(r), 'col': str(c), 'value': value, 'sign': sign})
    for u, v, d in sorted(x['g'].edges(data=True)):
        row('concept', 'adj', u, v, 1, d['sign'])
    for u in sorted(x['ps']):
        for v in sorted(x['ps'][u]):
            row('concept', 'reach', u, v, 1, net(x['ps'][u][v]))
    for level, kind, m in (('theme', 'adj', x['adj_t']), ('theme', 'reach', x['reach_t']), ('theme_acc', 'reach', x['t_acc'])):
        for (r, c), (n, signs) in sorted(m.items()):
            row(level, kind, r, c, n, net(signs))
    for (r, c), n in sorted(x['layer_flow'].items()):
        row('layer', 'adj', r, c, n, None)
    return rows


# ==================== 指标 ====================

def flow_vector(x):
    return [x['layer_flow'].get((a, b), 0) for a in LAYERS for b in LAYERS]


def pair_metrics(a, b):
    """一对研报（a 较早）的差异指标。"""
    ra, rb = a['rec'], b['rec']
    common = a['themes'] & b['themes']
    signed = [c for c in a['c_effect'] if a['c_effect'][c] and b['c_effect'].get(c)]
    return {'company': ra['company'], 'report_a': ra['report_id'], 'report_b': rb['report_id'],
            'broker_a': ra['broker'], 'broker_b': rb['broker'],
            'concept_jaccard': jaccard(a['concepts'], b['concepts']),
            'theme_jaccard': jaccard(a['themes'], b['themes']),
            'theme_cont_ab': len(common) / len(a['themes']) if a['themes'] else NAN,
            'theme_cont_ba': len(common) / len(b['themes']) if b['themes'] else NAN,
            'theme_cos': cosine(a['theme_w'], b['theme_w']),
            'acc_jaccard': jaccard(a['acc'], b['acc']),
            'theme_edge_jaccard': jaccard(a['adj_t'], b['adj_t']),
            'theme_reach_jaccard': jaccard(a['reach_t'], b['reach_t']),
            'theme_acc_jaccard': jaccard(a['t_acc'], b['t_acc']),
            'layer_flow_js': js(flow_vector(a), flow_vector(b)),
            'sign_agree': mean(a['c_effect'][c] == b['c_effect'][c] for c in signed),
            'days_apart': (to_date(rb['date']) - to_date(ra['date'])).days,
            'same_broker': ra['broker'] == rb['broker']}


def peers_of(x, group, peers, window_days):
    d = to_date(x['rec']['date'])
    out = []
    for y in group:
        if y is x:
            continue
        dy = to_date(y['rec']['date'])
        if peers == 'prior' and (dy >= d or (window_days is not None and (d - dy).days > window_days)):
            continue
        out.append(y)
    return out


def novelty_coverage(own, peer_sets):
    """own 中没有任何同行有的比例；半数以上同行都有的项中 own 覆盖的比例。"""
    if not peer_sets:
        return NAN, NAN
    cnt = Counter(k for s in peer_sets for k in s)
    cons = {k for k, n in cnt.items() if n >= len(peer_sets) / 2}
    return (sum(k not in cnt for k in own) / len(own) if own else NAN,
            len(cons & own) / len(cons) if cons else NAN)


def reach_agree(x, ps):
    """两端本篇都提到、且本篇或某个同行能走到的（主题，主题或科目）对中，两端都提到的同行与本篇可达与否一致的平均比例。"""
    rates = []
    for t in sorted(x['themes']):
        for end in sorted(x['ends']):
            if end == ('T', t):
                continue
            key = (end[0], t, end[1])
            voters = [y for y in ps if ('T', t) in y['ends'] and end in y['ends']]
            mine = key in x['links']
            if voters and (mine or any(key in y['links'] for y in voters)):
                rates.append(mean((key in y['links']) == mine for y in voters))
    return mean(rates)


def report_metrics(x, group, theme_names, peers, window_days):
    """一篇研报的逐篇指标：体量、形态，以及与同行比较的偏离度。theme_names 为这家公司的全部主题。"""
    rec, ps = x['rec'], peers_of(x, group, peers, window_days)
    theme_nov, theme_cov = novelty_coverage(x['themes'], [y['themes'] for y in ps])
    link_nov, link_cov = novelty_coverage(x['links'], [y['links'] for y in ps])
    share = lambda y: np.array([y['theme_w'].get(t, 0) for t in theme_names], float) / max(sum(y['theme_w'].values()), 1)
    conflicts = comparable = 0
    for c, e in x['c_effect'].items():
        votes = [y['c_effect'][c] for y in ps if y['c_effect'].get(c)]
        if e and votes:
            comparable += 1
            conflicts += sum(v == -e for v in votes) > sum(v == e for v in votes)
    return {'report_id': rec['report_id'], 'company': rec['company'], 'stock_code': rec['stock_code'], 'broker': rec['broker'],
            'date': rec['date'], 'n_paragraphs': rec['n_paragraphs'], 'n_concepts': len(x['concepts']),
            'n_edges': x['g'].number_of_edges(), 'n_themes': len(x['themes']), 'n_accounting': len(x['acc']),
            'n_peers': len(ps), **x['shape'],
            'theme_novelty': theme_nov, 'theme_coverage': theme_cov,
            'theme_js': js(share(x), np.mean([share(y) for y in ps], axis=0)) if ps else NAN,
            'link_novelty': link_nov, 'link_coverage': link_cov, 'reach_agree': reach_agree(x, ps),
            'sign_conflict_n': conflicts, 'sign_conflict': conflicts / comparable if comparable else NAN,
            'dist_theme_mean': mean(1 - jaccard(x['themes'], y['themes']) for y in ps),
            'dist_link_mean': mean(1 - jaccard(x['links'], y['links']) for y in ps)}


# ==================== 主流程 ====================

def main(graphs_path, out_dir, edge_set='all', peers='all', window_days=None):
    """graphs_path：report_logic 输出的 report_graphs.json；edge_set：'all' 全部边 / 'text' 去掉逻辑补全边；
    peers：'all' 同公司其他全部研报 / 'prior' 只用更早发布的（window_days 限定往前多少天，None 为不限）。"""
    if edge_set not in ('all', 'text') or peers not in ('all', 'prior'):
        raise SystemExit(f'edge_set 只能是 all 或 text，peers 只能是 all 或 prior，收到 {edge_set!r}、{peers!r}')
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_log_path(out_dir / '_progress.log')
    env = load_env()
    ctx = Ctx(env, out_dir, {})
    data = json.loads(Path(graphs_path).read_text(encoding='utf-8'))
    companies = defaultdict(list)
    for rec in sorted(data['reports'], key=lambda r: (r['date'], r['report_id'])):
        companies[rec['stock_code'] or rec['company']].append(rec)
    log(f'===== 框架对比：{len(data["reports"])} 篇研报、{len(companies)} 家公司；边 {edge_set}，同行 {peers}'
        + (f'（往前 {window_days} 天）' if peers == 'prior' and window_days is not None else '') + ' =====')
    themes = assign_themes(ctx, {k: (recs[0]['company'], recs) for k, recs in companies.items()})

    theme_rows, matrix, pairs, reports = [], [], [], []
    for key in sorted(companies):
        recs, th = companies[key], themes[key]
        names = sorted({t['name'] for t in th['themes']})
        group = [report_data(rec, edge_set, th['of'], len(names)) for rec in recs]
        theme_rows += [{'company': recs[0]['company'], 'stock_code': recs[0]['stock_code'], 'theme': t['name'],
                        'theme_definition': t['definition'], 'concept': c, 'layer': th['concepts'][c]['layer'],
                        'n_reports': th['concepts'][c]['n_reports']} for t in th['themes'] for c in t['concepts']]
        for x in group:
            matrix += matrix_rows(x)
        company_pairs = [pair_metrics(a, b) for a, b in itertools.combinations(group, 2)]
        company_reports = [report_metrics(x, group, names, peers, window_days) for x in group]
        pairs += company_pairs
        reports += company_reports
        log(f'== {recs[0]["company"]}：{len(recs)} 篇，{len(th["concepts"])} 个概念归成 {len(names)} 个主题；两两平均 '
            f'概念 Jaccard {mean(p["concept_jaccard"] for p in company_pairs):.2f}，主题 Jaccard '
            f'{mean(p["theme_jaccard"] for p in company_pairs):.2f}，主题可达 Jaccard {mean(p["theme_reach_jaccard"] for p in company_pairs):.2f}')
        log('   券商 日期 | 概念 主题 | 主题新颖 共识覆盖 | 连接新颖 连接覆盖 | 可达一致 | 平均主题距离')
        for m in company_reports:
            log(f'   {m["broker"]} {m["date"]} | {m["n_concepts"]} {m["n_themes"]} | {m["theme_novelty"]:.2f} {m["theme_coverage"]:.2f} | '
                f'{m["link_novelty"]:.2f} {m["link_coverage"]:.2f} | {m["reach_agree"]:.2f} | {m["dist_theme_mean"]:.2f}')

    write_parquet(theme_rows, out_dir / 'themes.parquet', ['company', 'stock_code', 'theme', 'theme_definition', 'concept'])
    write_parquet(matrix, out_dir / 'matrices.parquet', ['report_id', 'level', 'kind', 'row', 'col'])
    write_parquet(pairs, out_dir / 'pair_metrics.parquet', ['company', 'report_a', 'report_b'])
    write_parquet(reports, out_dir / 'report_metrics.parquet', ['report_id', 'company'])
    u = ctx.llm.usage
    log(f'===== 结束：{len(reports)} 篇、{len(pairs)} 对；矩阵 {len(matrix)} 行；调用 {u["calls"]} 次（不合格重试 {u["retries"]} 次），'
        f'输入 {u["prompt_tokens"]} tokens，输出 {u["completion_tokens"]} tokens =====')
