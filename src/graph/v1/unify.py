"""抽取版本 v1 的跨研报概念名称统一：概念 DAG 抽完之后（本目录的 extract.py），把不同研报里指完全相同的东西、只是说法不同的概念名称
统一成一个，方便在研报之间比较概念和边。只在细节层面合并同一个量的不同叫法（营收 / 营业收入，PE / 市盈率），
范围、口径、主体、指标不同的，有因果或上下位关系的，都不合并。提示词见本目录的 prompts/unify.md，
入口是 python run.py unify --version v1（命令行选项在 src/graph/unify_cmd.py，选好版本后调用这里的 main，对整个库统一）。

按公司分别统一：同一个说法在不同公司的研报里可能指不同的东西，所以只在同一家公司的研报之间统一，每家公司各有一套词条、
候选组和统一结果；只有 1 篇研报的公司没有可统一的，概念名保持原样。

流程：
  1. 词条：一家公司各篇概念表里的规范名按字面键（name_key：忽略空格、括号、大小写）归并，字面相同的本来就是同一个名称；
     每个词条带上各篇给的别名、定义、出现的篇数。
  2. 候选组（不调用大模型）：同一家公司的两个词条共用一个名称或别名，或者名称的两字片段重合得多（Dice ≥ MIN_DICE），
     记为候选对；候选对按相似度从高到低做单链接聚类，每组不超过 max_group 个词条（超过时不再并入），只有一个词条的组不送模型。
  3. 统一：每个候选组调用一次大模型，在组内找出完全相同的名称归为一组，并从组内名称里选一个作统一名称。
     组和组之间不再合并（候选组互不相交，结果不会互相矛盾）；某组调用失败时这组保持原样，下次运行重试。
  4. 复核：每个合并组再调用一次，逐个判断组内其他名称和统一名称是不是完全相同，判为不同的剔除，剩不到两个名称的组取消。
     第 3 步一次要看几十个名称、容易放宽；复核只看一组，专门挑差别，只会剔除不会新增，用来保证“只合并明显相同的”。
     复核调用失败的组整组不合并（宁可少合并）。
  5. 输出：每篇研报每个概念的统一名称、合并组的清单（含复核剔除的名称和理由）、按统一名称改写后的边表、核对页面。

调度：第 3、4 步放进同一个线程池流水推进（judge_all）：所有候选组排队统一，某组一统一完，它提出的合并组马上排进去复核，
不等其他候选组、也不分公司分批；线程池里始终有 workers 个调用在途，直到最后剩下的不足 workers 个
（日志里每分钟记一行在途调用数）。

调用缓存在 <库>/cache/unify_cache.jsonl、unify_verify_cache.jsonl（键含提示词指纹和这一组的全部输入），
重跑时没变的组直接取结果；概念 DAG 新增了研报、候选组变了的，只重做变了的组。

输出（<库>，即 output/graph/v1/）：
  概念统一.html                   核对页面（模板本目录的 unify_review.html）：合并了哪些名称、各出现在哪些研报、理由，复核剔除了哪些
  data/unify.js                   核对页面和 概念DAG.html 读取的数据（各篇改了名的概念 -> 统一名称、合并组、复核剔除、统计）
  data/concept_unify.parquet      每篇研报每个概念一行：name 原名、unified 统一名称、changed 是否改了名
  data/unify_groups.parquet       每个合并组一行：统一名称、组内原名、涉及的研报和公司数、理由、复核剔除的名称
  data/unified_edges.parquet      DAG 的边按统一名称改写：同一篇里改写后重复的边合并（support 相加），两端变成同一个
                                  概念的边去掉；改写后不保证仍然无环（同一篇里两个概念合并后可能连成环）
"""
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pyarrow.parquet as pq

from src.llm import LLM, Cache, FatalAPIError, InflightMeter, load_env, log, parse_json, set_log_path, text_hash, with_feedback

from .common import VIEWER_TEMPLATE, DagOut, fill, merge_direction, name_key, too_many, write_js, write_parquet

PROMPT_FILE = Path(__file__).resolve().parent / 'prompts' / 'unify.md'
REVIEW_TEMPLATE = Path(__file__).resolve().parent / 'unify_review.html'
SECTIONS = ('判断标准', '统一-System', '统一-User', '复核-System', '复核-User')
MIN_DICE = 0.5  # 名称两字片段的 Dice 系数达到这个值才算候选对（成本优势 / 成本领先优势 = 0.5）
ALIAS_LINK = 0.9  # 共用名称或别名的候选对的相似度（排在大多数字面相似的前面并组）
MAX_GROUP = 40  # 每个候选组最多这么多个词条
MAX_COMMON_GRAM = 300  # 出现在超过这么多个名称里的两字片段不用来找候选（只为省时间：这样的片段本身也撑不起 Dice ≥ 0.5）
MAX_ALIASES_SHOWN = 4
MAX_DEF_SHOWN = 50
MAX_REASON = 30
MAX_REPORTS_SHOWN = 12  # 核对页面上每个原名最多列出这么多篇研报


def load_prompts():
    """按二级标题切分 prompt 文件，返回 {标题: 内容}；两步的 System 里的 {criteria} 填入“判断标准”一节。"""
    out, cur = {}, None
    for line in PROMPT_FILE.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            cur = line[3:].strip()
            out[cur] = []
        elif cur:
            out[cur].append(line)
    out = {k: '\n'.join(v).strip() for k, v in out.items()}
    lack = [k for k in SECTIONS if k not in out]
    if lack:
        raise SystemExit(f'{PROMPT_FILE} 中缺少这些节：{"、".join(lack)}')
    for k in ('统一-System', '复核-System'):
        out[k] = fill(out[k], criteria=out['判断标准'])
    return out


# ==================== 1. 词条 ====================

def split_aliases(s):
    return [a.strip() for a in (s or '').split('、') if a.strip()]


def build_terms(concepts):
    """concepts.parquet 的行 -> {字面键: 词条}。词条：name 显示名（各篇最常用的写法）、aliases 别名计数、definitions 定义
    （按出现次数排序）、kinds 类型计数、reports 出现的研报、codes 各股票的篇数。"""
    terms = {}
    for c in concepts:
        k = name_key(c['name'])
        t = terms.setdefault(k, {'key': k, 'names': Counter(), 'aliases': Counter(), 'definitions': Counter(),
                                 'kinds': Counter(), 'reports': set(), 'codes': Counter()})
        t['names'][c['name']] += 1
        t['aliases'].update(a for a in split_aliases(c['aliases']) if name_key(a) != k)
        if c['definition']:
            t['definitions'][c['definition']] += 1
        t['kinds'][c['kind']] += 1
        if c['report_id'] not in t['reports']:
            t['reports'].add(c['report_id'])
            t['codes'][c['stock_code']] += 1
    for t in terms.values():
        t['name'] = t['names'].most_common(1)[0][0]
    return terms


# ==================== 2. 候选组 ====================

def grams(k):
    """名称字面键的两字片段（只有一个字的名称取它本身）。"""
    return {k[i:i + 2] for i in range(len(k) - 1)} or {k}


def candidate_pairs(terms):
    """候选对 {(键1, 键2): 相似度}，键1 < 键2：两字片段 Dice ≥ MIN_DICE，或共用一个名称或别名（ALIAS_LINK）。"""
    g = {k: grams(k) for k in terms}
    inv = defaultdict(set)
    for k, gs in g.items():
        for x in gs:
            inv[x].add(k)
    pairs = {}
    for k, gs in g.items():
        near = set().union(*(inv[x] for x in gs if len(inv[x]) <= MAX_COMMON_GRAM))
        for o in near:
            if o > k:
                d = 2 * len(gs & g[o]) / (len(gs) + len(g[o]))
                if d >= MIN_DICE:
                    pairs[(k, o)] = d
    owners = defaultdict(set)  # 名称或别名的字面键 -> 用它作名称或别名的词条
    for k, t in terms.items():
        owners[k].add(k)
        for a in t['aliases']:
            if len(name_key(a)) >= 2:
                owners[name_key(a)].add(k)
    for ks in owners.values():
        ks = sorted(ks)
        for i, a in enumerate(ks):
            for b in ks[i + 1:]:
                pairs[(a, b)] = max(pairs.get((a, b), 0), ALIAS_LINK)
    return pairs


def candidate_groups(terms, pairs, max_group=MAX_GROUP):
    """候选对按相似度从高到低做单链接聚类，并组后超过 max_group 个词条的不并；返回两个词条以上的组（组内按键排序）。"""
    parent, size = {k: k for k in terms}, {k: 1 for k in terms}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for (a, b), _ in sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0])):
        ra, rb = find(a), find(b)
        if ra != rb and size[ra] + size[rb] <= max_group:
            if size[ra] < size[rb]:
                ra, rb = rb, ra
            parent[rb] = ra
            size[ra] += size[rb]
    groups = defaultdict(list)
    for k in terms:
        groups[find(k)].append(k)
    return sorted((sorted(g) for g in groups.values() if len(g) > 1), key=lambda g: (-len(g), g))


# ==================== 3. 判定 ====================

def block_items(block, terms, company):
    """一个候选组给模型看的列表：编号 ｜ 名称 ｜ 其他叫法 ｜ 定义 ｜ 出现的公司和篇数。"""
    lines = []
    for i, k in enumerate(block, 1):
        t = terms[k]
        aliases = [a for a, _ in t['aliases'].most_common(MAX_ALIASES_SHOWN)]
        definition = t['definitions'].most_common(1)[0][0][:MAX_DEF_SHOWN] if t['definitions'] else '—'
        where = '、'.join(f'{company.get(c, c)} {n} 篇' for c, n in t['codes'].most_common(3))
        if len(t['codes']) > 3:
            where += f' 等 {len(t["codes"])} 家'
        lines.append(f'{i} ｜ {t["name"]} ｜ {"、".join(aliases) or "—"} ｜ {definition} ｜ {where}')
    return '\n'.join(lines)


def parse_groups(content, names, strict=True):
    """模型回复 -> [{'ids': [编号...], 'name': 统一名称的编号, 'reason': 理由}]。不合格的组（编号越界、不足两个、
    一个编号出现在几组里、统一名称不在组内）丢掉；strict 时丢掉的太多就抛出 ValueError 让模型重做。"""
    groups = parse_json(content, dict).get('groups')
    if not isinstance(groups, list):
        raise ValueError('缺少 groups 列表')
    n, used, good, errs = len(names), set(), [], []
    for g in groups:
        try:
            ids = list(dict.fromkeys(int(x) for x in g['ids']))
            if len(ids) < 2:
                raise ValueError(f'组 {ids} 不足两个编号')
            if any(not 1 <= x <= n for x in ids):
                raise ValueError(f'组 {ids} 里有不存在的编号（只有 1-{n}）')
            if used & set(ids):
                raise ValueError(f'编号 {sorted(used & set(ids))} 出现在不止一组里')
            keep = g['name']
            keep = int(keep) if str(keep).strip().isdigit() else names.index(str(keep).strip()) + 1
            if keep not in ids:
                raise ValueError(f'组 {ids} 的统一名称编号 {keep} 不在组内')
        except (KeyError, TypeError, ValueError) as e:
            errs.append(str(e) if not isinstance(e, KeyError) else f'某组缺少字段 {e}')
            continue
        used.update(ids)
        good.append({'ids': ids, 'name': keep, 'reason': str(g.get('reason') or '').strip()[:MAX_REASON]})
    if strict and errs and too_many(errs, len(groups)):
        raise ValueError('；'.join(errs[:5]))
    return good


def parse_verify(content, n, keep, strict=True):
    """复核回复 -> {编号: (是否相同, 理由)}，覆盖除统一名称 keep 以外的 1..n。strict 时缺了编号就抛出 ValueError 让模型重做；
    否则缺的按“不同”处理（宁可少合并）。"""
    results = parse_json(content, dict).get('results')
    if not isinstance(results, list):
        raise ValueError('缺少 results 列表')
    out = {}
    for x in results:
        try:
            i = int(x['id'])
        except (KeyError, TypeError, ValueError):
            continue
        if 1 <= i <= n and i != keep:
            same = x.get('same')
            out[i] = (same is True or str(same).strip().lower() in ('true', '是', '相同'), str(x.get('reason') or '').strip()[:MAX_REASON])
    lack = [i for i in range(1, n + 1) if i != keep and i not in out]
    if lack:
        if strict:
            raise ValueError(f'缺少编号 {lack} 的复核结果')
        out.update({i: (False, '复核没有给出结果') for i in lack})
    return out


class Judge:
    """两步判定共用的大模型、prompt、缓存。
    terms：{股票代码: 这家公司的词条}；
    propose((股票代码, 候选组)) -> [{'code': 股票代码, 'keys': [词条键...], 'keep': 统一名称的键, 'reason': 理由}]；
    verify(合并组) -> 复核后的合并组（多了 'dropped': [[键, 理由]...]），剩不到两个名称时 'keys' 只剩统一名称。"""

    def __init__(self, env, out, terms, company):
        self.llm, self.prompts = LLM(env), load_prompts()
        self.sig = {s: text_hash(self.prompts[f'{s}-System'], self.prompts[f'{s}-User']) for s in ('统一', '复核')}
        self.cache = {'统一': Cache(out.cache / 'unify_cache.jsonl'), '复核': Cache(out.cache / 'unify_verify_cache.jsonl')}
        self.terms, self.company = terms, company
        self.lock, self.retry_errors, self.hits = threading.Lock(), [], Counter()

    def ask(self, step, key, parse, lenient, label, extra, **user_kw):
        hit = self.cache[step].get(key)
        if hit is not None:
            with self.lock:
                self.hits[step] += 1
            return hit

        def checked(content):
            try:
                return parse(content)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                with self.lock:
                    self.retry_errors.append(f'{step}：{str(e)[:200]}')
                log(f'!!! {step}回复不合格（{label}），重做：{str(e)[:150]}')
                raise

        res = self.llm.ask(self.prompts[f'{step}-System'], with_feedback(fill(self.prompts[f'{step}-User'], **user_kw)),
                           checked, lenient)
        res = extra(res)
        self.cache[step].put(key, res, label=label)
        return res

    def propose(self, item):
        code, block = item
        terms = self.terms[code]
        items = block_items(block, terms, self.company)
        names = [terms[k]['name'] for k in block]
        res = self.ask('统一', text_hash(self.sig['统一'], items), lambda c: parse_groups(c, names),
                       lambda c: parse_groups(c, names, strict=False), f'{self.company.get(code, code)} {len(block)} 个名称的组，首个“{names[0]}”',
                       lambda res: [{'keys': [block[i - 1] for i in g['ids']], 'keep': block[g['name'] - 1],
                                     'reason': g['reason']} for g in res],
                       n=len(block), items=items)
        return [{**g, 'code': code} for g in res]  # 缓存里不存股票代码（候选组的输入里已有公司名）

    def verify(self, group):
        terms = self.terms[group['code']]
        keys = sorted(group['keys'], key=lambda k: (k != group['keep'], k))  # 统一名称排第一个
        items = block_items(keys, terms, self.company)
        name = terms[group['keep']]['name']
        res = self.ask('复核', text_hash(self.sig['复核'], name, items), lambda c: parse_verify(c, len(keys), 1),
                       lambda c: parse_verify(c, len(keys), 1, strict=False), f'统一名称“{name}”',
                       lambda r: {str(i): [same, reason] for i, (same, reason) in r.items()},
                       name=name, keep=1, items=items)
        kept = [keys[0]] + [k for i, k in enumerate(keys[1:], 2) if res[str(i)][0]]
        dropped = [[k, res[str(i)][1]] for i, k in enumerate(keys[1:], 2) if not res[str(i)][0]]
        return {**group, 'keys': kept, 'dropped': dropped}


def judge_all(judge, blocks, workers):
    """统一和复核放进同一个线程池流水推进：某个候选组一统一完，它提出的合并组马上排进去复核，不等其他候选组，
    线程池里始终有 workers 个调用在途（最后剩下的不足 workers 个时除外）。单项失败记为 None，不影响其他项；致命错误停下。
    返回 (proposals：各候选组提出的合并组列表，失败为 None；verdicts：{(候选组序号, 组内序号): 复核后的组，失败为 None})。"""
    t0, lock = time.time(), threading.Lock()
    proposals, verify_futs = [None] * len(blocks), {}
    meter = InflightMeter(workers)
    judge.llm.client.chat = meter.wrap(judge.llm.client.chat)
    pool = ThreadPoolExecutor(max_workers=max(1, workers))

    def propose(i):
        res = judge.propose(blocks[i])
        with lock:  # 提出的合并组马上排进复核
            for j, g in enumerate(res):
                verify_futs[(i, j)] = pool.submit(judge.verify, g)
        return res

    def queued():
        with lock:
            fs = list(verify_futs.values())
        return f'；复核已排入 {len(fs)} 组、做完 {sum(f.done() for f in fs)} 组'

    def collect(futs, label, total, store, extra=lambda: ''):
        for k, fut in enumerate(as_completed(futs), 1):
            key = futs[fut]
            try:
                store[key] = fut.result()
            except FatalAPIError as e:
                raise SystemExit(f'致命错误，已停止（已完成的部分保存在缓存中，处理后重新运行即可继续）：{e}')
            except Exception as e:  # 单项失败不影响其他项
                store[key] = None
                log(f'{label} {key} 失败：{e}')
            if k % max(1, total // 10) == 0 or k == total:
                log(f'{label}：{k}/{total}，用时 {time.time() - t0:.0f} 秒{extra()}')

    try:
        prop_store = {}
        collect({pool.submit(propose, i): i for i in range(len(blocks))}, '统一', len(blocks), prop_store, queued)
        for i, res in prop_store.items():
            proposals[i] = res
        verdicts = {}
        with lock:  # 统一全部做完以后不会再新增复核
            futs = {f: k for k, f in verify_futs.items()}
        collect(futs, '复核', len(futs), verdicts)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        meter.close()
    return proposals, verdicts


# ==================== 5. 输出 ====================

def report_label(r):
    return f'{r["company"]} {r["date"]} {r["broker"]}'


def shared_counts(rows, key, code_of):
    """在同一家公司 2 篇以上研报里出现的“东西”有多少个（rows 里每行一篇研报的一个东西，key(行) 取它）。"""
    seen = defaultdict(set)
    for x in rows:
        seen[(code_of[x['report_id']], key(x))].add(x['report_id'])
    return sum(len(v) > 1 for v in seen.values())


def unify_edges(edges, to_unified):
    """DAG 的边（in_dag 为真的）按统一名称改写：同一篇里改写后重复的合并，两端相同的去掉。返回 (新边表, 去掉的自环数)。"""
    agg, loops = {}, 0
    for e in edges:
        if not e['in_dag']:
            continue
        a, b = to_unified(e['report_id'], e['cause']), to_unified(e['report_id'], e['effect'])
        if a == b:
            loops += 1
            continue
        x = agg.setdefault((e['report_id'], a, b), {'report_id': e['report_id'], 'stock_code': e['stock_code'], 'cause': a,
                                                    'effect': b, 'directions': [], 'support': 0, 'raw': [], 'evidence': []})
        x['directions'].append(e['direction'])
        x['support'] += e['support']
        x['raw'].append(f'{e["cause"]} → {e["effect"]}')
        x['evidence'].append(e['evidence'])
    rows = [{**{k: x[k] for k in ('report_id', 'stock_code', 'cause', 'effect', 'support')},
             'direction': merge_direction(x['directions']), 'raw_edges': ' ｜ '.join(x['raw']),
             'evidence': ' ｜ '.join(x['evidence'])} for x in agg.values()]
    return rows, loops


def write_review_page(out):
    """写出核对页面，顺带用最新的模板刷新 概念DAG.html（它会读 data/unify.js，多出“统一名称”开关）。"""
    page = out.root / '概念统一.html'
    page.write_text(REVIEW_TEMPLATE.read_text(encoding='utf-8'), encoding='utf-8')
    shutil.copyfile(VIEWER_TEMPLATE, out.viewer)
    return page


def main(out_dir, companies=None, workers=50, max_group=MAX_GROUP, dry_run=False, sample=8):
    """companies：只统一这些股票代码的研报（其余研报不读、不进输出）；None 或空表示概念 DAG 里的全部研报。"""
    t0 = time.time()
    out = DagOut(out_dir)
    paths = {k: out.data / f'{k}.parquet' for k in ('concepts', 'edges', 'reports')}
    if not all(p.exists() for p in paths.values()):
        raise SystemExit(f'{out.data} 下缺少概念 DAG 的结果表，先运行 python run.py graph')
    concepts, edges, reports = (pq.read_table(paths[k]).to_pylist() for k in ('concepts', 'edges', 'reports'))
    if companies:
        concepts, edges, reports = ([x for x in rows if x['stock_code'] in companies] for rows in (concepts, edges, reports))
        if not reports:
            raise SystemExit(f'概念 DAG 里没有这些股票的研报：{"、".join(companies)}')
    company = {r['stock_code']: r['company'] for r in reports}
    rep = {r['report_id']: r for r in reports}
    code_of = {r['report_id']: r['stock_code'] for r in reports}
    n_rep = Counter(code_of.values())
    by_code = defaultdict(list)
    for c in concepts:
        by_code[c['stock_code']].append(c)
    terms = {code: build_terms(cs) for code, cs in by_code.items()}  # 每家公司各自的词条
    n_pairs, blocks = 0, []
    for code, ts in terms.items():
        if n_rep[code] > 1:  # 只有 1 篇研报的公司没有可统一的
            pairs = candidate_pairs(ts)
            n_pairs += len(pairs)
            blocks += [(code, b) for b in candidate_groups(ts, pairs, max_group)]
    blocks.sort(key=lambda x: (-len(x[1]), x))
    n_terms = sum(map(len, terms.values()))
    n_in = sum(len(b) for _, b in blocks)
    multi = [c for c in company if n_rep[c] > 1]
    summary = (f'{len(reports)} 篇研报、{len(company)} 家公司（2 篇以上、参与统一的 {len(multi)} 家 {sum(n_rep[c] for c in multi)} 篇），'
               f'概念 {len(concepts)} 个、各公司内不同名称合计 {n_terms} 个；候选对 {n_pairs} 个，分成 {len(blocks)} 个候选组'
               f'（共 {n_in} 个名称，最大的组 {max((len(b) for _, b in blocks), default=0)} 个），其余 {n_terms - n_in} 个名称'
               f'没有相近的或所在公司只有 1 篇，不送模型')
    if dry_run:
        print(summary)
        sizes = Counter(min(len(b), max_group) for _, b in blocks)
        print('候选组大小分布：', '，'.join(f'{s} 个 {n} 组' for s, n in sorted(sizes.items())))
        for code, b in blocks[:: max(1, len(blocks) // sample)][:sample]:
            print(f'  [{len(b)}] {company.get(code, code)}：' + '、'.join(terms[code][k]['name'] for k in b[:30])
                  + (' ……' if len(b) > 30 else ''))
        print('只做了统计，没有调用大模型；去掉 --dry-run 才开始判定')
        return

    out.make()
    set_log_path(out.cache / 'unify.log')
    env = load_env()
    judge = Judge(env, out, terms, company)
    log(f'===== 概念名称统一开始：模型 {env["MODEL"]}，{summary}；并发 {workers} =====')
    proposals, verdicts = judge_all(judge, blocks, workers)
    failed = sum(r is None for r in proposals)
    order = sorted(verdicts)  # (候选组序号, 组内序号)：结果顺序和完成先后无关
    proposed = [proposals[i][j] for i, j in order]
    verified = [verdicts[k] for k in order]
    failed_verify = sum(v is None for v in verified)
    rejected = [v for v in verified if v and len(v['keys']) < 2]  # 复核后剩不到两个名称、取消的组

    def pick_keep(g):
        """统一名称：组内名称复核后都和第一步选的那个相同，取研报里最常用的（出现篇数最多），带本公司名的排后，
        一样多时用第一步模型的选择（模型常选不常用的说法，如 5 篇用“销量”时选了只有 1 篇的“公司总销量”）。"""
        own_name = company.get(g['code'])

        def rank(k):
            t = terms[g['code']][k]
            return bool(own_name and own_name in t['name']), -len(t['reports']), k != g['keep'], k
        return min(g['keys'], key=rank)

    groups = [{**v, 'keep': pick_keep(v)} for v in verified if v and len(v['keys']) > 1]
    canon = {}  # (股票代码, 词条键) -> 统一名称的词条键；不在里面的保持原样
    for g in groups:
        for k in g['keys']:
            canon[(g['code'], k)] = g['keep']

    def to_unified(rid, name):
        code, k = code_of[rid], name_key(name)
        return terms[code][canon.get((code, k), k)]['name']

    # 每篇每个概念的统一名称
    map_rows, changes, collide = [], defaultdict(dict), 0
    by_report = defaultdict(Counter)
    for c in concepts:
        u = to_unified(c['report_id'], c['name'])
        changed = u != c['name']
        map_rows.append({'report_id': c['report_id'], 'stock_code': c['stock_code'], 'name': c['name'], 'unified': u,
                         'changed': changed, 'in_dag': c['in_dag']})
        if changed:
            changes[c['report_id']][c['name']] = u
        by_report[c['report_id']][u] += 1
    collide = sum(n - 1 for cnt in by_report.values() for n in cnt.values() if n > 1)
    write_parquet(map_rows, out.data / 'concept_unify.parquet',
                  ['report_id', 'stock_code', 'name', 'unified', 'changed', 'in_dag'])

    # 合并组（复核剔除的名称一并列出，复核后取消的组单独列出）
    def member(code, k, reason=None):
        t = terms[code][k]
        m = {'name': t['name'], 'n': len(t['reports']),
             'definition': t['definitions'].most_common(1)[0][0] if t['definitions'] else '',
             'reports': sorted(report_label(rep[r]) for r in t['reports'] if r in rep)[:MAX_REPORTS_SHOWN]}
        return m if reason is None else {**m, 'reason': reason}

    def group_reports(g):
        return set().union(*(terms[g['code']][k]['reports'] for k in g['keys']))

    group_rows, page_groups = [], []
    for gid, g in enumerate(sorted(groups, key=lambda g: (-len(group_reports(g)), g['code'], terms[g['code']][g['keep']]['name'])), 1):
        ts, code = terms[g['code']], g['code']
        members = sorted(g['keys'], key=lambda k: (k != g['keep'], -len(ts[k]['reports']), k))
        rids = group_reports(g)
        group_rows.append({'group_id': gid, 'stock_code': code, 'company': company.get(code, code),
                           'unified': ts[g['keep']]['name'], 'members': '、'.join(ts[k]['name'] for k in members),
                           'n_members': len(members), 'n_reports': len(rids), 'reason': g['reason'],
                           'dropped': '、'.join(f'{ts[k]["name"]}（{why}）' for k, why in g['dropped'])})
        page_groups.append({'id': gid, 'name': ts[g['keep']]['name'], 'reason': g['reason'], 'n_reports': len(rids),
                            'companies': [company.get(code, code)],
                            'members': [member(code, k) for k in members],
                            'dropped': [member(code, k, why) for k, why in g['dropped']]})
    page_rejected = [{'name': terms[g['code']][g['keep']]['name'], 'reason': g['reason'], 'company': company.get(g['code'], g['code']),
                      'keep': member(g['code'], g['keep']), 'dropped': [member(g['code'], k, why) for k, why in g['dropped']]}
                     for g in sorted(rejected, key=lambda g: (g['code'], terms[g['code']][g['keep']]['name']))]
    write_parquet(group_rows, out.data / 'unify_groups.parquet',
                  ['group_id', 'stock_code', 'company', 'unified', 'members', 'n_members', 'n_reports', 'reason', 'dropped'])

    # 边
    new_edges, loops = unify_edges(edges, to_unified)
    write_parquet(new_edges, out.data / 'unified_edges.parquet',
                  ['report_id', 'stock_code', 'cause', 'effect', 'direction', 'support', 'raw_edges', 'evidence'])

    # 统计：统一前后，同一家公司 2 篇以上研报共有的概念、边
    dag_concepts = [x for x in map_rows if x['in_dag']]
    dag_edges = [e for e in edges if e['in_dag']]
    n_after = sum(len({canon.get((code, k), k) for k in ts}) for code, ts in terms.items())  # 各公司内不同名称合计
    stats = {
        'n_reports': len(reports), 'n_companies': len(company), 'n_multi_companies': len(multi), 'n_concepts': len(concepts),
        'n_terms': n_terms, 'n_terms_after': n_after, 'n_blocks': len(blocks), 'n_blocked_terms': n_in,
        'n_failed_blocks': failed, 'n_proposed': len(proposed), 'n_failed_verify': failed_verify,
        'n_rejected': len(rejected), 'n_dropped': sum(len(g['dropped']) for g in verified if g),
        'n_groups': len(groups), 'n_merged_terms': sum(len(g['keys']) for g in groups),
        'n_changed': sum(x['changed'] for x in map_rows),
        'n_changed_dag': sum(x['changed'] for x in dag_concepts), 'n_dag_concepts': len(dag_concepts),
        'n_collide': collide, 'n_loops': loops, 'n_dag_edges': len(dag_edges), 'n_edges_after': len(new_edges),
        'shared_concepts_before': shared_counts(dag_concepts, lambda x: name_key(x['name']), code_of),
        'shared_concepts_after': shared_counts(dag_concepts, lambda x: x['unified'], code_of),
        'shared_edges_before': shared_counts(dag_edges, lambda e: (name_key(e['cause']), name_key(e['effect'])), code_of),
        'shared_edges_after': shared_counts(new_edges, lambda e: (e['cause'], e['effect']), code_of),
    }
    write_js(out.data / 'unify.js', 'dagUnify',
             {'model': env['MODEL'], 'generated': time.strftime('%Y-%m-%d %H:%M'), 'stats': stats,
              'groups': page_groups, 'rejected': page_rejected, 'map': changes})
    page = write_review_page(out)
    u = judge.llm.usage
    log(f'===== 结束：{len(blocks)} 个候选组（缓存命中 {judge.hits["统一"]} 组，失败 {failed} 组）提出 {len(proposed)} 个合并组；'
        f'复核（缓存命中 {judge.hits["复核"]} 组，失败 {failed_verify} 组）剔除 {stats["n_dropped"]} 个名称、取消 {len(rejected)} 组，'
        f'最终合并 {len(groups)} 组、涉及 {stats["n_merged_terms"]} 个名称；各公司内不同名称合计 {n_terms} -> {n_after} 个，改了名的概念 {stats["n_changed"]} 个'
        f'（进图的 {stats["n_changed_dag"]}/{len(dag_concepts)} 个）；同一篇里合并到同一名称的 {collide} 处，改写后变成自环去掉的边 '
        f'{loops} 条；同公司 2 篇以上共有的进图概念 {stats["shared_concepts_before"]} -> {stats["shared_concepts_after"]} 个、'
        f'边 {stats["shared_edges_before"]} -> {stats["shared_edges_after"]} 条；本次调用 {u["calls"]} 次（不合格重试 '
        f'{u["retries"]} 次），输入 {u["prompt_tokens"]} tokens、输出 {u["completion_tokens"]} tokens；用时 {time.time() - t0:.0f} 秒 =====')
    if judge.retry_errors:
        for err, n in Counter(e[:80] for e in judge.retry_errors).most_common(5):
            log(f'    不合格原因 {n} 次：{err}')
    log(f'核对页面：{page}（双击打开）；概念DAG.html 里勾选“统一名称”可按统一名称显示')


# ==================== 默认运行参数（入口 src/graph/unify_cmd.py 用） ====================
WORKERS = 50  # 同时在途的大模型调用数
# MAX_GROUP 见上面的常量：每个候选组最多这么多个名称（太大时模型一次要比较的名称多、容易漏，太小时相近的名称可能分到不同组）
