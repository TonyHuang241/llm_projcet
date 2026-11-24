"""研报概念有向无环图（DAG）抽取：和 src/report_logic.py 的五层估值逻辑图是两套独立的流程，输入相同、输出另放。
提示词见 src/prompts/concept_dag_prompt.md，入口脚本是仓库根目录的 extract_concept_dag.py。

每篇研报只走两步，不预设会计科目、估值等固定概念，也不做任何补全：
  0. 合并段落：clear_txt.py 清理出的正文段落按顺序合并成不超过 chunk_chars 字的块（单个段落超长的自成一块），
     后面两步都按块调用大模型，调用次数大致是“块数 × 2”，多数研报只有一块。
  1. 抽概念（概念）：按块顺序通读、滚动维护“本篇概念表”（规范名、别名、定义、类型）。每个新概念要给出原文中的叫法，
     程序逐字核对，对不上原文的不收。
  2. 连边（连边）：每块一次调用（各块并行），在原文中找概念表里概念之间“A 影响 B”的关系，引文逐字核对，
     两端都必须是概念表里的概念。
  3. 建图（不调用大模型）：各块的边按（A, B）合并，同一条边的多处引文都留下；有环时每次在环上删掉引文最少的一条边
     （一样多时删在原文中出现得晚的），直到无环。没有连上任何边的概念不进图，但仍留在概念表里。

调度：同时处理 workers 篇研报（线程池，一篇做完马上接着开始下一篇，不分批，长研报先开始），同时在途的大模型调用
也限定为 workers 个（信号量）。一篇研报里抽概念要按块顺序做（后一块要看前面建好的概念表），概念抽完马上并行连边，
不等其他研报；正在处理的每篇至少占一个调用，所以除了最后剩下不足 workers 篇的时候，在途调用一直是满的
（进度日志里每分钟记一行在途调用数）。

每篇研报做完即存档到 <输出目录>/cache/reports/（JSON，带指纹：段落文字、公司和标题、提示词、chunk_chars、ARCHIVE_VERSION），
重跑时存档有效的直接读回；各次大模型调用缓存在 cache/*_cache.jsonl，重新处理的研报只补做缺少的调用。
某一篇有调用失败时不存档、不进输出，记到 cache/failed_reports.jsonl，下次运行重试；余额不足、鉴权失败让整个运行停下。

输出（<输出目录>，默认 output_dag/）：
  概念DAG.html       入口页面（模板 src/viz/dag_viewer.html），双击打开，左侧选研报，选中时才读取这一篇的数据并绘图
  data/graphs/       每篇研报一个数据文件（概念、DAG 的边和引文、因成环删掉的边、正文段落），加上目录 index.js
  data/concepts.parquet、edges.parquet、reports.parquet   扁平表（各列见 write_outputs）
  cache/             调用缓存、各篇存档、出错记录、进度日志
"""
import hashlib
import json
import re
import shutil
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx

from src.config import PKG_DIR, PROMPT_DIR
from src.llm_client import STOP, FatalAPIError, InflightMeter, load_env, log, set_log_path
from src.logic.base import (LLM, CallFailed, append_jsonl, load_jsonl, name_key, norm_direction, norm_name, para_block,
                            parse_json, quote_in, squash, text_hash, with_feedback, write_parquet)
from src.report_logic import load_reports, quarter_of, select_reports

PROMPT_FILE = PROMPT_DIR / 'concept_dag_prompt.md'
VIEWER_TEMPLATE = PKG_DIR / 'viz' / 'dag_viewer.html'
STEPS = ('概念', '连边')  # 每步有 “-System”“-User” 两节
KINDS = ('指标', '事件', '状态')
MAX_NAME_LEN = 20
MAX_DEF_LEN = 60
MAX_EVIDENCE_LEN = 200
MIN_QUOTE = 6  # 连边引文至少这么多个字（去掉空白和引号后）
MAX_ALIAS_LEN = 12  # 原文叫法作别名的字数上限（再长的多半是带了数值、变化的短句，只用来核对概念出自原文）
CHUNK_CHARS = 2500  # 合并段落时每块的字数上限
SLOW_CALL = 120  # 一次调用超过这么多秒就在日志里记一笔（便于发现输出过长、被截断）
ARCHIVE_VERSION = 1  # 改了流程代码、或存档结构不兼容时加一，已有的存档全部失效（调用缓存还在）


class DagOut:
    """输出目录的布局。"""

    def __init__(self, root):
        self.root = Path(root)
        self.viewer = self.root / '概念DAG.html'
        self.data = self.root / 'data'
        self.graphs = self.data / 'graphs'
        self.cache = self.root / 'cache'
        self.reports = self.cache / 'reports'
        self.failed = self.cache / 'failed_reports.jsonl'
        self.log = self.cache / 'progress.log'

    def make(self):
        for d in (self.data, self.cache):
            d.mkdir(parents=True, exist_ok=True)
        return self


# ==================== prompt、缓存 ====================

def load_prompts():
    """按二级标题切分 prompt 文件，返回 {标题: 内容}。"""
    out, cur = {}, None
    for line in PROMPT_FILE.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            cur = line[3:].strip()
            out[cur] = []
        elif cur:
            out[cur].append(line)
    out = {k: '\n'.join(v).strip() for k, v in out.items()}
    lack = [f'{s}-{p}' for s in STEPS for p in ('System', 'User') if f'{s}-{p}' not in out]
    if lack:
        raise SystemExit(f'{PROMPT_FILE} 中缺少这些节：{"、".join(lack)}')
    return out


def too_many(errs, total):
    """一次回复里不合格的条目多到要让模型重做：超过总数的五分之一（至少 1 条）。少量不合格的直接丢掉，免得为一两条反复重试。"""
    return len(errs) > max(1, total // 5)


def alias_ok(s):
    """原文叫法能不能作别名：2 到 MAX_ALIAS_LEN 个字，不带数字和百分号。"""
    return 2 <= len(s) <= MAX_ALIAS_LEN and not re.search(r'[0-9%％]', s)


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace('{' + k + '}', str(v))
    return template


class Cache:
    """一类调用的结果缓存：jsonl 文件，每行 {key, result, ...}，重跑时同样的输入直接取结果。"""

    def __init__(self, path):
        self.path, self.lock = path, threading.Lock()
        self.rows = {r['key']: r['result'] for r in load_jsonl(path)}

    def get(self, key):
        return self.rows.get(key)

    def put(self, key, result, **extra):
        with self.lock:
            self.rows[key] = result
        append_jsonl(self.path, {'key': key, **extra, 'result': result})


class Ctx:
    """整次运行共用：大模型（同时在途的调用数用信号量限定为 workers）、prompt、两步的缓存、并行连边用的线程池。"""

    def __init__(self, env, out, workers, chunk_chars):
        self.llm, self.prompts, self.out = LLM(env), load_prompts(), out
        self.workers, self.chunk_chars = workers, chunk_chars
        self.sig = {s: text_hash(self.prompts[f'{s}-System'], self.prompts[f'{s}-User']) for s in STEPS}
        self.cache = {'概念': Cache(out.cache / 'concept_cache.jsonl'), '连边': Cache(out.cache / 'edge_cache.jsonl')}
        self.pool, self.lock, self.retry_errors = None, threading.Lock(), defaultdict(list)
        self.meter = InflightMeter(workers)  # 每分钟在日志里记一行在途调用数，确认并发一直是满的
        sem, raw = threading.BoundedSemaphore(workers), self.meter.wrap(self.llm.client.chat)

        def chat(*a, **kw):  # 抽概念在各篇自己的线程里调用、连边在共用线程池里调用，统一在这里限定并发
            with sem:
                if STOP.is_set():
                    raise FatalAPIError('任务已因致命错误停止')
                t = time.time()
                content, finish, usage = raw(*a, **kw)
            dt = time.time() - t
            if finish == 'length' or dt > SLOW_CALL:
                log(f'!!! 一次调用用了 {dt:.0f} 秒，finish_reason={finish}，输出 {usage.get("completion_tokens")} tokens')
            return content, finish, usage
        self.llm.client.chat = chat

    def ask(self, step, key, parse, lenient, extra, **user_kw):
        """有缓存直接返回；否则调用（不合格时带上错误说明重试，用完重试次数后用 lenient 取合格的部分）并写缓存。"""
        hit = self.cache[step].get(key)
        if hit is not None:
            return hit
        def checked(content):  # 记下不合格的原因，结束时汇总
            try:
                return parse(content)
            except (ValueError, KeyError, TypeError, AttributeError) as e:
                with self.lock:
                    self.retry_errors[step].append(str(e)[:200])
                log(f'!!! {step}回复不合格（{extra.get("report_id")} 第 {extra.get("part")} 块），重做：{str(e)[:150]}')
                raise
        res = self.llm.ask(self.prompts[f'{step}-System'], with_feedback(fill(self.prompts[f'{step}-User'], **user_kw)),
                           checked, lenient)
        self.cache[step].put(key, res, **extra)
        return res


# ==================== 0. 合并段落 ====================

def make_chunks(paras, max_chars):
    """段落按顺序合并成块，每块字数不超过 max_chars（单个段落超长的自成一块），返回段落列表的列表。"""
    chunks, cur, n = [], [], 0
    for p in paras:
        size = len(p['text'])
        if cur and n + size > max_chars:
            chunks.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += size
    if cur:
        chunks.append(cur)
    return chunks


def chunk_text(chunk):
    """核对引文用的原文：各段文字直接相连（squash 会去掉空白，跨段的引文也能对上）。"""
    return '\n'.join(p['text'] for p in chunk)


# ==================== 1. 抽概念 ====================
# 概念表：{规范名: {'aliases': [别名...], 'definition': 定义, 'kind': 类型, 'para_id': 首次出现的段落}}

def table_index(table):
    """名称或别名的字面键 -> 规范名（规范名优先）。"""
    idx = {name_key(n): n for n in table}
    for n, t in table.items():
        for a in t['aliases']:
            idx.setdefault(name_key(a), n)
    return idx


def table_text(table):
    if not table:
        return '（空）'
    return '\n'.join(f'{n} ｜ {"、".join(t["aliases"]) or "—"} ｜ {t["definition"]}' for n, t in table.items())


def parse_concepts(content, table, text, strict=True):
    """抽概念的回复 -> {'new': [...], 'aliases': [[别名, 规范名], ...]}。新概念的 mentions 要逐字出自这部分原文
    （都对不上时，规范名本身出现在原文中也算）；strict 时有不合格的条目就抛出 ValueError（附说明让模型重试），否则跳过它们。"""
    obj = parse_json(content, dict)
    new, aliases = obj.get('new') or [], obj.get('aliases') or []
    if not isinstance(new, list) or not isinstance(aliases, list):
        raise ValueError('new 和 aliases 都应是数组')
    sq, idx = squash(text), table_index(table)
    good_new, good_alias, errs = [], [], []
    for it in new:
        if not isinstance(it, dict):
            continue
        name = norm_name(it.get('name'))
        if not name:
            errs.append('有新概念缺少 name')
            continue
        if len(name) > MAX_NAME_LEN:
            errs.append(f'“{name}”超过 {MAX_NAME_LEN} 个字')
            continue
        ms = it.get('mentions') or []
        ms = [ms] if isinstance(ms, str) else ms
        found = [norm_name(m) for m in ms if isinstance(m, str) and len(squash(m)) >= 2 and squash(m) in sq]
        if not found and squash(name) in sq:
            found = [name]
        if not found:
            errs.append(f'“{name}”的 mentions 不是逐字摘自这部分原文')
            continue
        kind = str(it.get('kind') or '').strip()
        good_new.append({'name': name, 'mentions': list(dict.fromkeys(found))[:5],
                         'definition': norm_name(it.get('definition'))[:MAX_DEF_LEN],
                         'kind': kind if kind in KINDS else '指标'})
    for c in good_new:  # 别名也可以挂在这次新建的概念上（概念表还空着的第一块常见）
        idx.setdefault(name_key(c['name']), c['name'])
    for it in aliases:
        if not isinstance(it, dict):
            continue
        alias, target = norm_name(it.get('alias')), idx.get(name_key(it.get('name')))
        if not target:
            errs.append(f'aliases 中的“{norm_name(it.get("name"))}”不是概念表里的名称')
            continue
        if alias_ok(alias):
            good_alias.append([alias, target])
    if strict and too_many(errs, len(new) + len(aliases)):
        raise ValueError('；'.join(errs[:8]))
    return {'new': good_new, 'aliases': good_alias}


def apply_concepts(table, res, para_id):
    """把一部分原文的抽概念结果并入概念表。新概念和表中已有概念同名的，叫法并成别名；别名已经属于别的概念的不再收。"""
    idx = table_index(table)
    for c in res['new']:
        hit = idx.get(name_key(c['name']))
        if hit is None:
            table[c['name']] = {'aliases': [], 'definition': c['definition'], 'kind': c['kind'], 'para_id': para_id}
            idx[name_key(c['name'])] = hit = c['name']
        for m in c['mentions']:
            if alias_ok(m) and name_key(m) not in idx:
                table[hit]['aliases'].append(m)
                idx[name_key(m)] = hit
    for alias, target in res['aliases']:
        target = idx.get(name_key(target))  # 新建概念和表中已有概念同名时，别名挂到已有的那个上
        if target and name_key(alias) not in idx:
            table[target]['aliases'].append(alias)
            idx[name_key(alias)] = target


def build_concepts(ctx, r):
    """按块顺序抽概念，滚动更新概念表（后一块看得到前面各块建好的表）。"""
    about = ctx_about(r)
    for i, chunk in enumerate(r.chunks, 1):
        text = chunk_text(chunk)
        snapshot = {n: {**t, 'aliases': list(t['aliases'])} for n, t in r.table.items()}
        tt = table_text(snapshot)
        key = text_hash(ctx.sig['概念'], about['company'], about['title'], tt, str(i), str(len(r.chunks)), text)
        res = ctx.ask('概念', key, lambda c: parse_concepts(c, snapshot, text),
                      lambda c: parse_concepts(c, snapshot, text, strict=False), {'report_id': r.rid, 'part': i},
                      **about, table=tt, part=i, parts=len(r.chunks), text=para_block(chunk))
        apply_concepts(r.table, res, chunk[0]['para_id'])


def ctx_about(r):
    return {'company': r.meta.get('company') or '未知', 'title': r.meta.get('title') or '（无）'}


# ==================== 2. 连边 ====================

def locate(quote, chunk, pid):
    """引文所在的段落：先看模型给的段落编号，对不上时在这一块里找。"""
    by_id = {p['para_id']: p for p in chunk}
    if pid in by_id and quote_in(quote, by_id[pid]['text'], 1):
        return pid
    return next((p['para_id'] for p in chunk if quote_in(quote, p['text'], 1)), pid if pid in by_id else chunk[0]['para_id'])


def parse_edges(content, idx, chunk, strict=True):
    """连边的回复 -> 合格的边 [{cause, effect, direction, evidence, para_id}]（两端换成规范名，同一条边只留第一处）。
    strict 时有不合格的边就抛出 ValueError（附说明让模型重试），否则跳过它们。"""
    items = parse_json(content, list)
    text, out, seen, errs = chunk_text(chunk), [], set(), []
    for it in items:
        if not isinstance(it, dict):
            continue
        raw_c, raw_e = norm_name(it.get('cause')), norm_name(it.get('effect'))
        cause, effect = idx.get(name_key(raw_c)), idx.get(name_key(raw_e))
        bad = [x for x, y in ((raw_c, cause), (raw_e, effect)) if not y]
        if bad:
            errs.append(f'“{"”“".join(bad)}”不在概念表中')
            continue
        if cause == effect:
            errs.append(f'“{raw_c}”和“{raw_e}”是同一个概念，不能连边')
            continue
        quote = str(it.get('evidence') or '').strip()
        if not quote_in(quote, text, MIN_QUOTE):
            errs.append(f'{cause} → {effect} 的引文不是逐字摘自原文：{quote[:40]}')
            continue
        if (cause, effect) in seen:
            continue
        seen.add((cause, effect))
        try:
            pid = int(it.get('para_id'))
        except (TypeError, ValueError):
            pid = None
        out.append({'cause': cause, 'effect': effect, 'direction': norm_direction(it.get('direction')) or '未说明',
                    'evidence': quote[:MAX_EVIDENCE_LEN], 'para_id': locate(quote, chunk, pid)})
    if strict and too_many(errs, len(items)):
        raise ValueError('；'.join(errs[:8]))
    return out


def extract_edges(ctx, r):
    """各块并行连边，返回按块、块内顺序排列的边（每条带 chunk 序号）。有一块调用失败就抛出 CallFailed（其余块的结果已缓存）。"""
    about, idx, tt = ctx_about(r), table_index(r.table), table_text(r.table)

    def one(i):
        chunk = r.chunks[i]
        key = text_hash(ctx.sig['连边'], about['company'], about['title'], tt, chunk_text(chunk))
        return ctx.ask('连边', key, lambda c: parse_edges(c, idx, chunk), lambda c: parse_edges(c, idx, chunk, strict=False),
                       {'report_id': r.rid, 'part': i + 1}, **about, table=tt, text=para_block(chunk))

    futs = [ctx.pool.submit(one, i) for i in range(len(r.chunks))] if r.table else []
    edges, err = [], None
    for i, f in enumerate(futs):
        try:
            edges += [{**e, 'chunk': i + 1} for e in f.result()]
        except FatalAPIError:
            raise
        except Exception as e:  # 等其余各块都做完（结果写进缓存），再把这一篇记为出错
            err = err or e
    if err is not None:
        raise err
    return edges


# ==================== 3. 建图 ====================

def merge_direction(ds):
    """同一条边各处引文的方向：只有一种明确方向时取它，正负都有记为“矛盾”，都没说明记为“未说明”。"""
    known = set(ds) - {'未说明'}
    return known.pop() if len(known) == 1 else '矛盾' if known else '未说明'


def build_dag(r):
    """各块的边合并成有向图，删边去环，算每个概念的层级（从起点算起的最长路径步数）和图的统计量。"""
    agg = {}
    for k, e in enumerate(r.raw_edges):
        a = agg.setdefault((e['cause'], e['effect']), {'cause': e['cause'], 'effect': e['effect'], 'order': k,
                                                        'directions': [], 'evidence': []})
        a['directions'].append(e['direction'])
        if all(x['quote'] != e['evidence'] for x in a['evidence']):
            a['evidence'].append({'quote': e['evidence'], 'para_id': e['para_id'], 'direction': e['direction']})
    for a in agg.values():
        a['direction'], a['support'] = merge_direction(a.pop('directions')), len(a['evidence'])
    g = nx.DiGraph()
    for (u, v), a in agg.items():
        g.add_edge(u, v, support=a['support'], order=a['order'])
    removed = []
    while True:
        try:
            cyc = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            break
        u, v = min(((x[0], x[1]) for x in cyc), key=lambda e: (g.edges[e]['support'], -g.edges[e]['order']))
        removed.append({**agg[(u, v)], 'cycle': [x[0] for x in cyc]})
        g.remove_edge(u, v)
    level = {}
    for n in nx.topological_sort(g):
        level[n] = max((level[p] + 1 for p in g.predecessors(n)), default=0)
    r.edges = [agg[e] for e in sorted(g.edges, key=lambda e: agg[e]['order'])]
    r.removed, r.level = removed, level
    n_nodes = g.number_of_nodes()
    r.stats = {
        'n_chunks': len(r.chunks), 'n_concepts': len(r.table), 'n_nodes': n_nodes, 'n_edges': g.number_of_edges(),
        'n_raw_edges': len(r.raw_edges), 'n_removed': len(removed),
        'n_isolated': len(r.table) - n_nodes,
        'n_sources': sum(1 for n in g if g.in_degree(n) == 0), 'n_sinks': sum(1 for n in g if g.out_degree(n) == 0),
        'n_components': nx.number_weakly_connected_components(g) if n_nodes else 0,
        'longest': max(level.values(), default=0),
        'n_chars': sum(len(p['text']) for p in r.paras),
    }


# ==================== 存档 ====================

def archive_base(chunk_chars):
    return text_hash(str(ARCHIVE_VERSION), str(chunk_chars), PROMPT_FILE.read_text(encoding='utf-8'))


def archive_sig(r, base):
    return text_hash(base, r.meta.get('company') or '', r.meta.get('title') or '',
                     *(f'{p["para_id"]}\x1e{p["text"]}' for p in r.paras))


def data_file(rid):
    """一篇研报的数据文件在 graphs/ 下的相对路径（按名称前两个字符分子目录）。"""
    key = rid if re.fullmatch(r'[A-Za-z0-9_-]{4,80}', rid) else hashlib.md5(rid.encode('utf-8')).hexdigest()[:16]
    return f'{key[:2].lower()}/{key}.js'


def archive_path(out, rid):
    return out.reports / Path(data_file(rid)).with_suffix('.json')


RESULT_FIELDS = ('table', 'raw_edges', 'edges', 'removed', 'level', 'stats')


def read_archive(out, r, sig):
    path = archive_path(out, r.rid)
    if not path.exists():
        return False
    try:
        obj = json.loads(path.read_text(encoding='utf-8'))
        if obj['sig'] != sig:
            return False
        for k in RESULT_FIELDS:
            setattr(r, k, obj[k])
    except Exception:
        return False
    return True


def write_archive(out, r, sig):
    path = archive_path(out, r.rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps({'sig': sig, **{k: getattr(r, k) for k in RESULT_FIELDS}}, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


# ==================== 输出 ====================

def write_js(path, func, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{func}({json.dumps(obj, ensure_ascii=False, separators=(",", ":"))});\n', encoding='utf-8')


META_KEYS = ('company', 'stock_code', 'date', 'broker', 'title', 'report_type')


def write_outputs(out, finished, model):
    """写出入口页面、每篇研报的数据文件和目录、三张扁平表。
    concepts.parquet：每篇研报的每个概念一行（in_dag 是否进图，level 在图中的层级，n_in/n_out 入边、出边数）；
    edges.parquet：每条边一行（in_dag 为假的是因成环删掉的边；evidence 多处引文用“ ｜ ”分隔）；
    reports.parquet：每篇研报一行（研报信息和 build_dag 的统计量）。"""
    if out.graphs.exists():
        shutil.rmtree(out.graphs)
    index, concept_rows, edge_rows, report_rows = [], [], [], []
    for r in finished:
        meta = {k: r.meta.get(k) or '' for k in META_KEYS}
        n_in, n_out = Counter(e['effect'] for e in r.edges), Counter(e['cause'] for e in r.edges)
        nodes = [{'id': n, 'kind': t['kind'], 'definition': t['definition'], 'aliases': t['aliases'],
                  'para_id': t.get('para_id'), 'level': r.level.get(n), 'in_dag': n in r.level} for n, t in r.table.items()]
        edges = [{k: e[k] for k in ('cause', 'effect', 'direction', 'support', 'evidence')} for e in r.edges]
        removed = [{k: e[k] for k in ('cause', 'effect', 'direction', 'support', 'evidence', 'cycle')} for e in r.removed]
        paras = [{'para_id': p['para_id'], 'text': p['text']} for p in r.paras]
        f = data_file(r.rid)
        write_js(out.graphs / f, 'dagReport', {'rid': r.rid, 'meta': meta, 'stats': r.stats, 'nodes': nodes, 'edges': edges,
                                               'removed': removed, 'paras': paras})
        index.append({'rid': r.rid, 'file': f, **meta, **{k: r.stats[k] for k in ('n_concepts', 'n_nodes', 'n_edges')}})
        for x in nodes:
            concept_rows.append({'report_id': r.rid, 'stock_code': meta['stock_code'], 'name': x['id'], 'kind': x['kind'],
                                 'definition': x['definition'], 'aliases': '、'.join(x['aliases']), 'in_dag': x['in_dag'],
                                 'level': x['level'], 'n_in': n_in[x['id']], 'n_out': n_out[x['id']]})
        for e, kept in [(e, True) for e in r.edges] + [(e, False) for e in r.removed]:
            edge_rows.append({'report_id': r.rid, 'stock_code': meta['stock_code'], 'cause': e['cause'], 'effect': e['effect'],
                              'direction': e['direction'], 'support': e['support'], 'in_dag': kept,
                              'evidence': ' ｜ '.join(x['quote'] for x in e['evidence']),
                              'para_ids': ','.join(str(x['para_id']) for x in e['evidence'])})
        report_rows.append({'report_id': r.rid, **meta, 'quarter': quarter_of(meta['date']), **r.stats})
    index.sort(key=lambda x: (x['stock_code'], x['date'], x['rid']))
    write_js(out.graphs / 'index.js', 'dagIndex', {'model': model, 'generated': time.strftime('%Y-%m-%d %H:%M'),
                                                   'reports': index})
    write_parquet(concept_rows, out.data / 'concepts.parquet', ['report_id', 'name'])
    write_parquet(edge_rows, out.data / 'edges.parquet', ['report_id', 'cause', 'effect'])
    write_parquet(report_rows, out.data / 'reports.parquet', ['report_id'])
    shutil.copyfile(VIEWER_TEMPLATE, out.viewer)


# ==================== 主流程 ====================

def process(ctx, r):
    """一篇研报：抽概念 -> 连边 -> 建图。"""
    if STOP.is_set():
        raise FatalAPIError('任务已停止')
    r.table = {}
    build_concepts(ctx, r)
    r.raw_edges = extract_edges(ctx, r)
    build_dag(r)
    return r


def clear_results(out):
    """清空上一次的结果和各篇存档（调用缓存保留，重做不花钱）。"""
    for path in (out.data, out.reports):
        if path.exists():
            shutil.rmtree(path)
    for path in (out.viewer, out.failed, out.log):
        path.unlink(missing_ok=True)


def main(paras_path, out_dir, reports_path=None, select=None, limit=None, seed=0, workers=50, chunk_chars=CHUNK_CHARS,
         min_per_quarter=1, fresh=False, dry_run=False, keep=None, archived_only=False):
    """paras_path：clear_txt.py 写出的段落 parquet；reports_path：配套的研报元数据表，select(行) 为真的研报才抽取
    （None 时不筛选）；limit、seed：在选中的研报里按种子随机抽多少篇（同 extract_logic_chain.py）；
    workers：同时在途的大模型调用数，也是同时在处理的研报篇数；chunk_chars：合并段落时每块的字数上限；
    min_per_quarter：同一家公司同一个季度不足这么多篇的不抽（1 表示不按这条筛）；fresh：先清空上一次的结果和存档；
    dry_run：只统计选中的研报和合并后的块数，不调用大模型；keep(研报)：读入正文后再筛一遍，为真的才抽（如按正文字数），
    在按 limit 抽样之前筛；archived_only：只用已有的存档写输出、不调用大模型，没有有效存档的研报这次不进输出
    （少数研报的调用一直卡住时用，下次正常运行再补）。"""
    t0 = time.time()
    out = DagOut(out_dir)
    info, note = select_reports(reports_path, select, min_per_quarter)
    reports, _ = load_reports(paras_path, None if keep else limit, info, seed)
    if keep:
        n0 = len(reports)
        reports = [r for r in reports if keep(r)]
        note += f'；读入正文后剔除 {n0 - len(reports)} 篇（{getattr(keep, 'note', None) or (keep.__doc__ or 'keep').strip().splitlines()[0]}），剩 {len(reports)} 篇'
        if limit is not None:  # 抽样方式同 load_reports
            ids = set(sorted((r.rid for r in reports), key=lambda rid: hashlib.md5(f'{seed}:{rid}'.encode('utf-8')).hexdigest())[:limit])
            reports = [r for r in reports if r.rid in ids]
    if not reports:
        raise SystemExit(f'{note}；没有可抽取的研报')
    for r in reports:
        r.chunks = make_chunks(r.paras, chunk_chars)
    n_paras, n_chunks = sum(len(r.paras) for r in reports), sum(len(r.chunks) for r in reports)
    summary = (f'{len(reports)} 篇研报、{len({r.meta["stock_code"] for r in reports})} 家公司；{n_paras} 个段落合并成 '
               f'{n_chunks} 块（每块不超过 {chunk_chars} 字），每块抽概念、连边各一次调用')
    if dry_run:
        print(note)
        print(summary)
        by_c = defaultdict(lambda: [0, 0, 0])
        for r in reports:
            x = by_c[(r.meta['stock_code'], r.meta['company'])]
            x[0], x[1], x[2] = x[0] + 1, x[1] + len(r.paras), x[2] + len(r.chunks)
        for (code, name), (n, p, c) in sorted(by_c.items(), key=lambda kv: kv[1][0]):
            print(f'  {code} {name}：{n} 篇，{p} 段 -> {c} 块')
        print('只做了统计，没有调用大模型；去掉 --dry-run 才开始抽取')
        return
    if fresh:
        clear_results(out)
    out.make()
    set_log_path(out.log)
    env = load_env()
    ctx = Ctx(env, out, workers, chunk_chars)
    log(f'===== 概念 DAG 抽取开始：模型 {env["MODEL"]}，{summary}；并发 {workers} =====')
    log(note)

    base = archive_base(chunk_chars)
    sigs = {r.rid: archive_sig(r, base) for r in reports}
    done = {r.rid: r for r in reports if read_archive(out, r, sigs[r.rid])}
    todo = sorted((r for r in reports if r.rid not in done), key=lambda r: (-len(r.chunks), -len(r.paras), r.rid))
    log(f'从存档读回 {len(done)} 篇，本次待处理 {len(todo)} 篇、{sum(len(r.chunks) for r in todo)} 块（长研报先开始）')
    if archived_only and todo:
        log(f'只用存档：{len(todo)} 篇没有有效存档，这次不抽、不进输出：' + '、'.join(f'{r.meta["company"]} {r.rid[:8]}' for r in todo))
        todo = []

    failed, fatal = {}, None
    ctx.pool = ThreadPoolExecutor(max_workers=workers)
    runner = ThreadPoolExecutor(max_workers=max(1, min(workers, len(todo))))
    try:
        futures = {runner.submit(process, ctx, r): r for r in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            r = futures[fut]
            try:
                fut.result()
            except (FatalAPIError, SystemExit) as e:
                fatal = fatal or e
                continue
            except Exception as e:
                err = f'{type(e).__name__}: {e}'
                failed[r.rid] = err
                append_jsonl(out.failed, {'report_id': r.rid, 'error': err, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
                log(f'!!! {r.rid} 出错，不存档、不进输出（下次运行重试）：{err[:300]}')
                continue
            write_archive(out, r, sigs[r.rid])
            done[r.rid] = r
            s, u = r.stats, ctx.llm.usage
            log(f'已做完 {k}/{len(todo)} 篇：{r.meta["company"]} {r.meta["date"]} {r.meta["broker"]}，{len(r.paras)} 段 {s["n_chunks"]} 块；'
                f'概念 {s["n_concepts"]} 个、进图 {s["n_nodes"]} 个，边 {s["n_edges"]} 条（成环删 {s["n_removed"]} 条），'
                f'最长链 {s["longest"]} 步；累计调用 {u["calls"]} 次，用时 {time.time() - t0:.0f} 秒')
    finally:
        runner.shutdown(wait=True, cancel_futures=True)
        ctx.pool.shutdown(wait=True, cancel_futures=True)
        ctx.meter.close()
    if fatal is not None:
        log(f'!!! 致命错误，已停止：{fatal}。已做完的研报都已存档，处理后重新运行即接着跑')
        raise SystemExit(1)
    finished = [done[r.rid] for r in reports if r.rid in done]
    if not finished:
        raise SystemExit('没有完成的研报，见上面的日志')
    write_outputs(out, finished, env['MODEL'])
    u = ctx.llm.usage
    st = [r.stats for r in finished]
    avg = lambda k: sum(s[k] for s in st) / len(st)
    log(f'===== 结束：输出 {len(finished)} 篇（出错 {len(failed)} 篇）；平均每篇概念 {avg("n_concepts"):.1f} 个、进图 {avg("n_nodes"):.1f} 个、'
        f'边 {avg("n_edges"):.1f} 条、成环删边 {avg("n_removed"):.2f} 条、最长链 {avg("longest"):.1f} 步；本次调用 {u["calls"]} 次'
        f'（不合格重试 {u["retries"]} 次），输入 {u["prompt_tokens"]} tokens、输出 {u["completion_tokens"]} tokens；'
        f'用时 {time.time() - t0:.0f} 秒 =====')
    for step, errs in ctx.retry_errors.items():
        log(f'{step}：回复不合格、让模型重做 {len(errs)} 次，例如：' + ' ／ '.join(errs[:3]))
    log(f'页面：{out.viewer}（双击打开）；结果数据在 {out.data}')
