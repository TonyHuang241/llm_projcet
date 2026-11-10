"""从研报中提取推理逻辑链，整理成每篇研报一张以“公司估值”为根节点的有向无环图（DAG）。

输入是 src/report_text.py 写出的段落 parquet（也可以换成自己处理好的文本，列的要求见该模块说明）。
流程（大模型调用都走 .env 中配置的接口，见 src/neutralize_llm.py 的说明）：
  1. 抽取：每个段落调用一次大模型，抽出三元组（影响因素, 影响方向, 被影响因素），方向只有“正向”“负向”：
     正向表示影响因素越大、被影响因素越大，负向相反；节点是不带方向和数值的具体要素（如“海外销量”“毛利率”）。
  2. 去重：所有研报的节点放在一起，先按规范化后的字面合并（含去掉“公司”、研报公司名这类主语前缀），
     再用 BERT 类句向量模型编码、球面 k-means 聚类，每个簇交给大模型初判哪些名称指代同一要素，
     有可合并的组时再让大模型逐组复核，剔除相似但含义不同的成员（部分与整体、不同地区、比率与绝对额、毛利率与净利率等），
     得到“原名 -> 规范名”映射；聚类做 DEDUP_ROUNDS 轮，每轮换簇大小和随机种子，减少同义词恰好被分到不同簇的情况。
  3. 建图：三元组换成规范名后按研报首尾相连成图；同一条边出现多次时计数，正负方向冲突时取多数；
     去掉自环和从“公司估值”出发的边，有环时删掉环上支持次数最少的边，保证是 DAG。
  4. 连接估值：每篇研报的图里，除“公司估值”外还有终点（出度为 0 的节点）时，把各个子图交给大模型，
     先假设子图之间相互独立，为每个终点补一条通往“公司估值”的路径（可以经过新增的中间要素，如“归母净利润”）；
     校验路径首尾相接、不成环、所有节点都能到达“公司估值”，不合格带着错误说明重试，多次失败时直接连到“公司估值”（标记 fallback）。
  5. 输出：report_graphs.json（每篇研报的节点和边，含层级、对估值的净影响方向、原文证据），以及几张扁平的 parquet 表，
     并调用 src/plot_report_logic.py 生成可视化 HTML。

断点续跑：抽取、合并、连接三步的大模型结果都追加缓存在输出目录的 *_cache.jsonl 中，重新运行只会补做缺少的部分
（段落文字变化、簇的成员变化、子图变化时自动重新调用）。

用法：在 main.py 中设置 TASK = 'report_logic'
"""
import hashlib
import json
import math
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src import neutralize_llm
from src.neutralize_llm import ContentFilterError, FatalAPIError, LLMClient, load_env, log

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT_DIR / 'prompts' / 'report_logic_prompt.md'

ROOT = '公司估值'  # 所有逻辑的落脚点（DAG 的根节点）
# 字面上就是估值的说法，直接并入根节点（其余同义词交给去重时的大模型判断）
ROOT_ALIASES = {'估值', '公司估值', '目标价', '估值水平', '合理估值', '目标估值', '投资评级', '评级', '股价上行空间', '上行空间',
                '公司价值', '股票估值', '目标市值'}
DIRECTIONS = {'正向': 1, '负向': -1}
DIRECTION_ALIASES = {'正': '正向', '+': '正向', 'positive': '正向', '正相关': '正向',
                     '负': '负向', '-': '负向', 'negative': '负向', '负相关': '负向', '反向': '负向'}
MAX_NODE_LEN = 20  # 节点名的最大字数，超过视为抽取错误
MAX_EVIDENCE_LEN = 120
MAX_RETRIES = 4  # 单次调用输出不合格时的重试次数
DEFAULT_CONCURRENCY = 50

EMBED_MODEL = 'BAAI/bge-base-zh-v1.5'  # 中文句向量模型（BERT 结构），用 [CLS] 向量
DEDUP_ROUNDS = 2
CLUSTER_SIZES = (25, 35)  # 每轮 k-means 的目标簇大小（k = 节点数 / 簇大小），轮数多于元组长度时沿用最后一个
MAX_CLUSTER = 60  # 超过该大小的簇再细分，避免一次交给大模型的名称太多
CONTEXT_EXAMPLES = 2  # 合并时每个名称附带的例句（所在三元组）条数
N_DRIVERS = 20  # 连接估值时提示的“常用中间要素”个数（按指向公司估值的次数取）
DEFAULT_DRIVERS = ['归母净利润', '盈利预测', '营业收入', '毛利率', '净利率', 'ROE', '分红率', '行业估值中枢']

_lock = threading.Lock()


# ==================== 工具函数 ====================

def norm_name(s):
    """节点名规范化：全角转半角、去引号书名号、去掉汉字两侧的空格和首尾标点。"""
    s = unicodedata.normalize('NFKC', str(s or '')).strip()
    s = re.sub(r'[“”"\'‘’「」『』《》]', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])', '', s)
    return s.strip(' ，,。.;；:：、')


def name_key(s):
    """字面比较用的键：再去掉空格、连接符和括号，忽略英文大小写。"""
    return re.sub(r'[\s\-_·/、，,()]', '', norm_name(s)).casefold()


ROOT_KEYS = {name_key(a) for a in ROOT_ALIASES}


def norm_direction(d):
    d = str(d or '').strip()
    d = DIRECTION_ALIASES.get(d.lower(), d)
    return d if d in DIRECTIONS else None


def text_hash(*parts):
    return hashlib.sha1('\x1f'.join(parts).encode('utf-8')).hexdigest()[:16]


def load_prompts():
    """按二级标题切分 prompt 文件，返回 {标题: 内容}。"""
    prompts, cur = {}, None
    for line in PROMPT_FILE.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            cur = line[3:].strip()
            prompts[cur] = []
        elif cur:
            prompts[cur].append(line)
    prompts = {k: '\n'.join(v).strip() for k, v in prompts.items()}
    for k in ('抽取-System', '抽取-User', '合并-System', '合并-User', '复核-System', '复核-User', '连接-System', '连接-User'):
        if k not in prompts:
            raise SystemExit(f'{PROMPT_FILE} 中缺少 “## {k}” 一节')
    return prompts


def prompt_sig(prompts, step):
    """某一步 prompt 的指纹，放进缓存键里：改了 prompt 措辞后，这一步的缓存自动失效。"""
    return text_hash(*(prompts[k] for k in sorted(prompts) if k.startswith(step)))


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace('{' + k + '}', str(v))
    return template


def parse_json(content, want):
    """从模型回复中取出 JSON（容忍代码块标记、思考过程和前后说明），want 为 list 或 dict。"""
    text = re.sub(r'<think>.*?</think>', '', content, flags=re.S).strip()
    text = re.sub(r'^```(?:json)?|```$', '', text.strip()).strip()
    pairs = [('[', ']'), ('{', '}')] if want is list else [('{', '}')]
    for left, right in pairs:
        i, j = text.find(left), text.rfind(right)
        if i >= 0 and j > i:
            try:
                obj = json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                continue
            if want is list and isinstance(obj, dict):  # 有的模型会包一层 {"triples": [...]}
                obj = next((v for v in obj.values() if isinstance(v, list)), None)
            if isinstance(obj, want):
                return obj
    raise ValueError('回复中没有合法的 JSON')


def load_jsonl(path):
    rows = []
    if path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:  # 中断时写了一半的行
                continue
    return rows


def append_jsonl(path, row):
    with _lock, path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_parquet(rows, path, columns):
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({c: pa.array([], pa.string()) for c in columns}), path)


class LLM:
    """在 LLMClient 外面加上：输出解析失败时重试（可把上次的错误告诉模型）、用量统计。"""

    def __init__(self, env):
        self.client = LLMClient(env)
        self.usage = Counter()

    def ask(self, system, user, parse):
        """user 可以是字符串，或接收上一次错误（首次为 None）返回 prompt 的函数。"""
        err = None
        for _ in range(MAX_RETRIES):
            prompt = user(err) if callable(user) else user
            try:
                content, finish, usage = self.client.chat(system, prompt)
                with _lock:
                    self.usage['calls'] += 1
                    self.usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                    self.usage['completion_tokens'] += usage.get('completion_tokens', 0)
                if finish == 'length':
                    raise ValueError('输出被截断')
                return parse(content)
            except (ValueError, KeyError, TypeError, AttributeError, ContentFilterError) as e:
                err = e
                with _lock:
                    self.usage['retries'] += 1
        raise RuntimeError(f'连续 {MAX_RETRIES} 次不合格：{err}')


def run_parallel(items, fn, workers, label):
    """并行执行 fn(item)，返回结果列表（与 items 同序，失败为 None）；致命错误时停止全部任务。"""
    results, t0 = [None] * len(items), time.time()
    if not items:
        return results
    with ThreadPoolExecutor(max_workers=min(workers, len(items))) as pool:
        futures = {pool.submit(fn, it): i for i, it in enumerate(items)}
        for k, fut in enumerate(as_completed(futures), 1):
            try:
                results[futures[fut]] = fut.result()
            except FatalAPIError as e:
                pool.shutdown(wait=False, cancel_futures=True)
                raise SystemExit(f'致命错误，已停止（已完成的部分保存在缓存中，处理后重新运行即可继续）：{e}')
            except Exception as e:  # 单项失败不影响其他项
                log(f'{label} 第 {futures[fut] + 1} 项失败：{e}')
            if k % max(1, len(items) // 10) == 0 or k == len(items):
                log(f'{label}：{k}/{len(items)}，用时 {time.time() - t0:.0f} 秒')
    return results


# ==================== 第 1 步：段落 -> 三元组 ====================

def clean_triple(t):
    """校验并规范化模型给出的一条三元组，不合格返回 None。"""
    if not isinstance(t, dict):
        return None
    cause, effect = norm_name(t.get('cause')), norm_name(t.get('effect'))
    direction = norm_direction(t.get('direction'))
    if name_key(cause) in ROOT_KEYS:
        cause = ROOT
    if name_key(effect) in ROOT_KEYS:
        effect = ROOT
    if (not cause or not effect or direction is None or name_key(cause) == name_key(effect)
            or len(cause) > MAX_NODE_LEN or len(effect) > MAX_NODE_LEN):
        return None
    return {'cause': cause, 'direction': direction, 'effect': effect,
            'evidence': norm_name(t.get('evidence'))[:MAX_EVIDENCE_LEN]}


def extract_triples(paras, out_dir, llm, prompts, workers):
    """每个段落调用一次大模型；结果按 (report_id, para_id, 文字哈希) 缓存。返回扁平的三元组列表。"""
    cache_path = out_dir / 'extract_cache.jsonl'
    cache = {r['key']: r for r in load_jsonl(cache_path)}
    for p in paras:
        p['key'] = text_hash(prompt_sig(prompts, '抽取'), p['report_id'], str(p['para_id']), p['text'])
    todo = [p for p in paras if p['key'] not in cache]
    log(f'三元组抽取：共 {len(paras)} 段，已缓存 {len(paras) - len(todo)} 段，本次调用 {len(todo)} 段')

    def work(p):
        user = fill(prompts['抽取-User'], company=p.get('company') or '未知', title=p.get('title') or '',
                    section=p.get('section') or '无', text=p['text'])
        items = llm.ask(prompts['抽取-System'], user, lambda c: parse_json(c, list))
        row = {'key': p['key'], 'report_id': p['report_id'], 'para_id': p['para_id'],
               'triples': [t for t in map(clean_triple, items) if t], 'n_raw': len(items)}
        append_jsonl(cache_path, row)  # 只缓存成功的段落，失败的下次重跑
        return row

    for row in run_parallel(todo, work, workers, '三元组抽取'):
        if row:
            cache[row['key']] = row
    triples, failed = [], 0
    for p in paras:
        row = cache.get(p['key'])
        if row is None:
            failed += 1
            continue
        triples += [{'report_id': p['report_id'], 'para_id': p['para_id'], **t} for t in row['triples']]
    if failed:
        log(f'!!! {failed} 段抽取失败，这些段落没有三元组；重新运行会再试')
    log(f'三元组抽取完成：{len(triples)} 条，涉及 {len({t["report_id"] for t in triples})} 篇研报')
    return triples


# ==================== 第 2 步：节点去重 ====================

def embed(names, model_name):
    """用 BERT 类句向量模型编码，返回 L2 归一化后的矩阵。"""
    import torch  # 依赖 torch，只在用到时导入
    from transformers import AutoModel, AutoTokenizer
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(names), 256):
            batch = tok(names[i:i + 256], padding=True, truncation=True, max_length=64, return_tensors='pt').to(device)
            out.append(torch.nn.functional.normalize(model(**batch).last_hidden_state[:, 0], dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def spherical_kmeans(x, k, seed, iters=50):
    """余弦距离下的 k-means（k-means++ 初始化），x 为归一化向量，返回每行的簇号。"""
    rng = np.random.default_rng(seed)
    n = len(x)
    centers = [x[rng.integers(n)]]
    dist = 1 - x @ centers[0]
    for _ in range(1, k):
        p = np.clip(dist, 0, None).astype(float)
        centers.append(x[rng.choice(n, p=p / p.sum()) if p.sum() > 0 else rng.integers(n)])
        dist = np.minimum(dist, 1 - x @ centers[-1])
    c = np.array(centers)
    labels = None
    for _ in range(iters):
        sim = x @ c.T
        new = sim.argmax(1)
        if labels is not None and (new == labels).all():
            break
        labels = new
        for j in range(k):
            members = x[labels == j]
            if len(members):
                v = members.sum(0)
                c[j] = v / (np.linalg.norm(v) or 1)
            else:  # 空簇：重新放到离各自中心最远的点上
                c[j] = x[sim.max(1).argmin()]
    return labels


def make_clusters(x, idx, target, seed):
    """把 idx 指向的向量切成大小约为 target、不超过 MAX_CLUSTER 的簇，返回 [[下标...]]。"""
    if len(idx) <= MAX_CLUSTER:
        return [idx]
    k = math.ceil(len(idx) / target)
    labels = spherical_kmeans(x[idx], k, seed)
    groups = [[idx[i] for i in np.flatnonzero(labels == j)] for j in range(k)]
    groups = [g for g in groups if g]
    if len(groups) == 1:  # 分不开（向量几乎相同）：按顺序硬切
        return [idx[i:i + MAX_CLUSTER] for i in range(0, len(idx), MAX_CLUSTER)]
    return [c for g in groups for c in make_clusters(x, g, target, seed + 1)]


def parse_groups(content, names):
    """解析合并结果：返回 [(规范名, [成员名...])]；编号越界等硬错误抛 ValueError 以便重试。"""
    groups = parse_json(content, dict).get('groups')
    if not isinstance(groups, list):
        raise ValueError('缺少 groups 数组')
    out, seen = [], set()
    for g in groups:
        ids = [int(i) for i in g.get('members', [])]
        if any(i < 1 or i > len(names) for i in ids):
            raise ValueError(f'编号越界：{ids}')
        ids = [i for i in dict.fromkeys(ids) if i not in seen]
        seen.update(ids)
        if len(ids) >= 2:
            out.append((norm_name(g.get('canonical')), [names[i - 1] for i in ids]))
    return out


def parse_verified(content, proposed):
    """解析复核结果：每组成员必须来自同一个初判组，返回 [(规范名, [成员名...])]。"""
    groups = parse_json(content, dict).get('groups')
    if not isinstance(groups, list):
        raise ValueError('缺少 groups 数组')
    owner = {m: i for i, (_, members) in enumerate(proposed) for m in members}
    out, seen = [], set()
    for g in groups:
        names = [m for m in dict.fromkeys(norm_name(x) for x in g.get('members', [])) if m not in seen]
        unknown = [m for m in names if m not in owner]
        if unknown:
            raise ValueError(f'这些名称不在待复核的组里：{"、".join(unknown)}')
        if len({owner[m] for m in names}) > 1:
            raise ValueError(f'一组成员来自不同的待复核组：{"、".join(names)}')
        seen.update(names)
        if len(names) >= 2:
            out.append((norm_name(g.get('canonical')), names))
    return out


def dedup_nodes(triples, out_dir, llm, prompts, workers, embed_model, rounds, companies=()):
    """返回 {原节点名: 规范名}。companies 为各研报的公司简称，节点名带这些主语前缀时按去掉前缀后的字面合并。"""
    count, examples = Counter(), defaultdict(list)
    for t in triples:
        for n in (t['cause'], t['effect']):
            count[n] += 1
            if len(examples[n]) < CONTEXT_EXAMPLES:
                examples[n].append(f'{t["cause"]}→{t["effect"]}')
    names = sorted(set(count) | {ROOT})
    count[ROOT] += 0

    # 字面合并：规范化后相同的直接合并，规范名取出现次数最多的写法；节点默认以研报公司为主语，
    # 所以“公司海外销量”“比亚迪海外销量”这类去掉主语前缀后与已有名称相同的，也直接合并
    keys = {name_key(n) for n in names}
    subjects = sorted({name_key(c) for c in companies if c} | {'公司', '本公司'}, key=len, reverse=True)

    def literal_key(n):
        k = name_key(n)
        if k in ROOT_KEYS:
            return ROOT
        return next((k[len(sk):] for sk in subjects if k.startswith(sk) and k[len(sk):] in keys), k)

    by_key = defaultdict(list)
    for n in names:
        by_key[literal_key(n)].append(n)
    rep = {}
    for k, group in by_key.items():
        canon = ROOT if k == ROOT else max(group, key=lambda n: (count[n], -len(n), n))
        rep.update({n: canon for n in group})
    log(f'节点去重：原始 {len(names)} 个名称，字面合并后 {len(set(rep.values()))} 个')
    if len(set(rep.values())) < 2:
        return rep

    log(f'加载句向量模型 {embed_model}')
    vec = dict(zip(names, embed(names, embed_model)))
    cache_path = out_dir / 'dedup_cache.jsonl'
    cache = {r['key']: r for r in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '合并') + prompt_sig(prompts, '复核')

    for r in range(rounds):
        canon = sorted(set(rep.values()))
        members = defaultdict(list)
        for n, c in rep.items():
            members[c].append(n)
        c_count = {c: sum(count[n] for n in members[c]) for c in canon}
        x = np.array([vec[c] for c in canon])
        target = CLUSTER_SIZES[min(r, len(CLUSTER_SIZES) - 1)]
        clusters = [[canon[i] for i in g] for g in make_clusters(x, list(range(len(canon))), target, seed=r * 1000)]
        clusters = [sorted(g) for g in clusters if len(g) >= 2]
        todo = [g for g in clusters if text_hash(sig, *g) not in cache]
        log(f'第 {r + 1} 轮聚类：{len(canon)} 个名称分成 {len(clusters)} 簇（目标簇大小 {target}），'
            f'已缓存 {len(clusters) - len(todo)} 簇，本次调用 {len(todo)} 簇')

        def work(g):
            """先对整簇给出合并初判，有可合并的组时再逐组复核，只保留复核通过的成员。"""
            ctx = lambda c: '；'.join(e for n in members[c] for e in examples[n])[:80] or '无'
            line = lambda c: f'{c} ｜ {c_count[c]}次 ｜ {ctx(c)}'
            items = '\n'.join(f'{i}. {line(c)}' for i, c in enumerate(g, 1))
            proposed = llm.ask(prompts['合并-System'], fill(prompts['合并-User'], items=items), lambda s: parse_groups(s, g))
            verified = []
            if proposed:
                text = '\n\n'.join(f'【第{i}组】规范名称：{canonical}\n' + '\n'.join(f'  - {line(m)}' for m in group)
                                    for i, (canonical, group) in enumerate(proposed, 1))
                verified = llm.ask(prompts['复核-System'], fill(prompts['复核-User'], groups=text),
                                   lambda s: parse_verified(s, proposed))
            row = {'key': text_hash(sig, *g), 'names': g, 'proposed': proposed, 'groups': verified}
            append_jsonl(cache_path, row)
            return row

        for row in run_parallel(todo, work, workers, f'第 {r + 1} 轮合并'):
            if row:
                cache[row['key']] = row
        rows = [cache[text_hash(sig, *g)] for g in clusters if text_hash(sig, *g) in cache]
        n_merges = lambda field: sum(len(group) - 1 for row in rows for _, group in row[field])
        dropped = n_merges('proposed') - n_merges('groups')
        merged = 0
        for row in rows:
            for canonical, group in row['groups']:
                group = [m for m in group if m in c_count]
                if len(group) < 2:
                    continue
                if ROOT in group:
                    canonical = ROOT
                elif canonical not in group:  # 规范名必须是组内名称之一，否则取出现次数最多的
                    canonical = next((m for m in group if name_key(m) == name_key(canonical)),
                                     max(group, key=lambda m: (c_count[m], -len(m), m)))
                group = set(group)
                for n, c in rep.items():
                    if c in group:
                        rep[n] = canonical
                merged += len(group) - 1
        log(f'第 {r + 1} 轮合并完成：初判可合并的名称中复核否决 {dropped} 个，最终合并掉 {merged} 个，剩 {len(set(rep.values()))} 个')
    return rep


# ==================== 第 3 步：建图 ====================

def build_graph(triples, rep):
    """一篇研报的三元组 -> DAG。返回 (图, 诊断计数, 为去环删掉的边)。"""
    agg, info = {}, Counter()
    for order, t in enumerate(triples):
        u, v = rep.get(t['cause'], t['cause']), rep.get(t['effect'], t['effect'])
        if u == v:
            info['self_loops'] += 1
            continue
        if u == ROOT:  # 根节点只能是终点
            info['root_out_edges'] += 1
            continue
        e = agg.setdefault((u, v), {'pos': 0, 'neg': 0, 'evidence': [], 'para_ids': [], 'order': order})
        e['pos' if t['direction'] == '正向' else 'neg'] += 1
        if t['evidence'] and t['evidence'] not in e['evidence']:
            e['evidence'].append(t['evidence'])
        if t['para_id'] not in e['para_ids']:
            e['para_ids'].append(t['para_id'])
    g = nx.DiGraph()
    g.add_node(ROOT)
    for (u, v), e in agg.items():
        sign = 1 if e['pos'] >= e['neg'] else -1
        info['conflicts'] += bool(e['pos'] and e['neg'])
        g.add_edge(u, v, sign=sign, origin='extracted', weight=e['pos'] + e['neg'], n_pos=e['pos'], n_neg=e['neg'],
                   evidence=e['evidence'], para_ids=sorted(e['para_ids']), order=e['order'])
    removed = []
    while True:
        try:
            cycle = nx.find_cycle(g)
        except nx.NetworkXNoCycle:
            break
        u, v = min(((a, b) for a, b in cycle), key=lambda ab: (g.edges[ab]['weight'], -g.edges[ab]['order']))
        removed.append({'source': u, 'target': v, 'direction': '正向' if g.edges[u, v]['sign'] > 0 else '负向',
                        'weight': g.edges[u, v]['weight']})
        g.remove_edge(u, v)
    info['n_triples'] = len(triples)
    return g, info, removed


# ==================== 第 4 步：连接到估值 ====================

def pending_components(g):
    """需要补连接的子图：[(节点集合, 终点列表)]，终点是除根节点外出度为 0 的节点。"""
    out = []
    for comp in sorted(nx.weakly_connected_components(g), key=lambda c: (-len(c), sorted(c))):
        sinks = sorted(n for n in comp if g.out_degree(n) == 0 and n != ROOT)
        if sinks:
            out.append((comp, sinks))
    return out


def describe_components(g, pending):
    blocks = []
    for i, (comp, sinks) in enumerate(pending, 1):
        edges = [f'  {u} —{"正向" if d["sign"] > 0 else "负向"}→ {v}' for u, v, d in g.edges(data=True) if u in comp]
        blocks.append(f'【子图{i}】\n' + '\n'.join(edges or [f'  （孤立节点）{sinks[0]}']) + f'\n  终点：{"、".join(sinks)}')
    return '\n\n'.join(blocks)


def make_resolver(g, rep):
    """把模型写的节点名对应到图中已有节点或全局规范名；都对不上时作为新节点。"""
    local = {name_key(n): n for n in g}
    canon = {name_key(c): c for c in set(rep.values())}

    def resolve(name):
        n = norm_name(name)
        k = name_key(n)
        if k in ROOT_KEYS:
            return ROOT
        return local.get(k) or rep.get(n) or canon.get(k) or n
    return resolve


def apply_links(g, links, terminals, resolve):
    """校验模型给的路径并加到图的副本上，不合格抛 ValueError（错误说明会反馈给模型）。"""
    if not isinstance(links, list):
        raise ValueError('缺少 links 数组')
    h, errors, got = g.copy(), [], {}
    for link in links:
        term = resolve(link.get('terminal'))
        if term not in terminals:
            errors.append(f'“{link.get("terminal")}”不是待连接的终点')
            continue
        hops, prev = [], term
        for hop in link.get('path') or []:
            s, t, d = resolve(hop.get('source')), resolve(hop.get('target')), norm_direction(hop.get('direction'))
            if s != prev or d is None or not t or len(t) > MAX_NODE_LEN or s == ROOT:
                errors.append(f'终点“{term}”的路径在“{hop.get("source")}→{hop.get("target")}”处不合格'
                              f'（需首尾相接、方向为正向或负向）')
                break
            hops.append((s, t, d))
            prev = t
        else:
            if prev != ROOT:
                errors.append(f'终点“{term}”的路径没有到达“{ROOT}”')
            else:
                got[term] = (hops, norm_name(link.get('reason'))[:MAX_EVIDENCE_LEN])
    missing = [t for t in terminals if t not in got]
    if missing:
        errors.append(f'缺少这些终点的路径：{"、".join(missing)}')
    if errors:
        raise ValueError('；'.join(errors))
    for term, (hops, reason) in got.items():
        for s, t, d in hops:
            if h.has_edge(s, t):  # 原文抽取的边优先，不覆盖
                continue
            h.add_edge(s, t, sign=DIRECTIONS[d], origin='linked', weight=0, n_pos=0, n_neg=0,
                       evidence=[reason] if reason else [], para_ids=[], order=10 ** 9)
            if not nx.is_directed_acyclic_graph(h):
                raise ValueError(f'添加“{s}→{t}”会形成环，请换一条路径')
    left = sorted(n for n in h if h.out_degree(n) == 0 and n != ROOT)
    if left:
        raise ValueError(f'添加后这些节点仍未落到“{ROOT}”：{"、".join(left)}')
    return h


def link_to_root(g, meta, llm, prompts, rep, drivers, cache, cache_path):
    """返回 (连接后的图, 状态)。状态：already（本来就都落到估值）、linked、fallback（大模型多次失败，直接连到估值）。"""
    pending = pending_components(g)
    if not pending:
        return g, 'already'
    terminals = [s for _, sinks in pending for s in sinks]
    subgraphs = describe_components(g, pending)
    key = text_hash(prompt_sig(prompts, '连接'), meta['report_id'], subgraphs)
    resolve = make_resolver(g, rep)
    if key in cache:
        try:
            return apply_links(g, cache[key], terminals, resolve), 'linked'
        except ValueError:
            pass
    base = fill(prompts['连接-User'], company=meta.get('company') or '未知', title=meta.get('title') or '',
                subgraphs=subgraphs, terminals='、'.join(terminals), drivers='、'.join(drivers))
    state = {}

    def parse(content):
        links = parse_json(content, dict).get('links')
        h = apply_links(g, links, terminals, resolve)
        state['links'] = links
        return h

    user = lambda err: base if err is None else f'{base}\n\n上一次的输出有以下问题，请修正后重新输出完整结果：{err}'
    try:
        h = llm.ask(prompts['连接-System'], user, parse)
        append_jsonl(cache_path, {'key': key, 'report_id': meta['report_id'], 'links': state['links']})
        return h, 'linked'
    except RuntimeError as e:
        log(f'!!! {meta["report_id"]} 连接估值失败，{len(terminals)} 个终点直接连到“{ROOT}”（方向记为正向，需人工复核）：{e}')
        h = g.copy()
        for t in terminals:
            h.add_edge(t, ROOT, sign=1, origin='fallback', weight=0, n_pos=0, n_neg=0, evidence=[], para_ids=[], order=10 ** 9)
        return h, 'fallback'


def top_drivers(graphs):
    """连接时优先复用的中间要素：常见的估值驱动要素，加上在至少 2 篇研报中直接指向公司估值的要素（按研报数排序）。
    只出现在一篇研报里的（多是风险提示中的具体风险）不放进来。"""
    c = Counter(u for g in graphs for u in g.predecessors(ROOT))
    names = [n for n, k in c.most_common() if k >= 2 and n not in DEFAULT_DRIVERS]
    return (DEFAULT_DRIVERS + names)[:N_DRIVERS]


# ==================== 第 5 步：整理输出 ====================

def annotate(g):
    """层级：到根节点的最长路径步数（根为 0）；净影响：沿各条路径相乘得到的对估值的影响方向集合。"""
    level, effect = {}, {}
    for n in reversed(list(nx.topological_sort(g))):
        succ = list(g.successors(n))
        if n == ROOT:
            level[n], effect[n] = 0, {1}
        elif succ:
            level[n] = 1 + max(level[s] for s in succ)
            effect[n] = {g.edges[n, s]['sign'] * e for s in succ for e in effect[s]}
        else:  # 正常情况下不会出现（连接后只剩根节点一个终点）
            level[n], effect[n] = 1, set()
    label = lambda e: '+' if e == {1} else '-' if e == {-1} else '±' if e else '?'
    return level, {n: label(e) for n, e in effect.items()}


def graph_record(g, meta, info, removed, status, aliases):
    level, effect = annotate(g)
    mentions = Counter()
    for u, v, d in g.edges(data=True):
        mentions[u] += d['weight']
        mentions[v] += d['weight']
    nodes = [{'id': n, 'is_root': n == ROOT, 'level': level[n], 'effect': effect[n], 'mentions': mentions[n],
              'aliases': sorted(aliases.get(n, [])), 'new': n != ROOT and n not in aliases}
             for n in sorted(g, key=lambda n: (level[n], n))]
    edges = [{'source': u, 'target': v, 'direction': '正向' if d['sign'] > 0 else '负向', 'origin': d['origin'],
              'weight': d['weight'], 'n_pos': d['n_pos'], 'n_neg': d['n_neg'], 'evidence': d['evidence'],
              'para_ids': d['para_ids']}
             for u, v, d in sorted(g.edges(data=True), key=lambda e: (level[e[0]], e[0], e[1]))]
    stats = {'n_triples': info['n_triples'], 'n_nodes': len(nodes), 'n_edges': len(edges),
             'n_extracted': sum(e['origin'] == 'extracted' for e in edges),
             'n_linked': sum(e['origin'] == 'linked' for e in edges),
             'n_fallback': sum(e['origin'] == 'fallback' for e in edges),
             'self_loops': info['self_loops'], 'root_out_edges': info['root_out_edges'], 'conflicts': info['conflicts'],
             'removed_cycle_edges': len(removed), 'components_before_link': info['components'],
             'depth': max(level.values()) if level else 0, 'link_status': status}
    return {**meta, 'stats': stats, 'nodes': nodes, 'edges': edges, 'removed_cycle_edges': removed}


def main(paras_path, out_dir, limit=None, embed_model=EMBED_MODEL, rounds=DEDUP_ROUNDS, html_name='report_logic_graphs.html'):
    """paras_path：段落 parquet（report_text.py 的输出或自备文本）；limit：只处理前多少篇研报（按 report_id 排序），None 为全部。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    neutralize_llm.LOG_PATH = out_dir / '_progress.log'
    env = load_env()
    workers = int(env.get('CONCURRENCY') or DEFAULT_CONCURRENCY)
    prompts, llm = load_prompts(), LLM(env)

    table = pq.read_table(paras_path)
    missing = {'report_id', 'para_id', 'text'} - set(table.column_names)
    if missing:
        raise SystemExit(f'{paras_path} 缺少必需的列：{missing}')
    paras = [p for p in table.to_pylist() if (p.get('text') or '').strip()]
    report_ids = sorted({p['report_id'] for p in paras})[:limit]
    paras = [p for p in paras if p['report_id'] in set(report_ids)]
    meta_cols = ('file', 'date', 'broker', 'company', 'stock_code', 'title')
    metas = {}
    for p in paras:
        metas.setdefault(p['report_id'], {'report_id': p['report_id'], **{c: p.get(c) or '' for c in meta_cols},
                                          'n_paragraphs': 0})['n_paragraphs'] += 1
    log(f'===== 开始：模型 {env["MODEL"]}，{len(report_ids)} 篇研报、{len(paras)} 个段落，并行 {workers} =====')

    # 1. 抽取
    triples = extract_triples(paras, out_dir, llm, prompts, workers)
    write_parquet(triples, out_dir / 'triples_raw.parquet',
                  ['report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence'])

    # 2. 去重
    rep = dedup_nodes(triples, out_dir, llm, prompts, workers, embed_model, rounds, {m['company'] for m in metas.values()})
    aliases = defaultdict(set)
    for n, c in rep.items():
        aliases[c].add(n)
    for t in triples:
        t['cause_raw'], t['effect_raw'] = t['cause'], t['effect']
        t['cause'], t['effect'] = rep.get(t['cause'], t['cause']), rep.get(t['effect'], t['effect'])
    write_parquet([{'name': n, 'node': c} for n, c in sorted(rep.items())], out_dir / 'node_map.parquet', ['name', 'node'])
    write_parquet([t for t in triples if t['cause'] != t['effect']], out_dir / 'triples_clean.parquet',
                  ['report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence', 'cause_raw', 'effect_raw'])

    # 3. 建图
    by_report = defaultdict(list)
    for t in triples:
        by_report[t['report_id']].append(t)
    built = {}
    for rid in report_ids:
        g, info, removed = build_graph(by_report[rid], {})  # 三元组已换成规范名
        info['components'] = nx.number_weakly_connected_components(g)
        built[rid] = (g, info, removed)
        if removed:
            log(f'{rid}：删掉 {len(removed)} 条成环的边')

    # 4. 连接估值
    drivers = top_drivers([g for g, _, _ in built.values()])
    cache_path = out_dir / 'link_cache.jsonl'
    cache = {r['key']: r['links'] for r in load_jsonl(cache_path)}
    log(f'连接估值：常用中间要素 {drivers}')

    def work(rid):
        g, info, removed = built[rid]
        return link_to_root(g, metas[rid], llm, prompts, rep, drivers, cache, cache_path)

    linked = run_parallel(report_ids, work, workers, '连接估值')

    # 5. 输出
    records, edge_rows = [], []
    for rid, res in zip(report_ids, linked):
        g, info, removed = built[rid]
        h, status = res if res else (g, 'failed')
        if status == 'failed':  # 连接时出了意外错误（已记录日志）：保留未连接的图，标记出来
            log(f'!!! {rid} 未完成连接，输出的是未连接的图')
            h = g
        rec = graph_record(h, metas[rid], info, removed, status, aliases)
        records.append(rec)
        edge_rows += [{'report_id': rid, **{k: e[k] for k in ('source', 'target', 'direction', 'origin', 'weight')},
                       'evidence': ' | '.join(e['evidence']), 'para_ids': ','.join(map(str, e['para_ids']))}
                      for e in rec['edges']]
    node_rows = []
    in_reports = defaultdict(set)
    for rec in records:
        for n in rec['nodes']:
            in_reports[n['id']].add(rec['report_id'])
    for c in sorted(in_reports, key=lambda c: (-len(in_reports[c]), c)):
        node_rows.append({'node': c, 'n_reports': len(in_reports[c]), 'aliases': '、'.join(sorted(aliases.get(c, [])))})
    (out_dir / 'report_graphs.json').write_text(
        json.dumps({'root': ROOT, 'model': env['MODEL'], 'embed_model': embed_model, 'reports': records},
                   ensure_ascii=False, indent=1), encoding='utf-8')
    write_parquet(edge_rows, out_dir / 'edges.parquet',
                  ['report_id', 'source', 'target', 'direction', 'origin', 'weight', 'evidence', 'para_ids'])
    write_parquet(node_rows, out_dir / 'nodes.parquet', ['node', 'n_reports', 'aliases'])

    from src import plot_report_logic
    plot_report_logic.main(out_dir / 'report_graphs.json', out_dir / html_name)

    status = Counter(r['stats']['link_status'] for r in records)
    u = llm.usage
    log(f'===== 结束：{len(records)} 篇研报，连接状态 {dict(status)}；共 {len(node_rows)} 个节点、{len(edge_rows)} 条边；'
        f'调用 {u["calls"]} 次（不合格重试 {u["retries"]} 次），输入 {u["prompt_tokens"]} tokens，'
        f'输出 {u["completion_tokens"]} tokens =====')
