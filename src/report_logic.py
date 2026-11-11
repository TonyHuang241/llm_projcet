"""从研报中提取估值逻辑图：每篇研报整理成一张分五层的有向图，原文概念逐层连到预置的会计层和公司估值
（框架见 reports/研报逻辑图抽取框架设计.md）。

五层：1 公司估值、2 会计层（这两层预置，见 src/fin_items.py，不从原文抽取）、3 经营指标、4 公司经营情况、5 外部环境。
边只能从层号大的指向层号小的（可以跨层），第 3-5 层内部也可以有原文陈述的关系。

输入是 src/report_text.py 写出的段落 parquet（也可以换成自己处理好的文本，列的要求见该模块说明）；
读入时会再剔除一遍风险提示（规则同 report_text.strip_risk），自备文本里的风险提示也不会进入逻辑图。
流程（大模型调用都走 .env 中配置的接口，见 src/neutralize_llm.py 的说明）：
  1. 概念分层填表：每篇研报按顺序分块交给大模型，滚动维护“本篇概念表”（规范名、别名、定义、层级 3/4/5、类型）。
     会计科目、估值、目标价、PE/PB 等说法按 fin_items 的别名归到预置节点，不进表；篇内去重只在这里做。
     全篇抽完关系、补完表后，再调用一次大模型整理第 3 层的“上位指标”（东盟销量 → 海外销量 → 销量），
     然后向上合并：每棵上位指标树只保留最上层的指标，下位指标的边都改连到它上面，避免同层互相包含。
  2. 原文连边：每段调用一次大模型，带上概念表、会计层清单、小节标题和前一段末尾（只用于理解指代），抽出原文明说的一跳关系
     （影响因素, 正向/负向, 被影响因素, 原文片段）。表里没有的新概念附层级和定义，全篇抽完后按第 1 步的规则补进表。
     程序核对：原文片段出自本段、名称来自概念表或会计层、出处为小节标题或前文时引用属实、指向估值倍数或公司估值时
     片段中有相应字样；不合格的带错误说明重抽。之后按两端的层级归类：
       - 从层号大的指向层号小的（或同在第 3-5 层）：进图；
       - 层号小的指向层号大的：记为反向关系，不进图；
       - 两端都是会计科目（或会计科目指向估值）：不新建边，和预置关系对得上的作为该关系的原文证据；
       - 第 3-5 层概念直接指向“公司估值”（如“看好出海，维持买入评级”）：不连边，留给第 3 步作为线索。
     缺影响因素的悬挂记录：被影响因素在本篇别处已有入边时直接并入；否则到提到它的段落和前两段里找（逐字引用，程序核对）。
  3. 概念补全：先删掉孤立的单个概念（第 1、2 步之后没有任何边的，即使原文说它影响估值，也不做补全）。
     再按第 3、4、5 层的顺序，找出还没有通往会计层的概念（只需处理这些子图的终点），
     先取提到它的段落，让大模型找原文写明的影响对象（只能在同层或层号更小、已经接入的概念和会计科目中选，逐字引用、程序核对）；
     找不到的，再让大模型按定义和本篇语境从同样的候选（外加同一批待连接的概念）中选出最直接影响的 1-2 个
     （闭集选择，不造新概念；同层只选明显的构成因素或直接驱动因素，否则从上一层开始找），记为“逻辑补全”边，
     程序检查不成环、每个概念最终都能连到会计层。先处理第 3 层，处理第 4 层时第 3 层已全部接入，所以一遍就能接通。
  4. 建图与收尾：预置的会计层（全部科目、科目之间的关系、科目到估值的边）和以上各类边合成一张图，有环时先删逻辑补全边。
     剪枝：任意两个节点之间既有直连、又有方向一致的更长链路时，删掉直连，其余链路都保留（不论边的类型）；
     更长的链路只有方向相反的，直连保留并标记（可能是方向抽错，也可能是两个渠道，删掉会让净影响翻转）；
     最后反复删掉没有任何入边的会计科目，留下的科目都有下层概念直接或经由其他科目连入。
  5. 全局去重：所有研报的图建好后，同一层的概念用“名称：定义”的 BGE 向量取近邻和名称互相包含的作为候选对，
     交给大模型判断是否同一种概念；同一篇研报里的两个概念不合并。只给节点加“全局名称”，不改图的结构。
  6. 输出：report_graphs.json（每篇研报的节点、边、原文证据、反向关系、被路径替换和被剔除的部分、概念表，以及全局名称表）、
     几张扁平的 parquet 表，并调用 src/plot_report_logic.py 生成可视化 HTML。
边的 origin：extracted 段内原文 / bridged 跨段原文（悬挂补全或概念补全，引文经程序核对）/
logic 逻辑补全（大模型按定义选择，没有原文）/ preset 预置的会计关系。

断点续跑：各步的大模型结果都追加缓存在输出目录的 *_cache.jsonl 中，重新运行只会补做缺少的部分
（段落文字、概念表、候选等输入变化时自动重新调用）。

用法：在 main.py 中设置 TASK = 'report_logic'
"""
import hashlib
import itertools
import json
import re
import threading
import time
import unicodedata
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import networkx as nx
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from src import fin_items, neutralize_llm
from src.neutralize_llm import ContentFilterError, FatalAPIError, LLMClient, load_env, log
from src.report_text import strip_risk

ROOT_DIR = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT_DIR / 'prompts' / 'report_logic_prompt.md'

ROOT, MULTIPLE = fin_items.ROOT, fin_items.MULTIPLE
ACCOUNTING_TEXT = fin_items.accounting_list()
LAYER_NAMES = {1: '公司估值', 2: '会计层', 3: '经营指标', 4: '公司经营情况', 5: '外部环境'}
CONCEPT_LAYERS = (3, 4, 5)  # 从原文抽取的三层
KINDS = ('指标', '事件', '状态')
# 指向估值倍数、公司估值时，原文片段中必须出现的字样
FIXED_CUES = {
    ROOT: (re.compile(r'目标价|评级|买入|增持|中性|减持|卖出|跑赢|跑输|优于大市|推荐|估值|市值|股价'), '目标价、评级、估值、股价等'),
    MULTIPLE: (re.compile(r'倍|P/?E|P/?B|P/?S|EV|溢价|折价|估值', re.I), '倍数、PE/PB、溢价等'),
}
DIRECTIONS = {'正向': 1, '负向': -1}
DIRECTION_ALIASES = {'正': '正向', '+': '正向', 'positive': '正向', '正相关': '正向',
                     '负': '负向', '-': '负向', 'negative': '负向', '负相关': '负向', '反向': '负向'}
SOURCES = ('本段', '小节标题', '前文')
PARENT_RELATIONS = ('加总', '相乘')
ORIGIN_PRIORITY = ('extracted', 'bridged', 'preset', 'logic')  # 同一条边有多种来源时取靠前的
TEXT_ORIGINS = ('extracted', 'bridged')  # 有原文证据的边
CYCLE_RANK = {'logic': 0, 'bridged': 2, 'extracted': 3, 'preset': 9}  # 去环时先删排在前面的
MAX_NODE_LEN = 20  # 概念名的最大字数，超过视为抽取错误
MAX_DEF_LEN = 60
MAX_EVIDENCE_LEN = 120
MAX_RETRIES = 4  # 单次调用输出不合格时的重试次数
DEFAULT_CONCURRENCY = 50

EMBED_MODEL = 'BAAI/bge-base-zh-v1.5'  # 中文句向量模型（BERT 结构），用 [CLS] 向量
TABLE_CHUNK = 3000  # 分层填表时每块原文的字数
PREV_SENTENCES, PREV_CHARS = 2, 150  # 逐段抽取时附带的前一段末尾：句数、字数上限
DANGLING_MAX_PARAS = 5  # 悬挂补全时最多附带的段落数
COMPLETE_MAX_PARAS = 6  # 概念补全（原文）时最多附带的段落数
COMPLETE_MAX_TARGETS = 2  # 概念补全时一个概念最多连到几个上层概念
LOGIC_BATCH = 25  # 逻辑补全时每次交给大模型的概念数（同一批的概念可以互相选）
GLOBAL_SIM = 0.80  # 全局去重：“名称：定义”的 BGE 余弦达到该值算候选
GLOBAL_TOPK = 5  # 每个名称最多取的近邻数
GLOBAL_BATCH = 12  # 每次交给大模型的候选对数

_lock = threading.Lock()
_embedder = {}


# ==================== 工具函数 ====================

def norm_name(s):
    """名称规范化：全角转半角、去引号书名号、去掉汉字两侧的空格和首尾标点。"""
    s = unicodedata.normalize('NFKC', str(s or '')).strip()
    s = re.sub(r'[“”"\'‘’「」『』《》]', '', s)
    s = re.sub(r'\s+', ' ', s)
    s = re.sub(r'(?<=[^\x00-\x7f]) | (?=[^\x00-\x7f])', '', s)
    return s.strip(' ，,。.;；:：、')


def name_key(s):
    """字面比较用的键：再去掉空格、连接符和括号，忽略英文大小写。"""
    return re.sub(r'[\s\-_·/、，,()]', '', norm_name(s)).casefold()


def norm_direction(d):
    d = str(d or '').strip()
    d = DIRECTION_ALIASES.get(d.lower(), d)
    return d if d in DIRECTIONS else None


def is_null(x):
    return x is None or norm_name(x) in ('', 'null', 'None', '无', '未说明', '原文未说明')


def as_bool(x):
    if isinstance(x, bool):
        return x
    if str(x).strip().lower() in ('true', '是', '1'):
        return True
    if str(x).strip().lower() in ('false', '否', '0'):
        return False
    raise ValueError(f'应为 true 或 false，收到 {x!r}')


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
    for k in ('建表-System', '建表-User', '补表-User', '归并-System', '归并-User', '抽取-System', '抽取-User',
              '悬挂-System', '悬挂-User', '补全原文-System', '补全原文-User', '补全逻辑-System', '补全逻辑-User',
              '归一-System', '归一-User'):
        if k not in prompts:
            raise SystemExit(f'{PROMPT_FILE} 中缺少 “## {k}” 一节')
    return prompts


def prompt_sig(prompts, *steps):
    """某几步 prompt 的指纹，放进缓存键里：改了 prompt 措辞后，这一步的缓存自动失效。"""
    return text_hash(*(prompts[k] for k in sorted(prompts) if k.startswith(steps)))


def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace('{' + k + '}', str(v))
    return template


def with_feedback(base):
    """重试时把上一次的错误附在 prompt 后面。"""
    return lambda err: base if err is None else f'{base}\n\n上一次的输出有以下问题，请修正后重新输出完整结果：{err}'


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


def squash(s):
    """比对引文用：全角转半角，去掉空白和引号。"""
    return re.sub(r'[\s“”"\'‘’「」『』]', '', unicodedata.normalize('NFKC', str(s or '')))


def quote_in(quote, text, min_len=6):
    """引文是否逐字出自原文：允许用省略号连接几段原文，每段都要能在原文中找到，合计至少 min_len 个字。"""
    parts = [p for p in re.split(r'…+|\.{3,}', squash(quote)) if p]
    return sum(map(len, parts)) >= min_len and all(p in squash(text) for p in parts)


def tail(text):
    """段落末尾的几句，供下一段理解指代。"""
    parts = [s for s in re.split(r'(?<=[。！？；])', text) if s.strip()]
    return ''.join(parts[-PREV_SENTENCES:])[-PREV_CHARS:]


class LLM:
    """在 LLMClient 外面加上：输出解析失败时重试（可把上次的错误告诉模型）、用量统计。"""

    def __init__(self, env):
        self.client = LLMClient(env)
        self.usage = Counter()

    def ask(self, system, user, parse, lenient=None):
        """user 可以是字符串，或接收上一次错误（首次为 None）返回 prompt 的函数。
        重试用完仍不合格时，若给了 lenient，用它从最后一次回复中取出合格的部分。"""
        err, content = None, None
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
        if lenient and content is not None:
            try:
                return lenient(content)
            except (ValueError, KeyError, TypeError, AttributeError):
                pass
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


# ==================== 第 1 步：概念分层填表 ====================
# 概念表：{规范名: {'aliases': [别名...], 'definition': 定义, 'layer': 3/4/5, 'kind': 类型, 'parent': 上位指标}}

def table_index(table):
    """名称或别名的字面键 -> 规范名。"""
    idx = {}
    for name in table:
        idx.setdefault(name_key(name), name)
    for name, t in table.items():
        for a in t['aliases']:
            idx.setdefault(name_key(a), name)
    return idx


def table_text(table):
    if not table:
        return '（空）'
    return '\n'.join(f'{n} ｜ 第{t["layer"]}层 ｜ {"、".join(t["aliases"]) or "—"} ｜ {t["definition"]}' for n, t in table.items())


def layer_of(name, table):
    """节点所在的层：公司估值为 1，会计科目和估值倍数为 2，其余查概念表。"""
    if name == ROOT:
        return 1
    if name in fin_items.ITEMS:
        return 2
    t = table.get(name)
    return t['layer'] if t else None


def resolve_name(raw, index):
    """名称 -> 预置节点或概念表中的规范名，都不是时返回 None。"""
    if is_null(raw):
        return None
    return fin_items.match(raw) or index.get(name_key(raw))


def parse_concept(it, errors):
    """校验一个新概念（填表、补表、逐段抽取共用），合格时返回 {'name', 'definition', 'layer', 'kind'}，否则记下错误返回 None。"""
    name, definition = norm_name(it.get('name')), norm_name(it.get('definition'))
    m = re.search(r'\d', str(it.get('layer') or ''))
    layer, kind = int(m.group()) if m else None, str(it.get('kind') or '').strip()
    if len(name) > MAX_NODE_LEN or not definition:
        errors.append(f'新概念“{name}”需要不超过 {MAX_NODE_LEN} 字的名称和一句定义')
    elif layer not in CONCEPT_LAYERS:
        errors.append(f'新概念“{name}”的 layer 必须是 3、4、5 之一（第 1、2 层是预置的；分部、地区的收入和毛利率放第 3 层）')
    elif layer != 3 and kind not in KINDS:
        errors.append(f'新概念“{name}”的 kind 必须是{"、".join(KINDS)}之一')
    else:
        return {'name': name, 'definition': definition[:MAX_DEF_LEN], 'layer': layer, 'kind': '指标' if layer == 3 else kind}
    return None


def parse_table(content, table):
    """校验填表（或补表）结果，返回 (新概念列表, [(别名, 规范名)])。与已有名称字面相同的新概念视为已有，会计科目不进表。"""
    obj = parse_json(content, dict)
    new, aliases = obj.get('new') or [], obj.get('aliases') or []
    if not isinstance(new, list) or not isinstance(aliases, list):
        raise ValueError('new 和 aliases 必须是数组')
    index, added, out_new, out_alias, errors = table_index(table), {}, [], [], []
    for it in new:
        if not isinstance(it, dict):
            continue
        name = norm_name(it.get('name'))
        k = name_key(name)
        if not name or fin_items.match(name) or k in index or k in added:
            continue
        v = parse_concept(it, errors)
        if v:
            added[k] = v['name']
            out_new.append(v)
    for it in aliases:
        if not isinstance(it, dict):
            continue
        alias, target = norm_name(it.get('alias')), norm_name(it.get('name'))
        canon = index.get(name_key(target)) or added.get(name_key(target))
        if canon is None:
            if not fin_items.match(target):  # 指向会计科目的别名不需要记
                errors.append(f'aliases 中“{alias}”指向的“{target}”不是表中的规范名')
        elif alias and len(alias) <= MAX_NODE_LEN and not fin_items.match(alias) and name_key(alias) != name_key(canon):
            out_alias.append((alias, canon))
    if errors:
        raise ValueError('；'.join(errors))
    return out_new, out_alias


def apply_table(table, new, aliases):
    for t in new:
        table.setdefault(t['name'], {'aliases': [], 'definition': t['definition'], 'layer': t['layer'], 'kind': t['kind'],
                                     'parent': ''})
    index = table_index(table)
    for alias, canon in aliases:
        if canon in table and name_key(alias) not in index:  # 已经属于别的概念的叫法不改
            table[canon]['aliases'].append(alias)
            index[name_key(alias)] = canon


def chunk_paras(ps):
    out, cur, n = [], [], 0
    for p in ps:
        if cur and n + len(p['text']) > TABLE_CHUNK:
            out.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += len(p['text'])
    return out + [cur] if cur else out


def build_tables(by_rid, metas, out_dir, llm, prompts, workers):
    """每篇研报按顺序分块通读、滚动更新概念表（块与块之间串行，研报之间并行）。返回 {report_id: 概念表}。"""
    cache_path = out_dir / 'table_cache.jsonl'
    cache = {r['key']: r for r in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '建表')
    n_chunks = {rid: len(chunk_paras(ps)) for rid, ps in by_rid.items()}
    log(f'概念分层填表：{len(by_rid)} 篇研报共 {sum(n_chunks.values())} 块（每块约 {TABLE_CHUNK} 字）')

    def work(rid):
        meta, table = metas[rid], {}
        for chunk in chunk_paras(by_rid[rid]):
            text = '\n\n'.join(f'[{p["para_id"]}] {p["text"]}' for p in chunk)
            cur = table_text(table)
            key = text_hash(sig, rid, text, cur)
            if key not in cache:
                base = fill(prompts['建表-User'], company=meta['company'] or '未知', title=meta['title'],
                            accounting=ACCOUNTING_TEXT, table=cur, text=text)
                try:
                    new, aliases = llm.ask(prompts['建表-System'], with_feedback(base), lambda c: parse_table(c, table))
                except RuntimeError as e:  # 这一块没建成，后面逐段抽取时出现的新概念还会补进表
                    log(f'!!! {rid} 有一块填表失败：{e}')
                    continue
                cache[key] = {'key': key, 'report_id': rid, 'new': new, 'aliases': aliases}
                append_jsonl(cache_path, cache[key])
            apply_table(table, cache[key]['new'], [tuple(a) for a in cache[key]['aliases']])
        return table

    rids = sorted(by_rid)
    tables = dict(zip(rids, run_parallel(rids, work, workers, '概念分层填表')))
    tables = {rid: t or {} for rid, t in tables.items()}
    by_layer = Counter(t['layer'] for tb in tables.values() for t in tb.values())
    log(f'概念分层填表完成：每篇平均 {np.mean([len(t) for t in tables.values()]):.0f} 个概念（第 3/4/5 层共 '
        f'{by_layer[3]}/{by_layer[4]}/{by_layer[5]} 个），共 {sum(len(t["aliases"]) for tb in tables.values() for t in tb.values())} 个别名')
    return tables


def supplement_tables(new_vars, tables, metas, out_dir, llm, prompts, workers):
    """把逐段抽取时出现的新概念按填表的规则补进概念表。返回 {report_id: {新名称: 规范名}}。"""
    cache_path = out_dir / 'table_cache.jsonl'
    cache = {r['key']: r for r in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '建表', '补表')
    rids = sorted(rid for rid in new_vars if new_vars[rid])
    log(f'补表：{len(rids)} 篇研报共 {sum(len(new_vars[r]) for r in rids)} 个新概念')

    def work(rid):
        table, cands = tables[rid], new_vars[rid]
        text = '\n'.join(f'{i}. {v["name"]} ｜ 第{v["layer"]}层 ｜ {v["kind"]} ｜ {v["definition"]}' for i, v in enumerate(cands, 1))
        cur = table_text(table)
        key = text_hash(sig, rid, cur, text)
        if key not in cache:
            base = fill(prompts['补表-User'], company=metas[rid]['company'] or '未知', title=metas[rid]['title'],
                        accounting=ACCOUNTING_TEXT, table=cur, candidates=text)
            try:
                new, aliases = llm.ask(prompts['建表-System'], with_feedback(base), lambda c: parse_table(c, table))
            except RuntimeError as e:
                log(f'!!! {rid} 补表失败，新概念全部按新建处理：{e}')
                new, aliases = [], []
            cache[key] = {'key': key, 'report_id': rid, 'new': new, 'aliases': aliases}
            append_jsonl(cache_path, cache[key])
        row = cache[key]
        defs = {name_key(t['name']): t for t in row['new']}
        alias_of = {name_key(a): c for a, c in row['aliases']}
        rename, new = {}, []
        for v in cands:
            k = name_key(v['name'])
            if k in alias_of and alias_of[k] != v['name']:
                rename[v['name']] = alias_of[k]
            else:
                rename[v['name']] = v['name']
                new.append({**v, **{f: defs[k][f] for f in ('definition', 'layer', 'kind') if k in defs and defs[k][f]}})
        apply_table(table, new, [(v['name'], c) for v, c in ((v, rename[v['name']]) for v in cands) if c != v['name']])
        return rename

    return {rid: r or {v['name']: v['name'] for v in new_vars[rid]}
            for rid, r in zip(rids, run_parallel(rids, work, workers, '补表'))}


def parse_parents(content, table):
    """校验第 3 层上位指标的整理结果，返回 [{'name', 'parent', 'relation', 'reason'}]。"""
    items = parse_json(content, dict).get('parents') or []
    if not isinstance(items, list):
        raise ValueError('parents 必须是数组')
    index, out, errors = table_index(table), [], []
    for it in items:
        if not isinstance(it, dict):
            continue
        child, parent = index.get(name_key(it.get('name'))), index.get(name_key(it.get('parent')))
        rel = str(it.get('relation') or '').strip()
        if not child or not parent or child == parent or table[child]['layer'] != 3 or table[parent]['layer'] != 3:
            errors.append(f'“{norm_name(it.get("name"))}”和“{norm_name(it.get("parent"))}”必须是列表中两个不同的指标，名称原样书写')
        elif rel not in PARENT_RELATIONS:
            errors.append(f'“{child}”的 relation 必须是{"或".join(PARENT_RELATIONS)}')
        else:
            out.append({'name': child, 'parent': parent, 'relation': rel, 'reason': norm_name(it.get('reason'))[:MAX_EVIDENCE_LEN]})
    if errors:
        raise ValueError('；'.join(errors))
    return out


def assign_parents(tables, metas, out_dir, llm, prompts, workers):
    """第 1 步的最后一环：每篇研报调用一次大模型，给第 3 层指标找上位指标（原地写进概念表），会成环的跳过。"""
    cache_path = out_dir / 'parent_cache.jsonl'
    cache = {r['key']: r['parents'] for r in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '归并')
    rids = sorted(rid for rid, t in tables.items() if sum(x['layer'] == 3 for x in t.values()) >= 2)

    def work(rid):
        table = tables[rid]
        text = '\n'.join(f'{n} ｜ {t["definition"]}' for n, t in table.items() if t['layer'] == 3)
        key = text_hash(sig, rid, text)
        if key not in cache:
            base = fill(prompts['归并-User'], company=metas[rid]['company'] or '未知', title=metas[rid]['title'], metrics=text)
            try:
                res = llm.ask(prompts['归并-System'], with_feedback(base), lambda c: parse_parents(c, table))
            except RuntimeError as e:
                log(f'!!! {rid} 整理上位指标失败：{e}')
                return 0
            cache[key] = res
            append_jsonl(cache_path, {'key': key, 'report_id': rid, 'parents': res})
        n = 0
        for it in cache[key]:
            child, parent = it['name'], it['parent']
            if child not in table or parent not in table or table[child]['parent']:
                continue
            cur = parent  # 沿上位指标往上走，走回自己就是成环
            while cur and cur != child:
                cur = table[cur]['parent']
            if not cur:
                table[child].update(parent=parent, parent_relation=it['relation'], parent_reason=it['reason'])
                n += 1
        return n

    counts = run_parallel(rids, work, workers, '整理上位指标')
    log(f'整理上位指标完成：{len(rids)} 篇研报共设上位指标 {sum(c or 0 for c in counts)} 个')


# ==================== 第 2 步：原文连边 ====================

def check_relations(obj, p, table):
    """校验一段的抽取结果，返回 (合格的关系, 用到的新概念, 错误说明)。"""
    rels_in, vars_in = obj.get('relations'), obj.get('new_vars') or []
    if not isinstance(rels_in, list) or not isinstance(vars_in, list):
        return [], [], ['输出必须是 {"relations": [...], "new_vars": [...]}']
    index, new_vars, errors = table_index(table), {}, []
    for v in vars_in:
        if not isinstance(v, dict):
            continue
        name = norm_name(v.get('name'))
        k = name_key(name)
        if not name or k in index or fin_items.match(name):
            continue
        c = parse_concept(v, errors)
        if c:
            new_vars[k] = c

    def resolve(raw):
        if is_null(raw):
            return None, None
        name = resolve_name(raw, index) or (new_vars[name_key(raw)]['name'] if name_key(raw) in new_vars else None)
        if name is None:
            return None, (f'“{norm_name(raw)}”既不在概念表中，也不是会计科目：请改用表中的名称；确属新概念的，'
                          f'在 new_vars 中给出名称、层级、类型和定义')
        return name, None

    rels, seen = [], set()
    for i, r in enumerate(rels_in, 1):
        if not isinstance(r, dict):
            continue
        (cause, e1), (effect, e2) = resolve(r.get('cause')), resolve(r.get('effect'))
        direction, ev = norm_direction(r.get('direction')), norm_name(r.get('evidence'))
        problems = [e for e in (e1, e2) if e]
        if not problems and (cause is None and effect is None or cause == effect or cause == ROOT):
            continue  # 两端都空、自环、从“公司估值”出发的，直接丢弃
        if direction is None:
            problems.append('direction 必须是正向或负向')
        if not quote_in(ev, p['text'], min_len=4):
            problems.append('evidence 在本段中找不到，请从段落中逐字摘录（可以用……省略中间部分）')
        elif effect in FIXED_CUES and not FIXED_CUES[effect][0].search(ev):
            problems.append(f'被影响因素为“{effect}”时，evidence 中必须有{FIXED_CUES[effect][1]}字样；原文没有这样说，就不要连到“{effect}”')
        context = []
        for side, name, label in (('cause', cause, '影响因素'), ('effect', effect, '被影响因素')):
            src = str(r.get(f'{side}_from') or '本段').strip()
            if name is None or src not in SOURCES[1:]:
                continue
            where = (p.get('section') or '') if src == '小节标题' else p.get('prev_tail', '')
            q = squash(r.get('from_quote'))
            if len(q) < 2 or q not in squash(where):
                problems.append(f'{side}_from 为{src}时，from_quote 必须逐字摘自{src}')
            else:
                context.append(f'{label}“{name}”出自{src}：“{norm_name(r.get("from_quote"))}”')
        if problems:
            errors += [f'第{i}条：{x}' for x in problems]
            continue
        if (cause, effect, direction) in seen:
            continue
        seen.add((cause, effect, direction))
        rels.append({'cause': cause, 'direction': direction, 'effect': effect, 'evidence': ev[:MAX_EVIDENCE_LEN],
                     'context': context})
    used = {n for r in rels for n in (r['cause'], r['effect'])}
    return rels, [v for v in new_vars.values() if v['name'] in used], errors


def extract_relations(by_rid, tables, metas, out_dir, llm, prompts, workers):
    """每个段落调用一次大模型。返回 (关系列表, {report_id: [新概念...]}, {report_id: 校验不合格重抽次数})。"""
    cache_path = out_dir / 'extract_cache.jsonl'
    cache = {r['key']: r for r in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '抽取')
    paras, ttext = [], {rid: table_text(tables[rid]) for rid in by_rid}
    for rid, ps in by_rid.items():
        for p in ps:
            p['key'] = text_hash(sig, rid, str(p['para_id']), p['text'], p.get('section') or '', p['prev_tail'], ttext[rid],
                                 ACCOUNTING_TEXT)
            paras.append(p)
    todo = [p for p in paras if p['key'] not in cache]
    log(f'原文连边：共 {len(paras)} 段，已缓存 {len(paras) - len(todo)} 段，本次调用 {len(todo)} 段')

    def work(p):
        rid, table, bad = p['report_id'], tables[p['report_id']], Counter()

        def parse(content):
            rels, new, errors = check_relations(parse_json(content, dict), p, table)
            if errors:
                bad['n'] += 1
                raise ValueError('；'.join(errors[:10]))
            return rels, new

        base = fill(prompts['抽取-User'], company=metas[rid]['company'] or '未知', title=metas[rid]['title'],
                    section=p.get('section') or '无', accounting=ACCOUNTING_TEXT, table=ttext[rid],
                    prev=p['prev_tail'] or '（无）', text=p['text'])
        rels, new = llm.ask(prompts['抽取-System'], with_feedback(base), parse,
                            lenient=lambda c: check_relations(parse_json(c, dict), p, table)[:2])
        row = {'key': p['key'], 'report_id': rid, 'para_id': p['para_id'], 'relations': rels, 'new_vars': new,
               'rejected': bad['n']}
        append_jsonl(cache_path, row)  # 只缓存成功的段落，失败的下次重跑
        return row

    for row in run_parallel(todo, work, workers, '原文连边'):
        if row:
            cache[row['key']] = row
    rels, new_vars, rejected, failed = [], defaultdict(dict), Counter(), 0
    for p in paras:
        row = cache.get(p['key'])
        if row is None:
            failed += 1
            continue
        rels += [{'report_id': p['report_id'], 'para_id': p['para_id'], **r} for r in row['relations']]
        for v in row['new_vars']:
            new_vars[p['report_id']].setdefault(name_key(v['name']), v)
        rejected[p['report_id']] += row['rejected']
    if failed:
        log(f'!!! {failed} 段抽取失败，这些段落没有关系；重新运行会再试')
    n_dangling = sum(r['cause'] is None or r['effect'] is None for r in rels)
    log(f'原文连边完成：{len(rels)} 条关系（其中悬挂记录 {n_dangling} 条），程序校验不合格重抽 {sum(rejected.values())} 次')
    return rels, {rid: list(v.values()) for rid, v in new_vars.items()}, rejected


def classify(r, table):
    """按两端的层级给一条完整的关系归类：edge 进图 / accounting 两端都是会计科目（或会计科目指向估值）/
    to_root 第 3-5 层概念直接指向公司估值 / reverse 层号小的指向层号大的。"""
    lc, le = layer_of(r['cause'], table), layer_of(r['effect'], table)
    if lc == 2 and le <= 2:
        return 'accounting'
    if lc is None or le is None or lc < le:
        return 'reverse'
    return 'to_root' if le == 1 else 'edge'


def note_of(r):
    """悬挂记录或线索的简要信息，存进节点。"""
    return {'para_id': r['para_id'] if 'para_id' in r else r['para_ids'][0], 'direction': r['direction'],
            'evidence': r['evidence'] if isinstance(r['evidence'], str) else r['evidence'][0]}


def dangling_paras(r, var, ps, table):
    """缺影响因素的悬挂记录可供查找的段落：先取同一小节的前两段，再按距离远近加上提到该变量（规范名或别名）的段落，
    最多 DANGLING_MAX_PARAS 段，不含所在段落。"""
    pos = {p['para_id']: i for i, p in enumerate(ps)}
    i = pos[r['para_id']]
    near = [q for q in ps[max(0, i - 2):i][::-1] if q.get('section') == ps[i].get('section')] or ps[max(0, i - 1):i]
    fixed = fin_items.ITEMS.get(var, ('', ''))[0].split()
    names = [squash(n) for n in [var, *table.get(var, {}).get('aliases', []), *fixed] if len(squash(n)) >= 2]
    mention = sorted((q for q in ps if q is not ps[i] and any(n in squash(q['text']) for n in names)),
                     key=lambda q: abs(pos[q['para_id']] - i))
    out = []
    for q in near + mention:
        if q not in out:
            out.append(q)
    return out[:DANGLING_MAX_PARAS]


def parse_dangling(content, var, texts, index):
    """校验悬挂补全结果，返回 None（没找到）或 {'name', 'direction', 'para_id', 'quote', 'reason'}。"""
    obj = parse_json(content, dict)
    if not as_bool(obj.get('found', False)):
        return None
    raw = norm_name(obj.get('name'))
    name = resolve_name(raw, index)
    d = norm_direction(obj.get('direction'))
    try:
        pid = int(obj.get('para_id'))
    except (TypeError, ValueError):
        pid = None
    errors = []
    if name is None:
        errors.append(f'“{raw}”不在概念表中，也不是会计科目')
    elif name in (var, ROOT):
        errors.append(f'“{raw}”不能作为缺失的影响因素')
    if d is None:
        errors.append('direction 必须是正向或负向')
    if pid not in texts or not quote_in(obj.get('quote'), texts[pid], min_len=4):
        errors.append(f'引文在第 {obj.get("para_id")} 段中找不到，请从所给段落中逐字摘录；找不到就回答没有找到')
    if errors:
        raise ValueError('；'.join(errors))
    return {'name': name, 'direction': d, 'para_id': pid, 'quote': norm_name(obj.get('quote'))[:MAX_EVIDENCE_LEN],
            'reason': norm_name(obj.get('reason'))[:MAX_EVIDENCE_LEN]}


def resolve_dangling(dangling, complete, by_rid, tables, metas, out_dir, llm, prompts, workers):
    """处理缺影响因素的悬挂记录（缺被影响因素的留给第 3 步概念补全）。
    返回 (补出的跨段关系, {report_id: {变量: [并入的记录]}}, {report_id: {变量: [未补上的记录]}})。"""
    ins = defaultdict(set)
    for r in complete:
        ins[r['report_id']].add(r['effect'])
    attached, unresolved = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list))
    tasks = []
    for r in dangling:
        rid, var = r['report_id'], r['effect']
        if var in ins[rid]:  # 本篇别处已有入边：作为补充证据并入
            attached[rid][var].append(note_of(r))
            continue
        cands = dangling_paras(r, var, by_rid[rid], tables[rid])
        if not cands:
            unresolved[rid][var].append(note_of(r))
            continue
        tasks.append((r, var, cands))
    log(f'悬挂补全（缺影响因素）：程序直接并入 {sum(len(v) for d in attached.values() for v in d.values())} 条，'
        f'交给大模型查找 {len(tasks)} 条')

    cache_path = out_dir / 'dangling_cache.jsonl'
    cache = {row['key']: row['result'] for row in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '悬挂')

    def work(task):
        r, var, cands = task
        rid, meta, table = r['report_id'], metas[r['report_id']], tables[r['report_id']]
        source = next(p['text'] for p in by_rid[rid] if p['para_id'] == r['para_id'])
        texts = {p['para_id']: p['text'] for p in cands}
        record = f'影响因素：（原文未写明）；被影响因素：{var}；方向：{r["direction"]}；原文：“{r["evidence"]}”'
        block = '\n\n'.join(f'[{pid}] {t}' for pid, t in texts.items())
        key = text_hash(sig, rid, record, source, block)
        if key not in cache:
            base = fill(prompts['悬挂-User'], company=meta['company'] or '未知', title=meta['title'],
                        names='、'.join([*table, *(n for n in fin_items.ITEMS if n != ROOT)]), record=record,
                        pid=r['para_id'], source=source, paragraphs=block)
            index = table_index(table)
            try:
                res = llm.ask(prompts['悬挂-System'], with_feedback(base), lambda c: parse_dangling(c, var, texts, index),
                              lenient=lambda c: None)
            except RuntimeError as e:
                log(f'!!! {rid} 第 {r["para_id"]} 段的悬挂记录补全失败，按未说明处理：{e}')
                return None
            cache[key] = res
            append_jsonl(cache_path, {'key': key, 'report_id': rid, 'result': res})
        return cache[key]

    extra = []
    for (r, var, _), res in zip(tasks, run_parallel(tasks, work, workers, '悬挂补全')):
        rid = r['report_id']
        if not res:
            unresolved[rid][var].append(note_of(r))
            continue
        extra.append({'report_id': rid, 'cause': res['name'], 'direction': res['direction'], 'effect': var, 'origin': 'bridged',
                      'evidence': [r['evidence'], res['quote']], 'para_ids': sorted({r['para_id'], res['para_id']}),
                      'context': [f'悬挂补全：{res["reason"]}'] if res['reason'] else []})
    log(f'悬挂补全完成：大模型补出 {len(extra)} 条跨段关系，'
        f'未说明影响因素 {sum(len(v) for d in unresolved.values() for v in d.values())} 条')
    return extra, attached, unresolved


# ==================== 建图（第 3、4 步共用） ====================

def preset_items():
    return [{'cause': u, 'effect': v, 'direction': '正向' if s > 0 else '负向', 'origin': 'preset', 'evidence': [],
             'para_ids': [], 'context': []} for u, v, s in fin_items.PRESET_EDGES]


def top_parents(table):
    """第 3 层上位指标树：每个下位指标 -> 它所在的树最上层的指标（没有上位指标的不列）。"""
    tops = {}
    for n in table:
        cur = n
        while table.get(cur, {}).get('parent'):
            cur = table[cur]['parent']
        if cur != n:
            tops[n] = cur
    return tops


def merge_up(r, tops):
    """把关系两端的下位指标换成最上层的指标（向上合并），换过的在 context 里记下原来的说法。"""
    c, e = tops.get(r['cause'], r['cause']), tops.get(r['effect'], r['effect'])
    if (c, e) == (r['cause'], r['effect']):
        return r
    return {**r, 'cause': c, 'effect': e, 'context': [*r.get('context', []), f'合并前：{r["cause"]} → {r["effect"]}']}


def build_graph(items):
    """一篇研报的各类关系 -> DAG。items 的 evidence、para_ids 为列表。返回 (图, 诊断计数, 为去环删掉的边)。"""
    agg, info = {}, Counter()
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
        removed.append({'source': u, 'target': v, 'direction': '正向' if g.edges[u, v]['sign'] > 0 else '负向',
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


# ==================== 第 3 步：概念补全 ====================

FIN_ORDER = {n: i for i, n in enumerate(fin_items.ITEMS)}


def candidates_text(cands, lay, layer):
    """候选概念按层列出：同层在前，然后从上一层往上；会计科目按清单顺序排。"""
    by = defaultdict(list)
    for n in cands:
        by[lay[n]].append(n)
    return '\n'.join(f'第 {k} 层（{LAYER_NAMES[k]}{"，同层" if k == layer else ""}）：'
                     f'{"、".join(sorted(by[k], key=lambda n: (FIN_ORDER.get(n, -1), n)))}' for k in sorted(by, reverse=True))


def mention_paras(name, table, ps):
    names = [squash(x) for x in [name, *table.get(name, {}).get('aliases', [])] if len(squash(x)) >= 2]
    return [q for q in ps if any(x in squash(q['text']) for x in names)]


def complete_paras(name, g, table, ps, notes):
    """概念补全时查找原文的段落：线索所在段落、它作为被影响因素出现过的段落、提到它的段落，各自再带上后一段；
    最多 COMPLETE_MAX_PARAS 段，按段落顺序排列。"""
    pos = {p['para_id']: i for i, p in enumerate(ps)}
    first = ([x['para_id'] for x in notes] + [pid for _, _, d in g.in_edges(name, data=True) for pid in d['para_ids']]
             + [q['para_id'] for q in mention_paras(name, table, ps)])
    out = []
    for pid in first:
        if pid in pos:
            out += [q['para_id'] for q in ps[pos[pid]:pos[pid] + 2] if q['para_id'] not in out]
    return [ps[pos[pid]] for pid in sorted(out[:COMPLETE_MAX_PARAS], key=pos.get)]


def snippet(name, g, notes, ps, table):
    """逻辑补全时附带的一句原文：线索、入边的原文证据，或第一次提到它的句子。"""
    if notes:
        return notes[0]['evidence']
    for _, _, d in g.in_edges(name, data=True):
        if d['evidence']:
            return d['evidence'][0]
    names = [squash(x) for x in [name, *table.get(name, {}).get('aliases', [])] if len(squash(x)) >= 2]
    for p in mention_paras(name, table, ps):
        for s in re.split(r'(?<=[。！？；])', p['text']):
            if any(x in squash(s) for x in names):
                return s.strip()[:80]
    return '—'


def parse_complete_text(content, cands, texts, index):
    """校验概念补全（原文）结果，返回 None（没找到）或 {'links': [{'name', 'direction', 'para_id', 'quote'}], 'reason'}。"""
    obj = parse_json(content, dict)
    if not as_bool(obj.get('found', False)):
        return None
    links_in = obj.get('links')
    if not isinstance(links_in, list) or not links_in:
        raise ValueError('found 为 true 时 links 不能为空')
    links, errors = [], []
    for it in links_in[:COMPLETE_MAX_TARGETS]:
        if not isinstance(it, dict):
            continue
        raw, d = norm_name(it.get('name')), norm_direction(it.get('direction'))
        name = resolve_name(raw, index)
        try:
            pid = int(it.get('para_id'))
        except (TypeError, ValueError):
            pid = None
        if name not in cands:
            errors.append(f'“{raw}”不在候选概念中')
        elif d is None:
            errors.append(f'“{raw}”的 direction 必须是正向或负向')
        elif pid not in texts or not quote_in(it.get('quote'), texts[pid], min_len=4):
            errors.append(f'“{raw}”的引文在第 {it.get("para_id")} 段中找不到，请从所给段落中逐字摘录；找不到就回答没有找到')
        elif name in FIXED_CUES and not FIXED_CUES[name][0].search(str(it.get('quote'))):
            errors.append(f'连到“{name}”时引文中必须有{FIXED_CUES[name][1]}字样')
        elif name not in [x['name'] for x in links]:
            links.append({'name': name, 'direction': d, 'para_id': pid, 'quote': norm_name(it.get('quote'))[:MAX_EVIDENCE_LEN]})
    if errors:
        raise ValueError('；'.join(errors))
    return {'links': links, 'reason': norm_name(obj.get('reason'))[:MAX_EVIDENCE_LEN]} if links else None


def parse_complete_logic(content, items, index, g):
    """校验逻辑补全结果。items 为 [(概念, 候选集合)]，返回与之等长的 [[概念, [{'name', 'direction'}], 理由]]。
    同一批的概念可以互相选，所以最后把所选的边加到 g 的副本上，检查没有环、每个概念都能连到公司估值。"""
    links_in = parse_json(content, dict).get('links')
    if not isinstance(links_in, list):
        raise ValueError('缺少 links 数组')
    out, errors = {}, []
    for it in links_in:
        try:
            i = int(it.get('id'))
        except (TypeError, ValueError, AttributeError):
            errors.append('有一条缺少合法的 id')
            continue
        if not 1 <= i <= len(items):
            errors.append(f'编号 {i} 不存在')
            continue
        n, cands = items[i - 1]
        links, bad = [], []
        for t in (it.get('targets') or [])[:COMPLETE_MAX_TARGETS]:
            if not isinstance(t, dict):
                continue
            raw, d = norm_name(t.get('name')), norm_direction(t.get('direction'))
            name = resolve_name(raw, index)
            if name not in cands:
                bad.append(f'“{raw}”不在候选中，或会与“{n}”的上游形成环')
            elif d is None:
                bad.append(f'“{raw}”的 direction 必须是正向或负向')
            elif name not in [x['name'] for x in links]:
                links.append({'name': name, 'direction': d})
        if bad:
            errors.append(f'第{i}个概念：' + '；'.join(bad))
        elif not links:
            errors.append(f'第{i}个概念至少要选一个上层概念')
        else:
            out[i] = [n, links, norm_name(it.get('reason'))[:MAX_EVIDENCE_LEN]]
    missing = [str(i) for i in range(1, len(items) + 1) if i not in out]
    if missing and not errors:
        errors.append(f'缺少这些编号：{"、".join(missing)}')
    if not errors:
        h = g.copy()
        h.add_edges_from((n, x['name']) for n, links, _ in out.values() for x in links)
        try:
            cycle = nx.find_cycle(h)
            errors.append(f'所选的关系形成了环（{"→".join(u for u, _ in cycle)}），请改选')
        except nx.NetworkXNoCycle:
            conn = connected(h)
            errors += [f'第{i}个概念“{n}”沿所选的概念连不到会计层，请改选已连到会计层的概念' for i, (n, _) in enumerate(items, 1)
                       if n not in conn]
    if errors:
        raise ValueError('；'.join(errors))
    return [out[i] for i in range(1, len(items) + 1)]


def complete_concepts(graphs, tables, hints, by_rid, metas, out_dir, llm, prompts, workers):
    """第 3 步：按第 3、4、5 层的顺序，把还没有通往会计层的概念接上去（原地给 graphs 加边）。
    hints 为 {report_id: {概念: [原文只写了它、没写影响对象的记录]}}。返回 {report_id: Counter}。"""
    text_path, logic_path = out_dir / 'complete_text_cache.jsonl', out_dir / 'complete_logic_cache.jsonl'
    text_cache = {r['key']: r['result'] for r in load_jsonl(text_path)}
    logic_cache = {r['key']: r['result'] for r in load_jsonl(logic_path)}
    sig_t, sig_l = prompt_sig(prompts, '补全原文'), prompt_sig(prompts, '补全逻辑')
    counts, seq = defaultdict(Counter), itertools.count(10 ** 9)  # 补出的边排在原文边之后

    def add(rid, u, v, direction, origin, evidence, pids, context):
        sign = DIRECTIONS[direction]
        graphs[rid].add_edge(u, v, sign=sign, origin=origin, weight=1, n_pos=int(sign > 0), n_neg=int(sign < 0),
                             evidence=evidence, para_ids=pids, context=context, order=next(seq))
        counts[rid][origin] += 1

    def work_text(task):
        rid, n, cands, layer = task
        table, g, notes = tables[rid], graphs[rid], hints[rid].get(n, [])
        paras = complete_paras(n, g, table, by_rid[rid], notes)
        if not paras or not cands:
            return None
        texts = {p['para_id']: p['text'] for p in paras}
        block = '\n\n'.join(f'[{pid}] {t}' for pid, t in texts.items())
        ctext = candidates_text(cands, layer_map(g, table), layer)
        hint = '；'.join(f'第 {x["para_id"]} 段“{x["evidence"]}”' for x in notes) or '无'
        key = text_hash(sig_t, rid, n, ctext, block, hint)
        if key not in text_cache:
            base = fill(prompts['补全原文-User'], company=metas[rid]['company'] or '未知', title=metas[rid]['title'],
                        concept=n, layer=layer, layer_name=LAYER_NAMES[layer], definition=table.get(n, {}).get('definition', ''),
                        hint=hint, candidates=ctext, paragraphs=block)
            index = table_index(table)
            try:
                res = llm.ask(prompts['补全原文-System'], with_feedback(base),
                              lambda c: parse_complete_text(c, cands, texts, index), lenient=lambda c: None)
            except RuntimeError as e:
                log(f'!!! {rid} “{n}”的概念补全（原文）失败，改用逻辑补全：{e}')
                return None
            text_cache[key] = res
            append_jsonl(text_path, {'key': key, 'report_id': rid, 'concept': n, 'result': res})
        return text_cache[key]

    def work_logic(batch):
        rid, items, layer = batch
        table, g = tables[rid], graphs[rid]
        mates = {n for n, _ in items}  # 同一批的概念可以互相选
        items = [(n, c | (mates - {n})) for n, c in items]
        cands = set().union(*(c for _, c in items))
        if not cands:
            return None
        ctext = candidates_text(cands, layer_map(g, table), layer)
        lines = '\n'.join(f'{i}. {n} ｜ {table.get(n, {}).get("definition") or "—"} ｜ '
                          f'{snippet(n, g, hints[rid].get(n, []), by_rid[rid], table)}' for i, (n, _) in enumerate(items, 1))
        key = text_hash(sig_l, rid, lines, ctext)
        if key not in logic_cache:
            base = fill(prompts['补全逻辑-User'], company=metas[rid]['company'] or '未知', title=metas[rid]['title'],
                        layer=layer, concepts=lines, candidates=ctext)
            index = table_index(table)
            try:
                res = llm.ask(prompts['补全逻辑-System'], with_feedback(base), lambda c: parse_complete_logic(c, items, index, g))
            except RuntimeError as e:
                log(f'!!! {rid} 有一批逻辑补全（{len(items)} 个概念）失败，这些概念保持未接入：{e}')
                return None
            logic_cache[key] = res
            append_jsonl(logic_path, {'key': key, 'report_id': rid, 'result': res})
        return logic_cache[key]

    def candidates(g, table, n, layer):
        """同层或更上层、已经接入的概念，去掉 n 的上游（选它们会成环）。"""
        lay = layer_map(g, table)
        return {m for m in connected(g) if m != ROOT and lay[m] <= layer} - nx.ancestors(g, n)

    for layer in CONCEPT_LAYERS:
        tasks = []  # (report_id, 概念, 候选概念, 层)
        for rid in sorted(graphs):
            g = graphs[rid]
            lay, conn = layer_map(g, tables[rid]), connected(g)
            todo = {n for n in g if lay[n] == layer and n not in conn}
            for n in sorted(todo):
                if not any(s in todo for s in g.successors(n)):  # 只处理未接入部分的终点
                    tasks.append((rid, n, candidates(g, tables[rid], n, layer), layer))
        rest = defaultdict(list)
        for (rid, n, cands, _), res in zip(tasks, run_parallel(tasks, work_text, workers, f'第 {layer} 层概念补全（原文）')):
            if res:
                for x in res['links']:
                    add(rid, n, x['name'], x['direction'], 'bridged', [x['quote']], [x['para_id']],
                        ['概念补全（原文）' + (f'：{res["reason"]}' if res['reason'] else '')])
            else:
                rest[rid].append(n)
        # 原文找到的概念已经接入，可以作为其余概念的候选，所以重新算一遍候选
        rest = {rid: [(n, candidates(graphs[rid], tables[rid], n, layer)) for n in ns] for rid, ns in rest.items()}
        batches = [(rid, items[i:i + LOGIC_BATCH], layer) for rid, items in rest.items()
                   for i in range(0, len(items), LOGIC_BATCH)]
        for (rid, _, _), res in zip(batches, run_parallel(batches, work_logic, workers, f'第 {layer} 层概念补全（逻辑）')):
            for n, links, reason in res or []:
                for x in links:
                    add(rid, n, x['name'], x['direction'], 'logic', [], [], [reason] if reason else [])
        log(f'第 {layer} 层概念补全：{len(tasks)} 个未接入的终点，原文找到 {len(tasks) - sum(map(len, rest.values()))} 个，'
            f'逻辑补全 {sum(map(len, rest.values()))} 个（{len(batches)} 批）')
    return counts


# ==================== 第 4 步：剪枝 ====================

def other_path(g, u, v, sign=None):
    """DAG g 中不经过直连 u→v、从 u 到 v 的一条更长路径（广度优先，取最短的那条），返回节点列表或 None；
    sign 不为 None 时只找各段方向相乘等于 sign 的路径。按 (节点, 累计方向) 搜索，DAG 中找到的一定是简单路径。"""
    start = (u, 1)
    parent, queue = {start: None}, deque([start])
    while queue:
        n, s = queue.popleft()
        for m in g.successors(n):
            if n == u and m == v:
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


def prune_shortcuts(g):
    """剪枝：任意两个节点之间既有直连、又有方向一致的更长链路时，删掉直连，其余链路都保留（不论各段是哪类边）。
    更长的链路只有方向相反的，直连保留，并在边上记下 conflict_path（可能是某条边的方向抽错了，也可能是两个不同的渠道，
    删掉直连会让净影响翻转）。
    删掉方向一致的直连不改变任何两点之间的可达性和可能的方向，所以逐条删即可。返回被删的直连，各附一条代替它的链路。"""
    removed = []
    for u, v in sorted(g.edges, key=lambda e: g.edges[e]['order']):
        d = g.edges[u, v]
        path = other_path(g, u, v, d['sign'])
        if path:
            removed.append({'source': u, 'target': v, 'direction': '正向' if d['sign'] > 0 else '负向', 'origin': d['origin'],
                            'evidence': d['evidence'], 'para_ids': d['para_ids'], 'context': d.get('context', []), 'path': path})
            g.remove_edge(u, v)
    for u, v, d in g.edges(data=True):  # 剩下的直连如果还有更长的链路，方向一定相反
        d['conflict_path'] = other_path(g, u, v) or []
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


# ==================== 第 5 步：全局去重 ====================

def parse_same(content, n):
    items = parse_json(content, dict).get('decisions')
    if not isinstance(items, list):
        raise ValueError('缺少 decisions 数组')
    out = {}
    for it in items:
        try:
            i = int(it.get('id'))
        except (TypeError, ValueError, AttributeError):
            continue
        if 1 <= i <= n:
            out[i] = (as_bool(it.get('same')), norm_name(it.get('reason'))[:MAX_EVIDENCE_LEN])
    missing = [str(i) for i in range(1, n + 1) if i not in out]
    if missing:
        raise ValueError(f'缺少这些编号的判断：{"、".join(missing)}')
    return [out[i] for i in range(1, n + 1)]


def global_dedup(graphs, tables, embed_model, out_dir, llm, prompts, workers):
    """统一不同研报对同一种概念的叫法（只比较同一层的概念）。返回 ({名称: 全局名称}, 全局名称表)。
    字面相同的名称本来就统一；同一篇研报里的两个不同名称（填表时判定为不同概念）不会被并到一起；预置节点不参与。"""
    info = {}
    for rid, g in graphs.items():
        for n in g:
            t = tables[rid].get(n)
            if not t:  # 公司估值和会计层
                continue
            e = info.setdefault(n, {'reports': set(), 'definition': t['definition'], 'layer': t['layer']})
            e['reports'].add(rid)
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

    cache_path = out_dir / 'global_cache.jsonl'
    cache = {r['key']: r['same'] for r in load_jsonl(cache_path)}
    sig = prompt_sig(prompts, '归一')
    pkey = lambda a, b: text_hash(sig, *sorted(f'{n}|{info[n]["layer"]}|{info[n]["definition"]}' for n in (a, b)))
    todo = [(a, b) for a, b, _ in pairs if pkey(a, b) not in cache]
    batches = [todo[i:i + GLOBAL_BATCH] for i in range(0, len(todo), GLOBAL_BATCH)]
    log(f'全局去重：{len(names)} 个不同名称，候选对 {len(pairs)} 对，已缓存 {len(pairs) - len(todo)} 对，本次调用 {len(batches)} 次')
    line = lambda n: f'{n} ｜ 第{info[n]["layer"]}层 ｜ {info[n]["definition"] or "—"}'

    def work(batch):
        text = '\n'.join(f'【第{i}对】\n  A：{line(a)}\n  B：{line(b)}' for i, (a, b) in enumerate(batch, 1))
        res = llm.ask(prompts['归一-System'], with_feedback(fill(prompts['归一-User'], pairs=text)), lambda c: parse_same(c, len(batch)))
        rows = [{'key': pkey(a, b), 'a': a, 'b': b, 'same': same, 'reason': reason} for (a, b), (same, reason) in zip(batch, res)]
        for row in rows:
            append_jsonl(cache_path, row)
        return rows

    for rows in run_parallel(batches, work, workers, '全局去重'):
        for row in rows or []:
            cache[row['key']] = row['same']

    parent, reps, blocked = {n: n for n in names}, {n: set(info[n]['reports']) for n in names}, 0

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for a, b, _ in pairs:
        if not cache.get(pkey(a, b)):
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


def graph_record(g, meta, info, table, mapping, notes):
    lay = layer_map(g, table)
    level, effect = annotate(g, lay)
    conn = connected(g)
    mentions = Counter()
    for u, v, d in g.edges(data=True):
        if d['origin'] in TEXT_ORIGINS:
            mentions[u] += d['weight']
            mentions[v] += d['weight']
    nodes = []
    for n in sorted(g, key=lambda n: (level[n], n)):
        t = table.get(n, {})
        nodes.append({'id': n, 'label': mapping.get(n, n), 'layer': lay[n], 'layer_name': LAYER_NAMES[lay[n]], 'level': level[n],
                      'is_root': n == ROOT, 'is_preset': n in fin_items.ITEMS, 'kind': t.get('kind', ''), 'effect': effect[n],
                      'connected': n in conn, 'mentions': mentions[n],
                      'definition': t.get('definition') or fin_items.DEFS.get(n, ''), 'aliases': t.get('aliases', []),
                      'merged': sorted(notes['merged'].get(n, [])), 'unresolved': notes['unresolved'].get(n, []),
                      'attached': notes['attached'].get(n, []), 'hints': notes['hints'].get(n, [])})
    edges = [{'source': u, 'target': v, 'direction': '正向' if d['sign'] > 0 else '负向', 'origin': d['origin'],
              'gap': lay[u] - lay[v], 'weight': d['weight'], 'n_pos': d['n_pos'], 'n_neg': d['n_neg'],
              'evidence': d['evidence'], 'para_ids': d['para_ids'], 'context': d.get('context', []),
              'conflict_path': d.get('conflict_path', [])}
             for u, v, d in sorted(g.edges(data=True), key=lambda e: (level[e[0]], e[0], e[1]))]
    concepts = [n for n in g if lay[n] >= 3]
    by_origin = Counter(e['origin'] for e in edges)
    stats = {'n_table_vars': len(table), 'n_relations': info['n_relations'], 'n_dangling': info['n_dangling'],
             'n_dangling_resolved': info['n_dangling_resolved'], 'n_rejected': info['n_rejected'],
             'n_reverse': len(notes['reverse']), 'n_accounting_text': info['n_accounting_text'],
             'n_nodes': len(nodes), 'n_edges': len(edges), 'n_concepts': len(concepts),
             'n_by_layer': {k: sum(lay[n] == k for n in g) for k in LAYER_NAMES},
             **{f'n_{o}': by_origin[o] for o in ORIGIN_PRIORITY},
             'n_logic_skip': sum(e['origin'] == 'logic' and e['gap'] >= 2 for e in edges),
             'reach_before_completion': info['reach_before'], 'reach': sum(n in conn for n in concepts) / max(len(concepts), 1),
             'n_unconnected': sum(n not in conn for n in concepts), 'n_pruned': len(notes['pruned']),
             'n_isolated': len(notes['isolated']), 'n_merged': sum(map(len, notes['merged'].values())),
             'n_replaced': len(notes['replaced']),
             'n_conflict': sum(bool(e['conflict_path']) for e in edges),
             'n_components': nx.number_weakly_connected_components(g),
             'self_loops': info['self_loops'], 'conflicts': info['conflicts'], 'removed_cycle_edges': len(notes['removed']),
             'depth': max(level.values()) if level else 0}
    return {**meta, 'stats': stats, 'nodes': nodes, 'edges': edges, 'replaced_edges': notes['replaced'],
            'removed_cycle_edges': notes['removed'], 'reverse_relations': notes['reverse'],
            'accounting_relations': notes['accounting'], 'pruned_accounting': notes['pruned'], 'isolated_concepts': notes['isolated'],
            'variable_table': [{'name': n, **t} for n, t in table.items()]}


def rel_row(r):
    return {'cause': r['cause'], 'direction': r['direction'], 'effect': r['effect'], 'evidence': r['evidence'],
            'para_ids': r['para_ids'], 'context': r.get('context', [])}


def main(paras_path, out_dir, limit=None, embed_model=EMBED_MODEL, html_name='report_logic_graphs.html'):
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
    rows = [p for p in table.to_pylist() if (p.get('text') or '').strip()]
    paras = strip_risk(rows)
    n_risk = sum(len(p['text']) for p in rows) - sum(len(p['text']) for p in paras)
    report_ids = sorted({p['report_id'] for p in paras})[:limit]
    by_rid = defaultdict(list)
    for p in sorted(paras, key=lambda p: (p['report_id'], int(p['para_id']))):
        if p['report_id'] in set(report_ids):
            p['para_id'] = int(p['para_id'])
            by_rid[p['report_id']].append(p)
    meta_cols = ('file', 'date', 'broker', 'company', 'stock_code', 'title')
    metas = {rid: {'report_id': rid, **{c: ps[0].get(c) or '' for c in meta_cols}, 'n_paragraphs': len(ps)}
             for rid, ps in by_rid.items()}
    for ps in by_rid.values():
        for i, p in enumerate(ps):
            p['prev_tail'] = tail(ps[i - 1]['text']) if i else ''
    log(f'===== 开始：模型 {env["MODEL"]}，{len(report_ids)} 篇研报、{sum(map(len, by_rid.values()))} 个段落，并行 {workers} =====')
    if n_risk:
        log(f'读入时剔除风险提示 {n_risk} 字（全部研报合计）')

    # 1. 概念分层填表
    tables = build_tables(by_rid, metas, out_dir, llm, prompts, workers)

    # 2. 原文连边，再把新概念补进表、整理第 3 层上位指标（第 1 步的最后一环，要等表补全后再做）
    rels, new_vars, rejected = extract_relations(by_rid, tables, metas, out_dir, llm, prompts, workers)
    renames = supplement_tables(new_vars, tables, metas, out_dir, llm, prompts, workers)
    for r in rels:
        ren = renames.get(r['report_id'], {})
        r['cause'], r['effect'] = ren.get(r['cause'], r['cause']), ren.get(r['effect'], r['effect'])
    rels = [r for r in rels if r['cause'] != r['effect']]
    assign_parents(tables, metas, out_dir, llm, prompts, workers)
    write_parquet([{**{k: r[k] for k in ('report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence')},
                    'cause_layer': layer_of(r['cause'], tables[r['report_id']]),
                    'effect_layer': layer_of(r['effect'], tables[r['report_id']])} for r in rels],
                  out_dir / 'relations.parquet', ['report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence'])

    complete = [{**r, 'origin': 'extracted', 'evidence': [r['evidence']], 'para_ids': [r['para_id']]}
                for r in rels if r['cause'] and r['effect']]
    dangling = [r for r in rels if r['cause'] is None and r['effect']]
    extra, attached, unresolved = resolve_dangling(dangling, complete, by_rid, tables, metas, out_dir, llm, prompts, workers)

    # 第 3 层上位指标向上合并：每棵上位指标树只保留最上层的指标，下位指标的边都改连到它上面，树内部的关系去掉
    tops = {rid: top_parents(tables[rid]) for rid in report_ids}
    merged_rels, n_inner = [], 0
    for r in complete + extra:
        r = merge_up(r, tops[r['report_id']])
        if r['cause'] == r['effect']:
            n_inner += 1
        else:
            merged_rels.append(r)
    log(f'上位指标向上合并：{sum(map(len, tops.values()))} 个下位指标并入最上层的指标，树内部的关系 {n_inner} 条随之去掉')

    # 按两端的层级归类
    groups = {k: defaultdict(list) for k in ('edge', 'accounting', 'reverse')}
    hints = defaultdict(lambda: defaultdict(list))
    for r in merged_rels:
        rid = r['report_id']
        kind = classify(r, tables[rid])
        if kind == 'to_root':  # 原文直接指向公司估值：作为第 3 步的线索
            hints[rid][r['cause']].append({**note_of(r), 'note': '原文直接指向公司估值'})
        else:
            groups[kind][rid].append(r)
    for r in rels:
        if r['cause'] and r['effect'] is None and (layer_of(r['cause'], tables[r['report_id']]) or 0) >= 3:
            hints[r['report_id']][tops[r['report_id']].get(r['cause'], r['cause'])].append(note_of(r))
    log(f'按层归类：进图 {sum(map(len, groups["edge"].values()))} 条，会计科目之间 {sum(map(len, groups["accounting"].values()))} 条，'
        f'反向 {sum(map(len, groups["reverse"].values()))} 条，直接指向公司估值转为线索 '
        f'{sum(x.get("note") == "原文直接指向公司估值" for d in hints.values() for v in d.values() for x in v)} 条')

    # 4（前半）. 建图：预置的会计层 + 原文边。没有任何边的概念是孤立的单个节点，不做补全、直接删掉；
    # 只有原文线索（原文说它影响估值或别的什么，但没写明对象）而没有边的概念也算孤立，线索只供子图里的概念补全时参考
    built = {}
    for rid in report_ids:
        tb = tables[rid]
        g, info, removed = build_graph(preset_items() + groups['edge'][rid])
        isolated = sorted(n for n in tb if tops[rid].get(n, n) not in g)  # 并入了孤立节点的下位指标也一并删掉
        merged = defaultdict(list)
        for child, top in tops[rid].items():
            if top in g:
                merged[top].append(child)
        acc_unmatched = attach_accounting(g, groups['accounting'][rid])
        concepts = [n for n in g if (layer_of(n, tb) or 0) >= 3]
        conn = connected(g)
        info.update(n_relations=sum(r['report_id'] == rid for r in rels), n_rejected=rejected[rid],
                    n_dangling=sum(r['report_id'] == rid and not (r['cause'] and r['effect']) for r in rels),
                    n_dangling_resolved=sum(r['report_id'] == rid for r in extra),
                    n_accounting_text=len(groups['accounting'][rid]) - len(acc_unmatched),
                    reach_before=sum(n in conn for n in concepts) / max(len(concepts), 1))
        built[rid] = {'g': g, 'info': info, 'removed': removed, 'acc_unmatched': acc_unmatched, 'isolated': isolated,
                      'merged': merged, 'n_concepts': len(concepts), 'n_reach': sum(n in conn for n in concepts)}
    n_concepts = sum(b['n_concepts'] for b in built.values())
    log(f'建图：删掉孤立概念 {sum(len(b["isolated"]) for b in built.values())} 个，剩余概念 {n_concepts} 个，'
        f'补全前能连到公司估值的 {sum(b["n_reach"] for b in built.values()) / max(n_concepts, 1):.0%}')

    # 3. 概念补全
    graphs = {rid: built[rid]['g'] for rid in report_ids}
    complete_concepts(graphs, tables, hints, by_rid, metas, out_dir, llm, prompts, workers)
    for rid in report_ids:
        built[rid]['removed'] += break_cycles(graphs[rid])

    # 4（后半）. 剪枝：去掉有方向一致的更长链路的直连（方向相反的保留并标记）；再删掉没有任何下层概念连入的会计科目
    for rid in report_ids:
        built[rid].update(replaced=prune_shortcuts(graphs[rid]), pruned=prune_accounting(graphs[rid]))
    log(f'剪枝完成：删掉有方向一致的更长链路的直连 {sum(len(b["replaced"]) for b in built.values())} 条，'
        f'只有方向相反的链路而保留的直连 {sum(bool(d["conflict_path"]) for g in graphs.values() for *_, d in g.edges(data=True))} 条；'
        f'删掉没有下层概念连入的会计科目平均每篇 {np.mean([len(b["pruned"]) for b in built.values()]):.0f} 个')

    # 5. 全局去重：只给节点加全局名称，不改图的结构
    mapping, global_table = global_dedup(graphs, tables, embed_model, out_dir, llm, prompts, workers)

    # 6. 输出
    records, edge_rows = [], []
    for rid in report_ids:
        b = built[rid]
        up = lambda d: {k: sum((v for x, v in d.items() if tops[rid].get(x, x) == k), []) for k in {tops[rid].get(x, x) for x in d}}
        notes = {'unresolved': up(unresolved[rid]), 'attached': up(attached[rid]), 'hints': hints[rid], 'replaced': b['replaced'],
                 'removed': b['removed'], 'pruned': b['pruned'], 'isolated': b['isolated'], 'merged': b['merged'],
                 'reverse': [rel_row(r) for r in groups['reverse'][rid]], 'accounting': [rel_row(r) for r in b['acc_unmatched']]}
        rec = graph_record(graphs[rid], metas[rid], b['info'], tables[rid], mapping, notes)
        records.append(rec)
        node = {n['id']: n for n in rec['nodes']}
        edge_rows += [{'report_id': rid, 'source': e['source'], 'target': e['target'], 'source_global': node[e['source']]['label'],
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
        json.dumps({'root': ROOT, 'layers': LAYER_NAMES, 'model': env['MODEL'], 'embed_model': embed_model, 'reports': records,
                    'global_names': global_table}, ensure_ascii=False, indent=1), encoding='utf-8')
    write_parquet(edge_rows, out_dir / 'edges.parquet',
                  ['report_id', 'source', 'target', 'source_global', 'target_global', 'source_layer', 'target_layer',
                   'direction', 'origin', 'weight', 'evidence', 'para_ids'])
    write_parquet(node_rows, out_dir / 'nodes.parquet', ['node', 'layer', 'n_reports', 'members'])

    from src import plot_report_logic
    plot_report_logic.main(out_dir / 'report_graphs.json', out_dir / html_name)

    u = llm.usage
    reach = [r['stats']['reach'] for r in records]
    log(f'===== 结束：{len(records)} 篇研报；共 {len(node_rows)} 个全局节点、{len(edge_rows)} 条边；'
        f'概念连到公司估值的比例平均 {np.mean(reach):.0%}；调用 {u["calls"]} 次（不合格重试 {u["retries"]} 次），'
        f'输入 {u["prompt_tokens"]} tokens，输出 {u["completion_tokens"]} tokens =====')
