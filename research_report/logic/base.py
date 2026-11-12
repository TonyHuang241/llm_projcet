"""研报估值逻辑图的公用部分：常量、名称和引文工具、prompt 读取、大模型调用（带缓存、校验重试和并行）、每篇研报的数据容器。"""
import hashlib
import json
import re
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import pyarrow as pa
import pyarrow.parquet as pq

from common.llm_client import ContentFilterError, FatalAPIError, LLMClient, log
from research_report import fin_items
from research_report.config import PROMPT_DIR

PROMPT_FILE = PROMPT_DIR / 'report_logic_prompt.md'

ROOT, MULTIPLE = fin_items.ROOT, fin_items.MULTIPLE
ACCOUNTING_TEXT = fin_items.accounting_list()
LAYER_NAMES = {1: '公司估值', 2: '会计层', 3: '经营指标', 4: '公司经营与行业', 5: '外部环境'}
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
ORIGIN_PRIORITY = ('extracted', 'bridged', 'preset', 'logic', 'part_of')  # 同一条边有多种来源时取靠前的；part_of 为包含关系（只在缩减前的阶段图里）
TEXT_ORIGINS = ('extracted', 'bridged')  # 有原文证据的边
MAX_NODE_LEN = 20  # 概念名的最大字数，超过视为抽取错误
MAX_DEF_LEN = 60
MAX_EVIDENCE_LEN = 120
MAX_RETRIES = 4  # 单次调用输出不合格时的重试次数
DEFAULT_CONCURRENCY = 50

_lock = threading.Lock()


# ==================== 名称和引文 ====================

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


def direction_of(sign):
    return '正向' if sign > 0 else '负向'


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


def to_int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def short(s):
    """写进输出的理由、引文：规范化并截断。"""
    return norm_name(s)[:MAX_EVIDENCE_LEN]


def text_hash(*parts):
    return hashlib.sha1('\x1f'.join(parts).encode('utf-8')).hexdigest()[:16]


def squash(s):
    """比对引文用：全角转半角，去掉空白和引号。"""
    return re.sub(r'[\s“”"\'‘’「」『』]', '', unicodedata.normalize('NFKC', str(s or '')))


def quote_in(quote, text, min_len=6):
    """引文是否逐字出自原文：允许用省略号连接几段原文，每段都要能在原文中找到，合计至少 min_len 个字。"""
    parts = [p for p in re.split(r'…+|\.{3,}', squash(quote)) if p]
    return sum(map(len, parts)) >= min_len and all(p in squash(text) for p in parts)


def cue_missing(name, quote):
    """连到公司估值或估值倍数时，引文中没有相应字样：返回应有的字样说明，否则返回 None。"""
    if name in FIXED_CUES and not FIXED_CUES[name][0].search(str(quote or '')):
        return FIXED_CUES[name][1]
    return None


def para_block(paras):
    """[段落编号] 段落文字，空行分隔。"""
    return '\n\n'.join(f'[{p["para_id"]}] {p["text"]}' for p in paras)


# ==================== 概念表 ====================
# 概念表：{规范名: {'aliases': [别名...], 'definition': 定义, 'layer': 3/4/5, 'kind': 类型, 'parent': 上位概念（同一层）}}

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


# ==================== JSON、缓存文件 ====================

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


def parse_decisions(content, n, value, missing):
    """逐条判断的结果 {"decisions": [{"id": 编号, ...}]}（并列复核、全局去重共用）。value(条目) 返回这一条的判断，
    不合格时返回 None（按缺少处理）；有编号缺少判断时抛出 ValueError，说明由 missing(缺少的编号) 给出。返回按编号排列的判断。"""
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
            v = value(it)
            if v is not None:
                out[i] = v
    lack = [str(i) for i in range(1, n + 1) if i not in out]
    if lack:
        raise ValueError(missing(lack))
    return [out[i] for i in range(1, n + 1)]


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


# ==================== prompt ====================

STEPS = ('建表', '整理', '抽取', '并列', '悬挂', '补全', '归一')  # 每步有 “-System”“-User” 两节
SHARED = '共用-'  # 以此开头的节只是共用的规则，在别的节里用 {{共用-…}} 引用，加载时原样展开


def load_prompts():
    """按二级标题切分 prompt 文件，展开 {{共用-…}} 引用，返回 {标题: 内容}。"""
    raw, cur = {}, None
    for line in PROMPT_FILE.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            cur = line[3:].strip()
            raw[cur] = []
        elif cur:
            raw[cur].append(line)
    raw = {k: '\n'.join(v).strip() for k, v in raw.items()}

    def expand(m):
        name = m.group(1)
        if name not in raw or not name.startswith(SHARED):
            raise SystemExit(f'{PROMPT_FILE} 中引用的 “{{{{{name}}}}}” 不存在，或不是以“{SHARED}”开头的共用节')
        return re.sub(r'\{\{(.+?)\}\}', expand, raw[name])

    prompts = {k: re.sub(r'\{\{(.+?)\}\}', expand, v) for k, v in raw.items() if not k.startswith(SHARED)}
    need = [f'{s}-{part}' for s in STEPS for part in ('System', 'User')]
    lack = [k for k in need if k not in prompts]
    if lack:
        raise SystemExit(f'{PROMPT_FILE} 中缺少这些节：{"、".join(f"“## {k}”" for k in lack)}')
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


# ==================== 大模型调用 ====================

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


class Ctx:
    """整次运行共用的东西：大模型、prompt、输出目录、各篇研报的元信息、并发数。"""

    def __init__(self, env, out_dir, metas):
        self.llm, self.prompts, self.out_dir, self.metas = LLM(env), load_prompts(), out_dir, metas
        self.workers = int(env.get('CONCURRENCY') or DEFAULT_CONCURRENCY)

    def about(self, rid):
        """各步 User prompt 开头的研报公司、标题。"""
        return {'company': self.metas[rid]['company'] or '未知', 'title': self.metas[rid]['title']}

    def parallel(self, items, fn, label):
        return run_parallel(items, fn, self.workers, label)


_MISSING = object()


class Step:
    """一类大模型调用。结果按输入指纹（这一步 prompt 的指纹 + 调用方给出的输入文字）缓存在输出目录的 jsonl 文件中，
    重新运行只补做缺少的部分；fields 为缓存行中存放结果的字段（多个字段时结果是同样长度的元组）。"""

    def __init__(self, ctx, cache_file, step, system=None, user=None, sig=None, fields=('result',)):
        self.ctx, self.path, self.fields = ctx, ctx.out_dir / cache_file, fields
        self.system = ctx.prompts[system or f'{step}-System']
        self.user = ctx.prompts[user or f'{step}-User']
        self.sig = prompt_sig(ctx.prompts, *(sig or (step,)))
        self.cache = {r['key']: r for r in load_jsonl(self.path)}

    def key(self, *parts):
        return text_hash(self.sig, *parts)

    def get(self, key, default=_MISSING):
        row = self.cache.get(key)
        if row is None:
            return default
        return row[self.fields[0]] if len(self.fields) == 1 else tuple(row[f] for f in self.fields)

    def has(self, key):
        return key in self.cache

    def put(self, key, value, **extra):
        row = {'key': key, **extra, **dict(zip(self.fields, (value,) if len(self.fields) == 1 else value))}
        with _lock:
            self.cache[key] = row
        append_jsonl(self.path, row)

    def call(self, parse, lenient=None, **user_kw):
        """不查缓存，直接调用：输出不合格时带上错误说明重试，用完重试次数抛出 RuntimeError。"""
        return self.ctx.llm.ask(self.system, with_feedback(fill(self.user, **user_kw)), parse, lenient)

    def ask(self, key, parse, lenient=None, extra=None, **user_kw):
        """有缓存直接返回；否则调用并写入缓存（extra 为缓存行中另外记下的字段）。失败时抛出 RuntimeError，不写缓存。"""
        hit = self.get(key)
        if hit is not _MISSING:
            return hit
        res = self.call(parse, lenient, **user_kw)
        self.put(key, res, **(extra or {}))
        return self.get(key)


# ==================== 每篇研报 ====================

class Report:
    """一篇研报在各步之间传递的数据，各步按顺序填入：
    第 1 步 table（概念表）；第 2 步 rels（原文抽取记录）、new_vars、rejected、extra/attached/unresolved（悬挂补全）、
    kept/parallel（并列复核）、cut/tops（包含关系缩减）、groups（按层归类）、hints（原文线索）；
    建图之后 g（缩减后的图）、g0（缩减前的图）、info、removed、isolated、merged、acc_unmatched、ref_text；
    第 3 步 added、ref_done；第 4 步 replaced、pruned。"""

    def __init__(self, rid, meta, paras):
        self.rid, self.meta, self.paras = rid, meta, paras
        self.table = {}

    def para(self, pid):
        return next(p for p in self.paras if p['para_id'] == pid)
