"""调用大模型对MD&A做逐句中性化改写。

适用于任何兼容 OpenAI Chat Completions 接口的模型（智谱GLM、DeepSeek、通义千问、Kimi、OpenAI等），
换模型只需修改项目根目录的 .env：
    API_KEY=...                必填
    BASE_URL=...               必填，如 https://open.bigmodel.cn/api/paas/v4（写到 /chat/completions 也可以）
    MODEL=...                  必填，如 glm-4.7-flash
    REASONING_EFFORT=low       可选，思考强度（不支持该参数的模型请留空）
    TEMPERATURE=0.2            可选，默认 0.2
    MAX_TOKENS=8192            可选，单次最大输出token数，默认不传
    CONCURRENCY=50             可选，同时处理的文章数，默认 50（每篇文章一个任务，批次按顺序请求，
                               所以这也是同时在途的最大请求数）
    EXTRA_BODY={"thinking": {"type": "disabled"}}   可选，厂商特有参数（JSON），原样并入请求体
    MERGE_SYSTEM=1             可选，把 system prompt 拼进 user 消息发送（用于会丢弃 system 消息的中转站）

流程：每篇文章切句并在全文内连续编号 -> 每 BATCH_SIZE 句一批，按顺序调用模型 ->
校验返回的 JSON（id 与输入一一对应、无空句），不合格就重试，多次失败则把这一批拆成两半再试 ->
整篇完成后写出 <原文件名>.json（id、original、neutral、numbers_missing、fallback）和 .txt（每行一句改写句）。
被服务方内容安全过滤拦截的批次同样重试并拆半；单句仍被拦截时保留原句，fallback 标记为 true。
每批结果实时追加到 _partial/<文件名>.jsonl，中断后重新运行会跳过已完成的文章和批次
（中断时写了一半的行会被丢弃并重新改写；最终 .json 原子写入，存在即表示该篇已完成）。
可传入 coverage_path（分析师预测 parquet，需含 stock_code、create_date、report_count 列）只处理有分析师覆盖的股票，
coverage_year 限定为该年内有一年后(fy1)预测（Y 年年报传 Y-1）；可传入 st_path、st_year 剔除该年被 ST 过的股票（Y 年年报传 Y-1）。
进度（时间、完成篇数、失败、预计剩余时间）同时追加写入 <输出目录>/_progress.log。

用法：在 main.py 中设置 TASK = 'neutralize_llm'
"""
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = ROOT / 'prompts' / 'mda_neutralize_prompt.md'
SENTENCE_RE = re.compile(r'[^。！？；]+[。！？；]')
NUM_RE = re.compile(r'\d+(?:[.,]\d+)*')
BATCH_SIZE = 100  # 每次调用改写的句数；实测到 160 句未见截断或语调变弱，按次计费时批次越大越省
MAX_RETRIES = 4  # 同一批内容不合格时的重试次数，用尽后拆成两半
DEFAULT_CONCURRENCY = 50  # .env 未设置 CONCURRENCY 时同时处理的文章数
BSE_PREFIXES = ('43', '83', '87', '88', '92')  # 北交所（含原新三板精选层）股票代码前缀
RETRYABLE_HTTP = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}  # 52x：Cloudflare 网关与源站之间的临时错误
LOG_PATH = None  # 进度日志文件，main() 中设为 <输出目录>/_progress.log
_log_lock = threading.Lock()


def log(msg):
    """打印并追加写入进度日志（带时间戳），多线程安全。"""
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    with _log_lock:
        print(line, flush=True)
        if LOG_PATH:
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


# 以下错误码按智谱设置，换模型时按需补充（匹配错误响应体中的文本）
CONTENT_FILTER_CODES = ('"1301"',)  # 内容安全拦截：重试、拆半，单句仍被拦截则保留原句
# 余额不足/无可用资源包（智谱 1113；new-api 类中转站 insufficient_user_quota；laneai insufficient_balance）：
# 重试无意义，立即停止全部任务；已完成的批次保留在 _partial 中，充值后重新运行即可接上
FATAL_CODES = ('"1113"', 'insufficient_user_quota', 'insufficient_balance')
STOP = threading.Event()  # 遇到致命错误时置位，所有线程尽快退出


class ContentFilterError(RuntimeError):
    """模型服务方的内容安全过滤拦截了输入或输出。"""


class FatalAPIError(RuntimeError):
    """余额不足等无法通过重试解决的错误，需要人工处理后重新运行。"""


def load_env(path=ROOT / '.env'):
    env = {}
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1)
            env[k.strip()] = v.strip().strip('"\'')
    for k in ('API_KEY', 'BASE_URL', 'MODEL'):
        if not env.get(k):
            raise SystemExit(f'.env 缺少 {k}')
    return env


def load_prompts():
    """从 prompt 文件中取出 System Prompt 和 User Prompt 模板两部分。"""
    text = PROMPT_FILE.read_text(encoding='utf-8')
    system = text.split('## System Prompt', 1)[1].split('\n---', 1)[0].strip()
    user = text.split('## User Prompt 模板', 1)[1].split('\n（`{sentences_json}`', 1)[0].strip()  # 去掉末尾的格式说明
    if '{sentences_json}' not in user:
        raise SystemExit(f'{PROMPT_FILE} 的 User Prompt 模板中缺少 {{sentences_json}}')
    return system, user


def split_sentences(text):
    """逐行切句；不以句号等结尾的小标题不参与改写。"""
    return [s.strip() for line in text.splitlines() for s in SENTENCE_RE.findall(line) if len(s) >= 5]


class LLMClient:
    def __init__(self, env):
        base = env['BASE_URL'].rstrip('/')  # 兼容 BASE_URL 写到 /v1 或直接写完整的 /chat/completions
        self.url = base if base.endswith('/chat/completions') else base + '/chat/completions'
        self.headers = {'Authorization': f'Bearer {env["API_KEY"]}', 'Content-Type': 'application/json',
                        'User-Agent': 'llm-project-neutralizer/1.0'}  # 部分服务的 Cloudflare 会拒绝 urllib 默认 UA（403/1010）
        self.body = {'model': env['MODEL'], 'temperature': float(env.get('TEMPERATURE') or 0.2)}
        if env.get('REASONING_EFFORT'):
            self.body['reasoning_effort'] = env['REASONING_EFFORT']
        if env.get('MAX_TOKENS'):
            self.body['max_tokens'] = int(env['MAX_TOKENS'])
        if env.get('EXTRA_BODY'):
            self.body.update(json.loads(env['EXTRA_BODY']))
        self.http_retries = 0  # 因限流、服务端错误或网络问题而重发的次数（多线程下为近似值）
        self.timeouts = 0  # 其中网络错误/超时的次数
        self.merge_system = env.get('MERGE_SYSTEM', '').lower() in ('1', 'true', 'yes')

    def chat(self, system, user, timeout=300):
        """返回 (回复内容, finish_reason, usage)。限流、服务端错误和网络错误会退避后无限重试；
        余额不足等致命错误则置位 STOP 并抛出 FatalAPIError。"""
        if self.merge_system:  # 部分中转站会丢弃 system 消息：把它拼到 user 消息前面
            messages = [{'role': 'user', 'content': f'{system}\n\n---\n\n{user}'}]
        else:
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        data = json.dumps({**self.body, 'messages': messages}).encode('utf-8')
        wait = 5
        while True:
            if STOP.is_set():
                raise FatalAPIError('任务已因致命错误停止')
            try:
                req = urllib.request.Request(self.url, data=data, headers=self.headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    r = json.load(resp)
                choice = r['choices'][0]
                return choice['message'].get('content') or '', choice.get('finish_reason'), r.get('usage') or {}
            except urllib.error.HTTPError as e:
                body = e.read().decode('utf-8', 'replace')[:300]
                if e.code in (401, 402, 403) or any(c in body for c in FATAL_CODES):
                    STOP.set()
                    raise FatalAPIError(f'HTTP {e.code}: {body}') from None
                if e.code not in RETRYABLE_HTTP:
                    if any(c in body for c in CONTENT_FILTER_CODES):
                        raise ContentFilterError('内容安全过滤') from None
                    raise RuntimeError(f'HTTP {e.code}: {body}') from None
                self.http_retries += 1
                if self.http_retries <= 3:  # 只记录前几次，便于判断触发的是哪种限制
                    log(f'HTTP {e.code}，退避后重试：{body[:200]}')
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                self.http_retries += 1
                self.timeouts += 1
                if self.timeouts <= 3 or self.timeouts % 50 == 0:  # 超时/断网会静默重发，这里留下记录
                    log(f'网络错误或超时（累计 {self.timeouts} 次），退避后重试：{type(e).__name__} {e}')
            STOP.wait(wait + random.uniform(0, wait))  # 退避等待（加随机抖动）；STOP 置位时立即醒来
            wait = min(wait * 2, 120)


def parse_batch(content, batch):
    """解析并校验模型输出，合格返回 {id: 改写句}，否则抛出 ValueError。"""
    text = re.sub(r'^\s*```(?:json)?|```\s*$', '', content.strip()).strip()
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end < 0:
        raise ValueError('未找到JSON数组')
    items = json.loads(text[start:end + 1])
    got = {int(it['id']): str(it['text']).strip() for it in items}
    want = [s['id'] for s in batch]
    if len(items) != len(want) or sorted(got) != want:
        raise ValueError(f'句子不对应：输入{len(want)}句，输出{len(items)}句')
    if any(not got[i] for i in want):
        raise ValueError('存在空句')
    return got


class Neutralizer:
    def __init__(self, env, year):
        self.client = LLMClient(env)
        self.system, self.user_tpl = load_prompts()
        self.year = year
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0, 'retries': 0}
        self.fallbacks = set()  # 被内容过滤拦截、保留原句的 (股票代码, 句子id)

    def rewrite(self, batch, tag):
        """改写一批句子；多次不合格时拆半递归，直到单句。"""
        user = (self.user_tpl.replace('{year}', self.year).replace('{n}', str(len(batch)))
                .replace('{sentences_json}', json.dumps(batch, ensure_ascii=False)))
        err = None
        for _ in range(MAX_RETRIES):
            self.usage['calls'] += 1
            try:
                content, finish, usage = self.client.chat(self.system, user)
                self.usage['prompt_tokens'] += usage.get('prompt_tokens', 0)
                self.usage['completion_tokens'] += usage.get('completion_tokens', 0)
                if finish == 'length':
                    raise ValueError('输出被截断')
                return parse_batch(content, batch)
            except (ContentFilterError, ValueError, KeyError, TypeError) as e:
                err = e
                self.usage['retries'] += 1
        if len(batch) == 1:
            if isinstance(err, ContentFilterError):
                # 单句仍被拦截：保留原句，并在结果中标记 fallback，便于之后统计或人工处理
                log(f'{tag} id={batch[0]["id"]} 被内容安全过滤拦截，保留原句：{batch[0]["text"][:40]}')
                self.fallbacks.add((tag, batch[0]['id']))
                return {batch[0]['id']: batch[0]['text']}
            raise RuntimeError(f'{tag} id={batch[0]["id"]} 改写失败：{err}')
        log(f'{tag} 第{batch[0]["id"]}-{batch[-1]["id"]}句连续{MAX_RETRIES}次不合格（{err}），拆成两半重试')
        mid = len(batch) // 2
        return {**self.rewrite(batch[:mid], tag), **self.rewrite(batch[mid:], tag)}

    def process(self, src, out_dir):
        """处理一篇文章，返回 (文件名, 句数, 用时秒)。"""
        if STOP.is_set():
            raise FatalAPIError('任务已因致命错误停止')
        stem, t0 = src.stem, time.time()
        sents = [{'id': i, 'text': s} for i, s in enumerate(split_sentences(src.read_text(encoding='utf-8')), 1)]
        partial = out_dir / '_partial' / f'{stem}.jsonl'
        done = {}  # 已完成批次的结果，用于断点续跑
        code = stem.split('_')[0]
        if partial.exists():
            for line in partial.read_text(encoding='utf-8').splitlines():
                try:
                    got = json.loads(line)
                except json.JSONDecodeError:  # 中断时写了一半的行：丢弃，这批会重新改写
                    continue
                self.fallbacks.update((code, i) for i in got.pop('_fallback', []))
                done.update({int(k): v for k, v in got.items()})
        for k in range(0, len(sents), BATCH_SIZE):
            batch = [s for s in sents[k:k + BATCH_SIZE] if s['id'] not in done]
            if batch:
                got = self.rewrite(batch, code)
                fb = [s['id'] for s in batch if (code, s['id']) in self.fallbacks]
                with partial.open('a', encoding='utf-8') as f:
                    f.write(json.dumps({**got, '_fallback': fb} if fb else got, ensure_ascii=False) + '\n')
                done.update(got)
        rows = [{'id': s['id'], 'original': s['text'], 'neutral': done[s['id']],
                 'numbers_missing': [n for n in NUM_RE.findall(s['text']) if n not in done[s['id']]],
                 'fallback': (code, s['id']) in self.fallbacks}
                for s in sents]
        # 先写 .txt，再原子写入 .json（.json 是“已完成”标记，避免中断后留下半个文件却被当成已完成）
        (out_dir / f'{stem}.txt').write_text(''.join(r['neutral'] + '\n' for r in rows), encoding='utf-8')
        tmp = out_dir / f'{stem}.json.tmp'
        tmp.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(out_dir / f'{stem}.json')
        partial.unlink()
        return stem, len(rows), time.time() - t0


def load_covered_codes(path, year=None):
    """读取分析师预测数据中有覆盖的股票代码（6位字符串）。
    year 为 None：数据中出现过（report_count > 0）即算覆盖。
    year 给出：只算该自然年内（按 create_date）有“一年后”预测（任一 *_fy1_* 列非空）的股票。
    Y 年年报传 year=Y-1，即看 Y-1 年内是否有对下一财年（Y 年）的预测。"""
    import pyarrow.compute as pc  # 只在需要过滤时导入
    import pyarrow.parquet as pq
    names = pq.read_schema(path).names
    fy1 = [c for c in names if '_fy1_' in c]
    t = pq.read_table(path, columns=['stock_code', 'create_date', 'report_count'] + (fy1 if year else []))
    mask = pc.greater(t['report_count'], 0)
    if year is not None:
        if not fy1:
            raise SystemExit(f'{path} 中没有 *_fy1_* 预测列')
        has_fy1 = pc.invert(pc.is_null(t[fy1[0]]))
        for c in fy1[1:]:
            has_fy1 = pc.or_(has_fy1, pc.invert(pc.is_null(t[c])))
        mask = pc.and_(pc.and_(mask, has_fy1), pc.equal(pc.year(t['create_date']), int(year)))
    codes = t.filter(mask)['stock_code'].unique().to_pylist()
    return {str(c).strip().zfill(6) for c in codes if c is not None}


def load_st_codes(path, year):
    """读取 ST 状态数据（每日 date/code/is_st），返回 year 自然年内任一交易日被 ST 的股票代码（6位字符串）。
    Y 年年报传 year=Y-1，即剔除 Y-1 年内被 ST 过的股票。"""
    import pyarrow.compute as pc  # 只在需要过滤时导入
    import pyarrow.parquet as pq
    y = str(year)
    t = pq.read_table(path, columns=['date', 'code', 'is_st'],
                      filters=[('date', '>=', f'{y}0101'), ('date', '<=', f'{y}1231'), ('is_st', '==', True)])
    if not t.num_rows and not pq.read_table(path, columns=['date'],
                                            filters=[('date', '>=', f'{y}0101'), ('date', '<=', f'{y}1231')]).num_rows:
        raise SystemExit(f'{path} 中没有 {y} 年的数据，无法做 ST 过滤')
    return {str(c).strip().zfill(6) for c in pc.unique(t['code']).to_pylist() if c is not None}


def main(mda_dir, out_dir, year, limit=50, coverage_path=None, coverage_year=None, st_path=None, st_year=None,
         exclude_bse=False):
    """coverage_path：分析师预测 parquet 路径；给出时只处理有分析师覆盖的股票，None 表示不过滤。
    coverage_year：只保留该年内有一年后(fy1)预测的股票（Y 年年报传 Y-1），None 表示任意年份出现过即可。
    st_path：ST 状态 parquet 路径；给出时剔除 st_year 年内被 ST 过的股票（Y 年年报传 Y-1），None 表示不过滤。
    exclude_bse：True 时剔除北交所股票。"""
    global LOG_PATH
    env = load_env()
    out_dir = Path(out_dir)
    (out_dir / '_partial').mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / '_progress.log'
    files = sorted(Path(mda_dir).glob('*.txt'))
    if coverage_path:
        covered = load_covered_codes(coverage_path, coverage_year)
        n_all = len(files)
        files = [f for f in files if f.stem.split('_')[0] in covered]
        scope = f'{coverage_year} 年内有一年后(fy1)预测' if coverage_year else '任意年份出现过'
        log(f'分析师覆盖过滤（{scope}）：共 {len(covered)} 只股票，MD&A {n_all} 篇中有覆盖 {len(files)} 篇，'
            f'跳过 {n_all - len(files)} 篇')
        if not files:
            raise SystemExit(f'按“{scope}”筛选后没有任何文章，请检查分析师数据是否覆盖 {coverage_year} 年')
    else:
        log(f'不按分析师覆盖筛选：MD&A {len(files)} 篇全部保留')
    if st_path:
        st = load_st_codes(st_path, st_year)
        n_all = len(files)
        files = [f for f in files if f.stem.split('_')[0] not in st]
        log(f'ST 过滤（{st_year} 年内被 ST 过）：共 {len(st)} 只股票，剔除 {n_all - len(files)} 篇，剩 {len(files)} 篇')
    if exclude_bse:
        n_all = len(files)
        files = [f for f in files if not f.stem.startswith(BSE_PREFIXES)]
        log(f'北交所过滤：剔除 {n_all - len(files)} 篇，剩 {len(files)} 篇')
    files = files[:limit]
    todo =[f for f in files if not (out_dir / f'{f.stem}.json').exists()]
    workers = min(int(env.get('CONCURRENCY') or DEFAULT_CONCURRENCY), max(len(todo), 1))
    log(f'===== 开始：模型 {env["MODEL"]}，目标 {len(files)} 篇，已完成 {len(files) - len(todo)} 篇，'
        f'本次处理 {len(todo)} 篇，并行 {workers} 个任务 =====')
    if not todo:
        return

    nz, t0, failed, fatal, n_done = Neutralizer(env, year), time.time(), [], None, 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(nz.process, f, out_dir): f for f in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            eta = (time.time() - t0) / k * (len(todo) - k) / 60
            try:
                stem, n, sec = fut.result()
                n_done += 1
                log(f'[{k}/{len(todo)}] {stem[:20]} 完成 {n} 句，用时 {sec:.0f} 秒；'
                    f'累计限流/网络重发 {nz.client.http_retries} 次，预计还需 {eta:.0f} 分钟')
            except FatalAPIError as e:  # 余额不足等：只记录第一次，其余文章随即自动退出
                if fatal is None:
                    fatal = e
                    log(f'!!!!! 致命错误，停止全部任务（已完成的批次已保存，处理后重新运行即可继续）：{e}')
            except Exception as e:  # 单篇失败不影响其他文章，已完成的批次保留在 _partial 中
                failed.append(futures[fut].name)
                log(f'[{k}/{len(todo)}] {futures[fut].name[:20]} 失败：{e}')

    if fatal:
        log(f'本次完成 {n_done} 篇后因致命错误停止，未完成 {len(todo) - n_done - len(failed)} 篇；'
            f'原因：{fatal}')
    u = nz.usage
    log(f'===== 结束：用时 {(time.time() - t0) / 60:.1f} 分钟；调用 {u["calls"]} 次（其中不合格重试 {u["retries"]} 次，'
        f'另有限流/网络重发 {nz.client.http_retries} 次），'
        f'输入 {u["prompt_tokens"]} tokens，输出 {u["completion_tokens"]} tokens =====')
    if nz.fallbacks:
        log(f'被内容安全过滤拦截、保留原句的句子 {len(nz.fallbacks)} 句：{sorted(nz.fallbacks)}')
    if failed:
        log(f'失败 {len(failed)} 篇（重新运行即可从断点继续）：{failed}')
