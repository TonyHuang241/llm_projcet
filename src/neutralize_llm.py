"""调用大模型对MD&A做逐句中性化改写。

适用于任何兼容 OpenAI Chat Completions 接口的模型（智谱GLM、DeepSeek、通义千问、Kimi、OpenAI等），
换模型只需修改项目根目录的 .env：
    API_KEY=...                必填
    BASE_URL=...               必填，如 https://open.bigmodel.cn/api/paas/v4
    MODEL=...                  必填，如 glm-4.7-flash
    REASONING_EFFORT=low       可选，思考强度（不支持该参数的模型请留空）
    TEMPERATURE=0.2            可选，默认 0.2
    MAX_TOKENS=8192            可选，单次最大输出token数，默认不传
    CONCURRENCY=50             可选，同时处理的文章数，默认 50（每篇文章一个任务，最多同时 50 篇）
    EXTRA_BODY={"thinking": {"type": "disabled"}}   可选，厂商特有参数（JSON），原样并入请求体

流程：每篇文章切句并在全文内连续编号 -> 每 BATCH_SIZE 句一批，按顺序调用模型 ->
校验返回的 JSON（id 与输入一一对应、无空句），不合格就重试，多次失败则把这一批拆成两半再试 ->
整篇完成后写出 <原文件名>.json（id、original、neutral、numbers_missing、fallback）和 .txt（每行一句改写句）。
被服务方内容安全过滤拦截的批次同样重试并拆半；单句仍被拦截时保留原句，fallback 标记为 true。
每批结果实时追加到 _partial/<文件名>.jsonl，中断后重新运行会跳过已完成的文章和批次。
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
SENTENCE_RE = re.compile(r'[^。！？；]+[。！？；]')  # 与 finbert_sentiment.py 的切句规则保持一致
NUM_RE = re.compile(r'\d+(?:[.,]\d+)*')
BATCH_SIZE = 40
MAX_RETRIES = 4  # 同一批内容不合格时的重试次数，用尽后拆成两半
DEFAULT_CONCURRENCY = 50  # .env 未设置 CONCURRENCY 时同时处理的文章数
RETRYABLE_HTTP = {429, 500, 502, 503, 504}
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
FATAL_CODES = ('"1113"',)  # 余额不足/无可用资源包：重试无意义，立即停止全部任务
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
        self.url = env['BASE_URL'].rstrip('/') + '/chat/completions'
        self.headers = {'Authorization': f'Bearer {env["API_KEY"]}', 'Content-Type': 'application/json'}
        self.body = {'model': env['MODEL'], 'temperature': float(env.get('TEMPERATURE') or 0.2)}
        if env.get('REASONING_EFFORT'):
            self.body['reasoning_effort'] = env['REASONING_EFFORT']
        if env.get('MAX_TOKENS'):
            self.body['max_tokens'] = int(env['MAX_TOKENS'])
        if env.get('EXTRA_BODY'):
            self.body.update(json.loads(env['EXTRA_BODY']))
        self.http_retries = 0  # 因限流、服务端错误或网络问题而重发的次数（多线程下为近似值）

    def chat(self, system, user, timeout=300):
        """返回 (回复内容, finish_reason, usage)。限流、服务端错误和网络错误会退避后无限重试；
        余额不足等致命错误则置位 STOP 并抛出 FatalAPIError。"""
        data = json.dumps({**self.body, 'messages': [
            {'role': 'system', 'content': system}, {'role': 'user', 'content': user}]}).encode('utf-8')
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
                if any(c in body for c in FATAL_CODES):
                    STOP.set()
                    raise FatalAPIError(f'HTTP {e.code}: {body}') from None
                if e.code not in RETRYABLE_HTTP:
                    if any(c in body for c in CONTENT_FILTER_CODES):
                        raise ContentFilterError('内容安全过滤') from None
                    raise RuntimeError(f'HTTP {e.code}: {body}') from None
                self.http_retries += 1
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
                self.http_retries += 1
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
                got = json.loads(line)
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
        (out_dir / f'{stem}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding='utf-8')
        (out_dir / f'{stem}.txt').write_text(''.join(r['neutral'] + '\n' for r in rows), encoding='utf-8')
        partial.unlink()
        return stem, len(rows), time.time() - t0


def main(mda_dir, out_dir, year, limit=50):
    global LOG_PATH
    env = load_env()
    out_dir = Path(out_dir)
    (out_dir / '_partial').mkdir(parents=True, exist_ok=True)
    LOG_PATH = out_dir / '_progress.log'
    files = sorted(Path(mda_dir).glob('*.txt'))[:limit]
    todo = [f for f in files if not (out_dir / f'{f.stem}.json').exists()]
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
