"""大模型调用层：.env 读取、带时间戳的进度日志、兼容 OpenAI Chat Completions 接口的客户端。

适用于任何兼容 OpenAI Chat Completions 接口的模型（智谱GLM、DeepSeek、通义千问、Kimi、OpenAI等），
换模型只需修改项目根目录的 .env：
    API_KEY=...                必填
    BASE_URL=...               必填，如 https://open.bigmodel.cn/api/paas/v4（写到 /chat/completions 也可以）
    MODEL=...                  必填，如 glm-4.7-flash
    REASONING_EFFORT=low       可选，思考强度（不支持该参数的模型请留空）
    TEMPERATURE=0.2            可选，默认 0.2
    MAX_TOKENS=8192            可选，单次最大输出token数，默认不传
    CONCURRENCY=50             可选，同时在途的任务数，默认由调用方决定
    EXTRA_BODY={"thinking": {"type": "disabled"}}   可选，厂商特有参数（JSON），原样并入请求体
    HTTP_MAX_TRIES=6           可选，一次调用遇到限流、服务端错误或网络错误时最多发几次（每次之间退避等待），默认 6 次。
                               有的中转站对它拒绝的内容也返回 503“模型繁忙”，无限重试会让整次运行卡在这一个请求上
    STREAM=0                   可选，默认用流式接收（边生成边返回）；设为 0 改回一次性返回。有的中转站对一次性返回的
                               请求有约 30 秒的时限，生成慢的请求会被掐断（HTTP 502/503），流式接收不受这个限制
    MERGE_SYSTEM=1             可选，把 system prompt 拼进 user 消息发送（用于会丢弃 system 消息的中转站）

使用方：src/report_logic.py、src/report_compare.py。
"""
import http.client
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # 仓库根目录，.env 放在这里
RETRYABLE_HTTP = {429, 500, 502, 503, 504, 520, 521, 522, 523, 524}  # 52x：Cloudflare 网关与源站之间的临时错误
# 以下错误码按智谱设置，换模型时按需补充（匹配错误响应体中的文本）
CONTENT_FILTER_CODES = ('"1301"',)  # 内容安全拦截：由调用方决定重试、拆半或保留原句
# 余额不足/无可用资源包（智谱 1113；new-api 类中转站 insufficient_user_quota；laneai insufficient_balance）：
# 重试无意义，立即停止全部任务；调用方已完成的部分保留在缓存中，充值后重新运行即可接上
FATAL_CODES = ('"1113"', 'insufficient_user_quota', 'insufficient_balance')
STOP = threading.Event()  # 遇到致命错误时置位，所有线程尽快退出

LOG_PATH = None  # 进度日志文件，调用方用 set_log_path() 设为 <输出目录>/cache/progress.log
_log_lock = threading.Lock()


def set_log_path(path):
    """设置进度日志文件（None 表示只打印不写文件）。"""
    global LOG_PATH
    LOG_PATH = Path(path) if path else None


_quiet = threading.local()


def set_quiet(on):
    """在当前线程关掉（或恢复）过程日志：逐篇并行处理时，各篇各步的过程行不打，只留 “!!!” 开头的警告。"""
    _quiet.on = on


def log(msg):
    """打印并追加写入进度日志（带时间戳），多线程安全。"""
    if getattr(_quiet, 'on', False) and not msg.startswith('!!!'):
        return
    line = f'{time.strftime("%Y-%m-%d %H:%M:%S")} {msg}'
    with _log_lock:
        print(line, flush=True)
        if LOG_PATH:
            with open(LOG_PATH, 'a', encoding='utf-8') as f:
                f.write(line + '\n')


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


class StreamError(RuntimeError):
    """流式返回的数据里带着错误，或者流没有正常结束。"""


class StreamTimeout(StreamError):
    """一次请求的流式返回总时长超过 STREAM_MAX：不重发，这次调用直接算失败（调用方跳过）。"""


STREAM_IDLE = 300  # 流式返回超过这么多秒没有新生成的内容（回复或思考过程）就断开重发
STREAM_MAX = 600  # 一次请求的流式返回总共超过这么多秒就断开、这次调用算失败不重发（正常调用最慢四百多秒）


def read_stream(resp):
    """读完一个流式响应（SSE，每行 “data: {JSON}”，以 “data: [DONE]” 结束），返回 (回复内容, finish_reason, usage)。
    思考过程（reasoning_content）不要；流里带 error，或者没有等到结束标记就断了，抛出 StreamError（调用方重试）。
    连接没断、却超过 STREAM_IDLE 秒没有新生成的内容时也抛出 StreamError：中转站卡住时可能一直发保活行，
    socket 的读取超时不会触发，调用会永远挂着（2026-11-20 名称统一时 72 个调用这样卡住）。
    总共超过 STREAM_MAX 秒抛出 StreamTimeout：有的上游卡住时一直极慢地吐内容，空闲检查不触发，一次调用拖到一个多小时
    （2026-11-20 补抽深度报告时最长 4610 秒）；重发也常常同样慢，所以不重发，这次调用直接算失败（概念抽取里这篇不存档、
    下次运行重试，名称统一里这组不合并）。"""
    parts, finish, usage, ended = [], None, {}, False
    start = last = time.time()
    for raw in resp:
        if time.time() - last > STREAM_IDLE:
            raise StreamError(f'流式返回 {STREAM_IDLE} 秒没有新内容')
        if time.time() - start > STREAM_MAX:
            raise StreamTimeout(f'流式返回超过 {STREAM_MAX} 秒还没结束')
        line = raw.decode('utf-8', 'replace').strip()
        if not line.startswith('data:'):
            continue
        data = line[5:].strip()
        if data == '[DONE]':
            ended = True
            break
        chunk = json.loads(data)
        if chunk.get('error'):
            raise StreamError(json.dumps(chunk['error'], ensure_ascii=False)[:300])
        usage = chunk.get('usage') or usage
        for choice in chunk.get('choices') or []:
            delta = choice.get('delta') or {}
            piece = delta.get('content')
            if piece:
                parts.append(piece)
            if piece or delta.get('reasoning_content'):
                last = time.time()
            finish = choice.get('finish_reason') or finish
    if not ended and finish is None:
        raise StreamError('流没有正常结束')
    return ''.join(parts), finish, usage


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
        self.max_tries = int(env.get('HTTP_MAX_TRIES') or 6)
        self.stream = env.get('STREAM', '1').lower() not in ('0', 'false', 'no')
        if self.stream:
            self.body.update({'stream': True, 'stream_options': {'include_usage': True}})
        self.http_retries = 0  # 因限流、服务端错误或网络问题而重发的次数（多线程下为近似值）
        self.timeouts = 0  # 其中网络错误/超时的次数
        self.merge_system = env.get('MERGE_SYSTEM', '').lower() in ('1', 'true', 'yes')

    def chat(self, system, user, timeout=300):
        """返回 (回复内容, finish_reason, usage)。限流、服务端错误和网络错误会退避后重试，最多发 max_tries 次，
        仍不成功抛出 RuntimeError（调用方按这次调用失败处理）；余额不足等致命错误则置位 STOP 并抛出 FatalAPIError。"""
        if self.merge_system:  # 部分中转站会丢弃 system 消息：把它拼到 user 消息前面
            messages = [{'role': 'user', 'content': f'{system}\n\n---\n\n{user}'}]
        else:
            messages = [{'role': 'system', 'content': system}, {'role': 'user', 'content': user}]
        def payload(tries):
            """重试时在第一条消息最前面加一行请求序号：中转站似乎按开头的内容把请求固定分给同一个上游，那个上游卡住时，
            原样重发总是落到同一处、一直没有回复（2026-11-20 实测：开头加一行就能正常返回，只在末尾加不行）。"""
            msgs = messages if tries == 1 else [{**messages[0], 'content': f'（第 {tries} 次请求）\n{messages[0]["content"]}'},
                                                *messages[1:]]
            return json.dumps({**self.body, 'messages': msgs}).encode('utf-8')

        wait, tries, last = 5, 0, ''
        while True:
            tries += 1
            if STOP.is_set():
                raise FatalAPIError('任务已因致命错误停止')
            try:
                req = urllib.request.Request(self.url, data=payload(tries), headers=self.headers)
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    if self.stream:
                        return read_stream(resp)
                    r = json.load(resp)
                choice = r['choices'][0]
                return choice['message'].get('content') or '', choice.get('finish_reason'), r.get('usage') or {}
            except StreamTimeout as e:  # 总时长超限：不重发，这次调用算失败
                log(f'{e}，跳过这次调用（不重发）')
                raise RuntimeError(f'{e}，跳过这次调用') from None
            except StreamError as e:  # 流里返回的错误：致命的停下，其余按服务端错误退避重试
                if any(c in str(e) for c in FATAL_CODES):
                    STOP.set()
                    raise FatalAPIError(str(e)) from None
                if any(c in str(e) for c in CONTENT_FILTER_CODES):
                    raise ContentFilterError('内容安全过滤') from None
                self.http_retries += 1
                last = f'流式返回出错 {e}'
                if self.http_retries <= 3:
                    log(f'流式返回出错，退避后重试：{str(e)[:200]}')
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
                last = f'HTTP {e.code} {body}'
                if self.http_retries <= 3:  # 只记录前几次，便于判断触发的是哪种限制
                    log(f'HTTP {e.code}，退避后重试：{body[:200]}')
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError, http.client.HTTPException) as e:
                self.http_retries += 1
                self.timeouts += 1
                last = f'{type(e).__name__} {e}'
                if self.timeouts <= 3 or self.timeouts % 50 == 0:  # 超时/断网会静默重发，这里留下记录
                    log(f'网络错误或超时（累计 {self.timeouts} 次），退避后重试：{type(e).__name__} {e}')
            if tries >= self.max_tries:  # 最后一次失败后不再多等一轮
                raise RuntimeError(f'接口连续 {tries} 次没有成功（最后一次：{last[:120]}），放弃这次调用')
            STOP.wait(wait + random.uniform(0, wait))  # 退避等待（加随机抖动）；STOP 置位时立即醒来
            wait = min(wait * 2, 120)


# ==================== 在途调用数 ====================

class InflightMeter:
    """统计同时在途的大模型调用数，用来确认并发一直是满的：wrap(chat) 给调用函数包一层计数（调用期间加一，含失败重试的
    退避等待）；后台每秒采样一次，每 interval 秒在进度日志里记一行（这段时间的平均、最少、最多）。用完调用 close()。"""

    def __init__(self, limit, label='在途调用', interval=60):
        self.limit, self.label, self.interval = limit, label, interval
        self.n, self.lock, self.done = 0, threading.Lock(), threading.Event()
        threading.Thread(target=self.watch, daemon=True).start()

    def wrap(self, chat):
        def counted(*a, **kw):
            with self.lock:
                self.n += 1
            try:
                return chat(*a, **kw)
            finally:
                with self.lock:
                    self.n -= 1
        return counted

    def watch(self):
        buf, last = [], time.time()
        while not self.done.wait(1):
            buf.append(self.n)
            if time.time() - last >= self.interval:
                if max(buf):
                    log(f'{self.label}：最近 {len(buf)} 秒平均 {sum(buf) / len(buf):.1f} 个、最少 {min(buf)}、最多 {max(buf)}'
                        f'（上限 {self.limit}）')
                buf, last = [], time.time()

    def close(self):
        self.done.set()


# ==================== 额度上限 ====================

QUOTA_PER_USD = 500000  # new-api 类中转站的额度单位：500000 = 1 美元
BUDGET_MARGIN = 0.3  # 离上限还差这么多美元时就停：停下时在途的调用还会各自扣完费


def used_usd(env, timeout=30):
    """这把密钥在中转站上累计已用的额度（美元）。查的是 new-api 类中转站的 /api/usage/token；不支持时抛出异常。"""
    base = env['BASE_URL'].rstrip('/')
    base = base[:-len('/chat/completions')] if base.endswith('/chat/completions') else base
    root = base[:-3] if base.endswith('/v1') else base
    req = urllib.request.Request(root + '/api/usage/token', headers={
        'Authorization': f'Bearer {env["API_KEY"]}', 'User-Agent': 'llm-project-neutralizer/1.0'})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)['data']['total_used'] / QUOTA_PER_USD


class Budget:
    """本次运行的额度上限（美元）：后台线程每隔 interval 秒查一次中转站的已用额度，本次运行花掉的达到上限（留出
    BUDGET_MARGIN）就置位 STOP，各线程尽快停下。按中转站实际扣的费计算，同一把密钥同时在别处使用的也会算进来。
    连续 5 次查不到额度时同样停下：没法确认花了多少，不能接着花。"""

    def __init__(self, env, usd, interval=15):
        self.env, self.usd, self.interval = env, float(usd), interval
        self.start = used_usd(env)  # 查不到直接抛出：没法监控就不开始
        self.spent, self.hit, self.done = 0.0, False, threading.Event()
        threading.Thread(target=self.watch, daemon=True).start()

    def watch(self):
        fails, last_log = 0, time.time()
        while not self.done.wait(self.interval) and not STOP.is_set():
            try:
                self.spent, fails = used_usd(self.env) - self.start, 0
            except Exception as e:
                fails += 1
                if fails >= 5:
                    self.hit = True
                    log(f'!!! 连续 {fails} 次查不到已用额度（{type(e).__name__} {e}），无法确认花费，停止运行')
                    STOP.set()
                continue
            if self.spent >= self.usd - BUDGET_MARGIN:
                self.hit = True
                log(f'!!! 额度上限已到：本次运行已花 {self.spent:.2f} 美元（上限 {self.usd:.2f}），停止运行')
                STOP.set()
            elif time.time() - last_log >= 300:
                last_log = time.time()
                log(f'!!! 额度：本次运行已花 {self.spent:.2f} / {self.usd:.2f} 美元')

    def close(self):
        """结束监控，返回本次运行花掉的美元数（再查一次；查不到就用最后一次的数）。"""
        self.done.set()
        try:
            self.spent = used_usd(self.env) - self.start
        except Exception:
            pass
        return self.spent
