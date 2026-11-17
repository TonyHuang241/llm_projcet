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
    MERGE_SYSTEM=1             可选，把 system prompt 拼进 user 消息发送（用于会丢弃 system 消息的中转站）

使用方：src/report_logic.py、src/report_compare.py。
"""
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


def log(msg):
    """打印并追加写入进度日志（带时间戳），多线程安全。"""
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
