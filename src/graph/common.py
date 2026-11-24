"""概念 DAG 抽取（extract.py）和名称统一（unify.py）共用、又不属于大模型调用的部分：输出目录的布局、名称和引文的规范化与核对、
提示词填空、写 parquet / js 等小函数。读本步骤成品的函数在 load.py。"""
import json
import re
import unicodedata
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

VIEWER_TEMPLATE = Path(__file__).resolve().parent / 'viewer.html'  # 查看页面的模板，复制到输出目录最外层的 概念DAG.html
DIRECTIONS = {'正向': 1, '负向': -1}
DIRECTION_ALIASES = {'正': '正向', '+': '正向', 'positive': '正向', '正相关': '正向',
                     '负': '负向', '-': '负向', 'negative': '负向', '负相关': '负向', '反向': '负向'}


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


def squash(s):
    """比对引文用：全角转半角，去掉空白和引号。"""
    return re.sub(r'[\s“”"\'‘’「」『』]', '', unicodedata.normalize('NFKC', str(s or '')))


def quote_in(quote, text, min_len=6):
    """引文是否逐字出自原文：允许用省略号连接几段原文，每段都要能在原文中找到，合计至少 min_len 个字。"""
    parts = [p for p in re.split(r'…+|\.{3,}', squash(quote)) if p]
    return sum(map(len, parts)) >= min_len and all(p in squash(text) for p in parts)


def para_block(paras):
    """[段落编号] 段落文字，空行分隔。"""
    return '\n\n'.join(f'[{p["para_id"]}] {p["text"]}' for p in paras)


# ==================== 提示词、回复校验、建图 ====================

def fill(template, **kw):
    for k, v in kw.items():
        template = template.replace('{' + k + '}', str(v))
    return template


def too_many(errs, total):
    """一次回复里不合格的条目多到要让模型重做：超过总数的五分之一（至少 1 条）。少量不合格的直接丢掉，免得为一两条反复重试。"""
    return len(errs) > max(1, total // 5)


def merge_direction(ds):
    """同一条边各处引文的方向：只有一种明确方向时取它，正负都有记为“矛盾”，都没说明记为“未说明”。"""
    known = set(ds) - {'未说明'}
    return known.pop() if len(known) == 1 else '矛盾' if known else '未说明'


# ==================== 写文件 ====================

def write_parquet(rows, path, columns):
    pq.write_table(pa.Table.from_pylist(rows) if rows else pa.table({c: pa.array([], pa.string()) for c in columns}), path)


def write_js(path, func, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f'{func}({json.dumps(obj, ensure_ascii=False, separators=(",", ":"))});\n', encoding='utf-8')
