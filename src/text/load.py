"""读 text 步骤的成品：正文段落表（data/hs300_2024_paragraphs.parquet）和研报元数据表（data/hs300_2024_reports.parquet），
按元数据表选研报、按研报读入段落。下游步骤通过这里读，不自己拼路径。"""
import hashlib
import re
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.config import PARAGRAPHS, REPORTS
from src.text.clean import read_text, reports_table_path, strip_risk  # noqa: F401（read_text、reports_table_path 供下游用）

META_COLS = ('file', 'date', 'broker', 'company', 'stock_code', 'title')
TABLE_COLS = ('date', 'broker', 'company', 'stock_code', 'report_type', 'analysts')  # 有元数据表时以它为准的研报信息
PREV_SENTENCES, PREV_CHARS = 2, 150  # 每段附带的前一段末尾（prev_tail）：句数、字数上限


def read_paragraphs(columns=None, path=PARAGRAPHS):
    """段落表（每行一个段落），返回 pandas.DataFrame；columns 只读这些列。"""
    return pq.read_table(path, columns=columns).to_pandas()


def read_meta(columns=None, path=REPORTS):
    """研报元数据表（每篇研报一行，列见 src/text/clean.py 开头），返回 pandas.DataFrame；columns 只读这些列。"""
    return pq.read_table(path, columns=columns).to_pandas()


def quarter_of(date):
    """YYYYMMDD -> 'YYYYQn'；日期缺失或格式不对时返回空串。"""
    return f'{date[:4]}Q{(int(date[4:6]) - 1) // 3 + 1}' if date and len(date) >= 6 and date[:6].isdigit() else ''


def tail(text):
    """段落末尾的几句，供下一段理解指代。"""
    parts = [s for s in re.split(r'(?<=[。！？；])', text) if s.strip()]
    return ''.join(parts[-PREV_SENTENCES:])[-PREV_CHARS:]


class Report:
    """一篇研报：rid、meta（研报信息）、paras（段落列表，每段含 report_id、para_id、text 等列和 prev_tail）。
    概念 DAG 抽取（src/graph/extract.py）在上面依次挂上 chunks（合并后的块）、table（概念表）、raw_edges、
    edges、removed、level、stats。"""

    def __init__(self, rid, meta, paras):
        self.rid, self.meta, self.paras = rid, meta, paras
        self.table = {}

    def para(self, pid):
        return next(p for p in self.paras if p['para_id'] == pid)


def select_reports(reports_path, select, min_per_quarter=1):
    """按元数据表筛选要抽取的研报，返回 ({report_id: 元数据表的一行}, 说明)；reports_path 为 None 时不筛选，返回 (None, 说明)。
    指定了元数据表而文件不存在时直接停下：不筛选就全部抽取的话，调用大模型的次数会多出几十倍。
    min_per_quarter：select 选中的研报再按（股票代码, 发布季度）分组，一组里不足这么多篇的整组不抽（同一家公司同一个季度
    只有 1 篇时没有别的研报可比）；没有股票代码或日期的研报不分组、原样保留。"""
    if reports_path is None:
        return None, '不按元数据表筛选，段落表里的研报全部抽取'
    if not Path(reports_path).exists():
        raise SystemExit(f'找不到研报元数据表 {reports_path}：请重新运行 python run.py text 生成（它和段落表一起写出）；'
                         '确实要把段落表里的研报全部抽取时加 --all')
    rows = [r for r in pq.read_table(reports_path).to_pylist() if r.get('status', 'ok') == 'ok']
    keep = {r['report_id']: r for r in rows if select is None or select(r)}
    note = f'研报元数据表 {Path(reports_path).name}：有正文的 {len(rows)} 篇里选中 {len(keep)} 篇'
    if min_per_quarter and min_per_quarter > 1:
        group = lambda r: (r.get('stock_code'), quarter_of(r.get('date')))
        size = defaultdict(int)
        for r in keep.values():
            size[group(r)] += 1
        thin = {rid for rid, r in keep.items() if all(group(r)) and size[group(r)] < min_per_quarter}
        keep = {rid: r for rid, r in keep.items() if rid not in thin}
        note += f'；其中 {len(thin)} 篇所在的公司当季不足 {min_per_quarter} 篇，不抽，剩 {len(keep)} 篇'
    return keep, note


def load_reports(paras_path, limit, info=None, seed=0):
    """读入段落 parquet（读入时再剔除一遍风险提示），按研报分组，返回 (按 report_id 排序的 Report 列表, 剔除的风险提示字数)。
    info：{report_id: 元数据表的一行}，给出时只读这些研报，研报的日期、券商、公司、代码、分析师以元数据表为准。
    limit：在这些研报里按种子 seed 随机抽多少篇（按 md5(种子:report_id) 排序取前 limit 篇，同一个种子下抽得少的
    总是抽得多的子集，放大样本时做完的可以沿用），None 为全部。"""
    table = pq.read_table(paras_path)
    missing = {'report_id', 'para_id', 'text'} - set(table.column_names)
    if missing:
        raise SystemExit(f'{paras_path} 缺少必需的列：{missing}')
    if info is not None:
        table = table.filter(pc.is_in(table['report_id'], value_set=pa.array(list(info), pa.string())))
    rows = [p for p in table.to_pylist() if (p.get('text') or '').strip()]
    paras = strip_risk(rows)
    n_risk = sum(len(p['text']) for p in rows) - sum(len(p['text']) for p in paras)
    report_ids = {p['report_id'] for p in paras}
    if limit is not None:
        report_ids = set(sorted(report_ids, key=lambda rid: hashlib.md5(f'{seed}:{rid}'.encode('utf-8')).hexdigest())[:limit])
    by_rid = defaultdict(list)
    for p in sorted(paras, key=lambda p: (p['report_id'], int(p['para_id']))):
        if p['report_id'] in report_ids:
            p['para_id'] = int(p['para_id'])
            by_rid[p['report_id']].append(p)
    reports = []
    for rid, ps in by_rid.items():
        for i, p in enumerate(ps):
            p['prev_tail'] = tail(ps[i - 1]['text']) if i else ''
        meta = {'report_id': rid, **{c: ps[0].get(c) or '' for c in META_COLS}, 'n_paragraphs': len(ps)}
        if info is not None:
            meta.update({c: info[rid].get(c) or '' for c in TABLE_COLS})
        reports.append(Report(rid, meta, ps))
    return reports, n_risk
