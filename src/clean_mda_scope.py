"""清理 2011-2014 年 MD&A 文本：去掉“董事会报告”章中不属于经营讨论和展望的小节，使范围与 2015 年以后一致。

2015 年之前的年报没有单独的“管理层讨论与分析”章，extract_mda 提取的是整章“董事会报告”，混有利润分配、社会责任、
内幕信息、董事会日常工作等内容。这里按小标题整节删除，只删整行，所以留下的句子与原文本一字不差。
原文本不会被修改，结果写到新目录，文件名不变。

用法：在 main.py 中设置 TASK = 'clean_mda_scope'
"""
import re
from pathlib import Path

from src.extract_mda import SENTENCE_END, heading_level

# 小标题含这些词的小节整节删除（只看前两级小标题：“一、”和“（一）”）
EXCLUDE_KEYS = (
    '利润分配', '分红派息', '分红政策', '现金分红', '公积金转增',
    '社会责任',
    '内幕信息',
    '董事会日常工作', '董事会会议', '董事会下设', '审计委员会', '薪酬与考核',
    '非标准', '会计差错', '会计政策', '会计估计', '合并报表范围', '财务会计报告审计', '会计师事务所',
    '其他披露事项', '其他需要披露', '其它需要披露', '其他事项', '其他重要事项',
    '信息披露媒体', '投资者关系', '内部控制',
    '核心竞争力',  # 2015-2020 年在“公司业务概要”章，不在提取范围内
)
# 这些一级小标题一定是要保留的小节，即使编号比被删小节小也结束删除
CORE_KEYS = ('未来发展', '主营业务分析', '经营情况', '投资状况', '投资情况', '资产、负债', '资产及负债',
             '控股参股', '接待调研', '特殊目的', '结构化主体', '管理层讨论')
TOP_RE = re.compile(r'([一二三四五六七八九十]+)、')
DIGITS = {c: i for i, c in enumerate('一二三四五六七八九', 1)}


def top_number(text):
    """一级小标题的编号（“十四、”返回 14），不是一级小标题返回 None。"""
    m = TOP_RE.match(text)
    if not m:
        return None
    s = m.group(1)
    if '十' not in s:
        return DIGITS.get(s[0]) if len(s) == 1 else None
    tens, _, ones = s.partition('十')
    return (DIGITS.get(tens, 1) if tens else 1) * 10 + (DIGITS.get(ones, 0) if ones else 0)


def clean_lines(lines, exclude_keys=EXCLUDE_KEYS):
    """返回 (保留的行, 被删小节的小标题列表)。

    遇到要删的小节后一直删到下一个同级或更高级的小标题；被删小节内部常有自己的“一、二、”，
    所以一级小标题的编号不大于被删小节时仍算在被删小节内，除非它是 CORE_KEYS 中的小节。
    """
    kept, removed = [], []
    drop_level = drop_number = None
    for line in lines:
        level = None if line.endswith(SENTENCE_END) else heading_level(line)
        if drop_level is not None:
            if level is None or level > drop_level:
                continue
            number = top_number(line) if level == 0 else None
            is_core = level == 0 and any(k in line for k in CORE_KEYS)
            if number is not None and drop_number is not None and number <= drop_number and not is_core:
                continue
            drop_level = drop_number = None
        if level in (0, 1) and any(k in line for k in exclude_keys):
            drop_level, drop_number = level, top_number(line) if level == 0 else None
            removed.append(line)
            continue
        kept.append(line)
    return kept, removed


def main(src_root, out_root, years=('2011', '2012', '2013', '2014')):
    for year in years:
        out = Path(out_root) / year
        out.mkdir(parents=True, exist_ok=True)
        files = sorted((Path(src_root) / year).glob('*.txt'))
        n_changed = 0
        for f in files:
            text = f.read_text(encoding='utf-8')
            kept, removed = clean_lines(text.splitlines())
            n_changed += bool(removed)
            (out / f.name).write_text('\n'.join(kept), encoding='utf-8')
        print(f'{year}: {len(files)} 篇，其中 {n_changed} 篇有删减，结果保存在 {out}', flush=True)
