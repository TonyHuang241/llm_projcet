"""从A股年报txt中提取财务报表审计报告的正文。

审计报告位于“财务报告”一节开头，正文结构固定：
一、审计意见 …… 注册会计师对财务报表审计的责任 -> 会计师事务所签章（很多年报省略签章，直接接财务报表）。
思路：定位正文 -> 去掉页眉页脚 -> 把折行拼回段落 -> 与MD&A相同，只保留完整句子和带正文的小标题。
关键审计事项常排成两栏表格，栏里是正文，要拼回段落；表头、事项名称、混入的数据表等单元格则被过滤。

用法：在 main.py 中设置 TASK = 'extract_audit'
"""
import re
import zipfile
from pathlib import Path

from .extract_mda import DATA_DIR, SENTENCE_END, keep_narrative, remove_page_noise, visual_len

NUMBER = r'([一二三四五六七八九十]+、|[（(]([一二三四五六七八九十]+|\d{1,2})[）)]|\d{1,2}[、.．)）](?!\d))'


def title_re(name):
    """标题行的正则：标题前可能带项目符号和编号，末尾可能带冒号。"""
    return re.compile(r'[\uf0b7·●•]?\s*' + NUMBER + r'?\s*(' + name + r')\s*[：:]?')


# 起点：“形成审计意见的基础”之前最近的意见标题，如“一、审计意见”（非标意见为“一、保留意见”等）。
# 内部控制审计报告、H股核数师报告的结构不同，不会被选中
OPINION_RE = title_re('审计意见|保留意见|否定意见|无法表示意见')
BASIS_RE = title_re(r'形成\S{0,6}意见的基础')
# 终点：正文最后一部分“注册会计师对财务报表审计的责任”之后，第一个签章行或财务报表标题
LAST_PART_RE = title_re(r'注册会计师对\S{0,4}财务报表\S{0,2}的责任')  # 如“注册会计师对合并财务报表审计的责任”
END_RE = re.compile(
    r'.*无正文.*|[（(]此页.*'  # “（此页无正文，为……审计报告之签字盖章页）”“（此页为……报告之签字盖章页。）”
    r'|[^，。；]*(会计师事务所|中国注册会计师(?!审计|职业)|项目合伙人)[^，。；]*'  # 签章行
    r'|(二、\s*)?财务报表|(\d[、.．]\s*)?(合并)?资产负债表')  # 财务报表标题
# 单独成行的小标题：带编号的标题、审计报告的标准章节名（可能不带编号），或关键审计事项的栏目名
HEADING_RE = re.compile(NUMBER + r'\s*[^，。；：]{1,20}')
SECTION_RE = title_re(r'审计意见|保留意见|否定意见|无法表示意见|形成\S{0,6}意见的基础|强调事项|其他事项'
                      r'|其他信息|与持续经营相关的重大不确定性|管理层和治理层对财务报表的责任|注册会计师对\S{0,4}财务报表\S{0,2}的责任')
LABEL_RE = re.compile(r'[^，。；：]{0,12}(关键审计事项|应对|事项描述)[^，。；：]{0,6}')
# 跨页时重复的标题和页眉，如“三、关键审计事项（续）”“审计报告 - 续(第3页，共5页)”“审计报告 第4页”
PAGE_HEADER_RE = re.compile(r'.*[（(]\s*续\s*[）)]|.{0,30}审计报告[\s\-—(（]*(续|第\s*\d+\s*页).{0,20}')
CELL_PUNCTUATION_RE = re.compile('[，、。；：！？]')


def heading_text(lines, i):
    """第i行的文字；编号和标题被拆成两行时（如“一、”“审计意见”）拼在一起。"""
    text = lines[i].strip()
    if re.fullmatch(NUMBER, text) and i + 1 < len(lines):
        text += lines[i + 1].strip()
    return text


def find_starts(lines):
    """所有可能的正文起始行：先找“形成审计意见的基础”，再往前找离它最近的意见标题（避开审计报告前
    摘要表里的“审计意见”字样）和“我们审计了”，取靠前的一个。
    （有的报告意见标题在“我们审计了”之后，如“(一) 我们审计的内容……(二) 保留意见”）"""
    starts = []
    for basis in (i for i in range(len(lines)) if BASIS_RE.fullmatch(heading_text(lines, i))):
        window = range(basis - 1, max(basis - 40, -1), -1)
        opinion = next((i for i in window if OPINION_RE.fullmatch(heading_text(lines, i))), None)
        audited = next((i for i in window if lines[i].strip().startswith('我们审计了')), None)
        candidates = [i for i in (opinion, audited) if i is not None]
        if candidates:
            starts.append(min(candidates))
    return starts


def find_audit_section(lines):
    """返回审计报告正文的所有行（已去掉页眉页脚），找不到返回空列表。"""
    for start in find_starts(lines):
        # 财务报表每页都有“编制单位：”，会被当成页眉去掉，所以先在原文中截断
        stop = next((i for i in range(start, len(lines)) if lines[i].strip().startswith('编制单位')), len(lines))
        # 再去页眉页脚，避免页脚里的事务所名称被误当成终点
        body = remove_page_noise(lines[start:stop], lines)
        last_part = next((i for i, line in enumerate(body) if LAST_PART_RE.fullmatch(line)), 0)
        end = next((i for i in range(last_part + 1, len(body)) if END_RE.fullmatch(body[i])),
                   len(body) if stop < len(lines) else None)
        section = body[:end] if end else []
        # 非标意见的公司会在“重要事项”中引用审计报告的标题，从那里截取会把真正的审计报告也包进来，跳过
        if section and sum(bool(BASIS_RE.fullmatch(line)) for line in section) <= 1:
            return [line for line in section if not PAGE_HEADER_RE.fullmatch(line)]
    return []


def looks_like_heading(text):
    """只看格式判断是否像小标题。"""
    if SECTION_RE.fullmatch(text):  # 标准章节名末尾可能带冒号，如“一、审计意见：”
        return True
    return bool(HEADING_RE.fullmatch(text) or LABEL_RE.fullmatch(text)) and not text.endswith(SENTENCE_END)


def is_heading(lines, i):
    """第i行是否为单独成行的小标题。带编号的短行也可能是窄栏里一句话的开头
    （如“1.了解并测试与存货可变现净值的评估相”），而真正的小标题后面通常是更长的正文或另一个标题。"""
    text = lines[i]
    if not looks_like_heading(text):
        return False
    if SECTION_RE.fullmatch(text) or not HEADING_RE.fullmatch(text) or i + 1 == len(lines):
        return True
    return looks_like_heading(lines[i + 1]) or visual_len(text) < 0.8 * visual_len(lines[i + 1])


def is_table_cell(lines, i):
    """第i行是否为单独的表格单元格：没有汉字的数据，如“295,781,382.22”；或关键审计事项名称，
    如“收入确认”（右边一栏是事项描述）。窄栏里的正文每行也很短，但宽度相近；名称明显比下一行短，且不带标点。"""
    text = lines[i]
    if all(ord(c) < 128 for c in text):
        return True
    return (visual_len(text) < 12 and not CELL_PUNCTUATION_RE.search(text)
            and i + 1 < len(lines) and visual_len(text) < 0.5 * visual_len(lines[i + 1]))


def merge_paragraphs(lines):
    """把折行拼回段落：一直拼到句号等结尾为止，小标题和表格单元格单独成行。
    （关键审计事项的表格栏很窄，不能像MD&A那样按页面宽度判断折行）"""
    joined = []  # 先把被拆成两行的编号和标题拼起来，如“一、”“审计意见”
    for line in lines:
        if joined and re.fullmatch(NUMBER, joined[-1]):
            joined[-1] += line
        else:
            joined.append(line)

    paragraphs, buffer = [], ''
    for i, line in enumerate(joined):
        # 单元格（如事项名称）只出现在段首，句子中间的短行是正文折行
        if is_heading(joined, i) or not buffer and is_table_cell(joined, i):
            paragraphs += split_unfinished(buffer) + [line]
            buffer = ''
            continue
        buffer += line
        if buffer.endswith(SENTENCE_END):
            paragraphs.append(buffer)
            buffer = ''
    return paragraphs + split_unfinished(buffer)


def split_unfinished(buffer):
    """没以句号等结尾就被打断的段落：把前面完整的句子和末尾的残句分开，残句之后会被过滤掉。"""
    cut = max(buffer.rfind(c) for c in SENTENCE_END) + 1
    return [p for p in (buffer[:cut], buffer[cut:]) if p]


def extract_audit(text):
    # 审计程序多为短句，如“（6）检查期后回款情况。”，最短长度比MD&A放宽
    paragraphs = merge_paragraphs(find_audit_section(text.splitlines()))
    return '\n'.join(keep_narrative(paragraphs, min_len=5))


def main(zip_path=DATA_DIR / 'annual_reports' / '2024_5405份.zip', out_dir=DATA_DIR / 'audit_texts'):
    out = Path(out_dir)
    failed = []
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith('.txt')]
        for name in names:
            audit = extract_audit(z.read(name).decode('utf-8', errors='ignore'))
            if audit:
                path = out / name.split('_')[1] / name  # 按年份分子文件夹，文件名如“000001_2024_平安银行_….txt”
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(audit, encoding='utf-8')
            else:
                failed.append(name)

    print(f'成功 {len(names) - len(failed)} / {len(names)}，结果保存在 {out}')
    if failed:
        print('未找到审计报告：\n' + '\n'.join(failed))
