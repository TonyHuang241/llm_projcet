"""从A股年报txt中提取“管理层讨论与分析”章节，去掉表格，只保留文字叙述。

年报txt由PDF转换而来：正文按页面宽度折行，表格被拆成每个单元格一行。
思路：定位章节 -> 去掉页眉页脚 -> 把折行拼回段落 -> 只保留完整句子和带正文的小标题。

用法: python src/extract_mda.py [年报zip路径] [输出目录]
"""
import re
import statistics
import sys
import zipfile
from collections import Counter
from pathlib import Path

# 各公司对该章节的叫法，按优先级排列（部分H股/银行放在“董事会报告”里）
MDA_TITLES = ['管理层讨论与分析', '管理层讨论和分析', '经营情况讨论与分析', '董事会报告']
# 章节标题行，如“第三节 管理层讨论与分析”；也可能“第三节”单独一行、标题在下一行
CHAPTER_RE = re.compile(r'第[一二三四五六七八九十]{1,3}[节章]\s*([\u4e00-\u9fa5、，\s]*)')
# 小标题编号，下标即层级：一、 > （一） > 1、 > （1） > 1）
HEADING_RES = [re.compile(p) for p in (
    r'[一二三四五六七八九十]+、',
    r'[（(][一二三四五六七八九十]+[）)]',
    r'\d{1,2}(\.\d{1,2})*[、．.\s](?!\d)',
    r'[（(]\d{1,2}[）)]',
    r'\d{1,2}[）)]',
)]
SENTENCE_END = tuple('。！？；：')
DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
PAGE_NUMBER_RE = re.compile(r'[-—\s]*(第\s*)?\d+\s*(页.*|/\s*\d+)?[-—\s]*')  # 如“12”“第12页”“12 / 230”
# 带页码的页脚，每页不同，无法按重复次数识别，如“78丨2024 年年度报告”“XX公司 2024年年度报告 第102页”
FOOTER_RE = re.compile(r'\d+\s*[丨|].{0,30}年度报告.{0,15}|.{0,40}年度报告\s*[丨|]?\s*(第\s*)?\d+\s*页?')


def visual_len(s):
    """近似显示宽度：中文算1，数字字母等半角字符算0.5。"""
    return sum(1 if ord(c) > 127 else 0.5 for c in s)


def find_mda_section(lines):
    """返回“管理层讨论与分析”章节的所有行，找不到返回空列表。"""
    chapters = []  # [(行号, 章节名)]
    for i, line in enumerate(lines):
        m = CHAPTER_RE.fullmatch(line.strip())  # 目录行带“....”，不会完整匹配
        if m:
            title = m.group(1) or (lines[i + 1] if i + 1 < len(lines) else '')
            chapters.append((i, re.sub(r'\s', '', title)))

    for name in MDA_TITLES:
        # 每次出现该标题（目录、正文、页眉）都算一个候选，到下一个其他章节为止；
        # 目录里的候选很短，取最长的一段就是正文章节
        spans = []
        for k, (start, title) in enumerate(chapters):
            if name in title:
                end = next((i for i, t in chapters[k:] if name not in t), len(lines))
                spans.append((start, end))
        if spans:
            start, end = max(spans, key=lambda s: s[1] - s[0])
            return lines[start:end]
    return find_old_mda_section(lines)


# 2010年前后的旧版年报，章节标题多写成“八、董事会报告”“§7 董事会报告”“第 七 章 董事局报告”“第八部分：董事会报告”。
# 章内小标题也用“一、”编号，所以只认“董事会报告”作起点、其后的固定章节名作终点
OLD_CHAPTER_MARK = (r'(?:§\d{1,2}|\d{1,2}[、．.]?|[一二三四五六七八九十]{1,3}、'
                    r'|第[一二三四五六七八九十\d]{1,3}(?:节|章|部分)[：:.、]?)')
OLD_MARK_ONLY_RE = re.compile(OLD_CHAPTER_MARK)
OLD_START_RE = re.compile(OLD_CHAPTER_MARK + r'董事[会局](?:工作)?报告')
OLD_END_RE = re.compile(OLD_CHAPTER_MARK + r'(?:监事会(?:工作)?报告|重要事项|财务(?:会计)?报告|备查文件目录)')


def find_old_mda_section(lines):
    """旧版年报的兜底定位：从“X、董事会报告”到下一个“Y、监事会报告”等章节，取最长的一段（目录中的很短）。"""
    compact = [re.sub(r'\s', '', line) for line in lines]
    for i, line in enumerate(compact):
        # 章节号单独一行（可能因PDF转换重复多次）、标题在后面几行，拼成一行再匹配
        if OLD_MARK_ONLY_RE.fullmatch(line):
            compact[i] = line + next((t for t in compact[i + 1:i + 6] if t and t != line), '')
    ends = [i for i, line in enumerate(compact) if OLD_END_RE.fullmatch(line)]
    spans = []
    for start in (i for i, line in enumerate(compact) if OLD_START_RE.fullmatch(line)):
        end = next((e for e in ends if e > start), None)
        if end is not None:
            spans.append((start + 1, end))  # 不含章节标题行本身
    if not spans:
        return []
    start, end = max(spans, key=lambda s: s[1] - s[0])
    return lines[start:end]


def remove_page_noise(section, all_lines):
    """去掉页码，以及页眉页脚这类在全文反复出现、又不是句子结尾的行。"""
    freq = Counter(line.strip() for line in all_lines)
    kept = []
    for line in map(str.strip, section):
        is_page_number = PAGE_NUMBER_RE.fullmatch(line) or FOOTER_RE.fullmatch(line)
        is_header_footer = freq[line] >= 5 and not line.endswith(SENTENCE_END)
        if line and not is_page_number and not is_header_footer:
            kept.append(line)
    return kept


def merge_paragraphs(lines):
    """把折行拼回段落：一行写满了页面宽度、且没以句号等结尾，说明段落还没结束。"""
    # 长行大多是写满的正文行，用其中位数估计页面宽度（表格单元格多为短行，不参与）
    long_lines = [visual_len(line) for line in lines if visual_len(line) >= 20]
    page_width = statistics.median(long_lines) if long_lines else 40

    paragraphs, buffer = [], ''
    for line in lines:
        is_short = visual_len(line) < page_width * 0.85
        # 小标题（不满一行、带编号、不以句号结尾）总是新段落的开头；
        # 满行或以句号结尾的可能是“21.3 亿元。”这类恰好以数字开头的折行
        if buffer and is_short and heading_level(line) is not None and not line.endswith(SENTENCE_END):
            paragraphs.append(buffer)
            buffer = ''
        buffer += line
        if is_short or line.endswith(SENTENCE_END):
            paragraphs.append(buffer)
            buffer = ''
    if buffer:
        paragraphs.append(buffer)
    return paragraphs


def heading_level(text):
    """是小标题则返回层级（越小越高），否则返回None。"""
    # 要求含3个以上连续汉字，排除“1.23 亿元”这类表格单元格
    if visual_len(text) > 30 or not re.search(r'[\u4e00-\u9fa5]{3,}', text):
        return None
    return next((level for level, p in enumerate(HEADING_RES) if p.match(text)), None)


def keep_narrative(paragraphs, min_len=15):
    """保留完整的叙述句段落（以句号等结尾，且不短于min_len）和小标题；表格单元格既短又不成句，会被过滤掉。"""
    blocks = []  # [(层级, 文本)]，正文层级为None
    for p in paragraphs:
        if p.endswith(SENTENCE_END) and visual_len(p) >= min_len:
            blocks.append((None, p))
        elif (level := heading_level(p)) is not None:
            blocks.append((level, p))

    # 倒序扫描，去掉下面没有正文的小标题（多为表格标题）
    kept = []
    for level, text in reversed(blocks):
        is_empty_heading = level is not None and not (
            kept and (kept[-1][0] is None or kept[-1][0] > level))
        if not is_empty_heading:
            kept.append((level, text))
    return [text for _, text in reversed(kept)]


def extract_mda(text):
    lines = text.splitlines()
    section = remove_page_noise(find_mda_section(lines), lines)
    return '\n'.join(keep_narrative(merge_paragraphs(section)))


def lower_priority():
    """Windows下把当前进程设为“低于正常”优先级，避免抢占其他正在运行的程序。"""
    if sys.platform == 'win32':
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


_zip = None  # 每个子进程各自打开一份zip（ZipFile对象不能跨进程传递）
_out = None


def _init_worker(zip_path, out_dir):
    global _zip, _out
    lower_priority()
    _zip, _out = zipfile.ZipFile(zip_path), Path(out_dir)


def _process_one(name):
    """处理一篇年报，返回 (文件名, 是否成功)。"""
    mda = extract_mda(_zip.read(name).decode('utf-8', errors='ignore'))
    if not mda:
        return name, False
    path = _out / name.split('_')[1] / name  # 按年份分子文件夹，文件名如“000001_2024_平安银行_….txt”
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(mda, encoding='utf-8')
    return name, True


def main(zip_path=DATA_DIR / 'annual_reports' / '2024_5405份.zip', out_dir=DATA_DIR / 'mda_texts', workers=16):
    """多进程并行处理一个年份的zip（正则解析是CPU密集型，用进程而非线程才能绕过GIL）。"""
    from concurrent.futures import ProcessPoolExecutor
    out = Path(out_dir)
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith('.txt')]

    failed = []
    with ProcessPoolExecutor(int(workers), initializer=_init_worker, initargs=(str(zip_path), str(out))) as pool:
        for i, (name, ok) in enumerate(pool.map(_process_one, names, chunksize=8), 1):
            if not ok:
                failed.append(name)
            if i % 500 == 0:
                print(f'  {Path(zip_path).name}: {i} / {len(names)}', flush=True)

    print(f'{Path(zip_path).name} 成功 {len(names) - len(failed)} / {len(names)}，结果保存在 {out}', flush=True)
    if failed:
        print('未找到章节：\n' + '\n'.join(failed), flush=True)


def main_years(start, end, out_dir=DATA_DIR / 'mda_texts', workers=16):
    """按年份从 start 往 end 依次处理（start > end 时倒序），每个年份内部多进程并行。"""
    step = -1 if start > end else 1
    for year in range(start, end + step, step):
        zips = sorted((DATA_DIR / 'annual_reports').glob(f'{year}_*.zip'))
        if not zips:
            print(f'{year}: 找不到年报zip，跳过', flush=True)
            continue
        main(zips[0], out_dir, workers)


if __name__ == '__main__':
    # 用法1: python src/extract_mda.py --years 2023 2010 [进程数]   按年份倒序批量处理
    # 用法2: python src/extract_mda.py [年报zip路径] [输出目录] [进程数]
    lower_priority()
    args = sys.argv[1:]
    if args and args[0] == '--years':
        main_years(int(args[1]), int(args[2]), workers=int(args[3]) if len(args) > 3 else 16)
    else:
        main(*args)
