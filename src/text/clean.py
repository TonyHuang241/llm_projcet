"""研报纯文本正文抽取：只保留正文段落，图表、表格、侧栏、页眉页脚、水印、风险提示和末尾的免责声明全部剔除，结果写成 parquet。

输入是已经从 PDF 抽成纯文本的研报（data/HS300_reports_2024/MM/DD/<GUID>.txt，UTF-8 为主，读不了时依次试 GB18030），
文件名是 GUID，正文和表格、图表刻度、侧栏、页眉页脚、免责声明混在一起，一段话常被拆成多行，几个表格单元格也常挤在一行。

这是概念 DAG 抽取（python run.py graph，src/graph/）之前的文本清理，单独运行，也可以单独替换：已有处理好的文本时，只要自己写出同样结构的 parquet，
直接从第二步开始即可。parquet 每行一个段落，必需的列：
    report_id  研报唯一标识（字符串，这里是文件名里的 GUID）
    para_id    段落在研报内的序号（整数，从 1 开始）
    text       段落文字
可选列（缺失时留空，只用于给大模型提供上下文和在 HTML 中展示）：
    file（相对输入目录的路径）、date、broker、company、stock_code（取法见 src/text/meta.py）、
    title（纯文本里没有可靠的标题位置，留空）、page（纯文本没有页码，留空）、section（段落所属的小标题）

同时写一张研报元数据表（每篇研报一行，路径见 reports_table_path：段落表文件名里的 paragraphs 换成 reports，另有同名 .csv
便于用 Excel 查看）：report_id、file、date、broker、report_type、is_stock_report（是不是个股公司研报）、company、stock_code、
analysts、analyst_ids、status（ok 有正文 / empty 空文件、扫描件、乱码、外文 / failed 读取失败）、n_paragraphs、n_chars。
输入目录里的每个文件都有一行，和段落表用 report_id 对应；概念 DAG 抽取按这张表筛选要抽取的研报（src/text/load.py 的 select_reports）。

纯文本没有字号和坐标，这里只看文字本身，按行（行内再按项目符号和连续空格切开）判断：
  1. 预处理：去掉模板标记（[Table_Title]、＃summary＃ 等）和行内的“请务必阅读…声明”类水印（“…信息披露和法律声明”
     “…重要声明及评级说明”连同和、及后面的部分一起删）；
     抽成“?”的项目符号和行首的 ●▌ 等符号处切开并强制分段；连续 3 个以上空格、句末标点后的 2 个以上空格处切开；
     资料来源和下一段正文抽在同一行的（“资料来源：公司公告，东北证券公司已经形成…”），按本篇资料来源惯用的结尾（券商名）切开；
  2. 末尾声明：从“免责声明”“分析师声明”“评级说明”等小标题（出现在全文字数的 40% 之后）开始，之后全部丢弃；
     以“和、及”开头的水印残段、带句读的句子、图表标题不算声明小标题，“信息披露”只认单独成行的（正文里有“加强信息披露监管”）；
  3. 逐行剔除：在全篇中重复出现 3 次以上的短行（页眉页脚）、页码、图表标题和资料来源、目录、相关研究、
     分析师联系方式；没有汉字的行（坐标轴刻度、图例数字）；数字和数字串占比高、没有句读的行（表格）；
  4. 拼段：没有句读的短行是侧栏、图例或小标题，后面紧跟正文的记为小标题（作为后续段落的 section），其余只作分段点；
     上一行没有以句号等结尾就接着拼，项目符号处另起一段；只有一行、没写满一行（按本篇折行的行宽估计）且不以句号等结尾的
     短段是小标题；后面跟着日期的行是相关研究的条目，丢弃；
  5. 段落后处理：剔除风险提示（风险小节整节丢弃，段首或句末之后出现的“风险提示：…”“3.8. 风险提示 …”截到段尾），丢掉声明套话、分析师简介、
     汉字占比或句读过少的段落（表格残留）、没有句末标点的短段（标题、侧栏残留）和重复段落（首页摘要与正文重复），
     过短的段落并入同一小节的相邻段落，过长的按句子切开。

各篇研报互不相关，用多进程并行处理（默认用满 CPU 核心，Windows 上一个进程池最多 60 个进程）；结果按文件顺序写出，
和单进程运行的结果完全相同。

运行：在仓库根目录执行 python run.py text（输入、输出路径、试跑篇数和进程数的默认值在 src/text/__init__.py 开头设置，也可以用 --limit N 试跑）。
"""
import csv
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.text.meta import META_COLUMNS, fill_analysts, fill_stocks, parse_meta

CJK_RE = re.compile(r'[一-鿿]')
SENT_END = '。！？；…'
SENT_RE = re.compile(rf'[^{SENT_END}]+(?:[{SENT_END}]+[”’）)]*|$)')
PROSE_PUNCT = '，。；、：'
BODY_PUNCT = '，。；、！？…'  # 有这些标点才算成句；冒号不算（侧栏字段“当前价格：”“投资评级：”也带冒号）

# 末尾声明部分的小标题（允许前面带编号和券商名）；只在全文字数的后 60% 生效，避免误伤首页“投资评级”一类的字段。
# 每页页脚的“请务必阅读…免责声明”“敬请参阅最后一页特别声明”之类的提示不算
# “免责”后面要跟声明、条款（正文里有“尽职免责”“期限免责”）；“信息披露”后面只能是结尾、冒号或分析师声明一类
# （正文里有“加强信息披露监管”“自愿性信息披露公告”“公司信息披露 长源电力”）
DISCLAIMER_HEAD_RE = re.compile(
    r'^[\s\d一二三四五六七八九十、.．【\[]*(.{0,8}(免责(声明|申明|条款)|分析师声明|分析师承诺|法律声明|特别声明|一般声明|重要声明|郑重声明|版权声明|'
    r'信息披露(?=$|[：:】\]]|(证券)?分析师)|分析师(介绍|简介)|研究团队简介|证券投资咨询业务)|.{0,8}评[级等]的?(说明|标准|体系|定义|比率))')
DISCLAIMER_HINT_RE = re.compile(r'请|敬请|阅读|参阅|关注|详见|见文末|见末页')
# 不是声明小标题：以“和、及、与”开头（页脚提示被水印规则删掉一半剩下的“和法律声明”“及评级说明”）、带句读（正文句子）、
# 图表标题（“表26：电子病历评级标准修订要点”）
DISCLAIMER_NOT_HEAD_RE = re.compile(r'^([和及、]|与(?!公司有关))|[，。；！？]|^(图|表|图表)\s*\d')
DISCLAIMER_HEAD_MAX_LEN = 16  # 评级体系类标题后面常跟括号说明，放宽到 40 字
DISCLAIMER_FROM = 0.4  # 声明小标题只在全文字数的这个比例之后查找
# 声明、评级定义、联系地址、分析师简介一类的套话：段落里出现就整段丢弃（声明小标题被截掉、只剩残段时靠这条兜底）
DISCLAIMER_TEXT_RE = re.compile(
    r'证券市场是一个风险无时不在|本(研究)?报告(仅|由|的|中的|所|版权|不构成|是基于|基于|旨在)|本公司(不会|及其|具有|保留|对本报告|已获|不对|不保证)|'
    r'投资者(应当?|不应|务必|自主|据此|需自行|应自行)|报告发布(之|当)?日(起|后)|以报告发布日|版权(所有|归)|未经.{0,15}(许可|授权|同意)|'
    r'本(申明|声明)|最终解释权|等地设有分支机构|分支机构正在建设|'
    r'邮政编码|邮编[:：]|地址[:：]|网址[:：]|传真[:：]|证券投资咨询|经营证券期货业务|执业资格|评级(说明|标准|体系|定义)|'
    r'(加入|就职于|任职于).{0,12}(证券|研究所|研究院|研究中心|研究部)|\d+年.{0,8}(证券|卖方|行业)?研究经验|'
    r'(买入|增持|中性|减持|卖出|推荐|回避|看好|看淡|持有|强于大市|弱于大市|同步大市|优于大市|跑赢行业|跑输行业)'
    r'[^，。；]{0,16}[：:（(].{0,20}(相对|超越|强于|弱于|高于|低于|介于|涨幅|跌幅|之间|以上|以下)')

# 模板标记和行内水印：先从行里删掉再判断
TEMPLATE_RE = re.compile(r'\[Table_\w*\]|＃[A-Za-z0-9_]+＃|glzqdatemark\d*')
# 页脚提示常是“…信息披露和法律声明”“…重要声明及评级说明”，和、及后面的部分要一起删掉，否则剩下的“和法律声明”会被当成声明小标题
WATERMARK_RE = re.compile(
    r'(请务必|请仔细|敬请|请)(阅读|参阅|参考|查阅|关注)[^。！？\n]{0,40}?(声明|申明|说明|条款|披露|内容)(部分|及其项下所有内容)?'
    r'(\s*[和及与、]\s*[^。！？\n]{0,10}?(声明|申明|说明|条款|披露|内容)(部分)?)*[。.]?|'
    r'请通过合法途径获取[^。\n]*?(。|$)(如经由[^。\n]*?(。|$))?|[\s　]*[1-9]\d{0,2}\s*/\s*[1-9]\d{0,2}\s*$')
# 抽成“?”的项目符号（左边是行首、空白、汉字或标点，右边是汉字、括号、数字）、私用区的符号字体字符；
# 左边是行首、空白或句末标点时右边也可以是字母（“。?Q3自产煤…”），汉字中间的“?X Elite”多半是 ® 等符号，不切
BULLET_Q_RE = re.compile(r'(?:^|(?<=[\s。；！]))[?-]+\s*(?=[A-Za-z一-鿿（(【“\d])|'
                         r'(?<=[一-鿿，：、）)”])[?-]+\s*(?=[一-鿿（(【“\d])')
LEAD_BULLET_RE = re.compile(r'^\s*(?:[•■-◿▀-▟☀-➿√✓※]+|[lnuqv](?=\s+[一-鿿]))\s*')
SPLIT_RE = re.compile(r'\s{3,}|(?<=[。！？；])\s{2,}|(?<=[。！？；])\s*(?=(?:图表?|表|资料来源|数据来源)\s*\d{0,2}\s*[：:])')

# 逐行剔除（任意长度）：图表标题、资料来源、目录引导点、相关研究、联系方式
CAPTION_RE = re.compile(
    r'^\s*(图|表|图表|附图|附表|Figure|Exhibit|Table|Chart)\s*([\d一二三四五六七八九十]+|$)|'
    r'^\s*(资料来源|数据来源|来源|注|备注|Source|Sources)\s*[:：]|\.{5,}|…{3,}|·{5,}|^\s*(图表|插图|表格|图|表)?目\s*录\s*$', re.I)
SHORT_CAPTION_RE = re.compile(r'(资料|数据)来源[:：]')  # 短行中间出现的资料来源（“历史表现 数据来源：wind…”）
# 资料来源行（前面可带页码）；有的版式资料来源和下一段正文抽在同一行（“资料来源：公司公告，东北证券公司已经形成…”）
SOURCE_LINE_RE = re.compile(r'^\d{0,3}\s*(资料|数据)?来源\s*[:：]')
# 资料来源后面切下来的部分以这些开头时是注释或来源的延续，不当正文
SOURCE_TAIL_SKIP_RE = re.compile(r'^([和及与、，,；;。.（(）)\[【]|(注|说明|备注|单位|其中)\d?\s*[:：]?|预测|整理|测算|估算|计算|绘制|制图)')
SOURCE_NOTE_RE = re.compile(r'\s*(预测|整理|测算)?[，,]?\s*(注|说明|备注)\d?\s*[:：][^。]*。')
META_RE = re.compile(
    r'^\s*(\d+[.、]\s*)?《|》\s*$|》[，,]?\s*[—\-]*\s*\d{4}[.\-/年]\d{1,2}|^\s*相关(研究|报告)|^\s*附(录|表)|'
    r'执业证书|执业编号|登记编号|执证编号|SAC\s*(执业|执证|编号|No)|S\d{13}|[\w.\-]+[@＠][\w\-]+\.[\w.]+|电话[:：]|邮箱[:：]|\d{3,4}-\d{7,8}', re.I)
# 只对短行生效（长行里出现这些词多半是正文）：页脚提示、分析师署名、副标题、侧栏字段
SHORT_DROP_RE = re.compile(
    r'请务必阅读|请仔细阅读|敬请参阅|免责声明|法律声明|仅供内部使用|请勿传阅|金融终端下载|版权所有|证券研究报告|扫码|二维码|'
    r'^.{0,12}(分析员|分析师|研究员|联系人|研究助理)|——.{0,6}(简评|点评|深度|报告)')
# 侧栏字段名：只对没有句读的短行生效（“…按照8月26日收盘价12.43元，对应PE…”是正文）
SIDEBAR_FIELD_RE = re.compile(r'^.{0,20}(市价|收盘价|总市值|流通市值|总股本|流通股本|每股净资产|市净率|资产负债率|一年最低|52\s*周|最新收盘)')
SHORT_DROP_MAX = 60
PAGE_NO_RE = re.compile(r'[-—第]*\d{1,3}[-—页]*|\d{1,3}/\d{1,3}|Page\d+', re.I)
DATE_ONLY_RE = re.compile(r'[-—]*\d{4}[-./年]\d{1,2}([-./月]\d{1,2}日?)?')
DATE_TAIL_RE = re.compile(r'[-—–]\s*\d{4}[-./]\d{1,2}([-./]\d{1,2})?\s*$')  # “…：标题-2024/02/25”式的相关研究条目
NUM_TOKEN_RE = re.compile(r'[-+−–(（]*\d[\d,.]*(%|[AEFaef]|[QH]\d?|x|X|倍|pct|pp|bp|bps)?[)）%]*|--?|—|N/?A|NM|nan', re.I)
# 段首的项目符号或编号：上一段已结束时据此另起一段；前面只有一行短句时，短句是这一组条目的小标题
BULLET_RE = re.compile(
    r'^\s*([•■-◿☀-➿-√✓①-⑳]|[\-–—*]\s|[(（]?\d{1,2}[)）]|\d{1,2}[、.．](?!\d)|[(（][一二三四五六七八九十]+[)）]|'
    r'[一二三四五六七八九十]+、|第[一二三四五六七八九十]+[，,、])')
RATING_RE = re.compile(r'买入|增持|推荐|中性|减持|跑赢|优于大市|评级|评等|^——')  # 首页的评级、副标题不作小标题
# 侧栏、图表的栏目名：即使后面紧跟正文也不作小标题
SIDEBAR_LABEL_RE = re.compile(r'市场走势|股价表现|股价走势|历史表现|市场表现|行情走势|走势图|股价相对|基本数据|主要数据|公司信息|公司数据|'
                              r'基础数据|股票数据|市场数据|交易数据|最高/最低|作者|分析师|研究员|联系人|沪深300|上证指数|深证成指|创业板指|收益率')
# 风险提示是按惯例罗列的风险清单，不属于推理链，整节剔除。小节标题（含首页侧栏的“风险”栏目名）整行是这些词时算风险小节，
# “关注贸易政策调整风险”这类正文小节不受影响
RISK_HEAD_RE = re.compile(r'^\W*([一二三四五六七八九十\d]+(\.\d+)*[、.．\s]\s*)?(主要|投资)?风险(提示|因素|分析|因素提示)?\W*$')
# 段首或句末之后出现的“风险提示”（前面可带项目符号和编号，“3.8. 风险提示”），以及任何位置的“风险提示：”（侧栏插进来的半句后面常直接接风险提示）：
# 从这里截到段尾
RISK_TEXT_RE = re.compile(r'(?:^|(?<=[。！？；]))\W{0,3}([一二三四五六七八九十\d]+(\.\d+)*[、.．]?\s*)?风险(提示|因素)|风险提示[:：]')

HEADING_MAX_LEN = 30
MIN_PARA_CHARS, MAX_PARA_CHARS = 60, 1500
MIN_CJK_SHARE = 0.3  # 汉字占比低于该值的段落视为表格残留
MIN_PUNCT_DENSITY = 1.5  # 每 100 字的句读数低于该值的段落视为表格、图例残留
REPEAT_MIN = 3  # 全篇重复出现这么多次的短行视为页眉页脚
MIN_REPORT_CJK = 200  # 汉字少于该数的文件视为空文件（扫描件、乱码、只有一串字符）或外文研报，不抽取

SCHEMA = pa.schema([('report_id', pa.string()), ('para_id', pa.int64()), ('file', pa.string()), ('date', pa.string()),
                    ('broker', pa.string()), ('company', pa.string()), ('stock_code', pa.string()), ('title', pa.string()),
                    ('page', pa.int64()), ('section', pa.string()), ('text', pa.string())])
REPORT_SCHEMA = pa.schema([('report_id', pa.string()), ('file', pa.string()), ('date', pa.string()), ('broker', pa.string()),
                           ('report_type', pa.string()), ('is_stock_report', pa.bool_()), ('company', pa.string()),
                           ('stock_code', pa.string()), ('analysts', pa.string()), ('analyst_ids', pa.string()),
                           ('status', pa.string()), ('n_paragraphs', pa.int64()), ('n_chars', pa.int64())])
PARA_META = ('file', 'date', 'broker', 'company', 'stock_code', 'title')  # 段落表里每行都带的研报信息
# 汇总里各类剔除内容的名称（字数都按去掉空白后计）
DROP_NAMES = {'disclaimer': '免责声明', 'risk': '风险提示', 'table': '表格', 'chart': '图表数字/无汉字行',
              'caption': '图表标题/资料来源/目录', 'header': '页眉页脚/水印/联系方式/相关研究', 'sidebar': '侧栏/图例/短行',
              'heading': '小标题（记入 section）', 'residue': '段落过滤（表格残留/短标题/重复段）', 'template': '模板标记'}


def cjk_share(text):
    t = re.sub(r'\s', '', text)
    return len(CJK_RE.findall(t)) / len(t) if t else 0.0


def nchars(text):
    """去掉空白后的字数，汇总统计都用它。"""
    return len(re.sub(r'\s', '', text))


def width(text):
    """排版宽度：汉字和全角符号算 1，半角字符算 0.5（判断一行是否写满）。"""
    return sum(1 if ord(c) > 0x2E7F else 0.5 for c in text if not c.isspace())


def is_prose(text):
    return len(text) >= 10 and cjk_share(text) >= 0.5 and any(p in text for p in PROSE_PUNCT)


def punct_density(text):
    """每 100 字中的中文句读数；正文一般在 5 以上，表格、图例拼出来的文字接近 0。"""
    return sum(text.count(p) for p in PROSE_PUNCT + '！？') * 100 / max(len(text), 1)


def ends_sentence(text):
    return text.rstrip()[-1:] in SENT_END + '”’）)' and not text.rstrip().endswith(('（', '('))


def read_text(path):
    """读文件并解码：UTF-8（含 BOM）→ UTF-16（有 BOM 时）→ GB18030，都不行时按 UTF-8 替换坏字节。返回 (文本, 编码)。"""
    raw = path.read_bytes()
    if raw[:2] in (b'\xff\xfe', b'\xfe\xff'):
        return raw.decode('utf-16', errors='replace'), 'utf-16'
    for enc in ('utf-8-sig', 'gb18030'):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            pass
    return raw.decode('utf-8', errors='replace'), 'utf-8(replace)'


def split_segments(text, drop):
    """全文 -> 片段列表 [{'text','hard'}]；hard 表示片段前有项目符号或空行，必须另起一段。删掉的模板标记、水印计入 drop。"""
    segs, hard = [], False
    for line in text.replace('\r', '\n').split('\n'):
        line = line.replace('　', ' ').replace('\xa0', ' ').replace('\t', ' ')
        clean = TEMPLATE_RE.sub(' ', line)
        drop['template'] += nchars(line) - nchars(clean)
        line = WATERMARK_RE.sub(' ', clean)
        drop['header'] += nchars(clean) - nchars(line)
        if not line.strip():
            hard = True
            continue
        m = LEAD_BULLET_RE.match(line)
        if m and m.end() < len(line):
            line, hard = line[m.end():], True
        for part in SPLIT_RE.split(line):
            pieces = BULLET_Q_RE.split(part)
            for i, p in enumerate(pieces):
                p = p.strip()
                if p:
                    segs.append({'text': p, 'hard': hard or i > 0})
                    hard = False
                elif i > 0:
                    hard = True
    return segs


def split_source_tail(segs):
    """资料来源和正文抽在同一行时从中切开。资料来源的最后一项是券商名（“东北证券”“国信证券经济研究所整理”），
    用本篇单独成行（不长、没有句末标点）的资料来源里出现两次以上的最后一项作为结尾，在片段里最后一次出现的结尾处切开
    （结尾后面紧跟一句“注：……。”时切在这句之后）；切下来的部分像正文（有句读、汉字过半、不以注释或标点开头）时
    另起一个片段，交给后面的规则判断，否则不动。"""
    srcs = [re.sub(r'\s', '', s['text']) for s in segs if SOURCE_LINE_RE.match(s['text'])]
    ends = Counter()
    for t in srcs:
        if width(t) <= SHORT_DROP_MAX and not re.search(f'[{SENT_END}]', t):
            last = SOURCE_LINE_RE.sub('', re.split(r'[，,、；;]', t)[-1])
            if 2 <= len(last) <= 20 and CJK_RE.search(last):
                ends[last] += 1
    ends = [k for k, v in ends.items() if v >= 2]
    if not ends:
        return segs
    out = []
    for s in segs:
        t = s['text']
        if SOURCE_LINE_RE.match(t) and width(t) > SHORT_DROP_MAX:
            cut = max((t.rfind(k) + len(k) for k in ends if k in t), default=0)
            note = SOURCE_NOTE_RE.match(t[cut:]) if cut else None
            if note:  # 资料来源后面先跟一句图注（“注：……。”）再接正文：图注留在资料来源里
                cut += note.end()
            rest = t[cut:].strip()
            if (cut and len(rest) >= 10 and cjk_share(rest) >= 0.5 and any(p in rest for p in BODY_PUNCT)
                    and not SOURCE_TAIL_SKIP_RE.match(rest)):
                out += [{**s, 'text': t[:cut].strip()}, {'text': rest, 'hard': True}]
                continue
        out.append(s)
    return out


def find_disclaimer_start(segs):
    """返回末尾声明部分第一个片段的序号；没有找到时返回 None。只在全文字数的 DISCLAIMER_FROM 之后查找。"""
    total = sum(len(s['text']) for s in segs)
    seen = 0
    for i, s in enumerate(segs):
        seen += len(s['text'])
        if seen < total * DISCLAIMER_FROM:
            continue
        t = re.sub(r'\s', '', s['text'])
        m = DISCLAIMER_HEAD_RE.match(t)
        if (m and not DISCLAIMER_HINT_RE.search(t) and not DISCLAIMER_NOT_HEAD_RE.search(t)
                and len(t) <= (40 if '评' in m.group(0) else DISCLAIMER_HEAD_MAX_LEN)):
            return i
    return None


def strip_repeated(segs):
    """标出全篇重复出现 REPEAT_MIN 次以上的短行（页眉、页脚、页码），按去掉数字和空白后的文字比较。"""
    key = lambda t: re.sub(r'[\d\s/]', '', t)
    cand = [s for s in segs if s['kind'] in ('body', 'label') and width(s['text']) <= SHORT_DROP_MAX]  # 表格行本来就会剔除
    seen = Counter(key(s['text']) for s in cand)
    for s in cand:
        k = key(s['text'])
        if len(CJK_RE.findall(k)) >= 2 and seen[k] >= REPEAT_MIN:
            s['kind'] = 'header'
    return segs


def is_table(plain, tokens):
    """表格行：没有句读、且有两个以上数字串或数字占比高；有句读的，只有数字串占多数又没有句号时才算。"""
    digits = sum(c.isdigit() for c in plain) / len(plain)
    nums = sum(bool(NUM_TOKEN_RE.fullmatch(t)) for t in tokens)
    if not any(p in plain for p in BODY_PUNCT):
        return nums >= 2 or digits >= 0.3
    return '。' not in plain and nums >= 4 and nums >= len(tokens) * 0.5


def classify(text):
    """返回 'body'、'label'（没有句读的短行：小标题、侧栏、图例，在 assemble 中再细分）或剔除的类别。"""
    plain = re.sub(r'\s', '', text)
    if PAGE_NO_RE.fullmatch(plain):
        return 'header'
    short = width(plain) <= SHORT_DROP_MAX
    if CAPTION_RE.search(text) or short and SHORT_CAPTION_RE.search(plain):
        return 'caption'
    if META_RE.search(text) or short and SHORT_DROP_RE.search(plain):  # 按去掉空白的文字匹配（“方  正  证  券  研  究  所”）
        return 'header'
    if short and SIDEBAR_FIELD_RE.search(plain) and not any(p in plain for p in BODY_PUNCT):
        return 'sidebar'
    if not CJK_RE.search(plain):
        # 没有汉字的行：只保留像英文句子的长行（有 the/and/of 等虚词，正文中的英文换行），图例、表格单元格、坐标轴刻度丢弃
        if len(plain) >= 20 and re.search(r'\b(the|and|of|to|in|for|with|is|are|on|by)\b', text, re.I) and not is_table(plain, text.split()):
            return 'body'
        return 'chart'
    if is_table(plain, text.split()):
        return 'table'
    return 'body' if any(p in plain for p in BODY_PUNCT) else 'label'


def line_width(segs):
    """本篇折行的“写满一行”宽度：句子没写完就换行的正文行宽，取较宽一半的中位数；这样的行少于 3 行时返回 None。"""
    ws = sorted(width(a['text']) for a, b in zip(segs, segs[1:])
                if a['kind'] == 'body' and b['kind'] == 'body' and not b['hard'] and 12 <= width(a['text']) <= 80
                and a['text'].rstrip()[-1:] not in SENT_END + '：:')
    if len(ws) < 3:
        return None
    top = ws[len(ws) // 2:]
    return top[len(top) // 2]


def is_title(lines, full_w, max_w=HEADING_MAX_LEN):
    """只有一行、没写满、不以句号逗号等结尾、句中没有句号分号的短句：小标题。"""
    if len(lines) != 1:
        return False
    t = lines[0].strip()
    limit = min(max_w, full_w * 0.75) if full_w else max_w
    return (width(t) <= limit and len(CJK_RE.findall(t)) >= 2 and t[-1:] not in SENT_END + '，,、”’）)'
            and not re.search('[。；]', t))


def assemble(segs, full_w, drop):
    """把片段拼成段落：返回 [{'section','text'}]。页眉页脚、图表标题、无汉字行不打断句子；表格和侧栏短行另起一段。"""
    seq = []
    for s in segs:
        if s['kind'] in ('header', 'caption', 'chart'):
            drop[s['kind']] += nchars(s['text'])
        else:
            seq.append(s)
    # 后面紧跟日期的行、以“-2024/02/25”结尾的行：相关研究条目
    for a, b in zip(seq, seq[1:] + [None]):
        if a['kind'] in ('body', 'label') and (b and DATE_ONLY_RE.fullmatch(re.sub(r'\s', '', b['text'])) or DATE_TAIL_RE.search(a['text'])):
            a['kind'] = 'meta'
    paras, cur, section = [], [], ''

    def flush():
        nonlocal section
        if cur:
            if is_title(cur, full_w):
                section = re.sub(r'\s+', ' ', cur[0]).strip()
                drop['heading'] += nchars(section)
            else:
                paras.append({'section': section, 'text': join_lines(cur)})
            cur.clear()

    for i, s in enumerate(seq):
        kind, t = s['kind'], s['text']
        nxt = next((x for x in seq[i + 1:i + 3] if x['kind'] != 'meta'), None)
        if kind == 'meta':
            drop['header'] += nchars(t)
            continue
        if kind == 'label':
            if full_w and width(t) >= full_w * 0.8 and cjk_share(t) >= 0.6:
                kind = 'body'  # 写满一行、只是恰好没有标点的正文行
            elif (RISK_HEAD_RE.match(t) or nxt and nxt['kind'] == 'body' and not RATING_RE.search(t) and not SIDEBAR_LABEL_RE.search(t)
                  and width(t) <= HEADING_MAX_LEN + 10 and len(CJK_RE.findall(t)) >= 2):
                flush()
                section = re.sub(r'\s+', ' ', t).strip()  # 紧跟正文的短行是小标题；风险栏目名要记成小节，后面的风险清单才能整节剔除
                drop['heading'] += nchars(t)
                continue
        if kind == 'body':
            if cur and (s['hard'] or ends_sentence(cur[-1]) or is_title(cur, full_w)
                        or BULLET_RE.match(t) and is_title(cur, None, HEADING_MAX_LEN + 10)):
                flush()
            cur.append(t)
            continue
        flush()
        drop['sidebar' if kind == 'label' else kind] += nchars(t)
    flush()
    return paras


def join_lines(lines):
    """中文行之间直接相连，英文/数字之间补一个空格。"""
    out = ''
    for t in lines:
        t = t.strip()
        if out and re.match(r'[A-Za-z0-9]', t) and re.search(r'[A-Za-z0-9,.%]$', out):
            out += ' '
        out += t
    return re.sub(r'\s+', ' ', out).strip()


def split_long(text, limit=MAX_PARA_CHARS):
    if len(text) <= limit:
        return [text]
    chunks, cur = [], ''
    for s in SENT_RE.findall(text):
        if cur and len(cur) + len(s) > limit:
            chunks.append(cur)
            cur = ''
        cur += s
    return chunks + [cur] if cur else chunks


def strip_risk(paras):
    """剔除风险提示：风险小节的段落整段丢弃，段首或句末之后出现“风险提示”的从那里截到段尾。"""
    out = []
    for p in paras:
        if RISK_HEAD_RE.match(p.get('section') or ''):
            continue
        m = RISK_TEXT_RE.search(p['text'])
        text = p['text'][:m.start()].rstrip() if m else p['text']
        if text:
            out.append({**p, 'text': text})
    return out


def tidy(paras, drop=None):
    """丢掉声明套话、表格残留、无句末标点的短段和重复段，短段并入同一小节的相邻段，长段按句切开。
    汉字占比低但有句号、句读密的段落（数字很多的业绩描述）保留。重复按去掉标点空白后的文字比较（首页摘要常与正文逐字重复）。"""
    drop = Counter() if drop is None else drop
    kept, seen = [], set()
    for p in paras:
        t = p['text']
        key = re.sub(r'[\W_]', '', t)
        if DISCLAIMER_TEXT_RE.search(t):
            drop['disclaimer'] += nchars(t)
        elif (len(CJK_RE.findall(t)) < 10 or punct_density(t) < MIN_PUNCT_DENSITY or key in seen
              or cjk_share(t) < MIN_CJK_SHARE and not ('。' in t and punct_density(t) >= 3)
              or len(t) < MIN_PARA_CHARS and not re.search(f'[{SENT_END}]', t)):
            drop['residue'] += nchars(t)
        else:
            kept.append(p)
            seen.add(key)
    merged = []
    for p in kept:
        if merged and merged[-1]['section'] == p['section'] and (
                len(p['text']) < MIN_PARA_CHARS or len(merged[-1]['text']) < MIN_PARA_CHARS):
            merged[-1]['text'] += p['text']
        else:
            merged.append(dict(p))
    return [{**p, 'text': t} for p in merged for t in split_long(p['text'])]


def extract_text(text, drop=None):
    """一篇研报的全文 -> 段落列表 [{'section','text'}]。drop 累计各类被剔除内容的字数。"""
    drop = Counter() if drop is None else drop
    segs = split_source_tail(split_segments(text, drop))
    cut = find_disclaimer_start(segs)
    if cut is not None:
        drop['disclaimer'] += sum(nchars(s['text']) for s in segs[cut:])
        segs = segs[:cut]
    for s in segs:
        s['kind'] = classify(s['text'])
    strip_repeated(segs)
    paras = assemble(segs, line_width(segs), drop)
    no_risk = strip_risk(paras)
    drop['risk'] += sum(nchars(p['text']) for p in paras) - sum(nchars(p['text']) for p in no_risk)
    return tidy(no_risk, drop)


def process_file(path, root):
    """返回 (状态, 元数据, 段落列表, 剔除字数, 原文字数)；状态为 ok / empty（空文件、扫描件、乱码、外文研报）。"""
    text, _ = read_text(path)
    total = nchars(text)
    if len(CJK_RE.findall(text)) < MIN_REPORT_CJK:
        return 'empty', {'file': path.relative_to(root).as_posix()}, [], Counter(), total
    drop = Counter()
    paras = extract_text(text, drop)
    return 'ok', parse_meta(path, root, text), paras, drop, total


# ==================== 多进程 ====================

MAX_POOL = 60  # Windows 上一个进程池最多等待 63 个句柄，进程数超过约 60 会报错
CHUNK = 16  # 每次交给一个进程的文件数：太小进程间通信开销大，太大进度条不平滑


def default_workers():
    n = os.cpu_count() or 1
    return min(n, MAX_POOL) if os.name == 'nt' else n


def _init_worker(root):
    global _ROOT
    _ROOT = root


def _work(path):
    """在子进程里处理一篇，返回 (process_file 的结果, 错误信息)；单个文件出错（读不了、格式异常）只记下来，不影响其他文件。"""
    try:
        return process_file(path, _ROOT), None
    except Exception as e:
        return None, f'{type(e).__name__}: {e}'


def run_files(files, root, workers):
    """按 files 的顺序逐篇给出 (文件, 结果, 错误信息)；workers > 1 时用进程池并行，结果仍按原顺序返回。"""
    if workers <= 1:
        _init_worker(root)
        for f in files:
            yield (f, *_work(f))
        return
    with mp.Pool(workers, initializer=_init_worker, initargs=(root,)) as pool:
        for f, (res, err) in zip(files, pool.imap(_work, files, chunksize=CHUNK)):
            yield f, res, err


def reports_table_path(paras_path):
    """段落表对应的研报元数据表路径：文件名里的 paragraphs 换成 reports（没有这个词就在后面加 _reports）。"""
    p = Path(paras_path)
    stem = p.stem.replace('paragraphs', 'reports') if 'paragraphs' in p.stem else p.stem + '_reports'
    return p.with_name(stem + '.parquet')


def report_row(f, root, status, meta=None, n_paragraphs=0, n_chars=0):
    """元数据表的一行；没有正文或读取失败的文件只有 report_id、file、date、status。"""
    meta = meta or {}
    parts = f.relative_to(root).parts[:-1]
    year = re.search(r'(?:19|20)\d{2}', root.name)
    date = year.group(0) + parts[-2] + parts[-1] if year and len(parts) >= 2 and parts[-2].isdigit() and parts[-1].isdigit() else ''
    return {'report_id': f.stem, 'file': f.relative_to(root).as_posix(), 'date': meta.get('date') or date,
            **{k: meta.get(k, False if k == 'is_stock_report' else '') for k in META_COLUMNS if k != 'date'},
            'status': status, 'n_paragraphs': n_paragraphs, 'n_chars': n_chars}


def sample_files(files, limit, seed):
    """跨月份抽样：按所在月份目录分组，每组用固定种子打乱后轮流取，保证小样本覆盖所有月份。"""
    groups = {}
    for f in files:
        groups.setdefault(f.parent.parent.name, []).append(f)
    rng = random.Random(seed)
    for k in sorted(groups):
        rng.shuffle(groups[k])
    picked, i = [], 0
    while len(picked) < limit and any(i < len(g) for g in groups.values()):
        picked += [groups[k][i] for k in sorted(groups) if i < len(groups[k])]
        i += 1
    return sorted(picked[:limit])


class SimpleBar:
    """没有 tqdm 时的进度条：已处理/总数、百分比、速度、预计剩余时间，用 \\r 原地刷新。"""

    def __init__(self, total):
        self.total, self.n, self.t0, self.last = total, 0, time.time(), 0.0

    def update(self, k=1):
        self.n += k
        now = time.time()
        if now - self.last < 0.2 and self.n < self.total:
            return
        self.last = now
        rate = self.n / max(now - self.t0, 1e-9)
        eta = (self.total - self.n) / rate if rate else 0
        filled = int(30 * self.n / max(self.total, 1))
        sys.stderr.write(f'\r|{"#" * filled}{"-" * (30 - filled)}| {self.n}/{self.total} {self.n * 100 / max(self.total, 1):5.1f}% '
                         f'{rate:6.1f} 篇/秒 剩余 {int(eta // 60):02d}:{int(eta % 60):02d}')
        sys.stderr.flush()

    def close(self):
        sys.stderr.write('\n')


def progress(total):
    try:
        from tqdm import tqdm
        return tqdm(total=total, unit='篇', dynamic_ncols=True, smoothing=0.05)
    except ImportError:
        return SimpleBar(total)


def main(txt_dir, out_path, limit=None, seed=0, workers=None):
    """txt_dir 下的所有 .txt -> out_path（parquet，每行一个段落），同时在旁边写一份 .txt 便于人工检查抽取效果。
    limit：只跨月份随机抽这么多篇（固定种子 seed，可复现），None 为全部。
    workers：并行的进程数，None 为用满 CPU 核心（见 default_workers），1 为不开子进程。"""
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    root, out_path = Path(txt_dir), Path(out_path)
    files = sorted(root.rglob('*.txt'))
    if not files:
        raise SystemExit(f'{root} 下没有 .txt 文件')
    if limit:
        files = sample_files(files, limit, seed)
    workers = max(1, min(workers or default_workers(), len(files) // CHUNK or 1))  # 文件很少时不必开那么多进程
    print(f'{len(files)} 个文件，{workers} 个进程并行' if workers > 1 else f'{len(files)} 个文件，单进程')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_parquet, tmp_preview = out_path.with_name(out_path.name + '.tmp'), out_path.with_suffix('.txt.tmp')
    status, drop, failed, report_rows, analyst_pairs = Counter(), Counter(), [], [], {}
    total_chars = kept_chars = n_paras = 0
    filled = Counter()
    rows, t0 = [], time.time()
    writer = pq.ParquetWriter(tmp_parquet, SCHEMA)
    bar = progress(len(files))
    try:
        with open(tmp_preview, 'w', encoding='utf-8') as preview:
            for f, res, err in run_files(files, root, workers):
                if err:  # 单个文件出错（读不了、格式异常）记下来继续
                    failed.append((f.relative_to(root).as_posix(), err))
                    status['failed'] += 1
                    report_rows.append(report_row(f, root, 'failed'))
                    bar.update()
                    continue
                st, meta, paras, d, total = res
                status[st] += 1
                total_chars += total
                drop.update(d)
                if st == 'ok' and not paras:
                    status['ok_no_paragraph'] += 1
                report_rows.append(report_row(f, root, st if paras or st != 'ok' else 'empty', meta, len(paras),
                                              sum(nchars(p['text']) for p in paras)))
                if meta.get('analyst_pairs'):
                    analyst_pairs[f.stem] = meta['analyst_pairs']
                if paras:
                    rid = f.stem
                    filled.update(k for k in ('date', 'broker', 'company', 'stock_code', 'analysts', 'analyst_ids', 'is_stock_report')
                                  if meta.get(k))
                    filled['type_' + meta['report_type']] += 1
                    filled['reports'] += 1
                    preview.write(f'########## {rid} {meta["date"]} {meta["broker"]} {meta["company"]} {meta["stock_code"]} {meta["file"]}\n')
                    for i, p in enumerate(paras, 1):
                        rows.append({'report_id': rid, 'para_id': i, **{k: meta[k] for k in PARA_META}, 'page': None,
                                     'section': p['section'], 'text': p['text']})
                        preview.write(f'[{i}] 〔{p["section"]}〕 {p["text"]}\n')
                        kept_chars += nchars(p['text'])
                    n_paras += len(paras)
                if len(rows) >= 20000:
                    writer.write_table(pa.Table.from_pylist(rows, SCHEMA))
                    rows = []
                bar.update()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, SCHEMA))
    finally:
        writer.close()
        bar.close()
    # 全部处理完后，用其他研报里认出的对应关系再补一遍：有编号没有姓名的分析师；纯文本里没有代码或公司名的个股研报
    # （公司和代码同步到段落表；预览 txt 每篇开头那一行不再改，仍是补之前的）
    fill_analysts(report_rows, analyst_pairs)
    changed = set(fill_stocks(report_rows, lambda r: read_text(root / r['file'])[0]))
    if changed:
        info = {r['report_id']: r for r in report_rows if r['report_id'] in changed}
        paras = pq.read_table(tmp_parquet)
        rids = paras['report_id'].to_pylist()
        for col in ('company', 'stock_code'):
            old = paras[col].to_pylist()
            paras = paras.set_column(paras.schema.get_field_index(col), col,
                                     pa.array([info[rid][col] if rid in info else v for rid, v in zip(rids, old)], pa.string()))
        pq.write_table(paras, tmp_parquet)
    tmp_parquet.replace(out_path)
    tmp_preview.replace(out_path.with_suffix('.txt'))
    table_path = reports_table_path(out_path)
    table = pa.Table.from_pylist(report_rows, REPORT_SCHEMA)
    pq.write_table(table, table_path)
    with open(table_path.with_suffix('.csv'), 'w', encoding='utf-8-sig', newline='') as fh:  # 带 BOM，Excel 直接打开不乱码
        w = csv.writer(fh)
        w.writerow(REPORT_SCHEMA.names)
        w.writerows([r[k] for k in REPORT_SCHEMA.names] for r in report_rows)
    err_path = out_path.with_name(out_path.stem + '_errors.txt')
    if failed:
        err_path.write_text(''.join(f'{f}\t{e}\n' for f, e in failed), encoding='utf-8')

    n_rep = filled['reports']
    done = [r for r in report_rows if r['status'] == 'ok']
    for k in ('stock_code', 'is_stock_report', 'analysts'):  # 补过之后重新数
        filled[k] = sum(bool(r[k]) for r in done)
    stock_known = sum(bool(r['stock_code']) for r in done if r['is_stock_report'])
    print(f'\n===== 完成：用时 {time.time() - t0:.0f} 秒 =====')
    print(f'文件 {len(files)} 篇：有正文 {n_rep} 篇，空文件/扫描件/乱码/外文 {status["empty"]} 篇，'
          f'有中文但没留下段落 {status["ok_no_paragraph"]} 篇，读取失败 {status["failed"]} 篇'
          + (f'（清单见 {err_path.name}）' if failed else ''))
    print(f'保留段落 {n_paras} 个、{kept_chars} 字，占原文 {total_chars} 字（不含空白）的 {kept_chars / max(total_chars, 1):.1%}')
    print('剔除内容字数：' + '，'.join(f'{name} {drop[k]}（{drop[k] / max(total_chars, 1):.1%}）'
                                  for k, name in DROP_NAMES.items() if drop[k]))
    other = total_chars - kept_chars - sum(drop.values())
    print(f'空文件等未抽取的字数及拼接、截断误差 {other}（{other / max(total_chars, 1):.1%}）')
    print('元数据填充率（有正文的研报中）：' + '，'.join(
        f'{name} {filled[k] / max(n_rep, 1):.1%}' for k, name in (('date', '发布日期'), ('broker', '券商'), ('stock_code', '公司和代码'),
                                                              ('analysts', '分析师姓名'), ('analyst_ids', '执业证书编号'))))
    print('报告类型：' + '，'.join(f'{k[5:]} {v}' for k, v in filled.most_common() if k.startswith('type_'))
          + f'；个股公司研报 {filled["is_stock_report"]} 篇（{filled["is_stock_report"] / max(n_rep, 1):.1%}），'
          f'其中认出是哪只股票的 {stock_known} 篇（{stock_known / max(filled["is_stock_report"], 1):.1%}）')
    print(f'已写入 {out_path}（预览：{out_path.with_suffix(".txt").name}）')
    print(f'研报元数据表 {table_path}（{len(report_rows)} 行，另有同名 .csv），和段落表用 report_id 对应')

