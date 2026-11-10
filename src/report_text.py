"""研报 PDF 正文抽取：只保留正文段落，图表、表格、侧栏、页眉页脚、下载水印和末尾的免责声明全部剔除，结果写成 parquet。

这是推理链抽取（src/report_logic.py）的第一步，可以单独替换：已有处理好的文本时，只要自己写出同样结构的 parquet，
直接从第二步开始即可。parquet 每行一个段落，必需的列：
    report_id  研报唯一标识（字符串）
    para_id    段落在研报内的序号（整数，从 1 开始）
    text       段落文字
可选列（缺失时留空，只用于给大模型提供上下文和在 HTML 中展示）：
    file、date、broker、company、stock_code、title、page、section（段落所属的小标题）

各家券商版式不同（字号、侧栏位置、表格样式都不一样），这里不写死页码和坐标，而是按行判断：
  1. 末尾声明：从“免责声明”“分析师声明”“评级说明”等小标题（出现在后半篇）开始，之后全部丢弃；
  2. 正文字号：取“像正文”的行（含中文标点、汉字占比高）按字数加权的主要字号，图表内文字、表格、侧栏通常字号更小；
  3. 逐行剔除：页边距内的行、多页重复出现的页眉页脚、图表标题和资料来源、下载水印、分析师联系方式、竖排文字；
  4. 拼段：上一行没有以句号等结尾就接着拼；以句号结尾且下一行缩进、换项目符号、上一行没写满或行距变大时另起一段；
     小标题单独识别，作为后续段落的 section；
  5. 段落后处理：丢掉汉字占比过低的段落（表格残留），过短的段落并入同一小节的相邻段落，过长的按句子切开。

用法：在 main.py 中设置 TASK = 'report_text'
"""
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pymupdf

FILE_RE = re.compile(r'^(?P<date>\d{8})-(?P<broker>[^-]+)-(?P<company>[^（(]+)[（(](?P<code>[0-9A-Za-z.]+)[）)].*?[：:](?P<title>.+)$')
CJK_RE = re.compile(r'[\u4e00-\u9fff]')
SENT_END = '。！？；…'
SENT_RE = re.compile(rf'[^{SENT_END}]+(?:[{SENT_END}]+[”’）)]*|$)')
PROSE_PUNCT = '，。；、：'

# 末尾声明部分的小标题（允许前面带编号和券商名）；只在后半篇生效，避免误伤首页“投资评级”一类的字段。
# 每页页脚的“请务必阅读…免责声明”“敬请参阅最后一页特别声明”之类的提示不算
DISCLAIMER_HEAD_RE = re.compile(
    r'^[\s\d一二三四五六七八九十、.．]*(.{0,8}(免责|分析师声明|分析师承诺|法律声明|特别声明|一般声明|重要声明|版权声明|'
    r'信息披露|分析师(介绍|简介)|研究团队简介|证券投资咨询业务)|.{0,8}评级的?(说明|标准|体系|定义|比率))')
DISCLAIMER_HINT_RE = re.compile(r'请|敬请|阅读|参阅|关注|详见|见文末|见末页')
DISCLAIMER_HEAD_MAX_LEN = 16  # 评级体系类标题后面常跟括号说明，放宽到 40 字
# 逐行剔除：水印、免责提示、分析师联系方式、图表标题和资料来源
DROP_LINE_RE = re.compile(
    r'请务必阅读|请仔细阅读|敬请参阅|免责声明|法律声明|仅供内部使用|请勿传阅|金融终端下载|版权所有|证券研究报告|'
    r'\.{5,}|…{3,}|·{5,}|^\s*(图表)?目\s*录\s*$|'  # 目录页的引导点
    r'^\s*(\d+[.、]\s*)?《|》\s*$|》[，,]?\s*\d{4}[.\-/年]\d{1,2}|^\s*相关(研究|报告)|^\s*附(录|表)|'
    r'^.{0,12}(分析员|分析师|研究员|联系人|研究助理)|——.{0,6}(简评|点评|深度|报告)|'
    r'^.{0,20}(市价|收盘价|总市值|流通市值|总股本|流通股本|每股净资产|市净率|资产负债率|一年最低|52\s*周|最新收盘)|'
    r'执业证书|执业编号|S\d{13}|[\w.\-]+[@＠][\w\-]+\.[\w.]+|电话[:：]|邮箱[:：]|\d{3,4}-\d{7,8}|'
    r'^\s*(图|表|图表|附图|附表|Figure|Exhibit|Table)\s*([\d一二三四五六七八九十]+|$)|'
    r'^\s*(资料来源|数据来源|来源|注|备注|Source)\s*[:：]', re.I)
# 段首的项目符号或编号：上一段已结束时据此另起一段
BULLET_RE = re.compile(
    r'^\s*([\u2022\u25a0-\u25ff\u2600-\u27bf\uf020-\uf0ff√✓]|[\-–—*]\s|[(（]?\d{1,2}[)）]|\d{1,2}[、.．](?!\d)|[(（][一二三四五六七八九十]+[)）]|'
    r'[一二三四五六七八九十]+、|第[一二三四五六七八九十]+[，,、])')
RATING_RE = re.compile(r'买入|增持|推荐|中性|减持|跑赢|优于大市|评级|^——')  # 首页的评级、副标题不作小标题
HEADING_NUM_RE = re.compile(r'^\s*([一二三四五六七八九十]+、|\d{1,2}(\.\d{1,2})*[、.．\s](?!\d)|[(（][一二三四五六七八九十\d]+[)）])')

MARGIN_TOP, MARGIN_BOTTOM = 0.045, 0.955  # 页面高度的比例，之外是页眉页脚
REPEAT_PAGE_SHARE = 0.5  # 去掉数字后在这么多比例（且至少 2 页）的页面上重复出现的行视为页眉页脚
BODY_SIZE_SHARE = 0.15  # 某字号的正文样式字数占最多那个字号的比例达到该值，才算正文字号
SIZE_TOL = 0.35  # 与正文字号相差在此之内的行算正文
MIN_SIZE = 8.0  # 小于该字号的一律不算正文（图内文字、表格）
HEADING_MAX_LEN = 30
MIN_PARA_CHARS, MAX_PARA_CHARS = 60, 1500
MIN_CJK_SHARE = 0.3  # 汉字占比低于该值的段落视为表格残留
MIN_PUNCT_DENSITY = 1.5  # 每 100 字的句读数低于该值的段落视为表格、图例残留
BOLD_COMMON_SHARE = 0.3  # 某正文字号中粗体字数超过该比例时，粗体不再算小标题的依据


def parse_filename(stem):
    """从文件名“日期-券商-公司（代码）…：标题”中取出元数据，格式不符时只保留标题。"""
    m = FILE_RE.match(stem)
    if not m:
        return {'date': '', 'broker': '', 'company': '', 'stock_code': '', 'title': stem}
    d = m.groupdict()
    return {'date': d['date'], 'broker': d['broker'].strip(), 'company': d['company'].strip(),
            'stock_code': d['code'].upper(), 'title': d['title'].strip()}


def cjk_share(text):
    t = re.sub(r'\s', '', text)
    return len(CJK_RE.findall(t)) / len(t) if t else 0.0


def page_lines(page, page_no):
    """页面上所有横排文字行，带坐标、主要字号（按字数加权）、是否粗体和所在文字块编号。"""
    lines = []
    for b in page.get_text('dict')['blocks']:
        if b['type'] != 0:
            continue
        for ln in b['lines']:
            if abs(ln['dir'][1]) > 0.1:  # 竖排或旋转的文字：坐标轴标签、水印
                continue
            spans = [s for s in ln['spans'] if s['text'].strip()]
            if not spans:
                continue
            sizes, bold = Counter(), 0
            for s in spans:
                n = len(s['text'].strip())
                sizes[round(s['size'], 1)] += n
                if s['flags'] & 16 or re.search(r'bold|heavy|black|hei|黑体', s['font'], re.I):
                    bold += n
            n_chars = sum(sizes.values())
            text = ''.join(s['text'] for s in ln['spans']).strip()
            x0, y0, x1, y1 = ln['bbox']
            lines.append({'page': page_no, 'block': (page_no, b['number']), 'text': text,
                          'x0': x0, 'y0': y0, 'x1': x1, 'y1': y1,
                          'size': sizes.most_common(1)[0][0], 'bold': bold > n_chars / 2})
    return merge_fragments(lines)


def merge_fragments(lines):
    """两端对齐时同一行有时被拆成左右几段（如“…费用率为”和“3.5%/2.3% ，同比”），按基线和间距拼回一行。"""
    out = []
    for ln in lines:
        prev = out[-1] if out else None
        if (prev and abs(ln['y0'] - prev['y0']) < (prev['y1'] - prev['y0']) * 0.3
                and 0 <= ln['x0'] - prev['x1'] < ln['size'] * 2 and abs(ln['size'] - prev['size']) <= SIZE_TOL):
            prev.update(text=prev['text'] + ln['text'], x1=ln['x1'], y1=max(prev['y1'], ln['y1']))
        else:
            out.append(ln)
    return out


def is_prose(text):
    return len(text) >= 10 and cjk_share(text) >= 0.5 and any(p in text for p in PROSE_PUNCT)


def punct_density(text):
    """每 100 字中的中文句读数；正文一般在 5 以上，表格、图例拼出来的文字接近 0。"""
    return sum(text.count(p) for p in PROSE_PUNCT + '！？') * 100 / max(len(text), 1)


def find_disclaimer_start(pages):
    """返回 (页序号, 行序号)：末尾声明部分的第一行；没有找到时返回 None。只在后半篇查找。"""
    first = max(1, len(pages) // 2)
    for pi in range(first, len(pages)):
        for li, ln in enumerate(pages[pi]):
            t = re.sub(r'\s', '', ln['text'])
            m = DISCLAIMER_HEAD_RE.match(t)
            if m and not DISCLAIMER_HINT_RE.search(t) and len(t) <= (40 if '评级' in m.group(0) else DISCLAIMER_HEAD_MAX_LEN):
                return pi, li
    return None


def strip_repeated(pages):
    """去掉多页重复出现的行（页眉、页脚、页码），按去掉数字和空白后的文字比较。"""
    if len(pages) < 2:
        return pages
    key = lambda t: re.sub(r'[\d\s/]', '', t)
    seen = Counter(k for lines in pages for k in {key(ln['text']) for ln in lines})
    limit = max(2, len(pages) * REPEAT_PAGE_SHARE)
    return [[ln for ln in lines if seen[key(ln['text'])] < limit] for lines in pages]


def body_style(lines):
    """正文字号，以及其中粗体很常见的字号（这些字号上粗体不能作为小标题的依据，如中金正文整体用黑体）。
    正文样式的行按字数加权，字数达到最多者 BODY_SIZE_SHARE 的字号都算（同一篇首页和正文页字号可能不同）。"""
    w, w_bold = Counter(), Counter()
    for ln in lines:
        if ln['size'] >= MIN_SIZE and is_prose(ln['text']):
            w[ln['size']] += len(ln['text'])
            w_bold[ln['size']] += len(ln['text']) * ln['bold']
    if not w:
        return [], set()
    top = max(w.values())
    sizes = sorted(s for s, n in w.items() if n >= top * BODY_SIZE_SHARE)
    return sizes, {s for s in sizes if w_bold[s] > w[s] * BOLD_COMMON_SHARE}


def column_widths(lines, sizes):
    """“写满一行”的宽度：同一页、同一左边界（10pt 一档）的正文行宽取较宽的一半的中位数；
    这一档行数太少时（单独一行的小标题）改用整页正文行的同一统计量，键为 (页码, None)。"""
    widths = defaultdict(list)
    for ln in lines:
        if any(abs(ln['size'] - s) <= SIZE_TOL for s in sizes):
            widths[(ln['page'], round(ln['x0'] / 10))].append(ln['x1'] - ln['x0'])
            widths[(ln['page'], None)].append(ln['x1'] - ln['x0'])
    return {k: statistics.median(sorted(v)[len(v) // 2:]) for k, v in widths.items() if k[1] is None or len(v) >= 3}


def is_full(ln, full_w):
    w = full_w.get((ln['page'], round(ln['x0'] / 10))) or full_w.get((ln['page'], None), 0)
    return ln['x1'] - ln['x0'] >= w * 0.85


def classify(ln, sizes, bold_common, full_w, company=''):
    """返回 'body'、'heading'（小标题，作为后续段落的 section）、'label'（只作分段点，文字丢弃）或 None（丢弃）。"""
    t = ln['text']
    if not CJK_RE.search(t):
        # 没有汉字的行：只保留像英文句子的长行（正文中的英文名称换行），表格单元格、坐标轴刻度丢弃
        return 'body' if len(t) >= 20 and re.search(r'[A-Za-z]{2,}\s+[A-Za-z]{2,}', t) else None
    near = [s for s in sizes if abs(ln['size'] - s) <= SIZE_TOL]
    plain = re.sub(r'\s', '', t)
    if plain == company or RATING_RE.search(plain) and len(plain) <= 15:
        return 'label' if near or ln['size'] > max(sizes) else None
    # 明显大于正文的字号：章节标题，可以较长、带逗号（跨两行的标题在 assemble 中拼接）
    if (max(sizes) + 1.5 <= ln['size'] < max(sizes) + 8 and len(plain) <= HEADING_MAX_LEN * 2
            and plain[-1] not in '。；' and len(CJK_RE.findall(plain)) >= 2):
        return 'heading'
    if (len(plain) <= HEADING_MAX_LEN and plain[-1] not in SENT_END + '，,、：:' and len(CJK_RE.findall(plain)) >= 2
            and not is_full(ln, full_w)):
        if max(sizes) + SIZE_TOL < ln['size'] < max(sizes) + 8:
            return 'heading'
        if near and (ln['bold'] and near[0] not in bold_common or HEADING_NUM_RE.match(t) and not is_prose(t)):
            return 'heading'
        # 正文字号的独立短行：没写满、句中没有逗号句号，可能是小标题，也可能是侧栏栏目名（如“基本数据”）或日期，只作分段点；
        # 带编号的短行（如“2）行业竞争加剧”）是列表项的标题，留在正文里
        if near and not any(c in plain for c in '，。；,;') and not BULLET_RE.match(t):
            return 'label'
    return 'body' if near else None


def assemble(lines, full_w):
    """把逐行文字拼成段落：返回 [{'page','section','text'}]。"""
    paras, cur, section, prev, prev_head = [], [], '', None, None

    def flush():
        if cur:
            paras.append({'page': cur[0]['page'], 'section': section, 'text': join_lines(cur)})
            cur.clear()

    for ln in lines:
        if ln['kind'] == 'label' and prev is not None and continues(prev, ln):
            ln['kind'] = 'body'  # 句子还没结束：这是正文的最后一行（如图表前的半行），不是栏目名
        if ln['kind'] in ('heading', 'label'):
            flush()
            if ln['kind'] == 'heading':
                text = re.sub(r'\s+', ' ', ln['text']).strip()
                # 紧接在上一行标题下方的标题行是同一个标题的续行
                joined = prev_head and prev_head['page'] == ln['page'] and 0 <= ln['y0'] - prev_head['y1'] < (ln['y1'] - ln['y0'])
                section = section + text if joined else text
                prev_head = ln
            prev = None
            continue
        prev_head = None
        if prev is not None and (ln['page'] - prev['page'] > 1 or len(cur) > 1 and punct_density(join_lines(cur)) < MIN_PUNCT_DENSITY):
            flush()  # 隔页的、或接在表格残留后面的行，不当作句子的延续
        elif prev is not None and prev['text'].rstrip()[-1:] in SENT_END + '”’）)':  # 上一行以句末标点结尾才可能分段
            line_h = max(prev['y1'] - prev['y0'], 1)
            same_page = ln['page'] == prev['page']
            if (BULLET_RE.match(ln['text']) or not is_full(prev, full_w) or ln['block'] != prev['block']
                    or same_page and (ln['x0'] - prev['x0'] > line_h * 0.8 or ln['y0'] - prev['y1'] > line_h * 1.2)):
                flush()
        cur.append(ln)
        prev = ln
    flush()
    return paras


def continues(prev, ln):
    """ln 是否紧接在 prev 下方、同一栏，且 prev 没有以句末标点结尾。"""
    line_h = max(prev['y1'] - prev['y0'], 1)
    return (prev['text'].rstrip()[-1:] not in SENT_END and ln['page'] == prev['page']
            and abs(ln['x0'] - prev['x0']) < line_h * 2 and 0 <= ln['y0'] - prev['y1'] < line_h * 1.2)


def join_lines(lines):
    """中文行之间直接相连，英文/数字之间补一个空格。"""
    out = ''
    for ln in lines:
        t = ln['text'].strip()
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


def tidy(paras):
    """丢掉表格残留，短段并入同一小节的相邻段，长段按句切开。"""
    paras = [p for p in paras if len(CJK_RE.findall(p['text'])) >= 10 and cjk_share(p['text']) >= MIN_CJK_SHARE
             and punct_density(p['text']) >= MIN_PUNCT_DENSITY]
    merged = []
    for p in paras:
        if merged and merged[-1]['section'] == p['section'] and (
                len(p['text']) < MIN_PARA_CHARS or len(merged[-1]['text']) < MIN_PARA_CHARS):
            merged[-1]['text'] += p['text']
        else:
            merged.append(dict(p))
    return [{**p, 'text': t} for p in merged for t in split_long(p['text'])]


def extract_pdf(path, company=''):
    """一篇 PDF -> 段落列表 [{'page','section','text'}]，以及诊断信息。company 为公司简称，用来去掉首页的公司名标题。"""
    doc = pymupdf.open(path)
    pages = [page_lines(page, i + 1) for i, page in enumerate(doc)]
    cut = find_disclaimer_start(pages)
    if cut:
        pi, li = cut
        pages = pages[:pi] + [pages[pi][:li]]
    pages = strip_repeated(pages)
    heights = [page.rect.height for page in doc]
    lines = [ln for page in pages for ln in page
             if heights[ln['page'] - 1] * MARGIN_TOP <= ln['y1'] and ln['y0'] <= heights[ln['page'] - 1] * MARGIN_BOTTOM
             and not DROP_LINE_RE.search(ln['text'])]
    sizes, bold_common = body_style(lines)
    info = {'pages': len(doc), 'disclaimer_page': cut and cut[0] + 1, 'body_sizes': sizes}
    if not sizes:
        return [], info
    full_w = column_widths(lines, sizes)
    for ln in lines:
        ln['kind'] = classify(ln, sizes, bold_common, full_w, company)
    return tidy(assemble([ln for ln in lines if ln['kind']], full_w)), info


def main(pdf_dir, out_path, limit=None):
    """pdf_dir 下的所有 PDF -> out_path（parquet，每行一个段落），同时在旁边写一份 .txt 便于人工检查抽取效果。"""
    files = sorted(Path(pdf_dir).rglob('*.pdf'))[:limit]
    rows, used_ids, preview = [], set(), []
    for f in files:
        meta = parse_filename(f.stem)
        rid = '_'.join(x for x in (meta['date'], meta['broker'], meta['stock_code']) if x) or f.stem
        base, k = rid, 2
        while rid in used_ids:  # 同一天同一券商对同一公司发了多篇
            rid, k = f'{base}_{k}', k + 1
        used_ids.add(rid)
        paras, info = extract_pdf(f, meta['company'])
        print(f'{f.name[:40]}：{info["pages"]} 页，声明从第 {info["disclaimer_page"]} 页起，'
              f'正文字号 {info["body_sizes"]}，{len(paras)} 段 {sum(len(p["text"]) for p in paras)} 字')
        preview.append(f'########## {rid} {meta["title"]}')
        for i, p in enumerate(paras, 1):
            rows.append({'report_id': rid, 'para_id': i, 'file': f.name, **meta, 'page': p['page'],
                         'section': p['section'], 'text': p['text']})
            preview.append(f'[{i}] p{p["page"]} 〔{p["section"]}〕 {p["text"]}')
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), out_path)
    out_path.with_suffix('.txt').write_text('\n'.join(preview) + '\n', encoding='utf-8')
    print(f'共 {len(files)} 篇研报、{len(rows)} 个段落，已写入 {out_path}（预览：{out_path.with_suffix(".txt").name}）')
