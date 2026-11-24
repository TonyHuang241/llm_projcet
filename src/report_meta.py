"""研报元数据：从纯文本和文件路径里取出发布日期、券商、报告类型、公司和股票代码、分析师，并判断是不是个股公司研报。

clear_txt.py（src/report_text.py）清理正文时顺带调用，每篇研报一行写成元数据表（*_reports.parquet 和同名 .csv），
和段落表用 report_id 对应。逻辑链抽取（extract_logic_chain.py）按这张表筛选要抽取的研报，默认只抽个股公司研报。

各列的取法（取不到的留空，不硬猜）：
  date          发布日期，取文件所在目录的月/日（抽样核对：87% 的研报开头写着同一天，没有一篇的落款日期晚于目录日期，
                少数比落款晚 1-3 天）。
  broker        券商。先在“资料来源：XX证券研究所”一类的行里计数，取出现最多且没有并列的（前面粘着别的字的
                “预测均为东吴证券”计入“东吴证券”；只写“德邦研究所”的认作德邦证券）；没有结果时再在全文里找
                “XX证券股份有限公司 / 研究所”这样的全称。
  report_type   报告类型，看开头 REPORT_TYPE_LINES 行（不算相关研究的条目）里最先出现的类型字样：公司（公司研究、公司点评、
                年报点评、首次覆盖……）、行业（行业周报、行业研究……）、策略、宏观、固收、金工、晨报、基金；都没有的记“未知”。
                只认“公司点评”“行业周报”这样的完整说法：个股研报的侧栏里也常有“行业”“个股表现”等字样，单个词不算数。
  company、stock_code   这篇研报讲的那一只股票，认 A 股、北交所和港股代码，依次尝试：
                1. 开头 META_HEAD_LINES 行的短行（标题、侧栏）里只出现一个“公司名（代码）”，取它；
                2. 报告类型是“公司”时，取全文里出现次数明显占多数的那个“公司名（代码）”；
                3. 报告类型是“公司”但全文没有代码的（有些券商的纯文本里只有公司名）：全部研报处理完后，用其他研报里
                   认出的“公司名 → 代码”去对，取正文里出现次数明显占多数的公司名（研报自己券商的名字不算；fill_stocks，
                   由 report_text.main 调用）。
                开头的行里数字多、汉字少的（is_table_row）当作表格行，不算标题、侧栏：行业研报首页的可比公司估值表拆成多行后
                常有一行只剩一只股票。侧栏里同名的 A 股、港股各列一行时（两地上市），取 A 股。
                A 股、北交所的交易所后缀按代码开头定（60/68 是 SH，00/30 是 SZ，43/83/87/92 是 BJ），文本里写错的后缀
                （“燕京啤酒(000729.SH)”）不用；名称是指数的（“上证综指(000001.SH)”）、只有代码且后缀和代码开头对不上的不算股票；
                公司名前面粘着的评级字样（“增持良信股份”）去掉。
  is_stock_report   是不是个股公司研报：报告类型是“公司”，或者类型未知但开头的标题位置只有一只股票。
                行业周报、晨报里即使开头只提到一只股票也不算。个别个股研报认不出是哪只股票，这时 stock_code 为空。
  analysts      分析师姓名（含研究助理、联系人），用“、”连接。在“分析师：张三”一类的署名和执业证书编号附近找，
                姓名要以常见姓氏开头；只有编号没有姓名的，全部研报处理完后用其他研报里“编号 ↔ 姓名”的对应补上
                （fill_analysts）。很多研报的署名栏在转成纯文本时已经丢了，这一列只有三成左右的研报能取到。
  analyst_ids   执业证书编号（S 加 13 位数字），用“、”连接。
"""
import re
from collections import Counter

CJK = '一-鿿'
META_HEAD_LINES = 40  # 公司名和代码先在开头这么多行里找
REPORT_TYPE_LINES = 15  # 报告类型只看开头这么多行
DOMINANT_SHARE = 0.6  # 开头取不到唯一的股票时，全文里某只股票占“公司名（代码）”出现次数的比例达到该值才算这篇的股票

# ==================== 报告类型 ====================

CJK_SPACE_RE = re.compile(rf'(?<=[{CJK}])\s+(?=[{CJK}])')  # “行业 研 究”：汉字之间的空格去掉再判断
COMPANY_TYPE_RE = re.compile(
    r'(?<![本贵该我])公司(研究|点评|简评|深度|报告|动态|快报|跟踪|首次|季报|年报|半年报|中报|调研|评论|简报|定期|事件点评|公告点评|更新报告)|'
    r'个股(研究|报告|点评|深度)|(年报|季报|中报|半年报|业绩|业绩预告|业绩快报|财报|公告)(点评|简评)|首次覆盖|新股(研究|报告)|调整盈利预测')
# 个股研报首页的行情侧栏（行业报告开头不会有单只股票的行情；“行业总市值”和推荐股票表格的表头不算）
STOCK_SIDEBAR_RE = re.compile(r'(收盘|当前|最新|现)价|当前股价|(?<!行业)(?<!板块)总市值|总股本|目标(价|估值)[^\n]{0,12}\d')
NOT_SIDEBAR_RE = re.compile(r'股票名称|股票代码|证券代码|公司简称|重点公司|重点推荐|行业总市值|股票家数')
TYPE_LINE_MAX = 60  # 类型字样只在不超过这么多字的行里找（正文句子里的“公司公告”“市场策略”不算）
OTHER_TYPES = [  # (类型, 字样)，靠前的优先
    ('晨报', re.compile(r'晨报|晨会|早报|早参|晨间|(投资|市场|策略|研究)日报|每日(观点|策略|点评|视点|资讯)|午间')),
    ('金工', re.compile(r'金融工程|金工(研究|专题|周报|点评|报告)|量化(研究|专题|周报|策略|选股)')),
    ('固收', re.compile(r'固定收益|固收|债券(研究|周报|月报|点评|专题|策略|市场)|转债(研究|周报|月报|点评|专题|策略|市场)')),
    ('基金', re.compile(r'基金(研究|报告|周报|月报|点评|专题)')),
    ('宏观', re.compile(r'宏观(研究|经济|周报|月报|点评|专题|策略|报告|深度|数据|快评|简评|观察|分析)')),
    ('策略', re.compile(r'策略(研究|周报|月报|点评|专题|报告|深度|快评|简评|观察|日报)|投资策略|(年度|中期|季度|月度)策略|金股|'
                      r'市场(周报|月报|点评|观察|策略)|(A股|港股|大势)(策略|研判|周报)')),
    ('行业', re.compile(r'行业(周报|月报|日报|点评|深度|专题|研究|报告|跟踪|动态|快报|快评|策略|观察|简评|分析|数据|事件|简报|定期|投资)|'
                      r'(周|月|双周)(报|观点|观察|跟踪|度报告|度观点|度跟踪)|板块(周报|月报|点评|跟踪)|产业(研究|跟踪|周报|专题)')),
]


def report_type(lines):
    """开头若干行里最先出现的类型字样；同一行里公司和别的类型都有时算公司。没有“公司点评”这类说法、只有行情侧栏
    （总市值、收盘价等）的，侧栏出现在别的类型字样之前才算公司。"""
    for line in lines[:REPORT_TYPE_LINES]:
        if RELATED_RE.search(line):
            continue
        line = CJK_SPACE_RE.sub('', line)
        if STOCK_SIDEBAR_RE.search(line) and not NOT_SIDEBAR_RE.search(line):
            return '公司'
        if len(line) > TYPE_LINE_MAX:
            continue
        if COMPANY_TYPE_RE.search(line):
            return '公司'
        other = next((name for name, pat in OTHER_TYPES if pat.search(line)), '')
        if other:
            return other
    return '未知'


# 深度报告：首页标注“公司深度（研究）”“深度报告”“深度研究系列”“XX深度：”“系列深度二”之类的字样（正文里的“深度绑定”
# “深度合作”不算），以及首次覆盖报告（“公司首次覆盖报告”“评级：买入（首次覆盖）”，篇幅和深度报告相当）、“公司专题”
DEEP_RE = re.compile(r'(公司|个股)深度|深度(报告|研究|分析|拆解|复盘|系列|专题)|深度[:：丨|]|深度[一二三四五六七八九十\d]|'
                     r'首次覆盖|(公司|个股)专题')
NOT_DEEP_RE = re.compile(r'(上次|前次)评级')  # “上次评级：首次覆盖”说的是上一篇
DEEP_HEAD_LINES = 20  # 只看开头这么多行
NUMBERED_RE = re.compile(r'^\s*\d{1,2}\s*[.、．]')  # “1.徐工机械深度报告：……-2024/06/02”：带编号的是相关研究列表里的条目


def deep_line(lines):
    """开头 DEEP_HEAD_LINES 行里标注深度报告的那一行，没有时返回空串；相关研究条目、“上次评级”和超过 TYPE_LINE_MAX 字的
    正文句子不算。元数据表里没有这一列（report_type 只区分公司、行业等大类），需要时读原文首页判断。
    只认“深度”字样时全库个股研报里命中 798 篇，清理后正文中位数 1.2 万字，是点评类研报的五六倍；不排除带编号的行时
    会多出 34 篇，几乎都是只有一两千字的点评。加上首次覆盖、深度加编号、公司专题后，个股研报（股票代码、公司名、券商都认出的）
    里命中 1217 篇，其中 273 篇正文不超过 5000 字（多是只有一两页的首次覆盖点评，或正文没提取出来），extract_concept_dag.py
    另按正文字数剔除（MIN_DEEP_CHARS），剩 944 篇、408 家公司。"""
    for line in lines[:DEEP_HEAD_LINES]:
        if RELATED_RE.search(line) or NUMBERED_RE.match(line) or NOT_DEEP_RE.search(line):
            continue
        line = CJK_SPACE_RE.sub('', line)
        if len(line) <= TYPE_LINE_MAX and DEEP_RE.search(line):
            return line
    return ''


# ==================== 公司和股票代码 ====================

A_CODE = r'(?:00|30|60|68)\d{4}'
BJ_CODE = r'(?:43|83|87|92)\d{4}'
NAME = rf'[{CJK}A-Za-z*]{{2,12}}'
A_PART = rf'({A_CODE}|{BJ_CODE})(?:\s*[.．\s]\s*(SH|SZ|BJ|sh|sz|bj|SS|CH|CN))?'
HK_PART = r'(\d{3,5})\s*[.．\s]\s*(HK|hk)'
# “贵州茅台(600519.SH)”“锦江酒店（600754）”“牧原股份 (002714 CH)”“中信证券（600030.SH/6030.HK）”“腾讯控股（0700.HK）”
PAREN_CODE_RE = re.compile(rf'({NAME})\s*[（(]\s*(?:{A_PART}|{HK_PART})\s*[）)/／,，、;；]')
# “新澳股份 603889.SH”：没有括号时必须带交易所后缀
BARE_CODE_RE = re.compile(rf'({NAME})\s+(?:({A_CODE}|{BJ_CODE})\s*[.．]\s*(SH|SZ|BJ)|(\d{{4,5}})\s*[.．]\s*(HK))(?![\dA-Za-z])')
# 只有代码没有公司名（“调整盈利预测 2024年9月2日 603296.SH”）
CODE_ONLY_RE = re.compile(rf'(?<![\dA-Za-z.])({A_CODE}|{BJ_CODE})\s*[.．]\s*(SH|SZ|BJ)(?![\dA-Za-z])')
SPACED_NAME_RE = re.compile(rf'((?:[{CJK}] ){{1,5}}[{CJK}])(?=\s*[（(]\s*\d)')  # “贝 泰 妮(300957.SZ)”
RELATED_RE = re.compile(r'《|相关(研究|报告|阅读)|可比公司|同行')
DUAL_A_RE = re.compile(rf'^[\s/／、,，]*({A_CODE})')  # “（1339 HK/601319 CH）”：两地上市的取 A 股代码
TITLE_LINE_MAX = 80  # 不超过这么多字的行，或者公司名出现在行首 TITLE_START 个字以内，才算标题、侧栏（正文句子里顺带提到的股票不算）
TITLE_START = 25
MIN_FILL_NAME = 3  # 用公司名对代码时，名称至少这么多字（两个字的简称容易撞上普通词）
# 开头的行里数字至少这么多个、而且不少于汉字数的一半的当作表格行，不算标题、侧栏：行业研报首页的可比公司估值表拆成多行后，
# 常有一行只剩一只股票（“福莱特玻璃 06865.HK HKD 15.94 2023/11/1 买入 20.84 1.28 1.77 ……”），会被当成个股研报；
# 标题句里也常有不少数字（“爱美客（300896.SZ）2023年归母净利润预计增长43%-50%，……”），但汉字多得多
TABLE_ROW_NUMS = 6
TABLE_ROW_WINDOW = 60
NUM_RE = re.compile(r'(?<![\dA-Za-z])\d+(?:[.,/]\d+)*%?|(?<!\S)(?:-+|—+|/|N/?A|NM)(?!\S)')  # 数字和表格里的占位符“-”“/”
RATING_PREFIX_RE = re.compile(rf'^(强烈推荐|买入|增持|推荐|持有|中性|减持|优于大市|跑赢行业)(?=[{CJK}]{{2}})')  # “增持良信股份”
INDEX_NAME_RE = re.compile(r'^(上证|沪深|中证|深证|国证)|指数|综指|成指|创业板指|科创50')  # “上证综指(000001.SH)”不是股票


def exchange(code):
    return 'SH' if code[0] == '6' else 'BJ' if code[0] in '489' else 'SZ'


def a_code(code, suffix=''):
    """标准代码：交易所后缀按代码开头定，文本里写错的后缀（“燕京啤酒(000729.SH)”）不用。"""
    return f'{code}.{exchange(code)}'


def is_table_row(line):
    """只看前 TABLE_ROW_WINDOW 个字：表格最后一行常接着“资料来源：Wind，XX证券研究部预测……”，看整行汉字就多了。"""
    head = line[:TABLE_ROW_WINDOW]
    n = len(NUM_RE.findall(head))
    return n >= TABLE_ROW_NUMS and n * 2 >= len(re.findall(f'[{CJK}]', head))


def stock_mentions(line):
    """一行里的 (公司名, 标准代码, 在行里的位置)；只有代码没有公司名时公司名为空。"""
    line = SPACED_NAME_RE.sub(lambda m: m.group(1).replace(' ', ''), line)
    out = []
    for m in list(PAREN_CODE_RE.finditer(line)) + list(BARE_CODE_RE.finditer(line)):
        name, a, suffix, hk, _ = m.groups()
        if INDEX_NAME_RE.search(name):
            continue
        if hk:
            dual = DUAL_A_RE.match(line[m.end() - 1:])
            code = a_code(dual.group(1)) if dual else f'{int(hk):05d}.HK'
        else:
            code = a_code(a, suffix or '')
        out.append((name, code, m.start()))
    if not out:  # 只有代码时，后缀和代码开头对不上的多半是指数（“000001.SH”是上证综指），不算
        out = [('', a_code(m.group(1)), m.start()) for m in CODE_ONLY_RE.finditer(line) if m.group(2) == exchange(m.group(1))]
    return out


def dual_a(names_by_code):
    """{代码: {公司名}} 正好是一只 A 股、一只港股，而且公司名相关（相同或一个包含另一个，如“光大银行”和“中国光大银行”）时，
    认为是同一家公司两地上市，返回 A 股代码；否则返回 None。"""
    hk = [c for c in names_by_code if c.endswith('.HK')]
    a = [c for c in names_by_code if not c.endswith('.HK')]
    if len(hk) != 1 or len(a) != 1:
        return None
    pairs = [(x, y) for x in names_by_code[hk[0]] for y in names_by_code[a[0]] if x and y]
    return a[0] if any(x in y or y in x for x, y in pairs) else None


def pick_stock(lines, is_company):
    """这篇研报讲的那只股票：(公司名, 代码, 是否出现在标题位置)，取不到时公司名和代码为空。"""
    names, total = {}, Counter()
    for line in lines:
        for name, code, _ in stock_mentions(line):
            names.setdefault(code, Counter())[name] += 1
            total[code] += bool(name)  # 只有代码的行多半是可比公司的表格，不计数
    if not names:
        return '', '', False
    named, bare = Counter(), Counter()  # 开头的标题、侧栏里带公司名的代码、只有代码的
    for line in lines[:META_HEAD_LINES]:
        found = stock_mentions(line)
        codes = {code for _, code, _ in found}
        if len(codes) == 2:  # 两地上市写在同一行（“光大银行(601818.SH)/中国光大银行(06818.HK)”）：算 A 股
            a = dual_a({c: {n for n, c2, _ in found if c2 == c} for c in codes})
            if a:
                codes, found = {a}, [x for x in found if x[1] == a]
        if len(codes) != 1 or RELATED_RE.search(line) or is_table_row(line):  # 相关研究、表格行不算
            continue
        if len(line) <= TITLE_LINE_MAX or found[0][2] <= TITLE_START:
            (named if found[0][0] else bare).update(codes)
    head = named or bare
    if len(head) == 2:  # 两地上市的公司在侧栏里 A 股、港股各列一行（“中国中车 601766 CH”“中国中车 1766 HK”）：取 A 股
        a = dual_a({c: set(names[c]) for c in head})
        if a:
            head = Counter({a: head[a]})
    in_title = len(head) == 1
    if in_title:
        code = next(iter(head))
    elif is_company and total and total.most_common(1)[0][1] >= DOMINANT_SHARE * sum(total.values()):
        code = total.most_common(1)[0][0]
    else:
        return '', '', False
    # 公司名取全文里和这个代码一起出现次数最多的写法；前面粘着别的字（“航空军工中航西飞”）时取反复出现的最长后缀
    text = '\n'.join(lines)
    cand = Counter()
    for k, v in names[code].items():
        k = RATING_PREFIX_RE.sub('', k)
        if k:
            cand[k] += v
    name = min(cand, key=lambda x: (-cand[x], len(x))) if cand else ''
    if len(name) > 6:
        name = next((name[-k:] for k in range(6, 1, -1) if text.count(name[-k:]) >= 3), '')
    return (name if re.search(f'[{CJK}]', name) else ''), code, in_title


def fill_stocks(rows, read):
    """全部研报处理完后补公司和代码（rows 为元数据表的行，原地修改；read(row) 返回研报全文），返回改动过的 report_id：
    1. 公司名统一成其他研报里这个代码最常见的写法（补上只认出代码的，去掉名字前面粘着的字，如“家具欧派家居”）；
    2. 个股研报但全文没有代码的，用“公司名 → 代码”去对正文，取出现次数明显占多数的公司名（研报自己券商的名字不算：
       “广发证券”在广发的研报里到处都是）。"""
    seen = {}
    for r in rows:
        if r['company'] and r['stock_code']:
            seen.setdefault(r['stock_code'], Counter())[r['company']] += 1
    name_of = {code: c.most_common(1)[0][0] for code, c in seen.items()}
    code_of = {}
    for code, c in seen.items():
        for name, n in c.items():
            if len(name) >= MIN_FILL_NAME and n >= 2 and re.fullmatch(f'[{CJK}A-Za-z*]+', name):
                code_of.setdefault(name, code)
    changed = []
    for r in rows:
        if r['stock_code'] and r['stock_code'] in name_of:
            if r['company'] != name_of[r['stock_code']]:
                r['company'] = name_of[r['stock_code']]
                changed.append(r['report_id'])
        elif r['is_stock_report'] and not r['stock_code'] and code_of:
            text = read(r)
            hits = Counter()
            for name, code in code_of.items():
                n = text.count(name) if name != r.get('broker') else 0
                if n:
                    hits[code] += n
            top = hits.most_common(2)
            if top and top[0][1] >= 3 and top[0][1] >= DOMINANT_SHARE * sum(hits.values()):
                r['stock_code'], r['company'] = top[0][0], name_of[top[0][0]]
                changed.append(r['report_id'])
    return changed


# ==================== 券商 ====================

BROKERS = {  # 规范名: 文本里的各种写法（不带“证券”二字或简称和规范名不同的才需要列在这里）
    '中金公司': ('中金公司', '中国国际金融'), '中信证券': ('中信证券',), '中信建投': ('中信建投',), '华泰证券': ('华泰证券', '华泰研究'),
    '国泰君安': ('国泰君安',), '海通证券': ('海通证券',), '申万宏源': ('申万宏源', '申银万国'), '中国银河': ('中国银河', '银河证券'),
    '东方财富证券': ('东方财富证券',), '群益证券': ('群益证券',), '华兴证券': ('华兴证券',), '第一上海': ('第一上海',),
    '德邦证券': ('德邦证券', '德邦研究所'), '太平洋证券': ('太平洋证券', '太平洋研究院'),  # 资料来源里只写“德邦研究所”
}
SOURCE_LINE_RE = re.compile(r'来源|预测|整理|Source', re.I)
BOUNDARY = r'(?:^|(?<=[：:，,、\s;；(（|]))'
BROKER_RE = re.compile(rf'{BOUNDARY}([{CJK}]{{2,6}}?证券)(?=研究|研发|整理|预测|经济|发展|产业|股份|$|[，,；;。\s)）])')
# 全称：“XX证券股份有限公司”“XX证券研究所”“德邦研究所”
FIRM_RE = re.compile(rf'{BOUNDARY}([{CJK}]{{2,6}}?)(证券)?(?:股份有限公司|有限责任公司|有限公司|研究所|研究院|研究发展中心|研究部)')
NOT_BROKER = re.compile(r'本|该|我|公司|上市|集团|控股|银行|保险|基金|期货|信托|交易|中国证券|投资|咨询|资产|科技|实业|股份|产业|经济')


def vote(counter):
    top = counter.most_common(2)
    return top[0][0] if top and (len(top) == 1 or top[0][1] > top[1][1]) else ''


def pick_broker(lines):
    known = lambda line: {canon for canon, aliases in BROKERS.items() if any(a in line for a in aliases)}
    is_alias = lambda m: any(m in a or a in m for al in BROKERS.values() for a in al)
    votes = Counter()
    for line in lines:
        if SOURCE_LINE_RE.search(line):
            votes.update(known(line) | {m for m in BROKER_RE.findall(line) if not is_alias(m)})
    for m in list(votes):  # 前面粘着别的字的（“预测均为东吴证券”）并到以它结尾的那个名字上（“东吴证券”），否则两者并列、取不出券商
        short = [x for x in votes if x != m and m.endswith(x)]
        if short:
            votes[min(short, key=len)] += votes.pop(m)
    broker = vote(votes)
    if broker:
        return broker
    votes = Counter()
    for line in lines:
        votes.update(known(line))
        for name, sec in FIRM_RE.findall(line):
            if not NOT_BROKER.search(name) and (sec or name.endswith('证券')):
                full = name if name.endswith('证券') else name + '证券'
                if not is_alias(full):
                    votes[full] += 1
    return vote(votes)


# ==================== 分析师 ====================

SNO_RE = re.compile(r'S\s?(\d{13})(?!\d)')  # 执业证书编号
SURNAMES = set(
    '王李张刘陈杨黄赵吴周徐孙马朱胡郭何林罗郑梁谢宋唐许韩冯邓曹彭曾肖田董袁潘于蒋蔡余杜叶程苏魏吕丁任沈姚卢姜崔钟谭陆汪范'
    '石廖贾夏韦付方白邹孟熊秦邱江尹薛闫段雷侯龙史陶黎贺顾毛郝龚邵万钱严覃武戴莫孔向汤常温康施文牛樊葛邢安齐易乔伍庞颜倪庄'
    '聂章鲁岳翟殷詹申欧耿关兰焦俞左柳甘祝包宁尚符舒阮柯纪梅童凌毕单季裴霍涂成苗谷盛曲翁冉骆蓝路游辛靳管柴蒙鲍华喻祁蒲房滕'
    '屈饶解牟艾尤阳时穆农司卓古吉缪简车项连芦麦褚娄窦戚岑景党宫费卜冷晏席卫米柏宗瞿桂全佟应臧闵苟邬边卞姬师和仇栾隋商刁沙'
    '荣巫寇桑郎甄丛仲虞敖巩明佘池查麻苑迟邝官封谈匡鞠惠荆乐冀郁胥南班储原栗燕楚鄢劳谌奚皮粟冼蔺楼盘满闻位厉伊仝区郜海阚花'
    '权强帅屠豆朴盖练廉禹井祖漆巴丰支卿国狄平计索宣晋相初门云容敬来扈晁芮都普阙浦戈伏鹿薄邸雍辜羊阿乌母裘亓修邰赫杭况那宿'
    '鲜印逯隆茹诸战慕危玉银亢嵇公哈湛宾戎勾茅利於呼居揭干但尉冶斯元束檀衣信展阴昝智幸奉植衡富尧闭由'
    '高金洪赖傅萧贝钮阎山郦贡蔚莘代')
COMPOUND_SURNAMES = ('欧阳', '上官', '司马', '诸葛', '皇甫', '尉迟', '公孙', '慕容', '令狐', '司徒', '夏侯', '端木', '东方', '独孤',
                     '南宫', '万俟', '闻人', '澹台', '宇文', '长孙', '轩辕')
# 署名栏里姓名后面常粘着的字样：职务、行业、联系方式
AFTER_NAME = re.compile(
    r'策略|宏观|固收|固定|金工|金融|首席|联席|资深|高级|分析师|研究员|研究|助理|联系|行业|团队|负责|执业|证书|登记|资格|编号|邮箱|电话|邮件|'
    r'电子|汽车|医药|化工|石化|能源|军工|计算机|传媒|通信|银行|非银|地产|房地产|食品|饮料|食饮|机械|电新|电力|有色|建筑|建材|交运|公用|环保|'
    r'家电|轻工|纺服|纺织|商贸|社服|农业|农林|钢铁|煤炭|海外|中小盘|新能源|设备|零售|消费|科技|制造|互联网|TMT|基础|大宗|新材料|PhD|CFA|CPA')
# 以姓氏开头但不是姓名的词
NOT_NAME = set(
    '周报 月报 季报 年报 日报 石化 石油 金融 金工 金属 计算机 农业 农林 房地产 商贸 商业 高级 高端 华泰 华创 华安 华西 华福 华鑫 华金 华龙 '
    '国泰 国金 国信 国盛 国海 国联 国元 国投 国都 海通 方正 兴业 安信 东吴 东方 东北 东兴 东海 东莞 申万 长江 长城 中信 中金 中泰 中邮 中航 '
    '中原 中银 平安 万联 万和 光大 广发 招商 民生 浙商 天风 开源 西南 西部 信达 德邦 山西 财通 太平洋 首创 甬兴 湘财 渤海 上海 银河 '
    '于大市 强于 强烈 强于大市 相对 相关 成长 成本 时间 时期 平均 全球 全国 全年 全部 向上 向下 元件 原因 原料 原油 明显 明确 利润 利率 利好 '
    '盈利 应用 应收 方面 方向 方法 方案 支持 支撑 经营 经济 季度 年度 包括 关注 关键 关于 管理 路径 连续 连接 项目 时点 白酒 白电 '
    '黄金 汽车 文化 易方达 水泥 水电 温氏 施工 常规 安全 安装 齐全 景气 游戏 权益 智能 展望 修复 信息 信用 相比 初步 来源 来自 '
    '和谐 都是 公用 公司 公告 海外 海运 银行 国防 国际 国内 国企 平台 元器件 计划 计算 普通 富时 盘中 盘面 盖板 干散 阿里 支付 丰富 '
    '容量 云计算 门店 玉米 银价 阴极 阳极 阳光 宿主 修订 利用 幸福 居民 居住 斯达 元素 居前 由于 但是 单位 单季 单月 单价 成交 成立 '
    '包装 费用 费率 简称 简介 简评 连锁 项目 明年 名称 钢铁 银保 幅度 管线 路线 风险 祝贺 解读 解决 时代 古井 景区 党建 '
    '卫星 米线 应对 边际 沙特 荣耀 桑叶 满足 位于 位居 位列 区间 区域 花生 帅康 平衡 宣布 晋升 相应 门槛 云南 都市 普及 '
    '高位 高速 高于 高点 高增 高频 高度 高效 高质 高景 高分 高股 金额 金价 金股 金融 洪水 山东 山寨 贝塔 贝泰 代表 代码 代理 代工 '
    '傅里 萧条 赖于 阎良'.split())
LABEL_RE = re.compile(rf'(?:证券分析师|首席分析师|分析师|研究员|研究助理|联系人|作者|执笔人?)\s*(?:[:：]|\s)\s*([{CJK}]{{2,10}})')
SNO_LABEL_TAIL = re.compile(r'(?:SAC|S\.A\.C|执业|证书|编号|登记|资格|证号|执证号|中央|No\.?|NO\.?|号|[:：()（）\s|])+$')
CJK_TAIL = re.compile(rf'[{CJK}]+$')
CJK_HEAD = re.compile(rf'^[\s:：|]*([{CJK}]+)')
NAME_LINE = re.compile(rf'^(?:[{CJK}]{{2,3}}[\s、，,]+){{0,5}}[{CJK}]{{2,3}}$')
TITLE_LINE = re.compile(r'首席|分析师|研究员|助理|行业|联系人')


def is_name(s):
    if s in NOT_NAME or (len(s) == 3 and s[:2] in NOT_NAME):  # “国泰君”“石化行”：前两个字是券商、行业等词
        return False
    if len(s) == 4:
        return s[:2] in COMPOUND_SURNAMES
    return len(s) in (2, 3) and (s[0] in SURNAMES or s[:2] in COMPOUND_SURNAMES)


def lead_name(run):
    """一串汉字开头的姓名：后面要么没有别的字，要么紧跟职务、行业一类的字样（“樊继拓策略首席分析师”）。"""
    for k in (3, 2, 4):
        cand, rest = run[:k], run[k:]
        if len(cand) == k and is_name(cand) and (not rest or AFTER_NAME.match(rest)):
            return cand
    return ''


def tail_name(run):
    """一串汉字结尾的姓名：前面要么没有别的字，要么是“分析师”“研究员”一类的标签。"""
    for k in (3, 2, 4):
        cand, rest = run[-k:], run[:-k]
        if len(run) >= k and is_name(cand) and (not rest or re.search(r'(分析师|研究员|助理|联系人|首席|作者)$', rest)):
            return cand
    return ''


def pick_analysts(lines):
    """(姓名列表, 执业证书编号列表, {编号: 紧挨着它的姓名})，前两个按出现顺序去重。"""
    names, ids, pairs = [], [], {}
    for i, line in enumerate(lines):
        for m in LABEL_RE.finditer(line):  # “分析师：张三”“研究员 张三”
            names.append(lead_name(m.group(1)))
        found = list(SNO_RE.finditer(line))
        if not found:
            continue
        line_ids = ['S' + m.group(1) for m in found]
        ids += line_ids
        got = []
        for m in found:  # 编号前后紧挨着的姓名：“汪刘胜 S1090511040037”“SAC 编号:S1440522030002 何昱灵”
            before = SNO_LABEL_TAIL.sub('', line[:m.start()])
            run = CJK_TAIL.search(before)
            name = tail_name(run.group(0)) if run else ''
            if not name:
                after = CJK_HEAD.match(line[m.end():])
                name = lead_name(after.group(1)) if after else ''
            got.append(name)
        if not any(got) and i:  # 姓名在上一行（一行可以有几个姓名），或者隔着一行职务（“李想 / 公用环保行业首席 / S10105…”）
            prev = lines[i - 1]
            if NAME_LINE.match(prev) and all(is_name(x) for x in re.split(r'[\s、，,]+', prev)):
                got = re.split(r'[\s、，,]+', prev)
            elif i >= 2 and TITLE_LINE.search(prev) and len(prev) <= 20 and is_name(lines[i - 2]):
                got = [lines[i - 2]]
            else:
                run = CJK_TAIL.search(prev)
                got = [tail_name(run.group(0))] if run and re.search(r'(分析师|研究员|助理|联系人|作者)\s*[:：]?\s*' + re.escape(run.group(0)) + '$', prev) else []
        for sid, name in zip(line_ids, got):  # 姓名和编号按顺序一一对应
            if name:
                pairs.setdefault(sid, name)
        names += got
    return list(dict.fromkeys(n for n in names if n)), list(dict.fromkeys(ids)), pairs


def fill_analysts(rows, pairs):
    """全部研报处理完后补分析师姓名（rows 为元数据表的行，原地修改；pairs 为 {report_id: {编号: 姓名}}）：
    有编号没有姓名的，用其他研报里这个编号最常对应的姓名补上。返回补过的篇数。"""
    seen = {}
    for d in pairs.values():
        for sid, name in d.items():
            seen.setdefault(sid, Counter())[name] += 1
    name_of = {sid: c.most_common(1)[0][0] for sid, c in seen.items()}
    n = 0
    for r in rows:
        if not r['analyst_ids']:
            continue
        have = r['analysts'].split('、') if r['analysts'] else []
        own = pairs.get(r['report_id'], {})
        extra = [name_of[sid] for sid in r['analyst_ids'].split('、') if sid not in own and sid in name_of]
        names = list(dict.fromkeys(have + extra))
        if len(names) > len(have):
            r['analysts'] = '、'.join(names)
            n += 1
    return n


# ==================== 汇总 ====================

META_COLUMNS = ('date', 'broker', 'report_type', 'is_stock_report', 'company', 'stock_code', 'analysts', 'analyst_ids')


def parse_meta(path, root, text):
    """一篇研报的元数据（见本模块开头的说明）；file 为相对输入目录的路径。"""
    rel = path.relative_to(root)
    meta = {'file': rel.as_posix(), 'date': '', 'title': ''}
    year = re.search(r'(?:19|20)\d{2}', root.name)
    parts = rel.parts[:-1]
    if year and len(parts) >= 2 and re.fullmatch(r'\d{2}', parts[-2]) and re.fullmatch(r'\d{2}', parts[-1]):
        meta['date'] = year.group(0) + parts[-2] + parts[-1]
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    meta['broker'] = pick_broker(lines)
    meta['report_type'] = report_type(lines)
    meta['company'], meta['stock_code'], in_title = pick_stock(lines, meta['report_type'] == '公司')
    meta['is_stock_report'] = meta['report_type'] == '公司' or (meta['report_type'] == '未知' and in_title)
    names, ids, pairs = pick_analysts(lines)
    meta['analysts'], meta['analyst_ids'] = '、'.join(names), '、'.join(ids)
    meta['analyst_pairs'] = pairs  # 只供 fill_analysts 用，不写进表里
    return meta
