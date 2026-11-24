"""用规则（正则、关键词，不调用大模型）从研报原文识别本次个股评级、评级动作，并按券商自己的评级体系定档位（tier、TOP）。
由根目录 extract_report_rating.py 调用；样本是 output_dag/experiments/report_car/report_car_panel.parquet 的 790 篇。

一、本次评级（按优先级）
  S  结构化写法（页眉/摘要栏）
     S1  评级(动作)：买入（维持）、推荐(首次)、强烈推荐(调升)、买入-A(维持)、增持（首次覆盖）、买入（维持评级）、增持 ( 首次 ）
     S2  评级 分隔符 动作：买入丨维持、买入|首次推荐、买入 / 维持
     S3  评级(动作): 评级：投资评级(维持): 买入、投资评级(首评): 买入
     S4  评级字段：投资评级 增持、评级：持有 首次覆盖、评级 买入 评级变动 维持（只在开头 HEAD_CHARS 字内）
     S5  动作 评级：维持 买入（中信建投页眉；只在开头，用未拼接的原文，两词之间要有空白）
     A+H 页眉同时列两个评级时（华泰：857 HK 601857 CH 投资评级：买入(首评) 增持(首评)），取 A 股对应的第二个
     T   开头的重点公司估值表里本公司那一行的评级（广发：中国巨石 600176.SH CNY 10.55 2024/04/26 买入 ……）
     S6  开头部分单独成行的档位（[Table_Invest] 买入），只给评级
  B  正文句式：B_chg 评级上调/下调（宾语必须是评级）> B_first 首次覆盖……给予 X > B_keep 维持 X 评级 > B_give 给予 X 评级
  L  上次评级字段（上次评级 / 前次评级 / 原评级），用来和本次比较档位，作为上调/下调的第二个来源
  H  开头 1500 字里的“首次覆盖”“维持评级”（不在《》书名号里的），作为动作的弱证据
  排除：在后一半正文里第一次出现评级说明/免责声明关键词处截断；档位后面紧跟“：”“（预计”等定义写法的不算；
       前面紧挨着“行业评级/板块评级/上次评级/原评级”的不算；B 句式里出现“行业”“板块”的不算。
  取舍：本次评级取 S1–S5、T 中位置最靠前的一条 > B（B_chg > B_first > B_keep > B_give，同类取最靠前）> S6；
       动作取 S 自带的动作 > 本次与上次评级的档位比较 > B_chg > B_first/B_keep（取靠前的）> H > 上次评级为“无评级”记首次 > 未说明；
       conflict = 各来源给出的动作不止一种，或 S 与 B 给出的评级在该券商体系里档位不同。
二、档位（tier：1 最高档，2 次高档且仍是正面，3 中性及以下或更低的正面档）
  1. 评级说明解析：在正文里找“档位 + 相对/预期/股价…… + 百分比/指数”的定义块，按“公司评级/股票评级”与“行业评级”标签切开，
     取个股那一段的档位顺序（出现顺序即从高到低）。本篇解析到且包含本次评级 → def_self；
  2. 按券商汇总（用语料库里该券商全部研报的评级说明，取最常见的顺序）→ def_broker；
  3. 都没有时，用该券商全年个股研报里实际用过的档位（用本模块的规则识别，每个档位至少 USE_MIN 篇、占比至少 USE_SHARE）按通用强弱排序 → usage；
     券商年中换过一套互不相交的档位时（国信 2024 年 6 月起 买入/增持 → 优于大市/中性），按月份分段各自定体系；
  4. 还不行用兜底词表 → fallback。另存 tier_fb：只用兜底词表时的档位，便于对照。
  第二档是否“仍是正面”：它不是中性类词（中性/持有/同步大市/观望/跟随大市），或者它后面还有中性类词（西南：买入/持有/中性/回避/卖出）。
"""
import re
from collections import Counter

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- 词表
RATINGS = ['强烈推荐', '强推', '谨慎推荐', '审慎推荐', '推荐', '买入', '审慎增持', '谨慎增持', '增持', '优于大市', '强于大市', '跑赢行业', '跑赢大市',
           '领先大市', '同步大市', '跟随大市', '中性', '持有', '观望', '减持', '卖出', '回避', '弱于大市', '落后大市', '跑输行业', '跑输大市',
           '无评级', '未评级']
STRENGTH = {'强烈推荐': 5, '强推': 5, '买入': 5, '推荐': 4, '增持': 4, '优于大市': 4, '强于大市': 4, '跑赢行业': 4, '跑赢大市': 4, '领先大市': 4,
            '谨慎推荐': 3.5, '审慎推荐': 3.5, '审慎增持': 3.5, '谨慎增持': 3.5,
            '同步大市': 3, '跟随大市': 3, '中性': 3, '持有': 3, '观望': 3,
            '减持': 2, '弱于大市': 2, '落后大市': 2, '跑输行业': 2, '跑输大市': 2, '卖出': 1, '回避': 1}  # 只用于 usage 排序和兜底
NEUTRAL = {'中性', '持有', '同步大市', '观望', '跟随大市'}
SYN = {'强推': '强烈推荐', '强烈推荐': '强推'}
FALLBACK = {'买入': 1, '强烈推荐': 1, '强推': 1, '增持': 2, '推荐': 2, '谨慎推荐': 2, '审慎推荐': 2, '审慎增持': 2, '谨慎增持': 2,
            '优于大市': 2, '强于大市': 2, '跑赢行业': 2, '跑赢大市': 2, '领先大市': 2}  # 其余有评级的记 3
R = '(' + '|'.join(sorted(RATINGS, key=len, reverse=True)) + ')'
GRADE = r'(?:\s*-\s*[A-E])?'  # 买入-A
ACT = r'(首次覆盖|首次评级|首次推荐|首评|首次|维持评级|维持|上调|下调|调高|调低|调升|调降)'
Q, QE = r'[“"「『‘]?', r'[”"」』’]?'
GAP = r'[^。；;，,\n]'  # 同一分句内
HEAD_CHARS, FIRST_CHARS, USE_MIN, USE_SHARE = 8000, 1500, 3, 0.02
DISCLAIMER = ['评级说明', '投资评级说明', '评级标准', '分析师声明', '免责声明', '重要声明', '特别声明', '法律声明', '分析师承诺', '评级定义', '评级体系']
DEF_AFTER = re.compile(r'^\s*(?:[:：]|[（(]\s*(?:预计|预期|相对|未来|股价|指|Buy|BUY|Outperform))')
BAD_BEFORE = ('行业评级', '行业投资评级', '板块评级', '上次评级', '前次评级', '原评级', '上期评级', '行业：', '板块：')
ACT_MAP = {'首次覆盖': '首次', '首次评级': '首次', '首次推荐': '首次', '首评': '首次', '首次': '首次', '维持评级': '维持', '维持': '维持',
           '上调': '上调', '调高': '上调', '调升': '上调', '下调': '下调', '调低': '下调', '调降': '下调'}

S1 = re.compile(R + GRADE + r'\s*[（(]\s*' + ACT + r'\s*(?:评级|覆盖)?\s*[）)]?')
S2 = re.compile(R + GRADE + r'\s*[丨|｜/]\s*' + ACT)
S3 = re.compile(r'评级\s*[（(]\s*' + ACT + r'\s*[）)]\s*[:：]?\s*' + R)
S4 = re.compile(r'评级\s*[:：]?\s*' + R + GRADE + r'(?:\s*(?:评级变动|变动)?\s*[:：]?\s*' + ACT + ')?')
S5 = re.compile(r'(维持|首次覆盖|首次)\s+' + R + r'(?=\s)')
L = re.compile(r'(上次评级|前次评级|原评级|上期评级|原有评级)\s*[:：]?\s*' + R + '?')
S6 = re.compile(r'^\s*(?:\[[A-Za-z_]+\]\s*)*' + R + GRADE + r'\s*$', re.M)
UP, DOWN = r'(?:上调|调高|调升|上修|提高|提升)', r'(?:下调|调低|调降|下修|降低)'
QR = rf'{Q}{R}{GRADE}{QE}'
B_CHG = [(re.compile(rf'(?:{UP}){GAP}{{0,8}}?评级{GAP}{{0,6}}?(?:至|为|到){GAP}{{0,4}}?{QR}'), '上调'),
         (re.compile(rf'评级{GAP}{{0,12}}?(?:{UP})(?:至|为|到){GAP}{{0,4}}?{QR}'), '上调'),
         (re.compile(rf'(?:{UP})(?:公司)?(?:至|为|到){QR}\s*(?:投资)?评级'), '上调'),
         (re.compile(rf'由{QR}(?:评级)?(?:{UP})(?:至|为|到){QR}'), '上调'),
         (re.compile(rf'(?:{DOWN}){GAP}{{0,8}}?评级{GAP}{{0,6}}?(?:至|为|到){GAP}{{0,4}}?{QR}'), '下调'),
         (re.compile(rf'评级{GAP}{{0,12}}?(?:{DOWN})(?:至|为|到){GAP}{{0,4}}?{QR}'), '下调'),
         (re.compile(rf'(?:{DOWN})(?:公司)?(?:至|为|到){QR}\s*(?:投资)?评级'), '下调'),
         (re.compile(rf'由{QR}(?:评级)?(?:{DOWN})(?:至|为|到){QR}'), '下调')]
B_FIRST = re.compile(rf'首次(?:覆盖|评级|推荐)?[^。；;\n]{{0,25}}?(?:给予|予以|给与|评级为){GAP}{{0,12}}?{QR}')
B_KEEP = re.compile(rf'维持{GAP}{{0,10}}?(?:{QR}\s*(?:投资)?评级|[“"「『‘]{R}{GRADE}[”"」』’]|{R}(?=[。；;，,]))')
B_GIVE = re.compile(rf'(?:给予|予以|给与){GAP}{{0,10}}?{QR}\s*(?:投资)?评级')
H_ACT = re.compile(r'首次覆盖|首次评级|维持评级')
AH = re.compile(r'\d+\s*HK\s*\d{6}\s*CH[^。]{0,30}?' + R + r'\s*[（(]\s*' + ACT + r'?\s*[）)]\s*' + R + r'\s*[（(]\s*' + ACT + r'?')
FC = '(?:盈利预测|业绩预测|利润预测|EPS|每股收益预测|收入预测|营收预测)'

# 评级说明（定义块）
DEF_WORDS = RATINGS + ['看好', '看淡', '超配', '标配', '低配', '跑赢', '跑输', '同步', '谨慎', '领先', '落后']
DR = '(' + '|'.join(sorted(set(DEF_WORDS), key=len, reverse=True)) + ')'
DEF = re.compile(DR + r'\s*(?:[（(][A-Za-z ]+[）)])?\s*[:：—\-\s（(]*\s*(?:预计|预期|相对|未来|股价|个股|公司股价|以报告|报告日后|超越|超过|涨幅|表现|高于|低于|介于|强于|弱于|优于)'
                 r'[^。]{0,70}?(?:\d+\s*[%％]|指数|基准|大盘|市场)')
LAB_STOCK = re.compile(r'公司评级|股票评级|个股评级|股票投资评级|公司投资评级|证券的投资评级|股票评级标准|公司投资评级标准|投资评级（另有说明的除外）|个股投资评级')
LAB_IND = re.compile(r'行业评级|行业投资评级|行业的投资评级|行业评级标准|板块评级|行业指数')
LAB_OTHER = re.compile(r'港股|美股|香港市场评级|海外')


# ---------------------------------------------------------------- 工具
def body_of(text):
    """去掉文末评级说明/免责声明：后一半正文里第一次出现 DISCLAIMER 关键词处截断；再把 PDF 的句中换行拼回去（中文不加空格）。"""
    text = re.sub(r'＃[A-Za-z_ ]*＃', ' ', text)  # 兴业页眉的 ＃investSuggestion＃ 占位符
    cut = [text.find(k, len(text) // 2) for k in DISCLAIMER]
    cut = [c for c in cut if c >= 0]
    raw = text[:min(cut)] if cut else text
    return raw, re.sub(r'[ \t]*\n[ \t]*', '', raw)


def evid(text, s, e, k=30):
    return re.sub(r'\s+', ' ', text[max(0, s - k):e + k])


def rgroup(m):
    for i, g in enumerate(m.groups(), start=1):
        if g in STRENGTH or g in ('无评级', '未评级'):
            return i
    return None


def ok_struct(text, m, g):
    s, e = m.span(g)
    if DEF_AFTER.match(text[e:e + 8]) or re.match(r'\s*(?:行业基本情况|行业相对|行业指数)', text[m.end():m.end() + 20]):
        return False  # 定义写法；或行业研报页眉（强于大市|维持 行业基本情况）
    return not any(b in text[max(0, m.start() - 8):s] for b in BAD_BEFORE)


def not_in_title(text, pos):
    """pos 不在《……》书名号里。"""
    left = text.rfind('《', 0, pos)
    return left < 0 or text.rfind('》', left, pos) >= 0


# ---------------------------------------------------------------- 本次评级与动作
def extract(text, company=''):
    raw, body = body_of(text)
    head = body[:HEAD_CHARS]
    hits = []  # (来源, 规则, 评级, 动作, 位置, 证据)
    for name, pat, region in [('S1', S1, body), ('S2', S2, body), ('S3', S3, head), ('S4', S4, head), ('S5', S5, raw[:HEAD_CHARS])]:
        for m in pat.finditer(region):
            g = rgroup(m)
            if g is None or not ok_struct(region, m, g) or not not_in_title(region, m.start()):
                continue
            acts = [x for x in m.groups() if x in ACT_MAP]
            hits.append(('S', name, m.group(g), ACT_MAP[acts[0]] if acts else None, m.start(), evid(region, *m.span())))
    if company:
        tm = re.search(re.escape(company) + r'\s*\d{6}\.(?:SH|SZ)[^。]{0,45}?\s' + R + r'\s', head)
        if tm:
            hits.append(('S', 'T', tm.group(1), None, tm.start(), evid(head, *tm.span())))
    last = next(((m.group(2) or '', evid(head, *m.span())) for m in L.finditer(head)), None)
    for m in S6.finditer(raw[:HEAD_CHARS]):
        hits.append(('S6', 'S6', m.group(1), None, m.start(), evid(raw, *m.span())))
    for pat, act in B_CHG:
        for m in pat.finditer(body):
            g = [i for i, x in enumerate(m.groups(), start=1) if x in STRENGTH][-1:]  # 由 X 调至 Y：取 Y
            if not g or any(b in body[max(0, m.start() - 4):m.start()] + m.group(0) for b in ('行业', '板块')):
                continue
            hits.append(('B', 'B_chg', m.group(g[0]), act, m.start(), evid(body, *m.span())))
    for name, pat, act in [('B_first', B_FIRST, '首次'), ('B_keep', B_KEEP, '维持'), ('B_give', B_GIVE, None)]:
        for m in pat.finditer(body):
            g = rgroup(m)
            if g is None or any(b in m.group(0) for b in ('行业', '板块')) or DEF_AFTER.match(body[m.end(g):m.end(g) + 8]):
                continue
            hits.append(('B', name, m.group(g), act, m.start(), evid(body, *m.span())))
    h_act = next((m for m in H_ACT.finditer(body[:FIRST_CHARS]) if not_in_title(body, m.start())), None)

    s_hits = sorted([h for h in hits if h[0] == 'S'], key=lambda h: h[4])
    ah = AH.search(head)
    if ah:  # 华泰 A+H：代码按“HK 在前、CH 在后”排列，评级同序，A 股评级是第二个
        acts = [x for x in ah.groups() if x in ACT_MAP]
        s_hits.insert(0, ('S', 'S_AH', ah.group(3), ACT_MAP[acts[-1]] if acts else None, -1, evid(head, *ah.span())))
    b_order = {'B_chg': 0, 'B_first': 1, 'B_keep': 2, 'B_give': 3}
    b_hits = sorted([h for h in hits if h[0] == 'B'], key=lambda h: (b_order[h[1]], h[4]))
    s6_hits = sorted([h for h in hits if h[0] == 'S6'], key=lambda h: h[4])
    top = (s_hits or b_hits or s6_hits or [None])[0]
    rating = top[2] if top else None

    src = {}
    s_act = next((h for h in s_hits if h[3]), None)
    if s_act:
        src['S'] = (s_act[3], s_act[1], s_act[5])
    if last and last[0] in STRENGTH and rating in STRENGTH:
        d = STRENGTH[rating] - STRENGTH[last[0]]
        src['L'] = ('上调' if d > 0 else '下调' if d < 0 else '维持', 'L', last[1])
    chg = next((h for h in b_hits if h[1] == 'B_chg'), None)
    if chg:
        src['B_chg'] = (chg[3], 'B_chg', chg[5])
    fk = sorted([h for h in b_hits if h[1] in ('B_first', 'B_keep')], key=lambda h: h[4])
    if fk:
        src['B_fk'] = (fk[0][3], fk[0][1], fk[0][5])
    if h_act:
        src['H'] = ('维持' if '维持' in h_act.group(0) else '首次', 'H', evid(body, *h_act.span()))
    if last and last[0] in ('', '无评级', '未评级'):
        src['L0'] = ('首次', 'L0', last[1])
    action, a_rule, a_ev = '未说明', '', ''
    for k in ['S', 'L', 'B_chg', 'B_fk', 'H', 'L0']:
        if k in src:
            action, a_rule, a_ev = src[k]
            break
    ev = [top[5]] if top else []
    if a_ev and a_ev not in ev:
        ev.append(a_ev)
    return {'rating_raw': rating, 'action': action, 'last_rating_raw': last[0] if last else None,
            'rule_rating': top[1] if top else '', 'rule_action': a_rule, 'evidence': ' || '.join(ev),
            'actions_found': '/'.join(sorted({v[0] for k, v in src.items() if k != 'L0'})),
            'ratings_found': '/'.join(dict.fromkeys(h[2] for h in s_hits + b_hits)),
            'fc_up': int(bool(re.search(rf'(?:{UP}){GAP}{{0,14}}?{FC}|{FC}{GAP}{{0,6}}?(?:{UP})', body))),
            'fc_down': int(bool(re.search(rf'(?:{DOWN}){GAP}{{0,14}}?{FC}|{FC}{GAP}{{0,6}}?(?:{DOWN})', body)))}


# ---------------------------------------------------------------- 评级说明 → 个股档位顺序
def parse_scale(text):
    """从评级说明解析个股评级档位顺序（从高到低），解析不到返回 None。"""
    t = re.sub(r'[ \t]*\n[ \t]*', '', text)
    ms = list(DEF.finditer(t))
    if len(ms) < 3:
        return None
    blocks, cur = [], [ms[0]]
    for a, b in zip(ms, ms[1:]):
        if b.start() - a.end() < 250:
            cur.append(b)
        else:
            blocks.append(cur)
            cur = [b]
    blocks.append(cur)
    for bl in blocks:
        start = bl[0].start()
        ctx = t[max(0, start - 120):bl[-1].end()]
        off = start - max(0, start - 120)
        seq = []
        for m in bl:
            pre = ctx[:m.start() - start + off]
            ls, li, lo = [max((x.end() for x in P.finditer(pre)), default=-1) for P in (LAB_STOCK, LAB_IND, LAB_OTHER)]
            kind = 'stock' if ls > max(li, lo) else 'ind' if li > max(ls, lo) else 'other' if lo > -1 else 'none'
            seq.append((m.group(1), kind))
        stock = [w for w, k in seq if k == 'stock']
        if not stock:  # 没有标签：去掉行业专用词后剩下的
            stock = [w for w, k in seq if k == 'none' and w not in ('看好', '看淡', '超配', '标配', '低配')]
        stock = list(dict.fromkeys(stock))
        if len(stock) >= 3 and stock[0] in STRENGTH and STRENGTH.get(stock[0], 0) > 3:
            return stock
    return None


def tier_in(scale, rating):
    """rating 在 scale（从高到低）里的档位；不在里面返回 None。"""
    if not scale or rating not in STRENGTH:
        return None
    r = rating if rating in scale else SYN.get(rating)
    if r not in scale:
        return None
    i = scale.index(r)
    if STRENGTH[r] < 3 or (i == 0 and r in NEUTRAL):  # 负面档；或体系第一档就是中性类（只用过“持有”）
        return 3
    if i == 0:
        return 1
    if i == 1:
        later_neutral = any(w in NEUTRAL for w in scale[2:])
        return 2 if (r not in NEUTRAL or later_neutral) else 3
    return 3


def usage_scale(counts):
    """该券商实际用过的档位（至少 USE_MIN 篇且占该券商识别出评级篇数的 USE_SHARE 以上）按通用强弱排序；最强的不是正面词时返回 None。"""
    tot = counts.sum()
    used = [w for w, n in counts.items() if n >= USE_MIN and n >= USE_SHARE * tot and w in STRENGTH]
    used = sorted(used, key=lambda w: -STRENGTH[w])
    return used if used and STRENGTH[used[0]] > 3 else None


def tier_fallback(rating):
    if rating not in STRENGTH:
        return np.nan
    return FALLBACK.get(rating, 3)


# ---------------------------------------------------------------- 主流程
def broker_tables(meta, text_dir, brokers):
    """语料库里这些券商的全部研报：评级说明解析结果（每篇一个顺序）；个股研报用 extract() 识别出的本次评级（算页眉/正文实际用词）。"""
    defs, use = [], []
    sub = meta[meta.broker.isin(brokers) & meta.file.notna()]
    for r in sub.itertuples():
        try:
            text = (text_dir / r.file).read_text(encoding='utf-8')
        except OSError:
            continue
        sc = parse_scale(text)
        if sc:
            defs.append({'broker': r.broker, 'report_id': r.report_id, 'date': r.date, 'scale': '/'.join(sc)})
        if r.is_stock_report:
            e = extract(text, r.company or '')
            if e['rating_raw'] in STRENGTH:
                use.append({'broker': r.broker, 'report_id': r.report_id, 'date': r.date, 'rating': e['rating_raw']})
    return (pd.DataFrame(defs, columns=['broker', 'report_id', 'date', 'scale']),
            pd.DataFrame(use, columns=['broker', 'report_id', 'date', 'rating']))


def regimes(u):
    """券商全年用词是否在某月前后换了一套互不相交的档位（如国信 2024 年 6 月起由 买入/增持 换成 优于大市）。
    返回 [(起始月, 结束月)]；只看每边占比不少于 3% 的档位。"""
    months = sorted(u.date.str[:6].unique())
    for k in range(1, len(months)):
        a = u[u.date.str[:6] < months[k]].rating.value_counts(normalize=True)
        b = u[u.date.str[:6] >= months[k]].rating.value_counts(normalize=True)
        sa, sb = set(a[a >= 0.03].index), set(b[b >= 0.03].index)
        n_a, n_b = (u.date.str[:6] < months[k]).sum(), (u.date.str[:6] >= months[k]).sum()
        if sa and sb and not (sa & sb) and min(n_a, n_b) >= 10:
            return [('000000', months[k - 1]), (months[k], '999999')]
    return [('000000', '999999')]


def scale_table(defs, use, brokers):
    """每家券商（换过体系的分时段）一行：档位顺序取该时段评级说明里最常见的顺序，没有则取实际用词。"""
    rows = []
    for b in sorted(brokers):
        ub, db = use[use.broker == b], defs[defs.broker == b]
        for lo, hi in regimes(ub) if len(ub) else [('000000', '999999')]:
            inr = lambda x: (x.date.str[:6] >= lo) & (x.date.str[:6] <= hi)  # noqa: E731
            d, u = db[inr(db)].scale, ub[inr(ub)].rating.value_counts()
            vc = d.value_counts()
            def_scale = vc.index[0].split('/') if len(vc) else None
            us = usage_scale(u)
            scale = def_scale or us
            rows.append({'broker': b, 'month_from': lo, 'month_to': hi,
                         'scale_source': 'def' if def_scale else 'usage' if us else 'fallback',
                         'scale': '/'.join(scale) if scale else '', 'top': scale[0] if scale else None,
                         'second': scale[1] if scale and len(scale) > 1 and tier_in(scale, scale[1]) == 2 else None,
                         'n_def_reports': int(len(d)), 'n_def_variants': int(len(vc)),
                         'def_consistent': (int(len(vc) == 1) if len(vc) else np.nan),
                         'def_majority_share': (round(vc.iloc[0] / len(d), 3) if len(vc) else np.nan),
                         'def_variants': '；'.join(f'{k}×{v}' for k, v in vc.head(4).items()),
                         'usage_scale': '/'.join(us) if us else '', 'n_usage_reports': int(u.sum()),
                         'usage_counts': '；'.join(f'{k}×{v}' for k, v in u.items())})
    return pd.DataFrame(rows)


def run(panel_path, meta_path, text_dir):
    p = pd.read_parquet(panel_path, columns=['report_id', 'code', 'company', 'date', 'broker'])
    meta = pd.read_parquet(meta_path, columns=['report_id', 'file', 'broker', 'date', 'company', 'is_stock_report'])
    p = p.merge(meta[['report_id', 'file']], on='report_id', how='left')
    defs, use = broker_tables(meta, text_dir, set(p.broker))
    st = scale_table(defs, use, set(p.broker))
    rows = []
    for r in p.itertuples():
        text = (text_dir / r.file).read_text(encoding='utf-8')
        e = extract(text, r.company)
        rating = e['rating_raw']
        own = parse_scale(text)
        b = st[(st.broker == r.broker) & (st.month_from <= r.date[:6]) & (st.month_to >= r.date[:6])].iloc[0]
        dscale = b.scale.split('/') if b.scale_source == 'def' else None
        uscale = b.usage_scale.split('/') if b.usage_scale else None
        tier, src, sc_used = None, '', None
        for name, sc in [('def_self', own), ('def_broker', dscale), ('usage', uscale)]:
            tier = tier_in(sc, rating)
            if tier is not None:
                src, sc_used = name, sc
                break
        if tier is None and rating in STRENGTH:
            tier, src = tier_fallback(rating), 'fallback'
        found = [x for x in e['ratings_found'].split('/') if x in STRENGTH]
        tiers = {tier_in(sc_used, x) if sc_used else tier_fallback(x) for x in found}
        conflict = int(len({t for t in tiers if t is not None}) > 1 or len(e['actions_found'].split('/')) > 1)
        rows.append({'report_id': r.report_id, **e, 'own_scale': '/'.join(own) if own else '', 'broker_scale': b.scale,
                     'tier': tier if tier is not None else np.nan, 'tier_source': src, 'tier_fb': tier_fallback(rating),
                     'conflict': conflict})
    out = p.drop(columns='file').merge(pd.DataFrame(rows), on='report_id')
    out['TOP'] = out.tier.map({1: 1.0, 2: 0.0})
    out['rule'] = out.rule_rating + '|' + out.rule_action + '|' + out.tier_source
    return out, st, defs, use
