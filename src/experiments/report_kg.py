"""从单篇研报 PDF 提取知识图谱并生成单文件 HTML（框架参照 KGGen, NeurIPS 2025）。

语义判断（抽取、去重）由 Claude 子代理按 src/prompts/report_kg_prompt.md 完成，这里只做确定性工作，分三个阶段：
  parse：抽出正文（不含表格、图表、财务报表、目录和法律声明），切句编号（sid）并切块，写出 chunks.json；
  check：校验子代理写的 raw_extractions.jsonl，聚合后写出待去重列表 entities_to_resolve.json、predicates_to_resolve.json；
  build：校验并应用 entity_resolution.json、predicate_resolution.json，写出 kg.json、stats.json、review.md 和 HTML。

与 KGGen 的差异：实体只有几百个，省略了 S-BERT + k-means + BM25 的检索环节，按类型分组后让子代理看整组（每组上限 128，沿用论文的簇大小）；
数值和状态、趋势等描述性取值仍存成三元组（object_kind=literal），但不参与实体消解，默认不画成节点；
地区可以作为限定条件（place）挂在三元组上，这样谓词和属性名里不再夹带地区，便于复用。

用法：在 main.py 中运行 report_kg_parse、report_kg_check、report_kg_build；子代理写完一块后用 lint 自查
"""
import json
import math
import random
import re
import unicodedata
from collections import Counter, defaultdict

import numpy as np

from src.llm_client import NUM_RE

SUMMARY_PAGE = 1  # 首页“观点聚焦”
BODY_PAGES = range(5, 38)  # 正文页码（从 1 计）；2-4 页为财务报表和目录，38 页起为法律声明
H1_SIZE, H2_SIZE, BODY_SIZE = 16, 14, 11  # 只认这三种字号，p32 的 13.1 号字是图内文字
MIN_SPAN_SIZE = 10  # 小于该字号的 span（脚注上标等）直接丢弃
BODY_MIN_X, SUMMARY_MIN_X, SUMMARY_MIN_Y = 140, 225, 260  # 正文栏的左边界；图内文字和左侧栏都在此之外
PAGE_TOP, PAGE_BOTTOM = 60, 775  # 页眉（含下载水印）和页脚（版权声明）之外的区域
NOT_BODY_RE = re.compile(r'^(图表\s*\d|资料来源|注[:：])')
SENT_END = '。！？；'
SENT_RE = re.compile(rf'[^{SENT_END}]+(?:[{SENT_END}]+[”’）)]*|$)')  # 段尾没有句号的残句也保留
CJK = r'　-〿一-鿿＀-￯'
SPACE_NEAR_CJK_RE = re.compile(rf'(?<=[{CJK}])\s+|\s+(?=[{CJK}])')
MAX_CHUNK_CHARS = 2000
SUMMARY_SECTION = '观点聚焦'

ENTITY_TYPES = ('公司', '品牌', '车型', '技术', '地区', '政策', '人物', '机构', '概念')
RESOLVE_GROUPS = {'公司': '组织与品牌', '品牌': '组织与品牌', '机构': '组织与品牌', '车型': '车型', '技术': '技术',
                  '地区': '地区与政策', '政策': '地区与政策', '人物': '人物与概念', '概念': '人物与概念'}
MAX_GROUP_SIZE = 128
STANCES = ('事实', '预测', '观点')
PRONOUNS = {'公司', '该公司', '本公司', '我们', '其', '它', '它们', '该', '此', '这', '这些', '当地', '该市场', '该地区',
            '上述', '对方', '我国', '笔者', '本报告'}
NUMERIC_NAME_RE = re.compile(r'^[\d.,%+\-–~/]+[万亿千百]?(辆|元|港元|美元|欧元|台|个|款|家|倍|x|ppt|pct|gwh|kw|km|公里|年|月)?$')
TIME_PREFIX_RE = re.compile(r'^(19|20)\d{2}\s*(年|-)|^[1-4]q\d{2}|^[12]h\d{2}|^\d{1,2}m\d{2}')

# 抽取校验：概念须是可复用的名词，谓词须短且不夹带实体和地区（规则见 src/prompts/report_kg_prompt.md）
PHRASE_MARKS = ('的', '、', '，', ',')
CONCEPT_MAX_LEN, PREDICATE_MAX_LEN, ATTRIBUTE_MAX_LEN, VALUE_MAX_LEN = 10, 6, 12, 10
STATE_SUFFIXES = ('加剧', '提升', '下降', '增长', '下滑', '回落', '改善', '恶化', '优异', '紧迫', '受限', '放缓', '加快', '加速',
                  '趋势', '突破', '承压', '向好', '初期', '阶段', '明显', '显著')
EVAL_PREFIXES = ('强劲', '突出', '深厚', '优质', '优异', '显著', '明显', '持续', '快速', '小幅', '大幅', '较强', '较高', '较低', '领先')
SCOPE_REGIONS = {'中国', '海外', '国内', '全球', '境外', '境内'}  # 泛称，概念名和属性名里可以保留（如“海外销量”）

# 去重校验：数字集合按阿拉伯数字比较；禁并表每一行内的各类互不合并（同一类内的别名不受影响）
CN_DIGITS = dict(zip('一二三四五六七八九十', range(1, 11)))
FORBIDDEN_MERGES = [
    [{'欧盟'}, {'欧盟委员会'}, {'欧洲'}],
    [{'上汽集团'}, {'上汽乘用车'}],
    [{'bev', '纯电', '纯电动', '纯电动车', '纯电动汽车'}, {'phev', '插混', '插电混动', '插电式混合动力', '插电式混合动力汽车'}],
]
INVERSE_PREDICATES = [('上调', '下调'), ('提升', '下降'), ('增长', '下滑'), ('增加', '减少'), ('扩大', '缩小'), ('进入', '退出'),
                      ('出口', '进口'), ('高于', '低于'), ('领先', '落后'), ('加征', '取消')]
SYMMETRIC_PREDICATES = {'合作', '合资', '竞争', '对标'}
MUST_MERGE = [('phev', '插电混动'), ('bev', '纯电'), ('byd', '比亚迪')]  # 两个名称都出现时才检查
MUST_NOT_MERGE = [('bev', 'phev'), ('元plus', '元up'), ('u7', 'u8'), ('u8', 'u8l'), ('海狮06', '海狮07ev'), ('欧盟', '欧洲'),
                  ('欧盟', '欧盟委员会'), ('上汽集团', '上汽乘用车'), ('第一代刀片电池', '第二代刀片电池'), ('刀片电池', '第二代刀片电池')]
FORECAST_WORDS = ('预计', '有望', '我们认为')
MIN_COVERAGE, MAX_HUB_SHARE = 0.6, 0.6  # 经验阈值，只提示不报错
REVIEW_SAMPLE = 40

# 画图：配色分组与顺序（机构、人物合用一种颜色，概念用中性灰），形状作为颜色之外的第二编码
COLOR_GROUPS = {'公司': '公司', '品牌': '品牌', '车型': '车型', '技术': '技术', '地区': '地区', '政策': '政策',
                '机构': '机构/人物', '人物': '机构/人物', '概念': '概念'}
GROUP_ORDER = ['公司', '品牌', '车型', '技术', '地区', '政策', '机构/人物', '概念']
ANCHOR_WEIGHT, LAYOUT_K, LAYOUT_ITERATIONS, LAYOUT_SCALE, LAYOUT_SEED = 0.5, 0.11, 300, 900, 42
HTML_NAME = 'byd_cicc_kg.html'


def tidy(text):
    """去掉中文两侧的空白（PDF 在数字和中文之间插了空格），保留西文名内部的空格。"""
    return re.sub(r'\s+', ' ', SPACE_NEAR_CJK_RE.sub('', text)).strip()


def block_text(block):
    """拼接一个 block 的文字并返回 (文字, 主要字号)；行尾行首都是西文时补一个空格，避免断行把单词粘在一起。"""
    parts, sizes = [], Counter()
    for line in block['lines']:
        spans = [s for s in line['spans'] if s['size'] >= MIN_SPAN_SIZE]
        text = ''.join(s['text'] for s in spans)
        for s in spans:
            sizes[round(s['size'])] += len(s['text'].strip())
        if parts and re.search(r'[A-Za-z]$', parts[-1].rstrip()) and re.match(r'\s*[A-Za-z]', text):
            text = ' ' + text.lstrip()
        parts.append(text)
    return tidy(''.join(parts)), (sizes.most_common(1)[0][0] if sizes else 0)


def read_items(doc):
    """按阅读顺序返回 ('h1'|'h2'|'p', 文字, 页码)。"""
    items = []
    for page_no in [SUMMARY_PAGE, *BODY_PAGES]:
        rows = []
        for b in doc[page_no - 1].get_text('dict')['blocks']:
            if b['type'] != 0:
                continue
            x0, y0 = b['bbox'][:2]
            if not PAGE_TOP < y0 < PAGE_BOTTOM:
                continue
            text, size = block_text(b)
            if not text:
                continue
            if page_no == SUMMARY_PAGE:
                is_label = len(text) <= 10 and text[-1] not in SENT_END  # “投资建议”“理由”“风险”等小标签
                if size == BODY_SIZE and x0 >= SUMMARY_MIN_X and y0 >= SUMMARY_MIN_Y and not is_label:
                    rows.append((y0, 'p', text))
            elif x0 >= BODY_MIN_X and size == H1_SIZE:
                rows.append((y0, 'h1', text))
            elif x0 >= BODY_MIN_X and size == H2_SIZE:
                rows.append((y0, 'h2', text))
            elif x0 >= BODY_MIN_X and size == BODY_SIZE and not NOT_BODY_RE.match(text):
                rows.append((y0, 'p', text))
        items += [(kind, text, page_no) for _, kind, text in sorted(rows, key=lambda r: r[0])]
    return items


def build_paragraphs(items):
    """把跨块、跨页断开的句子拼回段落，返回 [{h1, h2, pieces: [(文字, 页码), ...]}]。"""
    paras, h1, h2, cur = [], SUMMARY_SECTION, '', None
    for kind, text, page in items:
        if kind == 'h1':
            h1, h2, cur = text, '', None
        elif kind == 'h2':
            h2, cur = text, None
        elif cur is not None and cur['pieces'][-1][0][-1] not in SENT_END + '”）':
            cur['pieces'].append((text, page))
        else:
            cur = {'h1': h1, 'h2': h2, 'pieces': [(text, page)]}
            paras.append(cur)
    return paras


def paragraph_sentences(para):
    """切句并按句子起点所在的片段确定页码，返回 [(句子, 页码)]。"""
    text = ''.join(p[0] for p in para['pieces'])
    ends, total = [], 0
    for piece, page in para['pieces']:
        total += len(piece)
        ends.append((total, page))
    return [(m.group().strip(), next(page for end, page in ends if m.start() < end))
            for m in SENT_RE.finditer(text) if m.group().strip()]


def make_chunks(paras):
    """按章节切块，单块超过 MAX_CHUNK_CHARS 时在段落边界换块（单段过长时在句子边界换块）。"""
    chunks, sid = [], 0

    def new_chunk(para):
        chunks.append({'chunk_id': f'c{len(chunks) + 1:02d}', 'h1': para['h1'], 'h2': para['h2'], 'sentences': []})

    for para in paras:
        last = chunks[-1] if chunks else None
        size = sum(len(s['text']) for s in last['sentences']) if last else 0
        para_len = sum(len(p[0]) for p in para['pieces'])
        if not last or (last['h1'], last['h2']) != (para['h1'], para['h2']) or size + para_len > MAX_CHUNK_CHARS:
            new_chunk(para)
        for text, page in paragraph_sentences(para):
            if sum(len(s['text']) for s in chunks[-1]['sentences']) + len(text) > MAX_CHUNK_CHARS:
                new_chunk(para)
            sid += 1
            chunks[-1]['sentences'].append({'sid': sid, 'page': page, 'text': text})
    return chunks


def leak_terms(pdf_path):
    """页眉水印里的下载人姓名（从 PDF 动态读取，不写进代码）。邮箱统一用 @ 判断。"""
    import pymupdf  # 只在解析 PDF 时导入，其他阶段不需要装 pymupdf
    with pymupdf.open(pdf_path) as doc:
        m = re.search(r'该报告由\s*(\S+?)\s*于\s*\d{4}\s*年', doc[0].get_text())
    if not m:
        raise SystemExit('未在首页找到下载水印，无法校验脱敏结果，请检查 PDF')
    return ['@', m.group(1)]


def assert_no_leak(text, terms, where):
    hit = [t for t in terms if t in text]
    if hit:
        raise SystemExit(f'{where} 中仍含下载水印或邮箱（命中 {len(hit)} 项），已停止')


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def parse(pdf_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    import pymupdf  # 只在解析 PDF 时导入，其他阶段不需要装 pymupdf
    with pymupdf.open(pdf_path) as doc:
        items = read_items(doc)
    paras = build_paragraphs(items)
    chunks = make_chunks(paras)
    text = json.dumps(chunks, ensure_ascii=False)
    assert_no_leak(text, leak_terms(pdf_path), 'chunks.json')
    write_json(out_dir / 'chunks.json', chunks)
    sentences = [s for c in chunks for s in c['sentences']]
    print(f'{len(chunks)} 块，{len(sentences)} 句，{sum(len(s["text"]) for s in sentences)} 字')
    for c in chunks:
        chars = sum(len(s['text']) for s in c['sentences'])
        print(f'  {c["chunk_id"]} {len(c["sentences"]):3d} 句 {chars:5d} 字  {c["h1"][:14]} / {c["h2"][:18]}')
    unfinished = [s for s in sentences if s['text'][-1] not in SENT_END + '”）)']
    print(f'无句末标点的句子 {len(unfinished)} 句：')
    for s in unfinished:
        print(f'  sid={s["sid"]} p{s["page"]}: {s["text"][:50]}')


def norm(name):
    """KGGen §4.2 的聚合规范化，中文下对应 NFKC、拉丁字母转小写、去空白。"""
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', str(name))).lower()


def load_raw(out_dir):
    """读取子代理逐块写的 raw/<chunk_id>.json，并合并成 raw_extractions.jsonl 留作基准。"""
    rows = {}
    for path in sorted((out_dir / 'raw').glob('*.json')):
        row = read_json(path)
        rows[row['chunk_id']] = row
    (out_dir / 'raw_extractions.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows.values()), encoding='utf-8')
    return rows


def bad_entity_name(name, etype=None):
    key = norm(name)
    if not key:
        return '空名称'
    if key in PRONOUNS:
        return '代词或泛指'
    if NUMERIC_NAME_RE.match(key):
        return '纯数值，应作 literal 宾语'
    if TIME_PREFIX_RE.match(key):
        return '名称带时间，应放到 time 字段'
    if any(m in key for m in PHRASE_MARKS):
        return '是短语或列举，应拆成单个名词'
    if etype == '概念' and len(key) > CONCEPT_MAX_LEN:
        return f'概念名超过 {CONCEPT_MAX_LEN} 个字，可能是短语'
    if etype == '概念' and (key.endswith(STATE_SUFFIXES) or key.startswith(EVAL_PREFIXES)):
        return '是状态、趋势或评价，应作 literal 描述性取值，或去掉评价词'
    return None


def validate_chunk(chunk, row):
    """返回该块的错误列表；校验项见 src/prompts/report_kg_prompt.md 的输出约束。"""
    errors, text_of = [], {s['sid']: s['text'] for s in chunk['sentences']}
    type_of = {}
    for e in row.get('entities', []):
        if e.get('type') not in ENTITY_TYPES:
            errors.append(f'实体“{e.get("name")}”类型非法：{e.get("type")}')
        reason = bad_entity_name(e.get('name', ''), e.get('type'))
        if reason:
            errors.append(f'实体“{e.get("name")}”：{reason}')
        type_of[norm(e.get('name', ''))] = e.get('type')
    known = set(type_of)
    regions = {k for k, v in type_of.items() if v == '地区' and k not in SCOPE_REGIONS}
    named = {k for k, v in type_of.items() if v != '概念' and len(k) >= 2}
    for e in row.get('entities', []):
        key = norm(e.get('name', ''))
        if e.get('type') == '概念' and any(key.startswith(r) and key != r for r in regions):
            errors.append(f'实体“{e.get("name")}”：名称带地区限定，地区应放到三元组的 place 字段')
    for i, t in enumerate(row.get('triples', [])):
        tag = f'三元组{i}（{t.get("subject")} - {t.get("predicate")} - {t.get("object")}）'
        if norm(t.get('subject', '')) not in known:
            errors.append(f'{tag}：主语不在实体表内')
        kind = t.get('object_kind')
        if kind not in ('entity', 'literal'):
            errors.append(f'{tag}：object_kind 非法')
        if kind == 'entity' and norm(t.get('object', '')) not in known:
            errors.append(f'{tag}：实体宾语不在实体表内')
        predicate = norm(t.get('predicate', ''))
        if not predicate or predicate.startswith(('被', '在')):
            errors.append(f'{tag}：谓词为空，或以“被”“在”开头（改成主动语态，地点放到 place）')
        if re.search(r'\d', predicate):
            errors.append(f'{tag}：谓词含数字（时间放到 time，数值作 literal 宾语）')
        if kind == 'entity':
            if len(predicate) > PREDICATE_MAX_LEN:
                errors.append(f'{tag}：关系谓词超过 {PREDICATE_MAX_LEN} 个字')
            hit = [n for n in named if n in predicate and n != predicate]
            if hit:
                errors.append(f'{tag}：谓词含实体名 {hit}')
        if kind == 'literal':
            if len(predicate) > ATTRIBUTE_MAX_LEN:
                errors.append(f'{tag}：属性名超过 {ATTRIBUTE_MAX_LEN} 个字')
            hit = [r for r in regions if r in predicate]
            if hit:
                errors.append(f'{tag}：属性名含地区 {hit}，地区应放到 place')
            value = norm(t.get('object', ''))
            if not re.search(r'\d', value) and len(value) > VALUE_MAX_LEN:
                errors.append(f'{tag}：描述性取值超过 {VALUE_MAX_LEN} 个字')
        place = t.get('place')
        if place and type_of.get(norm(place)) != '地区':
            errors.append(f'{tag}：place“{place}”不是实体表中类型为“地区”的实体')
        if t.get('stance') not in STANCES:
            errors.append(f'{tag}：stance 非法')
        sids = t.get('sids') or []
        if not sids or any(s not in text_of for s in sids):
            errors.append(f'{tag}：sids 为空或不属于本块')
            continue
        cited = norm(''.join(text_of[s] for s in sids))
        if kind == 'literal':
            missing = [n for n in NUM_RE.findall(norm(t.get('object', ''))) if n not in cited]
            if missing:
                errors.append(f'{tag}：数值 {missing} 不在所引句中')
    return errors


def lint(out_dir, chunk_ids):
    """只校验指定的块，供子代理写完一块后自查，不做聚合。"""
    chunks = {c['chunk_id']: c for c in read_json(out_dir / 'chunks.json')}
    for cid in chunk_ids:
        path = out_dir / 'raw' / f'{cid}.json'
        if not path.exists():
            print(f'{cid}：尚未写出 {path.name}')
            continue
        found = validate_chunk(chunks[cid], read_json(path))
        print(f'{cid}：' + (f'{len(found)} 个问题' if found else '通过'))
        for f in found:
            print('  ', f)


def aggregate_items(chunks, raw):
    """KGGen §4.2：跨块聚合并精确去重，只保留出现在三元组里（含 place）的实体。"""
    surfaces, types, sids_of = defaultdict(Counter), defaultdict(Counter), defaultdict(list)
    predicates, examples, kinds = Counter(), {}, defaultdict(set)
    for row in raw.values():
        type_of = {norm(e['name']): e['type'] for e in row['entities']}
        for t in row['triples']:
            names = [t['subject']] + ([t['object']] if t['object_kind'] == 'entity' else []) + ([t['place']] if t.get('place') else [])
            for name in names:
                key = norm(name)
                surfaces[key][name.strip()] += 1
                types[key][type_of[key]] += 1
                sids_of[key] += t['sids']
            predicates[t['predicate'].strip()] += 1
            examples.setdefault(t['predicate'].strip(), f'{t["subject"]} → {t["object"]}')
            kinds[t['predicate'].strip()].add('关系谓词' if t['object_kind'] == 'entity' else '属性名')
    entities = [{'name': surfaces[k].most_common(1)[0][0], 'type': types[k].most_common(1)[0][0],
                 'mentions': sum(surfaces[k].values()), 'surfaces': sorted(surfaces[k]),
                 'sids': sorted(set(sids_of[k]))[:3]} for k in surfaces]
    return entities, predicates, examples, kinds


def check(out_dir):
    """校验抽取结果并生成待去重列表；有错误时写出 validation_errors.json 并报错。"""
    chunks = read_json(out_dir / 'chunks.json')
    raw = load_raw(out_dir)
    errors = {}
    missing = [c['chunk_id'] for c in chunks if c['chunk_id'] not in raw]
    if missing:
        errors['_missing_chunks'] = missing
    for c in chunks:
        if c['chunk_id'] in raw:
            found = validate_chunk(c, raw[c['chunk_id']])
            if found:
                errors[c['chunk_id']] = found
    write_json(out_dir / 'validation_errors.json', errors)
    if errors:
        n = sum(len(v) for v in errors.values())
        for cid, found in list(errors.items())[:30]:
            print(cid, found[:8])
        raise SystemExit(f'校验未通过：{n} 个问题，详见 validation_errors.json')

    entities, predicates, examples, kinds = aggregate_items(chunks, raw)
    groups = defaultdict(list)
    for e in sorted(entities, key=lambda x: norm(x['name'])):
        groups[RESOLVE_GROUPS[e['type']]].append(e)
    out = {}
    for name, items in groups.items():
        for k in range(0, len(items), MAX_GROUP_SIZE):
            part = f'{name}-{k // MAX_GROUP_SIZE + 1}' if len(items) > MAX_GROUP_SIZE else name
            out[part] = items[k:k + MAX_GROUP_SIZE]
    write_json(out_dir / 'entities_to_resolve.json', out)
    write_json(out_dir / 'predicates_to_resolve.json',
               {kind: [{'predicate': p, 'count': n, 'example': examples[p]} for p, n in predicates.most_common() if kind in kinds[p]]
                for kind in ('关系谓词', '属性名')})

    triples = [t for r in raw.values() for t in r['triples']]
    n_literal = sum(t['object_kind'] == 'literal' for t in triples)
    n_place = sum(bool(t.get('place')) for t in triples)
    print(f'校验通过：{len(triples)} 条三元组（其中 literal {n_literal} 条，带 place {n_place} 条），'
          f'聚合后实体 {len(entities)} 个，谓词 {len(predicates)} 个')
    print('分组：', {k: len(v) for k, v in out.items()})


def merge_rejection(alias, canonical):
    """实体合并的代码校验（KGGen §4.3 的“相似但不同不得合并”）：返回拒绝原因，可以合并时返回 None。"""
    if number_set(alias) != number_set(canonical):
        return '数字或代际不同'
    if cjk_part(alias) == cjk_part(canonical) and latin_part(alias) != latin_part(canonical):
        return '中文部分相同而西文或型号不同'
    for classes in FORBIDDEN_MERGES:
        a, b = [i for i, c in enumerate(classes) if alias in c], [i for i, c in enumerate(classes) if canonical in c]
        if a and b and a != b:
            return '在禁并表内'
    return None


def number_set(key):
    """名称里的数字集合；量词前的中文数字也换成阿拉伯数字（第二代→2，六国→6）。"""
    key = re.sub(r'([一二三四五六七八九十])(?=[国代款家座门排])', lambda m: str(CN_DIGITS[m.group(1)]), key)
    return {float(n) for n in re.findall(r'\d+(?:\.\d+)?', key)}


def cjk_part(key):
    return ''.join(re.findall(r'[一-鿿]+', key))


def latin_part(key):
    return ''.join(re.findall(r'[a-z]+', key))


def predicate_rejection(alias, canonical):
    if alias.startswith('被') or canonical.startswith('被'):
        return '被动语态'
    for a, b in INVERSE_PREDICATES:
        if (a in alias and b in canonical) or (b in alias and a in canonical):
            return f'方向相反（{a}/{b}）'
    return None


def load_merges(path, known, guard):
    """读取子代理的去重结果并逐个别名对照规范名校验，返回 (映射, 规范名显示, 采纳的集合, 被拒的别名)。"""
    mapping, display, accepted, rejected, seen = {}, {}, [], [], set()
    for group in read_json(path):
        canonical = norm(group['canonical'])
        kept = []
        for alias in group['aliases']:
            key = norm(alias)
            if key == canonical:
                continue
            if key not in known:
                raise SystemExit(f'{path.name}：别名“{alias}”不在待去重列表中')
            if key in seen:
                raise SystemExit(f'{path.name}：别名“{alias}”出现在多个集合中')
            seen.add(key)
            reason = guard(key, canonical)
            if reason:
                rejected.append({'alias': alias, 'canonical': group['canonical'], 'reason': reason})
            else:
                mapping[key] = canonical
                kept.append(alias)
        if kept:
            display[canonical] = group['canonical'].strip()
            accepted.append({**group, 'aliases': kept})
    for key in list(mapping):  # 规范名本身又是别的集合的别名时，一路归到最终规范名
        while mapping[key] in mapping:
            mapping[key] = mapping[mapping[key]]
    return mapping, display, accepted, rejected


def section_name(h1):
    return h1.split('：')[0]


def build_graph(chunks, raw, entity_map, entity_display, predicate_map, predicate_display):
    """应用别名映射，相同 (主语, 谓词, 宾语) 合并为一条边，所有观测保留在 obs 中。"""
    sentence = {s['sid']: {**s, 'section': section_name(c['h1']), 'h2': c['h2']} for c in chunks for s in c['sentences']}
    info = defaultdict(lambda: {'surfaces': Counter(), 'types': Counter()})
    pred_surfaces = defaultdict(Counter)
    edges, literals, self_loops, before = {}, {}, [], set()

    def add(store, key, triple):
        obs = {'time': triple.get('time'), 'place': triple['_place'], 'stance': triple['stance'], 'sids': triple['sids'],
               'raw': triple['predicate'].strip()}
        item = store.setdefault(key, {'obs': []})
        if obs not in item['obs']:
            item['obs'].append(obs)

    for row in raw.values():
        type_of = {norm(e['name']): e['type'] for e in row['entities']}
        for t in row['triples']:
            subject_raw, predicate_raw = norm(t['subject']), norm(t['predicate'])
            subject = entity_map.get(subject_raw, subject_raw)
            predicate = predicate_map.get(predicate_raw, predicate_raw)
            pred_surfaces[predicate][t['predicate'].strip()] += 1
            info[subject]['surfaces'][t['subject'].strip()] += 1
            info[subject]['types'][type_of[subject_raw]] += 1
            t = {**t, '_place': None}
            if t.get('place'):
                place_raw = norm(t['place'])
                t['_place'] = entity_map.get(place_raw, place_raw)
                info[t['_place']]['surfaces'][t['place'].strip()] += 1
                info[t['_place']]['types'][type_of[place_raw]] += 1
            if t['object_kind'] == 'literal':
                add(literals, (subject, predicate, norm(t['object'])), t)
                literals[(subject, predicate, norm(t['object']))]['value'] = t['object'].strip()
                continue
            object_raw = norm(t['object'])
            target = entity_map.get(object_raw, object_raw)
            info[target]['surfaces'][t['object'].strip()] += 1
            info[target]['types'][type_of[object_raw]] += 1
            before.add((subject_raw, predicate_raw, object_raw))
            if subject == target:
                self_loops.append(f'{t["subject"]} - {t["predicate"]} - {t["object"]}')
                continue
            s, o = sorted((subject, target)) if predicate in SYMMETRIC_PREDICATES else (subject, target)
            add(edges, (s, predicate, o), t)

    degree = Counter()
    for s, _, o in edges:
        degree[s] += 1
        degree[o] += 1
    keys = sorted(info, key=lambda k: (-degree[k], k))
    index = {k: i for i, k in enumerate(keys)}
    pred_name = {k: predicate_display.get(k) or c.most_common(1)[0][0] for k, c in pred_surfaces.items()}

    def sections_of(obs_list):
        order = {section_name(c['h1']): i for i, c in reversed(list(enumerate(chunks)))}
        return sorted({sentence[sid]['section'] for o in obs_list for sid in o['sids']}, key=order.get)

    for item in [*edges.values(), *literals.values()]:
        for ob in item['obs']:
            ob['place'] = index[ob['place']] if ob['place'] else None
    edge_rows = [{'s': index[s], 't': index[o], 'p': pred_name[p], 'obs': e['obs'], 'sections': sections_of(e['obs']),
                  'fact': any(ob['stance'] == '事实' for ob in e['obs']), 'sym': p in SYMMETRIC_PREDICATES}
                 for (s, p, o), e in edges.items()]
    literal_rows = [{'s': index[s], 'p': pred_name[p], 'v': e['value'], 'obs': e['obs'], 'sections': sections_of(e['obs'])}
                    for (s, p, _), e in literals.items()]
    section_votes, context = defaultdict(Counter), Counter()  # context：以该地区为 place 的关系和属性条数
    for row in edge_rows + literal_rows:
        places = {ob['place'] for ob in row['obs'] if ob['place'] is not None}
        context.update(places)
        ends = [row['s']] + ([row['t']] if 't' in row else []) + sorted(places)
        for end in ends:
            for ob in row['obs']:
                for sid in ob['sids']:
                    section_votes[end][sentence[sid]['section']] += 1
    nodes = []
    for k in keys:
        name = entity_display.get(k) or info[k]['surfaces'].most_common(1)[0][0]
        etype = info[k]['types'].most_common(1)[0][0]
        votes = section_votes[index[k]]
        nodes.append({'id': index[k], 'key': k, 'name': name, 'type': etype, 'group': COLOR_GROUPS[etype],
                      'aliases': sorted(s for s in info[k]['surfaces'] if s != name), 'mentions': sum(info[k]['surfaces'].values()),
                      'deg': degree[k], 'ctx': context[index[k]], 'section': votes.most_common(1)[0][0], 'sections': sorted(votes)})
    return nodes, edge_rows, literal_rows, sentence, self_loops, len(before)


def add_layout(nodes, edges, sections):
    """预计算坐标：中心实体放在原点、不参与布局（否则所有节点都被它拉成一团），其余节点按所属章节聚集。"""
    hub = nodes[0]['id']
    others = [n for n in nodes if n['id'] != hub]
    size = Counter(n['section'] for n in others)
    total = sum(math.sqrt(v) for v in size.values())
    anchors, angle = {}, -math.pi / 2
    for section in sections:  # 章节锚点按节点数分配圆周，节点多的章节占更大的扇区
        share = 2 * math.pi * math.sqrt(size[section]) / total if size[section] else 0
        anchors[section] = (math.cos(angle + share / 2), math.sin(angle + share / 2))
        angle += share

    index = {n['id']: i for i, n in enumerate(others)}
    anchor_index = {s: len(others) + i for i, s in enumerate(sections)}
    n_all = len(others) + len(sections)
    weight = np.zeros((n_all, n_all))
    for e in edges:
        if hub not in (e['s'], e['t']):
            i, j = index[e['s']], index[e['t']]
            weight[i, j] += 1
            weight[j, i] += 1
    rng = np.random.default_rng(LAYOUT_SEED)
    pos = np.zeros((n_all, 2))
    for n in others:
        i, a = index[n['id']], anchor_index[n['section']]
        weight[i, a] = weight[a, i] = ANCHOR_WEIGHT
        pos[i] = anchors[n['section']] + rng.uniform(-0.3, 0.3, 2)
    for section, a in anchor_index.items():
        pos[a] = anchors[section]
    pos = force_layout(pos, weight, n_fixed=len(sections))
    for n in nodes:
        x, y = (0.0, 0.0) if n['id'] == hub else pos[index[n['id']]]
        n['x'], n['y'] = round(float(x) * LAYOUT_SCALE, 1), round(float(y) * LAYOUT_SCALE, 1)


def force_layout(pos, weight, n_fixed):
    """Fruchterman-Reingold 力导向布局（networkx 在 500 个节点以上要依赖 scipy，这里用 numpy 实现）；最后 n_fixed 个点固定不动。"""
    temperature = 0.12
    cooling = temperature / (LAYOUT_ITERATIONS + 1)
    for _ in range(LAYOUT_ITERATIONS):
        delta = pos[:, None, :] - pos[None, :, :]
        distance = np.maximum(np.linalg.norm(delta, axis=-1), 0.01)
        force = LAYOUT_K ** 2 / distance ** 2 - weight * distance / LAYOUT_K
        move = (delta * force[..., None]).sum(axis=1)
        length = np.maximum(np.linalg.norm(move, axis=-1), 0.01)
        step = move * (temperature / length)[:, None]
        step[-n_fixed:] = 0
        pos = pos + step
        temperature -= cooling
    return pos


def literal_found(literals, sentence, value, predicate=''):
    return any(value in norm(r['v']) and predicate in norm(r['p'])
               and any(value in norm(sentence[sid]['text']) for ob in r['obs'] for sid in ob['sids']) for r in literals)


def relation_found(nodes, edges, name, keyword):
    target = norm(name)
    label = {n['id']: norm(n['name']) for n in nodes}
    return any(target in (label[e['s']], label[e['t']]) and keyword in norm(e['p']) + label[e['s']] + label[e['t']] for e in edges)


def acceptance(nodes, edges, literals, sentence, chunks, entity_map, known):
    """验收检查，返回 (未通过项, 需人工复核的提示, 未覆盖的句子)。"""
    failures, notes = [], []
    if any(n['key'] in PRONOUNS or bad_entity_name(n['name']) for n in nodes):
        failures.append('存在代词、纯数值或带时间的节点：' + str([n['name'] for n in nodes if n['key'] in PRONOUNS or bad_entity_name(n['name'])]))
    if any(e['s'] == e['t'] for e in edges):
        failures.append('存在自环')
    if any(sid not in sentence for row in edges + literals for ob in row['obs'] for sid in ob['sids']):
        failures.append('存在无效 sid')
    for row in literals:
        for ob in row['obs']:
            cited = norm(''.join(sentence[sid]['text'] for sid in ob['sids']))
            if any(num not in cited for num in NUM_RE.findall(norm(row['v']))):
                failures.append(f'数值不在所引句中：{row["p"]} {row["v"]}')

    final = lambda k: entity_map.get(k, k)
    for a, b in MUST_MERGE:
        if a in known and b in known and final(a) != final(b):
            failures.append(f'应合并而未合并：{a} / {b}')
    for a, b in MUST_NOT_MERGE:
        if a in known and b in known and final(a) == final(b):
            failures.append(f'不应合并却合并了：{a} / {b}')

    required = [('2025 年海外销量 104.6 万辆', literal_found(literals, sentence, '104.6')),
                ('1H26 海外销量 79.2 万辆', literal_found(literals, sentence, '79.2')),
                ('比亚迪 BEV 反补贴税率 17%', literal_found(literals, sentence, '17%')),
                ('上汽集团 BEV 反补贴税率 35.3%', literal_found(literals, sentence, '35.3%')),
                ('目标价 120 元', literal_found(literals, sentence, '120', '目标价')),
                ('目标价 130 港元', literal_found(literals, sentence, '130', '目标价')),
                ('腾势属高端品牌矩阵', relation_found(nodes, edges, '腾势', '高端')),
                ('方程豹属高端品牌矩阵', relation_found(nodes, edges, '方程豹', '高端')),
                ('仰望属高端品牌矩阵', relation_found(nodes, edges, '仰望', '高端')),
                ('英国重启补贴', relation_found(nodes, edges, '英国', '补贴')),
                ('德国重启补贴', relation_found(nodes, edges, '德国', '补贴'))]
    failures += [f'关键事实未在图中找到：{name}' for name, ok in required if not ok]

    for row in edges + literals:
        for ob in row['obs']:
            if ob['stance'] == '事实' and any(w in sentence[sid]['text'] for sid in ob['sids'] for w in FORECAST_WORDS):
                notes.append({'triple': f'{nodes[row["s"]]["name"]} - {row["p"]} - {nodes[row["t"]]["name"] if "t" in row else row["v"]}',
                              'sids': ob['sids']})
    cited = {sid for row in edges + literals for ob in row['obs'] for sid in ob['sids']}
    coverage, uncovered = [], []
    for c in chunks:
        ids = [s['sid'] for s in c['sentences']]
        coverage.append((c['chunk_id'], c['h2'] or c['h1'], sum(i in cited for i in ids), len(ids)))
        uncovered += [s for s in c['sentences'] if s['sid'] not in cited]
    return failures, notes, coverage, uncovered


def write_review(out_dir, entity_groups, entity_rejected, predicate_groups, predicate_rejected, self_loops, notes, coverage,
                 uncovered, nodes, edges, sentence):
    lines = ['# 知识图谱复核清单', '', f'## 实体合并（{len(entity_groups)} 组）', '']
    lines += [f'- **{g["canonical"]}** ← {"、".join(g["aliases"])}（sid {g.get("sids")}）：{g.get("reason", "")}' for g in entity_groups]
    lines += ['', f'## 被代码拒绝的实体合并（{len(entity_rejected)} 项）', '']
    lines += [f'- {r["alias"]} → {r["canonical"]}：{r["reason"]}' for r in entity_rejected]
    lines += ['', f'## 谓词合并（{len(predicate_groups)} 组）', '']
    lines += [f'- **{g["canonical"]}** ← {"、".join(g["aliases"])}' for g in predicate_groups]
    lines += ['', f'## 被代码拒绝的谓词合并（{len(predicate_rejected)} 项）', '']
    lines += [f'- {r["alias"]} → {r["canonical"]}：{r["reason"]}' for r in predicate_rejected]
    lines += ['', f'## 删除的自环（{len(self_loops)} 条）', '', *[f'- {x}' for x in self_loops]]
    lines += ['', f'## 标为事实但所引句含预测词（{len(notes)} 条）', '']
    lines += [f'- {n["triple"]}：{"".join(sentence[s]["text"] for s in n["sids"])[:120]}' for n in notes]
    lines += ['', '## 各块句子覆盖率', '', *[f'- {c}：{a}/{b}  {h[:24]}' for c, h, a, b in coverage]]
    lines += ['', f'## 未被引用的句子（{len(uncovered)} 句）', '', *[f'- [{s["sid"]}] {s["text"]}' for s in uncovered]]
    sample = random.Random(LAYOUT_SEED).sample(edges, min(REVIEW_SAMPLE, len(edges)))
    lines += ['', f'## 随机抽查的关系（{len(sample)} 条，seed={LAYOUT_SEED}）', '']
    for e in sample:
        evidence = ' '.join(sentence[sid]['text'] for ob in e['obs'] for sid in ob['sids'])
        lines.append(f'- **{nodes[e["s"]]["name"]} - {e["p"]} - {nodes[e["t"]]["name"]}**（{e["obs"][0]["stance"]}）：{evidence}')
    (out_dir / 'review.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def build(pdf_path, out_dir):
    """应用去重结果，生成 kg.json、stats.json、review.md 和 HTML；硬性检查未通过时产物照常写出，最后报错。"""
    chunks = read_json(out_dir / 'chunks.json')
    raw = load_raw(out_dir)
    entities, predicates, *_ = aggregate_items(chunks, raw)
    known = {norm(s) for e in entities for s in e['surfaces']}
    known_predicates = {norm(p) for p in predicates}
    entity_map, entity_display, entity_groups, entity_rejected = load_merges(out_dir / 'entity_resolution.json', known, merge_rejection)
    predicate_map, predicate_display, predicate_groups, predicate_rejected = load_merges(
        out_dir / 'predicate_resolution.json', known_predicates, predicate_rejection)
    nodes, edges, literals, sentence, self_loops, edges_before = build_graph(
        chunks, raw, entity_map, entity_display, predicate_map, predicate_display)
    sections = list(dict.fromkeys(section_name(c['h1']) for c in chunks))
    add_layout(nodes, edges, sections)

    failures, notes, coverage, uncovered = acceptance(nodes, edges, literals, sentence, chunks, entity_map, known)
    write_review(out_dir, entity_groups, entity_rejected, predicate_groups, predicate_rejected, self_loops, notes, coverage,
                 uncovered, nodes, edges, sentence)
    hub_subject = sum(e['s'] == nodes[0]['id'] for e in edges) + sum(r['s'] == nodes[0]['id'] for r in literals)
    n_sentences = sum(b for *_, b in coverage)
    stats = {
        'chunks': len(chunks), 'sentences': n_sentences,
        'sentences_cited': n_sentences - len(uncovered),
        'entities_extracted': sum(len(r['entities']) for r in raw.values()),
        'entities_before': len(known), 'entities_after': len(nodes),
        'predicates_before': len(known_predicates), 'predicates_after': len({e['p'] for e in edges} | {r['p'] for r in literals}),
        'edges_before': edges_before, 'edges_after': len(edges), 'literals': len(literals),
        'literals_numeric': sum(bool(re.search(r'\d', r['v'])) for r in literals),
        'with_place': sum(any(ob['place'] is not None for ob in r['obs']) for r in edges + literals),
        'self_loops_removed': len(self_loops), 'merges_rejected': len(entity_rejected) + len(predicate_rejected),
        'hub': nodes[0]['name'], 'hub_subject_share': round(hub_subject / (len(edges) + len(literals)), 3),
    }
    relation_count = Counter(e['p'] for e in edges)
    stats['entity_dedup_ratio'] = round(stats['entities_after'] / stats['entities_before'], 3)
    stats['concepts'] = sum(n['type'] == '概念' for n in nodes)
    stats['entities_deg_le1'] = sum(n['deg'] + n['ctx'] <= 1 for n in nodes)  # 最多只有一条关系（含作为 place）的实体
    stats['relation_kinds'], stats['attribute_kinds'] = len(relation_count), len({r['p'] for r in literals})
    stats['relation_kinds_used_once'] = sum(v == 1 for v in relation_count.values())
    stats['relation_reuse'] = round(len(edges) / len(relation_count), 2)  # KGGen §6.5 的指标：边数 / 关系种类数
    write_json(out_dir / 'stats.json', stats)
    kg = {'stats': stats, 'sections': sections, 'groups': GROUP_ORDER, 'nodes': nodes, 'edges': edges, 'literals': literals,
          'sentences': {sid: [s['page'], s['text'], s['section']] for sid, s in sentence.items()}}
    text = json.dumps(kg, ensure_ascii=False)
    write_json(out_dir / 'kg.json', kg)
    payload = json.dumps(kg, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/')
    html = HTML_TEMPLATE.replace('/*KG_DATA*/null', payload)
    out = out_dir / HTML_NAME
    out.write_text(html, encoding='utf-8')

    terms = leak_terms(pdf_path)
    assert_no_leak(text, terms, 'kg.json')
    assert_no_leak(payload, terms, HTML_NAME)  # 只查嵌入的数据，模板里的 CSS @media 不算
    if re.search(r'https?://', html):
        failures.append('HTML 中存在外部链接')

    print(json.dumps(stats, ensure_ascii=False, indent=1))
    low = [c for c in coverage if c[3] >= 5 and c[2] / c[3] < MIN_COVERAGE]
    if low:
        print(f'提示：覆盖率低于 {MIN_COVERAGE:.0%} 的块：', low)
    if stats['hub_subject_share'] > MAX_HUB_SHARE:
        print(f'提示：{stats["hub"]} 作主语占比 {stats["hub_subject_share"]:.0%}，超过 {MAX_HUB_SHARE:.0%}')
    print(f'需复核：标为事实但含预测词 {len(notes)} 条，被拒合并 {stats["merges_rejected"]} 项，见 review.md')
    print(f'HTML 已保存到 {out}（{out.stat().st_size / 1e6:.2f} MB）')
    if failures:
        for f in failures[:40]:
            print('未通过：', f)
        raise SystemExit(f'验收未通过：{len(failures)} 项')
    print('硬性检查全部通过')


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>比亚迪研报知识图谱</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10); --wash: rgba(42,120,214,0.08); --accent: #2a78d6;
    --c1: #2a78d6; --c2: #eb6834; --c3: #1baf7a; --c4: #eda100; --c5: #e87ba4; --c6: #008300; --c7: #6250d6; --c8: #898781;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink-1: #f0efec; --ink-2: #c3c2b7; --ink-muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(57,135,229,0.12); --accent: #3987e5;
      --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #898781;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink-1: #f0efec; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10); --wash: rgba(57,135,229,0.12); --accent: #3987e5;
    --c1: #3987e5; --c2: #d95926; --c3: #199e70; --c4: #c98500; --c5: #d55181; --c6: #008300; --c7: #9085e9; --c8: #898781;
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink-1);
         font: 14px/1.6 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  main { padding: 16px 16px 32px; max-width: 1680px; margin: 0 auto; }
  h1 { font-size: 22px; margin: 0; font-weight: 600; }
  .lead { margin: 2px 0 12px; color: var(--ink-2); }
  .stats { display: flex; flex-wrap: wrap; gap: 6px 28px; margin: 0 0 12px; }
  .stats div { display: flex; align-items: baseline; gap: 6px; }
  .stats dt { color: var(--ink-2); font-size: 13px; }
  .stats dd { margin: 0; font-size: 18px; font-weight: 600; }
  .stats small { font-size: 12px; color: var(--ink-muted); font-weight: 400; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 8px; }
  .controls input[type=search], .controls select { padding: 6px 10px; border: 1px solid var(--axis); border-radius: 8px;
    background: var(--surface); color: var(--ink-1); font: inherit; }
  .controls input[type=search] { flex: 1 1 200px; max-width: 280px; min-width: 0; }
  button { font: inherit; color: var(--ink-1); cursor: pointer; }
  .chip, .tool { padding: 5px 10px; border: 1px solid var(--axis); border-radius: 8px; background: var(--surface); }
  .chip { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 13px; }
  .chip svg { display: block; }
  .chip[aria-pressed="true"] { color: var(--ink-1); border-color: var(--ink-muted); }
  .chip[aria-pressed="false"] { opacity: .45; }
  .chip small { color: var(--ink-muted); font-size: 12px; }
  .chip:hover, .tool:hover { background: var(--wash); }
  .check { display: inline-flex; align-items: center; gap: 6px; color: var(--ink-2); font-size: 13px; cursor: pointer; }
  .legend-note { display: inline-flex; align-items: center; gap: 12px; color: var(--ink-2); font-size: 13px; }
  .legend-note span { display: inline-flex; align-items: center; gap: 6px; }
  .key { width: 22px; height: 0; border-top: 2px solid var(--ink-muted); }
  .key.dash { border-top-style: dashed; }
  .count { color: var(--ink-muted); font-size: 13px; }
  .work { display: grid; grid-template-columns: minmax(0, 1fr) 380px; gap: 12px; }
  .stage, .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }
  .stage { position: relative; height: max(480px, calc(100vh - 208px)); overflow: hidden; }
  canvas { display: block; width: 100%; height: 100%; touch-action: none; cursor: grab; }
  canvas.grabbing { cursor: grabbing; } canvas.pointing { cursor: pointer; }
  .zoom { position: absolute; right: 10px; bottom: 10px; display: flex; gap: 4px; }
  .zoom button { min-width: 32px; padding: 4px 8px; border: 1px solid var(--axis); border-radius: 8px; background: var(--surface); }
  .zoom button:hover { background: var(--wash); }
  .tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
         border-radius: 8px; padding: 6px 10px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,.14);
         max-width: 300px; display: none; color: var(--ink-2); }
  .tip b { color: var(--ink-1); font-weight: 600; }
  .panel { height: max(480px, calc(100vh - 208px)); overflow-y: auto; padding: 14px 16px; }
  .panel h2 { font-size: 18px; font-weight: 600; margin: 0; }
  .panel h3 { font-size: 13px; font-weight: 600; color: var(--ink-2); margin: 16px 0 4px; }
  .panel p { margin: 0 0 6px; color: var(--ink-2); }
  .meta { color: var(--ink-muted); font-size: 13px; }
  .rel { border-top: 1px solid var(--grid); padding: 4px 0; }
  .rel summary { cursor: pointer; }
  .pred { color: var(--ink-2); }
  .link { border: 0; background: none; padding: 0; color: var(--accent); text-align: left; }
  .link:hover { text-decoration: underline; }
  .tag { font-size: 12px; color: var(--ink-muted); margin-left: 4px; }
  .ev { font-size: 13px; color: var(--ink-2); margin: 4px 0 4px 14px !important; }
  .pg { color: var(--ink-muted); margin-right: 6px; font-size: 12px; }
  .top { display: flex; flex-wrap: wrap; gap: 6px 12px; }
  details.table { margin-top: 12px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; padding: 10px 16px; }
  details.table > summary { cursor: pointer; font-weight: 600; }
  .scroll { overflow-x: auto; margin-top: 8px; }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 5px 8px; border-top: 1px solid var(--grid); text-align: left; vertical-align: top; }
  th { color: var(--ink-2); font-weight: 500; white-space: nowrap; }
  td.num { font-variant-numeric: tabular-nums; white-space: nowrap; }
  .note { margin: 12px 0 0; font-size: 12px; color: var(--ink-muted); }
  @media (max-width: 860px) {
    .work { grid-template-columns: minmax(0, 1fr); }
    .stage { height: 62vh; min-height: 360px; }
    .panel { height: auto; max-height: none; }
  }
</style>
</head>
<body>
<main>
  <h1>比亚迪研报知识图谱</h1>
  <p class="lead">中金公司 2026-08-07 研报《全球化拓展增长边界，技术底座构筑发展基石》的正文，按 KGGen 框架抽取并去重。不含表格和图片型图表。数值和状态等属性值、陈述所在的地区，显示在实体详情里。</p>
  <dl class="stats" id="stats"></dl>
  <div class="controls">
    <input type="search" id="search" placeholder="搜索实体或别名，回车定位" aria-label="搜索实体或别名">
    <select id="section" aria-label="按章节筛选"></select>
    <label class="check"><input type="checkbox" id="core">只看关系≥2 的实体</label>
    <label class="check"><input type="checkbox" id="literal">显示属性值节点</label>
    <span class="count" id="count"></span>
  </div>
  <div class="controls">
    <span id="chips" style="display:contents"></span>
    <span class="legend-note"><span><i class="key"></i>含事实</span><span><i class="key dash"></i>仅预测或观点</span></span>
  </div>
  <div class="work">
    <div class="stage" id="stage">
      <canvas id="canvas" role="img" aria-label="知识图谱网络图，实体为节点，关系为边。下方表格提供同样的内容。"></canvas>
      <div class="tip" id="tip"></div>
      <div class="zoom">
        <button type="button" id="zoom-in" aria-label="放大">+</button>
        <button type="button" id="zoom-out" aria-label="缩小">−</button>
        <button type="button" id="fit">复位</button>
      </div>
    </div>
    <aside class="panel" id="panel" aria-live="polite"></aside>
  </div>
  <details class="table" id="table-box">
    <summary>查看全部关系（表格，随上方筛选变化）</summary>
    <div class="scroll"><table>
      <thead><tr><th>主语</th><th>谓词</th><th>宾语</th><th>地区</th><th>时间</th><th>立场</th><th>页码</th></tr></thead>
      <tbody id="rows"></tbody>
    </table></div>
  </details>
  <p class="note">本页含研报原文摘录，仅供内部使用，请勿传阅。图谱由大模型抽取，使用前请对照证据句和页码核实。</p>
</main>

<script>
const KG = /*KG_DATA*/null;
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"']/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));

// ---------- 数据 ----------
const NODES = KG.nodes, EDGES = KG.edges, LITERALS = KG.literals, HUB = NODES[0].id;
const GROUP_STYLE = {'公司': ['--c1', 'circle'], '品牌': ['--c2', 'square'], '车型': ['--c3', 'diamond'], '技术': ['--c4', 'triangle'],
                     '地区': ['--c5', 'hexagon'], '政策': ['--c6', 'invert'], '机构/人物': ['--c7', 'pentagon'], '概念': ['--c8', 'dot']};
const POLY = {square: [4, Math.PI / 4, 1.2], diamond: [4, -Math.PI / 2, 1.3], triangle: [3, -Math.PI / 2, 1.35], invert: [3, Math.PI / 2, 1.35],
              hexagon: [6, 0, 1.12], pentagon: [5, -Math.PI / 2, 1.18]};
NODES.forEach(n => { n.r = Math.min(26, 3.5 + 2 * Math.sqrt(n.deg)); n.search = [n.name, ...n.aliases].join(' ').toLowerCase(); });
const radius = n => Math.max(3, n.r * Math.min(1, Math.max(.55, view.k / .45)));  // 全局视图下缩小节点，减少重叠
const LIT_BY = new Map(), CTX_BY = new Map();  // CTX_BY：以该地区为 place 的关系和属性
LITERALS.forEach(l => { if (!LIT_BY.has(l.s)) LIT_BY.set(l.s, []); LIT_BY.get(l.s).push(l); });
[...EDGES, ...LITERALS].forEach(row => new Set(row.obs.map(o => o.place).filter(p => p !== null)).forEach(p => {
  if (!CTX_BY.has(p)) CTX_BY.set(p, []); CTX_BY.get(p).push(row);
}));
const placesOf = obs => [...new Set(obs.map(o => o.place).filter(p => p !== null))].map(p => NODES[p].name);

const state = {groups: new Set(KG.groups), section: '', showLit: false, coreOnly: false, query: '', sel: null, hover: null};
const view = {k: 1, x: 0, y: 0};
let visNodes = [], visEdges = [], leaves = [], adj = new Map(), W = 0, H = 0, colors = {}, matches = null, userMoved = false;

// ---------- 主题 ----------
function readTheme() {
  const cs = getComputedStyle(document.documentElement);
  for (const v of ['--surface', '--ink-1', '--ink-2', '--ink-muted', '--axis', '--c1', '--c2', '--c3', '--c4', '--c5', '--c6', '--c7', '--c8'])
    colors[v] = cs.getPropertyValue(v).trim();
  schedule();
}
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readTheme);
new MutationObserver(readTheme).observe(document.documentElement, {attributes: true, attributeFilter: ['data-theme']});

// ---------- 筛选 ----------
function recompute() {
  const sec = state.section;
  visNodes = NODES.filter(n => state.groups.has(n.group) && (!sec || n.sections.includes(sec)) && (!state.coreOnly || n.deg >= 2));
  const ids = new Set(visNodes.map(n => n.id));
  visEdges = EDGES.filter(e => ids.has(e.s) && ids.has(e.t) && (!sec || e.sections.includes(sec)));
  adj = new Map();
  for (const e of visEdges) for (const id of [e.s, e.t]) { if (!adj.has(id)) adj.set(id, []); adj.get(id).push(e); }
  leaves = [];
  if (state.showLit) for (const n of visNodes) {
    const items = (LIT_BY.get(n.id) || []).filter(l => !sec || l.sections.includes(sec));
    items.forEach((lit, i) => {
      const r = n.r + 16 + 8 * Math.sqrt(i), a = i * 2.39996;
      leaves.push({parent: n, lit, dx: Math.cos(a) * r, dy: Math.sin(a) * r});
    });
  }
  const q = state.query;
  matches = q ? new Set(visNodes.filter(n => n.search.includes(q)).map(n => n.id)) : null;
  const shown = visEdges.length, lits = visNodes.reduce((t, n) => t + (LIT_BY.get(n.id) || []).filter(l => !sec || l.sections.includes(sec)).length, 0);
  $('count').textContent = (q ? `匹配 ${matches.size} 个 · ` : '') + `当前 ${visNodes.length} 个实体、${shown} 条关系、${lits} 个属性值`;
  if ($('table-box').open) renderTable();
  schedule();
}

function resize() {
  const box = $('stage').getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  W = box.width; H = box.height;
  $('canvas').width = Math.round(W * dpr); $('canvas').height = Math.round(H * dpr);
  if (!userMoved) fit();  // 脚本运行时画布可能还没有尺寸，拿到真实尺寸后再自适应
  schedule();
}

function fit() {
  userMoved = false;
  if (!visNodes.length || W <= 0 || H <= 0) return;
  const xs = visNodes.map(n => n.x), ys = visNodes.map(n => n.y);
  const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
  view.k = Math.min((W - 90) / Math.max(x1 - x0, 1), (H - 90) / Math.max(y1 - y0, 1), 2.2);
  view.x = W / 2 - view.k * (x0 + x1) / 2; view.y = H / 2 - view.k * (y0 + y1) / 2;
  schedule();
}

// ---------- 绘制 ----------
const sx = n => n.x * view.k + view.x, sy = n => n.y * view.k + view.y;
let pending = false;
function schedule() { if (!pending) { pending = true; requestAnimationFrame(draw); } }

function shape(ctx, kind, x, y, r) {
  ctx.beginPath();
  if (kind === 'circle' || kind === 'dot') { ctx.arc(x, y, kind === 'dot' ? r * .85 : r, 0, 7); return; }
  const [m, rot, scale] = POLY[kind];
  for (let i = 0; i < m; i++) {
    const a = rot + i * 2 * Math.PI / m, px = x + Math.cos(a) * r * scale, py = y + Math.sin(a) * r * scale;
    i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
  }
  ctx.closePath();
}

function focusIds() {
  const a = state.sel || state.hover;
  if (a && a.kind === 'node') {
    const s = new Set([a.n.id]);
    (adj.get(a.n.id) || []).forEach(e => { s.add(e.s); s.add(e.t); });
    return s;
  }
  if (a && a.kind === 'edge') return new Set([a.e.s, a.e.t]);
  return matches;
}
function isHot(e) {
  const a = state.sel || state.hover;
  if (!a) return false;
  if (a.kind === 'node') return e.s === a.n.id || e.t === a.n.id;
  return a.kind === 'edge' && a.e === e;
}

function draw() {
  pending = false;
  const c = $('canvas'), ctx = c.getContext('2d'), dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.lineCap = 'round';
  const focus = focusIds(), rects = [];
  const hot = [];
  for (const e of visEdges) {
    if (isHot(e)) { hot.push(e); continue; }
    const hubEdge = e.s === HUB || e.t === HUB;
    ctx.globalAlpha = focus ? .05 : hubEdge ? .10 : .55;
    ctx.strokeStyle = colors['--axis']; ctx.lineWidth = 1;
    ctx.setLineDash(e.fact ? [] : [4, 4]);
    ctx.beginPath(); ctx.moveTo(sx(NODES[e.s]), sy(NODES[e.s])); ctx.lineTo(sx(NODES[e.t]), sy(NODES[e.t])); ctx.stroke();
  }
  ctx.setLineDash([]); ctx.globalAlpha = 1;
  for (const e of hot) drawHotEdge(ctx, e);

  for (const f of leaves) {
    const x = sx(f.parent), y = sy(f.parent), on = !focus || focus.has(f.parent.id);
    ctx.globalAlpha = on ? .8 : .08;
    ctx.strokeStyle = colors['--axis']; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + f.dx, y + f.dy); ctx.stroke();
    ctx.fillStyle = colors['--ink-muted']; ctx.beginPath(); ctx.arc(x + f.dx, y + f.dy, 3, 0, 7); ctx.fill();
  }

  const ordered = [...visNodes].sort((a, b) => a.deg - b.deg);
  for (const n of ordered) {
    const [col, kind] = GROUP_STYLE[n.group], x = sx(n), y = sy(n);
    ctx.globalAlpha = !focus || focus.has(n.id) ? 1 : .14;
    const r = radius(n);
    shape(ctx, kind, x, y, r);
    ctx.fillStyle = colors[col]; ctx.fill();
    ctx.lineWidth = r < 5 ? 1 : 2; ctx.strokeStyle = colors['--surface']; ctx.stroke();
    if (state.sel && state.sel.kind === 'node' && state.sel.n === n) {
      shape(ctx, kind, x, y, r + 3); ctx.lineWidth = 2; ctx.strokeStyle = colors['--ink-1']; ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  drawLabels(ctx, focus, rects);
  drawHotLabels(ctx, hot, rects);
}

function drawHotEdge(ctx, e) {
  const a = NODES[e.s], b = NODES[e.t], x0 = sx(a), y0 = sy(a), x1 = sx(b), y1 = sy(b);
  const len = Math.hypot(x1 - x0, y1 - y0) || 1, ux = (x1 - x0) / len, uy = (y1 - y0) / len;
  const ex = x1 - ux * (radius(b) + 4), ey = y1 - uy * (radius(b) + 4);
  ctx.strokeStyle = colors['--ink-2']; ctx.fillStyle = colors['--ink-2']; ctx.lineWidth = 1.6;
  ctx.setLineDash(e.fact ? [] : [5, 4]);
  ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(ex, ey); ctx.stroke();
  ctx.setLineDash([]);
  if (!e.sym) {
    ctx.beginPath(); ctx.moveTo(ex, ey);
    ctx.lineTo(ex - ux * 8 - uy * 4, ey - uy * 8 + ux * 4); ctx.lineTo(ex - ux * 8 + uy * 4, ey - uy * 8 - ux * 4);
    ctx.closePath(); ctx.fill();
  }
}

function placeText(ctx, text, x, y, rects, color, weight) {
  ctx.font = `${weight} 12px system-ui, "Microsoft YaHei", sans-serif`;
  const w = ctx.measureText(text).width, box = [x - w / 2 - 2, y - 8, x + w / 2 + 2, y + 8];
  if (box[2] < 0 || box[0] > W || box[3] < 0 || box[1] > H) return;
  if (rects.some(r => box[0] < r[2] && box[2] > r[0] && box[1] < r[3] && box[3] > r[1])) return;
  rects.push(box);
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.lineJoin = 'round';
  ctx.strokeStyle = colors['--surface']; ctx.lineWidth = 3; ctx.strokeText(text, x, y);
  ctx.fillStyle = color; ctx.fillText(text, x, y);
}

function drawLabels(ctx, focus, rects) {
  const rank = n => (focus && focus.has(n.id) ? 1e6 : 0) + n.deg;
  const list = [...visNodes].sort((a, b) => rank(b) - rank(a));
  const budget = focus ? 90 : Math.round(45 * Math.max(view.k * view.k * 2, 1));
  let drawn = 0;
  for (const n of list) {
    if (drawn >= budget || (focus && !focus.has(n.id))) break;
    const before = rects.length;
    placeText(ctx, n.name, sx(n), sy(n) + radius(n) + 11, rects, colors['--ink-1'], n.id === HUB ? 600 : 400);
    if (rects.length > before) drawn++;
  }
  const sel = state.sel || state.hover;
  if (sel && sel.kind === 'node') for (const f of leaves) if (f.parent === sel.n)
    placeText(ctx, f.lit.v, sx(f.parent) + f.dx, sy(f.parent) + f.dy + 11, rects, colors['--ink-2'], 400);
}

function drawHotLabels(ctx, hot, rects) {
  if (hot.length > 40) return;
  for (const e of hot) {
    const a = NODES[e.s], b = NODES[e.t];
    placeText(ctx, e.p, (sx(a) + sx(b)) / 2, (sy(a) + sy(b)) / 2, rects, colors['--ink-2'], 400);
  }
}

// ---------- 命中测试和交互 ----------
function hit(px, py) {
  let best = null, dist = 1e9;
  for (const n of visNodes) {
    const d = Math.hypot(sx(n) - px, sy(n) - py);
    if (d <= radius(n) + 6 && d < dist) { best = {kind: 'node', n}; dist = d; }
  }
  if (best) return best;
  for (const f of leaves) {
    const d = Math.hypot(sx(f.parent) + f.dx - px, sy(f.parent) + f.dy - py);
    if (d <= 8 && d < dist) { best = {kind: 'leaf', f}; dist = d; }
  }
  if (best) return best;
  const sel = state.sel;
  for (const e of visEdges) {
    if (sel ? !isHot(e) : (e.s === HUB || e.t === HUB)) continue;
    const a = NODES[e.s], b = NODES[e.t], x0 = sx(a), y0 = sy(a), dx = sx(b) - x0, dy = sy(b) - y0;
    const t = Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / (dx * dx + dy * dy || 1)));
    const d = Math.hypot(x0 + t * dx - px, y0 + t * dy - py);
    if (d <= 5 && d < dist) { best = {kind: 'edge', e}; dist = d; }
  }
  return best;
}

function showTip(item, px, py) {
  const tip = $('tip');
  if (!item) { tip.style.display = 'none'; return; }
  if (item.kind === 'node') tip.innerHTML = `<b>${esc(item.n.name)}</b><div>${esc(item.n.type)} · ${item.n.deg} 条关系${item.n.ctx ? ` · ${item.n.ctx} 条陈述在此地区` : ''}</div>`;
  else if (item.kind === 'edge') tip.innerHTML = `<b>${esc(NODES[item.e.s].name)}</b> ${esc(item.e.p)} <b>${esc(NODES[item.e.t].name)}</b>`;
  else tip.innerHTML = `<b>${esc(item.f.parent.name)}</b> ${esc(item.f.lit.p)}：<b>${esc(item.f.lit.v)}</b>`;
  tip.style.display = 'block';
  const w = tip.offsetWidth, h = tip.offsetHeight;
  tip.style.left = Math.min(px + 14, W - w - 8) + 'px';
  tip.style.top = Math.min(py + 14, H - h - 8) + 'px';
}

const pointers = new Map();
let drag = null, pinch = null;
const canvas = $('canvas');
const local = ev => { const r = canvas.getBoundingClientRect(); return [ev.clientX - r.left, ev.clientY - r.top]; };

function zoomAt(px, py, factor) {
  const k = Math.max(.15, Math.min(8, view.k * factor)), f = k / view.k;
  view.x = px - (px - view.x) * f; view.y = py - (py - view.y) * f; view.k = k;
  userMoved = true;
  schedule();
}

canvas.addEventListener('wheel', ev => { ev.preventDefault(); const [x, y] = local(ev); zoomAt(x, y, Math.pow(1.0015, -ev.deltaY)); }, {passive: false});
canvas.addEventListener('pointerdown', ev => {
  canvas.setPointerCapture(ev.pointerId);
  const [x, y] = local(ev);
  pointers.set(ev.pointerId, [x, y]);
  if (pointers.size === 2) {
    const [a, b] = [...pointers.values()];
    pinch = {d: Math.hypot(a[0] - b[0], a[1] - b[1])}; drag = null; return;
  }
  const item = hit(x, y);
  drag = {x, y, moved: 0, node: item && item.kind === 'node' ? item.n : null, item};
  canvas.classList.add('grabbing');
});
canvas.addEventListener('pointermove', ev => {
  const [x, y] = local(ev);
  if (pointers.has(ev.pointerId)) pointers.set(ev.pointerId, [x, y]);
  if (pinch && pointers.size === 2) {
    const [a, b] = [...pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
    zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d / pinch.d); pinch.d = d; return;
  }
  if (drag) {
    const dx = x - drag.x, dy = y - drag.y;
    drag.moved += Math.abs(dx) + Math.abs(dy); drag.x = x; drag.y = y;
    if (drag.node) { drag.node.x += dx / view.k; drag.node.y += dy / view.k; } else { view.x += dx; view.y += dy; userMoved = true; }
    showTip(null); schedule(); return;
  }
  const item = hit(x, y);
  const hover = item && item.kind === 'leaf' ? null : item;
  const same = (hover && state.hover && hover.kind === state.hover.kind && (hover.n || hover.e) === (state.hover.n || state.hover.e)) || (!hover && !state.hover);
  if (!same) { state.hover = hover; schedule(); }
  canvas.classList.toggle('pointing', !!item);
  showTip(item, x, y);
});
function release(ev) {
  pointers.delete(ev.pointerId);
  if (pointers.size < 2) pinch = null;
  canvas.classList.remove('grabbing');
  if (drag && drag.moved < 5) {
    const item = drag.item;
    if (!item) select(null);
    else if (item.kind === 'leaf') select({kind: 'node', n: item.f.parent});
    else select(item);
  }
  drag = null;
}
canvas.addEventListener('pointerup', release);
canvas.addEventListener('pointercancel', release);
canvas.addEventListener('pointerleave', () => { if (!drag) { state.hover = null; showTip(null); schedule(); } });

function select(item, center) {
  state.sel = item;
  renderPanel();
  if (center && item && item.kind === 'node') {
    view.k = Math.max(view.k, 1.2);
    view.x = W / 2 - item.n.x * view.k; view.y = H / 2 - item.n.y * view.k;
  }
  schedule();
}

// ---------- 侧栏 ----------
function evidence(obs) {
  const sids = [...new Set(obs.flatMap(o => o.sids))];
  return sids.map(s => `<p class="ev"><span class="pg">p.${KG.sentences[s][0]}</span>${esc(KG.sentences[s][1])}</p>`).join('');
}
function tags(obs) {
  const times = [...new Set(obs.map(o => o.time).filter(Boolean))], stances = [...new Set(obs.map(o => o.stance))].filter(s => s !== '事实');
  return [...placesOf(obs), ...times, ...stances].map(t => `<span class="tag">${esc(t)}</span>`).join('');
}
const nodeLink = n => `<button type="button" class="link" data-node="${n.id}">${esc(n.name)}</button>`;
function contextRow(row) {
  const tail = 't' in row ? nodeLink(NODES[row.t]) : `<b>${esc(row.v)}</b>`;
  return `<details class="rel"><summary>${nodeLink(NODES[row.s])} <span class="pred">${esc(row.p)}</span> ${tail}${tags(row.obs)}</summary>${evidence(row.obs)}</details>`;
}
function literalRow(l) {
  return `<details class="rel"><summary><span class="pred">${esc(l.p)}</span> <b>${esc(l.v)}</b>${tags(l.obs)}</summary>${evidence(l.obs)}</details>`;
}
function edgeRow(e, id) {
  const out = e.s === id, other = NODES[out ? e.t : e.s];
  const link = `<button type="button" class="link" data-node="${other.id}">${esc(other.name)}</button>`;
  const body = out ? `<span class="pred">${esc(e.p)}</span> ${link}` : `${link} <span class="pred">${esc(e.p)}</span>`;
  return `<details class="rel"><summary>${body}${tags(e.obs)}</summary>${evidence(e.obs)}</details>`;
}
function renderPanel() {
  const panel = $('panel'), a = state.sel;
  panel.scrollTop = 0;
  if (a && a.kind === 'node') {
    const n = a.n, edges = EDGES.filter(e => e.s === n.id || e.t === n.id), out = edges.filter(e => e.s === n.id), inn = edges.filter(e => e.s !== n.id);
    const lits = LIT_BY.get(n.id) || [], ctx = CTX_BY.get(n.id) || [];
    panel.innerHTML = `<h2>${esc(n.name)}</h2><p class="meta">${esc(n.type)} · ${edges.length} 条关系 · ${lits.length} 个属性值${ctx.length ? ` · ${ctx.length} 条陈述在此地区` : ''} · 提及 ${n.mentions} 次</p>`
      + (n.aliases.length ? `<h3>别名</h3><p>${n.aliases.map(esc).join('、')}</p>` : '')
      + (lits.length ? `<h3>属性值（${lits.length}）</h3>${lits.map(literalRow).join('')}` : '')
      + (out.length ? `<h3>作为主语（${out.length}）</h3>${out.map(e => edgeRow(e, n.id)).join('')}` : '')
      + (inn.length ? `<h3>作为宾语（${inn.length}）</h3>${inn.map(e => edgeRow(e, n.id)).join('')}` : '')
      + (ctx.length ? `<h3>在该地区的陈述（${ctx.length}）</h3>${ctx.map(contextRow).join('')}` : '');
  } else if (a && a.kind === 'edge') {
    const e = a.e, s = NODES[e.s], t = NODES[e.t];
    panel.innerHTML = `<h2>${esc(e.p)}</h2><p><button type="button" class="link" data-node="${s.id}">${esc(s.name)}</button> → <button type="button" class="link" data-node="${t.id}">${esc(t.name)}</button>${tags(e.obs)}</p>`
      + `<h3>原谓词</h3><p>${[...new Set(e.obs.map(o => o.raw))].map(esc).join('、')}</p><h3>证据</h3>${evidence(e.obs)}`;
  } else {
    const top = NODES.slice(0, 16).map(n => `<button type="button" class="link" data-node="${n.id}">${esc(n.name)}</button><span class="tag">${n.deg}</span>`).join('</span><span>');
    panel.innerHTML = `<h2>怎么看这张图</h2><p>节点是实体，颜色和形状表示类型，大小表示关系数量。点击节点可以查看别名、属性值（数值和状态）、关系和原文证据；地区节点还列出发生在该地区的陈述。</p>`
      + `<p>滚轮缩放，拖动空白处平移，可以拖动节点。与中心实体“${esc(NODES[HUB].name)}”的连线默认调淡，点击它可以查看。</p>`
      + `<h3>关系最多的实体</h3><div class="top"><span>${top}</span></div>`;
  }
}
$('panel').addEventListener('click', ev => {
  const b = ev.target.closest('[data-node]');
  if (b) { ev.preventDefault(); select({kind: 'node', n: NODES[+b.dataset.node]}, true); }
});

// ---------- 表格 ----------
function renderTable() {
  const sec = state.section, ids = new Set(visNodes.map(n => n.id)), pages = obs => [...new Set(obs.flatMap(o => o.sids.map(s => KG.sentences[s][0])))].join(', ');
  const row = (a, p, b, obs) => `<tr><td>${esc(a)}</td><td>${esc(p)}</td><td>${esc(b)}</td><td>${esc(placesOf(obs).join('、'))}</td><td>${esc([...new Set(obs.map(o => o.time).filter(Boolean))].join(', '))}</td><td>${esc([...new Set(obs.map(o => o.stance))].join(', '))}</td><td class="num">${pages(obs)}</td></tr>`;
  $('rows').innerHTML = visEdges.map(e => row(NODES[e.s].name, e.p, NODES[e.t].name, e.obs)).join('')
    + LITERALS.filter(l => ids.has(l.s) && (!sec || l.sections.includes(sec))).map(l => row(NODES[l.s].name, l.p, l.v, l.obs)).join('');
}
$('table-box').addEventListener('toggle', () => { if ($('table-box').open) renderTable(); });

// ---------- 控件 ----------
const S = KG.stats;
$('stats').innerHTML = [['实体', `${S.entities_after}`, `去重前 ${S.entities_before}`], ['谓词', `${S.predicates_after}`, `去重前 ${S.predicates_before}`],
  ['关系', `${S.edges_after}`, `去重前 ${S.edges_before}`], ['属性值', `${S.literals}`, `其中数值 ${S.literals_numeric}`], ['正文句覆盖', `${S.sentences_cited}/${S.sentences}`, '']]
  .map(([k, v, m]) => `<div><dt>${k}</dt><dd>${v}</dd>${m ? `<small>${m}</small>` : ''}</div>`).join('');

function iconPath(kind) {
  const r = 5.5;
  if (kind === 'circle' || kind === 'dot') return `<circle r="${kind === 'dot' ? 4.6 : r}"/>`;
  const [m, rot, scale] = POLY[kind];
  const pts = Array.from({length: m}, (_, i) => { const a = rot + i * 2 * Math.PI / m; return `${(Math.cos(a) * r * scale).toFixed(1)},${(Math.sin(a) * r * scale).toFixed(1)}`; });
  return `<polygon points="${pts.join(' ')}"/>`;
}
const groupCount = g => NODES.filter(n => n.group === g).length;
$('chips').innerHTML = KG.groups.filter(groupCount).map(g => `<button type="button" class="chip" data-group="${esc(g)}" aria-pressed="true">
  <svg width="14" height="14" viewBox="-7.5 -7.5 15 15" fill="var(${GROUP_STYLE[g][0]})" aria-hidden="true">${iconPath(GROUP_STYLE[g][1])}</svg>${esc(g)}<small>${groupCount(g)}</small></button>`).join('');
$('chips').addEventListener('click', ev => {
  const b = ev.target.closest('[data-group]');
  if (!b) return;
  const g = b.dataset.group, on = !state.groups.has(g);
  on ? state.groups.add(g) : state.groups.delete(g);
  b.setAttribute('aria-pressed', on);
  clearHidden(); recompute();
});
function clearHidden() {
  const a = state.sel;
  if ((a && a.kind === 'node' && !state.groups.has(a.n.group)) || (a && a.kind === 'edge' && !(state.groups.has(NODES[a.e.s].group) && state.groups.has(NODES[a.e.t].group)))) select(null);
}
$('section').innerHTML = `<option value="">全部章节</option>` + KG.sections.map(s => `<option>${esc(s)}</option>`).join('');
$('section').addEventListener('change', ev => {
  state.section = ev.target.value;
  const a = state.sel;
  if (a && a.kind === 'node' && state.section && !a.n.sections.includes(state.section)) select(null);
  recompute(); fit();
});
$('literal').addEventListener('change', ev => { state.showLit = ev.target.checked; recompute(); });
$('core').addEventListener('change', ev => {
  state.coreOnly = ev.target.checked;
  const a = state.sel;
  if (state.coreOnly && a && ((a.kind === 'node' && a.n.deg < 2) || (a.kind === 'edge' && (NODES[a.e.s].deg < 2 || NODES[a.e.t].deg < 2)))) select(null);
  recompute();
});
$('search').addEventListener('input', ev => { state.query = ev.target.value.trim().toLowerCase(); recompute(); });
$('search').addEventListener('keydown', ev => {
  if (ev.key === 'Enter' && matches && matches.size) {
    select({kind: 'node', n: visNodes.filter(n => matches.has(n.id)).sort((a, b) => b.deg - a.deg)[0]}, true);
  }
});
document.addEventListener('keydown', ev => { if (ev.key === 'Escape') select(null); });
$('zoom-in').addEventListener('click', () => zoomAt(W / 2, H / 2, 1.3));
$('zoom-out').addEventListener('click', () => zoomAt(W / 2, H / 2, 1 / 1.3));
$('fit').addEventListener('click', fit);

new ResizeObserver(() => { resize(); }).observe($('stage'));
readTheme(); resize(); recompute(); fit(); renderPanel();
</script>
</body>
</html>
'''
