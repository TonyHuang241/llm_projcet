"""第 1、2 步：概念分层填表、原文连边，以及整理概念表（并入新概念、整理包含关系）、悬挂补全、并列复核。"""
import copy
import re
from collections import Counter, defaultdict

import numpy as np

from common.llm_client import log
from research_report import fin_items
from research_report.logic.base import (ACCOUNTING_TEXT, CONCEPT_LAYERS, FIXED_CUES, KINDS, LAYER_NAMES, MAX_DEF_LEN,
                                        MAX_EVIDENCE_LEN, MAX_NODE_LEN, ROOT, Step, as_bool, is_null, layer_of, name_key,
                                        norm_direction, norm_name, para_block, parse_decisions, parse_json, quote_in,
                                        resolve_name, short, squash, table_index, table_text, to_int)

TABLE_CHUNK = 3000  # 分层填表时每块原文的字数
PREV_SENTENCES, PREV_CHARS = 2, 150  # 逐段抽取时附带的前一段末尾：句数、字数上限
SOURCES = ('本段', '小节标题', '前文')
PARENT_RELATIONS = ('加总', '相乘', '细分')  # 包含关系的类型：第 3 层多为加总、相乘，第 4、5 层多为细分
DANGLING_MAX_PARAS = 5  # 悬挂补全时最多附带的段落数
PARALLEL_BATCH = 12  # 并列复核时每次交给大模型的关系数
PARALLEL_RELATIONS = ('因果', '并列', '拿不准')


def tail(text):
    """段落末尾的几句，供下一段理解指代。"""
    parts = [s for s in re.split(r'(?<=[。！？；])', text) if s.strip()]
    return ''.join(parts[-PREV_SENTENCES:])[-PREV_CHARS:]


# ==================== 第 1 步：概念分层填表 ====================

def parse_concept(it, errors):
    """校验一个新概念（填表、整理概念表、逐段抽取共用），合格时返回 {'name', 'definition', 'layer', 'kind'}，否则记下错误返回 None。"""
    name, definition = norm_name(it.get('name')), norm_name(it.get('definition'))
    m = re.search(r'\d', str(it.get('layer') or ''))
    layer, kind = int(m.group()) if m else None, str(it.get('kind') or '').strip()
    if len(name) > MAX_NODE_LEN or not definition:
        errors.append(f'新概念“{name}”需要不超过 {MAX_NODE_LEN} 字的名称和一句定义')
    elif layer not in CONCEPT_LAYERS:
        errors.append(f'新概念“{name}”的 layer 必须是 3、4、5 之一（第 1、2 层是预置的；分部、地区的收入和毛利率放第 3 层）')
    elif layer != 3 and kind not in KINDS:
        errors.append(f'新概念“{name}”的 kind 必须是{"、".join(KINDS)}之一')
    else:
        return {'name': name, 'definition': definition[:MAX_DEF_LEN], 'layer': layer, 'kind': '指标' if layer == 3 else kind}
    return None


def check_table(obj, table, errors):
    """校验填表（或整理概念表）结果中的 new、aliases，返回 (新概念列表, [(别名, 规范名)])，问题记进 errors。
    与已有名称字面相同的新概念视为已有，会计科目不进表。"""
    new, aliases = obj.get('new') or [], obj.get('aliases') or []
    if not isinstance(new, list) or not isinstance(aliases, list):
        raise ValueError('new 和 aliases 必须是数组')
    index, added, out_new, out_alias = table_index(table), {}, [], []
    for it in new:
        if not isinstance(it, dict):
            continue
        name = norm_name(it.get('name'))
        k = name_key(name)
        if not name or fin_items.match(name) or k in index or k in added:
            continue
        v = parse_concept(it, errors)
        if v:
            added[k] = v['name']
            out_new.append(v)
    for it in aliases:
        if not isinstance(it, dict):
            continue
        alias, target = norm_name(it.get('alias')), norm_name(it.get('name'))
        canon = index.get(name_key(target)) or added.get(name_key(target))
        if canon is None:
            if not fin_items.match(target):  # 指向会计科目的别名不需要记
                errors.append(f'aliases 中“{alias}”指向的“{target}”不是表中的规范名')
        elif alias and len(alias) <= MAX_NODE_LEN and not fin_items.match(alias) and name_key(alias) != name_key(canon):
            out_alias.append((alias, canon))
    return out_new, out_alias


def parse_table(content, table):
    errors = []
    res = check_table(parse_json(content, dict), table, errors)
    if errors:
        raise ValueError('；'.join(errors))
    return res


def apply_table(table, new, aliases):
    for t in new:
        table.setdefault(t['name'], {'aliases': [], 'definition': t['definition'], 'layer': t['layer'], 'kind': t['kind'],
                                     'parent': ''})
    index = table_index(table)
    for alias, canon in aliases:
        if canon in table and name_key(alias) not in index:  # 已经属于别的概念的叫法不改
            table[canon]['aliases'].append(alias)
            index[name_key(alias)] = canon


def chunk_paras(ps):
    out, cur, n = [], [], 0
    for p in ps:
        if cur and n + len(p['text']) > TABLE_CHUNK:
            out.append(cur)
            cur, n = [], 0
        cur.append(p)
        n += len(p['text'])
    return out + [cur] if cur else out


def build_tables(ctx, reports):
    """每篇研报按顺序分块通读、滚动更新概念表（块与块之间串行，研报之间并行），结果写进 r.table。"""
    step = Step(ctx, 'table_cache.jsonl', '建表', fields=('new', 'aliases'))
    log(f'概念分层填表：{len(reports)} 篇研报共 {sum(len(chunk_paras(r.paras)) for r in reports)} 块（每块约 {TABLE_CHUNK} 字）')

    def work(r):
        for chunk in chunk_paras(r.paras):
            text, cur = para_block(chunk), table_text(r.table)
            try:
                new, aliases = step.ask(step.key(r.rid, text, cur), lambda c: parse_table(c, r.table),
                                        extra={'report_id': r.rid}, **ctx.about(r.rid), accounting=ACCOUNTING_TEXT,
                                        table=cur, text=text)
            except RuntimeError as e:  # 这一块没建成，后面逐段抽取时出现的新概念还会补进表
                log(f'!!! {r.rid} 有一块填表失败：{e}')
                continue
            apply_table(r.table, new, [tuple(a) for a in aliases])

    ctx.parallel(reports, work, '概念分层填表')
    by_layer = Counter(t['layer'] for r in reports for t in r.table.values())
    log(f'概念分层填表完成：每篇平均 {np.mean([len(r.table) for r in reports]):.0f} 个概念（第 3/4/5 层共 '
        f'{by_layer[3]}/{by_layer[4]}/{by_layer[5]} 个），共 {sum(len(t["aliases"]) for r in reports for t in r.table.values())} 个别名')


def table_by_layer(table):
    """整理概念表时给模型看的概念表：按层列出（规范名 ｜ 别名 ｜ 定义），便于逐层找包含关系。"""
    blocks = [f'第 {k} 层 {LAYER_NAMES[k]}：\n' + '\n'.join(f'{n} ｜ {"、".join(t["aliases"]) or "—"} ｜ {t["definition"]}'
                                                       for n, t in table.items() if t['layer'] == k)
              for k in CONCEPT_LAYERS if any(t['layer'] == k for t in table.values())]
    return '\n\n'.join(blocks) or '（空）'


def merge_candidates(table, cands, new, aliases):
    """把逐段抽取时出现的新概念并入概念表（原地修改）：判为已有概念别名的改用规范名，其余新建（定义、层级、类型以模型修正的为准）。
    返回 {新名称: 规范名}。"""
    defs = {name_key(t['name']): t for t in new}
    alias_of = {name_key(a): c for a, c in aliases}
    rename, add = {}, []
    for v in cands:
        k = name_key(v['name'])
        if k in alias_of and alias_of[k] != v['name']:
            rename[v['name']] = alias_of[k]
        else:
            rename[v['name']] = v['name']
            add.append({**v, **{f: defs[k][f] for f in ('definition', 'layer', 'kind') if k in defs and defs[k][f]}})
    apply_table(table, add, [(v['name'], rename[v['name']]) for v in cands if rename[v['name']] != v['name']])
    return rename


def check_parents(obj, table, errors):
    """校验包含关系（同一层的两个概念），返回 [{'name', 'parent', 'relation', 'reason'}]，问题记进 errors。"""
    items = obj.get('parents') or []
    if not isinstance(items, list):
        errors.append('parents 必须是数组')
        return []
    index, out = table_index(table), []
    for it in items:
        if not isinstance(it, dict):
            continue
        child, parent = index.get(name_key(it.get('name'))), index.get(name_key(it.get('parent')))
        rel = str(it.get('relation') or '').strip()
        if not child or not parent:
            lack = '、'.join(f'“{norm_name(it.get(k))}”' for k, v in (('name', child), ('parent', parent)) if not v)
            errors.append(f'{lack}不在概念表中：包含关系的两端都只能用概念表中的名称（含这次新建的概念），原样书写；'
                          f'规则里举的例子不一定在本篇的表中')
        elif child == parent:
            errors.append(f'“{norm_name(it.get("name"))}”和“{norm_name(it.get("parent"))}”是同一个概念（或它的别名），不能设包含关系')
        elif table[child]['layer'] != table[parent]['layer']:
            errors.append(f'“{child}”（第 {table[child]["layer"]} 层）和“{parent}”（第 {table[parent]["layer"]} 层）不在同一层，不能设包含关系')
        elif rel not in PARENT_RELATIONS:
            errors.append(f'“{child}”的 relation 必须是{"或".join(PARENT_RELATIONS)}')
        else:
            out.append({'name': child, 'parent': parent, 'relation': rel, 'reason': short(it.get('reason'))})
    return out


def parse_organize(content, table, cands, strict=True):
    """校验整理概念表的结果：new、aliases 按填表的规则校验；parents 在并入新概念之后的表上校验。
    strict 为 False 时（重试用完之后）不合格的条目直接丢掉，保留合格的部分。"""
    obj, errors = parse_json(content, dict), []
    new, aliases = check_table(obj, table, errors)
    after = copy.deepcopy(table)
    merge_candidates(after, cands, new, aliases)
    parents = check_parents(obj, after, errors)
    if errors and strict:
        raise ValueError('；'.join(errors))
    return new, aliases, parents


def set_parents(table, parents):
    """把整理出的包含关系写进概念表（已有上位概念的、会成环的跳过），返回写入的条数。"""
    n = 0
    for it in parents:
        child, parent = it['name'], it['parent']
        if child not in table or parent not in table or table[child]['parent']:
            continue
        cur = parent  # 沿上位概念往上走，走回自己就是成环
        while cur and cur != child:
            cur = table[cur]['parent']
        if not cur:
            table[child].update(parent=parent, parent_relation=it['relation'], parent_reason=it['reason'])
            n += 1
    return n


def organize_tables(ctx, reports):
    """第 1 步的最后一环，要等逐段抽取完、表补全后再做：每篇研报调用一次大模型，把逐段抽取时出现的新概念并入概念表
    （是已有概念的记为别名，抽取记录里的名称随之换成规范名），并在第 3-5 层同一层的概念之间整理包含关系（写进表中的 parent）。"""
    step = Step(ctx, 'organize_cache.jsonl', '整理', fields=('new', 'aliases', 'parents'))
    todo = [r for r in reports if r.new_vars or max((Counter(t['layer'] for t in r.table.values())
                                                     + Counter(v['layer'] for v in r.new_vars)).values(), default=0) >= 2]
    log(f'整理概念表：{len(todo)} 篇研报，逐段抽取时出现的新概念共 {sum(len(r.new_vars) for r in todo)} 个')

    def work(r):
        cands = r.new_vars
        text = '\n'.join(f'{i}. {v["name"]} ｜ 第{v["layer"]}层 ｜ {v["kind"]} ｜ {v["definition"]}'
                         for i, v in enumerate(cands, 1)) or '（无）'
        cur = table_by_layer(r.table)
        try:
            new, aliases, parents = step.ask(step.key(r.rid, cur, text), lambda c: parse_organize(c, r.table, cands),
                                             lenient=lambda c: parse_organize(c, r.table, cands, strict=False),
                                             extra={'report_id': r.rid}, **ctx.about(r.rid), accounting=ACCOUNTING_TEXT,
                                             table=cur, candidates=text)
        except RuntimeError as e:
            log(f'!!! {r.rid} 整理概念表失败，新概念全部按新建处理、不设包含关系：{e}')
            new, aliases, parents = [], [], []
        return merge_candidates(r.table, cands, new, [tuple(a) for a in aliases]), set_parents(r.table, parents)

    n_alias = n_parent = 0
    for r, res in zip(todo, ctx.parallel(todo, work, '整理概念表')):
        ren, n = res or ({}, 0)
        n_alias += sum(a != b for a, b in ren.items())
        n_parent += n
        for x in r.rels:
            x['cause'], x['effect'] = ren.get(x['cause'], x['cause']), ren.get(x['effect'], x['effect'])
    for r in reports:
        r.rels = [x for x in r.rels if x['cause'] != x['effect']]
    by_layer = Counter(t['layer'] for r in reports for t in r.table.values() if t['parent'])
    log(f'整理概念表完成：新概念中 {n_alias} 个并入已有概念、其余新建；共设上位概念 {n_parent} 个（第 3/4/5 层 '
        f'{by_layer[3]}/{by_layer[4]}/{by_layer[5]} 个）')


# ==================== 第 2 步：原文连边 ====================

def check_relations(obj, p, table):
    """校验一段的抽取结果，返回 (合格的关系, 用到的新概念, 错误说明)。"""
    rels_in, vars_in = obj.get('relations'), obj.get('new_vars') or []
    if not isinstance(rels_in, list) or not isinstance(vars_in, list):
        return [], [], ['输出必须是 {"relations": [...], "new_vars": [...]}']
    index, new_vars, errors = table_index(table), {}, []
    for v in vars_in:
        if not isinstance(v, dict):
            continue
        name = norm_name(v.get('name'))
        k = name_key(name)
        if not name or k in index or fin_items.match(name):
            continue
        c = parse_concept(v, errors)
        if c:
            new_vars[k] = c

    def resolve(raw):
        if is_null(raw):
            return None, None
        name = resolve_name(raw, index) or (new_vars[name_key(raw)]['name'] if name_key(raw) in new_vars else None)
        if name is None:
            return None, (f'“{norm_name(raw)}”既不在概念表中，也不是会计科目：请改用表中的名称；确属新概念的，'
                          f'在 new_vars 中给出名称、层级、类型和定义')
        return name, None

    rels, seen = [], set()
    for i, r in enumerate(rels_in, 1):
        if not isinstance(r, dict):
            continue
        (cause, e1), (effect, e2) = resolve(r.get('cause')), resolve(r.get('effect'))
        direction, ev = norm_direction(r.get('direction')), norm_name(r.get('evidence'))
        problems = [e for e in (e1, e2) if e]
        if not problems and (cause is None and effect is None or cause == effect or cause == ROOT):
            continue  # 两端都空、自环、从“公司估值”出发的，直接丢弃
        if direction is None:
            problems.append('direction 必须是正向或负向')
        if not quote_in(ev, p['text'], min_len=4):
            problems.append('evidence 在本段中找不到，请从段落中逐字摘录（可以用……省略中间部分）')
        elif effect in FIXED_CUES and not FIXED_CUES[effect][0].search(ev):
            problems.append(f'被影响因素为“{effect}”时，evidence 中必须有{FIXED_CUES[effect][1]}字样；原文没有这样说，就不要连到“{effect}”')
        context = []
        for side, name, label in (('cause', cause, '影响因素'), ('effect', effect, '被影响因素')):
            src = str(r.get(f'{side}_from') or '本段').strip()
            if name is None or src not in SOURCES[1:]:
                continue
            where = (p.get('section') or '') if src == '小节标题' else p.get('prev_tail', '')
            q = squash(r.get('from_quote'))
            if len(q) < 2 or q not in squash(where):
                problems.append(f'{side}_from 为{src}时，from_quote 必须逐字摘自{src}')
            else:
                context.append(f'{label}“{name}”出自{src}：“{norm_name(r.get("from_quote"))}”')
        if problems:
            errors += [f'第{i}条：{x}' for x in problems]
            continue
        if (cause, effect, direction) in seen:
            continue
        seen.add((cause, effect, direction))
        rels.append({'cause': cause, 'direction': direction, 'effect': effect, 'evidence': ev[:MAX_EVIDENCE_LEN], 'context': context})
    used = {n for r in rels for n in (r['cause'], r['effect'])}
    return rels, [v for v in new_vars.values() if v['name'] in used], errors


def extract_relations(ctx, reports):
    """每个段落调用一次大模型，结果写进 r.rels（原文抽取记录）、r.new_vars（表里没有的新概念）、r.rejected（校验不合格重抽次数）。"""
    step = Step(ctx, 'extract_cache.jsonl', '抽取', fields=('relations', 'new_vars', 'rejected'))
    jobs = []
    for r in reports:
        ttext = table_text(r.table)
        for p in r.paras:
            key = step.key(r.rid, str(p['para_id']), p['text'], p.get('section') or '', p['prev_tail'], ttext, ACCOUNTING_TEXT)
            jobs.append((r, p, ttext, key))
    todo = [j for j in jobs if not step.has(j[3])]
    log(f'原文连边：共 {len(jobs)} 段，已缓存 {len(jobs) - len(todo)} 段，本次调用 {len(todo)} 段')

    def work(job):
        r, p, ttext, key = job
        bad = Counter()

        def parse(content):
            rels, new, errors = check_relations(parse_json(content, dict), p, r.table)
            if errors:
                bad['n'] += 1
                raise ValueError('；'.join(errors[:10]))
            return rels, new

        rels, new = step.call(parse, lenient=lambda c: check_relations(parse_json(c, dict), p, r.table)[:2],
                              **ctx.about(r.rid), section=p.get('section') or '无', accounting=ACCOUNTING_TEXT, table=ttext,
                              prev=p['prev_tail'] or '（无）', text=p['text'])
        step.put(key, (rels, new, bad['n']), report_id=r.rid, para_id=p['para_id'])  # 只缓存成功的段落，失败的下次重跑

    ctx.parallel(todo, work, '原文连边')
    failed, new_by = 0, {r.rid: {} for r in reports}
    for r in reports:
        r.rels, r.rejected = [], 0
    for r, p, _, key in jobs:
        res = step.get(key, None)
        if res is None:
            failed += 1
            continue
        rels, new, rejected = res
        r.rels += [{'report_id': r.rid, 'para_id': p['para_id'], **x} for x in rels]
        for v in new:
            new_by[r.rid].setdefault(name_key(v['name']), v)
        r.rejected += rejected
    for r in reports:
        r.new_vars = list(new_by[r.rid].values())
    if failed:
        log(f'!!! {failed} 段抽取失败，这些段落没有关系；重新运行会再试')
    rels = [x for r in reports for x in r.rels]
    n_dangling = sum(x['cause'] is None or x['effect'] is None for x in rels)
    log(f'原文连边完成：{len(rels)} 条关系（其中悬挂记录 {n_dangling} 条），程序校验不合格重抽 {sum(r.rejected for r in reports)} 次')


def classify(r, table):
    """按两端的层级给一条完整的关系归类：edge 进图 / accounting 两端都是会计科目（或会计科目指向估值）/
    to_root 第 3-5 层概念直接指向公司估值 / reverse 层号小的指向层号大的。"""
    lc, le = layer_of(r['cause'], table), layer_of(r['effect'], table)
    if lc == 2 and le <= 2:
        return 'accounting'
    if lc is None or le is None or lc < le:
        return 'reverse'
    return 'to_root' if le == 1 else 'edge'


def note_of(r):
    """悬挂记录或线索的简要信息，存进节点。"""
    return {'para_id': r['para_id'] if 'para_id' in r else r['para_ids'][0], 'direction': r['direction'],
            'evidence': r['evidence'] if isinstance(r['evidence'], str) else r['evidence'][0]}


def rel_row(r):
    return {'cause': r['cause'], 'direction': r['direction'], 'effect': r['effect'], 'evidence': r['evidence'],
            'para_ids': r['para_ids'], 'context': r.get('context', [])}


# ==================== 悬挂补全 ====================

def dangling_paras(r, var, ps, table):
    """缺影响因素的悬挂记录可供查找的段落：先取同一小节的前两段，再按距离远近加上提到该变量（规范名或别名）的段落，
    最多 DANGLING_MAX_PARAS 段，不含所在段落。"""
    pos = {p['para_id']: i for i, p in enumerate(ps)}
    i = pos[r['para_id']]
    near = [q for q in ps[max(0, i - 2):i][::-1] if q.get('section') == ps[i].get('section')] or ps[max(0, i - 1):i]
    fixed = fin_items.ITEMS.get(var, ('', ''))[0].split()
    names = [squash(n) for n in [var, *table.get(var, {}).get('aliases', []), *fixed] if len(squash(n)) >= 2]
    mention = sorted((q for q in ps if q is not ps[i] and any(n in squash(q['text']) for n in names)),
                     key=lambda q: abs(pos[q['para_id']] - i))
    out = []
    for q in near + mention:
        if q not in out:
            out.append(q)
    return out[:DANGLING_MAX_PARAS]


def parse_dangling(content, var, texts, index):
    """校验悬挂补全结果，返回 None（没找到）或 {'name', 'direction', 'para_id', 'quote', 'reason'}。"""
    obj = parse_json(content, dict)
    if not as_bool(obj.get('found', False)):
        return None
    raw = norm_name(obj.get('name'))
    name, d, pid = resolve_name(raw, index), norm_direction(obj.get('direction')), to_int(obj.get('para_id'))
    errors = []
    if name is None:
        errors.append(f'“{raw}”不在概念表中，也不是会计科目')
    elif name in (var, ROOT):
        errors.append(f'“{raw}”不能作为缺失的影响因素')
    if d is None:
        errors.append('direction 必须是正向或负向')
    if pid not in texts or not quote_in(obj.get('quote'), texts[pid], min_len=4):
        errors.append(f'引文在第 {obj.get("para_id")} 段中找不到，请从所给段落中逐字摘录；找不到就回答没有找到')
    if errors:
        raise ValueError('；'.join(errors))
    return {'name': name, 'direction': d, 'para_id': pid, 'quote': short(obj.get('quote')), 'reason': short(obj.get('reason'))}


def resolve_dangling(ctx, reports):
    """处理缺影响因素的悬挂记录（缺被影响因素的留给第 3 步概念补全）。写进 r.extra（补出的跨段关系）、
    r.attached（{变量: [并入的记录]}）、r.unresolved（{变量: [未补上的记录]}）。"""
    step = Step(ctx, 'dangling_cache.jsonl', '悬挂')
    tasks = []
    for r in reports:
        r.extra, r.attached, r.unresolved = [], defaultdict(list), defaultdict(list)
        ins = {x['effect'] for x in r.complete}
        for x in r.rels:
            if not (x['cause'] is None and x['effect']):
                continue
            var = x['effect']
            if var in ins:  # 本篇别处已有入边：作为补充证据并入
                r.attached[var].append(note_of(x))
                continue
            cands = dangling_paras(x, var, r.paras, r.table)
            if not cands:
                r.unresolved[var].append(note_of(x))
                continue
            tasks.append((r, x, var, cands))
    log(f'悬挂补全（缺影响因素）：程序直接并入 {sum(len(v) for r in reports for v in r.attached.values())} 条，'
        f'交给大模型查找 {len(tasks)} 条')

    def work(task):
        r, x, var, cands = task
        source = r.para(x['para_id'])['text']
        texts = {p['para_id']: p['text'] for p in cands}
        record = f'影响因素：（原文未写明）；被影响因素：{var}；方向：{x["direction"]}；原文：“{x["evidence"]}”'
        block = para_block(cands)
        index = table_index(r.table)
        try:
            return step.ask(step.key(r.rid, record, source, block), lambda c: parse_dangling(c, var, texts, index),
                            lenient=lambda c: None, extra={'report_id': r.rid}, **ctx.about(r.rid),
                            names='、'.join([*r.table, *(n for n in fin_items.ITEMS if n != ROOT)]), record=record,
                            pid=x['para_id'], source=source, paragraphs=block)
        except RuntimeError as e:
            log(f'!!! {r.rid} 第 {x["para_id"]} 段的悬挂记录补全失败，按未说明处理：{e}')
            return None

    for (r, x, var, _), res in zip(tasks, ctx.parallel(tasks, work, '悬挂补全')):
        if not res:
            r.unresolved[var].append(note_of(x))
            continue
        r.extra.append({'report_id': r.rid, 'cause': res['name'], 'direction': res['direction'], 'effect': var,
                        'origin': 'bridged', 'evidence': [x['evidence'], res['quote']],
                        'para_ids': sorted({x['para_id'], res['para_id']}),
                        'context': [f'悬挂补全：{res["reason"]}'] if res['reason'] else []})
    log(f'悬挂补全完成：大模型补出 {sum(len(r.extra) for r in reports)} 条跨段关系，'
        f'未说明影响因素 {sum(len(v) for r in reports for v in r.unresolved.values())} 条')


# ==================== 并列复核 ====================

def check_parallel(ctx, reports):
    """并列复核：逐段抽取时模型常把同一原因的几个结果、或同一句里并列的事项抽成因果（如“销量回升，并有望提升单车收入”
    抽成 销量 → 单车收入）。把要进图的原文关系按研报分批，连同所在段落交给大模型逐条判断，判为“并列”的删掉，
    “因果”和“拿不准”的保留。写进 r.kept（保留的关系）、r.parallel（删掉的关系，各附理由）。"""
    step = Step(ctx, 'parallel_cache.jsonl', '并列')
    batches = []
    for r in reports:
        r.to_check = [x for x in r.complete + r.extra if classify(x, r.table) == 'edge']
        batches += [(r, r.to_check[k:k + PARALLEL_BATCH]) for k in range(0, len(r.to_check), PARALLEL_BATCH)]

    def parse(content, n):
        def value(it):
            rel = str(it.get('relation') or '').strip()
            return (rel, short(it.get('reason'))) if rel in PARALLEL_RELATIONS else None
        return parse_decisions(content, n, value,
                               lambda lack: f'这些编号缺少判断，或 relation 不是{"、".join(PARALLEL_RELATIONS)}之一：{"、".join(lack)}')

    def work(batch):
        r, rels = batch
        text = {p['para_id']: p['text'] for p in r.paras}
        lines = '\n'.join(f'{k}. {x["cause"]} —{x["direction"]}→ {x["effect"]} ｜ 段落 '
                          f'{"、".join(map(str, x["para_ids"]))} ｜ 原文：“{" …… ".join(x["evidence"])}”'
                          for k, x in enumerate(rels, 1))
        block = '\n\n'.join(f'[{p}] {text[p]}' for p in sorted({p for x in rels for p in x['para_ids']}) if p in text)
        try:
            return step.ask(step.key(r.rid, lines, block), lambda c: parse(c, len(rels)), extra={'report_id': r.rid},
                            **ctx.about(r.rid), relations=lines, paragraphs=block)
        except RuntimeError as e:  # 这一批全部保留
            log(f'!!! {r.rid} 有一批并列复核（{len(rels)} 条）失败，这些关系全部保留：{e}')
            return None

    drop = {}
    for (r, rels), res in zip(batches, ctx.parallel(batches, work, '并列复核')):
        for x, (rel, reason) in zip(rels, res or []):
            if rel == '并列':
                drop[id(x)] = reason
    for r in reports:
        r.kept = [x for x in r.to_check if id(x) not in drop]
        r.parallel = [{**rel_row(x), 'reason': drop[id(x)]} for x in r.to_check if id(x) in drop]
    log(f'并列复核完成：{sum(len(r.to_check) for r in reports)} 条原文关系（{len(batches)} 批），判为并列而删掉 {len(drop)} 条')
