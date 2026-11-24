"""从研报中提取估值逻辑图：每篇研报整理成一张分五层的有向图，原文概念逐层连到预置的会计层和公司估值。
框架、规则的来由和各版本的结果见 docs/研报逻辑图抽取框架设计.md，提示词见 src/prompts/report_logic_prompt.md。

五层：1 公司估值、2 会计层（这两层预置，见 src/fin_items.py）、3 经营指标、4 公司经营与行业、5 外部环境（只放宏观和政策因素）。
边只能从层号大的指向层号小的（可以跨层），第 3-5 层内部也可以有原文陈述的关系。

输入是 clear_txt.py（src/report_text.py）写出的段落 parquet（也可以换成自己处理好的文本，至少含 report_id、para_id、text 三列）；
读入时会再剔除一遍风险提示。clear_txt.py 同时写出的研报元数据表（每篇研报一行：发布日期、券商、报告类型、是不是个股公司研报、
公司和代码、分析师，见 src/report_meta.py）用来筛选要抽取的研报，默认只抽认出了股票的个股公司研报；两张表用 report_id 对应。大模型调用都走 .env 中配置的接口（见 src/llm_client.py）。流程：
  1. 概念分层填表（建表）：每篇按顺序分块通读，滚动维护“本篇概念表”（规范名、别名、定义、层级、类型）。
  2. 原文连边（抽取）：每段一次调用，抽出原文明说的一跳关系并逐字核对引文；表里没有的新概念先记下。
     整理概念表（整理）：每篇一次调用，把新概念并入概念表（是已有概念的记为别名），并整理第 3-5 层同一层概念之间的包含关系。
     悬挂补全（悬挂）：缺影响因素的记录到相关段落里找原文。并列复核（并列）：删掉把并列事项抽成因果的关系。
     包含关系缩减：向上只有包含关系的概念并入上位概念，另有出边的去掉包含关系、保留原文边。
     按层归类：进图、会计科目之间、反向（不进图）；第 3-5 层概念直接指向公司估值的转为补全时参考的线索。
  3. 建图与子图补全（补全）：删掉孤立概念，会计层只留原文提到的科目；走不到估值的子图每个调用一次，
     原文写明的补原文边（逐字核对），没写的给每个终点补一条逻辑边（闭集选择，不造新概念）。
  4. 收尾：去环（先删逻辑边）、剪掉有方向一致的更长链路的直连、删掉没有下层概念连入的会计科目。
  5. 全局去重（归一）：各篇同一层的概念用 BGE 向量和名称包含取候选对，交给大模型判断，只加全局名称、不改图。
  6. 输出（布局见 src/config.py 的 OutDir）：入口页面 研报逻辑图.html；data/graphs/ 下每篇研报一个逻辑图数据文件
     （含缩减之前三个阶段的图），页面选中一篇研报时才读取它的数据、计算布局并绘图；data/ 下 edges/nodes/relations.parquet。
边的 origin：extracted 段内原文 / bridged 跨段原文（悬挂补全或子图补全，引文经程序核对）/ logic 逻辑补全 / preset 预置的会计关系。

第 1-4 步各篇互不影响：每篇研报各自走完这几步，同时在处理 IN_FLIGHT 篇（默认等于大模型并发数），做完一篇就接上下一篇
（长研报要逐块顺序通读，不会拖住别的研报；各篇的大模型调用共用一个线程池），每篇走完这几步，每批结束就把各篇的 Report 存档到输出目录的
cache/reports/（pickle，带指纹，见 archive_sig），中断后重新运行从上一批之后接着跑；存档有效的研报直接读回，不再进任何一步。
第 5、6 步对本次选中的、已完成的全部研报（读回的加新做的）统一做一遍。某一篇出错时只有它不存档、不进输出（记到
cache/failed_reports.jsonl，下次运行重试）；鉴权失败、余额不足这类致命错误让整个运行停下，此前各批的存档都已写入。
各步的大模型结果都缓存在 cache/*_cache.jsonl 中（进度日志在 cache/progress.log），重新处理的研报只补做缺少的调用
（输入或这一步的 prompt 变了会自动重新调用）。

代码分在 src/logic/ 下：base（常量、大模型调用与缓存、校验工具）、extract（第 1、2 步）、graph（包含关系缩减、
建图、剪枝）、complete（第 3 步）、output（第 5、6 步）；这里只负责按顺序调用、分批和存档。

用法：在仓库根目录运行 python extract_logic_chain.py（输入、输出和试跑篇数在该脚本开头设置）
"""
import hashlib
import json
import pickle
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src import fin_items
from src.config import OutDir
from src.llm_client import STOP, Budget, FatalAPIError, load_env, log, set_log_path, set_quiet
from src.logic.base import Ctx, Report, append_jsonl, layer_of, load_prompts, prompt_sig, text_hash, write_parquet
from src.logic.complete import complete_subgraphs
from src.logic.extract import (build_tables, check_parallel, extract_relations, organize_tables, resolve_dangling,
                                           tail)
from src.logic.graph import build_graphs, finish_graphs, merge_parents
from src.logic.output import EMBED_MODEL, global_dedup, graph_file, write_outputs
from src.report_text import strip_risk

META_COLS = ('file', 'date', 'broker', 'company', 'stock_code', 'title')
TABLE_COLS = ('date', 'broker', 'company', 'stock_code', 'report_type', 'analysts')  # 有元数据表时以它为准的研报信息
IN_FLIGHT = None  # 同时在处理的研报篇数，None 表示和大模型并发数（workers）相同。每篇研报各自走完第 1-4 步，做完一篇就接上下一篇；
#                   各篇各步的大模型调用共用一个线程池（线程数即并发数），长研报逐块通读时别的研报照常推进
ARCHIVE_STEPS = ('建表', '整理', '抽取', '并列', '悬挂', '补全')  # 第 1-4 步用到的提示词，改了其中任何一节存档就失效
ARCHIVE_VERSION = 1  # 存档版本：改了第 1-4 步的流程代码、或存档结构不兼容时手动加一，已有的存档随之全部失效、重新处理


def quarter_of(date):
    """YYYYMMDD -> 'YYYYQn'；日期缺失或格式不对时返回空串。"""
    return f'{date[:4]}Q{(int(date[4:6]) - 1) // 3 + 1}' if date and len(date) >= 6 and date[:6].isdigit() else ''


def select_reports(reports_path, select, min_per_quarter=1):
    """按元数据表筛选要抽取的研报，返回 ({report_id: 元数据表的一行}, 说明)；reports_path 为 None 时不筛选，返回 (None, 说明)。
    指定了元数据表而文件不存在时直接停下：不筛选就全部抽取的话，调用大模型的次数会多出几十倍。
    min_per_quarter：select 选中的研报再按（股票代码, 发布季度）分组，一组里不足这么多篇的整组不抽（同一家公司同一个季度
    只有 1 篇时没有别的研报可比）；没有股票代码或日期的研报不分组、原样保留。"""
    if reports_path is None:
        return None, '不按元数据表筛选，段落表里的研报全部抽取'
    if not Path(reports_path).exists():
        raise SystemExit(f'找不到研报元数据表 {reports_path}：请重新运行 python clear_txt.py 生成（它和段落表一起写出）；'
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


# ==================== 按篇存档 ====================

def archive_base():
    """存档指纹的公共部分：版本常量、第 1-4 步各步的提示词（展开共用规则之后）、预置的会计层清单和关系（src/fin_items.py）。
    全局去重、框架对比的提示词不在内：改它们不会让各篇的存档失效。"""
    return text_hash(str(ARCHIVE_VERSION), prompt_sig(load_prompts(), *ARCHIVE_STEPS),
                     json.dumps([fin_items.ITEMS, fin_items.PRESET_EDGES], ensure_ascii=False))


def archive_sig(r, base):
    """一篇研报的存档指纹：公共部分 + 进入流程的段落（剔除风险提示之后的编号、小节标题、文字）+ 提示词里用到的公司、标题。
    指纹对不上的存档当作没有，重新处理（调用缓存还在，重做不花钱）。"""
    return text_hash(base, r.meta['company'], r.meta['title'],
                     *(f'{p["para_id"]}\x1e{p.get("section") or ""}\x1e{p["text"]}' for p in r.paras))


def archive_path(out, rid):
    """存档文件：cache/reports/ 下按名称前两个字符分子目录，文件名同 data/graphs/ 下的逻辑图数据文件。"""
    return out.reports / Path(graph_file(rid)).with_suffix('.pkl')


def read_archive(out, r, sig):
    """读回一篇研报的存档，没有、读不回来或指纹对不上时返回 None。研报信息和段落换成本次读入的（元数据表可能更新过）。"""
    path = archive_path(out, r.rid)
    if not path.exists():
        return None
    try:
        with path.open('rb') as f:
            obj = pickle.load(f)
        done = obj['report']
        if obj['sig'] != sig or done.rid != r.rid:
            return None
    except Exception:  # 写了一半、流程代码改过等，读不回来的当作没有
        return None
    done.meta, done.paras = r.meta, r.paras
    return done


def write_archive(out, r, sig):
    """存档一篇走完第 1-4 步的研报：先写临时文件再改名，中断时不会留下写了一半的存档。"""
    path = archive_path(out, r.rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    with tmp.open('wb') as f:
        pickle.dump({'sig': sig, 'report': r}, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def clear_results(out):
    """清空输出目录里上一次的结果：data/（逻辑图、parquet、框架对比）、入口页面、各篇研报的存档、出错记录和进度日志。
    大模型调用缓存（cache/*_cache.jsonl）保留：研报重新处理时已经做过的调用直接取缓存，不会重复花钱。"""
    for path in (out.data, out.reports):
        if path.exists():
            shutil.rmtree(path)
    for path in (out.viewer, out.failed, out.log):
        path.unlink(missing_ok=True)


# ==================== 第 1-4 步 ====================

class StepError(Exception):
    """第 1-4 步中某一步抛出了异常（各步自己没有兜底的），说明里注明是哪一步。"""


def run_steps(ctx, reports):
    """第 1-4 步，各篇互不影响。并行任务里出错的研报（记在 ctx.errors）不再进后面的步骤，免得拿残缺的结果接着调用大模型。
    某一步抛出异常时抛出 StepError；遇到致命错误（鉴权失败、余额不足等，各步会把它当成调用失败兜底，所以这里每步之后
    看 STOP 是否置位）时抛出 FatalAPIError。"""
    live = list(reports)

    def step(name, fn, *args):
        if not live:
            return
        try:
            fn(*args)
        except FatalAPIError:
            raise
        except Exception as e:
            raise StepError(f'{name}：{type(e).__name__}: {e}') from e
        if STOP.is_set():
            raise FatalAPIError(f'{name}时遇到致命错误（详见上面的日志）')
        live[:] = [r for r in live if r.rid not in ctx.errors]

    # 1. 概念分层填表
    step('概念分层填表', build_tables, ctx, live)

    # 2. 原文连边，再整理概念表：并入新概念、整理第 3-5 层的包含关系（第 1 步的最后一环，要等表补全后再做）
    step('原文连边', extract_relations, ctx, live)
    step('整理概念表', organize_tables, ctx, live)
    for r in live:
        r.complete = [{**x, 'origin': 'extracted', 'evidence': [x['evidence']], 'para_ids': [x['para_id']]}
                      for x in r.rels if x['cause'] and x['effect']]
    step('悬挂补全', resolve_dangling, ctx, live)
    step('并列复核', check_parallel, ctx, live)  # 用合并前的名称判断

    # 包含关系缩减、按层归类，再建图（第 4 步的前半）
    step('包含关系缩减', merge_parents, live)
    step('建图', build_graphs, live)

    # 3. 子图补全：只给走不到估值的子图补边，已经能沿原文关系走到估值的概念不再补全
    step('子图补全', complete_subgraphs, ctx, live)

    # 4（后半）. 去环、剪枝、删掉没有任何下层概念连入的会计科目
    step('收尾', finish_graphs, live)


def run_batch(ctx, batch):
    """一批研报走完第 1-4 步，返回 (完成的研报, {report_id: 出错说明})。并行任务里的异常能对应到某一篇，直接记为这一篇出错；
    有异常中断了整批时，这一批改为逐篇重做（换成新的 Report；调用缓存在，已完成的调用不会重复花钱）。致命错误照常抛出。"""
    try:
        run_steps(ctx, batch)
    except StepError as e:
        log(f'!!! 本批 {len(batch)} 篇在{e}，改为逐篇处理')
    else:
        failed = {r.rid: '；'.join(dict.fromkeys(ctx.errors[r.rid])) for r in batch if r.rid in ctx.errors}
        return [r for r in batch if r.rid not in failed], failed
    done, failed = [], {}
    for r in batch:
        ctx.errors.pop(r.rid, None)
        ctx.call_failed.discard(r.rid)
        r = Report(r.rid, r.meta, r.paras)
        try:
            run_steps(ctx, [r])
        except StepError as e:
            ctx.errors[r.rid].append(str(e))
        if r.rid in ctx.errors:
            failed[r.rid] = '；'.join(dict.fromkeys(ctx.errors[r.rid]))
        else:
            done.append(r)
    return done, failed


def preview(paras_path, out_dir, limit, reports_path, select, seed, min_per_quarter):
    """只筛选不抽取：打印选中的研报数、已有存档的篇数、待处理的篇数和段落数、各季度的分布。不调用大模型、不写文件。
    已有存档只按文件是否存在来数，不核对指纹（正式运行时指纹对不上的会重新处理）。"""
    out = OutDir(out_dir)
    info, note = select_reports(reports_path, select, min_per_quarter)
    reports, _ = load_reports(paras_path, limit, info, seed)
    todo = [r for r in reports if not archive_path(out, r.rid).exists()]
    paras = lambda rs: sum(len(r.paras) for r in rs)
    print(note + (f'，按种子 {seed} 随机抽 {limit} 篇' if limit is not None else ''))
    print(f'选中 {len(reports)} 篇、{paras(reports)} 个段落、{len({r.meta["stock_code"] for r in reports})} 家公司；'
          f'已有存档 {len(reports) - len(todo)} 篇，待处理 {len(todo)} 篇、{paras(todo)} 个段落')
    by_q = defaultdict(lambda: [0, 0])
    for r in reports:
        by_q[quarter_of(r.meta['date']) or '无日期'][0] += 1
    for r in todo:
        by_q[quarter_of(r.meta['date']) or '无日期'][1] += 1
    print('各季度（选中 / 待处理）：' + '，'.join(f'{q} {a} / {b}' for q, (a, b) in sorted(by_q.items())))
    print('只做了筛选，没有调用大模型；去掉 --dry-run 才开始抽取')


def main(paras_path, out_dir, limit=None, embed_model=EMBED_MODEL, reports_path=None, select=None, seed=0, workers=None,
         fresh=False, min_per_quarter=1, dry_run=False, budget=None, order='length'):
    """paras_path：段落 parquet（clear_txt.py 的输出或自备文本）；limit：在选中的研报里按种子 seed 随机抽多少篇
    （同一个种子下嵌套），None 为全部。reports_path：研报元数据表（clear_txt.py 同时写出），select(行) 返回真的研报才抽取；
    reports_path 为 None 时不筛选。workers：同时在途的大模型调用数，None 为用 .env 的 CONCURRENCY。
    fresh：先清空输出目录里上一次的结果和各篇存档再跑（见 clear_results，调用缓存保留）。
    min_per_quarter：同一家公司同一个季度不足这么多篇的不抽（见 select_reports）。
    dry_run：只筛选、统计待处理的研报，不调用大模型，也不写任何文件。
    budget：本次运行最多花多少美元（按中转站实际扣费，见 src/llm_client.py 的 Budget），到了就停下：已做完的研报都已存档，
    没做完的下次接着跑；这时不做第 5、6 步，输出目录里的结果保持上一次的样子。None 为不设上限。
    order：待处理研报的先后。'length' 全部研报按篇幅从长到短（长研报最慢，先开始，适合一次跑完）；
    'company' 按公司排，一家公司的研报排完再排下一家，待处理段落少的公司在前（中途停下时，做完的公司是完整的，可以直接对比）。"""
    t0 = time.time()
    if not Path(paras_path).exists():
        raise SystemExit(f'找不到段落文件 {paras_path}，请先运行 python clear_txt.py，或把输入换成已有的段落 parquet')
    if dry_run:
        return preview(paras_path, out_dir, limit, reports_path, select, seed, min_per_quarter)
    if fresh:
        clear_results(OutDir(out_dir))
    out = OutDir(out_dir).make()
    out_dir = out.root
    set_log_path(out.log)
    env = load_env()
    info, note = select_reports(reports_path, select, min_per_quarter)
    reports, n_risk = load_reports(paras_path, limit, info, seed)
    if not reports:
        raise SystemExit(f'{note}；没有可抽取的研报')
    ctx = Ctx(env, out_dir, {r.rid: r.meta for r in reports}, workers)
    if fresh:
        log('已清空输出目录里上一次的结果和各篇存档（大模型调用缓存保留）')
    log(f'===== 开始：模型 {env["MODEL"]}，{len(reports)} 篇研报、{sum(len(r.paras) for r in reports)} 个段落，并行 {ctx.workers} =====')
    log(note + (f'，本次按种子 {seed} 随机抽 {limit} 篇' if limit is not None else ''))
    if n_risk:
        log(f'读入时剔除风险提示 {n_risk} 字（全部研报合计）')

    guard = None
    if budget is not None:
        try:
            guard = Budget(env, budget)
        except Exception as e:
            raise SystemExit(f'设了额度上限 {budget} 美元，但查不到这把密钥的已用额度（{type(e).__name__} {e}），没法监控花费，不开始运行')
        log(f'额度上限：本次运行最多花 {guard.usd:.2f} 美元（这把密钥此前累计已用 {guard.start:.2f} 美元）')

    # 存档有效的研报直接读回，不再进任何一步
    base = archive_base()
    sigs = {r.rid: archive_sig(r, base) for r in reports}
    done = {}
    for r in reports:
        old = read_archive(out, r, sigs[r.rid])
        if old is not None:
            done[r.rid] = old
    todo = [r for r in reports if r.rid not in done]
    n_skip, n_stale = len(done), sum(archive_path(out, r.rid).exists() for r in todo)
    log(f'选中 {len(reports)} 篇：已完成、从存档读回跳过 {n_skip} 篇，本次待处理 {len(todo)} 篇'
        + (f'（其中 {n_stale} 篇的存档指纹对不上或读不回来，重新处理）' if n_stale else ''))

    # 第 1-4 步：每篇研报各自走完，同时在处理 IN_FLIGHT 篇，做完一篇立即存档并接上下一篇；长研报先开始（它们最慢）
    if order not in ('length', 'company'):
        raise SystemExit(f"order 只能是 'length' 或 'company'，收到 {order!r}")
    code = lambda r: r.meta['stock_code'] or r.rid  # 没有股票代码的研报各自算一家
    left, left_paras = defaultdict(int), defaultdict(int)  # 各家公司待处理的篇数、段落数
    for r in todo:
        left[code(r)] += 1
        left_paras[code(r)] += len(r.paras)
    n_companies, n_full = len(left), 0
    if order == 'company':  # 一家排完再排下一家：待处理段落少的公司在前，公司内部长研报先开始
        todo.sort(key=lambda r: (left_paras[code(r)], code(r), -len(r.paras), r.rid))
    else:
        todo.sort(key=lambda r: (-len(r.paras), r.rid))
    failed, unsaved, n_new, n_saved, fatal = {}, [], 0, 0, None
    in_flight = min(IN_FLIGHT or ctx.workers, max(len(todo), 1))
    if todo:
        log(f'待处理的 {len(todo)} 篇逐篇处理：同时在处理 {in_flight} 篇，做完一篇就接上下一篇；各篇各步的过程不打日志，每做完一篇打一行')
        log(f'处理顺序：' + (f'按公司，{n_companies} 家公司里待处理段落少的在前，一家排完再排下一家'
                             if order == 'company' else '全部研报按篇幅从长到短'))

    def one(r):
        if STOP.is_set():  # 已经因致命错误或额度上限停下：排队中的研报不再开始
            raise FatalAPIError('任务已停止')
        set_quiet(True)
        return run_batch(ctx, [r])

    ctx.pool = ThreadPoolExecutor(max_workers=ctx.workers)
    runner = ThreadPoolExecutor(max_workers=in_flight)
    try:
        futures = {runner.submit(one, r): r for r in todo}
        for k, fut in enumerate(as_completed(futures), 1):
            src = futures[fut]
            try:
                ok, bad = fut.result()
            except (FatalAPIError, SystemExit) as e:  # 其余各篇看到 STOP 也会很快停下；已经做完的照常存档
                fatal = fatal or e
                continue
            for r in ok:
                if r.rid in ctx.call_failed:  # 有调用失败、按各步的方式兜底了：照常输出，但不存档，下次运行重试这些调用
                    unsaved.append(r.rid)
                else:
                    write_archive(out, r, sigs[r.rid])
                    n_saved += 1
                done[r.rid] = r
            for rid, err in bad.items():
                failed[rid] = err
                log(f'!!! {rid} 出错，不存档、不进输出（下次运行重试）：{err}')
                append_jsonl(out.failed, {'report_id': rid, 'error': err, 'time': time.strftime('%Y-%m-%d %H:%M:%S')})
            n_new += len(ok)
            u = ctx.llm.usage
            log(f'已做完 {k}/{len(todo)} 篇（这一篇 {len(src.paras)} 段，{"完成" if ok else "出错"}）；累计完成 {len(done)}/{len(reports)} 篇，'
                f'已存档 {n_saved} 篇，调用 {u["calls"]} 次，已用时 {time.time() - t0:.0f} 秒')
            left[code(src)] -= 1
            if not left[code(src)]:  # 这家公司待处理的研报都过了一遍（出错或有调用失败的不算存档，下次运行补做）
                n_full += 1
                log(f'== {src.meta["company"] or code(src)} {src.meta["stock_code"]} 的研报都做完了：第 {n_full}/{n_companies} 家')
    finally:
        runner.shutdown(wait=True, cancel_futures=True)
        ctx.pool.shutdown(wait=True, cancel_futures=True)
        ctx.pool = None
    if guard is not None and guard.hit:
        log(f'!!! 因额度上限停止：本次运行花了 {guard.close():.2f} 美元，新做完并存档 {n_saved} 篇（累计完成 {len(done)}/{len(reports)} 篇），'
            f'调用 {ctx.llm.usage["calls"]} 次，用时 {time.time() - t0:.0f} 秒；没做完的下次运行接着跑，输出目录里的结果未更新')
        raise SystemExit(0)
    if fatal is not None:
        log(f'!!! 致命错误，已停止：{fatal}。没做完的研报不存档；已做完的 {n_saved} 篇都已存档在 {out.reports}，处理后重新运行即接着跑')
        raise SystemExit(1)
    if unsaved:
        log(f'!!! {len(unsaved)} 篇有大模型调用失败（已按各步的方式兜底）：照常输出，但不存档，下次运行重试这些调用：'
            f'{"、".join(unsaved[:10])}{" 等" if len(unsaved) > 10 else ""}')
    if failed:
        log(f'!!! {len(failed)} 篇出错，不进输出，记在 {out.failed}，下次运行自动重试')
    finished = [done[r.rid] for r in reports if r.rid in done]  # 按 report_id 排序
    if not finished:
        raise SystemExit('没有完成的研报，见上面的日志')

    write_parquet([{**{k: x[k] for k in ('report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence')},
                    'cause_layer': layer_of(x['cause'], r.table), 'effect_layer': layer_of(x['effect'], r.table)}
                   for r in finished for x in r.rels],
                  out.data / 'relations.parquet', ['report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence'])

    # 5. 全局去重：只给节点加全局名称，不改图的结构
    mapping, global_table = global_dedup(ctx, finished, embed_model)

    # 6. 输出
    stats, node_rows, edge_table = write_outputs(finished, mapping, global_table, out_dir, env['MODEL'], embed_model)

    u = ctx.llm.usage
    reach = [x['reach'] for x in stats]
    log(f'===== 结束：{len(stats)} 篇研报（本次新处理 {n_new} 篇、从存档读回跳过 {n_skip} 篇、出错 {len(failed)} 篇）；'
        f'共 {len(node_rows)} 个全局节点、{len(edge_table)} 条边；概念连到公司估值的比例平均 {np.mean(reach):.0%}；'
        f'调用 {u["calls"]} 次（不合格重试 {u["retries"]} 次），输入 {u["prompt_tokens"]} tokens，'
        f'输出 {u["completion_tokens"]} tokens；总用时 {time.time() - t0:.0f} 秒 =====')
    if guard is not None:
        log(f'本次运行花了 {guard.close():.2f} 美元（上限 {guard.usd:.2f}）')
    log(f'逻辑图入口：{out.viewer}（双击打开，选中研报时才读取数据并绘图）；结果数据在 {out.data}')
