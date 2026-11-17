"""从研报中提取估值逻辑图：每篇研报整理成一张分五层的有向图，原文概念逐层连到预置的会计层和公司估值。
框架、规则的来由和各版本的结果见 docs/研报逻辑图抽取框架设计.md，提示词见 src/prompts/report_logic_prompt.md。

五层：1 公司估值、2 会计层（这两层预置，见 src/fin_items.py）、3 经营指标、4 公司经营与行业、5 外部环境（只放宏观和政策因素）。
边只能从层号大的指向层号小的（可以跨层），第 3-5 层内部也可以有原文陈述的关系。

输入是 clear_txt.py（src/report_text.py）写出的段落 parquet（也可以换成自己处理好的文本，至少含 report_id、para_id、text 三列）；
读入时会再剔除一遍风险提示。大模型调用都走 .env 中配置的接口（见 src/llm_client.py）。流程：
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
各步的大模型结果都缓存在输出目录的 cache/*_cache.jsonl 中（进度日志在 cache/progress.log），重新运行只补做缺少的部分（输入或这一步的 prompt 变了会自动重新调用）。

代码分在 src/logic/ 下：base（常量、大模型调用与缓存、校验工具）、extract（第 1、2 步）、graph（包含关系缩减、
建图、剪枝）、complete（第 3 步）、output（第 5、6 步）；这里只负责按顺序调用。

用法：在仓库根目录运行 python extract_logic_chain.py（输入、输出和试跑篇数在该脚本开头设置）
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from src.config import OutDir
from src.llm_client import load_env, log, set_log_path
from src.logic.base import Ctx, Report, layer_of, write_parquet
from src.logic.complete import complete_subgraphs
from src.logic.extract import (build_tables, check_parallel, extract_relations, organize_tables, resolve_dangling,
                                           tail)
from src.logic.graph import build_graphs, finish_graphs, merge_parents
from src.logic.output import EMBED_MODEL, global_dedup, write_outputs
from src.report_text import strip_risk

META_COLS = ('file', 'date', 'broker', 'company', 'stock_code', 'title')


def load_reports(paras_path, limit):
    """读入段落 parquet（读入时再剔除一遍风险提示），按研报分组，返回 (按 report_id 排序的 Report 列表, 剔除的风险提示字数)。"""
    table = pq.read_table(paras_path)
    missing = {'report_id', 'para_id', 'text'} - set(table.column_names)
    if missing:
        raise SystemExit(f'{paras_path} 缺少必需的列：{missing}')
    rows = [p for p in table.to_pylist() if (p.get('text') or '').strip()]
    paras = strip_risk(rows)
    n_risk = sum(len(p['text']) for p in rows) - sum(len(p['text']) for p in paras)
    report_ids = set(sorted({p['report_id'] for p in paras})[:limit])
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
        reports.append(Report(rid, meta, ps))
    return reports, n_risk


def main(paras_path, out_dir, limit=None, embed_model=EMBED_MODEL):
    """paras_path：段落 parquet（clear_txt.py 的输出或自备文本）；limit：只处理前多少篇研报（按 report_id 排序），None 为全部。"""
    if not Path(paras_path).exists():
        raise SystemExit(f'找不到段落文件 {paras_path}，请先运行 python clear_txt.py，或把输入换成已有的段落 parquet')
    out = OutDir(out_dir).make()
    out_dir = out.root
    set_log_path(out.log)
    env = load_env()
    reports, n_risk = load_reports(paras_path, limit)
    ctx = Ctx(env, out_dir, {r.rid: r.meta for r in reports})
    log(f'===== 开始：模型 {env["MODEL"]}，{len(reports)} 篇研报、{sum(len(r.paras) for r in reports)} 个段落，并行 {ctx.workers} =====')
    if n_risk:
        log(f'读入时剔除风险提示 {n_risk} 字（全部研报合计）')

    # 1. 概念分层填表
    build_tables(ctx, reports)

    # 2. 原文连边，再整理概念表：并入新概念、整理第 3-5 层的包含关系（第 1 步的最后一环，要等表补全后再做）
    extract_relations(ctx, reports)
    organize_tables(ctx, reports)
    write_parquet([{**{k: x[k] for k in ('report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence')},
                    'cause_layer': layer_of(x['cause'], r.table), 'effect_layer': layer_of(x['effect'], r.table)}
                   for r in reports for x in r.rels],
                  out.data / 'relations.parquet', ['report_id', 'para_id', 'cause', 'direction', 'effect', 'evidence'])
    for r in reports:
        r.complete = [{**x, 'origin': 'extracted', 'evidence': [x['evidence']], 'para_ids': [x['para_id']]}
                      for x in r.rels if x['cause'] and x['effect']]
    resolve_dangling(ctx, reports)
    check_parallel(ctx, reports)  # 用合并前的名称判断

    # 包含关系缩减、按层归类，再建图（第 4 步的前半）
    merge_parents(reports)
    build_graphs(reports)

    # 3. 子图补全：只给走不到估值的子图补边，已经能沿原文关系走到估值的概念不再补全
    complete_subgraphs(ctx, reports)

    # 4（后半）. 去环、剪枝、删掉没有任何下层概念连入的会计科目
    finish_graphs(reports)

    # 5. 全局去重：只给节点加全局名称，不改图的结构
    mapping, global_table = global_dedup(ctx, reports, embed_model)

    # 6. 输出
    stats, node_rows, edge_table = write_outputs(reports, mapping, global_table, out_dir, env['MODEL'], embed_model)

    u = ctx.llm.usage
    reach = [x['reach'] for x in stats]
    log(f'===== 结束：{len(stats)} 篇研报；共 {len(node_rows)} 个全局节点、{len(edge_table)} 条边；'
        f'概念连到公司估值的比例平均 {np.mean(reach):.0%}；调用 {u["calls"]} 次（不合格重试 {u["retries"]} 次），'
        f'输入 {u["prompt_tokens"]} tokens，输出 {u["completion_tokens"]} tokens =====')
    log(f'逻辑图入口：{out.viewer}（双击打开，选中研报时才读取数据并绘图）；结果数据在 {out.data}')
