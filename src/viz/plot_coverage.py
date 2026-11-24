"""生成抽取进度页：逻辑链抽取（各篇研报的存档）做到哪了：哪些公司做完了、做完的是否完整、没做完的还差多少篇
（单个自包含 HTML，不依赖外部库，页面模板在 coverage_template.html）。

页面上的数字都从结果文件算出：
  研报元数据表 + 抽取脚本的筛选条件（select、MIN_PER_QUARTER）：要抽取的研报，即抽取日志里“剩 N 篇”的那些；
  输出目录的 cache/reports/：各篇存档，元数据表里有、指纹对得上的算做完（规则同抽取脚本读回存档）；
    每篇的概念数、原文边数从存档的逻辑图算，口径同 sort_divergence.py 的研报大小（第 3-5 层概念，extracted、bridged 边）；
  cache/progress.log：最近一次记了花费的抽取运行（模型、并发、花费、用时），按段落数折算剩余工作量，日志里没有花费时不估；
  data/wave1_companies.txt（有的话）：名单里的公司在表里标“第一批”。
抽取还在后台运行时也能跑（存档整篇写完才改名生效），约一两分钟（要读回全部存档）。

用法：python extract_logic_chain.py --task coverage（默认统计全部公司，即 --companies all；给了 --companies 时只统计这些）
"""
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pyarrow.parquet as pq

from src.config import DATA_DIR, OutDir
from src.logic.base import TEXT_ORIGINS
from src.logic.output import graph_record
from src.report_logic import (archive_base, archive_path, archive_sig, load_reports, quarter_of, read_archive,
                              select_reports)

WAVE1 = DATA_DIR / 'wave1_companies.txt'  # 第一批公司名单（逗号分隔的股票代码），没有这个文件时不标
TEMPLATE = (Path(__file__).parent / 'coverage_template.html').read_text(encoding='utf-8')

RUN_START = re.compile(r'^(\S+ \S+) ===== 开始：模型 (\S+)，.*并行 (\d+) =====')
BUDGET_STOP = re.compile(r'^(\S+ \S+) !!! 因额度上限停止：本次运行花了 ([\d.]+) 美元，新做完并存档 (\d+) 篇.*用时 (\d+) 秒')
RUN_END = re.compile(r'^(\S+ \S+) ===== 结束：.*本次新处理 (\d+) 篇.*总用时 (\d+) 秒')
RUN_COST = re.compile(r'^(\S+ \S+) 本次运行花了 ([\d.]+) 美元')


def last_paid_run(log_path):
    """进度日志里最近一次记了花费的抽取运行：{start, end, model, workers, usd, n_saved, seconds}；没有时返回 None。
    额度上限停止的一行里有花费和新存档篇数；正常结束的看“===== 结束”一行和紧跟的“本次运行花了”一行。"""
    if not Path(log_path).exists():
        return None
    run, start, ended = None, None, None
    for line in Path(log_path).read_text(encoding='utf-8', errors='replace').splitlines():
        if m := RUN_START.match(line):
            start, ended = {'start': m[1], 'model': m[2], 'workers': int(m[3])}, None
        elif start and (m := BUDGET_STOP.match(line)):
            run = {**start, 'end': m[1], 'usd': float(m[2]), 'n_saved': int(m[3]), 'seconds': int(m[4])}
        elif start and (m := RUN_END.match(line)):
            ended = {**start, 'end': m[1], 'n_saved': int(m[2]), 'seconds': int(m[3])}
        elif ended and (m := RUN_COST.match(line)):
            run, ended = {**ended, 'usd': float(m[2])}, None
    return run if run and run['usd'] > 0 and run['n_saved'] > 0 else None


def collect(out_dir, paras_path, reports_path, select, min_per_quarter):
    """按公司汇总抽取进度，返回给页面用的 {summary, companies, excluded}。"""
    out = OutDir(out_dir)
    meta = pq.read_table(reports_path).to_pylist()
    ok = [r for r in meta if r.get('status', 'ok') == 'ok']
    loose, _ = select_reports(reports_path, select, min_per_quarter=1)  # 不按季度篇数筛
    sel, note = select_reports(reports_path, select, min_per_quarter=min_per_quarter)
    print(note)

    # 逐篇核对存档，做完的从逻辑图算概念数和原文边数
    base, rec = archive_base(), {}
    for r in load_reports(paras_path, None, info=sel)[0]:
        path = archive_path(out, r.rid)
        x = {'paras': len(r.paras), 'chars': sum(len(p['text']) for p in r.paras), 'status': 'todo'}
        if path.exists():
            done = read_archive(out, r, archive_sig(r, base))
            x['status'] = 'stale' if done is None else 'done'
            if done is not None:
                g = graph_record(done, {})
                x['mtime'] = path.stat().st_mtime
                x['concepts'] = sum(n['layer'] >= 3 for n in g['nodes'])
                x['edges'] = sum(e['origin'] in TEXT_ORIGINS for e in g['edges'])
        rec[r.rid] = x
    for rid in sel:  # 段落表里没有正文段落的
        rec.setdefault(rid, {'paras': 0, 'chars': 0, 'status': 'todo'})
    archives = {p.stem for p in out.reports.rglob('*.pkl')} if out.reports.exists() else set()
    thin_ids = set(loose) - set(sel)
    orphan = archives - set(sel)

    rows = [{**sel[rid], **rec[rid], 'quarter': quarter_of(sel[rid]['date'])} for rid in sel]
    quarters = sorted({r['quarter'] for r in rows if r['quarter']})
    wave1 = set(WAVE1.read_text(encoding='utf-8').replace('\n', ',').split(',')) - {''} if WAVE1.exists() else set()
    thin_by = defaultdict(list)
    for rid in thin_ids:
        thin_by[loose[rid]['stock_code']].append(loose[rid])
    by = defaultdict(list)
    for r in rows:
        by[r['stock_code']].append(r)

    def avg(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    companies = []
    for code, g in by.items():
        done = [r for r in g if r['status'] == 'done']
        todo = [r for r in g if r['status'] != 'done']
        names = [r['company'] for r in g]
        q = {qq: [sum(r['quarter'] == qq for r in g), sum(r['quarter'] == qq for r in done)] for qq in quarters}
        cm = avg([r['concepts'] for r in done])
        companies.append({
            'code': code, 'name': max(set(names), key=names.count),
            'total': len(g), 'done': len(done), 'stale': sum(r['status'] == 'stale' for r in g), 'todo': len(todo),
            'todo_paras': sum(r['paras'] for r in todo), 'paras': sum(r['paras'] for r in g),
            'brokers': len({r['broker'] for r in g}), 'brokers_done': len({r['broker'] for r in done}),
            'q': q, 'n_quarters': sum(v[0] > 0 for v in q.values()), 'usable_q': sum(v[1] >= 2 for v in q.values()),
            'thin': len(thin_by.get(code, [])),
            'concepts_mean': round(cm, 1) if cm is not None else None,
            'concepts_total': sum(r['concepts'] for r in done),
            'chars_done': round(avg([r['chars'] for r in done]) or 0) if done else None,
            'chars_todo': round(avg([r['chars'] for r in todo]) or 0) if todo else None,
            'wave1': code in wave1,
            'last_done': datetime.fromtimestamp(max(r['mtime'] for r in done)).strftime('%m-%d %H:%M') if done else '',
        })
    companies.sort(key=lambda c: (-c['done'] / c['total'], -c['total']))
    excluded = sorted(({'code': code, 'name': g[0]['company'], 'n': len(g)} for code, g in thin_by.items() if code not in by),
                      key=lambda c: -c['n'])  # 每个季度都不足 min_per_quarter 篇、整家不抽的公司

    done_rows = [r for r in rows if r['status'] == 'done']
    hours = defaultdict(int)  # 存档写入时间，按小时
    for r in done_rows:
        hours[datetime.fromtimestamp(r['mtime']).strftime('%m-%d %H:00')] += 1
    run = last_paid_run(out.log)
    if run:  # 那一轮新做完的研报（按存档写入时间认）的段落数，用来按段落折算
        t0, t1 = (datetime.strptime(run[k], '%Y-%m-%d %H:%M:%S').timestamp() for k in ('start', 'end'))
        batch = [r for r in done_rows if t0 <= r['mtime'] <= t1 + 60]
        run.update(reports=len(batch), paras=sum(r['paras'] for r in batch))
        if not run['paras']:
            run = None
    n_graphs = sum(1 for p in out.graphs.rglob('*.js') if p.parent != out.graphs) if out.graphs.exists() else 0
    cmp_path = out.compare / 'report_metrics.parquet'
    summary = {
        'generated': datetime.now().strftime('%Y-%m-%d %H:%M'),
        'latest': datetime.fromtimestamp(max(r['mtime'] for r in done_rows)).strftime('%Y-%m-%d %H:%M') if done_rows else '',
        'min_per_quarter': min_per_quarter,
        'n_meta': len(meta), 'n_ok': len(ok), 'n_stock_ok': sum(bool(r.get('is_stock_report')) for r in ok),
        'n_sel_noq': len(loose), 'n_sel': len(sel), 'n_thin': len(thin_ids),
        'n_done': len(done_rows), 'n_stale': sum(r['status'] == 'stale' for r in rows),
        'n_todo': sum(r['status'] != 'done' for r in rows),
        'n_archives': len(archives), 'n_orphan': len(orphan), 'orphan_thin': len(orphan & thin_ids),
        'paras_total': sum(r['paras'] for r in rows), 'paras_done': sum(r['paras'] for r in done_rows),
        'n_concepts_total': sum(r['concepts'] for r in done_rows), 'n_edges_total': sum(r['edges'] for r in done_rows),
        'quarters': {qq: [sum(r['quarter'] == qq for r in rows), sum(r['quarter'] == qq for r in done_rows)] for qq in quarters},
        'all_chars_mean': round(sum(r['chars'] for r in rows) / len(rows)) if rows else 0,
        'wave1_n': len(wave1), 'wave1_in_sel': sum(c['wave1'] for c in companies),
        'wave1_excluded': sorted({c['code'] for c in excluded} & wave1),
        'run': run, 'timeline': sorted(hours.items()),
        'n_graphs': n_graphs, 'n_compare': pq.read_metadata(cmp_path).num_rows if cmp_path.exists() else 0,
    }
    return {'summary': summary, 'companies': companies, 'excluded': excluded}


def main(out_dir, paras_path, reports_path, select, min_per_quarter, out_html):
    """out_dir：抽取的输出目录；paras_path、reports_path：抽取时用的段落表和研报元数据表；select、min_per_quarter：
    抽取脚本的筛选条件（同 extract_logic_chain.py）；out_html：写出的页面。"""
    data = collect(out_dir, paras_path, reports_path, select, min_per_quarter)
    html = TEMPLATE.replace('__DATA__', json.dumps(data, ensure_ascii=False, separators=(',', ':')).replace('</', '<\\/'))
    out_html = Path(out_html)
    out_html.write_text(html, encoding='utf-8')
    s = data['summary']
    print(f'已写出 {out_html}（{len(html.encode("utf-8")) / 1024:.0f} KB）：{len(data["companies"])} 家公司，'
          f'完成 {s["n_done"]}/{s["n_sel"]} 篇，存档指纹对不上的 {s["n_stale"]} 篇')
