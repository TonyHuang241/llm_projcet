"""研报概念有向无环图抽取（和 extract_logic_chain.py 的五层估值逻辑图是两套独立的流程）。读 clear_txt.py 清理好的正文段落，
按研报元数据表选出要抽取的个股研报，段落合并成块后用大模型先抽概念、再在原文里连“A 影响 B”的边，删边去环得到有向无环图；
不预设会计科目、估值等固定概念，也不做补全。流程见 src/concept_dag.py，提示词见 src/prompts/concept_dag_prompt.md。

结果写到 output_dag/（和 output/ 分开）：最外层的 概念DAG.html 是入口（双击打开，选中一篇研报时才读取它的数据并绘图），
结果数据在 output_dag/data/，大模型调用缓存、各篇存档和进度日志在 output_dag/cache/；中断后重新运行接着跑。

在仓库根目录运行：
    python extract_concept_dag.py                    按下面的运行参数抽取（默认是 COMPANIES 里的几家公司）
    python extract_concept_dag.py --dry-run          只统计选中的研报、段落合并后的块数，不调用大模型
    python extract_concept_dag.py --companies 600519.SH,002415.SZ   只抽这几只股票的研报；--companies all 不限公司
    python extract_concept_dag.py --deep-only 002594.SZ              这几只股票只抽深度报告（默认用 DEEP_ONLY）
    python extract_concept_dag.py --companies all --deep-only all    全部公司都只抽深度报告（全库 975 篇：有类型标注、正文超过 5000 字）
    python extract_concept_dag.py --min-deep-chars 3000              深度报告正文超过 3000 字就抽（默认用 MIN_DEEP_CHARS）
    python extract_concept_dag.py --limit 20         在选中的研报里随机抽 20 篇试跑（种子固定，同一个种子下嵌套）
    python extract_concept_dag.py --workers 50       同时在途的大模型调用数
    python extract_concept_dag.py --chunk-chars 3000 合并段落时每块的字数上限（改了之后各篇重新处理）
    python extract_concept_dag.py --fresh            先清空上一次的结果和各篇存档（调用缓存保留）
    python extract_concept_dag.py --archived-only    只用已有存档写输出、不调用大模型（少数研报的调用一直卡住时用，没存档的下次再补）
"""
import argparse
import sys

from src.config import DATA_DIR, ROOT_DIR
from src.report_meta import deep_line
from src.report_text import read_text, reports_table_path

# ==================== 运行参数 ====================
PARAGRAPHS = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # 输入：clear_txt.py 写出的正文段落
TXT_DIR = DATA_DIR / 'HS300_reports_2024'  # 研报纯文本，DEEP_ONLY 的股票要读原文首页判断是不是深度报告
REPORTS = None  # 研报元数据表；None 表示和输入配套的那张
OUT_DIR = ROOT_DIR / 'output_dag'  # 输出目录（和五层逻辑图的 output/ 分开）
# 试跑用的几家公司：研报 6-10 篇、行业各不相同，五层逻辑图那边也都已做完，方便对照
COMPANIES = ['603596.SH', '603885.SH', '600089.SH', '603919.SH', '600016.SH', '600196.SH', '002594.SZ']
# 伯特利、吉祥航空、特变电工、金徽酒、民生银行、复星医药、比亚迪；空列表表示不限公司
# 这些股票只抽深度报告（读原文首页的类型标注判断，深度、首次覆盖、公司专题等，见 src/report_meta.py 的 deep_line）：
# 比亚迪研报 266 篇，深度报告 19 篇，是全库深度报告最多的公司；写 'ALL' 表示所有股票都只抽深度报告
DEEP_ONLY = ['002594.SZ']
# 深度报告还要正文（清理后、去掉风险提示）超过这么多字：首页有深度报告的类型标注、正文却很短的多半不是真正的深度报告
# （正文没提取出来，或只是一两页的首次覆盖点评、摘要）。1252 篇有标注的里 277 篇不超过 5000 字，剩 975 篇、421 家公司
MIN_DEEP_CHARS = 5000
LIMIT = None  # 在选中的研报里随机抽多少篇，None 表示全部
SEED = 0
WORKERS = 50  # 同时在途的大模型调用数，也是同时在处理的研报篇数
CHUNK_CHARS = 2500  # 合并段落时每块的字数上限：块越大调用越少，但一次要读的原文越长、可能漏抽
MIN_PER_QUARTER = 1  # 同一家公司同一个季度不足这么多篇的不抽；1 表示不按这条筛（公司的研报全部抽取）


def select(r):
    """要抽取的研报（r 是研报元数据表的一行）：个股公司研报，股票代码、公司名、券商都识别出来了，COMPANIES 不为空时只要其中的股票；
    DEEP_ONLY 里的股票只要深度报告。"""
    return (r['is_stock_report'] and r['stock_code'] and r['company'] and r['broker']
            and (not COMPANIES or r['stock_code'] in COMPANIES)
            and (r['stock_code'] not in DEEP_ONLY and 'ALL' not in DEEP_ONLY or is_deep(r)))


def is_deep(r):
    """读原文首页看是不是深度报告（元数据表里没有这一列）。"""
    text, _ = read_text(TXT_DIR / r['file'])
    return bool(deep_line([x.strip() for x in text.splitlines() if x.strip()]))


def keep(r):
    """深度报告正文不超过 MIN_DEEP_CHARS 字的不抽
    （r 是读入正文后的研报；只管 DEEP_ONLY 里的股票，其他公司的研报照常全部抽取）。"""
    if r.meta['stock_code'] not in DEEP_ONLY and 'ALL' not in DEEP_ONLY:
        return True
    return sum(len(p['text']) for p in r.paras) > MIN_DEEP_CHARS
# ==================================================


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='研报概念有向无环图抽取')
    ap.add_argument('--input', default=PARAGRAPHS, help='正文段落 parquet')
    ap.add_argument('--reports', default=REPORTS, help='研报元数据表，默认用和输入配套的那张')
    ap.add_argument('--all', action='store_true', help='不按元数据表筛选，段落表里的研报全部抽取')
    ap.add_argument('--output', default=OUT_DIR, help='输出目录')
    ap.add_argument('--companies', default=None, help='只抽这些股票代码的研报，逗号分隔；all 表示不限公司；不给时用 COMPANIES')
    ap.add_argument('--deep-only', default=None, help='这些股票代码只抽深度报告，逗号分隔；all 表示所有股票，none 表示不限；不给时用 DEEP_ONLY')
    ap.add_argument('--min-deep-chars', type=int, default=MIN_DEEP_CHARS, help='深度报告正文要超过多少字才抽（只管只抽深度报告的股票）')
    ap.add_argument('--limit', type=int, default=LIMIT, help='在选中的研报里按种子随机抽多少篇试跑')
    ap.add_argument('--seed', type=int, default=SEED, help='随机抽样的种子')
    ap.add_argument('--workers', type=int, default=WORKERS, help='同时在途的大模型调用数')
    ap.add_argument('--chunk-chars', type=int, default=CHUNK_CHARS, help='合并段落时每块的字数上限')
    ap.add_argument('--fresh', action='store_true', help='先清空上一次的结果和各篇存档（调用缓存保留）')
    ap.add_argument('--dry-run', action='store_true', help='只统计选中的研报和块数，不调用大模型')
    ap.add_argument('--archived-only', action='store_true', help='只用已有存档写输出、不调用大模型，没有存档的研报这次不进输出')
    args = ap.parse_args()
    if args.companies is not None:
        COMPANIES[:] = [] if args.companies.lower() == 'all' else [x.strip().upper() for x in args.companies.split(',') if x.strip()]
    if args.deep_only is not None:
        DEEP_ONLY[:] = [] if args.deep_only.lower() == 'none' else [x.strip().upper() for x in args.deep_only.split(',') if x.strip()]
    MIN_DEEP_CHARS = args.min_deep_chars
    keep.note = f'深度报告正文不超过 {MIN_DEEP_CHARS} 字的不抽'  # 写进日志
    from src import concept_dag
    concept_dag.main(paras_path=args.input, out_dir=args.output,
                     reports_path=None if args.all else args.reports or reports_table_path(args.input), select=select,
                     limit=args.limit, seed=args.seed, workers=args.workers, chunk_chars=args.chunk_chars,
                     min_per_quarter=MIN_PER_QUARTER, fresh=args.fresh, dry_run=args.dry_run, keep=keep,
                     archived_only=args.archived_only)
