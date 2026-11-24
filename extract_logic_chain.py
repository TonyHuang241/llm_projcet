"""第二步：研报逻辑链抽取。读 clear_txt.py 清理好的正文段落，按研报元数据表选出要抽取的研报（默认只要个股公司研报，
见下面的 select），用大模型把每篇研报抽成一张五层估值逻辑图，
结果写到 output/：最外层的 研报逻辑图.html 是入口（双击打开，选中一篇研报时才读取它的数据并绘图），
结果数据在 output/data/；大模型缓存、各篇研报的存档和进度日志在 output/cache/：已完成的研报（存档有效的）重跑时直接读回、
不再处理；待处理的研报逐篇处理、同时在处理 WORKERS 篇，做完一篇即存档并接上下一篇，中断后重新运行接着跑；
重新处理的研报也只补做缺少的调用。
抽取流程见 src/report_logic.py，框架设计见 docs/研报逻辑图抽取框架设计.md。

在仓库根目录运行：
    python extract_logic_chain.py                按下面的运行参数抽取逻辑链
    python extract_logic_chain.py --limit 20     在选中的研报里随机抽 20 篇试跑（种子固定，同一个种子下 20 篇 ⊂ 100 篇 ⊂ 全部，
                                                 放大样本时做完的直接沿用）；--seed 1 换一组
    python extract_logic_chain.py --companies 600519.SH,002415.SZ   只抽这几只股票的研报（覆盖下面的 COMPANIES）；--companies all 不限公司
    python extract_logic_chain.py --budget 30    本次运行最多花 30 美元（按中转站实际扣费），到了就停，做完的都已存档
    python extract_logic_chain.py --dry-run      只筛选、统计待处理的研报，不调用大模型（可与 --companies、--limit 等连用）
    python extract_logic_chain.py --fresh        先清空输出目录里上一次的结果和各篇存档再跑（大模型调用缓存保留，不重复花钱）
    python extract_logic_chain.py --all          不筛选，段落表里的研报全部抽取
    python extract_logic_chain.py --input data/xxx.parquet --output output_sample   换输入、输出（试跑时免得覆盖正式结果）
    python extract_logic_chain.py --task compare     抽取之后：同一公司不同研报的逻辑框架对比
    python extract_logic_chain.py --task intro_page  抽取和对比之后：生成介绍页 docs/逻辑抽取与分歧度量介绍.html
    python extract_logic_chain.py --task coverage    抽取进度页 docs/研报概念数据整理进度.html：按公司统计各篇存档做到哪了（不调用大模型，
                                                     默认不限公司；抽取在后台运行时也能跑）
"""
import argparse
import sys

from src.config import DATA_DIR, OUTPUT_DIR, ROOT_DIR
from src.report_text import reports_table_path

# ==================== 运行参数 ====================
PARAGRAPHS = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # 输入：clear_txt.py 写出的正文段落（至少含 report_id、para_id、text 三列）
REPORTS = None  # 研报元数据表；None 表示和输入配套的那张（clear_txt.py 同时写出，文件名里的 paragraphs 换成 reports）
OUT_DIR = OUTPUT_DIR  # 输出目录
LIMIT = None  # 试跑时在选中的研报里随机抽多少篇，None 表示全部
COMPANIES = ['600519.SH', '002415.SZ', '600036.SH']  # 只抽这些股票的研报（贵州茅台、海康威视、招商银行）；空列表表示不限公司
MIN_PER_QUARTER = 2  # 同一家公司同一个季度（按发布日期）不足这么多篇的不抽：只有 1 篇时没有别的研报可比；1 表示不按这条筛
WORKERS = 120  # 同时在处理的研报篇数，也是同时在途的大模型调用数（线程数）上限；None 表示用 .env 里的 CONCURRENCY
ORDER = 'company'  # 待处理研报的先后：'company' 按公司排，一家做完再做下一家（待处理段落少的公司在前）；'length' 全部研报从长到短
BUDGET = None  # 本次运行最多花多少美元，None 表示不设上限；需要中转站支持额度查询（new-api 类的 /api/usage/token）
SEED = 0  # 随机抽样的种子：同一个种子下抽得少的总是抽得多的子集（按 md5(种子:report_id) 排序取前 LIMIT 篇）
TASK = 'extract'  # 默认抽取逻辑链；另外两个功能见下方 TASKS


def select(r):
    """要抽取逻辑链的研报。r 是研报元数据表的一行（report_id、date、broker、report_type、is_stock_report、company、stock_code、
    analysts、n_paragraphs 等，各列含义见 src/report_meta.py），返回真的才抽取。默认只要个股公司研报，而且股票代码、
    公司名、券商都明确识别出来了（缺任何一项的不抽，识别不出券商的也在这里剔除）；COMPANIES 不为空时只要其中的股票。
    选中之后还会按 MIN_PER_QUARTER 剔除“公司当季篇数不足”的研报（按组判断，见 src/report_logic.py 的 select_reports）。
    要加条件直接改这里，例如只要某段时间：and '20240101' <= r['date'] <= '20240630'。"""
    return (r['is_stock_report'] and r['stock_code'] and r['company'] and r['broker']
            and (not COMPANIES or r['stock_code'] in COMPANIES))


# 框架对比的参数（各列定义见 src/report_compare.py 开头）
EDGE_SET = 'all'  # 'all' 全部边；'text' 去掉逻辑补全边（稳健性检验）
PEERS = 'all'  # 逐篇指标的同行：'all' 同公司其他全部研报；'prior' 只用更早发布的（避免前视偏差）
WINDOW_DAYS = None  # PEERS='prior' 时往前看多少天，None 为不限
# ==================================================


def extract(args):
    """逻辑链抽取：概念分层填表 -> 原文连边 -> 整理概念表 -> 悬挂补全、并列复核 -> 包含关系缩减 -> 子图补全
    -> 剪枝 -> 全局去重 -> 写出每篇的逻辑图数据、parquet 和入口页面。"""
    from src import report_logic
    report_logic.main(paras_path=args.input, out_dir=args.output, limit=args.limit, seed=args.seed, workers=args.workers,
                      fresh=args.fresh, min_per_quarter=MIN_PER_QUARTER, dry_run=args.dry_run, budget=args.budget, order=ORDER,
                      reports_path=None if args.all else args.reports or reports_table_path(args.input), select=select)


def compare(args):
    """同一公司不同研报的逻辑框架对比（每家公司调用一次大模型把概念归成主题），结果写到 <输出目录>/data/compare/。"""
    from src import report_compare
    report_compare.main(out_dir=args.output, edge_set=EDGE_SET, peers=PEERS, window_days=WINDOW_DAYS, workers=args.workers)


def intro_page(args):
    """介绍页（不调用大模型）：抽取流程、用研报样例说明抽取可信度和分歧度量，页面模板在 src/viz/intro_template.html。"""
    from src.viz import plot_intro
    plot_intro.main(out_dir=args.output, paras_path=args.input, out_html=ROOT_DIR / 'docs' / '逻辑抽取与分歧度量介绍.html')


def coverage(args):
    """抽取进度页（不调用大模型）：按上面的筛选条件列出要抽取的研报，核对各篇存档，按公司统计做完、没做完的篇数，
    页面模板在 src/viz/coverage_template.html。不给 --companies 时不限公司（抽取正式跑的就是全部公司）。"""
    from src.viz import plot_coverage
    if args.companies is None:
        COMPANIES[:] = []
    plot_coverage.main(out_dir=args.output, paras_path=args.input, reports_path=args.reports or reports_table_path(args.input),
                       select=select, min_per_quarter=MIN_PER_QUARTER, out_html=ROOT_DIR / 'docs' / '研报概念数据整理进度.html')


TASKS = {'extract': extract, 'compare': compare, 'intro_page': intro_page, 'coverage': coverage}

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='研报逻辑链抽取')
    ap.add_argument('--task', default=TASK, choices=list(TASKS), help='要运行的功能，默认抽取逻辑链')
    ap.add_argument('--input', default=PARAGRAPHS, help='正文段落 parquet')
    ap.add_argument('--reports', default=REPORTS, help='研报元数据表，默认用和输入配套的那张')
    ap.add_argument('--all', action='store_true', help='不按元数据表筛选，段落表里的研报全部抽取')
    ap.add_argument('--output', default=OUT_DIR, help='输出目录')
    ap.add_argument('--limit', type=int, default=LIMIT, help='在选中的研报里按种子随机抽多少篇试跑（同一个种子下嵌套）')
    ap.add_argument('--seed', type=int, default=SEED, help='随机抽样的种子')
    ap.add_argument('--companies', default=None, help='只抽这些股票代码的研报，逗号分隔；all 表示不限公司；不给时用 COMPANIES')
    ap.add_argument('--workers', type=int, default=WORKERS, help='同时在途的大模型调用数（线程数）')
    ap.add_argument('--budget', type=float, default=BUDGET, help='本次运行最多花多少美元（按中转站实际扣费），到了就停')
    ap.add_argument('--dry-run', action='store_true', help='只筛选、统计待处理的研报，不调用大模型、不写文件')
    ap.add_argument('--fresh', action='store_true', help='先清空输出目录里上一次的结果和各篇存档再跑（调用缓存保留）')
    args = ap.parse_args()
    if args.companies is not None:
        COMPANIES[:] = [] if args.companies.lower() == 'all' else [x.strip().upper() for x in args.companies.split(',') if x.strip()]
    TASKS[args.task](args)
