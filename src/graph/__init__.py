"""步骤 graph：研报概念有向无环图抽取。读 python run.py text 清理好的正文段落，按研报元数据表选出要抽取的个股研报，
段落合并成块后用大模型先抽概念、再在原文里连“A 影响 B”的边，删边去环得到有向无环图；不预设会计科目、估值等固定概念，
也不做补全。流程见 src/graph/extract.py，提示词见 src/graph/prompts/extract.md。抽完后可以再跑 python run.py unify
（src/graph/unify.py）把同一家公司不同研报里同一个概念的不同叫法统一成一个。

结果按运行名分目录写到 output/graph/<运行名>/：最外层的 概念DAG.html 是入口（双击打开，选中一篇研报时才读取它的数据并绘图），
结果数据在 data/，大模型调用缓存、各篇存档和进度日志在 cache/；中断后重新运行接着跑。下游读结果用 src/graph/load.py。
各次运行的选样和参数写在下面的 RUNS 里，--run 选一个，命令行选项可以覆盖其中任何一项。

在仓库根目录运行：
    python run.py graph --dry-run                    只统计选中的研报、段落合并后的块数，不调用大模型（默认 --run deep）
    python run.py graph --run pilot                  按试跑那次的配置抽取（7 家公司）
    python run.py graph --companies 600519.SH,002415.SZ   只抽这几只股票的研报；--companies all 不限公司
    python run.py graph --deep-only 002594.SZ        这几只股票只抽深度报告；--deep-only all 全部只抽深度报告，none 不限
    python run.py graph --min-deep-chars 3000        深度报告正文超过 3000 字就抽
    python run.py graph --limit 20                   在选中的研报里随机抽 20 篇试跑（种子固定，同一个种子下嵌套）
    python run.py graph --workers 50                 同时在途的大模型调用数
    python run.py graph --chunk-chars 3000           合并段落时每块的字数上限（改了之后各篇重新处理）
    python run.py graph --fresh                      先清空上一次的结果和各篇存档（调用缓存保留）
    python run.py graph --archived-only              只用已有存档写输出、不调用大模型（少数研报的调用一直卡住时用，没存档的下次再补）
"""
from src.config import PARAGRAPHS, TXT_DIR, out_dir

# ==================== 运行参数 ====================
# 每次运行一项：输出到 output/graph/<运行名>/。各项的含义：
#   companies        只抽这些股票的研报，空列表表示不限公司
#   deep_only        这些股票只抽深度报告（读原文首页的类型标注判断，深度、首次覆盖、公司专题等，见 src/text/meta.py 的
#                    deep_line）；写 'ALL' 表示所有股票都只抽深度报告
#   min_deep_chars   深度报告还要正文（清理后、去掉风险提示）超过这么多字：首页有深度报告的类型标注、正文却很短的多半不是
#                    真正的深度报告（正文没提取出来，或只是一两页的首次覆盖点评、摘要）；只管 deep_only 里的股票
#   chunk_chars      合并段落时每块的字数上限：块越大调用越少，但一次要读的原文越长、可能漏抽
#   workers          同时在途的大模型调用数，也是同时在处理的研报篇数
#   limit、seed      在选中的研报里随机抽多少篇（None 表示全部）、抽样种子
#   min_per_quarter  同一家公司同一个季度不足这么多篇的不抽；1 表示不按这条筛（公司的研报全部抽取）
#   unify_companies  python run.py unify 默认统一哪些股票的研报，空列表表示这次运行里的全部研报（各公司分别统一）
RUNS = {
    # 全样本深度报告（重构前在概念 DAG 输出目录的 deep/ 子目录下）：当时用重构前的抽取入口脚本跑的，参数是
    #   --companies all --deep-only all --chunk-chars 5000 --workers 100 --output <deep 子目录>
    # 选中 975 篇、3200 块，974 篇有存档（视源股份 1 篇的一块回复始终不是合法 JSON，跳过，见 data/skipped_reports.csv）；
    # 名称统一用的是 --companies all --workers 100
    'deep': dict(companies=[], deep_only=['ALL'], min_deep_chars=5000, chunk_chars=5000, workers=100, limit=None, seed=0,
                 min_per_quarter=1, unify_companies=[]),
    # 试跑的 7 家公司（重构前是概念 DAG 输出目录根下的 data/、cache/）：研报 6-10 篇、行业各不相同，方便对照；
    # 伯特利、吉祥航空、特变电工、金徽酒、民生银行、复星医药、比亚迪。比亚迪研报 266 篇，只抽深度报告（全库深度报告最多的公司）。
    # 当时是重构前抽取入口脚本的默认参数（每块 2500 字、并发 50），2026-11-19 抽了 65 篇；那时还没有 min_deep_chars
    # 这一条，深度报告的判断（deep_line）和文本清理后来也改过，所以这套参数不能完全复现当时的选样：现在选出 68 篇，
    # 其中存档还有效的只有 56 篇（8 篇的段落文字在试跑之后变了、存档指纹对不上，4 篇从来没抽过；当时 65 篇里另有 1 篇
    # 现在因正文不足 5000 字不再选中）。所以不要对 pilot 跑 --archived-only：它会把 data/ 下的结果改写成 56 篇。
    # 名称统一只统一了比亚迪的 16 篇
    'pilot': dict(companies=['603596.SH', '603885.SH', '600089.SH', '603919.SH', '600016.SH', '600196.SH', '002594.SZ'],
                  deep_only=['002594.SZ'], min_deep_chars=5000, chunk_chars=2500, workers=50, limit=None, seed=0,
                  min_per_quarter=1, unify_companies=['002594.SZ']),
}
DEFAULT_RUN = 'deep'
# ==================================================


def make_filters(companies, deep_only, min_deep_chars, txt_dir=TXT_DIR):
    """按一次运行的配置给出 (select, keep)：select(r) 按研报元数据表的一行筛选，keep(r) 读入正文后再筛一遍。"""
    from src.text.load import read_text
    from src.text.meta import deep_line

    def select(r):
        """要抽取的研报（r 是研报元数据表的一行）：个股公司研报，股票代码、公司名、券商都识别出来了，companies 不为空时只要其中的股票；
        deep_only 里的股票只要深度报告。"""
        return (r['is_stock_report'] and r['stock_code'] and r['company'] and r['broker']
                and (not companies or r['stock_code'] in companies)
                and (r['stock_code'] not in deep_only and 'ALL' not in deep_only or is_deep(r)))

    def is_deep(r):
        """读原文首页看是不是深度报告（元数据表里没有这一列）。"""
        text, _ = read_text(txt_dir / r['file'])
        return bool(deep_line([x.strip() for x in text.splitlines() if x.strip()]))

    def keep(r):
        """深度报告正文不超过 min_deep_chars 字的不抽
        （r 是读入正文后的研报；只管 deep_only 里的股票，其他公司的研报照常全部抽取）。"""
        if r.meta['stock_code'] not in deep_only and 'ALL' not in deep_only:
            return True
        return sum(len(p['text']) for p in r.paras) > min_deep_chars

    keep.note = f'深度报告正文不超过 {min_deep_chars} 字的不抽'  # 写进日志
    return select, keep


def add_args(ap):
    ap.description = '研报概念有向无环图抽取（各次运行的配置见 src/graph/__init__.py 的 RUNS，命令行选项覆盖其中的项）'
    ap.add_argument('--run', default=DEFAULT_RUN, choices=sorted(RUNS),
                    help=f'按哪次运行的配置抽取，输出到 output/graph/<运行名>/，默认 {DEFAULT_RUN}')
    ap.add_argument('--input', default=PARAGRAPHS, help='正文段落 parquet')
    ap.add_argument('--reports', default=None, help='研报元数据表，默认用和输入配套的那张')
    ap.add_argument('--all', action='store_true', help='不按元数据表筛选，段落表里的研报全部抽取')
    ap.add_argument('--output', default=None, help='输出目录；不给时用 output/graph/<运行名>/')
    ap.add_argument('--companies', default=None, help='只抽这些股票代码的研报，逗号分隔；all 表示不限公司；不给时用运行配置')
    ap.add_argument('--deep-only', default=None, help='这些股票代码只抽深度报告，逗号分隔；all 表示所有股票，none 表示不限；不给时用运行配置')
    ap.add_argument('--min-deep-chars', type=int, default=None, help='深度报告正文要超过多少字才抽（只管只抽深度报告的股票）；不给时用运行配置')
    ap.add_argument('--limit', type=int, default=None, help='在选中的研报里按种子随机抽多少篇试跑；不给时用运行配置')
    ap.add_argument('--seed', type=int, default=None, help='随机抽样的种子；不给时用运行配置')
    ap.add_argument('--workers', type=int, default=None, help='同时在途的大模型调用数；不给时用运行配置')
    ap.add_argument('--chunk-chars', type=int, default=None, help='合并段落时每块的字数上限；不给时用运行配置')
    ap.add_argument('--fresh', action='store_true', help='先清空上一次的结果和各篇存档（调用缓存保留）')
    ap.add_argument('--dry-run', action='store_true', help='只统计选中的研报和块数，不调用大模型')
    ap.add_argument('--archived-only', action='store_true', help='只用已有存档写输出、不调用大模型，没有存档的研报这次不进输出')


def run(args):
    from src.text.load import reports_table_path
    cfg = dict(RUNS[args.run])
    if args.companies is not None:
        cfg['companies'] = [] if args.companies.lower() == 'all' else [x.strip().upper() for x in args.companies.split(',') if x.strip()]
    if args.deep_only is not None:
        cfg['deep_only'] = [] if args.deep_only.lower() == 'none' else [x.strip().upper() for x in args.deep_only.split(',') if x.strip()]
    for k in ('min_deep_chars', 'limit', 'seed', 'workers', 'chunk_chars'):
        if getattr(args, k) is not None:
            cfg[k] = getattr(args, k)
    select, keep = make_filters(cfg['companies'], cfg['deep_only'], cfg['min_deep_chars'])
    from src.graph import extract
    extract.main(paras_path=args.input, out_dir=args.output or out_dir('graph', args.run),
                 reports_path=None if args.all else args.reports or reports_table_path(args.input), select=select,
                 limit=cfg['limit'], seed=cfg['seed'], workers=cfg['workers'], chunk_chars=cfg['chunk_chars'],
                 min_per_quarter=cfg['min_per_quarter'], fresh=args.fresh, dry_run=args.dry_run, keep=keep,
                 archived_only=args.archived_only)
