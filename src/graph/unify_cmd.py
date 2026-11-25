"""步骤 unify 的入口：选抽取版本，对这个版本的库（output/graph/<版本>/）做跨研报概念名称统一，分发到 src/graph/<版本>/unify.py。
做法、提示词和输出见版本包里的 unify.py 开头。读库的 data/ 下的结果表，结果也写在那里，核对页面是库最外层的 概念统一.html；
库里新增了研报后重新运行即可，没变的候选组直接用缓存。

在仓库根目录运行：
    python run.py unify --dry-run                        只统计候选组，不调用大模型（默认版本 v1）
    python run.py unify                                  统一整个库
    python run.py unify --companies 600519.SH,002415.SZ  只统一这几家公司的研报（注意：结果表会改写成只有这几家）
"""
from src.config import DEFAULT_VERSION, graph_dir


def add_args(ap):
    from src.graph import VERSIONS
    ap.description = ('跨研报概念名称统一：在 python run.py graph 抽完的库上，把同一家公司不同研报里指完全相同的东西、'
                      '只是说法不同的概念名称统一成一个（营收 / 营业收入，PE / 市盈率），合并组再逐组复核。'
                      '读写 output/graph/<版本>/data/，核对页面是 output/graph/<版本>/概念统一.html')
    ap.add_argument('--version', default=DEFAULT_VERSION, help=f'抽取版本（{"、".join(VERSIONS)}），默认 {DEFAULT_VERSION}')
    ap.add_argument('--companies', default=None, help='只统一这些股票代码的研报，逗号分隔；不给时统一库里的全部研报')
    ap.add_argument('--workers', type=int, default=None, help='同时在途的大模型调用数；默认用版本包 unify.py 里的 WORKERS')
    ap.add_argument('--max-group', type=int, default=None, help='每个候选组最多多少个名称；默认用版本包 unify.py 里的 MAX_GROUP')
    ap.add_argument('--dry-run', action='store_true', help='只统计候选组，不调用大模型')


def run(args):
    from src.graph import version_module
    unify = version_module(args.version, 'unify', '统一名称')
    companies = [] if args.companies is None else [x.strip().upper() for x in args.companies.split(',') if x.strip()]
    unify.main(graph_dir(args.version), companies=companies,
               workers=unify.WORKERS if args.workers is None else args.workers,
               max_group=unify.MAX_GROUP if args.max_group is None else args.max_group, dry_run=args.dry_run)

