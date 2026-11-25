"""步骤 graph：研报概念有向无环图抽取。选抽取版本、选这次要处理的研报，再分发到版本包（src/graph/<版本>/extract.py）。

两个概念（详见 README 和 src/samples/__init__.py）：
  抽取版本（version）  一套抽取规则 = 版本包里的抽取代码 + 提示词 + 规则参数（每块字数、ARCHIVE_VERSION）。每个版本一个库
                       output/graph/<版本>/，同一版本下每篇研报的逻辑结构只存一份；补充抽取、重新抽取都往同一个库里加或改。
                       只有明确要开新版本时才把版本包整个复制成新的一套（见 README 的“开新版本”），平时直接改当前版本的代码。
  样本（sample）       分析用哪些研报：一张“研报 ID → 取自哪个版本的库”的清单（src/samples/<样本>.csv），不存逻辑结构。

库的全集是全部能识别公司的个股研报（src/graph/store.py 的 in_library）。一次运行只处理：--sample 给的样本里取自该版本的研报
（样本的主版本照清单；给的是别的版本时样本的全部研报，见 src/samples 的 assignment），或 --report-ids 给的几篇；其中缺有效
存档的才调用大模型，其余只读存档。写输出时写库里全部有有效存档的研报，所以补抽新样本时已有的研报原样保留、不重复存。
不给 --sample 也不给 --report-ids 时不抽取，只用存档重写库的输出。

库的布局：最外层的 概念DAG.html 是入口（双击打开），结果数据在 data/，大模型调用缓存、各篇存档和进度日志在 cache/；
中断后重新运行接着跑。下游读结果用 src/graph/load.py（按样本读）。抽完后可以再跑 python run.py unify 统一概念名称。

在仓库根目录运行：
    python run.py graph --sample hs300_full --dry-run   统计要处理的研报、已有存档和待抽的篇数、块数，不调用大模型
    python run.py graph --sample hs300_full             补抽样本里缺存档的研报（默认版本 v1），结果并入 output/graph/v1/
    python run.py graph --report-ids <研报ID>           只处理这一篇（逗号分隔可以给几篇），结果并入同一个库
    python run.py graph --sample hs300_full --limit 20  在要处理的研报里按种子随机抽 20 篇先试（--seed 换种子；同一个种子下嵌套）
    python run.py graph --archived-only                 只用已有存档重写库的输出、不调用大模型（带 --sample 时缺存档的也不抽）
    python run.py graph --report-ids <ID> --chunk-chars 3000   换每块字数的试跑：其余研报的存档会对不上，所以只在日志里
                                                        报告这几篇的结果，不存档、不写输出
    python run.py graph --version v2 --sample hs300_full   用新版本 v2 的规则把样本的全部研报抽进 output/graph/v2/
    python run.py graph --fresh --yes                   清空库里的结果和全部存档（调用缓存保留）；不加 --yes 只报错说明后果
"""
import importlib

from src.config import DEFAULT_VERSION, graph_dir

# 抽取版本登记表：package 是版本包（None 表示只有数据、没有对应代码）；readonly 的版本不能再往库里抽取或统一名称
VERSIONS = {
    'v1': dict(package='src.graph.v1', readonly=False,
               desc='现行规则：每块 5000 字，src/graph/v1/ 的代码和提示词；库里是全部公司的深度报告（重构前的 deep 运行）'),
    'v0': dict(package=None, readonly=True,
               desc='早期试跑（2026-11-19，重构前的 pilot 运行，7 家公司 65 篇）：每块 2500 字、重构前的代码，'
                    '代码在 git 标签 pre-cleanup 里，现在已不能完全复现；只读，不再往里抽'),
}


def version_spec(version):
    if version not in VERSIONS:
        raise SystemExit(f'没有抽取版本 {version}；已登记：{"、".join(VERSIONS)}（src/graph/__init__.py 的 VERSIONS）')
    return VERSIONS[version]


def version_module(version, name, action):
    """版本包里的模块（extract、unify）；只读或没有代码的版本报错说明。"""
    s = version_spec(version)
    if s['readonly'] or s['package'] is None:
        raise SystemExit(f'抽取版本 {version} 是只读的，不能{action}：{s["desc"]}')
    return importlib.import_module(f'{s["package"]}.{name}')


def add_args(ap):
    ap.description = ('研报概念有向无环图抽取：选抽取版本（库 output/graph/<版本>/）和这次要处理的研报（--sample 或 --report-ids），'
                      '缺存档的才调用大模型，其余只读存档，输出写库里全部有有效存档的研报；两个都不给时只用存档重写输出')
    ap.add_argument('--version', default=DEFAULT_VERSION, help=f'抽取版本（{"、".join(VERSIONS)}），默认 {DEFAULT_VERSION}')
    ap.add_argument('--sample', default=None,
                    help='处理这个样本里取自该版本的研报（样本见 python run.py sample）；版本不是样本的主版本时处理样本的全部研报')
    ap.add_argument('--report-ids', default=None, help='只处理这几篇研报（report_id，逗号分隔，要在库的全集里）；不能和 --sample 同时给')
    ap.add_argument('--limit', type=int, default=None, help='在要处理的研报里按种子随机抽多少篇试跑')
    ap.add_argument('--seed', type=int, default=0, help='随机抽样的种子，默认 0')
    ap.add_argument('--workers', type=int, default=None, help='同时在途的大模型调用数；默认用版本包里的 WORKERS（v1 是 100）')
    ap.add_argument('--chunk-chars', type=int, default=None,
                    help='覆盖版本的每块字数（v1 是 5000）：库里其余研报的存档会对不上，所以自动改成试跑，不存档、不写输出')
    ap.add_argument('--fresh', action='store_true', help='先清空库里的结果和全部存档（调用缓存保留）；要同时给 --yes')
    ap.add_argument('--yes', action='store_true', help='确认 --fresh')
    ap.add_argument('--dry-run', action='store_true', help='只统计要处理的研报、已有存档和待抽的篇数、块数，不调用大模型')
    ap.add_argument('--archived-only', action='store_true', help='只用已有存档写输出、不调用大模型，没有存档的研报这次不进输出')


def run(args):
    from src import samples
    extract = version_module(args.version, 'extract', '抽取')
    pkg = importlib.import_module(VERSIONS[args.version]['package'])
    if args.sample and args.report_ids:
        raise SystemExit('--sample 和 --report-ids 只能给一个')
    if args.fresh and not args.dry_run and not args.yes:
        raise SystemExit(f'--fresh 会删掉 {graph_dir(args.version)} 下的 data/ 和 cache/reports/（库里全部研报的存档；'
                         '调用缓存保留，重抽不花钱但很慢），所有用到这个版本的样本都要重抽才能恢复。确定的话同时给 --yes')
    if args.sample:
        t, _ = samples.assignment(args.sample, args.version)
        targets = list(t.report_id[t.version == args.version])
        note = f'；样本 {args.sample}：{len(t)} 篇里取自 {args.version} 的 {len(targets)} 篇是这次的处理范围'
        if not targets:
            raise SystemExit(f'样本 {args.sample} 里没有取自 {args.version} 的研报')
    elif args.report_ids:
        targets = [x.strip() for x in args.report_ids.split(',') if x.strip()]
        note = f'；--report-ids：处理 {len(targets)} 篇'
    else:
        targets, note = [], '；没有给 --sample 或 --report-ids：不抽取，只用存档重写库的输出'
    chunk_chars = pkg.CHUNK_CHARS if args.chunk_chars is None else args.chunk_chars
    if chunk_chars != pkg.CHUNK_CHARS:
        note += f'；每块字数改成 {chunk_chars}（版本 {args.version} 是 {pkg.CHUNK_CHARS}）'
    extract.main(graph_dir(args.version), targets=targets, chunk_chars=chunk_chars,
                 workers=pkg.WORKERS if args.workers is None else args.workers, limit=args.limit, seed=args.seed,
                 fresh=args.fresh, dry_run=args.dry_run, archived_only=args.archived_only, note=note)
