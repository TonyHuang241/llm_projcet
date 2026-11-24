"""步骤 text：研报纯文本清理。把 data/HS300_reports_2024/ 下的 txt 清理成正文段落，剔除图表、表格、侧栏、页眉页脚、
风险提示和免责声明，结果写到 data/ 下的 parquet（旁边同名 .txt 便于人工检查）。剔除规则见 src/text/clean.py。

同时写一张研报元数据表（data/hs300_2024_reports.parquet，另有同名 .csv 可用 Excel 打开），每篇研报一行：发布日期、券商、
报告类型、是不是个股公司研报、公司和股票代码、分析师等（取法见 src/text/meta.py）。它和段落表用 report_id 对应，
概念 DAG 抽取（python run.py graph）按这张表筛选要抽取的研报。下游读这两张表用 src/text/load.py。

在仓库根目录运行（不调用大模型；多进程并行，默认用满 CPU 核心；运行时显示进度条）：
    python run.py text              按下面的运行参数运行
    python run.py text --limit 100  只跨月份随机抽 100 篇试跑，结果写到 *_sample100.parquet，不覆盖全量结果

清理完再运行 python run.py graph 抽取概念 DAG。
"""
from src.config import PARAGRAPHS, TXT_DIR

# ==================== 运行参数 ====================
OUT_PATH = PARAGRAPHS  # 清理后的正文段落，python run.py graph 的输入（元数据表写在旁边，见 src/text/clean.py 的 reports_table_path）
LIMIT = None  # 试跑时跨月份随机抽多少篇（固定种子），None 表示全部
WORKERS = None  # 并行的进程数，None 表示用满 CPU 核心（Windows 上最多 60 个），1 表示不开子进程
# ==================================================


def add_args(ap):
    ap.description = '研报纯文本 -> 正文段落 parquet（同时写研报元数据表）'
    ap.add_argument('--input', default=TXT_DIR, help='txt 所在目录（递归查找 *.txt）')
    ap.add_argument('--output', default=None, help='输出 parquet 路径；不指定时全量写到 OUT_PATH，试跑写到 *_sampleN.parquet')
    ap.add_argument('--limit', type=int, default=LIMIT, help='只跨月份随机抽这么多篇试跑')
    ap.add_argument('--seed', type=int, default=0, help='抽样的随机种子')
    ap.add_argument('--workers', type=int, default=WORKERS, help='并行的进程数，默认用满 CPU 核心')


def run(args):
    from src.text import clean
    out = args.output or (OUT_PATH.with_name(f'{OUT_PATH.stem}_sample{args.limit}.parquet') if args.limit else OUT_PATH)
    clean.main(args.input, out, args.limit, args.seed, args.workers)
