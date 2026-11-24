"""第一步：研报纯文本清理。把 data/HS300_reports_2024/ 下的 txt 清理成正文段落，剔除图表、表格、侧栏、页眉页脚、
风险提示和免责声明，结果写到 data/ 下的 parquet（旁边同名 .txt 便于人工检查）。剔除规则见 src/report_text.py。

同时写一张研报元数据表（data/hs300_2024_reports.parquet，另有同名 .csv 可用 Excel 打开），每篇研报一行：发布日期、券商、
报告类型、是不是个股公司研报、公司和股票代码、分析师等（取法见 src/report_meta.py）。它和段落表用 report_id 对应，
extract_logic_chain.py 按这张表筛选要抽取的研报。

在仓库根目录运行（不调用大模型；多进程并行，默认用满 CPU 核心；运行时显示进度条）：
    python clear_txt.py              按下面的运行参数运行
    python clear_txt.py --limit 100  只跨月份随机抽 100 篇试跑，结果写到 *_sample100.parquet，不覆盖全量结果

清理完再运行 python extract_logic_chain.py 抽取逻辑链。
"""
import argparse
import sys

from src import report_text
from src.config import DATA_DIR

# ==================== 运行参数 ====================
TXT_DIR = DATA_DIR / 'HS300_reports_2024'  # 研报纯文本（MM/DD/<GUID>.txt，递归查找）
OUT_PATH = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # 清理后的正文段落，extract_logic_chain.py 的输入
LIMIT = None  # 试跑时跨月份随机抽多少篇（固定种子），None 表示全部
WORKERS = None  # 并行的进程数，None 表示用满 CPU 核心（Windows 上最多 60 个），1 表示不开子进程
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='研报纯文本 -> 正文段落 parquet')
    ap.add_argument('--input', default=TXT_DIR, help='txt 所在目录（递归查找 *.txt）')
    ap.add_argument('--output', default=None, help='输出 parquet 路径；不指定时全量写到 OUT_PATH，试跑写到 *_sampleN.parquet')
    ap.add_argument('--limit', type=int, default=LIMIT, help='只跨月份随机抽这么多篇试跑')
    ap.add_argument('--seed', type=int, default=0, help='抽样的随机种子')
    ap.add_argument('--workers', type=int, default=WORKERS, help='并行的进程数，默认用满 CPU 核心')
    args = ap.parse_args()
    out = args.output or (OUT_PATH.with_name(f'{OUT_PATH.stem}_sample{args.limit}.parquet') if args.limit else OUT_PATH)
    report_text.main(args.input, out, args.limit, args.seed, args.workers)
