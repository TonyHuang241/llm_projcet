"""第二步：研报逻辑链抽取。读 clear_txt.py 清理好的正文段落，用大模型把每篇研报抽成一张五层估值逻辑图，
结果写到 output/：最外层的 研报逻辑图.html 是入口（双击打开，选中一篇研报时才读取它的数据并绘图），
结果数据在 output/data/，大模型缓存和进度日志在 output/cache/（已完成的调用重跑时自动跳过）。
抽取流程见 src/report_logic.py，框架设计见 docs/研报逻辑图抽取框架设计.md。

在仓库根目录运行：
    python extract_logic_chain.py                按下面的运行参数抽取逻辑链
    python extract_logic_chain.py --limit 5      只处理前 5 篇试跑（按 report_id 排序）
    python extract_logic_chain.py --input data/xxx.parquet --output output_sample   换输入、输出（试跑时免得覆盖正式结果）
    python extract_logic_chain.py --task compare     抽取之后：同一公司不同研报的逻辑框架对比
    python extract_logic_chain.py --task intro_page  抽取和对比之后：生成介绍页 docs/逻辑抽取与分歧度量介绍.html
"""
import argparse
import sys

from src.config import DATA_DIR, OUTPUT_DIR, ROOT_DIR

# ==================== 运行参数 ====================
PARAGRAPHS = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # 输入：clear_txt.py 写出的正文段落（至少含 report_id、para_id、text 三列）
OUT_DIR = OUTPUT_DIR  # 输出目录
LIMIT = None  # 试跑时只处理前多少篇研报（按 report_id 排序），None 表示全部
TASK = 'extract'  # 默认抽取逻辑链；另外两个功能见下方 TASKS

# 框架对比的参数（各列定义见 src/report_compare.py 开头）
EDGE_SET = 'all'  # 'all' 全部边；'text' 去掉逻辑补全边（稳健性检验）
PEERS = 'all'  # 逐篇指标的同行：'all' 同公司其他全部研报；'prior' 只用更早发布的（避免前视偏差）
WINDOW_DAYS = None  # PEERS='prior' 时往前看多少天，None 为不限
# ==================================================


def extract(args):
    """逻辑链抽取：概念分层填表 -> 原文连边 -> 整理概念表 -> 悬挂补全、并列复核 -> 包含关系缩减 -> 子图补全
    -> 剪枝 -> 全局去重 -> 写出每篇的逻辑图数据、parquet 和入口页面。"""
    from src import report_logic
    report_logic.main(paras_path=args.input, out_dir=args.output, limit=args.limit)


def compare(args):
    """同一公司不同研报的逻辑框架对比（每家公司调用一次大模型把概念归成主题），结果写到 <输出目录>/data/compare/。"""
    from src import report_compare
    report_compare.main(out_dir=args.output, edge_set=EDGE_SET, peers=PEERS, window_days=WINDOW_DAYS)


def intro_page(args):
    """介绍页（不调用大模型）：抽取流程、用研报样例说明抽取可信度和分歧度量，页面模板在 src/viz/intro_template.html。"""
    from src.viz import plot_intro
    plot_intro.main(out_dir=args.output, paras_path=args.input, out_html=ROOT_DIR / 'docs' / '逻辑抽取与分歧度量介绍.html')


TASKS = {'extract': extract, 'compare': compare, 'intro_page': intro_page}

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='研报逻辑链抽取')
    ap.add_argument('--task', default=TASK, choices=list(TASKS), help='要运行的功能，默认抽取逻辑链')
    ap.add_argument('--input', default=PARAGRAPHS, help='正文段落 parquet')
    ap.add_argument('--output', default=OUT_DIR, help='输出目录')
    ap.add_argument('--limit', type=int, default=LIMIT, help='只处理前多少篇研报试跑')
    args = ap.parse_args()
    TASKS[args.task](args)
