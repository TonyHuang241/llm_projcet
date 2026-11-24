"""跨研报概念名称统一：在 extract_concept_dag.py 抽完的概念 DAG 上，调用大模型把同一家公司不同研报里指完全相同的东西、只是说法不同的
概念名称（营收 / 营业收入，PE / 市盈率）统一成一个；范围、口径、主体、指标不同的都不合并，合并组再逐组复核。
流程见 src/concept_unify.py，提示词见 src/prompts/concept_unify_prompt.md。

读 output_dag/data/ 下概念 DAG 的结果表，结果也写在那里（concept_unify.parquet、unify_groups.parquet、unified_edges.parquet、
unify.js），核对页面是 output_dag/概念统一.html；概念 DAG 新增了研报后重新运行即可，没变的候选组直接用缓存。

在仓库根目录运行：
    python unify_concept_dag.py --dry-run          只统计候选组，不调用大模型
    python unify_concept_dag.py                    按下面的运行参数判定并写出结果（默认只统一 COMPANIES 里的公司，并发 50）
    python unify_concept_dag.py --companies all    概念 DAG 里的研报全部参与统一（各公司分别统一，只有 1 篇的公司不统一）
    python unify_concept_dag.py --max-group 30     每个候选组最多 30 个名称
    python unify_concept_dag.py --task structure   不调用大模型：按统一后的名称算各篇的源汇比、唯一有向路径数、平均路径长度，
                                                   写出 output_dag/概念DAG结构.html（src/viz/plot_dag_structure.py）
    python unify_concept_dag.py --output output_dag/deep --task deep
                                                   不调用大模型：全样本深度报告的三个指标分布、明细表和每篇的结构图，
                                                   写出 output_dag/deep/深度报告结构.html（src/viz/plot_deep_structure.py；
                                                   默认输出目录里的研报全部画，--companies 可只画几家）
"""
import argparse
import sys

from src.config import ROOT_DIR

# ==================== 运行参数 ====================
OUT_DIR = ROOT_DIR / 'output_dag'  # 概念 DAG 的输出目录
COMPANIES = ['002594.SZ']  # 只统一这些股票的研报：比亚迪的 16 篇深度报告；空列表表示概念 DAG 里的全部研报
WORKERS = 50
MAX_GROUP = 40  # 每个候选组最多这么多个名称：太大时模型一次要比较的名称多、容易漏，太小时相近的名称可能分到不同组
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='跨研报概念名称统一')
    ap.add_argument('--output', default=OUT_DIR, help='概念 DAG 的输出目录')
    ap.add_argument('--companies', default=None, help='只统一这些股票代码的研报，逗号分隔；all 表示全部；不给时用 COMPANIES')
    ap.add_argument('--workers', type=int, default=WORKERS, help='同时在途的大模型调用数')
    ap.add_argument('--max-group', type=int, default=MAX_GROUP, help='每个候选组最多多少个名称')
    ap.add_argument('--dry-run', action='store_true', help='只统计候选组，不调用大模型')
    ap.add_argument('--task', choices=['unify', 'structure', 'deep'], default='unify',
                    help='unify：统一概念名称（默认）；structure：按统一结果写出 DAG 结构统计页面（每篇一行）；'
                         'deep：深度报告结构页面（指标分布、明细表、每篇的结构图）')
    args = ap.parse_args()
    if args.task == 'deep':  # 默认画输出目录里的全部研报
        from src.viz import plot_deep_structure
        codes = None if args.companies in (None, 'all') else [x.strip().upper() for x in args.companies.split(',') if x.strip()]
        plot_deep_structure.main(args.output, companies=codes)
        sys.exit()
    if args.companies is not None:
        COMPANIES[:] = [] if args.companies.lower() == 'all' else [x.strip().upper() for x in args.companies.split(',') if x.strip()]
    if args.task == 'structure':
        from src.viz import plot_dag_structure
        plot_dag_structure.main(args.output, companies=COMPANIES)
        sys.exit()
    from src import concept_unify
    concept_unify.main(args.output, companies=COMPANIES, workers=args.workers, max_group=args.max_group,
                       dry_run=args.dry_run)
