"""概念 DAG 结构指标与 2025 年股价、财务指标的回归（不调用大模型，可以反复运行）。
方案见 docs/DAG结构指标与2025年股价财务回归方案.md：241 家 2024 年沪深300 公司、791 篇抽出 DAG 的研报，
三个结构指标（源汇比、ln(1+唯一有向路径数)、平均路径长度）对 2025 年的股价指标（同步性、崩盘风险、波动、回报、年报窗口 CAR、
换手率、非流动性）和财务指标的回归：M1 公司截面（主）、M2 研报层面、M3 季度财务面板，以及稳健性。
变量构造在 src/dag_outcomes_vars.py，估计（手写 OLS + 固定效应 + CR1 / HC1 / 双向聚类）在 src/dag_outcomes_reg.py，
置换检验、秩相关、三等分、逐一剔除在 src/dag_outcomes_robust.py，表格片段在 src/dag_outcomes_tables.py，流程和输出在 src/dag_outcomes.py。

在仓库根目录运行：
    python regress_dag_outcomes.py               构造变量、跑全部回归，写出 output_dag/regression/ 下的结果（约几分钟）
    python regress_dag_outcomes.py --selfcheck   只跑自检（虚拟变量法 vs 组内去均值、CR1 vs HC1、双向聚类退化、手算小数据、
                                                 t 分布 p 值；置换检验的 FWL t 与完整回归一致、原假设下拒绝率）

结果在 <输出目录>/：
    firm_panel.parquet、report_panel.parquet、quarter_panel.parquet   回归用的面板（已缩尾）；*_raw.parquet 是未缩尾的
    market_returns.parquet    市值加权市场组合的周收益、日收益
    descriptives.csv          各变量的观测数、均值、标准差、分位数
    correlations.csv          三个 DAG 指标和各 Y 的 Pearson、Spearman 相关系数（公司层面，缩尾后）
    reg_results.csv           长表，一行一个系数（M1/M2/M3 各口径、逐步加控制 step、置换检验 p_perm）
    reg_coefs_full.csv        main 口径的全部系数（含控制变量）
    rank_ic.csv、group_sort.csv、influence.csv   M1 主样本的秩相关、按 X 三等分、逐一剔除
    tables/                   论文式表格片段 t01–t19
    回归结果.md               主要结果的表格
"""
import argparse
import sys

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
OUT_DIR = ROOT_DIR / 'output_dag' / 'regression'  # 输出目录
SAMPLE = DATA_DIR / 'hs300_2024_dag_covered.csv'  # 241 家样本公司（stock_code 带后缀）
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'  # 行业：证监会门类（主）、申万一级（稳健性）
DAG_DIRS = [ROOT_DIR / 'output_dag' / 'deep' / 'data', ROOT_DIR / 'output_dag' / 'data']  # DAG 输出目录，report_id 重复时用前面的
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'  # 日行情（后复权价、总市值、复权因子）
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'  # 季度财务数据
REPORTS_META = DATA_DIR / 'hs300_2024_reports.parquet'  # 研报元数据（算 LNCOV：2024 年个股研报篇数）
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='概念 DAG 结构指标与 2025 年股价、财务指标的回归')
    ap.add_argument('--selfcheck', action='store_true', help='只跑回归函数的自检')
    ap.add_argument('--output', default=OUT_DIR, help='输出目录')
    args = ap.parse_args()
    if args.selfcheck:
        from src.dag_outcomes_reg import selfcheck
        from src.dag_outcomes_robust import selfcheck_perm
        ok, lines = selfcheck()
        print('回归函数自检：')
        print('\n'.join(lines))
        ok2, lines = selfcheck_perm()
        print('置换检验自检：')
        print('\n'.join(lines))
        ok &= ok2
        print('全部通过' if ok else '有未通过的项')
        sys.exit(0 if ok else 1)
    from src import dag_outcomes
    dag_outcomes.main(out_dir=args.output, sample_path=SAMPLE, industry_path=INDUSTRY, dag_dirs=DAG_DIRS, daily_path=DAILY,
                      fin_path=FINANCIAL, reports_meta_path=REPORTS_META)
