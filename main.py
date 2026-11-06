"""项目启动入口：修改下面的 TASK 选择要运行的功能，然后执行 python main.py"""
from pathlib import Path

from src import extract_audit, extract_mda, neutralize_llm, plot_tone_change

DATA_DIR = Path(__file__).resolve().parent / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'  # 分析结果

# ==================== 运行参数 ====================
TASK = 'neutralize_llm'  # 要运行的功能，可选值见下方 TASKS
YEAR = '2023'  # 情感分析使用哪一年的文本（提取的文本按年份存放在子文件夹中）
# 中性化时不按分析师覆盖筛选、全部保留的年报年份（分析师数据只到 2023-08，2024 年报要看的 2023 年数据不完整）
KEEP_ALL_YEARS = {'2024'}


def run_finbert_sentiment():
    from src import finbert_sentiment  # 依赖 torch，只在用到时导入
    finbert_sentiment.main(
        audit_dir=DATA_DIR / 'audit_texts' / YEAR,
        mda_dir=DATA_DIR / 'mda_texts' / YEAR,
        out_path=OUTPUT_DIR / 'sentiment_analysis' / 'finbert_sentiment.csv',
        limit=100,  # 分析前多少家公司（按文件名排序），None 表示全部
    )


def run_compare_neutral_tone():
    from src import compare_neutral_tone  # 依赖 torch，只在用到时导入
    # 一次分析 mda_neutral_glm 下所有年份，结果合并到同一份文件；随时可以停止，重新运行自动跳过已完成的文章
    compare_neutral_tone.main(
        neutral_root=DATA_DIR / 'mda_neutral_glm',
        out_dir=OUTPUT_DIR / 'sentiment_analysis' / 'neutral_compare_glm_all',
        # 之前已分析过的结果（没有 year 列），导入后不再重新分析，原文件不会被修改
        legacy_dir=OUTPUT_DIR / 'sentiment_analysis' / 'neutral_compare_glm',
        legacy_year='2024',
    )


# 所有可运行的功能，新增功能时在这里加一项
TASKS = {
    # 从年报中提取“管理层讨论与分析”章节的文字叙述
    'extract_mda': lambda: extract_mda.main(
        zip_path=DATA_DIR / 'annual_reports' / '2024_5405份.zip',
        out_dir=DATA_DIR / 'mda_texts',
    ),
    # 从年报中提取财务报表审计报告的文字叙述
    'extract_audit': lambda: extract_audit.main(
        zip_path=DATA_DIR / 'annual_reports' / '2024_5405份.zip',
        out_dir=DATA_DIR / 'audit_texts',
    ),
    # 用中文FinBERT对审计报告和MD&A做情感分析（GPU加速）
    'finbert_sentiment': run_finbert_sentiment,
    # 比较MD&A中性化改写前后的FinBERT语调
    'compare_neutral_tone': run_compare_neutral_tone,
    # 调用大模型（.env 中配置）对MD&A逐句中性化改写，每篇文章一个并行任务
    'neutralize_llm': lambda: neutralize_llm.main(
        mda_dir=DATA_DIR / 'mda_texts' / YEAR,
        out_dir=DATA_DIR / 'mda_neutral_glm' / YEAR,
        year=YEAR,
        limit=None,  # 处理前多少篇（按文件名排序，在覆盖过滤之后），None 表示全部；已完成的会自动跳过
        # 只处理有分析师覆盖的股票，None 表示不过滤（KEEP_ALL_YEARS 中的年份全部保留）
        coverage_path=None if YEAR in KEEP_ALL_YEARS else DATA_DIR / 'analyst_forecast' / 'stock_forecast_means.parquet',
        # YEAR-1 年内有一年后(fy1)预测的公司才保留；None 表示任意年份出现过即可
        coverage_year=int(YEAR) - 1,
        # 剔除 YEAR-1 年内被 ST 过的股票，st_path 为 None 表示不过滤
        st_path=DATA_DIR / 'market_data' / 'stock_st_status.parquet',
        st_year=int(YEAR) - 1,
        exclude_bse=True,  # 剔除北交所股票
    ),
    # 把中性化前后的语调变化画成交互式HTML报告（文章间、文章内两个分布）
    'plot_tone_change': lambda: plot_tone_change.main(
        compare_dir=OUTPUT_DIR / 'sentiment_analysis' / 'neutral_compare_glm',
    ),
}
# ==================================================

if __name__ == '__main__':
    if TASK not in TASKS:
        raise SystemExit(f'未知的 TASK: {TASK}，可选: {list(TASKS)}')
    TASKS[TASK]()
