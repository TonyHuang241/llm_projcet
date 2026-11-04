"""项目启动入口：修改下面的 TASK 选择要运行的功能，然后执行 python main.py"""
from pathlib import Path

from src import extract_audit, extract_mda, neutralize_llm

DATA_DIR = Path(__file__).resolve().parent / 'data'
OUTPUT_DIR = Path(__file__).resolve().parent / 'output'  # 分析结果

# ==================== 运行参数 ====================
TASK = 'compare_neutral_tone'  # 要运行的功能，可选值见下方 TASKS
YEAR = '2024'  # 情感分析使用哪一年的文本（提取的文本按年份存放在子文件夹中）


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
    # 要分析的改写版本 {版本名: data下的文件夹}；Haiku版只有前50篇，需要时加回 'haiku': 'mda_neutral'
    versions = {'glm': 'mda_neutral_glm'}
    for name, folder in versions.items():
        print(f'\n########## 改写版本：{name} ##########')
        compare_neutral_tone.main(
            neutral_dir=DATA_DIR / folder / YEAR,
            out_dir=OUTPUT_DIR / 'sentiment_analysis' / f'neutral_compare_{name}',
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
        limit=2500,  # 处理前多少篇（按文件名排序），None 表示全部；已完成的会自动跳过
    ),
}
# ==================================================

if __name__ == '__main__':
    if TASK not in TASKS:
        raise SystemExit(f'未知的 TASK: {TASK}，可选: {list(TASKS)}')
    TASKS[TASK]()
