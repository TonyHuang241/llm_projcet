"""目录约定和跨步骤共用的数据文件路径：src/ 只放代码和提示词；数据、输出、文档、文献和 .env 都在仓库根目录。

各步骤自己的默认运行参数写在该步骤的目录里（src/<步骤>/），不放这里；下游步骤读上游的成品时通过上游目录里的 load.py，
不自己拼路径。
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'  # 原始数据与中间数据（不入 git）
OUTPUT_DIR = ROOT_DIR / 'output'  # 各步骤的结果（不入 git），output/<步骤>/

# 跨步骤共用的数据文件
TXT_DIR = DATA_DIR / 'HS300_reports_2024'  # 研报纯文本（MM/DD/<GUID>.txt）
PARAGRAPHS = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # text 步骤写出的正文段落表（旁边有同名 .txt 便于人工检查）
REPORTS = DATA_DIR / 'hs300_2024_reports.parquet'  # text 步骤写出的研报元数据表（旁边有同名 .csv）
EST_DIR = DATA_DIR / 'EST_COMP_TN'  # 分析师盈利预测，按发布日期分月：<年>/<年>-<月>.parquet
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'  # 财务数据（年报实际值、公告日、总资产、负债）
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'  # 个股日行情（收盘价、总市值、停牌）


def out_dir(step, run=None):
    """步骤的输出目录 output/<步骤>/，给了运行名时是 output/<步骤>/<运行名>/（如 out_dir('graph', run='deep')）。"""
    d = OUTPUT_DIR / step
    return d / run if run else d
