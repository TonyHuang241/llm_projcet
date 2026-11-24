"""实验：券商研报的概念 DAG 结构指标是否影响研报发布后的市场反应（事件研究，不调用大模型，可反复运行）。
791 篇 2024 年沪深300 研报，每篇一个观测：发布后的 CAR、|CAR|、异常换手、发布前的安慰剂 CAR 对三个 DAG 指标回归，
口径 A（申万一级+券商+月份固定效应）、B（公司+券商+月份固定效应）和 B 的稳健性。口径细节见 src/report_car.py 开头。
不改动已有流程（output_dag/regression/ 只读）。

在仓库根目录运行：
    python -X utf8 experiment_report_car.py

结果在 output_dag/experiments/report_car/：
    report_car_panel.parquet       每篇研报的事件日、各 Y、各控制变量（已缩尾）；report_car_panel_raw.parquet 未缩尾
    report_car_results.csv         长表，一行一个系数
    report_car_descriptives.csv    各 Y 和控制变量的描述统计（缩尾后）
"""
import sys

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
OUT_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
REPORT_PANEL = ROOT_DIR / 'output_dag' / 'regression' / 'report_panel.parquet'  # 791 篇研报的面板（DAG 指标已缩尾）
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'
REPORTS_META = DATA_DIR / 'hs300_2024_reports.parquet'  # 算 NREP（同股其他个股研报）
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'  # 申万一级（行业调整收益）
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src.report_car import main
    main(OUT_DIR, REPORT_PANEL, DAILY, FINANCIAL, REPORTS_META, INDUSTRY)
