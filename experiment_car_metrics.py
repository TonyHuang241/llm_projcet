"""实验：790 篇深度研报逐篇整理发布前后 6 个窗口的 CAR、换手率、异常换手率和概念 DAG 结构指标（含“到投资评级”的四个指标），
并生成一个可任选两个指标画散点图的页面。不调用大模型，可反复运行；只读已有输出和日行情。
口径见 src/report_car_metrics.py 开头；目标节点只认名称恰好为“投资评级”的概念（src/valuation_sinks.TARGET）。
需先运行 experiment_report_car.py（report_car_panel_raw.parquet，闸门用）和 experiment_dag_event.py（事件面板 pathlen_event_panel_raw.parquet）。

在仓库根目录运行：
    python -X utf8 experiment_car_metrics.py

结果在 output_dag/experiments/report_car/car_metrics/：
    report_car_metrics.parquet / .csv   一行一篇：代码、公司、券商、日期、6 个 CAR、6 个 TURN、6 个 ATURN、PRETURN、
                                        整图和到投资评级的指标（均未缩尾）
    （上一版“估值相关汇点”口径的 valuation_sinks.csv、sinks_all.csv 不再生成，运行时如存在会删掉）
页面：output_dag/experiments/report_car/研报CAR与DAG指标.html
"""
import sys
import time

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
EXP_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
OUT_DIR = EXP_DIR / 'car_metrics'
PAGE = EXP_DIR / '研报CAR与DAG指标.html'
EVENT_PANEL = EXP_DIR / 'pathlen_event' / 'pathlen_event_panel_raw.parquet'  # 790 篇，CAR 未缩尾
REPORTS_META = DATA_DIR / 'hs300_2024_reports.parquet'  # 券商名
DAG_DIRS = [ROOT_DIR / 'output_dag' / 'deep' / 'data', ROOT_DIR / 'output_dag' / 'data']  # report_id 重复时用前面的
REPORT_PANEL = ROOT_DIR / 'output_dag' / 'regression' / 'report_panel.parquet'  # 只读：核对整图三个指标（那边已缩尾）
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'  # 日行情：算换手率（同 experiment_report_car.py）
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'  # market_matrices 需要的申万一级映射（换手率用不到行业收益）
CAR_RAW_PANEL = EXP_DIR / 'report_car_panel_raw.parquet'  # 只读：闸门，PRETURN、ATURN_0_2 须逐篇相等（未缩尾）
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src.report_car_metrics import main
    from src.viz.plot_car_metrics import main as plot
    t0 = time.time()
    df, check = main(OUT_DIR, EVENT_PANEL, REPORTS_META, DAG_DIRS, DAILY, INDUSTRY, CAR_RAW_PANEL, REPORT_PANEL)
    plot(df, check, PAGE)
    print(f'完成，用时 {time.time() - t0:.1f} 秒')
