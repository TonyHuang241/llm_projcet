"""实验：PATHLEN 与研报前后 CAR，按评级强度（券商评级体系最高档 vs 次高档）区分：signed CAR、PATHLEN×TOP 交互、同样本对照。
不调用大模型，可反复运行。设计见 src/report_car_rating.py 开头。
需先运行 extract_report_rating.py（评级）和 experiment_dag_event.py PATHLEN market industry（事件面板）。

在仓库根目录运行：
    python -X utf8 experiment_pathlen_rating.py

结果在 output_dag/experiments/report_car/pathlen_rating_ind/（行业调整）和 pathlen_rating/（市场调整）：
    pathlen_rating_sample.parquet       回归样本（含 TOP、DIR、signed CAR、标准化 PATHLEN）
    pathlen_rating_results.csv          长表：model（signed / interact / interact_sum / plain / plain_top）× spec × Y × winsor × term
    tables/t7_signed_pooled_ctrl.md     表 7：signed CAR，无固定效应
    tables/t8_signed_firm_ctrl.md       表 8：signed CAR，公司固定效应
    tables/t9_interact_firm_ctrl.md     表 9：PATHLEN×TOP 交互，公司固定效应
    tables/t10_signed_pooled.md         表 10：signed CAR，无固定效应，不加研报字数和概念数
    tables/t11_signed_firm.md           表 11：signed CAR，公司固定效应，不加研报字数和概念数
    tables/t12_interact_firm.md         表 12：PATHLEN×TOP 交互，公司固定效应，不加研报字数和概念数
"""
import sys

from src.config import ROOT_DIR

# ==================== 运行参数 ====================
BASE = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
RATING = BASE / 'rating' / 'report_rating.parquet'
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src.report_car_rating import main
    main(BASE, RATING)
