"""实验：PATHLEN 与研报前后 CAR，按本篇 2024 年净利润预测相对同一券商上一次预测（发布前 365 天内）的上调/下调定方向：signed CAR、PATHLEN×UP 交互、同样本对照。
不调用大模型，可反复运行。设计见 src/report_car_frev.py 开头。
需先运行 experiment_dag_event.py PATHLEN market industry（事件面板）；盈利预测表在 data/EST_COMP_TN/。

在仓库根目录运行：
    python -X utf8 experiment_pathlen_frev.py

结果：
    output_dag/experiments/report_car/frev/report_frev.parquet / .csv   逐篇方向（主版本 UP，稳健性 UP_25、UP_180）
    output_dag/experiments/report_car/pathlen_frev_ind/（行业调整）和 pathlen_frev/（市场调整）：
        pathlen_frev_sample.parquet           回归样本
        pathlen_frev_results.csv              长表：dir_version × model（signed / interact / interact_sum / plain）× spec × Y × winsor × term
        tables/t16_frev_signed_pooled.md      表 16：signed CAR，无固定效应
        tables/t17_frev_signed_firm.md        表 17：signed CAR，公司固定效应
        tables/t18_frev_interact_firm.md      表 18：PATHLEN×UP 交互，公司固定效应
        tables/t16k_*、t17k_*、t18k_*          同上，加研报字数和概念数
"""
import sys

from src.config import ROOT_DIR

# ==================== 运行参数 ====================
BASE = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'
EST_DIR = ROOT_DIR / 'data' / 'EST_COMP_TN'
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src.report_car_frev import main
    main(BASE, EST_DIR)
