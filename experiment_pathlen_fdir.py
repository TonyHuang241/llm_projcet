"""实验：PATHLEN 与研报前后 CAR，按本篇 2024 年净利润预测相对发布前 90 天其他券商预测中位数的高低定方向：signed CAR、PATHLEN×HIGH 交互、同样本对照。
不调用大模型，可反复运行。设计见 src/report_car_fdir.py 开头。
需先运行 experiment_dag_event.py PATHLEN market industry（事件面板）；盈利预测表在 data/EST_COMP_TN/。

在仓库根目录运行：
    python -X utf8 experiment_pathlen_fdir.py

结果：
    output_dag/experiments/report_car/fdir/report_fdir.parquet / .csv   逐篇方向（主版本 HIGH，稳健性 HIGH_25、HIGH_180）
    output_dag/experiments/report_car/pathlen_fdir_ind/（行业调整）和 pathlen_fdir/（市场调整）：
        pathlen_fdir_sample.parquet           回归样本
        pathlen_fdir_results.csv              长表：dir_version × model（signed / interact / interact_sum / plain）× spec × Y × winsor × term
        tables/t13_fdir_signed_pooled.md      表 13：signed CAR，无固定效应
        tables/t14_fdir_signed_firm.md        表 14：signed CAR，公司固定效应
        tables/t15_fdir_interact_firm.md      表 15：PATHLEN×HIGH 交互，公司固定效应
        tables/t13k_*、t14k_*、t15k_*          同上，加研报字数和概念数
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
    from src.report_car_fdir import main
    main(BASE, EST_DIR)
