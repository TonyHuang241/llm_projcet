"""实验：概念 DAG 结构指标（PATHLEN、LNPATH、RATIO）与研报发布前后的分日超额收益（事件研究的分日展开，不调用大模型，可反复运行）。
在上一轮 experiment_report_car.py 的 790 篇研报上，取 21 天 AR 都能算出的平衡样本，窗口 CAR、逐日 AR、累计 CUM 分别对标准化的
单个 DAG 指标回归（无固定效应、行业、公司及其同样本对照和加控制变量版本）。口径细节见 src/report_car_event.py 开头。
需先运行过 experiment_report_car.py（读它的面板和结果）；不改动上一轮输出和 output_dag/regression/。

在仓库根目录运行（默认三个指标都跑、基准是市场收益；也可以只给一个或几个指标，基准写 market 或 industry）：
    python -X utf8 experiment_dag_event.py
    python -X utf8 experiment_dag_event.py LNPATH
    python -X utf8 experiment_dag_event.py PATHLEN industry     # 行业调整（申万一级、剔除本股）

结果在 output_dag/experiments/report_car/{pathlen,lnpath,ratio}_event/（行业调整版目录名加 _ind，如 pathlen_event_ind/）：
    *_event_panel.parquet       平衡样本，21 个 AR、窗口 CAR、CUM、三个 DAG 指标、固定效应和控制变量（已缩尾）；_raw 未缩尾；三个目录内容相同
    *_event_results.csv         长表，一行一个回归（winsor=1/0 区分缩尾与否）
    tables/t1_window.md         表 1：窗口 CAR
    tables/t2_daily.md          表 2：分日 AR
    tables/t3_window_ctrl.md    表 3：窗口 CAR，控制研报字数和概念数（none_k、ind_k、firm_k）
    tables/t4_daily_ctrl.md     表 4：分日 AR，控制研报字数和概念数
    tables/t5_windows_pooled_ctrl.md  表 5：六个窗口（CAR[−5,−1]、[0,1]、[0,2]、[0,3]、[0,5]、[0,10]），无固定效应，
                                      样本限定在有 2 篇及以上研报的公司，控制研报字数和概念数
    tables/t6_windows_firm_ctrl.md    表 6：六个窗口，公司固定效应 + 研报字数和概念数
"""
import sys

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
PREV_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car'  # 上一轮输出（790 篇面板）；各指标的结果放在它下面的 *_event/
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src.report_car_event import BENCHES, DAG_X, main
    args = sys.argv[1:]
    xs = [a.upper() for a in args if a.lower() not in BENCHES] or DAG_X
    benches = [a.lower() for a in args if a.lower() in BENCHES] or ['market']
    bad = [x for x in xs if x not in DAG_X]
    if bad:
        sys.exit(f'未知的参数：{bad}，解释变量可选 {DAG_X}，基准可选 {BENCHES}')
    main(PREV_DIR, PREV_DIR, DAILY, INDUSTRY, xs, benches)
