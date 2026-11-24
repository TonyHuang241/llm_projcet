"""概念 DAG 结构指标与研报盈利预测准确率的回归（不调用大模型，可以反复运行）。
791 篇抽出 DAG 的 2024 年研报（241 家沪深300 公司），被解释变量为这篇研报对本公司归母净利润预测的误差：
AFE1（主，T+1 年绝对误差 ÷ 过去五年一致预期绝对误差的标准差）、AFE2、AFE_FY24、AFE1_P（市值标度）、PMAFE1（相对同行）、AFE_FY25（对 FY2025 的预测）、PMAFE_FY24、PMAFE_FY25（对 FY2024/FY2025 的相对同行误差）。
三种固定效应结构（A 行业+目标年度、B 公司+目标年度、C 公司×目标年度+券商），按公司聚类。
面板、口径和回归在 src/dag_forecast.py，图和结构指标在 src/dag_measures.py，稳健性检验在 src/dag_forecast_robust.py，
表格在 src/dag_forecast_tables.py，估计函数在 src/dag_outcomes_reg.py。

在仓库根目录运行：
    python regress_dag_forecast.py              构造变量、跑全部回归，写出 output_dag/forecast/
    python regress_dag_forecast.py --main-only  敏感性：一致预期和同行只用研报主体公司的预测行，写到 output_dag/forecast/main_only/
    python regress_dag_forecast.py --x rating   解释变量换成“投资评级”汇点的源点数、路径数、平均路径长度，写到 output_dag/forecast/rating_sink/
                                                （可以和 --main-only 一起用，写到 rating_sink/main_only/）
    python regress_dag_forecast.py --x sink_np   目标节点换成“归母净利润”（sink_rev：“营业收入”；sink_fund：两个节点合起来），
                                                写到 output_dag/forecast/sink_np/ 等（同样可以加 --main-only）
    python regress_dag_forecast.py [--x ...] --peer-variants
                                                同行基准稳健性检验（PMAFE_FY24、PMAFE_FY25 的 8 种同行定义 × 2 种同行来源），
                                                只写该口径基础目录下的 peer_variants.csv、peer_variants_panel.parquet、tables/f11–f12
    python regress_dag_forecast.py --joint rating_rev
                                                “投资评级”节点和“营业收入”节点的指标放进同一回归（读 rating_sink/、sink_rev/ 的已有输出，
                                                先跑完那两个口径的普通命令和 --main-only），写到 output_dag/forecast/joint_rating_rev/
                                                （--joint val_cf：估值类 vs 现金流类汇点，读 sink_val/、sink_cf/，写到 joint_val_cf/；
                                                旧开关 --joint-rating-rev 等于 --joint rating_rev；--joint val_cfcore：估值类 vs 核心现金流类，
                                                读 sink_val_n/、sink_cf_core/，写到 joint_val_cfcore/）；--input-dir 改读别的根目录
    python regress_dag_forecast.py --x rating --with-nofe [--main-only]
                                                回归多一个不加固定效应的 Panel N（只有截距，控制变量同 Panel A，按公司聚类），顺序 N、A、B、C，
                                                写到 rating_sink_n/（[main_only/]）；--x sink_rev 同理写到 sink_rev_n/；
                                                --joint rating_rev --with-nofe 读 rating_sink_n/、sink_rev_n/，写到 joint_rating_rev_n/（--output 可改）
    python regress_dag_forecast.py --x rating_strict --with-nofe [--main-only]
                                                同 --x rating，但只有“投资评级”节点是终点（入度>0、出度=0）的研报才有指标（有出边的记缺失），
                                                在这些研报里缩尾；写到 rating_strict_n/。--x sink_rev_strict 同理（“营业收入”），写到 sink_rev_strict_n/；
                                                --x sink_np_strict 同理（“归母净利润”，与旧的非严格 sink_np 不同），写到 sink_np_strict_n/
    python regress_dag_forecast.py --x sink_cf_core_ratio [--with-nofe] [--main-only] [--output ...]
                                                核心现金流类汇点集合（同 sink_cf_core），但 R_RATIO 用源汇比（源点数 ÷ 汇点个数）
                                                （估值类 + 源汇比 + Panel N 用 --x sink_val --with-nofe --output <目录>，避免与已有 sink_val_n/ 重名）
    python regress_dag_forecast.py --x sink_final --with-nofe [--main-only] [--output ...]
                                                目标节点 = 名称在最终指标清单（output_dag/forecast/final_indicators/final_indicator_list.csv，
                                                status == '入选'）里的全部严格汇点（入度>0、出度=0）；R_RATIO 用源点数，另存 R_N_SINKS；写到 sink_final_n/
    python regress_dag_forecast.py --ratio rating_rev [--output ...]
                                                “投资评级”节点的源点数、ln(1+路径数)、平均路径长度分别除以“营业收入”节点的对应指标（另存对数口径），
                                                样本为 base 内两个节点都是终点的研报；对 PMAFE_FY24、PMAFE_FY25 的 8 种同行定义 × 2 种同行来源回归，
                                                Panel N、A、B、C；写到 output_dag/forecast/ratio_rating_rev/（见 src/dag_forecast_robust.py）
    python regress_dag_forecast.py --node-coverage [--input-dir ...] [--output ...]
                                                “投资评级”“营业收入”节点的覆盖篇数（读 rating_sink/、sink_rev/ 的 forecast_panel_raw.parquet），
                                                写到 output_dag/forecast/node_coverage.csv
    python regress_dag_forecast.py --classify-sinks [core] [--output ...]
                                                样本研报里每个汇点名称的类别（VAL / CF / OTHER）和篇数，写到 output_dag/forecast/sink_classification.csv
                                                （core：VAL / CFCORE / OTHER，写到 sink_classification_core.csv）

结果在 <输出目录>/：
    forecast_panel_raw.parquet、forecast_panel.parquet   每篇研报一行（未缩尾 / 缩尾后）
    consensus_errors.parquet   各公司各财年的一致预期、券商家数、绝对误差（算 STD_PAST 用）
    peer_detail.parquet        PMAFE1 的同行预测明细；peer_detail_fy24/fy25.parquet 为 PMAFE_FY24、PMAFE_FY25 的
                               同行预测明细
    funnel.csv、descriptives.csv、reg_results.csv
    tables/f01–f10             样本筛选、描述统计、八个被解释变量的回归表
    peer_variants.csv、peer_variants_panel.parquet、tables/f11–f12   --peer-variants 的结果（见 src/dag_forecast_robust.py）
"""
import argparse
import sys
import time
from pathlib import Path

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
OUT_DIR = ROOT_DIR / 'output_dag' / 'forecast'
PANEL = ROOT_DIR / 'output_dag' / 'regression' / 'report_panel.parquet'  # 研报层面面板（DAG 指标已缩尾）
EST_DIR = DATA_DIR / 'EST_COMP_TN'  # 分析师预测，按研报日期分月
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
DAG_DIRS = [ROOT_DIR / 'output_dag' / 'deep' / 'data', ROOT_DIR / 'output_dag' / 'data']  # 汇点口径用：report_id 重复时用前面的
# ==================================================
PATHS = dict(panel_path=PANEL, est_dir=EST_DIR, fin_path=FINANCIAL, daily_path=DAILY)

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src import dag_forecast as F
    ap = argparse.ArgumentParser(description='概念 DAG 结构指标与研报盈利预测准确率的回归')
    ap.add_argument('--main-only', action='store_true', help='一致预期和同行只用研报主体公司的预测行')
    ap.add_argument('--x', choices=F.X_MODES, default='dag',
                    help='解释变量：dag = 整张图的三个指标（默认）；rating = “投资评级”汇点（写到 rating_sink/）；'
                         'sink_np / sink_rev / sink_fund = “归母净利润” / “营业收入” / 两者合起来；'
                         'sink_val / sink_cf = 估值类 / 现金流类汇点集合（源汇比）；sink_val_n / sink_cf_core = 估值类 / 核心现金流类汇点集合'
                         '（源点数）；sink_cf_core_ratio = 核心现金流类汇点集合（源汇比）；sink_final = 名称在最终指标清单（入选）里的全部汇点（源点数）；rating_strict / sink_rev_strict / sink_np_strict = “投资评级” / “营业收入” / “归母净利润”节点且只在它是终点（入度>0、出度=0）时有指标'
                         '（都写到同名子目录）')
    ap.add_argument('--peer-variants', action='store_true', help='只做同行基准稳健性检验（两种同行来源都做，不能和 --main-only 同用）')
    ap.add_argument('--joint', choices=['rating_rev', 'val_cf', 'val_cfcore'], default=None, help='两种汇点口径指标的联合回归（见 src/dag_forecast_robust.py）')
    ap.add_argument('--joint-rating-rev', action='store_true', help='等于 --joint rating_rev')
    ap.add_argument('--classify-sinks', nargs='?', const='full', choices=['full', 'core'], default=None,
                    help='写出汇点名称的归类明细：full -> sink_classification.csv（VAL/CF/OTHER）；core -> sink_classification_core.csv（VAL/CFCORE/OTHER）')
    ap.add_argument('--with-nofe', action='store_true',
                    help='回归多一个不加固定效应的 Panel N（控制变量同 Panel A），顺序 N、A、B、C；默认输出目录为原目录名加 _n'
                         '（如 rating_sink_n/、sink_rev_n/、joint_rating_rev_n/；--joint 时读 <口径目录>_n/ 的输出），可用 --output 改')
    ap.add_argument('--node-coverage', action='store_true',
                    help='“投资评级”“营业收入”节点的覆盖统计（读 rating_sink/、sink_rev/ 的 forecast_panel_raw.parquet），写到 node_coverage.csv')
    ap.add_argument('--ratio', choices=['rating_rev'], default=None,
                    help='“投资评级”节点指标 ÷ “营业收入”节点指标的比值回归（见 src/dag_forecast_robust.py），写到 ratio_rating_rev/')
    ap.add_argument('--output', default=None, help='输出目录（--node-coverage、--classify-sinks 时为输出文件）')
    ap.add_argument('--input-dir', default=None, help='--joint、--node-coverage 读已有口径输出的根目录（默认 output_dag/forecast/）')
    args = ap.parse_args()
    in_dir = Path(args.input_dir) if args.input_dir else OUT_DIR
    t0 = time.time()
    if args.ratio:
        from src import dag_forecast_robust
        dag_forecast_robust.ratio(args.output or OUT_DIR / ('ratio_' + args.ratio), DAG_DIRS, **PATHS)
    elif args.node_coverage:
        cov = F.node_coverage(in_dir)
        cov.to_csv(args.output or OUT_DIR / 'node_coverage.csv', index=False, encoding='utf-8-sig')
        print(cov.to_string(index=False), flush=True)
        sys.exit(0)
    elif args.joint or args.joint_rating_rev:
        from src import dag_forecast_robust
        dag_forecast_robust.joint(in_dir, 'joint_' + (args.joint or 'rating_rev'), with_nofe=args.with_nofe, out_dir=args.output)
    elif args.classify_sinks:
        import pandas as pd
        core = args.classify_sinks == 'core'
        cls = F.sink_classification(list(pd.read_parquet(PANEL).report_id), DAG_DIRS, core)
        cls.to_csv(args.output or OUT_DIR / ('sink_classification_core.csv' if core else 'sink_classification.csv'), index=False, encoding='utf-8-sig')
        print(cls.groupby('class').agg(names=('name', 'size'), occurrences=('n_reports', 'sum')).to_string())
    else:
        if args.peer_variants and args.main_only:
            ap.error('--peer-variants 同时做全部行和主体行两种同行来源，不要加 --main-only')
        if args.peer_variants and args.with_nofe:
            ap.error('--with-nofe 不用于 --peer-variants')
        out = args.output or F.default_dir(OUT_DIR, args.x, args.main_only, args.with_nofe)
        F.run(out, args.x, main_only=args.main_only, with_nofe=args.with_nofe, peer_variants=args.peer_variants, dag_dirs=DAG_DIRS, **PATHS)
    print(f'完成，耗时 {time.time() - t0:.0f} 秒', flush=True)
