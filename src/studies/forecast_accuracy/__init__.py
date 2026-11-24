"""专题研究 forecast_accuracy：研报逻辑结构与分析师预测准确度（docs/研报逻辑结构与分析师预测准确度.md 的全部表、图和数字）。

输入只有上游步骤的成品（先跑 python run.py measure、python run.py accuracy）：src.graph.load、src.measure.load、src.accuracy.load，
以及 data/hs300_2024_dag_covered.csv、data/market_data/industry.xlsx。样本、缩尾、回归设定的口径见 sample.py、regressions.py
的模块说明；回归工具在 src/ols.py。

在仓库根目录运行：
    python run.py study forecast_accuracy                   样本、回归、表和图全部重做（一分钟以内）
    python run.py study forecast_accuracy --part tables     只做样本、回归和表（不画图）
    python run.py study forecast_accuracy --part figures    只做样本和两张图

输出在 output/studies/forecast_accuracy/（图在 docs/figures/）：
    sample_report.parquet        研报层面回归样本（786 篇基础样本；<变量>_RAW 为缩尾前的值）             sample.report_sample
    sample_pair.parquet          公司-券商层面回归样本（760 对）                                         sample.pair_sample
    reg_all.csv                  全部回归系数的长表（列见 regressions.py）                               regressions.run_all
    variance_decomposition.csv   表 1 的方差分解（六个变量 × 六组固定效应的 N、K、R²、调整 R²）           tables.variance_decomposition
    figure2_peer_lags.parquet    图 2 (a) 的同行明细（709 篇 × 同行，15266 行）                          figures.peer_lag_frames
    figure2_report_lags.parquet  图 2 (b) 的每篇研报同行时滞                                             figures.peer_lag_frames
    tables/table1_…–table8_….csv 文档表 1–8                                                             tables.all_tables
    tables/text_numbers.csv      正文、图注、第 4 节的数字（位置、数值、小数位、显示、星号、算法）        tables.text_numbers

文档的表 / 图 / 节 -> 输出文件 -> 生成它的函数
    1.1、1.2 节样本数字          tables/text_numbers.csv（位置 1.1、1.2）    sample.scope、tables.text_numbers
    图 1（逻辑结构变量的构建）   docs/figures/logic_structure_schematic.png   figures.schematic（例子的数字：位置“1.2 图 1 的例子”）
    1.3、1.4 节描述统计          tables/text_numbers.csv（位置 1.3、1.4）    tables.text_numbers
    表 1（固定效应分解）         tables/table1_fe_decomposition.csv          tables.variance_decomposition、tables.table1
    2.1 节                       tables/text_numbers.csv（位置 2.1、表 1 注） tables.text_numbers
    图 2（同行时滞分布）         docs/figures/peer_lag_distribution.png       figures.peer_lag_frames、figures.peer_lag_plot
    2.2 节、图 2 注              tables/text_numbers.csv（位置 2.2、图 2 注） tables.text_numbers
    表 2（基准结果）             tables/table2_benchmark.csv                 reg_all：report / fe_steps / PMAFE_FY25 / firm+broker
    表 3（不同固定效应）         tables/table3_fe_steps.csv                  reg_all：report / fe_steps / PMAFE_FY25
    表 4（成对放入）             tables/table4_paired.csv                    reg_all：report / paired / PMAFE_FY25 / firm+broker
    表 5（FY2024）               tables/table5_fy2024.csv                    reg_all：report / fe_steps / PMAFE_FY24
    表 6（排名分数）             tables/table6_score.csv                     reg_all：report / fe_steps / SCORE_FY25
    表 7（公司-券商层面）        tables/table7_firm_broker.csv               reg_all：firm_broker / fe_steps / PMAFE_FY25
    表 8（公司-券商层面成对）    tables/table8_firm_broker_paired.csv        reg_all：firm_broker / paired / PMAFE_FY25 / firm+broker
    3.1–3.6 节正文               tables/text_numbers.csv（位置 3.x）          tables.text_numbers
    第 4 节补充说明第 1–8 条     tables/text_numbers.csv（位置 4.1–4.8）      reg_all 的 no_peerlag、peerlag_median、peerlag_npeers、month_fe、
                                                                             window_plus30、paired（SCORE）、same_sample、log_sources、
                                                                             fe_steps、excl_000858 块；tables.text_numbers
"""
from src.config import out_dir

OUT = out_dir('studies', run='forecast_accuracy')
PARTS = ('all', 'tables', 'figures')


def add_args(ap):
    ap.description = '研报逻辑结构与分析师预测准确度：回归样本、回归、文档的表 1–8、正文数字和两张图'
    ap.add_argument('--part', choices=PARTS, default='all', help='all（默认）：表和图都做；tables：只做样本、回归和表；figures：只做样本和图')
    ap.add_argument('--output', default=None, help=f'输出目录，默认 {OUT}')
    ap.add_argument('--fig-dir', default=None, help='图的目录，默认 docs/figures/')


def run(args):
    import time
    from pathlib import Path

    from src.studies.forecast_accuracy import figures, regressions, sample, tables
    t0 = time.perf_counter()
    out = Path(args.output) if args.output else OUT
    fig_dir = Path(args.fig_dir) if args.fig_dir else figures.FIG_DIR
    out.mkdir(parents=True, exist_ok=True)

    S, D, P = sample.build()
    D.to_parquet(out / 'sample_report.parquet', index=False)
    P.to_parquet(out / 'sample_pair.parquet', index=False)
    print(f'研究范围 {len(S)} 篇；研报层面样本 {len(D)} 篇，公司-券商层面样本 {len(P)} 对（{time.perf_counter() - t0:.1f} 秒）')
    est = regressions.estimation_sample(D)
    pl, rl = figures.peer_lag_frames(est)
    pl.to_parquet(out / 'figure2_peer_lags.parquet', index=False)
    rl.to_parquet(out / 'figure2_report_lags.parquet', index=False)

    if args.part in ('all', 'tables'):
        reg = regressions.run_all(D, P)
        reg.to_csv(out / 'reg_all.csv', index=False, encoding='utf-8-sig')
        print(f'reg_all.csv：{len(reg)} 行（' + '，'.join(f'{lv} {n}' for lv, n in reg.level.value_counts(sort=False).items())
              + f'）（{time.perf_counter() - t0:.1f} 秒）')
        dec = tables.variance_decomposition(D)
        dec.to_csv(out / 'variance_decomposition.csv', index=False, encoding='utf-8-sig')
        td = out / 'tables'
        td.mkdir(exist_ok=True)
        for k, t in tables.all_tables(reg, dec).items():
            t.to_csv(td / tables.FILES[k], index=False, encoding='utf-8-sig')
        nums = tables.text_numbers(S, D, P, reg, dec, est, pl, rl, figures.schematic_numbers())
        nums.to_csv(td / tables.NUMBERS_FILE, index=False, encoding='utf-8-sig')
        print(f'表 1–8 和 {len(nums)} 个正文数字写到 {td}')
    if args.part in ('all', 'figures'):
        figures.schematic(fig_dir / figures.FIG1.name)
        figures.peer_lag_plot(pl, rl, fig_dir / figures.FIG2.name)
        print(f'图 1、图 2 写到 {fig_dir}')
    print(f'输出在 {out}，耗时 {time.perf_counter() - t0:.1f} 秒')
