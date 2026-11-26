"""专题研究 forecast_accuracy：研报逻辑结构与分析师预测准确度（文档 研报逻辑结构与分析师预测准确度.md 的全部表、图和数字）。

样本和版本：--sample（默认 hs300_deep）决定研究哪些研报，--version（默认样本的主版本，hs300_deep 是 v1）决定逻辑结构取自
哪个抽取版本的库（给的不是主版本时样本的全部研报都从那个版本取，见 src/samples 的 assignment）。输出目录
output/studies/forecast_accuracy/<样本>_<版本>/（--rating-only 时目录名加 _rating，--log1p 时再加 _log1p，两个都给时是
<样本>_<版本>_rating_log1p），文档 md、PDF 和 figures/ 也在里面（入 git）；
每次跑 sample 环节都写一份 manifest.json（样本、版本、是否 log1p、篇数、各版本篇数、生成时间；给了 --rating-only 时另记
rating_only: true，不给时不写这一项，manifest 与加这个开关之前相同）。

范围开关 --rating-only：研究范围（sample.scope）在原有规则（样本里 n_edges > 0、样本登记了公司名单时限定在名单内）之后，
再限定在“投资评级”是严格汇点（入度 > 0、出度 = 0；measure 的 RATING_STRICT_SINK 为真）的研报，所以 RT_* 三个变量在范围内全部
有定义。之后的基础样本、券商分组、缩尾、标准化、公司-券商层面的样本对都在这个范围内照原规则做，口径不另改。和 --log1p 一样，
它在 sample 环节决定范围，其他环节读同一个输出目录：单独跑某个环节时也要带上 --rating-only（或同一个 --output）。
不给时行为与加这个开关之前完全相同。

输入：上游步骤的成品（先跑 python run.py measure --version <版本>、python run.py accuracy --sample <样本>）src.graph.load、
src.measure.load、src.accuracy.load，样本登记的公司名单（hs300_deep 是 data/hs300_2024_dag_covered.csv）、data/market_data/industry.xlsx，以及外部的预测误差文件 data/forecast_errors.parquet（被解释变量，
见 errors.py；describe 环节的中位数核对另读 data/EST_COMP_TN）。样本、缩尾、回归设定的口径见 sample.py、errors.py、regressions.py
的模块说明；回归工具在 src/ols.py。
被解释变量：主变量 Y_MAIN 由 sample.MAIN_DEF 决定，现为 PMED180_NP_Hm = (|A − F| − M180) / M180（净利润；用数据文件的 180 天
同行绝对误差中位数派生，文件没有直接给出）；其余 53 个定义 × 口径 × 窗口用于稳健性；accuracy 自算的 PMAFE_Hm、SCORE_Hm 作对照。
下面 Ym 指主变量。预测误差按预测窗口 H1、H2、H3 定义（第 h 年预测，目标年度 = 预测行的 RPT_YR_T + h，逐篇不同）；
主窗口只由 sample.py 的 MAIN_H（现为 2）决定（m = MAIN_H），表 8 的两个 Panel 是其余两个窗口（OTHER_H）。

环节（--part，逗号分隔，按下表的顺序执行；不给时做 sample 到 figures 六个，pdf 要显式指定，因为它依赖本机的 Chrome / Edge）
每个环节只读前面环节写在输出目录里的成品，缺成品时报错并提示先跑哪个环节，不会悄悄重算上游。
    环节      做什么                          读（输出目录内，除注明外）                     写（输出目录内，除注明外）
    sample    选样本、并入数据文件的预测误差、 上游成品（graph / measure / accuracy）、      scope.parquet、sample_report.parquet、sample_pair.parquet、
              缩尾；图 2 的同行明细            data/（含 forecast_errors.parquet）           forecast_errors_rows.parquet、figure2_peer_lags.parquet、
                                                                                             figure2_report_lags.parquet
    describe  描述性统计（表 1、3）、一致性    sample_report.parquet、scope.parquet、         error_stats.csv、descriptive_stats.csv、
              （表 2）和中位数核对             forecast_errors_rows.parquet（另读数据文件     tables/table1_error_descriptive.csv、tables/consistency.csv、
                                              和 data/EST_COMP_TN）                          tables/consistency_fields.csv、tables/table3_descriptive.csv
    decomp    固定效应分解（表 4）            sample_report.parquet                          variance_decomposition.csv、tables/table4_fe_decomposition.csv
    regress   全部回归                        sample_report.parquet、sample_pair.parquet    reg_all.csv
    tables    回归表（表 5–13）和正文数字     reg_all.csv、descriptive_stats.csv、           tables/table5_…–table13_….csv、tables/text_numbers.csv
                                              variance_decomposition.csv、scope.parquet、
                                              两个 sample_*、两个 figure2_*（另读 graph 的概念表）
    figures   图 1、图 2                      figure2_*.parquet                              figures/ 下两张 png（--fig-dir 可改）
    pdf       文档 md -> PDF（src/mdpdf.py）  输出目录里的文档 md 和 figures/                <md 文件名>.html（中间文件）；md 旁边同名 .pdf
估计样本（regressions.estimation_sample）各环节需要时现算，不存盘。csv 读回时数值与写出前逐位相同（见 regressions.read_csv、
regressions.read_reg），所以分环节运行和一次跑完的结果完全相同。--part all 等于默认的六个环节。

在仓库根目录运行：
    python run.py study forecast_accuracy                           sample 到 figures 全部重做（一分半左右，回归约一分钟）
    python run.py study forecast_accuracy --part tables             只用已有的 reg_all 等成品重出表 5–13 和正文数字
    python run.py study forecast_accuracy --part describe,tables    重算描述性统计和一致性，再重出正文数字
    python run.py study forecast_accuracy --part regress --blocks paired,firm_broker:paired
                                                                    只重估这几块（块名见 regressions.py），其余行不动
    python run.py study forecast_accuracy --part pdf                输出目录里的文档转 PDF（--doc 换别的 md）
    python run.py study forecast_accuracy --log1p                   六个逻辑结构变量都取 ln(1 + x)（见 sample.py），输出在
                                                                    output/studies/forecast_accuracy/hs300_deep_v1_log1p/；--log1p
                                                                    只在 sample 环节改变变换，其他环节读这个目录（单独跑其他环节时也要带上）
    python run.py study forecast_accuracy --sample hs300_full       另一个样本，输出在 hs300_full_v1/
    python run.py study forecast_accuracy --sample hs300_full --rating-only
                                                                    只研究“投资评级”是严格汇点的研报，输出在 hs300_full_v1_rating/
                                                                    （共约 8 分钟，回归约 5 分钟）；单独跑其他环节时也要带上 --rating-only

输出在 output/studies/forecast_accuracy/<样本>_<版本>[_rating][_log1p]/（下面的篇数是 hs300_deep_v1 的）：
    scope.parquet                研究范围 791 篇（合并了 accuracy 的全部列和 measure 的原始值）           sample.scope
    sample_report.parquet        研报层面回归样本（786 篇基础样本；<变量>_RAW 为缩尾前的值）             sample.report_sample
    sample_pair.parquet          公司-券商层面回归样本（760 对）                                         sample.pair_sample
    forecast_errors_rows.parquet 数据文件里 report_id 属于研究范围或被评价预测的全部行                    errors.read_rows
    figure2_peer_lags.parquet    图 2 (a) 的同行明细（基准回归估计样本 × 主窗口比较组的同行）           figures.peer_lag_frames
    figure2_report_lags.parquet  图 2 (b) 的每篇研报同行时滞                                             figures.peer_lag_frames
    error_stats.csv              预测误差变量的描述性统计长表（786 篇里各列的非缺失值）                  tables.error_stats
    descriptive_stats.csv        表 3 的描述性统计长表（六个变量 × 缩尾后 / 缩尾前）                     tables.descriptive_stats
    variance_decomposition.csv   表 4 的方差分解（六个变量 × 六组固定效应的 N、K、R²、调整 R²）           tables.variance_decomposition
    reg_all.csv                  全部回归系数的长表（列见 regressions.py）                               regressions.run_all
    tables/                      文档表 1–13（文件名见 tables.FILES；表 2 = consistency.csv 的 H2 各行）、 tables.table1、table3、table4、
                                 consistency_fields.csv（字段层面的一致性、中位数核对）                  reg_tables；errors.consistency 等
    tables/text_numbers.csv      正文、图注、第 4 节的数字（位置、数值、小数位、显示、星号、算法）        tables.text_numbers
    figures/                     图 1、图 2（文档 md 用相对路径 figures/ 引用）                          figures.schematic、peer_lag_plot
    研报逻辑结构与分析师预测准确度.md、.pdf   文档和 PDF（对数变量版的目录里是 …_对数变量.md）           人工撰写；src.mdpdf
    <md 文件名>.html             pdf 环节的中间 HTML                                                     src.mdpdf
    manifest.json                这份输出用的样本、版本、是否 log1p（--rating-only 时另记 rating_only）、  stage_sample
                                 篇数、各版本篇数、生成时间

文档的表 / 图 / 节 -> 输出文件 -> 生成它的函数
    1.1、1.2 节样本数字          tables/text_numbers.csv（位置 1.1、1.2）    sample.scope、tables.text_numbers
    图 1（逻辑结构变量的构建）   figures/logic_structure_schematic.png        figures.schematic（例子的数字：位置“1.2 图 1 的例子”）
    表 1（预测误差变量）         tables/table1_error_descriptive.csv         tables.error_stats、tables.table1
    表 2（与自算 PMAFE 的一致性）tables/consistency.csv（窗口 = H2 的行）    errors.consistency
    1.3 节一致性、中位数核对     tables/consistency.csv、consistency_fields.csv  errors.consistency_fields、errors.median_check
    1.3、1.4 节其他数字          tables/text_numbers.csv（位置 1.3、1.3 窗口与财年、1.4） tables.text_numbers
    表 3（逻辑结构变量描述统计） tables/table3_descriptive.csv               tables.descriptive_stats、tables.table3
    2.1 节、表 3 注              tables/text_numbers.csv（位置 2.1、表 3 注） tables.text_numbers
    表 4（固定效应分解）         tables/table4_fe_decomposition.csv          tables.variance_decomposition、tables.table4
    2.2 节                       tables/text_numbers.csv（位置 2.2）          tables.text_numbers
    图 2（同行时滞分布）         figures/peer_lag_distribution.png            figures.peer_lag_frames、figures.peer_lag_plot
    2.3 节、图 2 注              tables/text_numbers.csv（位置 2.3、图 2 注） tables.text_numbers
    表 5（基准结果）             tables/table5_benchmark.csv                 reg_all：report / fe_steps / Ym_Hm / firm+broker
    表 6（不同固定效应）         tables/table6_fe_steps.csv                  reg_all：report / fe_steps / Ym_Hm
    表 7（成对放入）             tables/table7_paired.csv                    reg_all：report / paired / Ym_Hm（Panel A 公司 + 券商，B 公司）
    表 8（其他预测窗口）         tables/table8_other_horizons.csv            reg_all：report / fe_steps / Ym_Ho（OTHER_H 依次为 Panel A、B）
    表 9、10（不同误差定义 H2）  tables/table9_…、table10_….csv              reg_all：report / fe_steps、definitions、own_measures（公司 + 券商；公司）
    表 11（不同误差定义 H1、H3） tables/table11_definitions_other_horizons.csv  reg_all：report / fe_steps、definitions
    表 12（公司-券商层面）       tables/table12_firm_broker.csv              reg_all：firm_broker / fe_steps / Ym_Hm
    表 13（公司-券商层面成对）   tables/table13_firm_broker_paired.csv       reg_all：firm_broker / paired / Ym_Hm / firm+broker
    3.1–3.6 节正文               tables/text_numbers.csv（位置 3.x）          tables.text_numbers
    第 4 节补充说明第 1–9 条     tables/text_numbers.csv（位置 4.1–4.9）      reg_all 的 no_peerlag、peerlag_median、peerlag_npeers、month_fe、
                                                                             definitions（75 天窗口）、paired（自算 PMAFE）、same_sample、
                                                                             log_sources、fe_steps、excl_000858、firm_x_year 块；tables.text_numbers
对数变量版的文档（hs300_deep_v1_log1p/研报逻辑结构与分析师预测准确度_对数变量.md）用 --log1p 的输出，没有表 3，其余表号依次减 1。
"""
from src.config import DEFAULT_SAMPLE, study_dir

STUDY = 'forecast_accuracy'
DOC = '研报逻辑结构与分析师预测准确度.md'  # pdf 环节默认转换的文档（在输出目录里）
DOC_LOG1P = '研报逻辑结构与分析师预测准确度_对数变量.md'  # --log1p 时的默认文档
STAGES = ('sample', 'describe', 'decomp', 'regress', 'tables', 'figures', 'pdf')  # 执行顺序
DEFAULT_PARTS = STAGES[:-1]


def add_args(ap):
    ap.description = '研报逻辑结构与分析师预测准确度：回归样本、描述性统计、一致性、方差分解、回归、文档的表 1–13、正文数字、两张图和 PDF'
    ap.add_argument('--part', default=None,
                    help='要做的环节，逗号分隔，按固定顺序执行：' + '、'.join(STAGES)
                    + '；默认（或 all）为 ' + ','.join(DEFAULT_PARTS) + '，pdf 要显式指定')
    ap.add_argument('--blocks', default=None,
                    help='regress 环节只重估这些块（逗号分隔，块名同 reg_all 的 block 列；公司-券商层面写 firm_broker:fe_steps、'
                         'firm_broker:paired），其余行取自已有的 reg_all.csv；不给时全部重估')
    ap.add_argument('--sample', default=DEFAULT_SAMPLE, help=f'样本（研究哪些研报），默认 {DEFAULT_SAMPLE}；样本见 python run.py sample')
    ap.add_argument('--version', default=None,
                    help='逻辑结构取自哪个抽取版本；默认样本的主版本（照清单，个别研报可以取自别的版本），给别的版本时样本的全部研报都取自它')
    ap.add_argument('--output', default=None,
                    help='输出目录，默认 output/studies/forecast_accuracy/<样本>_<版本>/（--rating-only 时加 _rating，--log1p 时再加 _log1p）')
    ap.add_argument('--log1p', action='store_true', help='六个逻辑结构变量都取 ln(1 + x)（源点数、平均路径长度取 log1p，路径数不变）')
    ap.add_argument('--rating-only', action='store_true',
                    help='研究范围再限定在“投资评级”是严格汇点的研报（measure 的 RATING_STRICT_SINK），之后照原规则做；'
                         '只在 sample 环节起作用，单独跑其他环节时也要带上（决定输出目录）')
    ap.add_argument('--fig-dir', default=None, help='图的目录，默认 <输出目录>/figures/')
    ap.add_argument('--doc', default=None, help=f'pdf 环节要转换的 md，默认 <输出目录>/{DOC}（--log1p 时 {DOC_LOG1P}）；PDF 写在 md 旁边、同名')


def default_out(sample, version, log1p, rating_only=False):
    """默认输出目录：output/studies/forecast_accuracy/<样本>_<版本>[_rating][_log1p]/；version 为 None 时用样本的主版本。"""
    from src import samples
    suffix = ('_rating' if rating_only else '') + ('_log1p' if log1p else '')
    return study_dir(STUDY, sample, version or samples.main_version(sample), suffix)


def parse_parts(text):
    """--part 的取值 -> 按 STAGES 顺序排好的环节列表。"""
    if text is None or text.strip() == 'all':
        return list(DEFAULT_PARTS)
    parts = [s.strip() for s in text.split(',') if s.strip()]
    bad = [s for s in parts if s not in STAGES]
    if bad or not parts:
        raise SystemExit(f'没有环节 {"、".join(bad) or repr(text)}；可选：{"、".join(STAGES)}（或 all）')
    return [s for s in STAGES if s in parts]


class Ctx:
    """一次运行的参数：输出目录、图目录、--blocks 解析出的块，以及读前面环节成品的工具。"""

    def __init__(self, args, out, fig_dir, only):
        self.args, self.out, self.fig_dir, self.only = args, out, fig_dir, only
        self.src = dict(sample=args.sample, version=args.version)  # 读上游成品用的样本和版本
        self.hint = ((f' --sample {args.sample}' if args.sample != DEFAULT_SAMPLE else '')
                     + (f' --version {args.version}' if args.version else '')
                     + (' --rating-only' if args.rating_only else '')
                     + (' --log1p' if args.log1p else '') + (f' --output {args.output}' if args.output else ''))

    def load(self, name, stage):
        """读输出目录里的成品 name（由环节 stage 写出）；不存在时报错并提示先跑哪个环节。"""
        import pandas as pd

        from src.studies.forecast_accuracy import regressions
        p = self.out / name
        if not p.exists():
            raise SystemExit(f'缺少 {p}\n请先运行：python run.py study forecast_accuracy --part {stage}{self.hint}')
        if name.endswith('.parquet'):
            return pd.read_parquet(p)
        return regressions.read_reg(p) if name == 'reg_all.csv' else regressions.read_csv(p)

    def write(self, df, name):
        """写到输出目录（csv 用 utf-8-sig）；返回路径。"""
        p = self.out / name
        p.parent.mkdir(parents=True, exist_ok=True)
        if name.endswith('.parquet'):
            df.to_parquet(p, index=False)
        else:
            df.to_csv(p, index=False, encoding='utf-8-sig')
        return p


# ---------------------------------------------------------------- 各环节：返回写出的文件列表
def stage_sample(c):
    from src import samples
    from src.graph.load import owners
    from src.studies.forecast_accuracy import figures, regressions, sample
    S, D, P, R = sample.build(c.args.log1p, **c.src, rating_only=c.args.rating_only)
    pl, rl = figures.peer_lag_frames(regressions.estimation_sample(D), c.args.sample)
    print(f'  研究范围 {len(S)} 篇' + ('（--rating-only：“投资评级”是严格汇点）' if c.args.rating_only else '')
          + f'；研报层面样本 {len(D)} 篇，公司-券商层面样本 {len(P)} 对；数据文件 {len(R)} 行；'
          f'图 2：{len(rl)} 篇、{len(pl)} 条同行预测')
    owner = owners(**c.src)
    vs = [owner[r] for r in S.report_id]
    extra = dict(rating_only=True) if c.args.rating_only else {}  # 不给开关时不写这一项，manifest 与原来相同
    samples.write_manifest(
        c.out, study=STUDY, sample=c.args.sample, version=c.args.version or samples.main_version(c.args.sample),
        version_rule='照样本清单' if c.args.version in (None, samples.main_version(c.args.sample)) else '样本的全部研报都取自该版本',
        log1p=bool(c.args.log1p), **extra, n_sample=len(samples.read(c.args.sample)), n_with_graph=len(owner),
        n_scope=len(S), scope_by_version={v: vs.count(v) for v in dict.fromkeys(vs)},
        n_report_sample=len(D), n_pair_sample=len(P))
    return [c.write(S, 'scope.parquet'), c.write(D, 'sample_report.parquet'), c.write(P, 'sample_pair.parquet'),
            c.write(R, 'forecast_errors_rows.parquet'), c.write(pl, 'figure2_peer_lags.parquet'),
            c.write(rl, 'figure2_report_lags.parquet'), c.out / 'manifest.json']


def stage_describe(c):
    import pandas as pd

    from src.studies.forecast_accuracy import errors, tables
    S, D = c.load('scope.parquet', 'sample'), c.load('sample_report.parquet', 'sample')
    R = c.load('forecast_errors_rows.parquet', 'sample')
    desc = tables.descriptive_stats(D)
    n = desc[desc.口径 == '缩尾后'].set_index('变量').N
    print(f'  样本：RT 三个变量 {n["RT_NSRC"]} 篇，WG 三个变量 {n["WG_NSRC"]} 篇')
    es = tables.error_stats(D)
    cons = errors.consistency(D)
    med = errors.median_check(R, S).rename(columns={'口径': '窗口'}).assign(说明='M75 是否为同行绝对误差的中位数（窗口列为来源、口径、n75 奇偶）')
    fields = pd.concat([errors.consistency_fields(R, S), med], ignore_index=True)
    return [c.write(desc, 'descriptive_stats.csv'), c.write(es, 'error_stats.csv'),
            c.write(tables.table1(es), f'tables/{tables.FILES[1]}'), c.write(cons, f'tables/{tables.FILES[2]}'),
            c.write(fields, f'tables/{tables.FIELDS_FILE}'), c.write(tables.table3(desc), f'tables/{tables.FILES[3]}')]


def stage_decomp(c):
    from src.studies.forecast_accuracy import tables
    dec = tables.variance_decomposition(c.load('sample_report.parquet', 'sample'))
    return [c.write(dec, 'variance_decomposition.csv'), c.write(tables.table4(dec), f'tables/{tables.FILES[4]}')]


def stage_regress(c):
    from src.studies.forecast_accuracy import regressions
    old = None
    if c.only:
        if not (c.out / 'reg_all.csv').exists():
            raise SystemExit(f'--blocks 要替换已有 reg_all.csv 里的块，但 {c.out / "reg_all.csv"} 不存在\n'
                             f'请先运行：python run.py study forecast_accuracy --part regress{c.hint}（不带 --blocks）')
        old = c.load('reg_all.csv', 'regress')
        print('  只重估：' + '、'.join(':'.join(k) for k in c.only))
    D, P = c.load('sample_report.parquet', 'sample'), c.load('sample_pair.parquet', 'sample')
    reg = regressions.run_all(D, P, c.only, old)
    print(f'  {len(reg)} 行（' + '，'.join(f'{lv} {n}' for lv, n in reg.level.value_counts(sort=False).items()) + '）')
    return [c.write(reg, 'reg_all.csv')]


def stage_tables(c):
    from src.studies.forecast_accuracy import figures, regressions, tables
    reg = c.load('reg_all.csv', 'regress')
    desc = c.load('descriptive_stats.csv', 'describe')
    dec = c.load('variance_decomposition.csv', 'decomp')
    S, D, P = (c.load(n, 'sample') for n in ('scope.parquet', 'sample_report.parquet', 'sample_pair.parquet'))
    pl, rl = c.load('figure2_peer_lags.parquet', 'sample'), c.load('figure2_report_lags.parquet', 'sample')
    files = [c.write(t, f'tables/{tables.FILES[k]}') for k, t in tables.reg_tables(reg).items()]
    nums = tables.text_numbers(S, D, P, reg, desc, dec, regressions.estimation_sample(D), pl, rl, figures.schematic_numbers(),
                               **c.src, rating_only=c.args.rating_only)
    print(f'  表 5–13 和 {len(nums)} 个正文数字')
    return files + [c.write(nums, f'tables/{tables.NUMBERS_FILE}')]


def stage_figures(c):
    from src.studies.forecast_accuracy import figures
    pl, rl = c.load('figure2_peer_lags.parquet', 'sample'), c.load('figure2_report_lags.parquet', 'sample')
    f1, f2 = c.fig_dir / figures.FIG1.name, c.fig_dir / figures.FIG2.name
    figures.schematic(f1)
    figures.peer_lag_plot(pl, rl, f2)
    return [f1, f2]


def stage_pdf(c):
    from pathlib import Path

    from src import mdpdf
    doc = Path(c.args.doc).resolve() if c.args.doc else c.out / (DOC_LOG1P if c.args.log1p else DOC)
    if not doc.exists():
        raise SystemExit(f'没有文档 {doc}（文档 md 放在研究输出目录里；--doc 可指定别的 md）')
    html = c.out / (doc.stem + '.html')
    pdf = mdpdf.convert(doc, html)
    print(f'  {pdf.stat().st_size / 1024:.0f} KB')
    return [html, pdf]


STAGE_FUNCS = {'sample': stage_sample, 'describe': stage_describe, 'decomp': stage_decomp, 'regress': stage_regress,
               'tables': stage_tables, 'figures': stage_figures, 'pdf': stage_pdf}


def run(args):
    import time
    from pathlib import Path

    from src.studies.forecast_accuracy import figures, regressions
    parts = parse_parts(args.part)
    if args.blocks and 'regress' not in parts:
        raise SystemExit('--blocks 只用于 regress 环节：请同时给 --part regress')
    only = regressions.parse_blocks(args.blocks) if args.blocks else None  # 先检查块名，免得跑到一半才报错
    t0 = time.perf_counter()
    from src import samples
    from src.graph import version_spec
    samples.spec(args.sample)
    if args.version:
        version_spec(args.version)
    out = Path(args.output) if args.output else default_out(args.sample, args.version, args.log1p, args.rating_only)
    out.mkdir(parents=True, exist_ok=True)
    c = Ctx(args, out, Path(args.fig_dir) if args.fig_dir else out / figures.FIG_SUBDIR, only)
    for name in parts:
        t = time.perf_counter()
        print(f'[{name}]')
        files = STAGE_FUNCS[name](c)
        print(f'  写了 {len(files)} 个文件（{time.perf_counter() - t:.1f} 秒）：')
        for p in files:
            print(f'    {p}')
    print(f'输出在 {out}，{"、".join(parts)} 共 {time.perf_counter() - t0:.1f} 秒')
