"""专题研究 structure_distribution：六个逻辑结构变量（回归的自变量）的分布，用来判断偏态和取对数是否合适（独立的检查工具）。

只读 measure 的成品（src/measure/load.py 的 read_measures）和 graph 的研报表（src/graph/load.py 的 read_table，取 n_chars），
不调用大模型，不被其他步骤调用，也不改其他步骤的输出；可以反复运行，每次覆盖自己的输出目录。不读 forecast_accuracy 的输出。

口径
  统计对象      measure 的原始值：不缩尾、不标准化
  范围          默认是样本里库中有结果的全部研报（measure 表里有的）；每个变量在它有定义的研报上统计：RT_* 只在“投资评级”
                是严格汇点（RATING_STRICT_SINK）的研报上有值，WG_* 在图非空的研报上有值，其余记为缺失（n_missing）。
  --rating-only 只看 RATING_STRICT_SINK 为真的研报（与 forecast_accuracy 的同名开关同一个条件），但这里不套样本的公司名单，
                也不加 forecast_accuracy 的其他范围条件（严格汇点本身意味着图里有边，即 n_edges > 0）；所以篇数可能多于
                forecast_accuracy --rating-only 的研究范围。
  统计的列      回归用的六个变量 RT_NSRC、RT_LNPATH、RT_PATHLEN、WG_NSRC、WG_LNPATH、WG_PATHLEN（RateSrc、RatePath、RateLen、
                AllSrc、AllPath、AllLen）；路径条数 RT_N_PATHS、WG_N_PATHS；对数变换后的四列 <原名>_LOG1P = ln(1 + 原始值)，
                即 forecast_accuracy --log1p 换掉的 LOG1P_X 四列（源点数、平均路径长度；*_LNPATH 本来就是 ln(1 + 路径数)）。
  每列的统计    n（非缺失）、n_missing、mean、sd（ddof = 1）、min、p1–p99（numpy 线性插值，同 pandas quantile）、max、
                skewness、excess_kurtosis（偏差校正的样本偏度 G1、超额峰度 G2，同 pandas skew、kurt，用 numpy 算）、
                share_min（取最小值的研报占非缺失的比例）、n_unique。
  频数表        RT_NSRC、WG_NSRC、RT_N_PATHS、WG_N_PATHS 的取值频数（只列出现过的值）；≥ k 的取值合并成一行“≥k”，
                k 是使前面各行累计占比 ≥ 99% 的最小整数，且 k ≥ 21（0–20 总是逐个列出）。
  按正文字数    研报表的 n_chars 分组（<1000、1000–2000、2000–5000、5000–10000、≥10000，左闭右开）。

在仓库根目录运行：
    python run.py study structure_distribution                                   默认样本 hs300_deep、版本取样本的主版本
    python run.py study structure_distribution --sample hs300_full --rating-only

输出在 output/studies/structure_distribution/<样本>_<版本>[_rating]/（目录名同 forecast_accuracy：config.study_dir，
--rating-only 时加 _rating）：
    summary.csv            统计表：一列一行（variable、label 和上面的统计）
    frequency.csv          四个整数变量的频数表（variable、value、n_reports、share、cum_share）
    by_length.csv          按正文字数分组，六个变量各自的 n、mean、p50（n_reports 为该组研报数）
    histograms.png         2 行 3 列直方图：上排 RT、下排 WG 的三个回归变量（原始值；源点数整数柱、横轴截到 p99）
    histograms_log1p.png   同样布局，源点数和平均路径长度换成 ln(1 + x)（路径数那一列与上一张相同）
    summary.md             一页中文小结（三张统计表、频数表、按字数表和两张图，只有数字和口径）
    manifest.json          样本、版本、是否 rating_only、篇数、生成时间（src/samples 的 write_manifest）
"""
from src.config import DEFAULT_SAMPLE, study_dir

STUDY = 'structure_distribution'


def add_args(ap):
    ap.description = '六个逻辑结构变量（原始值）的分布：统计表、频数表、按字数分组和直方图（独立的检查工具，只读成品）'
    ap.add_argument('--sample', default=DEFAULT_SAMPLE, help=f'样本，默认 {DEFAULT_SAMPLE}；样本见 python run.py sample')
    ap.add_argument('--version', default=None, help='抽取版本，默认样本的主版本（照清单）；给别的版本时样本的全部研报都取自它')
    ap.add_argument('--rating-only', action='store_true',
                    help='只看“投资评级”是严格汇点的研报（不套公司名单），输出目录加 _rating')


def run(args):
    import time

    import pandas as pd

    from src import samples
    from src.graph import version_spec
    from src.graph.load import read_table
    from src.measure.load import read_measures
    from src.studies.structure_distribution import stats
    samples.spec(args.sample)
    if args.version:
        version_spec(args.version)
    version = args.version or samples.main_version(args.sample)
    out = study_dir(STUDY, args.sample, version, '_rating' if args.rating_only else '')
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    m = read_measures(sample=args.sample, version=args.version)
    r = read_table('reports', args.sample, columns=['n_chars'], version=args.version)
    x = m.merge(r[['report_id', 'n_chars']], on='report_id', how='left', validate='1:1')
    n_lib = len(x)
    if args.rating_only:
        x = x[x.RATING_STRICT_SINK.astype(bool)].reset_index(drop=True)
    x = stats.add_log1p(x)
    scope = '“投资评级”是严格汇点的研报（--rating-only，不套公司名单）' if args.rating_only else '库中有结果的全部研报'
    print(f'样本 {args.sample}，抽取版本 {version}，范围：{scope}，{len(x)} 篇（库中有结果 {n_lib} 篇）')

    tabs = {'summary.csv': stats.summary(x), 'frequency.csv': stats.frequency(x), 'by_length.csv': stats.by_length(x)}
    for name, df in tabs.items():
        df.to_csv(out / name, index=False, encoding='utf-8-sig', float_format='%.6f', lineterminator='\n')
    stats.histograms(x, out / 'histograms.png')
    stats.histograms(x, out / 'histograms_log1p.png', log1p=True)
    write_md(out / 'summary.md', args.sample, version, f'{scope}，{len(x)} 篇', tabs)
    samples.write_manifest(out, study=STUDY, sample=args.sample, version=version, rating_only=bool(args.rating_only),
                           n_reports=len(x), n_extracted=n_lib, n_sample=len(samples.read(args.sample)))

    with pd.option_context('display.width', 250, 'display.max_columns', 30):
        print('\n[summary]\n' + tabs['summary.csv'].drop(columns='label').to_string(index=False, float_format=lambda v: f'{v:.3f}'))
    print(f'\n输出在 {out}（{time.perf_counter() - t0:.1f} 秒）')


def md_table(df, pct=()):
    """DataFrame -> Markdown 表：浮点数保留 2 位小数，pct 里的列显示为百分比，缺失值留空。"""
    import pandas as pd

    def fmt(v, c):
        if v is None or (isinstance(v, float) and pd.isna(v)) or v is pd.NA:
            return ''
        if c in pct:
            return f'{v:.2%}'
        return f'{v:.2f}' if isinstance(v, float) else str(v)
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join(fmt(v, c) for v, c in zip(row, cols)) + ' |' for row in df.itertuples(index=False)]
    return '\n'.join(lines)


def write_md(path, sample, version, scope, tabs):
    """一页中文小结：只有数字和口径。"""
    import time

    from src.studies.structure_distribution.stats import INTS, LOG1P, NPATHS, X6
    s, f = tabs['summary.csv'], tabs['frequency.csv']
    parts = ['# 逻辑结构变量的分布', '',
             f'样本 {sample}；抽取版本 {version}；范围：{scope}；生成时间 {time.strftime("%Y-%m-%d %H:%M:%S")}', '',
             '## 统计表', '',
             '统计对象是 measure 的原始值（不缩尾、不标准化）；每个变量在它有定义的研报上统计，n 是非缺失篇数，n_missing 是范围内'
             '缺失的篇数（RT_* 只在“投资评级”是严格汇点时有值，WG_* 在图非空时有值）。分位数为线性插值；skewness、excess_kurtosis 为'
             '偏差校正的样本偏度和超额峰度；share_min 是取最小值的研报占非缺失的比例。', '']
    for title, cols, note in [
            ('回归用的六个变量', X6, '源点数为整数；路径数一列已是 ln(1 + 路径条数)；平均路径长度按边数计。'),
            ('路径条数', NPATHS, '取对数前的路径条数（整数）。'),
            ('对数变换后', LOG1P, '与回归 --log1p 相同的变换：源点数、平均路径长度换成 ln(1 + 原始值)。')]:
        parts += [f'### {title}', '', note, '', md_table(s[s.variable.isin(cols)]), '']
    parts += ['## 整数变量的频数表', '',
              '只列出现过的取值；≥ k 的取值合并成一行，k 是使前面各行累计占比 ≥ 99% 的最小整数且不小于 21。share、cum_share 的分母是'
              '该变量的非缺失篇数。每个变量最多显示 25 行，完整的表见 frequency.csv。', '']
    for c in INTS:
        g = f[f.variable == c]
        more = f'（共 {len(g)} 行，这里显示前 25 行）' if len(g) > 25 else ''
        parts += [f'### {c}{more}', '', md_table(g.drop(columns='variable').head(25), pct=('share', 'cum_share')), '']
    parts += ['## 按正文字数分组', '',
              '按研报表的 n_chars 分组（左闭右开）；n_reports 是该组的研报数，n 是该变量非缺失的篇数，mean、p50 在非缺失值上算。', '',
              md_table(tabs['by_length.csv']), '',
              '## 直方图', '',
              '上排为“投资评级”对应的三个变量，下排为全文的三个变量；虚线为中位数，图上的 n、中位数、偏度按全部非缺失值算。'
              '原始值图中源点数用整数柱，横轴截到 p99（截掉的篇数写在子图里）；对数图中源点数和平均路径长度换成 ln(1 + x)，'
              '路径数一列与原始值图相同。', '',
              '![原始值](histograms.png)', '', '![对数变换](histograms_log1p.png)', '']
    path.write_text('\n'.join(parts), encoding='utf-8')
