"""第三步：分歧度与股票数据的分组统计（不调用大模型，可以反复运行）。
用抽取和框架对比的结果算出每家公司每个季度的分歧度（总分歧 = 主题项 + 关系项），接上股票行情和财务数据，
再按两个分歧指标做双重分组，统计每个格子的下季度股票回报均值（带“高-低”和 t 值）、当季末市值均值和样本数。
回报按 Ang, Hodrick, Xing & Zhang (2006) 的组合做法：每组在季末建仓、持有下一个季度，算组合每天的回报；
“高-低”是最高组和最低组每日回报之差的平均，t 值用 Newey-West 标准误。
另外用几种算法量每篇研报逻辑图的大小和深度，把深度报告和财报点评分开，在公司-季度内取平均后分组，看每组的下季度回报。
分歧度要先跑完 extract_logic_chain.py 的抽取和 --task compare；逻辑链大小和深度直接读各篇研报的存档，抽取还在跑时也能用。
各列定义和算法见 src/divergence_sort.py 开头。

在仓库根目录运行：
    python sort_divergence.py                     按下面的运行参数：逻辑链大小和深度分组 + 算分歧度 -> 匹配股票数据 -> 双重分组
    python sort_divergence.py --task size         只做逻辑链大小和深度的分组（深度报告、财报点评各一张表，每个算法一行）
    python sort_divergence.py --task size --size-var depth,depth_mean    只用这几个算法
    python sort_divergence.py --task size --size-source graphs           用 data/graphs/ 里的逻辑图（上一次完整运行抽取时写出的）
    python sort_divergence.py --task divergence   只算公司-季度分歧度
    python sort_divergence.py --task sort         分歧度表已经有了，只重做匹配和分组（换分组方式时用，几秒钟）
    python sort_divergence.py --task sort --groups 5 --how dependent --weight value  五分组、条件分组、市值加权的组合
    python sort_divergence.py --task sort --row pair_theme --col pair_rel            换成两两平均口径的分歧度
    python sort_divergence.py --task reg          三种分歧度对股票收益、财务特征两组变量的回归（系数和聚类 t 值）
    python sort_divergence.py --task text         去掉补全边、只用原文的逻辑图：子图数等图结构指标对股票收益和财务特征的回归和分组（src/text_graph.py）
    python sort_divergence.py --output output_sample                                 换输出目录

结果在 <输出目录>/data/divergence/：
    company_quarter.parquet   公司-季度分歧度
    panel.parquet             接上季末市值、下季度回报和财务数据之后的面板；panel_sorted.parquet 另带分组编号 g_row、g_col
    sort_single.csv           按总分歧、主题项、关系项各自分组的日均回报（每个指标一行，含“高-低”和 t 值）；sort_single_detail.csv 是每组明细
    sort_next_ret.csv         双重分组每个格子持有下一个季度的日均回报（基点），含“高-低”和 t 值
    sort_mkcap.csv            季末市值均值（亿元）
    sort_count.csv            每个格子的公司-季度个数
    company_quarter_reports.parquet   参与分歧度计算的研报（算研报发布窗口的超额收益用）
    panel_reg.parquet         回归用的变量：分歧度、股票收益类变量、财务特征
    reg_returns.csv、reg_financial.csv   分歧度对股票收益、财务特征的回归（每个因变量、模型、回归、分歧指标一行）
    report_sizes.parquet      每篇研报的类型（deep 深度报告 / earn 财报点评 / other 其他）和逻辑图的各种大小、深度
    company_quarter_size.parquet   公司-季度内各类研报的平均值，接上季末市值和下季度回报
    size_sort_deep.csv、size_sort_earn.csv   深度报告、财报点评按每个算法分组后的下季度回报均值（每个算法一行，带高-低和 t 值）
    size_sort_detail.csv      每个算法每组的指标均值、下季度回报、季末市值和个数
    text_report.parquet、text_company_quarter.parquet   原文图的各项指标（每篇研报 / 公司-季度平均），接上股票和财务变量
    text_reg_returns.csv、text_reg_financial.csv、text_reg_report.csv   原文图指标的回归（公司-季度两组因变量、研报窗口收益）
    text_sort.csv、text_sort_detail.csv   按原文图指标分组后，组合持有下一个季度的日均回报
"""
import argparse
import sys

from src.config import DATA_DIR, OUTPUT_DIR

# ==================== 运行参数 ====================
OUT_DIR = OUTPUT_DIR  # 抽取和框架对比的输出目录，也是本脚本的输出目录
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'  # 日行情：date、code（6 位，不带后缀）、close（后复权）、mkcap（总市值，元）
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'  # 季度财务数据：code、report_date（报告期）、date（公告日）和各科目
FIN_COLS = ['total_assets', 'total_liability', 'operating_revenue_ttm', 'np_parent_company_owners_ttm']  # 接到面板里的财务科目，空列表表示不接
PARAGRAPHS = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # 正文段落（判断研报类型用：深度报告 / 财报点评 / 其他），和抽取用的同一个
REPORTS = None  # 研报元数据表（从存档读逻辑图时用）；None 表示和段落表配套的那张（文件名里的 paragraphs 换成 reports）
TASK = 'all'  # 'all' 全部；'divergence' 只算分歧度；'sort' 只做分歧度的匹配和双重分组；'size' 只做逻辑链大小的分组；'reg' 只做回归
MIN_REPORTS = 2  # 一个公司-季度至少要有几篇参与比较的研报
ROW_VAR = 'theme'  # 分组的行指标：theme 主题项、rel 关系项、total 总分歧，或两两平均口径的 pair_theme、pair_rel、pair_total
COL_VAR = 'rel'  # 分组的列指标
N_GROUPS = 3  # 双重分组时每个指标分几组
SINGLE_VARS = ['total', 'theme', 'rel']  # 双重分组之前先按这些分歧指标各自做单变量分组；空列表表示不做
SINGLE_GROUPS = 3  # 单变量分组分几组
HOW = 'independent'  # 'independent' 两个指标各自分组；'dependent' 先按行指标分组，再在每个行组内按列指标分组
WEIGHT = 'equal'  # 组合内的权重：'equal' 等权；'value' 按建仓时（当季末）的总市值加权
REG_VARS = ['total', 'theme', 'rel']  # 回归用的分歧指标：每个单独回归一次，再把主题项和关系项放在一起回归一次；空列表表示不做
CAR_WINDOW = (-3, 1)  # 研报发布窗口：发布日前 3 个到后 1 个交易日（目录日期有时比落款晚 1-3 天，所以往前多取几天）
NW_LAGS = None  # 高-低日回报差 t 值的 Newey-West 滞后阶数；None 按经验规则 floor(4·(交易日数/100)^(2/9))
SIZE_VARS = ['n_concepts', 'n_edges', 'n_text_edges', 'chains', 'density',  # 逻辑链大小的算法：概念数、边数、原文边数、链条数、每千字概念数
             'depth', 'depth_mean', 'depth_flow', 'mean_layer']  # 逻辑链深度的算法：最长链步数、平均链长、流量平均步数、平均层级
SIZE_SOURCE = 'archive'  # 逻辑链大小和深度从哪里读逻辑图：'archive' 各篇研报的存档（最新）；'graphs' data/graphs/
SIZE_GROUPS = 5  # 逻辑链大小和深度分几组
DEEP_CHARS = 6000  # 正文不少于这么多字算深度报告（写明“首次覆盖”的减半）
LABELS = {'theme': '主题项', 'rel': '关系项', 'total': '总分歧', 'pair_theme': '主题项', 'pair_rel': '关系项', 'pair_total': '总分歧'}  # 表头用的名字
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):  # 输出被重定向时 Windows 默认用本地编码，写中文会出错
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='分歧度与股票数据的分组统计')
    ap.add_argument('--task', default=TASK, choices=['all', 'divergence', 'sort', 'size', 'reg', 'text'], help='要运行的部分，默认全部')
    ap.add_argument('--input', default=PARAGRAPHS, help='正文段落 parquet（判断研报类型用）')
    ap.add_argument('--reports', default=REPORTS, help='研报元数据表，默认用和正文段落配套的那张')
    ap.add_argument('--size-var', default=','.join(SIZE_VARS), help='逻辑链大小和深度用哪些算法，逗号分隔（列名见 SIZE_VARS）')
    ap.add_argument('--size-source', default=SIZE_SOURCE, choices=['archive', 'graphs'], help='逻辑链大小和深度从哪里读逻辑图')
    ap.add_argument('--size-groups', type=int, default=SIZE_GROUPS, help='逻辑链大小和深度分几组')
    ap.add_argument('--output', default=OUT_DIR, help='输出目录（和抽取、框架对比用同一个）')
    ap.add_argument('--daily', default=DAILY, help='日行情 parquet')
    ap.add_argument('--financial', default=FINANCIAL, help='季度财务数据 parquet')
    ap.add_argument('--min-reports', type=int, default=MIN_REPORTS, help='一个公司-季度至少要有几篇研报')
    ap.add_argument('--row', default=ROW_VAR, help='分组的行指标')
    ap.add_argument('--col', default=COL_VAR, help='分组的列指标')
    ap.add_argument('--groups', type=int, default=N_GROUPS, help='双重分组时每个指标分几组')
    ap.add_argument('--single', default=','.join(SINGLE_VARS), help='单变量分组用哪些分歧指标，逗号分隔；空串表示不做')
    ap.add_argument('--single-groups', type=int, default=SINGLE_GROUPS, help='单变量分组分几组')
    ap.add_argument('--how', default=HOW, choices=['independent', 'dependent'], help='独立分组还是条件分组')
    ap.add_argument('--weight', default=WEIGHT, choices=['equal', 'value'], help='组合等权还是按季末市值加权')
    ap.add_argument('--nw-lags', type=int, default=NW_LAGS, help='Newey-West 滞后阶数，不给时按经验规则')
    ap.add_argument('--reg', default=','.join(REG_VARS), help='回归用哪些分歧指标，逗号分隔；空串表示不做')
    ap.add_argument('--car-window', default=','.join(map(str, CAR_WINDOW)), help='研报发布窗口，如 -3,1')
    args = ap.parse_args()
    if args.task == 'text':  # 只用原文边的逻辑图：图结构指标对股票收益和财务特征（src/text_graph.py）
        from src import text_graph
        text_graph.main(out_dir=args.output, paras_path=args.input, daily_path=args.daily, fin_path=args.financial,
                        reports_path=args.reports, deep_chars=DEEP_CHARS, car_window=tuple(int(v) for v in args.car_window.split(',')),
                        weight=args.weight, nw_lags=args.nw_lags)
        sys.exit()
    from src import divergence_sort
    divergence_sort.main(out_dir=args.output, daily_path=args.daily, fin_path=args.financial, fin_cols=FIN_COLS, task=args.task,
                         min_reports=args.min_reports, row_var=args.row, col_var=args.col, n_groups=args.groups, how=args.how,
                         weight=args.weight, nw_lags=args.nw_lags, labels=LABELS, paras_path=args.input, size_groups=args.size_groups, deep_chars=DEEP_CHARS,
                         size_vars=[v.strip() for v in args.size_var.split(',') if v.strip()], size_source=args.size_source,
                         reports_path=args.reports, single_vars=[v.strip() for v in args.single.split(',') if v.strip()],
                         single_groups=args.single_groups, reg_vars=[v.strip() for v in args.reg.split(',') if v.strip()],
                         car_window=tuple(int(v) for v in args.car_window.split(',')))
