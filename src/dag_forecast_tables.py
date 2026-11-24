"""预测准确度回归的 Markdown 表格，全部由回归长表生成、不手填（格式沿用 src/dag_outcomes_tables.py：系数 3 位有效数字、t 两位小数）。
  主回归      f01 样本筛选、f02 描述统计、f03–f10 每个被解释变量一张（write_tables）
  同行基准    f11–f12（peer_table）
  联合回归    j01–j04（joint_table）
  比值回归    每种同行基准 × 来源一张（ratio_table）"""
from src.dag_forecast import DAG_X, MIN_PEERS, PANELS, PEER_DAYS, SINKS, YS
from src.dag_measures import CF_RE, CFCORE_EXCLUDE, CFCORE_RE, VAL_EXCLUDE, VAL_RE
from src.dag_outcomes_tables import NAME as DAG_NAME, STAR_NOTE, f2, f3, md, sig3, stars, write  # noqa: F401（write 供调用方用）

NAME = {
    'AFE1': '预测误差·T+1', 'AFE2': '预测误差·T+2', 'AFE_FY24': '预测误差·FY2024', 'AFE1_P': '预测误差·T+1（市值标度）',
    'PMAFE1': '相对预测误差·T+1', 'LNHORIZON': '预测期限', 'LNHORIZON1': '预测期限·T+1', 'LNHORIZON2': '预测期限·T+2',
    'LNHORIZON24': '预测期限·FY2024', 'AFE_FY25': '预测误差·FY2025', 'LNHORIZON25': '预测期限·FY2025',
    'PMAFE_FY24': '相对预测误差·FY2024', 'PMAFE_FY25': '相对预测误差·FY2025', 'LNCHARS_R': '研报字数', 'LNNODES_R': '概念数', 'SIZE_M': '公司规模（市值）',
    'BM': '账面市值比', 'MOM': '动量', 'VOL': '波动率', 'ROA_T': 'T 年 ROA', 'LNAGE': '上市月数',
    'RATIO': DAG_NAME['RATIO'], 'LNPATH': DAG_NAME['LNPATH'], 'PATHLEN': DAG_NAME['PATHLEN'],
    'R_RATIO': '评级汇点的源点数', 'R_LNPATH': '评级汇点的路径数', 'R_PATHLEN': '评级汇点的平均路径长度',
}
X_SUFFIX = {'R_RATIO': '源点数', 'R_LNPATH': '路径数', 'R_PATHLEN': '平均路径长度'}
WINSOR_NOTE = '三个指标在有值的研报里按 1%、99% 缩尾。'
SET_PATHS = ('路径数（R_LNPATH）= ln(1 + 以 T 中各汇点为终点的路径条数之和)；'
             '平均路径长度（R_PATHLEN）= 这些路径合在一起的平均边数。')


def sink_def(sink):
    """汇点口径三个指标的定义（表注用）。"""
    s = SINKS[sink]
    pre, nodes = s.prefix, s.targets
    if nodes == 'FINAL':  # 最终指标清单汇点，R_RATIO 用源点数
        return (f'{pre}：图中入度大于 0、出度为 0 的节点（汇点，有出边的不算）中，名称在最终指标清单'
                '（output_dag/forecast/final_indicators/final_indicator_list.csv，status 为“入选”，tier A、B 都用）里的全部节点 T。'
                '子图 = T 中各汇点的祖先的并集加 T；源点数（R_RATIO）= 能到达 T 中任一汇点的源点个数（不重复计，不除以汇点个数）；'
                + SET_PATHS + '没有清单内汇点的研报记缺失、不进回归。' + WINSOR_NOTE)
    if s.by_class:  # 按类别取汇点：R_RATIO 用源点数（count）的口径和 CFCORE 用核心现金流类的归类规则，其余用现金流类
        kind = {'VAL': '估值类', 'CF': '现金流类', 'CFCORE': '核心现金流类'}[nodes]
        rule = (f'核心现金流类（匹配 {CFCORE_RE} 且不匹配 {CFCORE_EXCLUDE}），其余不用；' if s.count or nodes == 'CFCORE'
                else f'现金流类（匹配 {CF_RE}），都不匹配的不用；')
        ratio = ('源点数（R_RATIO）= 能到达 T 中任一汇点的源点个数（不重复计，不除以汇点个数）；' if s.count
                 else '源汇比（R_RATIO）= 能到达 T 中任一汇点的源点个数（不重复计）÷ T 的汇点个数；')
        return (f'{pre}：图中入度大于 0、出度为 0 的节点（汇点，有出边的不算）按名称归类，先判估值类（名称匹配正则 {VAL_RE}，'
                f'且不匹配 {VAL_EXCLUDE}），否则判{rule}取一篇研报里全部{kind}汇点 T。子图 = T 中各汇点的祖先的并集加 T；'
                + ratio + SET_PATHS + f'没有{kind}汇点的研报记缺失、不进回归。' + WINSOR_NOTE)
    if s.strict:
        n = nodes[0]
        return (f'{pre}的三个指标只看图中名称恰好为“{n}”的节点（不合并别名），且只有该节点是终点（入度大于 0、出度为 0）的研报才有指标；'
                f'“{n}”有出边的研报和没有该节点的研报一样记缺失、不进回归。源点数（R_RATIO）= 能到达“{n}”的源点（入度为 0）个数；'
                f'路径数（R_LNPATH）= ln(1 + 从源点到“{n}”的不同有向路径条数)；平均路径长度（R_PATHLEN）= 这些路径边数的均值。'
                f'三个指标在“{n}”是终点的研报里按 1%、99% 缩尾。')
    if len(nodes) == 1:
        n = nodes[0]
        return (f'{pre}的三个指标只看图中名称恰好为“{n}”的节点（不合并别名）和能到达它的全部节点（祖先）：'
                f'源点数（R_RATIO）= 能到达“{n}”的源点（入度为 0）个数，即只有这一个汇点时的源汇比；'
                f'路径数（R_LNPATH）= ln(1 + 从源点到“{n}”的不同有向路径条数)；平均路径长度（R_PATHLEN）= 这些路径边数的均值。'
                f'没有“{n}”节点或它没有入边的研报记缺失、不进回归；“{n}”还有出边的也只算以它为终点的路径。' + WINSOR_NOTE)
    a, b = nodes
    return (f'{pre}的三个指标看图中名称恰好为“{a}”和“{b}”的两个节点（不合并别名）及能到达其中任一节点的全部节点（祖先的并集）：'
            f'源点数（R_RATIO）= 能到达其中任一节点的源点（入度为 0）个数（不重复计）；'
            f'路径数（R_LNPATH）= ln(1 + 以“{a}”为终点的路径条数 + 以“{b}”为终点的路径条数)；'
            '平均路径长度（R_PATHLEN）= 这两组路径合在一起的平均边数。两个节点都没有路径的研报记缺失、不进回归；节点有出边的也只算以它为终点的路径。'
            + WINSOR_NOTE)


Y_DEF = {
    'AFE1': '|PROFIT_T1 − T+1 年实际归母净利润| ÷ STD_PAST',
    'AFE2': '|PROFIT_T2 − T+2 年实际归母净利润| ÷ STD_PAST',
    'AFE_FY24': '|对 FY2024 的预测 − FY2024 实际归母净利润| ÷ STD_PAST（T=2023 用 PROFIT_T1，T=2022 用 PROFIT_T2）',
    'AFE1_P': '|PROFIT_T1 − T+1 年实际归母净利润| ÷ 研报日前一交易日总市值 × 100',
    'PMAFE1': f'(本篇 T+1 绝对误差 − 同行绝对误差均值) ÷ 同行绝对误差均值；同行为其他券商在研报日前 {PEER_DAYS} 天内（含当天）'
              f'对同一公司 T+1 年的预测，每家取最近一条，不足 {MIN_PEERS} 家记缺失',
    'AFE_FY25': '|对 FY2025 的预测 − FY2025 实际归母净利润| ÷ STD_PAST（T=2023 用 PROFIT_T2，T=2022 用 PROFIT_T3）',
}
Y_DEF.update({f'PMAFE_FY{t}': f'(本篇对 FY20{t} 的绝对误差 − 同行绝对误差均值) ÷ 同行绝对误差均值；本篇预测同 AFE_FY{t}，绝对误差不除 STD_PAST；'
                                f'同行为其他券商在研报日前 {PEER_DAYS} 天内（含当天）对同一公司 FY20{t} 的预测（任一 k 年预测，RPT_YR_T + k = 20{t}），'
                                f'每家取最近一条，不足 {MIN_PEERS} 家记缺失' for t in ('24', '25')})
SINGLE_YEAR = {'AFE_FY24': 2024, 'AFE_FY25': 2025, 'PMAFE_FY24': 2024, 'PMAFE_FY25': 2025}  # 目标年度只有一个的 Y
FE_NAME = {'A': '申万一级行业 + 目标年度固定效应', 'B': '公司 + 目标年度固定效应', 'C': '公司×目标年度 + 券商固定效应',
           'N': '不加固定效应（只有截距）'}
_CTRL_FULL = '研报层面（预测期限、研报字数、概念数）和公司层面（市值规模、账面市值比、动量、波动率、T 年 ROA、上市月数）'
CTRL_NAME = {'N': _CTRL_FULL, 'A': _CTRL_FULL, 'B': '研报层面（预测期限、研报字数、概念数）和市值规模、动量、波动率',
             'C': '研报层面（预测期限、研报字数、概念数）'}


def lab(v, sink=None):
    """变量的“中文名（代码名）”；汇点口径下 R_ 指标写成“××节点的源点数（R_RATIO）”。"""
    if sink and v in X_SUFFIX:
        suffix = SINKS[sink].ratio_name if v == 'R_RATIO' else X_SUFFIX[v]
        return f'{SINKS[sink].prefix}的{suffix}（{v}）'
    return f'{NAME[v]}（{v}）' if v in NAME else v


def cell(r):
    """系数星号 (t)。"""
    return f'{sig3(r["coef"])}{stars(r["p"])} ({f2(r["t"])})' if r else ''


# ---------------------------------------------------------------- 主回归
def t_funnel(fn, sink=None):
    rows = [[r.step, r.n_reports, r.n_firms] for r in fn.itertuples()]
    pre = SINKS[sink].prefix if sink else ''
    note = ((f'注：{pre}口径下，“{SINKS[sink].step}”之后的各步（含各 Y 的非缺失数）都在有{pre}指标的研报内计。' if sink else '注：') +
            '“非缺失”在“研报日早于 T+1 年报公告日”这一步的样本内计。AFE1、AFE2、AFE_FY24、AFE_FY25 需要 STD_PAST；AFE1_P、PMAFE1、PMAFE_FY24、PMAFE_FY25 不需要，'
            '所以不受“STD_PAST 非缺失”这一步限制。回归样本为该 Panel 联合回归（三个 X 一起进入）的样本，已剔除固定效应 singleton。')
    return md(['步骤', '研报篇数', '公司数'], rows) + '\n\n' + note


def t_desc(desc, n_base, sink=None):
    rows = [[lab(r['var'], sink), int(r['n'])] + [sig3(r[k]) for k in ('mean', 'sd', 'min', 'p25', 'p50', 'p75', 'max')]
            for r in desc.to_dict('records')]
    note = (f'注：基础样本（匹配到本公司预测、有 T+1 实际值、研报日早于 T+1 年报公告日，{n_base} 篇）。被解释变量和控制变量已在基础样本内按 1%、99% '
            '缩尾；三个 DAG 指标用研报面板里已缩尾的值。预测期限 = ln(研报日到目标年度年报公告日的自然日数)；公司规模 = ln(研报日前一交易日总市值)；'
            '账面市值比 = T 年末净资产 ÷ 该市值；动量 = 前第 250 到前第 21 个交易日的累计收益；波动率 = 前 120 个交易日日收益的标准差；'
            'T 年 ROA = T 年归母净利润 ÷ T 年末总资产；上市月数 = ln(行情数据首个交易日到研报日的月数，行情从 2010 年开始，更早上市的被截断)。')
    if sink:
        note += sink_def(sink)
    return md(['变量', 'N', '均值', '标准差', '最小值', 'P25', '中位数', 'P75', '最大值'], rows) + '\n\n' + note


def by_y(res, y, xs=DAG_X, sink=None, pans=PANELS):
    """一个 Y 的表：各 Panel（pans，默认 A、B、C），每个 (1)–(3) 三个 X 单独进入、(4) 联合。"""
    idx = {(r['panel'], r['X_set'], r['X']): r for r in res[res.Y == y].to_dict('records')}
    cols = [[x] for x in xs] + [xs]
    rows = []
    for pan in pans:
        rows.append([f'**Panel {pan}：{FE_NAME[pan]}**'] + [''] * 4)
        for x in xs:
            vals = [idx.get((pan, '+'.join(c), x)) for c in cols]
            rows.append([lab(x, sink)] + [f'{sig3(r["coef"])}{stars(r["p"])}' if r else '' for r in vals])
            rows.append([''] + [f'({f2(r["t"])})' if r else '' for r in vals])
        first = [idx.get((pan, '+'.join(c), c[0])) for c in cols]
        rows.append(['控制变量'] + ['Y'] * 4)
        rows.append(['样本量'] + [r['n'] if r else '' for r in first])
        rows.append(['公司数'] + [r['n_clusters'] if r else '' for r in first])
        rows.append(['调整后 R²'] + [f'{r["adj_r2"]:.3f}' if r else '' for r in first])
    note = (f'注：被解释变量为{lab(y)} = {Y_DEF[y]}，越大表示预测越不准。每篇研报一个观测；(1)–(3) 为'
            + (f'{SINKS[sink].prefix}的{SINKS[sink].ratio_name}、路径数、平均路径长度' if sink else '源汇比、路径数、平均路径长度') + '各自单独进入，'
            '(4) 为三个指标同时进入；DAG 指标在各自回归样本内标准化，系数表示指标高 1 个标准差时被解释变量的变化。'
            + (sink_def(sink) + '控制变量里的研报字数、概念数仍是整篇研报（整张图）的值。' if sink else '')
            + '；'.join(f'Panel {p} 控制{CTRL_NAME[p]}' for p in pans) + '（控制变量系数不列，见 reg_results.csv）。'
            + ('Panel N 不加任何固定效应（只有截距）。' if 'N' in pans else '') +
            '预测期限 = ln(研报日到目标年度年报公告日的自然日数)' + ('，目标年度为 T+2' if y == 'AFE2' else f'，目标年度为 {SINGLE_YEAR[y]}' if y in SINGLE_YEAR else '，目标年度为 T+1') + '。'
            + (f'目标年度只有 {SINGLE_YEAR[y]} 一个，目标年度固定效应不起作用，公司×目标年度即公司。' if y in SINGLE_YEAR else '') +
            f'括号内为按公司聚类的稳健（CR1）t 值，星号按聚类 p 值（t(G−1) 分布）：{STAR_NOTE}。固定效应中只有一个观测的组先剔除。'
            '券商固定效应里样本不足 5 篇的券商并成“其他”。')
    return md(['变量', '(1)', '(2)', '(3)', '(4)'], rows) + '\n\n' + note


def write_tables(out, w, fn, desc, res, main_only=False, xs=DAG_X, sink=None, pans=PANELS):
    write(out / 'f01_funnel.md', t_funnel(fn, sink))
    write(out / 'f02_desc.md', t_desc(desc, int(w.base.sum()), sink))
    for i, y in enumerate(YS, start=3):
        txt = by_y(res, y, xs, sink, pans)
        if main_only:
            txt += '\n\n敏感性口径：一致预期（STD_PAST）和同行预测只用研报主体公司的预测行（SEC_CD = RPT_MAIN_CD）。'
        write(out / f'f{i:02d}_{y}.md', txt)
    print(f'表格：{len(list(out.glob("*.md")))} 个 -> {out}', flush=True)


# ---------------------------------------------------------------- 同行基准
VAR_LABEL = {'V0': 'V0 基准', 'V1': 'V1 窗口 30 天', 'V2': 'V2 窗口 60 天', 'V3': 'V3 窗口 180 天', 'V4': 'V4 同行误差取中位数',
             'V5': 'V5 至少 5 家', 'V6': 'V6 百分位排名', 'V7': 'V7 不含研报当天'}
VAR_DEF = {  # 每种同行基准一句话（比值回归的表注）
    'V0': '同行为研报日前 90 天内（含当天）其他券商对同一公司同一财年的预测、每家取最近一条，Y =（本篇绝对误差 − 同行绝对误差均值）÷ 同行绝对误差均值，至少 3 家同行（正文的 PMAFE）',
    'V1': '同 V0，但同行窗口改为研报日前 30 天（含当天）',
    'V2': '同 V0，但同行窗口改为研报日前 60 天（含当天）',
    'V3': '同 V0，但同行窗口改为研报日前 180 天（含当天）',
    'V4': '同 V0，但同行基准改为同行绝对误差的中位数',
    'V5': '同 V0，但至少 5 家同行',
    'V6': '同 V0 的同行（90 天、至少 3 家），Y =（同行绝对误差严格小于本篇的家数 + 0.5 × 相等的家数）÷ 同行家数，取值 0–1，越大越不准，不缩尾',
    'V7': '同 V0，但只取研报日之前发布的同行预测（不含研报当天）',
}
SOURCE_NAME = {'all': '全部预测行', 'main': '主体公司行'}
_MAIN_CTRL_NOTE = ('Panel A：预测期限、研报字数、概念数和市值规模、账面市值比、动量、波动率、T 年 ROA、上市月数；Panel B：预测期限、研报字数、'
                   '概念数和市值规模、动量、波动率；Panel C：预测期限、研报字数、概念数）')


def peer_table(res, yk, xs, sink):
    idx = {(r['variant'], r['source'], r['panel'], r['X']): r for r in res[res.Y == f'PMAFE_{yk}'].to_dict('records')}
    head = ['定义', '同行来源'] + [f'{p}：{x}' for p in PANELS for x in xs] + [f'N（{p}）' for p in PANELS]
    rows = []
    for v in VAR_LABEL:
        for src, sname in SOURCE_NAME.items():
            cells = [cell(idx.get((v, src, p, x))) for p in PANELS for x in xs]
            ns = [(idx.get((v, src, p, xs[0])) or {}).get('n', '') for p in PANELS]
            rows.append([VAR_LABEL[v], sname] + cells + ns)
    yr = '20' + yk[2:]
    note = (f'注：被解释变量为对 FY{yr} 预测的相对同行误差（PMAFE_{yk}），本篇预测同 AFE_{yk}（T=2023 用 '
            + ('PROFIT_T1，T=2022 用 PROFIT_T2' if yk == 'FY24' else 'PROFIT_T2，T=2022 用 PROFIT_T3') + '），绝对误差为元，不除 STD_PAST。'
            f'同行为其他券商（ORG_CL 不同）对同一公司 FY{yr} 的预测（同行那行的 RPT_YR_T + k = {yr}，k = 1、2、3），每家券商取窗口内最近一条。'
            'V0 基准：研报日前 90 天内（含当天），Y = (本篇绝对误差 − 同行绝对误差均值) ÷ 同行绝对误差均值，至少 3 家同行（即正文的 PMAFE 变量）；'
            'V1、V2、V3：窗口改为 30、60、180 天；V4：同行基准改为同行绝对误差的中位数；V5：至少 5 家同行；'
            'V6：Y = (同行绝对误差严格小于本篇的家数 + 0.5 × 与本篇相等的家数) ÷ 同行家数，取值 0–1，越大越不准（90 天、至少 3 家）；'
            'V7：只取研报日之前发布的同行预测（不含研报当天），其余同 V0。V0–V5、V7 在基础样本内按 1%、99% 缩尾，V6 不缩尾。'
            '同行来源：全部预测行 = 预测表里全部 IS_VLD = 1 的行；主体公司行 = 只用 SEC_CD = RPT_MAIN_CD 的行（研报主体公司的预测）。'
            '每格为 X 单独进入回归时的系数星号 (按公司聚类的 t)，X 在回归样本内标准化；X：'
            + '、'.join(lab(x, sink) for x in xs) + '。'
            + '；'.join(f'Panel {p} 为{FE_NAME[p]}' for p in PANELS) + f'（目标年度只有 {yr} 一个，年度固定效应不起作用）；'
            '控制变量同主回归（' + _MAIN_CTRL_NOTE + f'，预测期限 = ln(研报日到 FY{yr} 年报公告日的自然日数)。'
            f'N 为该 Panel 回归样本（第一个 X 的，三个 X 的样本相同）。星号按聚类 p 值（t(G−1) 分布）：{STAR_NOTE}。')
    return md(head, rows) + '\n\n' + note


# ---------------------------------------------------------------- 联合回归
def joint_table(res, y, src, n_obs, n_firms, cfg, pans=PANELS):
    q = res[(res.Y == y) & (res.source == src)]
    idx = {(r['panel'], r['kind'], r['reg'], r['X']): r for r in q.to_dict('records')}
    kinds, (xa, xb), (na, nb) = cfg['kinds'], cfg['xlab'], cfg['names']
    cols = [(k, reg) for k in kinds for reg in (1, 2, 3)]
    rows = [['指标类型'] + [kinds[k] for k, _ in cols]]
    for pan in pans:
        rows.append([f'**Panel {pan}：{FE_NAME[pan]}**'] + [''] * 9)
        for x, name in ((xa, na), (xb, nb)):
            vals = [idx.get((pan, k, reg, x)) for k, reg in cols]
            rows.append([name] + [f'{sig3(r["coef"])}{stars(r["p"])}' if r else '' for r in vals])
            rows.append([''] + [f'({f2(r["t"])})' if r else '' for r in vals])
        first = [idx.get((pan, k, reg, xa if reg != 2 else xb)) for k, reg in cols]
        rows.append(['两系数相等的检验 p 值'] + [f3(r['diff_p']) if r and reg == 3 else '' for r, (_, reg) in zip(first, cols)])
        rows.append(['控制变量'] + ['Y'] * 9)
        rows.append(['样本量'] + [r['n'] if r else '' for r in first])
        rows.append(['公司数'] + [r['n_clusters'] if r else '' for r in first])
        rows.append(['调整后 R²'] + [f'{r["adj_r2"]:.3f}' if r else '' for r in first])
    yr = '20' + y[-2:]
    note = (f'注：被解释变量为对 FY{yr} 预测的相对同行误差（{y}，基准定义：研报日前 90 天内含当天、其他券商每家最近一条、同行绝对误差取均值、'
            f'至少 3 家），同行来源为{SOURCE_NAME[src]}' + ('（预测表里全部 IS_VLD = 1 的行）' if src == 'all' else '（只用 SEC_CD = RPT_MAIN_CD 的行）') +
            f'。样本：基础样本里{cfg["sample"]}（两套指标都非缺失）的研报，{n_obs} 篇、{n_firms} 家公司；'
            '各 Panel 的样本量是剔除被解释变量或控制变量缺失、固定效应 singleton 之后的。'
            f'(1)–(3) 为{kinds["RATIO"]}，(4)–(6) 为{kinds["LNPATH"]}，(7)–(9) 为{kinds["PATHLEN"]}；每组中第一列只放{na}的该指标，第二列只放{nb}的该指标，'
            '第三列两个一起放，同组三列用同一样本。' + cfg['defs'] +
            f'两套指标用各自口径（{cfg["dirs"][0]}、{cfg["dirs"][1]}）已在有值的全部研报里按 1%、99% 缩尾的值，回归时在回归样本内标准化，系数表示指标高 1 个标准差时'
            '被解释变量的变化。两系数相等的检验：把两个标准化后的指标 a、b 换成 a + b 和 a − b 重新回归（其余不变），a − b 的系数等于两系数之差的一半，'
            '其按公司聚类的 t 检验即 H0：两系数相等，p 值用 t(G−1)。'
            + '；'.join(f'Panel {p} 为{FE_NAME[p]}' for p in pans) + f'（目标年度只有 {yr} 一个，年度固定效应不起作用）。控制变量同主回归'
            + ('（Panel N 同 Panel A；' if 'N' in pans else '（') + _MAIN_CTRL_NOTE +
            f'，预测期限 = ln(研报日到 FY{yr} 年报公告日的自然日数)。括号内为按公司聚类的稳健（CR1）t 值，'
            f'星号按聚类 p 值（t(G−1) 分布）：{STAR_NOTE}。')
    return md(['变量'] + [f'({i})' for i in range(1, 10)], rows) + '\n\n' + note


# ---------------------------------------------------------------- 比值回归
def ratio_table(res, v, src, form):
    from src.dag_forecast import panels
    from src.dag_forecast_robust import FORMS
    xs = FORMS[form]
    q = res[(res.variant == v) & (res.source == src) & (res.form == form)]
    idx = {(r['Y'], r['panel'], r['X']): r for r in q.to_dict('records')}
    pans = panels(with_nofe=True)
    rows = []
    for yk in ('FY24', 'FY25'):
        for pan in pans:
            rs = [idx.get((f'PMAFE_{yk}', pan, x)) for x in xs]
            ns = {(r['n'], r['n_clusters']) for r in rs if r}
            assert len(ns) <= 1, (v, src, yk, pan, ns)  # 三个 X 的回归样本相同
            n, g = ns.pop() if ns else ('', '')
            rows.append([f'FY20{yk[2:]} Panel {pan}'] + [cell(r) for r in rs] + [n, g])
    head = ['被解释变量 / Panel'] + [f'{name}（{x}）' for x, name in xs.items()] + ['样本量', '公司数']
    title = f'### 同行基准 {VAR_LABEL[v]}，同行来源：{SOURCE_NAME[src]}' + ('（对数口径）' if form == 'log' else '')
    xdef = ('源点数比值 = 评级节点源点数 ÷ 营收节点源点数；路径数比值 = 评级节点 ln(1 + 路径条数) ÷ 营收节点 ln(1 + 路径条数)；'
            '平均路径长度比值 = 评级节点平均路径长度 ÷ 营收节点平均路径长度' if form == 'ratio' else
            'ln 源点数比值 = ln(评级节点源点数 ÷ 营收节点源点数)；ln 路径条数比值 = ln(评级节点路径条数 ÷ 营收节点路径条数)；'
            'ln 平均路径长度比值 = ln(评级节点平均路径长度 ÷ 营收节点平均路径长度)')
    note = (f'同行基准定义：{VAR_DEF[v]}。同行来源：'
            + ('预测表里全部 IS_VLD = 1 的行。' if src == 'all' else '只用 SEC_CD = RPT_MAIN_CD 的行（研报主体公司的预测）。')
            + '\n\n注：被解释变量为对 FY2024（FY2025）预测的相对同行误差（PMAFE_FY24、PMAFE_FY25，在全部基础样本内计算和缩尾，越大越不准）。'
            '样本：基础样本里“投资评级”和“营业收入”两个节点都是终点（入度 > 0、出度 = 0）的研报。'
            f'{xdef}；用未缩尾的原始值构造，在本样本内按 1%、99% 缩尾，回归时在回归样本内标准化，三个指标各自单独进入回归。'
            '每格为标准化系数和星号，括号内为按公司聚类的稳健（CR1）t 值；'
            + '；'.join(f'Panel {p}：{FE_NAME[p]}，控制{CTRL_NAME[p]}' for p in pans) +
            '（目标年度只有一个，年度固定效应不起作用）。样本量、公司数为剔除缺失和固定效应 singleton 之后的回归样本（三个指标相同）。'
            f'星号按聚类 p 值（t(G−1) 分布）：{STAR_NOTE}。')
    return title + '\n\n' + md(head, rows) + '\n\n' + note
