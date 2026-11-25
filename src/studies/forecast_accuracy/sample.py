"""回归样本：研报层面和公司-券商层面（一切依赖样本的处理都在这里：选研报、券商分组、配行业、缩尾、换单位）。

输入：上游步骤的成品 src.graph.load（研报表、概念表）、src.measure.load（逻辑结构变量原始值）、src.accuracy.load
（自算的预测准确度、同行时滞、控制变量原始值），data/ 下的覆盖名单和行业表，以及外部的预测误差文件 data/forecast_errors.parquet
（被解释变量，读取和整理见 errors.py）。

研究范围（scope，791 篇）
  公司在 data/hs300_2024_dag_covered.csv 里、且 graph 研报表 n_edges > 0（至少有一条进图的边）的研报；行序同
  src.graph.load.read_table('reports')（deep 在前、pilot 在后，运行内按文件行序）。回归的行序就是这个顺序（求和顺序影响
  浮点数的末位，所以保持不变）。
  broker_g  在这 791 篇里按 graph 研报表的 broker 计数，不少于 MIN_BROKER 篇的用券商名，否则记“其他”。
  sw        申万一级行业：data/market_data/industry.xlsx（Wind 导出），缺的两只按 MANUAL_INDUSTRY 补。
  month     发布月份 YYYY-MM。

预测窗口和主窗口
  变量按预测窗口 H1、H2、H3 给出（列名后缀 _Hh：第 h 年预测，目标年度 TY_Hh = RPT_YR_T + h，逐篇不同；T = 2022 的研报
  H2 = FY2024，T = 2023 的 H2 = FY2025；数据文件的 horizon、target_year 与此逐篇相同）。主窗口 MAIN_H（= 2）只在这里定义一次：
  被解释变量 Y_MAIN（PMED180_NP_H2）、自算的对照 OWN_MAIN（PMAFE_H2）、SCORE_MAIN、同行时滞 LAG_MAIN 等变量名都由它
  派生（hv）；其余两个窗口 OTHER_H 用于“其他预测窗口”。改成 1 或 3 只动 MAIN_H 一处即可重跑。

被解释变量
  主变量的定义名只在这里定义一次：MAIN_DEF（现为 'PMED180'，errors.DEFS 的键）。Y_MAIN = ymain(MAIN_H) = PMED180_NP_H2 =
  (|A − F| − M180) / M180（净利润、第 2 个预测年度；用数据文件的 180 天同行绝对误差中位数派生，文件没有直接给出）；其他窗口的
  同名变量 ymain(h)、公司-券商层面的 Y、peerlag_npeers 块控制的同行条数对数 LNN_MAIN（窗口取 MAIN_DEF 的窗口，现为
  LNN180_NP_H2）都由它派生，换主变量只改 MAIN_DEF。其余定义、口径、窗口的 54 列（errors.ALL_Y）都并入样本，用于稳健性。
  自算的 PMAFE_Hh、SCORE_Hh 留在样本里作对照。

研报层面（report_sample，786 篇 = 791 篇里 accuracy 的基础样本 base）
  缩尾一律在这 786 篇内按 1%/99% 分位做（src.ols.winsor，只用非缺失值求分位数），各变量各自缩尾：
    Y        数据文件的 54 列 {定义}_{口径}_Hh（errors.ALL_Y，缩尾前为 errors.wide 的值）；自算的三个窗口 PMAFE_Hh、
             PMAFEW30_Hh（窗口延伸 30 天的比较组）；SCORE_Hh 不缩尾
    时滞     三个窗口的 PEERLAG_Hh、PEERLAGW30_Hh、LAGMED_Hh、LNNPEERS_Hh（自算的 90 天比较组），以及数据文件的同行条数
             对数 LNN75_{口径}_Hh、LNN180_{口径}_Hh；N_PEERS*、N75_*、N180_* 不缩尾
             （各窗口的变量在 786 篇里各自缩尾，只用该列的非缺失值：例如 T = 2023 研报的 PEERLAG_H3 有值、PMAFE_H3 缺失，
             PEERLAG_H3 的分位数也用到这些研报）
    控制     SIZE_M、BM、MOM、VOL、ROA_T、LNAGE：先缩尾、再把 MOM、VOL、ROA_T 乘 100（换成 %）
    X        六个逻辑结构变量和 RT_LNSRC = ln(RT_NSRC)、WG_LNSRC = ln(WG_NSRC)
    路径数   RT_N_PATHS、WG_N_PATHS（取对数之前的条数，只用于表 1 的描述性统计，不进回归）
  列 <变量>_RAW 是缩尾前的值（控制变量已换成文档单位：MOM、VOL、ROA_T 为 %），<变量> 是回归用的缩尾后的值。
  X 在回归时再在各自的估计样本内标准化（src.ols.fit），时滞和控制变量不标准化。

公司-券商层面（pair_sample，760 对）
  样本对    791 篇里有 ORG_CL（匹配到本篇预测行）的研报构成的（公司，券商）对；被评价的预测、自算的比较组和 PMAFE、DAGE、
            控制变量取自 accuracy 的 pair_accuracy 的主窗口列（*_H{MAIN_H}；被评价预测 = 券商 2024 年内最后一条第 MAIN_H 年
            预测，目标年度 TY_H{MAIN_H} 逐对不同；控制变量在该预测的发布日上度量），EVAL_IS_DEEP、EVAL_GAP 和六个控制变量去掉后缀。
  Y         Y_MAIN（PMED180_NP_H2）= 数据文件里被评价预测所在研报（EVAL_RPT_ID 大写、sec == code、PROFIT、target_year == TY）
            那一行算出的 MAIN_DEF（errors.pair_y；EVAL_FILE_FOUND、EVAL_FILE_HORIZON 记录是否找到、那一行的 horizon）；
            自算的 PMAFE_H{MAIN_H} 留作对照。
  X         每对取这些研报（不限 base）里最新的一篇：按 code、ORG_CL、date、report_id 排序取最后一篇的逻辑结构变量原始值；
            X 在 760 对内各自按 1%/99% 缩尾。
  broker_g  每个 ORG_CL 在这些研报里最常见的 broker_g（value_counts 的第一个）；sw 取每家公司的第一个。
  缩尾      Y_MAIN、自算的 PMAFE_H{MAIN_H} 和六个控制变量：只在“786 篇基础样本研报涉及的对”内求 1%/99% 分位，再用同一界限截断全部
            760 对；控制变量先把 MOM、VOL、ROA_T 乘 100、再缩尾。DAGE 不缩尾。
  两个层面缩尾和乘 100 的先后顺序不同，是沿用原有做法（乘 100 和缩尾的先后只影响浮点数末位，不影响结论）。
  回归用的辅助列：report_id = 'code|ORG_CL'，base = True，目标年度 TY_H{MAIN_H} 取自 pair_accuracy（年度固定效应用），
  LNHORIZON_H{MAIN_H} 记缺失（不进回归）。

对数变量开关（build(log1p=True)，命令行 --log1p）
  在 scope 里把 LOG1P_X 四列（源点数、平均路径长度）换成 ln(1 + 原始值)，之后照常缩尾、标准化，两个层面的样本都跟着变；
  两个 *_LNPATH 本来就是 ln(1 + 路径数)，不再变换；两个 *_N_PATHS（路径条数）也不变换。打开时 RT_LNSRC、WG_LNSRC（log_sources 块、text_numbers 的 4.5）是对
  已取对数的值再取对数，没有意义，不要使用。
"""
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from src.accuracy.forecasts import HORIZONS
from src.config import DATA_DIR
from src.ols import winsor
from src.studies.forecast_accuracy import errors

COVERED = DATA_DIR / 'hs300_2024_dag_covered.csv'  # DAG 覆盖的沪深300 成分股（241 家）
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'  # 申万一级行业（2021）、证监会门类
MANUAL_INDUSTRY = {'600837': ('金融业', '非银金融'), '601989': ('制造业', '国防军工')}  # 行业表里没有的两只（2025 年退市）：（证监会门类，申万一级）
MIN_BROKER = 5  # 研究范围内不足这么多篇的券商并成“其他”

CTRL = ['SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']
PCT = ['MOM', 'VOL', 'ROA_T']  # 小数 -> %
X6 = ['RT_NSRC', 'RT_LNPATH', 'RT_PATHLEN', 'WG_NSRC', 'WG_LNPATH', 'WG_PATHLEN']
XLOG = ['RT_LNSRC', 'WG_LNSRC']  # ln(源点数)，稳健性检验用
NPATHS = ['RT_N_PATHS', 'WG_N_PATHS']  # 路径条数（取对数之前），只用于描述性统计
LOG1P_X = ['RT_NSRC', 'RT_PATHLEN', 'WG_NSRC', 'WG_PATHLEN']  # --log1p 时换成 ln(1 + 原始值) 的列（*_LNPATH 已是 ln(1 + 路径数)）

MAIN_H = 2  # 主窗口（第 h 年预测）：研究层唯一决定主窗口的常量，下面的变量名都由它派生
OTHER_H = [h for h in HORIZONS if h != MAIN_H]  # 表 6 的其他窗口（Panel A、B 依次）


def hv(var, h=MAIN_H):
    """窗口 h 的变量名：hv('PMAFE') -> 'PMAFE_H2'（默认主窗口）。"""
    return f'{var}_H{h}'


MAIN_DEF = 'PMED180'  # 主被解释变量的定义名（errors.DEFS 的键）：研究层唯一决定主变量的常量
MAIN_WIN = re.search(r'(75|180)$', MAIN_DEF).group(1) if re.search(r'(75|180)$', MAIN_DEF) else '75'  # 主变量的同行窗口（RAWFE 没有窗口，取 75）


def ymain(h=MAIN_H):
    """窗口 h 的主变量（净利润口径）：ymain(2) -> 'PMED180_NP_H2'。"""
    return errors.yname(MAIN_DEF, 'NP', h)


Y_MAIN = ymain()  # 主被解释变量
OWN_MAIN, SCORE_MAIN = hv('PMAFE'), hv('SCORE')  # 自算的 PMAFE、Score（对照）
LAG_MAIN, LAGMED_MAIN, LNNPEERS_MAIN, NPEERS_MAIN = hv('PEERLAG'), hv('LAGMED'), hv('LNNPEERS'), hv('N_PEERS')
LNN_MAIN, N_MAIN = hv(f'LNN{MAIN_WIN}_NP'), hv(f'N{MAIN_WIN}_NP')  # 数据文件主变量窗口的同行条数（净利润）：ln(n)、n
DAGE_MAIN = hv('DAGE')  # 公司-券商层面的时效差
Y_W = errors.ALL_Y + [hv(v, h) for h in HORIZONS for v in ('PMAFE', 'PMAFEW30')]  # 研报层面缩尾的 Y
LAG_W = [hv(v, h) for h in HORIZONS for v in ('PEERLAG', 'PEERLAGW30', 'LAGMED', 'LNNPEERS')] + errors.LNN  # 研报层面缩尾的时滞类变量
PAIR_KEEP = ['EVAL_RPT_ID', 'EVAL_DT', 'EVAL_F', 'EVAL_T', 'TY', 'A', 'AFE', 'N_GROUP', 'AGE', 'MAFE', 'DAGE', 'EVAL_LAST_TY']  # 保留后缀
PAIR_STRIP = ['EVAL_IS_DEEP', 'EVAL_GAP'] + CTRL  # 主窗口的这些列去掉后缀


# ---------------------------------------------------------------- 行业
def read_xlsx_first_sheet(path):
    """不依赖 openpyxl 读 xlsx 的第一张表：返回 [{列字母: 文本}]，一行一个字典。"""
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    tag = lambda t: '{%s}%s' % (ns['m'], t)  # noqa: E731
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns):
            shared.append(''.join(t.text or '' for t in si.iter(tag('t'))))
    sheets = sorted((n for n in z.namelist() if re.match(r'xl/worksheets/sheet\d+\.xml$', n)),
                    key=lambda n: int(re.search(r'(\d+)\.xml$', n).group(1)))
    rows = []
    for row in ET.fromstring(z.read(sheets[0])).iter(tag('row')):
        r = {}
        for c in row.findall('m:c', ns):
            col = re.match(r'[A-Z]+', c.get('r')).group(0)
            v, t = c.find('m:v', ns), c.get('t')
            if t == 's' and v is not None:
                r[col] = shared[int(v.text)]
            elif t == 'inlineStr':
                r[col] = ''.join(x.text or '' for x in c.iter(tag('t')))
            else:
                r[col] = v.text if v is not None else None
        rows.append(r)
    return rows


def load_industry(path=INDUSTRY):
    """industry.xlsx（证券代码、证券简称、申万一级行业（2021）、证监会门类行业）-> code、name、sw、csrc。
    只留证券代码形如 000001.SZ 的行（去掉表头和最后的“数据来源”），空的行业记缺失；补上 MANUAL_INDUSTRY。"""
    out = []
    for r in read_xlsx_first_sheet(path):
        code = (r.get('A') or '').strip()
        if re.fullmatch(r'\d{6}\.[A-Z]{2}', code):
            clean = lambda v: (v or '').strip() or None  # noqa: E731
            out.append({'code': code[:6], 'name': clean(r.get('B')), 'sw': clean(r.get('C')), 'csrc': clean(r.get('D'))})
    ind = pd.DataFrame(out).drop_duplicates('code')
    add = [{'code': c, 'name': None, 'csrc': v[0], 'sw': v[1]} for c, v in MANUAL_INDUSTRY.items() if c not in set(ind.code)]
    return pd.concat([ind, pd.DataFrame(add)], ignore_index=True)


# ---------------------------------------------------------------- 研究范围
def scope(log1p=False):
    """研究范围的 791 篇研报（行序同 graph 研报表），合并 accuracy 的全部列和 measure 的六个变量、两个路径条数的原始值
    （列名同上游，未缩尾；路径条数转成 float）；另加 broker_g、sw、month。log1p=True 时 LOG1P_X 四列换成 ln(1 + 原始值)。"""
    from src.accuracy.load import report_accuracy
    from src.graph.load import read_table
    from src.measure.load import read_measures

    r = read_table('reports', columns=['stock_code', 'date', 'broker', 'n_edges'])
    covered = set(pd.read_csv(COVERED, dtype=str).stock_code.str[:6])
    r = r.assign(code=r.stock_code.str[:6])
    r = r[r.code.isin(covered) & (r.n_edges > 0)].reset_index(drop=True)
    bc = r.broker.value_counts()
    r['broker_g'] = np.where(r.broker.map(bc) >= MIN_BROKER, r.broker, '其他')
    r['sw'] = r.code.map(load_industry().set_index('code').sw)
    r['month'] = r.date.str[:4] + '-' + r.date.str[4:6]
    acc = report_accuracy().drop(columns=['code', 'date', 'broker'])
    meas = read_measures(X6 + NPATHS + ['HAS_RATING_NODE', 'RATING_STRICT_SINK'])
    meas[NPATHS] = meas[NPATHS].astype(float)
    if log1p:
        meas[LOG1P_X] = np.log1p(meas[LOG1P_X])
    S = (r[['report_id', 'code', 'date', 'month', 'broker', 'broker_g', 'sw', 'n_edges']]
         .merge(acc, on='report_id', how='left', validate='1:1').merge(meas, on='report_id', how='left', validate='1:1'))
    assert S.base.notna().all() and S.WG_NSRC.notna().all(), '研究范围内有研报不在 accuracy / measure 的成品里'
    S['base'] = S.base.astype(bool)
    return S


# ---------------------------------------------------------------- 研报层面
def report_sample(S, rows):
    """研报层面的回归样本（786 篇基础样本，行序同 S；口径见模块说明）。rows = errors.read_rows 读出的数据文件行。"""
    D = S[S.base].reset_index(drop=True)
    W = errors.wide(rows, D[['report_id', 'code']])
    assert (W.report_id.to_numpy() == D.report_id.to_numpy()).all()
    D = pd.concat([D, W.drop(columns='report_id')], axis=1)
    for x in XLOG:
        D[x] = np.log(D[x.replace('LNSRC', 'NSRC')])
    new = {}  # 先收集再一次并入（逐列插入几百列会让 DataFrame 碎片化）
    for c in Y_W + LAG_W + X6 + XLOG + NPATHS:
        new[c + '_RAW'] = D[c]
        new[c] = winsor(D[c])
    for c in CTRL:  # 先缩尾，再换成 %
        scale = 100 if c in PCT else 1
        raw = D[c]
        new[c + '_RAW'] = raw * scale
        new[c] = winsor(raw) * scale
    D = pd.concat([D.drop(columns=[c for c in new if c in D.columns]), pd.DataFrame(new)], axis=1)
    cols = (['report_id', 'code', 'ORG_CL', 'broker', 'broker_g', 'sw', 'date', 'month', 'base', 'RPT_YR_T']
            + [hv(v, h) for h in HORIZONS for v in ('TY', 'LNHORIZON', 'F', 'A', 'SCORE', 'N_PEERS', 'N_PEERSW30')] + errors.NCOLS
            + [c for v in Y_W + LAG_W + CTRL + X6 + XLOG + NPATHS for c in (v + '_RAW', v)])
    return D[cols]


# ---------------------------------------------------------------- 公司-券商层面
def winsor_by(df, cols, universe):
    """在 universe（布尔）内求 1%/99% 分位，截断全部行（缩尾前的值另存 <列>_RAW）；返回界限。"""
    bounds = {}
    for c in cols:
        lo, hi = df.loc[universe, c].quantile([.01, .99])
        bounds[c] = (lo, hi)
        df[c + '_RAW'] = df[c]
        df[c] = df[c].clip(lo, hi)
    return bounds


def pair_sample(S, rows):
    """公司-券商层面的回归样本（760 对，按 code、ORG_CL 排序；口径见模块说明）。rows = errors.read_rows 读出的数据文件行。"""
    from src.accuracy.load import pair_accuracy

    s = S[S.ORG_CL.notna()]
    bg = s.groupby('ORG_CL').broker_g.agg(lambda x: x.value_counts().index[0])
    sw = S.groupby('code').sw.first()
    P = s.groupby(['code', 'ORG_CL']).size().rename('_n').reset_index().drop(columns='_n')
    last = s.sort_values(['code', 'ORG_CL', 'date', 'report_id']).groupby(['code', 'ORG_CL']).tail(1)
    last = last[['code', 'ORG_CL', 'report_id', 'date'] + X6].rename(columns={'report_id': 'X_report_id', 'date': 'X_report_dt',
                                                                              **{x: x + '_RAW' for x in X6}})
    pa = pair_accuracy()
    pa = pa[['code', 'ORG_CL', 'n_reports', 'last_deep_dt', hv('PMAFE')] + [hv(v) for v in PAIR_KEEP + PAIR_STRIP]]
    pa = pa.rename(columns={hv(v): v for v in PAIR_STRIP})
    P = (P.merge(pa, on=['code', 'ORG_CL'], how='left', validate='1:1')
         .merge(last, on=['code', 'ORG_CL'], how='left', validate='1:1'))
    P = pd.concat([P, errors.pair_y(rows, P, MAIN_H, MAIN_DEF)], axis=1)
    U = s[s.base][['code', 'ORG_CL']].drop_duplicates().assign(_u=True)
    in_u = P[['code', 'ORG_CL']].merge(U, on=['code', 'ORG_CL'], how='left')._u.fillna(False).astype(bool).to_numpy()
    for c in PCT:  # 先换成 %，再缩尾
        P[c] = P[c] * 100
    winsor_by(P, [Y_MAIN, OWN_MAIN] + CTRL, in_u)
    for x in X6:
        P[x] = winsor(P[x + '_RAW'])
    P['broker_g'] = P.ORG_CL.map(bg)
    P['sw'] = P.code.map(sw)
    P['report_id'] = [f'{c}|{o}' for c, o in zip(P.code, P.ORG_CL)]
    P['base'] = True
    P[hv('LNHORIZON')] = np.nan
    P['IN_WINSOR_UNIVERSE'] = in_u
    cols = (['report_id', 'code', 'ORG_CL', 'broker_g', 'sw', 'base', hv('LNHORIZON'), 'n_reports', 'last_deep_dt', 'X_report_id',
             'X_report_dt', 'IN_WINSOR_UNIVERSE'] + [hv(v) for v in PAIR_KEEP] + ['EVAL_IS_DEEP', 'EVAL_GAP', 'EVAL_FILE_FOUND',
                                                                                  'EVAL_FILE_HORIZON']
            + [c for v in [Y_MAIN, OWN_MAIN] + CTRL + X6 for c in (v + '_RAW', v)])
    return P[cols]


def build(log1p=False):
    """(研究范围 791 篇, 研报层面样本, 公司-券商层面样本, 数据文件里用到的行)。log1p 见模块说明。
    数据文件的行 = report_id 属于研究范围或公司-券商层面被评价预测（EVAL_RPT_ID_H{MAIN_H}）的全部行（不管 sec）。"""
    from src.accuracy.load import pair_accuracy
    S = scope(log1p)
    rows = errors.read_rows(set(S.report_id) | set(pair_accuracy(columns=[hv('EVAL_RPT_ID')])[hv('EVAL_RPT_ID')].dropna()))
    return S, report_sample(S, rows), pair_sample(S, rows), rows
