"""回归样本：研报层面和公司-券商层面（一切依赖样本的处理都在这里：选研报、券商分组、配行业、缩尾、换单位）。

输入只有上游步骤的成品：src.graph.load（研报表、概念表）、src.measure.load（逻辑结构变量原始值）、src.accuracy.load
（预测准确度、同行时滞、控制变量原始值），以及 data/ 下的覆盖名单和行业表。

研究范围（scope，791 篇）
  公司在 data/hs300_2024_dag_covered.csv 里、且 graph 研报表 n_edges > 0（至少有一条进图的边）的研报；行序同
  src.graph.load.read_table('reports')（deep 在前、pilot 在后，运行内按文件行序）。回归的行序就是这个顺序（求和顺序影响
  浮点数的末位，所以保持不变）。
  broker_g  在这 791 篇里按 graph 研报表的 broker 计数，不少于 MIN_BROKER 篇的用券商名，否则记“其他”。
  sw        申万一级行业：data/market_data/industry.xlsx（Wind 导出），缺的两只按 MANUAL_INDUSTRY 补。
  month     发布月份 YYYY-MM。

研报层面（report_sample，786 篇 = 791 篇里 accuracy 的基础样本 base）
  缩尾一律在这 786 篇内按 1%/99% 分位做（src.ols.winsor，只用非缺失值求分位数），各变量各自缩尾：
    Y        PMAFE_FY25、PMAFE_FY24、PMAFEW30_FY25（窗口延伸 30 天的比较组）；SCORE_FY25 不缩尾
    时滞     PEERLAG25、PEERLAG24、PEERLAGW30、LAGMED25、LNNPEERS25；N_PEERS* 不缩尾
    控制     SIZE_M、BM、MOM、VOL、ROA_T、LNAGE：先缩尾、再把 MOM、VOL、ROA_T 乘 100（换成 %）
    X        六个逻辑结构变量和 RT_LNSRC = ln(RT_NSRC)、WG_LNSRC = ln(WG_NSRC)
  列 <变量>_RAW 是缩尾前的值（控制变量已换成文档单位：MOM、VOL、ROA_T 为 %），<变量> 是回归用的缩尾后的值。
  X 在回归时再在各自的估计样本内标准化（src.ols.fit），时滞和控制变量不标准化。

公司-券商层面（pair_sample，760 对）
  样本对    791 篇里有 ORG_CL（匹配到本篇预测行）的研报构成的（公司，券商）对；被评价的预测、比较组、PMAFE、DAGE、控制变量
            取自 accuracy 的 pair_accuracy（控制变量在被评价预测的发布日上度量）。
  X         每对取这些研报（不限 base）里最新的一篇：按 code、ORG_CL、date、report_id 排序取最后一篇的逻辑结构变量原始值；
            X 在 760 对内各自按 1%/99% 缩尾。
  broker_g  每个 ORG_CL 在这些研报里最常见的 broker_g（value_counts 的第一个）；sw 取每家公司的第一个。
  缩尾      PMAFE_FY25 和六个控制变量：只在“786 篇基础样本研报涉及的对”内求 1%/99% 分位，再用同一界限截断全部 760 对；
            控制变量先把 MOM、VOL、ROA_T 乘 100、再缩尾。DAGE25 不缩尾。
  两个层面缩尾和乘 100 的先后顺序不同，是沿用原有做法（乘 100 和缩尾的先后只影响浮点数末位，不影响结论）。
  回归用的辅助列：report_id = 'code|ORG_CL'，base = True，TY25 = 2025，LNHORIZON25 记缺失（不进回归）。
"""
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from src.config import DATA_DIR
from src.ols import winsor

COVERED = DATA_DIR / 'hs300_2024_dag_covered.csv'  # DAG 覆盖的沪深300 成分股（241 家）
INDUSTRY = DATA_DIR / 'market_data' / 'industry.xlsx'  # 申万一级行业（2021）、证监会门类
MANUAL_INDUSTRY = {'600837': ('金融业', '非银金融'), '601989': ('制造业', '国防军工')}  # 行业表里没有的两只（2025 年退市）：（证监会门类，申万一级）
MIN_BROKER = 5  # 研究范围内不足这么多篇的券商并成“其他”

CTRL = ['SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']
PCT = ['MOM', 'VOL', 'ROA_T']  # 小数 -> %
X6 = ['RT_NSRC', 'RT_LNPATH', 'RT_PATHLEN', 'WG_NSRC', 'WG_LNPATH', 'WG_PATHLEN']
XLOG = ['RT_LNSRC', 'WG_LNSRC']  # ln(源点数)，稳健性检验用
Y_W = ['PMAFE_FY25', 'PMAFE_FY24', 'PMAFEW30_FY25']  # 研报层面缩尾的 Y
LAG_W = ['PEERLAG25', 'PEERLAG24', 'PEERLAGW30', 'LAGMED25', 'LNNPEERS25']  # 研报层面缩尾的时滞变量


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
def scope():
    """研究范围的 791 篇研报（行序同 graph 研报表），合并 accuracy 的全部列和 measure 的六个变量原始值（列名同上游，未缩尾）；
    另加 broker_g、sw、month。"""
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
    meas = read_measures(X6 + ['HAS_RATING_NODE', 'RATING_STRICT_SINK'])
    S = (r[['report_id', 'code', 'date', 'month', 'broker', 'broker_g', 'sw', 'n_edges']]
         .merge(acc, on='report_id', how='left', validate='1:1').merge(meas, on='report_id', how='left', validate='1:1'))
    assert S.base.notna().all() and S.WG_NSRC.notna().all(), '研究范围内有研报不在 accuracy / measure 的成品里'
    S['base'] = S.base.astype(bool)
    return S


# ---------------------------------------------------------------- 研报层面
def report_sample(S):
    """研报层面的回归样本（786 篇基础样本，行序同 S；口径见模块说明）。"""
    D = S[S.base].reset_index(drop=True)
    for x in XLOG:
        D[x] = np.log(D[x.replace('LNSRC', 'NSRC')])
    for c in Y_W + LAG_W + X6 + XLOG:
        D[c + '_RAW'] = D[c]
        D[c] = winsor(D[c + '_RAW'])
    for c in CTRL:  # 先缩尾，再换成 %
        scale = 100 if c in PCT else 1
        raw = D[c]
        D[c + '_RAW'] = raw * scale
        D[c] = winsor(raw) * scale
    cols = (['report_id', 'code', 'ORG_CL', 'broker', 'broker_g', 'sw', 'date', 'month', 'base', 'TY24', 'TY25', 'LNHORIZON24', 'LNHORIZON25',
             'A24', 'A25', 'SCORE_FY25', 'N_PEERS25', 'N_PEERS24', 'N_PEERSW30']
            + [c for v in Y_W + LAG_W + CTRL + X6 + XLOG for c in (v + '_RAW', v)])
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


def pair_sample(S):
    """公司-券商层面的回归样本（760 对，按 code、ORG_CL 排序；口径见模块说明）。"""
    from src.accuracy.load import pair_accuracy

    s = S[S.ORG_CL.notna()]
    bg = s.groupby('ORG_CL').broker_g.agg(lambda x: x.value_counts().index[0])
    sw = S.groupby('code').sw.first()
    P = s.groupby(['code', 'ORG_CL']).size().rename('_n').reset_index().drop(columns='_n')
    last = s.sort_values(['code', 'ORG_CL', 'date', 'report_id']).groupby(['code', 'ORG_CL']).tail(1)
    last = last[['code', 'ORG_CL', 'report_id', 'date'] + X6].rename(columns={'report_id': 'X_report_id', 'date': 'X_report_dt',
                                                                              **{x: x + '_RAW' for x in X6}})
    pa = pair_accuracy()
    P = (P.merge(pa, on=['code', 'ORG_CL'], how='left', validate='1:1')
         .merge(last, on=['code', 'ORG_CL'], how='left', validate='1:1'))
    U = s[s.base][['code', 'ORG_CL']].drop_duplicates().assign(_u=True)
    in_u = P[['code', 'ORG_CL']].merge(U, on=['code', 'ORG_CL'], how='left')._u.fillna(False).astype(bool).to_numpy()
    for c in PCT:  # 先换成 %，再缩尾
        P[c] = P[c] * 100
    winsor_by(P, ['PMAFE_FY25'] + CTRL, in_u)
    for x in X6:
        P[x] = winsor(P[x + '_RAW'])
    P['broker_g'] = P.ORG_CL.map(bg)
    P['sw'] = P.code.map(sw)
    P['report_id'] = [f'{c}|{o}' for c, o in zip(P.code, P.ORG_CL)]
    P['base'] = True
    P['TY25'] = 2025
    P['LNHORIZON25'] = np.nan
    P['IN_WINSOR_UNIVERSE'] = in_u
    cols = (['report_id', 'code', 'ORG_CL', 'broker_g', 'sw', 'base', 'TY25', 'LNHORIZON25', 'n_reports', 'last_deep_dt', 'X_report_id',
             'X_report_dt', 'IN_WINSOR_UNIVERSE', 'EVAL_RPT_ID25', 'EVAL_DT25', 'EVAL_F25', 'EVAL_T25', 'A25', 'AFE25', 'N_GROUP25', 'AGE25',
             'MAFE25', 'DAGE25', 'EVAL_IS_DEEP', 'EVAL_GAP']
            + [c for v in ['PMAFE_FY25'] + CTRL + X6 for c in (v + '_RAW', v)])
    return P[cols]


def build():
    """(研究范围 791 篇, 研报层面样本, 公司-券商层面样本)。"""
    S = scope()
    return S, report_sample(S), pair_sample(S)
