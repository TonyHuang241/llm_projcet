"""回归设定和估计：文档用到的回归块，全部结果写成一张长表（reg_all.csv，每个系数一行）。

估计（src.ols.fit）
  每个（层面，块，Y，固定效应，X）一次 OLS：样本 = 回归样本中 Y、X、控制变量、固定效应、聚类变量都不缺失的观测，再反复剔除
  固定效应里的单例（只有一个观测的组）；X 在这个估计样本内标准化为均值 0、标准差 1，控制变量不标准化；固定效应用虚拟变量；
  标准误按公司（code）聚类，CR1：V = (X'X)⁻¹ Σ_g X_g'e_g e_g'X_g (X'X)⁻¹ · G/(G−1) · (N−1)/(N−K)，K 为设计矩阵的秩（常数项和
  全部固定效应虚拟变量都计入，包括嵌套在公司聚类内的公司固定效应）；p 值用 t(G−1)。

固定效应（FE，列名见 FE_COLS；样本只有一个目标年度，ty 是常数）
  none 无；industry 申万一级行业（sw、ty）；firm 公司（code、ty）；firm+broker 公司 + 券商（code_ty、broker_g）；
  firm+month、firm+broker+month 在前两者上再加发布月份（month）。

块（研报层面 report；控制变量 = 六个控制变量 + 同行时滞，PMAFE_FY25、SCORE_FY25 配 PEERLAG25，PMAFE_FY24 配 PEERLAG24）
  fe_steps            Y ∈ {PMAFE_FY25, PMAFE_FY24, SCORE_FY25} × FE ∈ {none, industry, firm, firm+broker} × 六个 X 各自单独放入
                      （表 2、3、5、6，第 4 节第 6、7 条）
  paired              RT 与 WG 同名变量成对放入，Y ∈ {PMAFE_FY25, SCORE_FY25} × FE ∈ {firm, firm+broker}（表 4，第 4 节第 3 条）
  no_peerlag          同 fe_steps 的 PMAFE_FY25、SCORE_FY25，不放同行时滞（第 4 节第 1 条）
  peerlag_median      同行时滞换成中位数 LAGMED25，PMAFE_FY25 × {firm, firm+broker}（第 4 节第 1 条）
  peerlag_npeers      PEERLAG25 之外再控制 ln(同行家数) LNNPEERS25（第 4 节第 1 条）
  month_fe            加发布月份固定效应，PMAFE_FY25、SCORE_FY25 × {firm+month, firm+broker+month}（第 4 节第 1 条）
  window_plus30       比较组窗口 [d − 90, d + 30]：Y = PMAFEW30_FY25，时滞 = PEERLAGW30（第 4 节第 2 条）
  same_sample         WG 三个变量单独放入，样本限 RT_NSRC 有定义的研报（第 4 节第 4 条）
  log_sources         RT_LNSRC、WG_LNSRC 单独放入和成对放入（第 4 节第 5 条）
  excl_000858         剔除五粮液（000858）：六个 X 单独、三对成对、六个同时放入，PMAFE_FY25 × {firm, firm+broker}（第 4 节第 8 条）
块（公司-券商层面 firm_broker；控制变量 = 六个控制变量 + DAGE25）
  fe_steps            PMAFE_FY25 × FE 四档 × 六个 X 单独放入（表 7）
  paired              成对放入 × {firm, firm+broker}（表 8）

reg_all.csv 的列：level、block、Y、fe（上面的固定效应名）、fe_cols（实际的固定效应列）、controls、X（单独放入时为变量名，
成对或同时放入时为 '+' 连接的变量名）、term（本行系数对应的变量）、role（x 逻辑结构变量 / control 控制变量 / peerlag 同行时滞类
控制变量）、coef、se（按公司聚类）、t、p、n、n_clusters（公司数）、n_singletons（剔除的单例数）、adj_r2、r2、k（设计矩阵的秩）。
系数是原始单位（PMAFE 的系数没有乘 100）。
"""
import pandas as pd

from src.ols import fit
from src.studies.forecast_accuracy.sample import CTRL, LAG_W, X6, XLOG

FE_COLS = {
    'none': [], 'industry': ['sw', 'ty'], 'firm': ['code', 'ty'], 'firm+broker': ['code_ty', 'broker_g'],
    'firm+month': ['code', 'ty', 'month'], 'firm+broker+month': ['code_ty', 'broker_g', 'month'],
}
FE4 = ['none', 'industry', 'firm', 'firm+broker']
FE_FIRM = ['firm', 'firm+broker']
TARGET = {'PMAFE_FY25': '25', 'SCORE_FY25': '25', 'PMAFEW30_FY25': '25', 'PMAFE_FY24': '24'}  # Y -> 目标年度（TY、LNHORIZON 列的后缀）
LAG = {'PMAFE_FY25': 'PEERLAG25', 'SCORE_FY25': 'PEERLAG25', 'PMAFE_FY24': 'PEERLAG24', 'PMAFEW30_FY25': 'PEERLAGW30'}
PAIRS = [['RT_NSRC', 'WG_NSRC'], ['RT_LNPATH', 'WG_LNPATH'], ['RT_PATHLEN', 'WG_PATHLEN']]
EXCL_CODE = '000858'  # 五粮液：2025 年实际值待核实
COLS = ['level', 'block', 'Y', 'fe', 'fe_cols', 'controls', 'X', 'term', 'role', 'coef', 'se', 't', 'p', 'n', 'n_clusters',
        'n_singletons', 'adj_r2', 'r2', 'k']


def y_sample(d, y):
    """Y 的回归样本：基础样本（base），加目标年度 ty（字符串）和 code_ty = code + '_' + ty。"""
    d = d[d.base].copy()
    tag = TARGET[y]
    d['LNHORIZON'] = d['LNHORIZON' + tag]
    d['ty'] = d['TY' + tag].astype(str)
    d['code_ty'] = d.code + '_' + d.ty
    return d


def single(xs):
    return [(x, [x]) for x in xs]


def joint(*groups):
    return [('+'.join(g), list(g)) for g in groups]


def estimate(level, block, data, ys, fes, xsets, controls):
    """data 上的一组回归：ys × fes × xsets（[(X 标签, 变量列表)]）；controls(y) 给出控制变量。返回 reg_all 格式的行列表。"""
    rows = []
    for y in ys:
        d = y_sample(data, y)
        ctrl = controls(y)
        for fe in fes:
            for label, cols in xsets:
                res, _ = fit(d, y, cols, ctrl, FE_COLS[fe], 'code', terms=True)
                for r in res:
                    term = r['X']
                    role = 'x' if term in cols else 'peerlag' if term in LAG_W else 'control'
                    rows.append({'level': level, 'block': block, 'Y': y, 'fe': fe, 'fe_cols': '+'.join(FE_COLS[fe]),
                                 'controls': '+'.join(ctrl), 'X': label, 'term': term, 'role': role,
                                 **{k: r[k] for k in ('coef', 'se', 't', 'p', 'n', 'n_clusters', 'n_singletons', 'adj_r2', 'r2', 'k')}})
    return rows


def report_blocks(D):
    """研报层面的全部块（D = 研报层面回归样本）。"""
    lag = lambda y: CTRL + [LAG[y]]  # noqa: E731
    y2 = ['PMAFE_FY25', 'SCORE_FY25']
    rows = []
    add = lambda block, data, ys, fes, xsets, controls=lag: rows.extend(estimate('report', block, data, ys, fes, xsets, controls))  # noqa: E731
    add('fe_steps', D, ['PMAFE_FY25', 'PMAFE_FY24', 'SCORE_FY25'], FE4, single(X6))
    add('paired', D, y2, FE_FIRM, joint(*PAIRS))
    add('no_peerlag', D, y2, FE4, single(X6), lambda y: CTRL)
    add('peerlag_median', D, ['PMAFE_FY25'], FE_FIRM, single(X6), lambda y: CTRL + ['LAGMED25'])
    add('peerlag_npeers', D, ['PMAFE_FY25'], FE_FIRM, single(X6), lambda y: CTRL + ['PEERLAG25', 'LNNPEERS25'])
    add('month_fe', D, y2, ['firm+month', 'firm+broker+month'], single(X6))
    add('window_plus30', D, ['PMAFEW30_FY25'], FE_FIRM, single(X6))
    add('same_sample', D[D.RT_NSRC_RAW.notna()], y2, FE_FIRM, single(X6[3:]))
    add('log_sources', D, ['PMAFE_FY25'], FE_FIRM, single(XLOG) + joint(XLOG))
    add('excl_000858', D[D.code != EXCL_CODE], ['PMAFE_FY25'], FE_FIRM, single(X6) + joint(*PAIRS, X6))
    return rows


def pair_blocks(P):
    """公司-券商层面的块（P = 公司-券商层面回归样本）。"""
    ctrl = lambda y: CTRL + ['DAGE25']  # noqa: E731
    return (estimate('firm_broker', 'fe_steps', P, ['PMAFE_FY25'], FE4, single(X6), ctrl)
            + estimate('firm_broker', 'paired', P, ['PMAFE_FY25'], FE_FIRM, joint(*PAIRS), ctrl))


def run_all(D, P):
    """全部回归 -> reg_all 长表（列见模块说明）。"""
    res = pd.DataFrame(report_blocks(D) + pair_blocks(P), columns=COLS)
    assert not res.duplicated(['level', 'block', 'Y', 'fe', 'X', 'term']).any()
    return res


def estimation_sample(D, y='PMAFE_FY25', x='WG_NSRC', fe='firm+broker'):
    """基准回归（表 2）的估计样本：默认 PMAFE_FY25、全文源点数、公司 + 券商固定效应（709 篇）；返回 D 的对应行。"""
    d = y_sample(D, y)
    _, s = fit(d, y, [x], CTRL + [LAG[y]], FE_COLS[fe], 'code')
    return d.loc[s.index]
