"""回归设定和估计：文档用到的回归块，全部结果写成一张长表（reg_all.csv，每个系数一行）。

估计（src.ols.fit）
  每个（层面，块，Y，固定效应，X）一次 OLS：样本 = 回归样本中 Y、X、控制变量、固定效应、聚类变量都不缺失的观测，再反复剔除
  固定效应里的单例（只有一个观测的组）；X 在这个估计样本内标准化为均值 0、标准差 1，控制变量不标准化；固定效应用虚拟变量；
  标准误按公司（code）聚类，CR1：V = (X'X)⁻¹ Σ_g X_g'e_g e_g'X_g (X'X)⁻¹ · G/(G−1) · (N−1)/(N−K)，K 为设计矩阵的秩（常数项和
  全部固定效应虚拟变量都计入，包括嵌套在公司聚类内的公司固定效应）；p 值用 t(G−1)。

固定效应（FE，列名见 FE_COLS；全部是加性的，对应文档方程的 μ_b + δ_f + λ_t）
  年度 ty = 该 Y 所属窗口的目标年度 TY_Hh（研报层面逐篇不同：主窗口 H2 下 T = 2022 的研报是 2024、T = 2023 的是 2025；
  公司-券商层面是每对的 TY_H2）；样本里只有一个目标年度时（如 H3 只有 T = 2022 的研报）ty 的虚拟变量为空，被常数项吸收。
  none 无；industry 申万一级行业 + 年度（sw、ty）；firm 公司 + 年度（code、ty）；firm+broker 公司 + 年度 + 券商（code、ty、broker_g）；
  firm+month、firm+broker+month 在前两者上再加发布月份（month）；
  firmxyear、firmxyear+broker 公司 × 目标年度（code_ty = code + '_' + ty），后者再加券商（firm_x_year 块用）。

主窗口和变量名：主窗口 MAIN_H、主变量的定义名 MAIN_DEF 定义在 sample.py，Y_MAIN = ymain(MAIN_H) = {MAIN_DEF}_NP_H{MAIN_H}
（现为 PMED180_NP_H2，用数据文件的 180 天同行中位数派生的比例误差）、OWN_MAIN = PMAFE_H{MAIN_H}（自算，对照）、SCORE_MAIN、
LAG_MAIN、LNN_MAIN 等由它们派生；下面写 Ym_Hm（= Y_MAIN）、Ym_Ho（= ymain(h)，其他窗口的同名变量）、PMAFE_Hm、PEERLAG_Hm 等时
m = MAIN_H（现为 2），Ho 为其余两个窗口 OTHER_H（现为 1、3）。数据文件的被解释变量列名为 {定义}_{口径}_Hh（见 errors.py）。
Y 的窗口由列名后缀 _Hh 决定（horizon），控制的同行时滞一律取同一窗口自算 90 天比较组的 PEERLAG_Hh（数据文件没有给同行预测的
时间；PMAFEW30_Hh 配 PEERLAGW30_Hh）。

块（研报层面 report；控制变量 = 六个控制变量 + 同行时滞）
  fe_steps            Y ∈ {Ym_Hm, Ym_Ho（两个）} × FE ∈ {none, industry, firm, firm+broker} × 六个 X 各自单独放入
                      （主回归表、不同固定效应表、其他预测窗口表）
  paired              RT 与 WG 同名变量成对放入，Y ∈ {Ym_Hm, PMAFE_Hm（自算，对照）} × FE ∈ {firm, firm+broker}
  definitions         数据文件的其余定义：errors.ALL_Y（9 个定义 × 2 个口径 × 3 个窗口）中除主变量三个窗口（Ym_H1/H2/H3）之外的 51 个
                      × {firm, firm+broker} × 六个 X 单独放入（“不同误差定义”表）
  own_measures        自算的 PMAFE_Hm、SCORE_Hm × FE 四档 × 六个 X 单独放入（上一版的主结果，原样保留作对照）
  no_peerlag          Ym_Hm × FE 四档，不放同行时滞
  peerlag_median      同行时滞换成中位数 LAGMED_Hm，Ym_Hm × {firm, firm+broker}
  peerlag_npeers      PEERLAG_Hm 之外再控制数据文件主变量窗口的同行条数对数 LNN_MAIN（现为 ln(n180)，LNN180_NP_Hm）
  month_fe            加发布月份固定效应，Ym_Hm × {firm+month, firm+broker+month}
  same_sample         WG 三个变量单独放入，样本限 RT_NSRC 有定义的研报
  log_sources         RT_LNSRC、WG_LNSRC 单独放入和成对放入
  excl_000858         剔除五粮液（000858）：六个 X 单独、三对成对、六个同时放入，Ym_Hm × {firm, firm+broker}
  firm_x_year         公司 × 目标年度固定效应：Ym_Hm × {firmxyear, firmxyear+broker} × 六个 X 单独放入
块（公司-券商层面 firm_broker；控制变量 = 六个控制变量 + DAGE_Hm；Y = 数据文件里被评价预测的 Ym_Hm）
  fe_steps            Ym_Hm × FE 四档 × 六个 X 单独放入
  paired              成对放入 × {firm, firm+broker}
某个格子样本不足（src.ols.fit 没有结果，如 H3 加券商固定效应后自由度不够）时 reg_all 里没有这一行，表里写 NA。
块的顺序（reg_all 的行序）同上，由 blocks() 定义；run_all(only=...) 只重估其中几块、其余行取自已有的 reg_all（命令行
--part regress --blocks，公司-券商层面的块写成 firm_broker:fe_steps、firm_broker:paired）。

reg_all.csv 的列：level、block、Y、fe（上面的固定效应名）、fe_cols（实际的固定效应列）、controls、X（单独放入时为变量名，
成对或同时放入时为 '+' 连接的变量名）、term（本行系数对应的变量）、role（x 逻辑结构变量 / control 控制变量 / peerlag 同行时滞类
控制变量）、coef、se（按公司聚类）、t、p、n、n_clusters（公司数）、n_singletons（剔除的单例数）、adj_r2、r2、k（设计矩阵的秩）。
系数是原始单位（没有乘 100）。
"""
import re

import pandas as pd

from src.ols import fit
from src.studies.forecast_accuracy.errors import ALL_Y
from src.studies.forecast_accuracy.sample import (CTRL, DAGE_MAIN, LAG_MAIN, LAG_W, LAGMED_MAIN, LNN_MAIN, OTHER_H, OWN_MAIN,
                                                  SCORE_MAIN, X6, XLOG, Y_MAIN, hv, ymain)

FE_COLS = {
    'none': [], 'industry': ['sw', 'ty'], 'firm': ['code', 'ty'], 'firm+broker': ['code', 'ty', 'broker_g'],
    'firm+month': ['code', 'ty', 'month'], 'firm+broker+month': ['code', 'ty', 'broker_g', 'month'],
    'firmxyear': ['code_ty'], 'firmxyear+broker': ['code_ty', 'broker_g'],
}
FE4 = ['none', 'industry', 'firm', 'firm+broker']
FE_FIRM = ['firm', 'firm+broker']
FE_FIRMXYEAR = ['firmxyear', 'firmxyear+broker']
PAIRS = [['RT_NSRC', 'WG_NSRC'], ['RT_LNPATH', 'WG_LNPATH'], ['RT_PATHLEN', 'WG_PATHLEN']]
EXCL_CODE = '000858'  # 五粮液：FY2025 实际值待核实
COLS = ['level', 'block', 'Y', 'fe', 'fe_cols', 'controls', 'X', 'term', 'role', 'coef', 'se', 't', 'p', 'n', 'n_clusters',
        'n_singletons', 'adj_r2', 'r2', 'k']


def horizon(y):
    """Y 所属的预测窗口 h（列名后缀 _Hh，如 PMED180_NP_H2、PMAFE_H2、SCORE_H1）。"""
    m = re.search(r'_H(\d)$', y)
    assert m, f'{y} 不是按窗口的变量'
    return int(m.group(1))


def lag_of(y):
    """Y 配的同行时滞：同一窗口自算 90 天比较组的 PEERLAG_Hh（数据文件的各定义也用它）；PMAFEW30_Hh 配 PEERLAGW30_Hh。"""
    return hv('PEERLAGW30' if y.startswith('PMAFEW30') else 'PEERLAG', horizon(y))


def y_sample(d, y):
    """Y 的回归样本：基础样本（base），加 Y 所属窗口的目标年度 ty（字符串，取自 TY_Hh；缺失时为缺失）、
    code_ty = code + '_' + ty 和预测期 LNHORIZON（取自 LNHORIZON_Hh）。"""
    d = d[d.base].copy()
    h = horizon(y)
    d['LNHORIZON'] = d[hv('LNHORIZON', h)]
    d['ty'] = [str(int(t)) if pd.notna(t) else None for t in d[hv('TY', h)]]
    d['code_ty'] = [f'{c}_{t}' if t is not None else None for c, t in zip(d.code, d.ty)]
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


def blocks(D, P):
    """{(层面, 块名): 估计这一块的函数（返回 reg_all 格式的行列表）}；D、P = 两个层面的回归样本（只列块名时可以是 None）。
    字典的顺序就是 reg_all 的行序。"""
    lag = lambda y: CTRL + [lag_of(y)]  # noqa: E731
    dage = lambda y: CTRL + [DAGE_MAIN]  # noqa: E731
    ys_main = [Y_MAIN] + [ymain(h) for h in OTHER_H]
    ys_def = [y for y in ALL_Y if y not in ys_main]
    out = {}

    def add(level, block, ys, fes, xsets, controls=lag, keep=None):
        def f():
            data = D if level == 'report' else P
            return estimate(level, block, data if keep is None else data[keep(data)], ys, fes, xsets, controls)
        out[(level, block)] = f
    add('report', 'fe_steps', ys_main, FE4, single(X6))
    add('report', 'paired', [Y_MAIN, OWN_MAIN], FE_FIRM, joint(*PAIRS))
    add('report', 'definitions', ys_def, FE_FIRM, single(X6))
    add('report', 'own_measures', [OWN_MAIN, SCORE_MAIN], FE4, single(X6))
    add('report', 'no_peerlag', [Y_MAIN], FE4, single(X6), lambda y: CTRL)
    add('report', 'peerlag_median', [Y_MAIN], FE_FIRM, single(X6), lambda y: CTRL + [LAGMED_MAIN])
    add('report', 'peerlag_npeers', [Y_MAIN], FE_FIRM, single(X6), lambda y: CTRL + [LAG_MAIN, LNN_MAIN])
    add('report', 'month_fe', [Y_MAIN], ['firm+month', 'firm+broker+month'], single(X6))
    add('report', 'same_sample', [Y_MAIN], FE_FIRM, single(X6[3:]), keep=lambda d: d.RT_NSRC_RAW.notna())
    add('report', 'log_sources', [Y_MAIN], FE_FIRM, single(XLOG) + joint(XLOG))
    add('report', 'excl_000858', [Y_MAIN], FE_FIRM, single(X6) + joint(*PAIRS, X6), keep=lambda d: d.code != EXCL_CODE)
    add('report', 'firm_x_year', [Y_MAIN], FE_FIRMXYEAR, single(X6))
    add('firm_broker', 'fe_steps', [Y_MAIN], FE4, single(X6), dage)
    add('firm_broker', 'paired', [Y_MAIN], FE_FIRM, joint(*PAIRS), dage)
    return out


def parse_blocks(text):
    """--blocks 的取值 -> [(层面, 块名)]：逗号分隔，块名默认是研报层面，公司-券商层面写 firm_broker:fe_steps、firm_broker:paired。"""
    keys = blocks(None, None)
    out = []
    for b in filter(None, (s.strip() for s in text.split(','))):
        key = tuple(b.split(':', 1)) if ':' in b else ('report', b)
        if key not in keys:
            raise SystemExit(f'没有块 {b}；可选：' + '、'.join(k[1] if k[0] == 'report' else ':'.join(k) for k in keys))
        out.append(key)
    return out


def run_all(D, P, only=None, old=None):
    """全部回归 -> reg_all 长表（列见模块说明）。only 给出 [(层面, 块名)] 时只重估这些块，其余块的行原样取自 old（已有的
    reg_all），行序与全部重估相同。"""
    parts = []
    for key, f in blocks(D, P).items():
        if only is None or key in only:
            parts.append(pd.DataFrame(f(), columns=COLS))
        else:
            part = old[(old.level == key[0]) & (old.block == key[1])]
            assert len(part), f'已有的 reg_all 里没有块 {key}'
            parts.append(part)
    res = pd.concat(parts, ignore_index=True)
    assert not res.duplicated(['level', 'block', 'Y', 'fe', 'X', 'term']).any()
    return res


def read_csv(path):
    """读回本研究写出的 csv（utf-8-sig），浮点数与写出前逐位相同：pandas 3.1 之前用 float_precision='round_trip'；
    3.1 起所有精度选项都用同一个（逐位还原的）转换器，这个参数已弃用、传了会警告，所以不再传。"""
    major, minor = (int(x) for x in pd.__version__.split('.')[:2])
    kw = {} if (major, minor) >= (3, 1) else {'float_precision': 'round_trip'}
    return pd.read_csv(path, encoding='utf-8-sig', **kw)


def read_reg(path):
    """读回 reg_all.csv（fe = none 的 fe_cols 是空串，read_csv 会读成缺失，换回空串）。"""
    reg = read_csv(path)
    reg['fe_cols'] = reg.fe_cols.fillna('')
    return reg


def estimation_sample(D, y=Y_MAIN, x='WG_NSRC', fe='firm+broker'):
    """基准回归（主回归表）的估计样本：默认主变量 Y_MAIN、全文源点数、公司 + 年度 + 券商固定效应；返回 D 的对应行。"""
    d = y_sample(D, y)
    _, s = fit(d, y, [x], CTRL + [lag_of(y)], FE_COLS[fe], 'code')
    if s is None:  # 例如 MAIN_H = 3：只有 T = 2022 的研报有 H3 的实际值，RT 各列加公司 + 券商固定效应后自由度不够
        raise SystemExit(f'基准回归（{y}、{x}、{fe}）样本不足，src.ols.fit 没有结果，取不到估计样本')
    return d.loc[s.index]
