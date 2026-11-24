"""DAG 结构指标与研报盈利预测准确率的回归：预测面板（变量构造、样本筛选、缩尾）、解释变量口径、回归、主流程。
入口 regress_dag_forecast.py；表格 src/dag_forecast_tables.py；同行基准 / 联合 / 比值三个稳健性检验 src/dag_forecast_robust.py；
图和结构指标 src/dag_measures.py；估计函数 src/dag_outcomes_reg.fit。

常用接口（第三方跑表脚本直接调用，路径参数都有默认值）：
  build_panel(main_only=False)        -> (未缩尾面板 d, 同行明细 {'1'/'24'/'25': DataFrame}, 一致预期表, 预测长表 {'all','main'})
  forecast_panel(x_mode, main_only)   -> (d, w)：面板并上 x_mode 的解释变量，w 为缩尾后（回归用）
  y_sample(w, y)                      -> y 的回归样本（基础样本，按 y 设 LNHORIZON、ty、code_ty）
  regress_each(w, {标签: 列名}, ys, pans, controls=None, joint=False, terms=False)
                                      -> 每个 Y × Panel 的回归长表（X 单独或同时进入；可换控制变量；可输出控制变量系数和调整 R²）
  run(out_dir, x_mode, ...)           -> 一个口径的全部输出（同命令行）
build_panel、load_est 在同一进程内有缓存，同一进程跑多个口径时面板只建一次。

口径（方案没写到的细节在这里定）：
  预测行      data/EST_COMP_TN，只用 IS_VLD == 1；本篇的预测 = RPT_ID 等于 report_id 转大写、SEC_CD 等于本公司的那一行。
              第 k 年预测（PROFIT_Tk）的目标年度 = RPT_YR_T + k。同一券商同一天有多条时按 CREAT_TM、SEQ 取最后一条。
  实际值      financial_data 里 report_date = 年度1231 的 np_parent_company_owners_ttm（归母净利润），公告日 = date。
  前一交易日  研报日期之前（不含当天）最后一个市场交易日；个股取该日或之前最后一个未停牌交易日的值。
  MOM         前第 21 个交易日收盘 / 前第 250 个交易日收盘 − 1（后复权；某日停牌取该日或之前最后一个未停牌收盘）。
  VOL         前第 1–120 个交易日内未停牌日的日收益（相对上一个未停牌收盘）的标准差，不足 60 个记缺失。
  LNAGE       ln(（研报日 − 行情里首个未停牌交易日）的天数 / 30.4375)。
  缩尾        被解释变量和控制变量在基础样本（匹配到预测、有 T+1 实际值、研报日早于 T+1 年报公告日）内按 1%/99% 缩尾一次。
"""
import glob
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import dag_outcomes_reg as R
from src.config import DATA_DIR, ROOT_DIR
from src.dag_measures import (CF_RE, CFCORE_EXCLUDE, CFCORE_RE, DAG_DIRS, FINAL_LIST, RATING_NODE, VAL_EXCLUDE, VAL_RE,  # noqa: F401
                              classify_sink, final_names, load_graphs, sink_classification, sink_measures, strict_sinks)
from src.dag_outcomes_vars import daily_returns, load_daily

# 默认输入（regress_dag_forecast.py 用同样的路径）
PANEL = ROOT_DIR / 'output_dag' / 'regression' / 'report_panel.parquet'  # 研报层面面板（DAG 指标已缩尾）
EST_DIR = DATA_DIR / 'EST_COMP_TN'  # 分析师预测，按研报日期分月
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'
EST_YEARS = range(2018, 2025)


# ---------------------------------------------------------------- 解释变量口径
class Sink(NamedTuple):
    """--x 的一个汇点口径。targets：节点名元组（按名称取节点）或类别名 'VAL' / 'CF' / 'CFCORE' / 'FINAL'（取该类的全部严格汇点）。"""
    targets: object
    dir: str  # 默认输出子目录
    prefix: str  # 表格里指标名的前缀
    step: str  # 样本筛选表里加的那一步
    count: bool = False  # 按类别取汇点时 R_RATIO 用源点数（False：源点数 ÷ 汇点个数）
    strict: bool = False  # 只在目标节点是严格汇点（入度 > 0、出度 = 0）时有指标

    @property
    def by_class(self):
        return isinstance(self.targets, str)

    @property
    def ratio_name(self):
        """R_RATIO 的中文名。"""
        return '源汇比' if self.by_class and not self.count else '源点数'


def _strict(key, node):
    return Sink((node,), key, f'{node}节点（终点）', f'{node}节点是终点（入度>0、出度=0）', strict=True)


DAG_X = ['RATIO', 'LNPATH', 'PATHLEN']  # --x dag：整张图的三个指标（report_panel 里已缩尾的值）
RATING_X = ['R_RATIO', 'R_LNPATH', 'R_PATHLEN']  # 汇点口径的三个指标
SINKS = {
    'rating': Sink((RATING_NODE,), 'rating_sink', '评级汇点', '有投资评级节点且有路径'),
    'sink_np': Sink(('归母净利润',), 'sink_np', '归母净利润节点', '有归母净利润节点且有路径'),
    'sink_rev': Sink(('营业收入',), 'sink_rev', '营业收入节点', '有营业收入节点且有路径'),
    'sink_fund': Sink(('归母净利润', '营业收入'), 'sink_fund', '归母净利润和营业收入节点', '有归母净利润或营业收入节点且有路径'),
    'sink_val': Sink('VAL', 'sink_val', '估值类汇点', '有估值类汇点'),
    'sink_cf': Sink('CF', 'sink_cf', '现金流类汇点', '有现金流类汇点'),
    'sink_val_n': Sink('VAL', 'sink_val_n', '估值类汇点', '有估值类汇点', count=True),
    'sink_cf_core': Sink('CFCORE', 'sink_cf_core', '核心现金流类汇点', '有核心现金流类汇点', count=True),
    'rating_strict': _strict('rating_strict', RATING_NODE),
    'sink_rev_strict': _strict('sink_rev_strict', '营业收入'),
    'sink_np_strict': _strict('sink_np_strict', '归母净利润'),
    'sink_cf_core_ratio': Sink('CFCORE', 'sink_cf_core_ratio', '核心现金流类汇点', '有核心现金流类汇点'),
    # 最终指标清单（output_dag/forecast/final_indicators/final_indicator_list.csv，status == '入选'）里的全部严格汇点
    'sink_final': Sink('FINAL', 'sink_final', '最终指标清单汇点', '有最终指标清单内的汇点（入度>0、出度=0）', count=True),
}
X_MODES = ['dag'] + list(SINKS)


def x_list(x_mode):
    return RATING_X if x_mode in SINKS else DAG_X


def default_dir(root, x_mode='dag', main_only=False, with_nofe=False):
    """命令行不给 --output 时的输出目录：root/<口径目录>[_n][/main_only]（dag 口径不加 --with-nofe 时就是 root）。"""
    name = SINKS[x_mode].dir if x_mode in SINKS else ('dag' if with_nofe else '')
    base = Path(root) / (name + ('_n' if with_nofe else '')) if name else Path(root)
    return base / 'main_only' if main_only else base


def sink_frame(report_ids, x_mode, dag_dirs=DAG_DIRS):
    """x_mode（SINKS 的键）的 R_ 指标，每篇研报一行（src/dag_measures.sink_measures）。"""
    s = SINKS[x_mode]
    return sink_measures(list(report_ids), dag_dirs, s.targets, count=s.count, strict=s.strict)


# ---------------------------------------------------------------- 被解释变量、控制变量、回归设定
YS = ['AFE1', 'AFE2', 'AFE_FY24', 'AFE1_P', 'PMAFE1', 'AFE_FY25', 'PMAFE_FY24', 'PMAFE_FY25']
RPT_CTRL = ['LNHORIZON', 'LNCHARS_R', 'LNNODES_R']
FIRM_CTRL = ['SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']
# 每个 Y 用哪一列作 LNHORIZON、哪一列作目标年度
Y_HORIZON = {'AFE1': 'LNHORIZON1', 'AFE2': 'LNHORIZON2', 'AFE_FY24': 'LNHORIZON24', 'AFE1_P': 'LNHORIZON1', 'PMAFE1': 'LNHORIZON1',
             'AFE_FY25': 'LNHORIZON25', 'PMAFE_FY24': 'LNHORIZON24', 'PMAFE_FY25': 'LNHORIZON25'}
Y_TY = {'AFE1': 'TY1', 'AFE2': 'TY2', 'AFE_FY24': 'TY24', 'AFE1_P': 'TY1', 'PMAFE1': 'TY1', 'AFE_FY25': 'TY25',
        'PMAFE_FY24': 'TY24', 'PMAFE_FY25': 'TY25'}
PANELS = {  # 固定效应、控制变量
    'A': (['sw', 'ty'], RPT_CTRL + FIRM_CTRL),
    'B': (['code', 'ty'], RPT_CTRL + ['SIZE_M', 'MOM', 'VOL']),
    'C': (['code_ty', 'broker_g'], RPT_CTRL),
}
PANEL_N = {'N': ([], RPT_CTRL + FIRM_CTRL)}  # 不加固定效应（只有截距），控制变量同 Panel A（--with-nofe 时排在 A 前面）


def panels(with_nofe=False):
    """回归用的 Panel：默认 A、B、C；with_nofe=True 时为 N、A、B、C。"""
    return {**PANEL_N, **PANELS} if with_nofe else PANELS


HORIZONS = ['LNHORIZON1', 'LNHORIZON2', 'LNHORIZON24', 'LNHORIZON25']
WINSOR_VARS = YS + HORIZONS + ['LNCHARS_R', 'LNNODES_R'] + FIRM_CTRL
PAST_YEARS = 5  # STD_PAST 用 T−4..T
MIN_PAST = 3  # STD_PAST 至少几年有值
PEER_DAYS = 90  # 同行预测的窗口（自然日，含研报当天）
MIN_PEERS = 3  # 同行至少几家券商
MIN_VOL_DAYS = 60  # VOL 至少几个日收益


# ---------------------------------------------------------------- 读数据
@lru_cache(maxsize=2)
def _load_est(est_dir, years, codes):
    cols = ['SEQ', 'CREAT_TM', 'RPT_ID', 'SEC_CD', 'RPT_MAIN_CD', 'ORG_CL', 'DATA_SRC', 'RPT_DT', 'RPT_YR_T',
            'PROFIT_T1', 'PROFIT_T2', 'PROFIT_T3']
    # 只读规范文件名 YYYY-MM.parquet（目录里可能有 YYYY-MM_时间戳.parquet、YYYY-MM(1).parquet 之类的备份，不能一起读进来）
    fs = [f for y in years for f in sorted(glob.glob(str(Path(est_dir) / str(y) / f'{y}-[0-9][0-9].parquet')))]
    e = pd.concat([pq.read_table(f, columns=cols, filters=[('IS_VLD', '=', 1), ('SEC_CD', 'in', list(codes))]).to_pandas()
                   for f in fs], ignore_index=True)
    e['RPT_DT'] = e.RPT_DT.dt.strftime('%Y%m%d')
    return e


def load_est(est_dir, years, codes):
    """读若干年的预测表（只读用到的列），只留 IS_VLD == 1 且 SEC_CD 在样本里的行；RPT_DT 转成 YYYYMMDD 字符串。
    同一进程内同样的参数只读一次（返回副本）。"""
    return _load_est(str(est_dir), tuple(years), tuple(codes)).copy()


def to_long(e, extra=()):
    """一行一个（预测行 × k）：SEC_CD、ORG_CL、RPT_DT、k、ty（目标年度）、F（预测值），按公司、目标年度、券商、时间排序
    （多列排序是稳定排序，所以先排序再筛行与先筛行再排序结果相同）。extra 为另外带上的列（放在 SEQ 后面）。"""
    keep = ['RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'CREAT_TM', 'SEQ'] + list(extra)
    parts = [e[keep + [f'PROFIT_T{k}']].rename(columns={f'PROFIT_T{k}': 'F'}).assign(k=k) for k in (1, 2, 3)]
    lg = pd.concat(parts, ignore_index=True)
    lg['ty'] = lg.RPT_YR_T + lg.k
    return lg[lg.F.notna()].sort_values(['SEC_CD', 'ty', 'ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ']).reset_index(drop=True)


def load_annual(fin_path, codes):
    """年报：(code, 财年) -> 归母净利润 A、净利润 NP、公告日 ANN、总资产 TA、净资产 EQ。"""
    cols = ['report_date', 'code', 'date', 'total_assets', 'total_liability', 'np_parent_company_owners_ttm', 'net_profit_ttm']
    f = pq.read_table(fin_path, columns=cols).to_pandas()
    f = f[f.code.isin(codes) & f.report_date.str.endswith('1231')].drop_duplicates(['code', 'report_date'], keep='last')
    return pd.DataFrame({'code': f.code, 'fy': f.report_date.str[:4].astype(int), 'A': f.np_parent_company_owners_ttm,
                         'NP': f.net_profit_ttm, 'ANN': f.date, 'TA': f.total_assets,
                         'EQ': f.total_assets - f.total_liability}).set_index(['code', 'fy'])


# ---------------------------------------------------------------- 一致预期误差和 STD_PAST
def consensus_errors(lg, ann):
    """每个（公司，财年 s）：日历年 s 内发布、目标年度为 s 的预测，每家券商取当年最后一条，取均值；返回 CONS、N_BROKER、ERR = |CONS − 实际|。"""
    c = lg[lg.RPT_DT.str[:4].astype(int) == lg.ty]
    last = c.groupby(['SEC_CD', 'ty', 'ORG_CL']).F.last()  # lg 已按时间排序，last 即最后一条
    g = last.groupby(level=[0, 1])
    out = pd.DataFrame({'CONS': g.mean(), 'N_BROKER': g.size()})
    out.index.names = ['code', 'fy']
    out['A'] = ann.A.reindex(out.index)
    out['ERR'] = (out.CONS - out.A).abs()
    return out


def std_past(err, code, T):
    """T−4..T 各年的一致预期绝对误差（列表）、STD_PAST（至少 MIN_PAST 年有值，ddof=1）、有值年数。err = {(code, 财年): ERR}。"""
    errs = [err.get((code, s), np.nan) for s in range(T - PAST_YEARS + 1, T + 1)]
    v = np.array([x for x in errs if np.isfinite(x)])
    sd = v.std(ddof=1) if len(v) >= MIN_PAST else np.nan
    return errs, (sd if sd > 0 else np.nan), len(v)


# ---------------------------------------------------------------- 同行
class Peers:
    """预测长表 lg（to_long 的结果，已按公司、目标年度、券商、时间排序）上查同行预测。"""

    def __init__(self, lg):
        self.lg = lg
        code, ty = lg.SEC_CD.to_numpy(), lg.ty.to_numpy()
        self.org, self.dt, self.f = lg.ORG_CL.to_numpy(), lg.RPT_DT.astype(int).to_numpy(), lg.F.to_numpy()
        start = np.flatnonzero(np.r_[True, (code[1:] != code[:-1]) | (ty[1:] != ty[:-1])])
        stop = np.r_[start[1:], len(lg)]
        self.slices = {(code[a], int(ty[a])): (a, b) for a, b in zip(start, stop)}

    def last_each(self, code, ty, org, date, days=PEER_DAYS, today=True):
        """其他券商（ORG_CL != org）在 [date − days 天, date]（today=False 时不含 date）内对 (code, ty) 的预测，
        每家取最近一条：返回这些行在 lg 里的位置（升序，同 groupby('ORG_CL').tail(1)）；没有同行时为 None。"""
        sl = self.slices.get((code, ty))
        if sl is None:
            return None
        a, b = sl
        lo = int((pd.Timestamp(date) - pd.Timedelta(days=days)).strftime('%Y%m%d'))
        dt, o, day = self.dt[a:b], self.org[a:b], int(date)
        idx = np.flatnonzero((o != org) & (dt >= lo) & ((dt <= day) if today else (dt < day)))
        oo = o[idx]
        return a + idx[np.r_[oo[1:] != oo[:-1], True]] if len(idx) else idx

    def abs_errors(self, pos, actual):
        """这些行的预测相对实际值的绝对误差（Series）。"""
        return pd.Series(np.abs(self.f[pos] - actual))

    def detail(self, rids, poss):
        """同行明细：各篇研报的同行行（lg 的列）加 report_id，按研报顺序接起来。"""
        keep = [(r, p) for r, p in zip(rids, poss) if p is not None and len(p)]
        rows = self.lg.iloc[np.concatenate([p for _, p in keep])].reset_index(drop=True)
        return rows.assign(report_id=pd.Series([r for r, p in keep for _ in p], dtype=str))


# ---------------------------------------------------------------- 行情
def market_vars(rep, daily_path):
    """每篇研报：MKCAP（前一交易日总市值）、MOM、VOL、FIRST_DAY（行情里首个未停牌交易日）。"""
    codes = sorted(set(rep.code))
    start = (pd.Timestamp(rep.date.min()) - pd.Timedelta(days=500)).strftime('%Y%m%d')
    d = load_daily(daily_path, start, rep.date.max())
    cal = np.array(sorted(d.date.unique()))
    t, _ = daily_returns(d[d.code.isin(codes)])
    by = {c: (g.date.to_numpy(), g.close.to_numpy(), g.mkcap.to_numpy(), g.ret.to_numpy()) for c, g in t.groupby('code')}
    first = pq.read_table(daily_path, columns=['code', 'date', 'close', 'is_suspended'], filters=[('code', 'in', codes)]).to_pandas()
    first = first[~first.is_suspended & (first.close > 0)].groupby('code').date.min()
    rows = []
    for rid, code, date in zip(rep.report_id, rep.code, rep.date):
        rec = {'report_id': rid, 'FIRST_DAY': first.get(code)}
        i0 = np.searchsorted(cal, date, 'left') - 1  # 前一交易日在日历里的位置
        if code in by and i0 >= 249:
            dt, cl, mk, rt = by[code]
            at = lambda day: np.searchsorted(dt, day, 'right') - 1  # noqa: E731  该日或之前最后一个未停牌交易日
            j = at(cal[i0])
            if j >= 0:
                rec.update(PREV_DAY=cal[i0], MKCAP_DAY=dt[j], MKCAP=mk[j])
            j21, j250 = at(cal[i0 - 20]), at(cal[i0 - 249])
            if j250 >= 0:
                rec['MOM'] = cl[j21] / cl[j250] - 1
            w = (dt >= cal[i0 - 119]) & (dt <= cal[i0]) & np.isfinite(rt)
            if w.sum() >= MIN_VOL_DAYS:
                rec['VOL'] = rt[w].std(ddof=1)
            rec['N_VOL'] = int(w.sum())
        rows.append(rec)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 研报层面面板
@lru_cache(maxsize=4)
def _build_panel(panel_path, est_dir, fin_path, daily_path, main_only):
    p = pd.read_parquet(panel_path)
    codes = sorted(set(p.code))
    est = load_est(est_dir, EST_YEARS, codes)
    ann = load_annual(fin_path, codes)
    lg = to_long(est.assign(_MAIN=est.SEC_CD == est.RPT_MAIN_CD), ['_MAIN'])  # 主体公司行 = 研报主体公司的预测
    lgs = {'all': lg.drop(columns='_MAIN'), 'main': lg[lg._MAIN].drop(columns='_MAIN').reset_index(drop=True)}
    lg = lgs['main'] if main_only else lgs['all']
    cons = consensus_errors(lg, ann)
    peers = Peers(lg)

    own = est.rename(columns={'RPT_ID': 'RID', 'SEC_CD': 'code'})
    # 预测表里有少量 RPT_ID 含小写字母，统一转大写再匹配；同一 (RPT_ID, 公司) 有多条 IS_VLD == 1 时取 RPT_DT、CREAT_TM、SEQ 最后一条
    # （2026-11-22 的数据里 2010–2026 年都没有这种重复，排序只为结果不依赖文件顺序）
    own['RID'] = own.RID.str.upper()
    own = own.sort_values(['RPT_DT', 'CREAT_TM', 'SEQ'], kind='mergesort')
    own = own.drop_duplicates(['RID', 'code'], keep='last')[['RID', 'code', 'ORG_CL', 'RPT_YR_T', 'RPT_MAIN_CD', 'DATA_SRC',
                                                             'PROFIT_T1', 'PROFIT_T2', 'PROFIT_T3']]
    d = p[['report_id', 'code', 'company', 'date', 'broker', 'broker_g', 'sw', 'csrc', 'fin'] + DAG_X + ['LNCHARS_R', 'LNNODES_R']].copy()
    d['RID'] = d.report_id.str.upper()
    d = d.merge(own, on=['RID', 'code'], how='left')
    d['matched'] = d.RPT_YR_T.notna()
    d['T'] = d.RPT_YR_T.astype('Int64')
    lut = {c: ann[c].to_dict() for c in ann.columns}
    get = lambda col, fy: [lut[col].get((c, int(y)), np.nan) if pd.notna(y) else np.nan for c, y in zip(d.code, fy)]  # noqa: E731
    d['TY1'], d['TY2'], d['TY24'], d['TY25'] = d['T'] + 1, d['T'] + 2, 2024, 2025
    d['F1'], d['F2'] = d.PROFIT_T1, d.PROFIT_T2
    d['F24'] = np.where(d.RPT_YR_T == 2023, d.PROFIT_T1, np.where(d.RPT_YR_T == 2022, d.PROFIT_T2, np.nan))
    d['F25'] = np.where(d.RPT_YR_T == 2023, d.PROFIT_T2, np.where(d.RPT_YR_T == 2022, d.PROFIT_T3, np.nan))
    for tag, fy in (('1', d.TY1), ('2', d.TY2), ('24', d.TY24), ('25', d.TY25), ('T', d['T'])):
        d[f'A{tag}'], d[f'ANN{tag}'] = get('A', fy), get('ANN', fy)
    d['NP1'] = get('NP', d.TY1)
    d['TA_T'], d['EQ_T'] = get('TA', d['T']), get('EQ', d['T'])

    # 目标年度核对
    d['has_act1'] = d.matched & d.F1.notna() & d.A1.notna()
    d['ann1_after'] = d.has_act1 & (d.ANN1.fillna('') > d.date)
    d['annT_after_rpt'] = d.matched & (d.ANNT.fillna('') > d.date)  # T 年年报在研报日之后才公告（方案要求只数不处理）
    base = d.ann1_after

    # STD_PAST
    err = cons.ERR.to_dict()
    sp = [std_past(err, c, int(T)) if pd.notna(T) else ([np.nan] * PAST_YEARS, np.nan, 0) for c, T in zip(d.code, d['T'])]
    for j in range(PAST_YEARS):
        d[f'PAST_ERR_L{PAST_YEARS - 1 - j}'] = [s[0][j] for s in sp]  # L4 = T−4 年，L0 = T 年
    d['STD_PAST'], d['N_PAST'] = [s[1] for s in sp], [s[2] for s in sp]

    # 同行：目标年度 T+1（N_PEER、PEER_ERR）、FY2024（N_PEER24、PEER_ERR24）、FY2025（N_PEER25、PEER_ERR25）
    detail = {}
    for tag, suf in (('1', ''), ('24', '24'), ('25', '25')):
        poss, n_peer, peer_err = [], [], []
        for code, ty, org, date, a, ok in zip(d.code, d[f'TY{tag}'], d.ORG_CL, d.date, d[f'A{tag}'], base):
            pos = peers.last_each(code, int(ty), org, date) if ok else None
            poss.append(pos)
            n_peer.append(0 if pos is None else len(pos))
            peer_err.append(peers.abs_errors(pos, a).mean() if pos is not None and len(pos) else np.nan)
        d[f'N_PEER{suf}'], d[f'PEER_ERR{suf}'] = n_peer, peer_err
        detail[tag] = peers.detail(d.report_id, poss)

    # 被解释变量（只在基础样本里算）
    mv = market_vars(d[['report_id', 'code', 'date']], daily_path)
    d = d.merge(mv, on='report_id', how='left')
    d['ABSERR1'], d['ABSERR2'], d['ABSERR24'] = (d.F1 - d.A1).abs(), (d.F2 - d.A2).abs(), (d.F24 - d.A24).abs()
    d['ABSERR25'] = (d.F25 - d.A25).abs()
    d['AFE1'] = d.ABSERR1 / d.STD_PAST
    d['AFE2'] = d.ABSERR2 / d.STD_PAST
    d['AFE_FY24'] = d.ABSERR24 / d.STD_PAST
    d['AFE_FY25'] = d.ABSERR25 / d.STD_PAST
    d['AFE1_P'] = d.ABSERR1 / d.MKCAP * 100
    d['PMAFE1'] = ((d.ABSERR1 - d.PEER_ERR) / d.PEER_ERR).where((d.N_PEER >= MIN_PEERS) & (d.PEER_ERR > 0))
    for t in ('24', '25'):
        pe, npr = d[f'PEER_ERR{t}'], d[f'N_PEER{t}']
        d[f'PMAFE_FY{t}'] = ((d[f'ABSERR{t}'] - pe) / pe).where((npr >= MIN_PEERS) & (pe > 0))
    for y in YS:
        d[y] = d[y].where(base).replace([np.inf, -np.inf], np.nan)

    # 控制变量
    days = lambda a: (pd.to_datetime(a, format='%Y%m%d', errors='coerce') - pd.to_datetime(d.date, format='%Y%m%d')).dt.days  # noqa: E731
    for tag in ('1', '2', '24', '25'):
        h = days(d[f'ANN{tag}'])
        d[f'HORIZON{tag}'] = h
        d[f'LNHORIZON{tag}'] = np.log(h.where(h > 0))
    d['SIZE_M'] = np.log(d.MKCAP)
    d['BM'] = d.EQ_T / d.MKCAP
    d['ROA_T'] = d.AT / d.TA_T
    d['AGE_M'] = (-days(d.FIRST_DAY)) / 30.4375
    d['LNAGE'] = np.log(d.AGE_M.where(d.AGE_M > 0))
    d['base'] = base
    return d.drop(columns='RID'), detail, cons, lgs


def build_panel(panel_path=PANEL, est_dir=EST_DIR, fin_path=FINANCIAL, daily_path=DAILY, main_only=False):
    """返回 (未缩尾面板, 同行明细 {目标年度标签 '1'/'24'/'25': DataFrame（同行行 + report_id）}, 一致预期表,
    长表字典 {'all': 全部预测行, 'main': 只用主体公司行})。面板每篇研报一行，base = 基础样本。
    main_only=True 时一致预期和同行只用 SEC_CD == RPT_MAIN_CD 的预测行（研报主体公司的预测；本篇自己的预测行不受影响）。
    同一进程内同样的参数只建一次（返回面板的副本；其余三项共用，不要原地修改）。"""
    d, detail, cons, lgs = _build_panel(str(panel_path), str(est_dir), str(fin_path), str(daily_path), bool(main_only))
    return d.copy(), detail, cons, lgs


def winsorize(d):
    """WINSOR_VARS 在基础样本内按 1%/99% 缩尾（基础样本外的行不动）。"""
    w = d.copy()
    for v in WINSOR_VARS:
        w.loc[w.base, v] = R.winsor(w.loc[w.base, v])
    return w


def forecast_panel(x_mode='dag', main_only=False, dag_dirs=DAG_DIRS, **paths):
    """预测面板并上 x_mode 的解释变量：返回 (未缩尾 d, 缩尾后 w)。汇点口径的 R_ 指标在有值的研报里已缩尾（不随 w 再缩）。
    paths 可传 panel_path、est_dir、fin_path、daily_path。"""
    d = build_panel(main_only=main_only, **paths)[0]
    if x_mode in SINKS:
        d = d.merge(sink_frame(d.report_id, x_mode, dag_dirs), on='report_id', how='left')
    return d, winsorize(d)


# ---------------------------------------------------------------- 回归
def set_target_year(d, horizon_col, ty_col):
    """副本：LNHORIZON = horizon_col，ty = ty_col 转字符串，code_ty = code + '_' + ty（Panel B、C 的固定效应用）。"""
    d = d.copy()
    d['LNHORIZON'] = d[horizon_col]
    d['ty'] = d[ty_col].astype(str)
    d['code_ty'] = d.code + '_' + d.ty
    return d


def y_sample(w, y):
    """y 的回归样本：基础样本，LNHORIZON、ty 按 Y_HORIZON、Y_TY 取。"""
    return set_target_year(w[w.base], Y_HORIZON[y], Y_TY[y])


def pick(r, cols):
    return {c: r[c] for c in cols}


def slim(d, ys, xs, pans):
    """只留回归要用的列（R.fit 每次都要从样本里取列，宽表很慢；值不变）：ys、xs、各 Panel 的控制变量和固定效应、code、report_id。"""
    cols = list(ys) + list(xs) + [c for fe, ctrl in pans.values() for c in ctrl + fe] + ['code', 'report_id']
    return d[list(dict.fromkeys(cols))]


def run_regressions(w, xs=DAG_X, pans=PANELS):
    """每个 Y × Panel（pans）× (三个 X 单独、联合)，控制变量的系数也输出。返回长表（reg_results.csv）。"""
    rows = []
    for y in YS:
        d = slim(y_sample(w, y), [y], xs, pans)
        for pan, (fe, ctrl) in pans.items():
            for reg, xx in [('单独', [x]) for x in xs] + [('联合', xs)]:
                res, _ = R.fit(d, y, xx, ctrl, fe, 'code', terms=True)
                rows += [{'Y': y, 'panel': pan, 'reg': reg, 'X_set': '+'.join(xx), 'X': r['X'],
                          **pick(r, ['coef', 'se', 't', 'p', 'n', 'adj_r2', 'r2', 'n_clusters', 'n_singletons']),
                          'fe': '+'.join(fe), 'controls': '+'.join(ctrl), 'y_mean': r['y_mean'], 'y_sd': r['y_sd']} for r in res]
    return pd.DataFrame(rows)


def regress_each(w, xs, ys=YS, pans=None, return_ids=False, controls=None, joint=False, terms=False, standardize=True):
    """Y × Panel 的回归（按公司聚类，固定效应里的 singleton 反复剔除，src/dag_outcomes_reg.fit）。
    w          缩尾后的面板（forecast_panel 的第二个返回值，或已有的 forecast_panel.parquet，可先并上自己的 X 列）；回归样本 = y_sample(w, y)
    xs         {结果里的 X 标签: 面板列名}，或列名列表
    ys         被解释变量列表（须是 YS 里的名字：决定 LNHORIZON 和目标年度 ty 用哪一列）
    pans       {Panel 名: (固定效应列表, 控制变量列表)}，默认 panels(with_nofe=True)，即 N、A、B、C
    controls   None：用 pans 里各 Panel 自带的控制变量；列表：所有 Panel 都换成这一组（固定效应不变）；{Panel 名: 列表}：逐个指定。
               与 X 同名的控制变量自动去掉（避免同一列进设计矩阵两次）
    joint      False（默认）：每个 X 单独进入一个回归；True：全部 X 同时进入一个回归
    terms      True：结果里另有控制变量的系数（role = 'control'，X 为列名；常数项和固定效应虚拟变量不输出）
    standardize X 在回归样本内标准化（默认；控制变量不标准化）
    返回长表，每个系数一行：Y、panel、X、coef、se（按公司聚类 CR1）、t、p（t(G−1)）、n、n_clusters、fe、controls，
    以及 spec（单独进入时 = 该 X 的标签，同时进入时 = 'joint'）、role（'x' / 'control'）、adj_r2、r2、k（设计矩阵的秩，含常数项和
    固定效应虚拟变量）、n_singletons（同一回归的各行相同）。样本不足的回归不出行。return_ids=True 时另返回进入任一回归的 report_id 集合。
    默认参数（controls=None、joint=False、terms=False）时前 11 列与原来的 regress_each 完全相同。"""
    xs = xs if isinstance(xs, dict) else {x: x for x in xs}
    label = {c: lab for lab, c in xs.items()}
    pans = pans or panels(True)
    if controls is not None:
        pans = {p: (fe, list(controls[p] if isinstance(controls, dict) else controls)) for p, (fe, _) in pans.items()}
    pans = {p: (fe, [c for c in ctrl if c not in label]) for p, (fe, ctrl) in pans.items()}
    specs = [('joint', list(xs.values()))] if joint else [(lab, [col]) for lab, col in xs.items()]
    rows, used = [], set()
    for y in ys:
        d = slim(y_sample(w, y), [y], xs.values(), pans)
        for pan, (fe, ctrl) in pans.items():
            for spec, cols in specs:
                res, s = R.fit(d, y, cols, ctrl, fe, 'code', standardize=standardize, terms=terms)
                if not res:
                    continue
                used |= set(d.loc[s.index, 'report_id'])
                for r in res:
                    is_x = r['X'] in cols
                    rows.append({'Y': y, 'panel': pan, 'X': label[r['X']] if is_x else r['X'],
                                 **pick(r, ['coef', 'se', 't', 'p', 'n', 'n_clusters']), 'fe': '+'.join(fe), 'controls': '+'.join(ctrl),
                                 'spec': spec, 'role': 'x' if is_x else 'control', **pick(r, ['adj_r2', 'r2', 'k', 'n_singletons'])})
    out = pd.DataFrame(rows)
    return (out, used) if return_ids else out


def funnel(d, res, xs=DAG_X, sink=None, pans=PANELS):
    """样本筛选：每一步剩多少篇、多少家公司；最后是各 Y 的非缺失数和各 Panel 联合回归的样本。
    汇点口径（sink 为 SINKS 的键）在“研报日早于 T+1 年报公告日”之后加一步“有××节点且有路径”，之后各步都在其内计。"""
    keep = d.has_rating if sink else pd.Series(True, index=d.index)
    steps = [('DAG 研报（report_panel）', pd.Series(True, index=d.index)),
             ('匹配到本公司预测行（IS_VLD=1）', d.matched),
             ('有 T+1 预测值和实际值', d.has_act1),
             ('研报日早于 T+1 年报公告日', d.ann1_after)]
    if sink:
        steps.append((SINKS[sink].step, d.ann1_after & keep))
    steps.append(('STD_PAST 非缺失', d.ann1_after & keep & d.STD_PAST.notna()))
    rows = [{'step': s, 'n_reports': int(m.sum()), 'n_firms': int(d.code[m].nunique())} for s, m in steps]
    for y in YS:
        m = d[y].notna() & keep
        rows.append({'step': f'{y} 非缺失', 'n_reports': int(m.sum()), 'n_firms': int(d.code[m].nunique())})
        for pan in pans:
            r = res[(res.Y == y) & (res.panel == pan) & (res.reg == '联合') & (res.X == xs[0])]
            if len(r):
                rows.append({'step': f'{y} Panel {pan} 回归样本', 'n_reports': int(r.n.iloc[0]), 'n_firms': int(r.n_clusters.iloc[0])})
    return pd.DataFrame(rows)


def describe(w, cols):
    s = w[w.base]
    rows = []
    for v in cols:
        x = s[v].dropna()
        rows.append({'var': v, 'n': len(x), 'mean': x.mean(), 'sd': x.std(), 'min': x.min(), 'p25': x.quantile(.25),
                     'p50': x.median(), 'p75': x.quantile(.75), 'max': x.max()})
    return pd.DataFrame(rows)


def node_coverage(fc_dir, dirs=(('投资评级', 'rating_sink'), ('营业收入', 'sink_rev'))):
    """两个按名称取的节点在全部研报里的覆盖：读各口径已有的 forecast_panel_raw.parquet（不重算），返回长表 node、item、n_reports、n_firms。"""
    fc_dir = Path(fc_dir)
    ps = {n: pd.read_parquet(fc_dir / sub / 'forecast_panel_raw.parquet') for n, sub in dirs}
    ref = next(iter(ps.values()))
    for p in ps.values():  # 两个口径的研报和基础样本必须一样
        assert list(p.report_id) == list(ref.report_id) and (p.base == ref.base).all()
    code, base = ref.code, ref.base.to_numpy()
    rows = []

    def add(node, item, m):
        m = np.asarray(m, dtype=bool)
        rows.append({'node': node, 'item': item, 'n_reports': int(m.sum()), 'n_firms': int(code[m].nunique())})

    has = {}
    for n, p in ps.items():
        h = p.R_LNPATH_RAW.notna().to_numpy()
        assert (h == (p.R_HAS_NODE.fillna(False) & (p.R_N_ANC.fillna(0) > 0)).to_numpy()).all()  # 有路径 = 节点在图里且有入边
        has[n] = h
        add(n, '全部研报', np.ones(len(p)))
        add(n, '图里有该节点', p.R_HAS_NODE.fillna(False))
        add(n, '有入边（三个指标非缺失）', h)
        add(n, '有指标且该节点出度为 0', h & (p.R_OUT_DEG.fillna(-1) == 0).to_numpy())
        add(n, '有指标且该节点出度大于 0', h & (p.R_OUT_DEG.fillna(-1) > 0).to_numpy())
        add(n, '基础样本（base）内有指标', h & base)
    (a, b), (ha, hb) = list(ps), list(has.values())
    both = f'{a}+{b}'
    add(both, '两个节点都有指标（全部）', ha & hb)
    add(both, '两个节点都有指标（基础样本内）', ha & hb & base)
    add(both, f'只有{a}有指标', ha & ~hb)
    add(both, f'只有{b}有指标', ~ha & hb)
    add(both, '两个节点都没有指标', ~ha & ~hb)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def run(out_dir, x_mode='dag', main_only=False, with_nofe=False, peer_variants=False, dag_dirs=DAG_DIRS, **paths):
    """一个口径的全部输出。x_mode = 'dag'：整张图的三个指标；SINKS 的键：该汇点口径的三个 R_ 指标。
    peer_variants=True 时只做同行基准稳健性检验（src/dag_forecast_robust.py），写 peer_variants.csv、peer_variants_panel.parquet、
    tables/f11–f12，不动其他输出。with_nofe=True 时回归多一个不加固定效应的 Panel N（控制变量同 Panel A），顺序 N、A、B、C。
    paths 可传 panel_path、est_dir、fin_path、daily_path。返回 (d, w, 回归长表)。"""
    from src import dag_forecast_tables as T
    pans = panels(with_nofe)
    out = Path(out_dir)
    (out / 'tables').mkdir(parents=True, exist_ok=True)
    xs, sink = x_list(x_mode), (x_mode if x_mode in SINKS else None)
    _, detail, cons, lgs = build_panel(main_only=main_only, **paths)
    d, w = forecast_panel(x_mode, main_only, dag_dirs, **paths)
    if peer_variants:
        from src import dag_forecast_robust
        dag_forecast_robust.peer_variants(out, w, lgs, xs, sink)
        return d, w, None
    d.to_parquet(out / 'forecast_panel_raw.parquet', index=False)
    w.to_parquet(out / 'forecast_panel.parquet', index=False)
    for tag, name in (('1', 'peer_detail'), ('24', 'peer_detail_fy24'), ('25', 'peer_detail_fy25')):
        detail[tag].to_parquet(out / f'{name}.parquet', index=False)
    cons.reset_index().to_parquet(out / 'consensus_errors.parquet', index=False)
    res = run_regressions(w, xs, pans)
    res.to_csv(out / 'reg_results.csv', index=False, encoding='utf-8-sig')
    fn = funnel(w, res, xs, sink, pans)
    fn.to_csv(out / 'funnel.csv', index=False, encoding='utf-8-sig')
    desc = describe(w, YS + DAG_X + (RATING_X if sink else []) + HORIZONS + ['LNCHARS_R', 'LNNODES_R'] + FIRM_CTRL)
    desc.to_csv(out / 'descriptives.csv', index=False, encoding='utf-8-sig')
    T.write_tables(out / 'tables', w, fn, desc, res, main_only, xs, sink, pans)
    chk = {'annT_after_rpt': int(d.annT_after_rpt.sum()), 'ann1_not_after': int((d.has_act1 & ~d.ann1_after).sum())}
    print('目标年度核对：', chk, flush=True)
    print(fn.to_string(index=False), flush=True)
    return d, w, res
