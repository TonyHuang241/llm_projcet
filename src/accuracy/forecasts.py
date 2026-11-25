"""分析师预测和实际值：读 data/EST_COMP_TN 的有效预测、financial_data 的年报实际值，整理成预测长表；匹配每篇研报本身的预测。

口径
  预测行      data/EST_COMP_TN 下 <年>/<年>-<月>.parquet（只读规范文件名；目录里可能有带时间戳或 (1) 的备份文件，不能一起读），
              只用 IS_VLD == 1、SEC_CD 在研报范围的公司里的行；RPT_DT 转成 YYYYMMDD 字符串。
              第 k 年预测（PROFIT_Tk，k = 1、2、3，归母净利润）的目标年度 = RPT_YR_T + k。
  预测长表    LONG_YEARS 各年文件的全部预测行，一行一个（预测行 × k），去掉预测值缺失的；按公司、目标年度、券商、RPT_DT、CREAT_TM、
              SEQ 稳定排序，所以同一券商同一天有多条时排在最后的是 CREAT_TM、SEQ 最大的那条（比较组“每家取最新一条”取的就是它）。
  本篇预测    RPT_ID 转大写后等于 report_id 转大写、SEC_CD 等于研报公司代码的那一行（只在 OWN_YEARS 各年文件里找）；
              同一（RPT_ID，公司）有多条时按 RPT_DT、CREAT_TM、SEQ 稳定排序取最后一条（现有数据里没有这种重复，排序只为结果不依赖
              文件顺序）。匹配到的行给出 ORG_CL（券商编号）和 RPT_YR_T（记为 T）。
  预测窗口    H1、H2、H3（h = 1、2、3）：本篇预测行的 PROFIT_Th 是对 T + h 年的预测，目标年度 TY_Hh = T + h 逐篇不同
              （T = 2022 的研报 H1、H2、H3 = FY2023、FY2024、FY2025；T = 2023 的是 FY2024、FY2025、FY2026）。
              F_Hh = PROFIT_Th，A_Hh = 该公司 TY_Hh 财年的实际值（没有就缺失），ABSERR_Hh = |F_Hh − A_Hh|。
  实际值      financial_data 里 report_date = <财年>1231 的 np_parent_company_owners_ttm（归母净利润），公告日 = date；
              同一（公司，report_date）有多行时取最后一行。另取总资产 TA、净资产 EQ = 总资产 − 总负债（控制变量用）。
  基础样本    base = 匹配到本篇预测 & 有 H1（T+1 年）的预测值 F_H1 和实际值 A_H1 & 研报日早于 T+1 年年报的公告日（字符串比较）。
              逐篇可判定，与研报范围无关。
  预测期      LNHORIZON_Hh = ln(TY_Hh 年报公告日 − 研报日 的天数)，天数 ≤ 0 或公告日缺失时缺失
              （研究层 src/studies/forecast_accuracy/regressions.py 的 y_sample 按 Y 的窗口取这一列）。
"""
import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import EST_DIR, FINANCIAL

OWN_YEARS = range(2018, 2025)  # 匹配本篇预测时读这些年份的文件
LONG_YEARS = range(2018, 2026)  # 预测长表（比较组）读这些年份的文件
EST_COLS = ['SEQ', 'CREAT_TM', 'RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'PROFIT_T1', 'PROFIT_T2', 'PROFIT_T3']
LONG_KEYS = ['SEC_CD', 'ty', 'ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ']
HORIZONS = (1, 2, 3)  # 预测窗口 H1、H2、H3：第 h 年预测 PROFIT_Th，目标年度 = RPT_YR_T + h


def load_est(codes, years=LONG_YEARS, est_dir=EST_DIR):
    """若干年的预测表（只读 EST_COLS），只留 IS_VLD == 1 且 SEC_CD 在 codes 里的行；file_year = 行所在文件的年份。"""
    parts = []
    for y in years:
        for f in sorted(glob.glob(str(est_dir / str(y) / f'{y}-[0-9][0-9].parquet'))):
            t = pq.read_table(f, columns=EST_COLS, filters=[('IS_VLD', '=', 1), ('SEC_CD', 'in', list(codes))]).to_pandas()
            parts.append(t.assign(file_year=y))
    e = pd.concat(parts, ignore_index=True)
    e['RPT_DT'] = e.RPT_DT.dt.strftime('%Y%m%d')
    return e


def to_long(est):
    """预测长表：一行一个（预测行 × k），列 RPT_ID、SEC_CD、ORG_CL、RPT_DT、RPT_YR_T、CREAT_TM、SEQ、F（预测值）、k、ty（目标年度），
    去掉 F 缺失的行，按 LONG_KEYS 排序（多列排序是稳定排序）。"""
    keep = ['RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'CREAT_TM', 'SEQ']
    lg = pd.concat([est[keep + [f'PROFIT_T{k}']].rename(columns={f'PROFIT_T{k}': 'F'}).assign(k=k) for k in (1, 2, 3)],
                   ignore_index=True)
    lg['ty'] = lg.RPT_YR_T + lg.k
    return lg[lg.F.notna()].sort_values(LONG_KEYS).reset_index(drop=True)


def load_annual(codes, fin_path=FINANCIAL):
    """年报：(code, 财年) -> 归母净利润 A、公告日 ANN、总资产 TA、净资产 EQ。"""
    cols = ['report_date', 'code', 'date', 'total_assets', 'total_liability', 'np_parent_company_owners_ttm']
    f = pq.read_table(fin_path, columns=cols).to_pandas()
    f = f[f.code.isin(codes) & f.report_date.str.endswith('1231')].drop_duplicates(['code', 'report_date'], keep='last')
    return pd.DataFrame({'code': f.code, 'fy': f.report_date.str[:4].astype(int), 'A': f.np_parent_company_owners_ttm,
                         'ANN': f.date, 'TA': f.total_assets, 'EQ': f.total_assets - f.total_liability}).set_index(['code', 'fy'])


def match_own(reports, est):
    """每篇研报本身的预测行。reports：report_id、code、date；est：load_est 的结果（只用 OWN_YEARS 的文件）。
    返回 reports 的副本（行序不变），加 ORG_CL、RPT_YR_T、PROFIT_T1..3、matched。"""
    own = est[est.file_year.isin(list(OWN_YEARS))].rename(columns={'RPT_ID': 'RID', 'SEC_CD': 'code'})
    own['RID'] = own.RID.str.upper()
    own = own.sort_values(['RPT_DT', 'CREAT_TM', 'SEQ'], kind='mergesort')
    own = own.drop_duplicates(['RID', 'code'], keep='last')[['RID', 'code', 'ORG_CL', 'RPT_YR_T', 'PROFIT_T1', 'PROFIT_T2', 'PROFIT_T3']]
    d = reports[['report_id', 'code', 'date']].copy()
    d['RID'] = d.report_id.str.upper()
    d = d.merge(own, on=['RID', 'code'], how='left').drop(columns='RID')
    d['matched'] = d.RPT_YR_T.notna()
    return d


def report_forecasts(reports, est, ann):
    """每篇研报一行：本篇预测（match_own）、各窗口 h 的目标年度 TY_Hh、预测 F_Hh、实际值 A_Hh、年报公告日 ANN_Hh、绝对误差
    ABSERR_Hh、预测期 LNHORIZON_Hh，基础样本 base，以及控制变量要用的 T 年总资产 TA_T、净资产 EQ_T。行序同 reports。"""
    d = match_own(reports, est)
    d['T'] = d.RPT_YR_T.astype('Int64')
    lut = {c: ann[c].to_dict() for c in ann.columns}
    get = lambda col, fy: [lut[col].get((c, int(y)), np.nan) if pd.notna(y) else np.nan for c, y in zip(d.code, fy)]  # noqa: E731
    days = lambda a: (pd.to_datetime(a, format='%Y%m%d', errors='coerce') - pd.to_datetime(d.date, format='%Y%m%d')).dt.days  # noqa: E731
    for h in HORIZONS:
        ty = d['T'] + h
        d[f'TY_H{h}'] = ty
        d[f'F_H{h}'] = d[f'PROFIT_T{h}']
        d[f'A_H{h}'], d[f'ANN_H{h}'] = get('A', ty), get('ANN', ty)
        d[f'ABSERR_H{h}'] = (d[f'F_H{h}'] - d[f'A_H{h}']).abs()
        n = days(d[f'ANN_H{h}'])
        d[f'LNHORIZON_H{h}'] = np.log(n.where(n > 0))
    d['TA_T'], d['EQ_T'] = get('TA', d['T']), get('EQ', d['T'])
    has_act1 = d.matched & d.F_H1.notna() & d.A_H1.notna()
    d['base'] = has_act1 & (d.ANN_H1.fillna('') > d.date)
    return d
