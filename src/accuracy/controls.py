"""六个公司层面的控制变量在给定日期上的取值（研报层面用研报发布日，公司-券商层面用被评价预测的发布日）。

口径（与 Liu, Liu & Sui (2026) Table 10 相同的六个变量；这里全部是原始值：不缩尾，MOM、VOL、ROA_T 是小数、没有乘 100）
  前一交易日  给定日期之前（不含当天）最后一个市场交易日（市场日历 = 日行情里沪深 A 股出现过的全部日期）；个股取该日或之前最后一个
              未停牌交易日的值。未停牌 = 非停牌、收盘价 > 0、总市值非缺失且 > 0。日行情只读沪深 A 股（代码以 0、3、6 开头），
              从最早的给定日期往前 500 天读起；市场日历上前一交易日之前不足 249 个交易日时 MKCAP、MOM、VOL 都缺失。
  SIZE_M      ln(前一交易日总市值 MKCAP)。
  BM          T 年年报的净资产（总资产 − 总负债）÷ MKCAP。T = 所用预测行的 RPT_YR_T（研报层面是本篇预测，公司-券商层面是被评价预测）。
  MOM         前第 21 个交易日收盘价 ÷ 前第 250 个交易日收盘价 − 1（后复权收盘价；某日停牌取该日或之前最后一个未停牌收盘价）。
  VOL         前第 1–120 个交易日内未停牌日的日收益（相对上一个未停牌收盘价）的标准差（ddof = 1），不足 60 个日收益时缺失。
  ROA_T       T 年归母净利润 ÷ T 年末总资产。
  LNAGE       ln((给定日期 − 行情里首个未停牌交易日) 的天数 ÷ 30.4375)，月数 ≤ 0 时缺失。首个交易日在全部日行情里找（行情数据始于
              2010 年 1 月，此前上市的公司取同一上限）。
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.config import DAILY

CTRL = ['SIZE_M', 'BM', 'MOM', 'VOL', 'ROA_T', 'LNAGE']
LOOKBACK_DAYS = 500  # 日行情从最早的给定日期往前读多少个自然日
MIN_VOL_DAYS = 60  # VOL 至少几个日收益


def load_daily(start, end, daily_path=DAILY):
    """日行情按日期裁剪，只留沪深 A 股（代码 0、3、6 开头），按公司、日期排序；traded = 未停牌且有收盘价和总市值。"""
    cols = ['date', 'code', 'close', 'is_suspended', 'mkcap']
    d = pq.read_table(daily_path, columns=cols, filters=[('date', '>=', start), ('date', '<=', end)]).to_pandas()
    d = d[d.code.str[0].isin(['0', '3', '6'])].sort_values(['code', 'date']).reset_index(drop=True)
    d['traded'] = (~d.is_suspended) & (d.close > 0) & d.mkcap.notna() & (d.mkcap > 0)
    return d


def daily_returns(d):
    """未停牌交易日的个股日收益 ret（相对同一公司上一个未停牌交易日的收盘价；每家公司第一行缺失）。"""
    t = d[d.traded].copy()
    same = t.code.eq(t.code.shift())
    t['ret'] = np.where(same, t.close / t.close.shift() - 1, np.nan)
    return t


def market_vars(rows, daily_path=DAILY):
    """rows：key、code、date（YYYYMMDD）。每行：MKCAP（前一交易日总市值）、MOM、VOL、FIRST_DAY（行情里首个未停牌交易日）。"""
    codes = sorted(set(rows.code))
    start = (pd.Timestamp(rows.date.min()) - pd.Timedelta(days=LOOKBACK_DAYS)).strftime('%Y%m%d')
    d = load_daily(start, rows.date.max(), daily_path)
    cal = np.array(sorted(d.date.unique()))
    t = daily_returns(d[d.code.isin(codes)])
    by = {c: (g.date.to_numpy(), g.close.to_numpy(), g.mkcap.to_numpy(), g.ret.to_numpy()) for c, g in t.groupby('code')}
    first = pq.read_table(daily_path, columns=['code', 'date', 'close', 'is_suspended'], filters=[('code', 'in', codes)]).to_pandas()
    first = first[~first.is_suspended & (first.close > 0)].groupby('code').date.min()
    out = []
    for key, code, date in zip(rows.key, rows.code, rows.date):
        rec = {'key': key, 'FIRST_DAY': first.get(code)}
        i0 = np.searchsorted(cal, date, 'left') - 1  # 前一交易日在日历里的位置
        if code in by and i0 >= 249:
            dt, cl, mk, rt = by[code]
            at = lambda day: np.searchsorted(dt, day, 'right') - 1  # noqa: E731  该日或之前最后一个未停牌交易日
            j = at(cal[i0])
            if j >= 0:
                rec['MKCAP'] = mk[j]
            j21, j250 = at(cal[i0 - 20]), at(cal[i0 - 249])
            if j250 >= 0:
                rec['MOM'] = cl[j21] / cl[j250] - 1
            w = (dt >= cal[i0 - 119]) & (dt <= cal[i0]) & np.isfinite(rt)
            if w.sum() >= MIN_VOL_DAYS:
                rec['VOL'] = rt[w].std(ddof=1)
        out.append(rec)
    return pd.DataFrame(out, columns=['key', 'FIRST_DAY', 'MKCAP', 'MOM', 'VOL'])


def controls_at(rows, ann, daily_path=DAILY):
    """rows：key（唯一）、code、date、T（年报年度，可缺失）；ann：forecasts.load_annual 的结果。
    返回 key 和六个控制变量（CTRL），行序同 rows。"""
    mv = market_vars(rows[['key', 'code', 'date']], daily_path)
    o = rows[['key', 'code', 'date', 'T']].merge(mv, on='key', how='left', validate='1:1')
    look = lambda col: [ann[col].get((c, int(t)), np.nan) if pd.notna(t) else np.nan for c, t in zip(o.code, o['T'])]  # noqa: E731
    o['SIZE_M'] = np.log(o.MKCAP)
    o['BM'] = np.array(look('EQ'), float) / o.MKCAP
    o['ROA_T'] = np.array(look('A'), float) / np.array(look('TA'), float)
    age = (-(pd.to_datetime(o.FIRST_DAY, format='%Y%m%d', errors='coerce') - pd.to_datetime(o.date, format='%Y%m%d')).dt.days) / 30.4375
    o['LNAGE'] = np.log(age.where(age > 0))
    return o[['key'] + CTRL]
