"""比较组：研报层面的 PMAFE、排名分数 Score、同行时滞 PeerLag 和同行明细；公司-券商层面的被评价预测、PMAFE 和时效差 DAGE。
全部是原始值（不缩尾），输入是 forecasts.py 的预测长表和每篇研报的本篇预测。

研报层面（只对基础样本 base 的研报计算，其余研报这些列缺失）
  比较组      研报 i（公司 f、券商 o、发布日 d）的同行 = 其他券商（ORG_CL ≠ o）对公司 f、目标年度 ty 发布日在 [d − back, d + fwd]
              （含两端，自然日）内的预测，每家券商取窗口内最新的一条（预测长表已按券商、RPT_DT、CREAT_TM、SEQ 排序，取每家最后一条）。
              三个窗口（WINDOWS）：'25' = FY2025、[d − 90, d]（主口径）；'24' = FY2024、[d − 90, d]；'W30' = FY2025、[d − 90, d + 30]
              （窗口向后延伸 30 天，同行可能晚于研报）。同行数 N_PEERS = 选出的其他券商家数（不含本篇）。
  PMAFE       同行至少 MIN_PEERS = 3 家、本篇绝对误差 AFE = |F − A| 非缺失、实际值 A 有限时：
              MAFE = (各同行 |F_j − A| 与本篇 AFE) 的平均（MAFE 含本篇，分母 = 同行数 + 1），PMAFE = (AFE − MAFE) / MAFE；MAFE = 0 时缺失。
              PMAFE_FY25、PMAFE_FY24、PMAFEW30_FY25 分别用窗口 '25'、'24'、'W30'。
  Score       窗口 '25' 的比较组上的 Hong-Kubik 排名分数：本篇 AFE 非缺失、同行 ≥ 3 家时，
              Score = 100 × (1 − (同行中误差比本篇小的家数 + 0.5 × 误差与本篇相等的家数) / 同行数)，
              即 100 − (r − 1)/(N − 1) × 100（r 为本篇在含本篇的组里的名次，误差最小为 1，并列取平均名次；N = 同行数 + 1）。越大越准。
  PeerLag     同行时滞：比较组内各同行 (d − 同行预测发布日) 的天数的平均值（窗口 'W30' 中可为负）；同行不足 3 家时缺失。
              PEERLAG25 / 24 / W30 对应三个窗口；LAGMED25 = 窗口 '25' 的中位数；LNNPEERS25 = ln(N_PEERS25)（同行不足 3 家时缺失）。
  同行明细    窗口 '25' 中每篇研报的每个同行一行（研报的同行数可以不足 3 家）。

公司-券商层面（每个（公司，券商）一行）
  被评价预测  券商 o 在 PAIR_YEAR（2024）年内（RPT_DT 在该年）对公司 f 目标年度 ty 归母净利润的最后一条预测（预测长表的顺序：
              RPT_DT、CREAT_TM、SEQ 最大的那条），不一定出自深度研报。AFE = |F − A_ty|。
  比较组      PAIR_YEAR 年内对公司 f 目标年度 ty 做过预测的全部券商，每家取当年最后一条（含本券商）；同一家公司的所有券商共用一个比较组。
              至少 MIN_GROUP = 4 家且组内各家 AFE 都有限时：MAFE = 组内 AFE 的平均（含本券商），PMAFE = (AFE − MAFE) / MAFE（MAFE = 0 时缺失）；
              DAGE = 本券商的 AGE − 组内 AGE 的平均，AGE = 预测发布日到 ty 年 12 月 31 日的天数（DAGE 为正 = 比比较组更陈旧）。
  与研报的关系 n_reports = 研报范围里该对的研报篇数（只算匹配到本篇预测、有 ORG_CL 的研报），last_deep_dt = 其中最晚的发布日；
              EVAL_IS_DEEP = 被评价的 FY2025 预测正好是该对名下某篇研报本身的预测（RPT_ID 转大写比较）；
              EVAL_GAP = 被评价的 FY2025 预测的发布日 − last_deep_dt（天）。
"""
import numpy as np
import pandas as pd

MIN_PEERS = 3  # 研报层面比较组至少几家同行（不含本篇）
WINDOWS = {'25': (2025, 90, 0), '24': (2024, 90, 0), 'W30': (2025, 90, 30)}  # 标签: (目标年度, 向前天数, 向后天数)
MIN_GROUP = 4  # 公司-券商层面比较组至少几家券商（含本券商）
PAIR_YEAR = '2024'  # 公司-券商层面：被评价预测和比较组都取这一年内发布的预测
PAIR_TY = 2025  # 公司-券商层面的目标年度


# ---------------------------------------------------------------- 研报层面
def pick_peers(sl, org, day, back, fwd):
    """sl = (日期 int 数组, 券商数组, 起始位置)；其他券商在 [day − back, day + fwd] 内每家最新一条的位置（back=None 不设下界）。"""
    dt, o, a0 = sl
    ts = pd.Timestamp(day)
    hi = int((ts + pd.Timedelta(days=fwd)).strftime('%Y%m%d'))
    m = (o != org) & (dt <= hi)
    if back is not None:
        m &= dt >= int((ts - pd.Timedelta(days=back)).strftime('%Y%m%d'))
    idx = np.flatnonzero(m)
    if not len(idx):
        return idx
    oo = o[idx]
    return a0 + idx[np.r_[oo[1:] != oo[:-1], True]]  # 同一券商的最后一条（切片已按券商、时间排序）


def ty_slices(lg, ty):
    """预测长表里目标年度 ty 的部分（仍按公司、券商、时间排序）和每家公司的切片 {code: (日期 int 数组, 券商数组, 起始位置)}。"""
    g = lg[lg.ty == ty].reset_index(drop=True)
    code_a = g.SEC_CD.to_numpy()
    start = np.flatnonzero(np.r_[True, code_a[1:] != code_a[:-1]])
    stop = np.r_[start[1:], len(g)]
    dt, org = g.RPT_DT.astype(int).to_numpy(), g.ORG_CL.to_numpy()
    return g, {code_a[a]: (dt[a:b], org[a:b], a) for a, b in zip(start, stop)}


def report_peers(rep, lg):
    """rep：基础样本的研报（report_id、code、ORG_CL、date、ABSERR25、ABSERR24、A25、A24）；lg：预测长表。
    返回 (每篇研报一行的比较组变量, 窗口 '25' 的同行明细)，见模块说明。"""
    out = rep[['report_id']].copy()
    detail = None
    for tag, (ty, back, fwd) in WINDOWS.items():
        g, slices = ty_slices(lg, ty)
        f_g = g.F.to_numpy(float)
        own_col, a_col = f'ABSERR{ty % 100}', f'A{ty % 100}'
        poss, pm, sc = [], [], []
        for code, org, day, own, a in zip(rep.code, rep.ORG_CL, rep.date, rep[own_col], rep[a_col]):
            p = pick_peers(slices[code], org, day, back, fwd) if code in slices else np.array([], int)
            poss.append(p)
            v, s = np.nan, np.nan
            if pd.notna(own) and np.isfinite(a) and code in slices and len(p) >= MIN_PEERS:
                pa = np.abs(f_g[p] - a)
                mafe = np.r_[pa, own].mean()
                v = (own - mafe) / mafe if mafe > 0 else np.nan
                rank = ((pa < own).sum() + 0.5 * (pa == own).sum()) / len(pa)
                s = 100 * (1 - rank)
            pm.append(v)
            sc.append(s)
        lens = np.array([len(p) for p in poss], int)
        pos = np.concatenate(poss).astype(int) if len(poss) else np.array([], int)
        pl = pd.DataFrame({'report_id': np.repeat(rep.report_id.to_numpy(), lens), 'report_dt': np.repeat(rep.date.to_numpy(), lens),
                           'peer_ORG_CL': g.ORG_CL.to_numpy()[pos], 'peer_RPT_ID': g.RPT_ID.to_numpy()[pos],
                           'peer_dt': g.RPT_DT.to_numpy()[pos], 'F': f_g[pos]})
        pl['LAG'] = (pd.to_datetime(pl.report_dt, format='%Y%m%d') - pd.to_datetime(pl.peer_dt, format='%Y%m%d')).dt.days
        lag = pl.groupby('report_id').LAG
        n = pd.Series(lens, index=rep.index)
        ok = n >= MIN_PEERS
        y = 'PMAFEW30_FY25' if tag == 'W30' else f'PMAFE_FY{tag}'
        out[y] = pd.Series(pm, index=rep.index, dtype=float).replace([np.inf, -np.inf], np.nan)
        if tag == '25':
            out['SCORE_FY25'] = pd.Series(sc, index=rep.index, dtype=float)
        out[f'N_PEERS{tag}'] = n
        out[f'PEERLAG{tag}'] = rep.report_id.map(lag.mean()).where(ok)
        if tag == '25':
            out['LAGMED25'] = rep.report_id.map(lag.median()).where(ok)
            out['LNNPEERS25'] = np.log(n.where(ok).astype(float))
            detail = pl.drop(columns='report_dt')[['report_id', 'peer_ORG_CL', 'peer_RPT_ID', 'peer_dt', 'LAG', 'F']]
    return out, detail


# ---------------------------------------------------------------- 公司-券商层面
def last_in_year(lg, ty, codes, year=PAIR_YEAR):
    """year 年内发布的、目标年度为 ty 的预测，每个（公司，券商）取最后一条（预测长表已按公司、目标年度、券商、时间排序）。"""
    g = lg[(lg.ty == ty) & (lg.RPT_DT.str[:4] == year) & lg.SEC_CD.isin(codes)]
    return g.groupby(['SEC_CD', 'ORG_CL']).tail(1).reset_index(drop=True)


def pair_forecasts(pairs, grp, ann, ty=PAIR_TY):
    """每个（code，ORG_CL）对的被评价预测和比较组变量；grp = last_in_year(...)。列名带目标年度的后两位（如 PMAFE_FY25、DAGE25）。"""
    tag = str(ty)[2:]
    fyend = pd.Timestamp(f'{ty}1231')
    grp = grp.copy()
    a_ty = ann.A.xs(ty, level='fy').to_dict()
    grp['A'] = [a_ty.get(c, np.nan) for c in grp.SEC_CD]
    grp['AFE'] = (grp.F - grp.A).abs()
    grp['AGE'] = (fyend - pd.to_datetime(grp.RPT_DT, format='%Y%m%d')).dt.days
    by = {c: (g.ORG_CL.to_numpy(), g.AFE.to_numpy(float), g.AGE.to_numpy(), g.RPT_ID.to_numpy(), g.RPT_DT.to_numpy(), g.F.to_numpy(),
              g.RPT_YR_T.to_numpy(), g.A.to_numpy()) for c, g in grp.groupby('SEC_CD')}
    cols = ['code', 'ORG_CL'] + [f'{c}{tag}' for c in ('EVAL_RPT_ID', 'EVAL_DT', 'EVAL_F', 'EVAL_T', 'A', 'AFE', 'N_GROUP', 'AGE', 'MAFE')] \
        + [f'PMAFE_FY{tag}', f'DAGE{tag}']
    rows = []
    for code, org in zip(pairs.code, pairs.ORG_CL):
        rec = {'code': code, 'ORG_CL': org}
        if code not in by or not (by[code][0] == org).any():
            rows.append(rec)
            continue
        orgs, e, ages, rids, dts, fs, ts, aa = by[code]
        i = int(np.flatnonzero(orgs == org)[0])
        a, n = e[i], len(e)
        rec.update({f'EVAL_RPT_ID{tag}': rids[i], f'EVAL_DT{tag}': dts[i], f'EVAL_F{tag}': fs[i], f'EVAL_T{tag}': ts[i],
                    f'A{tag}': aa[i], f'AFE{tag}': a, f'N_GROUP{tag}': n, f'AGE{tag}': ages[i]})
        if n >= MIN_GROUP and np.isfinite(e).all():
            mafe = e.mean()
            rec.update({f'MAFE{tag}': mafe, f'PMAFE_FY{tag}': (a - mafe) / mafe if mafe > 0 else np.nan,
                        f'DAGE{tag}': ages[i] - ages.mean()})
        rows.append(rec)
    return pd.DataFrame(rows, columns=cols)


def pair_table(reports, lg, ann):
    """reports：研报表（report_id、code、ORG_CL、date）。范围 = 有 ORG_CL 的研报出现过的全部（code，ORG_CL）对，按 code、ORG_CL 排序。
    返回每对一行：n_reports、last_deep_dt、FY2025 的被评价预测和比较组变量（pair_forecasts）、EVAL_IS_DEEP、EVAL_GAP。"""
    s = reports[reports.ORG_CL.notna()]
    P = s.groupby(['code', 'ORG_CL']).agg(n_reports=('report_id', 'size'), last_deep_dt=('date', 'max')).reset_index()
    ids = s.groupby(['code', 'ORG_CL']).report_id.agg(lambda x: set(x.str.upper()))
    codes = sorted(set(reports.code))
    Y = pair_forecasts(P[['code', 'ORG_CL']], last_in_year(lg, PAIR_TY, codes), ann, PAIR_TY)
    P = P.merge(Y, on=['code', 'ORG_CL'], how='left', validate='1:1')
    own = [ids[(c, o)] for c, o in zip(P.code, P.ORG_CL)]
    P['EVAL_IS_DEEP'] = [isinstance(e, str) and e.upper() in i for e, i in zip(P.EVAL_RPT_ID25, own)]
    P['EVAL_GAP'] = (pd.to_datetime(P.EVAL_DT25, format='%Y%m%d') - pd.to_datetime(P.last_deep_dt, format='%Y%m%d')).dt.days
    return P
