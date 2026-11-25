"""比较组：研报层面的 PMAFE、排名分数 Score、同行时滞 PeerLag 和同行明细；公司-券商层面的被评价预测、PMAFE 和时效差 DAGE。
全部是原始值（不缩尾），输入是 forecasts.py 的预测长表和每篇研报的本篇预测。两个层面都按预测窗口 h（HORIZONS = 1、2、3，
即第 h 年预测 PROFIT_Th、目标年度 = RPT_YR_T + h）分别计算，列名带后缀 _Hh；研究层只用其中的主窗口（见
src/studies/forecast_accuracy/sample.py 的 MAIN_H）。

研报层面（只对基础样本 base 的研报计算，其余研报这些列缺失）
  目标年度    研报 i（公司 f、券商 o、发布日 d）在窗口 h 的目标年度 TY_Hh = T + h（T = 本篇预测行的 RPT_YR_T），逐篇不同：
              同一窗口里 T = 2022 和 T = 2023 的研报目标年度相差一年。
  比较组      研报 i 在窗口 h 的同行 = 其他券商（ORG_CL ≠ o）对公司 f、同一目标年度 TY_Hh 发布日在 [d − back, d + fwd]
              （含两端，自然日）内的预测，每家券商取窗口内最新的一条（预测长表已按公司、目标年度、券商、RPT_DT、CREAT_TM、SEQ
              排序，取每家最后一条）。同行用预测长表的 ty = RPT_YR_T + k 去对，不管同行自己是第几年预测（k 可以不等于 h）。
              两种日期窗口（WINDOWS）：'' = [d − 90, d]（主口径）；'W30' = [d − 90, d + 30]（向后延伸 30 天，同行可能晚于研报）。
              同行数 N_PEERS = 选出的其他券商家数（不含本篇）。
  PMAFE       同行至少 MIN_PEERS = 3 家、本篇绝对误差 AFE = ABSERR_Hh = |F − A| 非缺失、实际值 A = A_Hh 有限时：
              MAFE = (各同行 |F_j − A| 与本篇 AFE) 的平均（MAFE 含本篇，分母 = 同行数 + 1），PMAFE = (AFE − MAFE) / MAFE；MAFE = 0 时缺失。
              PMAFE_Hh 用 [d − 90, d]，PMAFEW30_Hh 用 [d − 90, d + 30]。
  Score       [d − 90, d] 比较组上的 Hong-Kubik 排名分数 SCORE_Hh：本篇 AFE 非缺失、同行 ≥ 3 家时，
              Score = 100 × (1 − (同行中误差比本篇小的家数 + 0.5 × 误差与本篇相等的家数) / 同行数)，
              即 100 − (r − 1)/(N − 1) × 100（r 为本篇在含本篇的组里的名次，误差最小为 1，并列取平均名次；N = 同行数 + 1）。越大越准。
  PeerLag     同行时滞：比较组内各同行 (d − 同行预测发布日) 的天数的平均值（'W30' 中可为负）；同行不足 3 家时缺失（与实际值是否
              缺失无关）。PEERLAG_Hh / PEERLAGW30_Hh 对应两种日期窗口；LAGMED_Hh = [d − 90, d] 的中位数；
              LNNPEERS_Hh = ln(N_PEERS_Hh)（同行不足 3 家时缺失）。
  同行明细    三个窗口 h 的 [d − 90, d] 比较组中每篇研报的每个同行一行（列 h、ty 标明窗口和目标年度；研报的同行数可以不足 3 家）。

公司-券商层面（每个（公司，券商）一行，每个窗口 h 一组列）
  被评价预测  券商 o 在 PAIR_YEAR（2024）年内（RPT_DT 在该年）对公司 f 发布的、第 h 年预测 PROFIT_Th 非缺失的最后一条预测行
              （按 RPT_DT、CREAT_TM、SEQ 稳定排序取最后一条），不一定出自深度研报。EVAL_T_Hh = 该行的 RPT_YR_T，目标年度
              TY_Hh = EVAL_T_Hh + h（逐对不同），预测值 EVAL_F_Hh = 该行的 PROFIT_Th，AFE = |EVAL_F − A_TY|。
  比较组      PAIR_YEAR 年内对公司 f、目标年度 TY_Hh 发布过预测的全部券商（预测长表口径，不管是第几年预测），每家取当年对 TY_Hh
              的最后一条；本券商在组里的那一条用被评价预测本身。被评价预测通常就是本券商当年对 TY_Hh 的最后一条，不是时记
              EVAL_LAST_TY_Hh = False（accuracy 步骤运行时打印这种对数）。
              至少 MIN_GROUP = 4 家且组内各家 AFE 都有限时：MAFE = 组内 AFE 的平均（含本券商），PMAFE = (AFE − MAFE) / MAFE（MAFE = 0
              时缺失）；DAGE = 本券商的 AGE − 组内 AGE 的平均，AGE = 预测发布日到 TY_Hh 年 12 月 31 日的天数（DAGE 为正 = 比比较组更陈旧）。
  与研报的关系 n_reports = 研报范围里该对的研报篇数（只算匹配到本篇预测、有 ORG_CL 的研报），last_deep_dt = 其中最晚的发布日；
              EVAL_IS_DEEP_Hh = 窗口 h 的被评价预测正好是该对名下某篇研报本身的预测（RPT_ID 转大写比较）；
              EVAL_GAP_Hh = 窗口 h 的被评价预测的发布日 − last_deep_dt（天）。
"""
import numpy as np
import pandas as pd

from src.accuracy.forecasts import HORIZONS

MIN_PEERS = 3  # 研报层面比较组至少几家同行（不含本篇）
WINDOWS = {'': (90, 0), 'W30': (90, 30)}  # 日期窗口标签（列名里接在变量名后、_Hh 之前）: (向前天数, 向后天数)
MIN_GROUP = 4  # 公司-券商层面比较组至少几家券商（含本券商）
PAIR_YEAR = '2024'  # 公司-券商层面：被评价预测和比较组都取这一年内发布的预测
PAIR_SORT = ['SEC_CD', 'ORG_CL', 'RPT_DT', 'CREAT_TM', 'SEQ']  # 公司-券商层面取“最后一条”的排序


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


def code_ty_slices(lg):
    """预测长表（已按公司、目标年度、券商、时间排序，索引为 0..n−1）里每个（公司，目标年度）的切片
    {(code, ty): (日期 int 数组, 券商数组, 起始位置)}；位置是 lg 的行号。"""
    code_a, ty_a = lg.SEC_CD.to_numpy(), lg.ty.to_numpy()
    start = np.flatnonzero(np.r_[True, (code_a[1:] != code_a[:-1]) | (ty_a[1:] != ty_a[:-1])])
    stop = np.r_[start[1:], len(lg)]
    dt, org = lg.RPT_DT.astype(int).to_numpy(), lg.ORG_CL.to_numpy()
    return {(code_a[a], int(ty_a[a])): (dt[a:b], org[a:b], a) for a, b in zip(start, stop)}


def report_peers(rep, lg):
    """rep：基础样本的研报（report_id、code、ORG_CL、date，以及每个 h 的 TY_Hh、ABSERR_Hh、A_Hh）；lg：预测长表。
    返回 (每篇研报一行的比较组变量, 三个窗口 [d − 90, d] 比较组的同行明细)，见模块说明。"""
    assert lg.index.equals(pd.RangeIndex(len(lg)))
    slices = code_ty_slices(lg)
    f_lg, org_lg, rid_lg, dt_lg = lg.F.to_numpy(float), lg.ORG_CL.to_numpy(), lg.RPT_ID.to_numpy(), lg.RPT_DT.to_numpy()
    out = rep[['report_id']].copy()
    details = []
    for h in HORIZONS:
        tys = [int(t) if pd.notna(t) else None for t in rep[f'TY_H{h}']]
        for tag, (back, fwd) in WINDOWS.items():
            poss, pm, sc = [], [], []
            for code, ty, org, day, own, a in zip(rep.code, tys, rep.ORG_CL, rep.date, rep[f'ABSERR_H{h}'], rep[f'A_H{h}']):
                sl = slices.get((code, ty))
                p = pick_peers(sl, org, day, back, fwd) if sl is not None else np.array([], int)
                poss.append(p)
                v, s = np.nan, np.nan
                if pd.notna(own) and np.isfinite(a) and len(p) >= MIN_PEERS:
                    pa = np.abs(f_lg[p] - a)
                    mafe = np.r_[pa, own].mean()
                    v = (own - mafe) / mafe if mafe > 0 else np.nan
                    rank = ((pa < own).sum() + 0.5 * (pa == own).sum()) / len(pa)
                    s = 100 * (1 - rank)
                pm.append(v)
                sc.append(s)
            lens = np.array([len(p) for p in poss], int)
            pos = np.concatenate(poss).astype(int) if len(poss) else np.array([], int)
            pl = pd.DataFrame({'report_id': np.repeat(rep.report_id.to_numpy(), lens), 'h': h,
                               'ty': np.repeat(np.array([t if t is not None else -1 for t in tys]), lens),
                               'report_dt': np.repeat(rep.date.to_numpy(), lens), 'peer_ORG_CL': org_lg[pos],
                               'peer_RPT_ID': rid_lg[pos], 'peer_dt': dt_lg[pos], 'F': f_lg[pos]})
            pl['LAG'] = (pd.to_datetime(pl.report_dt, format='%Y%m%d') - pd.to_datetime(pl.peer_dt, format='%Y%m%d')).dt.days
            lag = pl.groupby('report_id').LAG
            n = pd.Series(lens, index=rep.index)
            ok = n >= MIN_PEERS
            out[f'PMAFE{tag}_H{h}'] = pd.Series(pm, index=rep.index, dtype=float).replace([np.inf, -np.inf], np.nan)
            if tag == '':
                out[f'SCORE_H{h}'] = pd.Series(sc, index=rep.index, dtype=float)
            out[f'N_PEERS{tag}_H{h}'] = n
            out[f'PEERLAG{tag}_H{h}'] = rep.report_id.map(lag.mean()).where(ok)
            if tag == '':
                out[f'LAGMED_H{h}'] = rep.report_id.map(lag.median()).where(ok)
                out[f'LNNPEERS_H{h}'] = np.log(n.where(ok).astype(float))
                details.append(pl[['report_id', 'h', 'ty', 'peer_ORG_CL', 'peer_RPT_ID', 'peer_dt', 'LAG', 'F']])
    return out, pd.concat(details, ignore_index=True)


# ---------------------------------------------------------------- 公司-券商层面
def pair_eval(lg, h, codes, year=PAIR_YEAR):
    """窗口 h 的被评价预测：year 年内发布、第 h 年预测非缺失（预测长表里 k == h）的预测，每个（公司，券商）取最后一条
    （按 PAIR_SORT 稳定排序）。保留 lg 的行号作索引（核对“是不是该券商对目标年度的最后一条”时用）。"""
    g = lg[(lg.k == h) & (lg.RPT_DT.str[:4] == year) & lg.SEC_CD.isin(codes)]
    return g.sort_values(PAIR_SORT, kind='mergesort').groupby(['SEC_CD', 'ORG_CL']).tail(1)


def last_by_ty(lg, codes, year=PAIR_YEAR):
    """year 年内发布的预测，每个（公司，目标年度，券商）取最后一条（预测长表已按公司、目标年度、券商、时间排序；不管第几年预测）。
    保留 lg 的行号作索引。"""
    g = lg[(lg.RPT_DT.str[:4] == year) & lg.SEC_CD.isin(codes)]
    return g.groupby(['SEC_CD', 'ty', 'ORG_CL']).tail(1)


def pair_forecasts(pairs, lg, ann, h, codes):
    """每个（code，ORG_CL）对在窗口 h 的被评价预测和比较组变量（见模块说明），列名带后缀 _Hh。"""
    ev = pair_eval(lg, h, codes)
    grp = last_by_ty(lg, codes)
    a_lut = ann.A.to_dict()
    grp = grp.assign(A=[a_lut.get((c, int(t)), np.nan) for c, t in zip(grp.SEC_CD, grp.ty)])
    grp['AFE'] = (grp.F - grp.A).abs()
    grp['AGE'] = (pd.to_datetime(grp.ty.astype(int).astype(str) + '1231', format='%Y%m%d')
                  - pd.to_datetime(grp.RPT_DT, format='%Y%m%d')).dt.days
    by = {k: (g.ORG_CL.to_numpy(), g.AFE.to_numpy(float), g.AGE.to_numpy(), g.index.to_numpy())
          for k, g in grp.groupby(['SEC_CD', 'ty'])}
    ev_by = {(c, o): r for c, o, r in zip(ev.SEC_CD, ev.ORG_CL, ev.itertuples())}
    names = ['EVAL_RPT_ID', 'EVAL_DT', 'EVAL_F', 'EVAL_T', 'TY', 'A', 'AFE', 'N_GROUP', 'AGE', 'MAFE', 'PMAFE', 'DAGE', 'EVAL_LAST_TY']
    sfx = lambda c: f'{c}_H{h}'  # noqa: E731
    rows = []
    for code, org in zip(pairs.code, pairs.ORG_CL):
        rec = {'code': code, 'ORG_CL': org}
        r = ev_by.get((code, org))
        if r is None:
            rows.append(rec)
            continue
        ty = int(r.RPT_YR_T) + h
        a = a_lut.get((code, ty), np.nan)
        afe = abs(r.F - a)
        age = (pd.Timestamp(f'{ty}1231') - pd.Timestamp(r.RPT_DT)).days
        orgs, e, ages, idx = by[(code, ty)]  # 被评价预测本身就是一条 year 年内对 ty 的预测，所以这一组一定存在、含本券商
        i = int(np.flatnonzero(orgs == org)[0])
        e, ages = e.copy(), ages.copy()
        e[i], ages[i] = afe, age  # 本券商用被评价预测本身
        n = len(e)
        rec.update({sfx('EVAL_RPT_ID'): r.RPT_ID, sfx('EVAL_DT'): r.RPT_DT, sfx('EVAL_F'): r.F, sfx('EVAL_T'): r.RPT_YR_T,
                    sfx('TY'): ty, sfx('A'): a, sfx('AFE'): afe, sfx('N_GROUP'): n, sfx('AGE'): age,
                    sfx('EVAL_LAST_TY'): bool(idx[i] == r.Index)})
        if n >= MIN_GROUP and np.isfinite(e).all():
            mafe = e.mean()
            rec.update({sfx('MAFE'): mafe, sfx('PMAFE'): (afe - mafe) / mafe if mafe > 0 else np.nan, sfx('DAGE'): age - ages.mean()})
        rows.append(rec)
    out = pd.DataFrame(rows, columns=['code', 'ORG_CL'] + [sfx(c) for c in names])
    out[sfx('TY')] = out[sfx('TY')].astype('Int64')
    out[sfx('N_GROUP')] = out[sfx('N_GROUP')].astype('Int64')
    out[sfx('EVAL_LAST_TY')] = out[sfx('EVAL_LAST_TY')].astype('boolean')
    return out


def pair_table(reports, lg, ann):
    """reports：研报表（report_id、code、ORG_CL、date）。范围 = 有 ORG_CL 的研报出现过的全部（code，ORG_CL）对，按 code、ORG_CL 排序。
    返回每对一行：n_reports、last_deep_dt，以及每个窗口 h 的被评价预测和比较组变量（pair_forecasts）、EVAL_IS_DEEP_Hh、EVAL_GAP_Hh。"""
    s = reports[reports.ORG_CL.notna()]
    P = s.groupby(['code', 'ORG_CL']).agg(n_reports=('report_id', 'size'), last_deep_dt=('date', 'max')).reset_index()
    ids = s.groupby(['code', 'ORG_CL']).report_id.agg(lambda x: set(x.str.upper()))
    own = [ids[(c, o)] for c, o in zip(P.code, P.ORG_CL)]
    codes = sorted(set(reports.code))
    for h in HORIZONS:
        P = P.merge(pair_forecasts(P[['code', 'ORG_CL']], lg, ann, h, codes), on=['code', 'ORG_CL'], how='left', validate='1:1')
        P[f'EVAL_IS_DEEP_H{h}'] = [isinstance(e, str) and e.upper() in i for e, i in zip(P[f'EVAL_RPT_ID_H{h}'], own)]
        P[f'EVAL_GAP_H{h}'] = (pd.to_datetime(P[f'EVAL_DT_H{h}'], format='%Y%m%d')
                               - pd.to_datetime(P.last_deep_dt, format='%Y%m%d')).dt.days
    return P
