"""实验：平均路径长度 PATHLEN 与研报发布前后 CAR，按本篇盈利预测相对同一券商上一次预测的上调/下调定方向。由根目录 experiment_pathlen_frev.py 调用。
结构照 src/report_car_fdir.py，只是把“预测高于其他券商 HIGH”换成“相对本券商上一次预测上调 UP”。

  方向      分析师盈利预测表 data/EST_COMP_TN（2023、2024 两年，IS_VLD == 1），转成长表（同 fdir 的 load_forecasts）
            cur      本篇（RPT_ID = report_id 转大写）本公司行（SEC_CD = 本公司代码）fy = 2024 的归母净利润预测
            prev     同 ORG_CL、同 SEC_CD、fy = 2024、RPT_DT 严格早于本篇且相差 ≤ 365 个自然日的各行中最近的一条（按 RPT_DT、SEQ 取最后），
                     gap = 相差天数
            REV = (cur − prev) / |prev|（prev = 0 记缺失）；UP = 1（REV > 0.001）/ 0（REV < −0.001），其余缺失；DIR = 2·UP − 1
            稳健性另算 UP_25（fy = 2025）和 UP_180（间隔上限 180 天）
  样本      UP 非缺失、且该公司有 2 篇及以上这样的研报；PATHLEN 在样本内标准化（PZ）
  signed    signed CAR = DIR × 未缩尾 CAR，再在样本内按 1%/99% 缩尾
  交互      CAR（事件面板里缩尾后的值）= b1·PZ + b2·PZ×UP + b3·UP (+ 公司固定效应)；b1+b2 把 PZ×UP 换成 PZ×(UP−1) 重新估计
  plain     同样本不乘方向、不加交互
  规格      none_s、firm（无控制变量），none_sk、firm_k（控制 LNCHARS_R、LNNODES_R）；缩尾和不缩尾都跑；按公司聚类，另存双向聚类 t
  稳健性    UP_25、UP_180 各自重建样本，只跑 firm 规格的 signed、interact、interact_sum、plain，行追加在 csv 末尾（dir_version 列区分）
"""
import time

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src import report_car_event as E
from src.report_car_fdir import load_forecasts
from src.report_car_rating import cellv, one

WIN6 = E.WIN6
CTRL_K = E.CTRL_K
SPECS = [('none_s', [], []), ('firm', ['code'], []), ('none_sk', [], CTRL_K), ('firm_k', ['code'], CTRL_K)]
FIRM_SPECS = ('firm_k', 'firm')
CTRL_SPECS = ('none_sk', 'firm_k')
LABEL = {'PZ': '平均路径长度', 'PZ_UP': '平均路径长度×上调预测', 'UP': '上调预测', 'LNCHARS_R': '研报字数', 'LNNODES_R': '研报概念数'}
TOL = 0.001
VERSIONS = {'UP': (2024, 365), 'UP_25': (2025, 365), 'UP_180': (2024, 180)}  # 方向版本：(预测年度, 间隔上限天数)
VERSION_NAME = {'UP': 'fy2024_365d', 'UP_25': 'fy2025_365d', 'UP_180': 'fy2024_180d'}
COLS = ['cur', 'prev', 'gap', 'REV', 'UP']


# ---------------------------------------------------------------- 方向
def direction(reports, long, fy, window):
    """逐篇：cur、prev、gap、REV、UP 和缺失原因。reports 需有 RID（大写 report_id）、code。"""
    own_all = long[long.RPT_ID.isin(reports.RID)]
    own_rows = own_all.merge(reports[['RID', 'code']], left_on='RPT_ID', right_on='RID')
    own_rows = own_rows[own_rows.SEC_CD == own_rows.code]
    has_own = set(own_rows.RPT_ID)
    cur_rows = own_rows[own_rows.fy == fy].sort_values(['RPT_DT', 'SEQ'])
    assert not cur_rows.RPT_ID.duplicated().any(), '同一研报本公司行有多条同年度预测'
    cur_map = cur_rows.set_index('RPT_ID')
    pool = {k: g for k, g in long[(long.fy == fy) & long.SEC_CD.isin(reports.code)].groupby(['SEC_CD', 'ORG_CL'])}
    out = []
    for rid, code in zip(reports.RID, reports.code):
        r = {'cur': np.nan, 'prev': np.nan, 'gap': np.nan, 'REV': np.nan, 'UP': np.nan, 'RPT_DT': pd.NaT, 'ORG_CL': np.nan,
             'RPT_YR_T': np.nan}
        if rid not in has_own:
            r['reason'] = '无本公司行'
        elif rid not in cur_map.index:
            r['reason'] = f'本公司行无{fy}年预测'
        else:
            c = cur_map.loc[rid]
            r.update(cur=c.profit, RPT_DT=c.RPT_DT, ORG_CL=c.ORG_CL, RPT_YR_T=c.RPT_YR_T)
            g = pool.get((code, c.ORG_CL))
            if g is not None:
                g = g[(g.RPT_DT < c.RPT_DT) & (g.RPT_DT >= c.RPT_DT - pd.Timedelta(days=window))]
            if g is None or g.empty:
                r['reason'] = f'{window}天内无上一次预测'
            else:
                p = g.sort_values(['RPT_DT', 'SEQ']).iloc[-1]
                r['prev'] = float(p.profit)
                r['gap'] = float((c.RPT_DT - p.RPT_DT).days)
                if r['prev'] == 0:
                    r['reason'] = '上一次预测为0'
                else:
                    r['REV'] = (r['cur'] - r['prev']) / abs(r['prev'])
                    if abs(r['REV']) <= TOL:
                        r['reason'] = '维持不变'
                    else:
                        r['UP'] = float(r['REV'] > TOL)
                        r['reason'] = ''
        out.append(r)
    return pd.DataFrame(out, index=reports.index)


def build_frev(panel, est_dir):
    """790 篇研报的方向表：主版本 UP（fy 2024、365 天）列全，稳健性版本另存 *_25、*_180 列。"""
    rep = panel[['report_id', 'code', 'date']].copy()
    rep['RID'] = rep.report_id.str.upper()
    long = load_forecasts(est_dir)
    out = rep[['report_id', 'code', 'date']].copy()
    for name, (fy, window) in VERSIONS.items():
        d = direction(rep, long, fy, window)
        if name == 'UP':
            out['ORG_CL'] = d.ORG_CL.astype('Int64')
            out['RPT_YR_T'] = d.RPT_YR_T.astype('Int64')
            for c in COLS:
                out[c] = d[c]
            out['DIR'] = 2 * out.UP - 1
            out['reason'] = d.reason
            assert (d.RPT_DT.dropna().dt.strftime('%Y%m%d') == rep.date[d.RPT_DT.notna()]).all(), 'RPT_DT 与研报日期不一致'
        else:
            sfx = name[len('UP'):]
            for c in COLS:
                out[f'{c}{sfx}'] = d[c]
            out[f'reason{sfx}'] = d.reason
    return out


# ---------------------------------------------------------------- 样本和回归
def sample(panel_w, panel_raw, frev, ucol='UP'):
    """UP 非缺失且公司有 2 篇及以上的研报；附上未缩尾和缩尾后的窗口 CAR、标准化的 PATHLEN 及交互项。"""
    keep = ['report_id', 'code', 'company', 'date', 'day0', 'sw', 'PATHLEN'] + CTRL_K
    d = panel_raw[keep + WIN6].merge(panel_w[['report_id'] + WIN6].add_suffix('_w').rename(columns={'report_id_w': 'report_id'}),
                                     on='report_id')
    sfx = ucol[len('UP'):]
    f = frev[['report_id', 'ORG_CL'] + [f'{c}{sfx}' for c in COLS[:-1]] + [ucol]]
    d = d.merge(f.rename(columns={f'{c}{sfx}': c for c in COLS[:-1]} | {ucol: 'UP'}), on='report_id')
    d = d[d.UP.notna()]
    d = d[d.code.map(d.code.value_counts()) >= 2].reset_index(drop=True)
    d['DIR'] = 2 * d.UP - 1
    d['PZ'] = (d.PATHLEN - d.PATHLEN.mean()) / d.PATHLEN.std()
    d['PZ_UP'] = d.PZ * d.UP
    d['PZ_UPm1'] = d.PZ * (d.UP - 1)  # 重新参数化：PZ 的系数 = b1 + b2
    for y in WIN6:
        d[f'S_{y}'] = d.DIR * d[y]  # signed CAR（未缩尾）
        d[f'S_{y}_w'] = R.winsor(d[f'S_{y}'])
    return d


def run_all(d, specs=SPECS, models=('signed', 'interact', 'interact_sum', 'plain')):
    rows = []
    for y in WIN6:
        for w, sfx in [(1, '_w'), (0, '')]:
            for spec, fe, ck in specs:
                if 'signed' in models:
                    rows += one(d, f'S_{y}{sfx}', ['PZ'], ck, fe, 'signed', spec, w, y)
                if 'interact' in models:
                    rows += one(d, f'{y}{sfx}', ['PZ', 'PZ_UP'], ['UP'] + ck, fe, 'interact', spec, w, y)
                if 'interact_sum' in models:
                    rows += [dict(r, term='PZ(b1+b2)') for r in one(d, f'{y}{sfx}', ['PZ'], ['PZ_UPm1', 'UP'] + ck, fe,
                                                                    'interact_sum', spec, w, y)]
                if 'plain' in models:
                    rows += one(d, f'{y}{sfx}', ['PZ'], ck, fe, 'plain', spec, w, y)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 表格
def table(res, model, spec, d, rows_spec, extra=None, note=''):
    g = res[(res.dir_version == VERSION_NAME['UP']) & (res.model == model) & (res.spec == spec) & (res.winsor == 1)]
    lines = ['| | ' + ' | '.join(E.Y_LABEL[y] for y in WIN6) + ' |', '|---|' + ':-:|' * 6,
             '| | ' + ' | '.join(f'({i})' for i in range(1, 7)) + ' |']
    row = lambda name, vals: lines.append(f'| {name} | ' + ' | '.join(vals) + ' |')  # noqa: E731
    first = {y: g[g.Y == y].iloc[0] for y in WIN6}
    for term in rows_spec:
        if term in ('PZ', 'PZ_UP'):
            cs = [cellv(*g[(g.Y == y) & (g.term == term)][['coef', 't', 'p']].iloc[0]) for y in WIN6]
        else:
            cs = [cellv(first[y][f'b_{term}'], first[y][f't_{term}'], first[y][f'p_{term}']) for y in WIN6]
        row(LABEL[term], [c[0] for c in cs])
        row('', [c[1] for c in cs])
    row('公司固定效应', ['Y' if spec in FIRM_SPECS else 'N'] * 6)
    row('样本量', [str(int(first[y].n)) for y in WIN6])
    row('公司数', [str(int(first[y].n_firms)) for y in WIN6])
    row('R²', [E.num(first[y].r2, 3) for y in WIN6])
    if extra is not None:
        e = res[(res.dir_version == VERSION_NAME['UP']) & (res.model == extra) & (res.spec == spec) & (res.winsor == 1)]
        cs = [cellv(*e[e.Y == y][['coef', 't', 'p']].iloc[0]) for y in WIN6]
        row('上调预测研报的路径长度效应（b1+b2）', [c[0] for c in cs])
        row('', [c[1] for c in cs])
    n1, n0 = int((d.UP == 1).sum()), int((d.UP == 0).sum())
    base = (f'样本为本篇研报（盈利预测表中本公司那一行）有 2024 年归母净利润预测、同一券商在发布前 365 个自然日内对同一公司有过 2024 年预测、'
            f'相对上一次预测有上调或下调、且所属公司有 2 篇及以上这样研报的研报（{len(d)} 篇、{d.code.nunique()} 家；'
            f'上调 {n1} 篇、下调 {n0} 篇），剔除了无本公司预测、365 天内无上一次预测（含首次覆盖）、维持不变（|REV| ≤ 0.1%）和只有 1 篇的公司；')
    fe_note = ('加入公司固定效应，R² 是包含公司虚拟变量的整体 R²（不是组内 R²）；' if spec in FIRM_SPECS else '未加固定效应，R² 是普通 OLS 的 R²；')
    ctrl_note = f'控制研报字数和研报概念数：{E.CTRL_NOTE}；' if spec in CTRL_SPECS else '未加控制变量；'
    lines += ['', f'注：{note}窗口以研报发布日（非交易日顺延到下一个交易日）为第 0 天；{base}平均路径长度 PATHLEN 已在样本内标准化；'
                  f'{ctrl_note}{fe_note}括号内为按公司聚类的 t 值；***、**、* 分别表示在 1%、5%、10% 水平上显著（按公司聚类的 p 值）。']
    return '\n'.join(lines) + '\n'


DEF_NOTE = ('方向按本篇对 2024 年归母净利润的预测相对同一券商上一次预测的修正确定：上一次预测指同一券商在发布前 365 个自然日内'
            '（严格早于发布日）对同一公司给出的最近一次 2024 年预测；修正幅度 REV = (本篇预测 − 上一次预测)/|上一次预测|；')


# ---------------------------------------------------------------- 主流程
def funnel(frev, panel_ids):
    f = frev[frev.report_id.isin(panel_ids)]
    n_up = f[f.UP.notna()]
    n_firm2 = n_up[n_up.code.map(n_up.code.value_counts()) >= 2]
    steps = [('事件面板', len(f)), ('有本公司行', int((f.reason != '无本公司行').sum())), ('有 cur', int(f.cur.notna().sum())),
             ('365 天内有上一次预测', int(f.prev.notna().sum())), ('上一次预测非 0', int(f.prev.notna().sum() - (f.prev == 0).sum())),
             ('非维持', len(n_up)), ('公司 ≥2 篇', len(n_firm2))]
    return steps


def main(out_base, est_dir, benches=('industry', 'market')):
    t0 = time.time()
    fd = out_base / 'frev'
    fd.mkdir(parents=True, exist_ok=True)
    panels = {}
    for bench in benches:
        src = out_base / E.out_name('PATHLEN', bench)
        panels[bench] = (pd.read_parquet(src / f'{E.out_name("PATHLEN", bench)}_panel.parquet'),
                         pd.read_parquet(src / f'{E.out_name("PATHLEN", bench)}_panel_raw.parquet'))
    ids = [set(p[1].report_id) for p in panels.values()]
    assert all(s == ids[0] for s in ids), '两个口径的事件面板研报不同'
    frev = build_frev(panels[benches[0]][1], est_dir)
    frev.to_parquet(fd / 'report_frev.parquet', index=False)
    frev.to_csv(fd / 'report_frev.csv', index=False, encoding='utf-8-sig')
    print(f'方向表：{fd}（{len(frev)} 篇，{time.time() - t0:.0f} 秒）', flush=True)
    print('缺失原因：' + '；'.join(f'{k or "有效"} {v}' for k, v in frev.reason.value_counts().items()), flush=True)
    print('漏斗：' + ' → '.join(f'{k} {v}' for k, v in funnel(frev, ids[0])), flush=True)
    for sfx in ('_25', '_180'):
        h = frev[frev[f'UP{sfx}'].notna()]
        print(f'  UP{sfx}：非缺失 {len(h)} 篇（上调 {int((h[f"UP{sfx}"] == 1).sum())}）；与主版本都非缺失的 '
              f'{int((h.UP.notna()).sum())} 篇里方向一致 {int((h.UP == h[f"UP{sfx}"]).sum())} 篇', flush=True)
    out = {}
    for bench in benches:
        pw, pr = panels[bench]
        d = sample(pw, pr, frev)
        g = d.groupby('code').UP.agg(['min', 'max', 'size'])
        both = g[(g['min'] == 0) & (g['max'] == 1)]
        print(f'[{bench}] 样本 {len(d)} 篇、{d.code.nunique()} 家；上调 {int((d.UP == 1).sum())}、下调 {int((d.UP == 0).sum())}；'
              f'公司内两个方向都有的 {len(both)} 家、{int(both["size"].sum())} 篇；gap 分位数（0/10/25/50/75/90/100）'
              f'{d.gap.quantile([0, .1, .25, .5, .75, .9, 1]).tolist()}', flush=True)
        mean = d.groupby('UP')[[f'{y}_w' for y in WIN6]].mean().T
        mean.index = [E.Y_LABEL[y] for y in WIN6]
        print(f'[{bench}] 缩尾后 CAR 按 UP 分组均值（百分点）：\n{mean.round(3).to_string()}', flush=True)
        res = run_all(d)
        res.insert(0, 'dir_version', VERSION_NAME['UP'])
        for ucol in ('UP_25', 'UP_180'):
            dr = sample(pw, pr, frev, ucol)
            print(f'[{bench}] 稳健性 {ucol}：样本 {len(dr)} 篇、{dr.code.nunique()} 家；上调 {int((dr.UP == 1).sum())}、'
                  f'下调 {int((dr.UP == 0).sum())}', flush=True)
            rr = run_all(dr, specs=[s for s in SPECS if s[0] == 'firm'])
            rr.insert(0, 'dir_version', VERSION_NAME[ucol])
            res = pd.concat([res, rr], ignore_index=True)
        res.insert(0, 'bench', bench)
        res.insert(0, 'X', 'PATHLEN')
        od = out_base / ('pathlen_frev_ind' if bench == 'industry' else 'pathlen_frev')
        td = od / 'tables'
        td.mkdir(parents=True, exist_ok=True)
        d.to_parquet(od / 'pathlen_frev_sample.parquet', index=False)
        res.to_csv(od / 'pathlen_frev_results.csv', index=False, encoding='utf-8-sig')
        bn = E.BENCH_NOTE[bench]
        sn = (f'因变量为 signed CAR = DIR × CAR，{DEF_NOTE}DIR 在 REV > 0.1%（上调预测）时取 +1、REV < −0.1%（下调预测）时取 −1，'
              f'CAR 为各窗口累积超额收益（{bn}，百分点），先乘方向再在样本内按 1%/99% 缩尾；')
        inote = (f'因变量为各窗口累积超额收益（{bn}，百分点，在 790 篇平衡样本上按 1%/99% 缩尾，同表 5、6）；{DEF_NOTE}'
                 f'上调预测 UP 在 REV > 0.1% 时取 1、REV < −0.1% 时取 0；平均路径长度×上调预测 = 标准化的 PATHLEN × UP；'
                 f'b1+b2 的 t 值用聚类协方差计算（把交互项换成 PATHLEN×(UP−1) 重新估计，PATHLEN 的系数即 b1+b2）；')
        tabs = [('t16_frev_signed_pooled.md', 'signed', 'none_s', ['PZ'], None, sn),
                ('t17_frev_signed_firm.md', 'signed', 'firm', ['PZ'], None, sn),
                ('t18_frev_interact_firm.md', 'interact', 'firm', ['PZ', 'PZ_UP', 'UP'], 'interact_sum', inote),
                ('t16k_frev_signed_pooled_ctrl.md', 'signed', 'none_sk', ['PZ'] + CTRL_K, None, sn),
                ('t17k_frev_signed_firm_ctrl.md', 'signed', 'firm_k', ['PZ'] + CTRL_K, None, sn),
                ('t18k_frev_interact_firm_ctrl.md', 'interact', 'firm_k', ['PZ', 'PZ_UP', 'UP'] + CTRL_K, 'interact_sum', inote)]
        for fn, model, spec, rows_spec, extra, note in tabs:
            (td / fn).write_text(table(res, model, spec, d, rows_spec, extra=extra, note=note), encoding='utf-8')
        out[bench] = (d, res)
        print(f'[{bench}] 完成：{od}（{len(res)} 行，{time.time() - t0:.0f} 秒）', flush=True)
    return frev, out
