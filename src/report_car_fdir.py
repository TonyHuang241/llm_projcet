"""实验：平均路径长度 PATHLEN 与研报发布前后 CAR，按本篇盈利预测相对其他券商的高低定方向。由根目录 experiment_pathlen_fdir.py 调用。
结构照 src/report_car_rating.py，只是把“最高档评级 TOP”换成“预测高于其他券商 HIGH”。

  方向      分析师盈利预测表 data/EST_COMP_TN（2023、2024 两年，IS_VLD == 1），转成长表：fy = RPT_YR_T + k、净利润 = PROFIT_Tk（k = 1,2,3），去缺失
            cur      本篇（RPT_ID = report_id 转大写）本公司行（SEC_CD = 本公司代码）fy = 2024 的归母净利润预测
            其他券商  同 SEC_CD、fy = 2024、ORG_CL ≠ 本篇、RPT_DT 严格早于本篇且相差 ≤ 90 个自然日；每家券商按 RPT_DT、SEQ 排序取最后一条
            要求家数 n_brk ≥ 3，med = 中位数；DEV = (cur − med) / |med|（med = 0 记缺失）
            HIGH = 1（DEV > 0.001）/ 0（DEV < −0.001），其余缺失；DIR = 2·HIGH − 1
            稳健性另算 HIGH_25（fy = 2025）和 HIGH_180（窗口 180 天）
  样本      HIGH 非缺失、且该公司有 2 篇及以上这样的研报；PATHLEN 在样本内标准化（PZ）
  signed    signed CAR = DIR × 未缩尾 CAR，再在样本内按 1%/99% 缩尾
  交互      CAR（事件面板里缩尾后的值）= b1·PZ + b2·PZ×HIGH + b3·HIGH (+ 公司固定效应)；b1+b2 把 PZ×HIGH 换成 PZ×(HIGH−1) 重新估计
  plain     同样本不乘方向、不加交互
  规格      none_s、firm（无控制变量），none_sk、firm_k（控制 LNCHARS_R、LNNODES_R）；缩尾和不缩尾都跑；按公司聚类，另存双向聚类 t
  稳健性    HIGH_25、HIGH_180 各自重建样本，只跑 firm 规格的 signed 和 interact_sum，行追加在 csv 末尾（dir_version 列区分）
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src import report_car_event as E
from src.report_car_rating import cellv, one

WIN6 = E.WIN6
CTRL_K = E.CTRL_K
SPECS = [('none_s', [], []), ('firm', ['code'], []), ('none_sk', [], CTRL_K), ('firm_k', ['code'], CTRL_K)]
FIRM_SPECS = ('firm_k', 'firm')
CTRL_SPECS = ('none_sk', 'firm_k')
LABEL = {'PZ': '平均路径长度', 'PZ_HIGH': '平均路径长度×预测高于其他券商', 'HIGH': '预测高于其他券商', 'LNCHARS_R': '研报字数',
         'LNNODES_R': '研报概念数'}
TOL = 0.001
MIN_BRK = 3
VERSIONS = {'HIGH': (2024, 90), 'HIGH_25': (2025, 90), 'HIGH_180': (2024, 180)}  # 方向版本：(预测年度, 窗口天数)
VERSION_NAME = {'HIGH': 'fy2024_90d', 'HIGH_25': 'fy2025_90d', 'HIGH_180': 'fy2024_180d'}
EST_COLS = ['SEQ', 'IS_VLD', 'RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'RPT_YR_T', 'PROFIT_T1', 'PROFIT_T2', 'PROFIT_T3']


# ---------------------------------------------------------------- 方向
def load_forecasts(est_dir, years=(2023, 2024)):
    """IS_VLD == 1 的预测转成长表：每行一篇研报 × 一只股票 × 一个预测年度 fy。"""
    # 只读规范文件名 YYYY-MM.parquet（同 src/dag_forecast.load_est；年份目录里若有备份文件不会被读进来）
    fs = sorted(f for y in years for f in (Path(est_dir) / str(y)).glob(f'{y}-[0-9][0-9].parquet'))
    e = pd.concat([pd.read_parquet(f, columns=EST_COLS) for f in fs], ignore_index=True)
    e = e[e.IS_VLD == 1]
    keys = ['RPT_ID', 'SEC_CD', 'ORG_CL', 'RPT_DT', 'SEQ', 'RPT_YR_T']
    long = pd.concat([e[keys].assign(fy=e.RPT_YR_T + k, profit=e[f'PROFIT_T{k}']) for k in (1, 2, 3)], ignore_index=True)
    return long.dropna(subset=['profit']).reset_index(drop=True)


def direction(reports, long, fy, window):
    """逐篇：cur、n_brk、med、DEV、HIGH 和缺失原因。reports 需有 RID（大写 report_id）、code。"""
    own_all = long[long.RPT_ID.isin(reports.RID)]
    own_rows = own_all.merge(reports[['RID', 'code']], left_on='RPT_ID', right_on='RID')
    own_rows = own_rows[own_rows.SEC_CD == own_rows.code]
    has_own = set(own_rows.RPT_ID)
    cur_rows = own_rows[own_rows.fy == fy].sort_values(['RPT_DT', 'SEQ'])
    assert not cur_rows.RPT_ID.duplicated().any(), '同一研报本公司行有多条同年度预测'
    cur_map = cur_rows.set_index('RPT_ID')
    pool = {c: g for c, g in long[(long.fy == fy) & long.SEC_CD.isin(reports.code)].groupby('SEC_CD')}
    out = []
    for rid, code in zip(reports.RID, reports.code):
        r = {'cur': np.nan, 'n_brk': 0, 'med': np.nan, 'DEV': np.nan, 'HIGH': np.nan, 'RPT_DT': pd.NaT, 'ORG_CL': np.nan,
             'RPT_YR_T': np.nan}
        if rid not in has_own:
            r['reason'] = '无本公司行'
        elif rid not in cur_map.index:
            r['reason'] = f'本公司行无{fy}年预测'
        else:
            c = cur_map.loc[rid]
            r.update(cur=c.profit, RPT_DT=c.RPT_DT, ORG_CL=c.ORG_CL, RPT_YR_T=c.RPT_YR_T)
            g = pool.get(code)
            if g is not None:
                g = g[(g.ORG_CL != c.ORG_CL) & (g.RPT_DT < c.RPT_DT) & (g.RPT_DT >= c.RPT_DT - pd.Timedelta(days=window))]
                g = g.sort_values(['RPT_DT', 'SEQ']).groupby('ORG_CL').tail(1)
                r['n_brk'] = len(g)
            if r['n_brk'] < MIN_BRK:
                r['reason'] = f'其他券商不足{MIN_BRK}家'
            else:
                r['med'] = float(g.profit.median())
                if r['med'] == 0:
                    r['reason'] = '中位数为0'
                else:
                    r['DEV'] = (r['cur'] - r['med']) / abs(r['med'])
                    if abs(r['DEV']) <= TOL:
                        r['reason'] = '与中位数持平'
                    else:
                        r['HIGH'] = float(r['DEV'] > TOL)
                        r['reason'] = ''
        out.append(r)
    return pd.DataFrame(out, index=reports.index)


def build_fdir(panel, est_dir):
    """790 篇研报的方向表：主版本 HIGH（fy 2024、90 天）列全，稳健性版本另存 *_25、*_180 列。"""
    rep = panel[['report_id', 'code', 'date']].copy()
    rep['RID'] = rep.report_id.str.upper()
    long = load_forecasts(est_dir)
    out = rep[['report_id', 'code', 'date']].copy()
    for name, (fy, window) in VERSIONS.items():
        d = direction(rep, long, fy, window)
        if name == 'HIGH':
            out['ORG_CL'] = d.ORG_CL.astype('Int64')
            out['RPT_YR_T'] = d.RPT_YR_T.astype('Int64')
            for c in ['cur', 'n_brk', 'med', 'DEV', 'HIGH']:
                out[c] = d[c]
            out['DIR'] = 2 * out.HIGH - 1
            out['reason'] = d.reason
            assert (d.RPT_DT.dropna().dt.strftime('%Y%m%d') == rep.date[d.RPT_DT.notna()]).all(), 'RPT_DT 与研报日期不一致'
        else:
            sfx = name[len('HIGH'):]
            for c in ['cur', 'n_brk', 'med', 'DEV', 'HIGH']:
                out[f'{c}{sfx}'] = d[c]
            out[f'reason{sfx}'] = d.reason
    return out


# ---------------------------------------------------------------- 样本和回归
def sample(panel_w, panel_raw, fdir, hcol='HIGH'):
    """HIGH 非缺失且公司有 2 篇及以上的研报；附上未缩尾和缩尾后的窗口 CAR、标准化的 PATHLEN 及交互项。"""
    keep = ['report_id', 'code', 'company', 'date', 'day0', 'sw', 'PATHLEN'] + CTRL_K
    d = panel_raw[keep + WIN6].merge(panel_w[['report_id'] + WIN6].add_suffix('_w').rename(columns={'report_id_w': 'report_id'}),
                                     on='report_id')
    sfx = hcol[len('HIGH'):]
    f = fdir[['report_id', 'ORG_CL', f'cur{sfx}', f'n_brk{sfx}', f'med{sfx}', f'DEV{sfx}', hcol]]
    d = d.merge(f.rename(columns={f'{c}{sfx}': c for c in ['cur', 'n_brk', 'med', 'DEV']} | {hcol: 'HIGH'}), on='report_id')
    d = d[d.HIGH.notna()]
    d = d[d.code.map(d.code.value_counts()) >= 2].reset_index(drop=True)
    d['DIR'] = 2 * d.HIGH - 1
    d['PZ'] = (d.PATHLEN - d.PATHLEN.mean()) / d.PATHLEN.std()
    d['PZ_HIGH'] = d.PZ * d.HIGH
    d['PZ_HIGHm1'] = d.PZ * (d.HIGH - 1)  # 重新参数化：PZ 的系数 = b1 + b2
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
                    rows += one(d, f'{y}{sfx}', ['PZ', 'PZ_HIGH'], ['HIGH'] + ck, fe, 'interact', spec, w, y)
                if 'interact_sum' in models:
                    rows += [dict(r, term='PZ(b1+b2)') for r in one(d, f'{y}{sfx}', ['PZ'], ['PZ_HIGHm1', 'HIGH'] + ck, fe,
                                                                    'interact_sum', spec, w, y)]
                if 'plain' in models:
                    rows += one(d, f'{y}{sfx}', ['PZ'], ck, fe, 'plain', spec, w, y)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 表格
def table(res, model, spec, d, rows_spec, extra=None, note=''):
    g = res[(res.dir_version == VERSION_NAME['HIGH']) & (res.model == model) & (res.spec == spec) & (res.winsor == 1)]
    lines = ['| | ' + ' | '.join(E.Y_LABEL[y] for y in WIN6) + ' |', '|---|' + ':-:|' * 6,
             '| | ' + ' | '.join(f'({i})' for i in range(1, 7)) + ' |']
    row = lambda name, vals: lines.append(f'| {name} | ' + ' | '.join(vals) + ' |')  # noqa: E731
    first = {y: g[g.Y == y].iloc[0] for y in WIN6}
    for term in rows_spec:
        if term in ('PZ', 'PZ_HIGH'):
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
        e = res[(res.dir_version == VERSION_NAME['HIGH']) & (res.model == extra) & (res.spec == spec) & (res.winsor == 1)]
        cs = [cellv(*e[e.Y == y][['coef', 't', 'p']].iloc[0]) for y in WIN6]
        row('预测高于其他券商研报的路径长度效应（b1+b2）', [c[0] for c in cs])
        row('', [c[1] for c in cs])
    n1, n0 = int((d.HIGH == 1).sum()), int((d.HIGH == 0).sum())
    base = (f'样本为本篇研报（盈利预测表中本公司那一行）有 2024 年归母净利润预测、发布前 90 个自然日内至少有 {MIN_BRK} 家其他券商的 2024 年预测、'
            f'且所属公司有 2 篇及以上这样研报的研报（{len(d)} 篇、{d.code.nunique()} 家；高于 {n1} 篇、低于 {n0} 篇），'
            f'剔除了无本公司预测、其他券商不足 {MIN_BRK} 家、与中位数持平（|DEV| ≤ 0.1%）和只有 1 篇的公司；')
    fe_note = ('加入公司固定效应，R² 是包含公司虚拟变量的整体 R²（不是组内 R²）；' if spec in FIRM_SPECS else '未加固定效应，R² 是普通 OLS 的 R²；')
    ctrl_note = f'控制研报字数和研报概念数：{E.CTRL_NOTE}；' if spec in CTRL_SPECS else '未加控制变量；'
    lines += ['', f'注：{note}窗口以研报发布日（非交易日顺延到下一个交易日）为第 0 天；{base}平均路径长度 PATHLEN 已在样本内标准化；'
                  f'{ctrl_note}{fe_note}括号内为按公司聚类的 t 值；***、**、* 分别表示在 1%、5%、10% 水平上显著（按公司聚类的 p 值）。']
    return '\n'.join(lines) + '\n'


DEF_NOTE = ('方向按本篇对 2024 年归母净利润的预测相对其他券商预测中位数的高低确定：其他券商指发布前 90 个自然日内（严格早于发布日）'
            '对同一公司给出 2024 年预测的其他券商，每家取最近一次预测；偏离 DEV = (本篇预测 − 中位数)/|中位数|；')


# ---------------------------------------------------------------- 主流程
def funnel(fdir, panel_ids):
    f = fdir[fdir.report_id.isin(panel_ids)]
    n_high = f[f.HIGH.notna()]
    n_firm2 = n_high[n_high.code.map(n_high.code.value_counts()) >= 2]
    steps = [('事件面板', len(f)), ('有本公司行', int((f.reason != '无本公司行').sum())), ('有 cur', int(f.cur.notna().sum())),
             (f'其他券商 ≥{MIN_BRK} 家', int((f.n_brk >= MIN_BRK).sum())), ('中位数非 0', int(f.med.notna().sum() - (f.med == 0).sum())),
             ('非持平', len(n_high)), ('公司 ≥2 篇', len(n_firm2))]
    return steps


def main(out_base, est_dir, benches=('industry', 'market')):
    t0 = time.time()
    fd = out_base / 'fdir'
    fd.mkdir(parents=True, exist_ok=True)
    panels = {}
    for bench in benches:
        src = out_base / E.out_name('PATHLEN', bench)
        panels[bench] = (pd.read_parquet(src / f'{E.out_name("PATHLEN", bench)}_panel.parquet'),
                         pd.read_parquet(src / f'{E.out_name("PATHLEN", bench)}_panel_raw.parquet'))
    ids = [set(p[1].report_id) for p in panels.values()]
    assert all(s == ids[0] for s in ids), '两个口径的事件面板研报不同'
    fdir = build_fdir(panels[benches[0]][1], est_dir)
    fdir.to_parquet(fd / 'report_fdir.parquet', index=False)
    fdir.to_csv(fd / 'report_fdir.csv', index=False, encoding='utf-8-sig')
    print(f'方向表：{fd}（{len(fdir)} 篇，{time.time() - t0:.0f} 秒）', flush=True)
    print('缺失原因：' + '；'.join(f'{k or "有效"} {v}' for k, v in fdir.reason.value_counts().items()), flush=True)
    print('漏斗：' + ' → '.join(f'{k} {v}' for k, v in funnel(fdir, ids[0])), flush=True)
    for sfx in ('_25', '_180'):
        h = fdir[fdir[f'HIGH{sfx}'].notna()]
        print(f'  HIGH{sfx}：非缺失 {len(h)} 篇（高于 {int((h[f"HIGH{sfx}"] == 1).sum())}）；与主版本都非缺失的 '
              f'{int((h.HIGH.notna()).sum())} 篇里方向一致 {int((h.HIGH == h[f"HIGH{sfx}"]).sum())} 篇', flush=True)
    out = {}
    for bench in benches:
        pw, pr = panels[bench]
        d = sample(pw, pr, fdir)
        g = d.groupby('code').HIGH.agg(['min', 'max', 'size'])
        both = g[(g['min'] == 0) & (g['max'] == 1)]
        print(f'[{bench}] 样本 {len(d)} 篇、{d.code.nunique()} 家；高于 {int((d.HIGH == 1).sum())}、低于 {int((d.HIGH == 0).sum())}；'
              f'公司内两个方向都有的 {len(both)} 家、{int(both["size"].sum())} 篇；n_brk 分位数（0/25/50/75/100）'
              f'{d.n_brk.quantile([0, .25, .5, .75, 1]).tolist()}', flush=True)
        mean = d.groupby('HIGH')[[f'{y}_w' for y in WIN6]].mean().T
        mean.index = [E.Y_LABEL[y] for y in WIN6]
        print(f'[{bench}] 缩尾后 CAR 按 HIGH 分组均值（百分点）：\n{mean.round(3).to_string()}', flush=True)
        res = run_all(d)
        res.insert(0, 'dir_version', VERSION_NAME['HIGH'])
        for hcol in ('HIGH_25', 'HIGH_180'):
            dr = sample(pw, pr, fdir, hcol)
            print(f'[{bench}] 稳健性 {hcol}：样本 {len(dr)} 篇、{dr.code.nunique()} 家；高于 {int((dr.HIGH == 1).sum())}、'
                  f'低于 {int((dr.HIGH == 0).sum())}', flush=True)
            rr = run_all(dr, specs=[s for s in SPECS if s[0] == 'firm'], models=('signed', 'interact_sum'))
            rr.insert(0, 'dir_version', VERSION_NAME[hcol])
            res = pd.concat([res, rr], ignore_index=True)
        res.insert(0, 'bench', bench)
        res.insert(0, 'X', 'PATHLEN')
        od = out_base / ('pathlen_fdir_ind' if bench == 'industry' else 'pathlen_fdir')
        td = od / 'tables'
        td.mkdir(parents=True, exist_ok=True)
        d.to_parquet(od / 'pathlen_fdir_sample.parquet', index=False)
        res.to_csv(od / 'pathlen_fdir_results.csv', index=False, encoding='utf-8-sig')
        bn = E.BENCH_NOTE[bench]
        sn = (f'因变量为 signed CAR = DIR × CAR，{DEF_NOTE}DIR 在 DEV > 0.1%（预测高于其他券商）时取 +1、DEV < −0.1% 时取 −1，'
              f'CAR 为各窗口累积超额收益（{bn}，百分点），先乘方向再在样本内按 1%/99% 缩尾；')
        inote = (f'因变量为各窗口累积超额收益（{bn}，百分点，在 790 篇平衡样本上按 1%/99% 缩尾，同表 5、6）；{DEF_NOTE}'
                 f'预测高于其他券商 HIGH 在 DEV > 0.1% 时取 1、DEV < −0.1% 时取 0；平均路径长度×预测高于其他券商 = 标准化的 PATHLEN × HIGH；'
                 f'b1+b2 的 t 值用聚类协方差计算（把交互项换成 PATHLEN×(HIGH−1) 重新估计，PATHLEN 的系数即 b1+b2）；')
        tabs = [('t13_fdir_signed_pooled.md', 'signed', 'none_s', ['PZ'], None, sn),
                ('t14_fdir_signed_firm.md', 'signed', 'firm', ['PZ'], None, sn),
                ('t15_fdir_interact_firm.md', 'interact', 'firm', ['PZ', 'PZ_HIGH', 'HIGH'], 'interact_sum', inote),
                ('t13k_fdir_signed_pooled_ctrl.md', 'signed', 'none_sk', ['PZ'] + CTRL_K, None, sn),
                ('t14k_fdir_signed_firm_ctrl.md', 'signed', 'firm_k', ['PZ'] + CTRL_K, None, sn),
                ('t15k_fdir_interact_firm_ctrl.md', 'interact', 'firm_k', ['PZ', 'PZ_HIGH', 'HIGH'] + CTRL_K, 'interact_sum', inote)]
        for fn, model, spec, rows_spec, extra, note in tabs:
            (td / fn).write_text(table(res, model, spec, d, rows_spec, extra=extra, note=note), encoding='utf-8')
        out[bench] = (d, res)
        print(f'[{bench}] 完成：{od}（{len(res)} 行，{time.time() - t0:.0f} 秒）', flush=True)
    return fdir, out
