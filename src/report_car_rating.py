"""实验：平均路径长度 PATHLEN 与研报发布前后 CAR，按评级强度（该券商评级体系的最高档 vs 次高档）区分。由根目录 experiment_pathlen_rating.py 调用。
读 extract_report_rating.py 的评级（TOP：最高档 1、次高档 0）和 experiment_dag_event.py 的事件面板（行业调整版、市场调整版）。

  样本      TOP 非缺失、且该公司有 2 篇及以上这样的研报（三张表同一样本，公司固定效应下没有单例）
  A signed  DIR = +1（最高档）/ −1（次高档），signed CAR = DIR × CAR（未缩尾的 CAR 乘方向后，在这个样本内按 1%/99% 缩尾）
            none_sk（无固定效应）和 firm_k（公司固定效应），都控制 LNCHARS_R、LNNODES_R → 表 7、表 8
  B 交互    CAR = b1·PATHLEN + b2·PATHLEN×TOP + b3·TOP + 控制变量 (+ 公司固定效应)；CAR 用事件面板里缩尾后的值（在 790 篇平衡样本上缩尾，同表 5、6）
            PATHLEN 在样本内标准化后再乘 TOP；b1+b2 用重新参数化得到：把 PATHLEN×TOP 换成 PATHLEN×(TOP−1)，PATHLEN 的系数就是 b1+b2，
            其聚类标准误即 b1+b2 的聚类标准误 → 表 9（firm_k）；无固定效应版只进 csv
  C 对照    同样本不乘方向、不加交互：none_sk、firm_k，以及另加一个 TOP 控制变量的版本（只进 csv）
  D 无控制  同一样本、不加 LNCHARS_R、LNNODES_R：none_s（无固定效应）、firm（公司固定效应），A、B、C 中 signed、interact
            （TOP 主效应保留）、interact_sum、plain 四类都跑，行追加在 csv 末尾 → 表 10（signed none_s）、表 11（signed firm）、
            表 12（interact firm）
  标准误    按公司聚类；另算公司 × 事件日双向聚类的 t（t_twoway）
"""
import time

import numpy as np
import pandas as pd

from src import dag_outcomes_reg as R
from src import report_car_event as E

WIN6 = E.WIN6
CTRL_K = E.CTRL_K
SPECS = [('none_sk', []), ('firm_k', ['code'])]
SPECS_NC = [('none_s', []), ('firm', ['code'])]  # 不加 CTRL_K 的规格
FIRM_SPECS = ('firm_k', 'firm')
CTRL_SPECS = ('none_sk', 'firm_k')
LABEL = {'PZ': '平均路径长度', 'PZ_TOP': '平均路径长度×最高档评级', 'TOP': '最高档评级', 'LNCHARS_R': '研报字数', 'LNNODES_R': '研报概念数'}


def sample(panel_w, panel_raw, rating):
    """TOP 非缺失且公司有 2 篇及以上的研报；附上未缩尾和缩尾后的窗口 CAR、标准化的 PATHLEN 及交互项。"""
    keep = ['report_id', 'code', 'company', 'date', 'day0', 'sw', 'PATHLEN'] + CTRL_K
    d = panel_raw[keep + WIN6].merge(panel_w[['report_id'] + WIN6].add_suffix('_w').rename(columns={'report_id_w': 'report_id'}),
                                     on='report_id')
    d = d.merge(rating[['report_id', 'broker', 'rating_raw', 'tier', 'TOP']], on='report_id')
    d = d[d.TOP.notna()]
    d = d[d.code.map(d.code.value_counts()) >= 2].reset_index(drop=True)
    d['DIR'] = 2 * d.TOP - 1
    d['PZ'] = (d.PATHLEN - d.PATHLEN.mean()) / d.PATHLEN.std()
    d['PZ_TOP'] = d.PZ * d.TOP
    d['PZ_TOPm1'] = d.PZ * (d.TOP - 1)  # 重新参数化：PZ 的系数 = b1 + b2
    for y in WIN6:
        d[f'S_{y}'] = d.DIR * d[y]  # signed CAR（未缩尾）
        d[f'S_{y}_w'] = R.winsor(d[f'S_{y}'])
    return d


def one(d, y, xs, ctrl, fe, model, spec, winsor, y_name):
    """一次回归（xs 不再标准化），返回每个 xs 一行，控制变量系数放在 b_/t_/p_ 列。"""
    res, s = R.fit(d, y, xs, ctrl, fe, 'code', standardize=False, terms=True)
    res2, _ = R.fit(d, y, xs, ctrl, fe, ['code', 'day0'], standardize=False)
    ctrl_rows = {r['X']: r for r in res[len(xs):]}
    rows = []
    for i, x in enumerate(xs):
        r = res[i]
        row = {'model': model, 'spec': spec, 'Y': y_name, 'Y_label': E.Y_LABEL[y_name], 'winsor': winsor, 'term': x,
               'coef': r['coef'], 'se': r['se'], 't': r['t'], 'p': r['p'], 't_twoway': res2[i]['t'], 'p_twoway': res2[i]['p'],
               'n': r['n'], 'n_firms': s.code.nunique(), 'r2': r['r2'], 'adj_r2': r['adj_r2'], 'fe': '+'.join(fe), 'controls': '、'.join(ctrl)}
        for c, cr in ctrl_rows.items():
            row.update({f'b_{c}': cr['coef'], f't_{c}': cr['t'], f'p_{c}': cr['p']})
        rows.append(row)
    return rows


def run_all(d):
    rows = []
    for y in WIN6:
        for w, sfx in [(1, '_w'), (0, '')]:
            for spec, fe in SPECS:
                rows += one(d, f'S_{y}{sfx}', ['PZ'], CTRL_K, fe, 'signed', spec, w, y)
                rows += one(d, f'{y}{sfx}', ['PZ', 'PZ_TOP'], ['TOP'] + CTRL_K, fe, 'interact', spec, w, y)
                rows += [dict(r, term='PZ(b1+b2)') for r in one(d, f'{y}{sfx}', ['PZ'], ['PZ_TOPm1', 'TOP'] + CTRL_K, fe,
                                                                'interact_sum', spec, w, y)]
                rows += one(d, f'{y}{sfx}', ['PZ'], CTRL_K, fe, 'plain', spec, w, y)
                rows += one(d, f'{y}{sfx}', ['PZ'], ['TOP'] + CTRL_K, fe, 'plain_top', spec, w, y)
    # 不加 CTRL_K 的规格追加在后面，不改变上面已有行的顺序和数值
    for y in WIN6:
        for w, sfx in [(1, '_w'), (0, '')]:
            for spec, fe in SPECS_NC:
                rows += one(d, f'S_{y}{sfx}', ['PZ'], [], fe, 'signed', spec, w, y)
                rows += one(d, f'{y}{sfx}', ['PZ', 'PZ_TOP'], ['TOP'], fe, 'interact', spec, w, y)
                rows += [dict(r, term='PZ(b1+b2)') for r in one(d, f'{y}{sfx}', ['PZ'], ['PZ_TOPm1', 'TOP'], fe,
                                                                'interact_sum', spec, w, y)]
                rows += one(d, f'{y}{sfx}', ['PZ'], [], fe, 'plain', spec, w, y)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 表格
def cellv(coef, t, p):
    return E.num(coef, 3) + R.stars(p), f'({E.num(t, 2)})'


def table(res, model, spec, bench, d, rows_spec, extra=None, note=''):
    g = res[(res.model == model) & (res.spec == spec) & (res.winsor == 1)]
    lines = ['| | ' + ' | '.join(E.Y_LABEL[y] for y in WIN6) + ' |', '|---|' + ':-:|' * 6,
             '| | ' + ' | '.join(f'({i})' for i in range(1, 7)) + ' |']
    row = lambda name, vals: lines.append(f'| {name} | ' + ' | '.join(vals) + ' |')  # noqa: E731
    first = {y: g[g.Y == y].iloc[0] for y in WIN6}
    for term in rows_spec:
        if term in ('PZ', 'PZ_TOP'):
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
        e = res[(res.model == extra) & (res.spec == spec) & (res.winsor == 1)]
        cs = [cellv(*e[e.Y == y][['coef', 't', 'p']].iloc[0]) for y in WIN6]
        row('最高档评级研报的路径长度效应（b1+b2）', [c[0] for c in cs])
        row('', [c[1] for c in cs])
    n1, n0 = int((d.TOP == 1).sum()), int((d.TOP == 0).sum())
    base = (f'样本为评级档位可识别为该券商体系最高档或次高档、且所属公司有 2 篇及以上这样研报的研报（{len(d)} 篇、{d.code.nunique()} 家；'
            f'最高档 {n1} 篇、次高档 {n0} 篇），剔除了无评级、未识别评级、中性及以下评级和只有 1 篇的公司；')
    fe_note = ('加入公司固定效应，R² 是包含公司虚拟变量的整体 R²（不是组内 R²）；' if spec in FIRM_SPECS else '未加固定效应，R² 是普通 OLS 的 R²；')
    ctrl_note = f'控制研报字数和研报概念数：{E.CTRL_NOTE}；' if spec in CTRL_SPECS else '未加控制变量；'
    lines += ['', f'注：{note}窗口以研报发布日（非交易日顺延到下一个交易日）为第 0 天；{base}平均路径长度 PATHLEN 已在样本内标准化；'
                  f'{ctrl_note}{fe_note}括号内为按公司聚类的 t 值；***、**、* 分别表示在 1%、5%、10% 水平上显著（按公司聚类的 p 值）。']
    return '\n'.join(lines) + '\n'


def main(out_base, rating_path, benches=('industry', 'market')):
    t0 = time.time()
    rating = pd.read_parquet(rating_path)
    out = {}
    for bench in benches:
        src = out_base / E.out_name('PATHLEN', bench)
        pw = pd.read_parquet(src / f'{E.out_name("PATHLEN", bench)}_panel.parquet')
        pr = pd.read_parquet(src / f'{E.out_name("PATHLEN", bench)}_panel_raw.parquet')
        d = sample(pw, pr, rating)
        g = d.groupby('code').TOP.agg(['min', 'max', 'size'])
        both = g[(g['min'] == 0) & (g['max'] == 1)]
        print(f'[{bench}] 样本 {len(d)} 篇、{d.code.nunique()} 家；最高档 {int((d.TOP == 1).sum())}、次高档 {int((d.TOP == 0).sum())}；'
              f'公司内两档都有的 {len(both)} 家、{int(both["size"].sum())} 篇', flush=True)
        res = run_all(d)
        res.insert(0, 'bench', bench)
        res.insert(0, 'X', 'PATHLEN')
        od = out_base / ('pathlen_rating_ind' if bench == 'industry' else 'pathlen_rating')
        (od / 'tables').mkdir(parents=True, exist_ok=True)
        d.to_parquet(od / 'pathlen_rating_sample.parquet', index=False)
        res.to_csv(od / 'pathlen_rating_results.csv', index=False, encoding='utf-8-sig')
        bn = E.BENCH_NOTE[bench]
        sn = f'因变量为 signed CAR = DIR × CAR，DIR 在最高档评级时取 +1、次高档取 −1，CAR 为各窗口累积超额收益（{bn}，百分点），先乘方向再在样本内按 1%/99% 缩尾；'
        (od / 'tables' / 't7_signed_pooled_ctrl.md').write_text(table(res, 'signed', 'none_sk', bench, d, ['PZ'] + CTRL_K, note=sn), encoding='utf-8')
        (od / 'tables' / 't8_signed_firm_ctrl.md').write_text(table(res, 'signed', 'firm_k', bench, d, ['PZ'] + CTRL_K, note=sn), encoding='utf-8')
        inote = (f'因变量为各窗口累积超额收益（{bn}，百分点，在 790 篇平衡样本上按 1%/99% 缩尾，同表 5、6）；最高档评级 TOP 在本次评级为该券商体系最高档时取 1、'
                 f'次高档取 0；平均路径长度×最高档评级 = 标准化的 PATHLEN × TOP；b1+b2 的 t 值用聚类协方差计算（把交互项换成 PATHLEN×(TOP−1) 重新估计，PATHLEN 的系数即 b1+b2）；')
        (od / 'tables' / 't9_interact_firm_ctrl.md').write_text(
            table(res, 'interact', 'firm_k', bench, d, ['PZ', 'PZ_TOP', 'TOP'] + CTRL_K, extra='interact_sum', note=inote), encoding='utf-8')
        (od / 'tables' / 't10_signed_pooled.md').write_text(table(res, 'signed', 'none_s', bench, d, ['PZ'], note=sn), encoding='utf-8')
        (od / 'tables' / 't11_signed_firm.md').write_text(table(res, 'signed', 'firm', bench, d, ['PZ'], note=sn), encoding='utf-8')
        (od / 'tables' / 't12_interact_firm.md').write_text(
            table(res, 'interact', 'firm', bench, d, ['PZ', 'PZ_TOP', 'TOP'], extra='interact_sum', note=inote), encoding='utf-8')
        out[bench] = (d, res)
        print(f'[{bench}] 完成：{od}（{len(res)} 行，{time.time() - t0:.0f} 秒）', flush=True)
    return out
