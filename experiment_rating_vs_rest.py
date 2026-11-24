"""FY2025 上的效果是不是都来自“投资评级”节点：最终指标清单去掉“评级”类后的汇点指标，单独回归和与评级终点指标联合回归。

口径：
  清单去掉评级  final_indicator_list.csv 里 status == '入选' 且 category != '评级' 的名称；目标节点 = 图里名称在这个子清单内的严格汇点
                （入度 > 0、出度 = 0）；三个指标同 sink_final（dag_forecast.sink_measures(targets='FINAL', count=True, final_exclude=('评级',))），
                在有值的研报里按 1%/99% 缩尾。
  评级终点      output_dag/forecast_newdata/rating_strict_n/ 面板里的 R_RATIO、R_LNPATH、R_PATHLEN（已缩尾）。
  完整清单      output_dag/forecast_newdata/sink_final_n/ 面板里的同名指标（只用于相关系数）。
被解释变量、控制变量、Panel N/A/B/C、聚类都同 src/dag_forecast.py（Y 和控制变量取 sink_final_n/ 的缩尾后面板），X 在回归样本内标准化。
  (1) 单独回归：清单去掉评级的三个指标各自进入；样本 all = 全部有指标的研报，no_rating = 其中“投资评级”不是严格汇点的研报
  (2) 联合回归：两套都有值的研报上，评级终点指标和清单去掉评级的同类型指标同时进入，报告两系数、两系数相等检验 p 值，
      以及同一样本上各自单独进入的系数

在仓库根目录运行：
    python experiment_rating_vs_rest.py
结果写到 output_dag/forecast_newdata/rating_vs_rest/（全部预测行）和 rating_vs_rest/main_only/（主体公司行）：
    rest_measures.parquet（清单去掉评级的指标，未缩尾和缩尾后）、counts.csv、corr.csv、reg_single.csv、reg_joint.csv
"""
import sys

import numpy as np
import pandas as pd

from experiment_fy_horizon import fitv, lincom, prep, wald
from src import dag_forecast as F
from src.config import ROOT_DIR

# ==================== 运行参数 ====================
ND = ROOT_DIR / 'output_dag' / 'forecast_newdata'
OUT = ND / 'rating_vs_rest'
DAG_DIRS = [ROOT_DIR / 'output_dag' / 'deep' / 'data', ROOT_DIR / 'output_dag' / 'data']  # 同 regress_dag_forecast.py
EXCLUDE = ('评级',)
XS = ['R_RATIO', 'R_LNPATH', 'R_PATHLEN']
YS = ['PMAFE_FY24', 'PMAFE_FY25']
SCOPES = {'all': '', 'main': 'main_only'}
# ==================================================


def rest_measures(report_ids):
    """清单去掉评级的指标；顺带核对不排除时与 sink_final_n/ 面板一致。"""
    q = F.sink_measures(report_ids, DAG_DIRS, 'FINAL', count=True, final_exclude=EXCLUDE)
    full = F.sink_measures(report_ids, DAG_DIRS, 'FINAL', count=True)
    ref = pd.read_parquet(ND / 'sink_final_n' / 'forecast_panel_raw.parquet').set_index('report_id').loc[full.report_id]
    for c in [x + '_RAW' for x in XS] + XS + ['R_N_SINKS']:
        assert np.allclose(full[c].to_numpy(float), ref[c].to_numpy(float), equal_nan=True), c
    print(f'清单去掉评级：{len(F.final_names(exclude=EXCLUDE))} 个名称；完整清单复算与 sink_final_n/ 一致', flush=True)
    return q


def run_scope(scope, q):
    sub = SCOPES[scope]
    out = OUT / sub if sub else OUT
    out.mkdir(parents=True, exist_ok=True)
    pdir = lambda name: ND / name / sub if sub else ND / name  # noqa: E731
    w = pd.read_parquet(pdir('sink_final_n') / 'forecast_panel.parquet')
    g = pd.read_parquet(pdir('rating_strict_n') / 'forecast_panel.parquet')
    assert list(w.report_id) == list(g.report_id) and w.PMAFE_FY25.equals(g.PMAFE_FY25)
    w = w.rename(columns={x: 'F_' + x[2:] for x in XS})  # 完整清单
    for x in XS:
        w['G_' + x[2:]] = g[x].to_numpy()  # 评级终点
    qq = q.set_index('report_id').loc[w.report_id]
    for x in XS:
        w['Q_' + x[2:]] = qq[x].to_numpy()  # 清单去掉评级
    w['Q_N_SINKS'] = qq.R_N_SINKS.to_numpy()
    w['has_q'] = qq.R_LNPATH_RAW.notna().to_numpy()
    w['has_g'] = g.R_LNPATH_RAW.notna().to_numpy()
    if scope == 'all':
        q.to_parquet(out / 'rest_measures.parquet', index=False)
        cnt = []
        for lab, m in (('清单去掉评级有指标（全部）', w.has_q), ('清单去掉评级有指标（base 内）', w.has_q & w.base),
                       ('其中投资评级不是严格汇点（全部）', w.has_q & ~w.has_g), ('其中投资评级不是严格汇点（base 内）', w.has_q & ~w.has_g & w.base),
                       ('两套都有值（全部）', w.has_q & w.has_g), ('两套都有值（base 内）', w.has_q & w.has_g & w.base)):
            cnt.append({'item': lab, 'n_reports': int(m.sum()), 'n_firms': int(w.code[m].nunique())})
        s = w.loc[w.has_q, 'Q_N_SINKS']
        cnt.append({'item': '清单去掉评级：每篇目标节点个数 均值 / 中位数', 'n_reports': round(float(s.mean()), 3), 'n_firms': float(s.median())})
        pd.DataFrame(cnt).to_csv(out / 'counts.csv', index=False, encoding='utf-8-sig')
        cr = []
        for x in ('RATIO', 'LNPATH', 'PATHLEN'):
            for a, b in (('G', 'Q'), ('G', 'F'), ('Q', 'F')):
                for lab, m in (('全部', pd.Series(True, index=w.index)), ('base 内', w.base)):
                    v = w.loc[m, [f'{a}_{x}', f'{b}_{x}']].dropna()
                    cr.append({'X': x, 'pair': f'{a}-{b}', 'sample': lab, 'n': len(v),
                               'pearson': float(np.corrcoef(v.iloc[:, 0], v.iloc[:, 1])[0, 1]),
                               'spearman': float(np.corrcoef(v.iloc[:, 0].rank(), v.iloc[:, 1].rank())[0, 1])})
        pd.DataFrame(cr).to_csv(out / 'corr.csv', index=False, encoding='utf-8-sig')
    pans = F.panels(with_nofe=True)
    single, joint = [], []
    for y in YS:
        d = prep(w, y)
        for pan, (fe, ctrl) in pans.items():
            for x in ('RATIO', 'LNPATH', 'PATHLEN'):
                meta = {'scope': scope, 'Y': y, 'panel': pan, 'X': x}
                # (1) 单独回归
                for samp, m in (('all', d.has_q), ('no_rating', d.has_q & ~d.has_g)):
                    r = fitv(d[m], y, f'Q_{x}', ctrl, fe)
                    est, se, t, p = lincom(r, [1])
                    single.append({**meta, 'sample': samp, 'coef': est, 'se': se, 't': t, 'p': p, 'n': r['n'], 'n_clusters': r['G']})
                # (2) 联合回归（两套都有值）和同一样本上各自单独进入
                gx, qx = f'G_{x}', f'Q_{x}'
                dj = d[d.has_q & d.has_g]

                def both(s, gx=gx, qx=qx):
                    return pd.DataFrame({gx: s[gx], qx: (s[qx] - s[qx].mean()) / s[qx].std()})
                rj = fitv(dj, y, gx, ctrl, fe, both, extra=[qx])
                rg = fitv(dj, y, gx, ctrl, fe, extra=[qx])
                rq = fitv(dj, y, qx, ctrl, fe, extra=[gx])
                rec = {**meta, 'n': rj['n'], 'n_clusters': rj['G'], 'p_equal': wald(rj, [[1, -1]])}
                for lab, r, wv in (('rating_joint', rj, [1, 0]), ('rest_joint', rj, [0, 1]), ('rating_alone', rg, [1]), ('rest_alone', rq, [1])):
                    est, se, t, p = lincom(r, wv)
                    rec.update({f'{lab}_coef': est, f'{lab}_t': t, f'{lab}_p': p})
                assert rg['n'] == rq['n'] == rj['n']
                joint.append(rec)
    pd.DataFrame(single).to_csv(out / 'reg_single.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(joint).to_csv(out / 'reg_joint.csv', index=False, encoding='utf-8-sig')
    print(f'[{scope}] 结果 -> {out}', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    rids = list(pd.read_parquet(ND / 'sink_final_n' / 'forecast_panel_raw.parquet').report_id)
    qm = rest_measures(rids)
    for sc in SCOPES:
        run_scope(sc, qm)
