"""experiment_rating_subgraph.py 分析三的补充：带方向的 CAR（signed CAR）。只读已有文件，不重算方向和 CAR。

方向  output_dag/experiments/report_car/fdir/report_fdir.parquet（定义见 src/report_car_fdir.py 开头）
      dir24（主口径）= DIR：本篇 FY2024 归母净利润预测高于其他券商中位数 +1、低于 −1、相差 0.1% 以内或无可比预测缺失
      dir25（第二口径）= 2·HIGH_25 − 1（同样定义，用 FY2025 预测）
样本  “投资评级”是严格汇点的 606 篇 ∩ 市场反应面板（790 篇）∩ 方向非缺失（两个口径各自一套）
Y     signed：方向 × 未缩尾 CAR（七个已有窗口取 report_car_panel_raw.parquet，CAR_0_40、CAR_0_60 取 rating_subgraph/car_long.parquet
      的 car_raw），在该口径的样本内按 1%/99% 缩尾；plain：不带方向的 CAR（面板缩尾值；CAR_0_40、CAR_0_60 取 car_w），同样本对照。× 100。
X     rating_subgraph/measures.parquet 里在 606 篇内缩尾好的五个指标，估计样本内标准化，每次一个
回归  Panel N（无固定效应）、A（申万一级 + 月份）、B（公司 + 月份）、C（公司 + 券商 + 月份），按公司聚类；
      控制变量同 experiment_rating_subgraph.py（N、A 含四个公司变量，B、C 不含；发布前窗口不放 PRECAR）
      ytype = signed       PRECAR 换成 方向 × PRECAR（主结果）
              signed_rawpre PRECAR 不乘方向
              plain        不带方向的 CAR，PRECAR 不乘方向

在仓库根目录运行：
    C:\\Users\\hung\\projects\\.venv\\Scripts\\python.exe -X utf8 experiment_rating_subgraph_signed.py
结果写到 output_dag/forecast_newdata/rating_subgraph/signed/：
    reg_signed.csv     长表（version、ytype、Y、panel、X、coef、se、t、p、n、n_clusters …）
    samples.csv        各口径方向 +1 / −1 / 缺失篇数、样本篇数和公司数、进入 Panel C 回归的篇数和公司数
    signed_panel.parquet  回归用数据（每个口径一套，含 signed Y、PRECAR_S）
"""
import sys

import pandas as pd

from experiment_rating_subgraph import CAR_DIR, CAR_PANELS, CTRL_CAR, CTRL_FIRM, MEAS, OUT, SG, Y_CAR_POST, Y_CAR_PRE
from src import dag_outcomes_reg as R

# ==================== 运行参数 ====================
OUT_S = OUT / 'signed'
FDIR = CAR_DIR / 'fdir' / 'report_fdir.parquet'
VERSIONS = {'dir24': 'FY2024 方向（DIR，主口径）', 'dir25': 'FY2025 方向（2·HIGH_25 − 1）'}
YS = Y_CAR_PRE + Y_CAR_POST
NEW = ['CAR_0_40', 'CAR_0_60']
# ==================================================


def load():
    m = pd.read_parquet(OUT / 'measures.parquet')[['report_id'] + MEAS].rename(columns=SG)
    w = pd.read_parquet(CAR_DIR / 'report_car_panel.parquet')
    raw = pd.read_parquet(CAR_DIR / 'report_car_panel_raw.parquet').set_index('report_id')
    cl = pd.read_parquet(OUT / 'car_long.parquet')
    for name in NEW:
        c = cl[cl.window == name].set_index('report_id')
        w[name] = w.report_id.map(c.car_w)
        raw[name] = c.car_raw.reindex(raw.index)
    for y in YS:
        w['RAW_' + y] = w.report_id.map(raw[y])
    f = pd.read_parquet(FDIR).set_index('report_id')
    w['dir24'] = w.report_id.map(f.DIR)
    w['dir25'] = w.report_id.map(2 * f.HIGH_25 - 1)
    d = w.merge(m, on='report_id', how='inner')
    assert len(d) == 605
    return d


def main():
    OUT_S.mkdir(parents=True, exist_ok=True)
    d0 = load()
    rows, samp, panels = [], [], []
    for ver, label in VERSIONS.items():
        cnt = d0[ver].value_counts(dropna=False)
        d = d0[d0[ver].notna()].copy()
        for y in YS:
            d['S_' + y] = R.winsor(d[ver] * d['RAW_' + y]) * 100  # signed，在该口径样本内缩尾
            d['P_' + y] = d[y] * 100  # plain（面板缩尾值）
        d['PRECAR_S'] = d[ver] * d.PRECAR
        panels.append(d.assign(version=ver))
        used_c = set()
        for ytype in ('signed', 'signed_rawpre', 'plain'):
            pre = 'PRECAR_S' if ytype == 'signed' else 'PRECAR'
            for y in YS:
                yc = ('P_' if ytype == 'plain' else 'S_') + y
                for pan, fe in CAR_PANELS.items():
                    ctrl = [pre if c == 'PRECAR' else c for c in CTRL_CAR
                            if not (pan in ('B', 'C') and c in CTRL_FIRM) and not (y in Y_CAR_PRE and c == 'PRECAR')]
                    for x in MEAS:
                        res, s = R.fit(d, yc, [SG[x]], ctrl, fe, 'code')
                        r = res[0]
                        if ytype == 'signed' and pan == 'C':
                            used_c |= set(d.loc[s.index, 'report_id'])
                        rows.append({'version': ver, 'ytype': ytype, 'Y': y, 'panel': pan, 'X': x, 'coef': r['coef'], 'se': r['se'],
                                     't': r['t'], 'p': r['p'], 'n': r['n'], 'n_clusters': r['n_clusters'],
                                     'n_singletons': r['n_singletons'], 'fe': '+'.join(fe), 'controls': '+'.join(ctrl)})
        samp.append({'version': ver, 'label': label, 'n_pos': int(cnt.get(1.0, 0)), 'n_neg': int(cnt.get(-1.0, 0)),
                     'n_missing': int(d0[ver].isna().sum()), 'n_sample': len(d), 'firms_sample': d.code.nunique(),
                     'n_reg_C': len(used_c), 'firms_reg_C': d[d.report_id.isin(used_c)].code.nunique()})
    res = pd.DataFrame(rows)
    res.to_csv(OUT_S / 'reg_signed.csv', index=False, encoding='utf-8-sig')
    sm = pd.DataFrame(samp)
    sm.to_csv(OUT_S / 'samples.csv', index=False, encoding='utf-8-sig')
    keep = ['version', 'report_id', 'code', 'date', 'month', 'sw', 'broker_g', 'dir24', 'dir25', 'PRECAR', 'PRECAR_S'] + \
        [p + y for y in YS for p in ('RAW_', 'S_', 'P_')] + list(SG.values())
    pd.concat(panels, ignore_index=True)[keep].to_parquet(OUT_S / 'signed_panel.parquet', index=False)
    print(sm.to_string(index=False), flush=True)
    print(f'完成 -> {OUT_S}（{len(res)} 个系数）', flush=True)


if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    main()
