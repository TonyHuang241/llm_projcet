"""读 accuracy 步骤的成品（output/accuracy/ 下的三张表）。下游通过这里读，不自己拼路径。

这里一切都是原始值：没有缩尾、没有标准化，MOM、VOL、ROA_T 是小数（文档里以 % 报告，研究层自己乘 100）。口径细节见
src/accuracy/forecasts.py、peers.py、controls.py 的模块说明。

report_accuracy.parquet：每篇研报一行（研报范围 = graph 步骤的全部研报，行序同 src.graph.load.read_table('reports')）
  report_id       研报 ID（graph 研报表的 report_id）
  code            公司代码，graph 研报表 stock_code 的前 6 位
  date            发布日 YYYYMMDD（graph 研报表的 date）
  broker          券商名（graph 研报表的 broker，原样；券商分组留给研究层）
  ORG_CL          本篇预测行的券商编号（EST_COMP_TN 的 ORG_CL）；没匹配到本篇预测时缺失（浮点数）
  RPT_YR_T        本篇预测行的 RPT_YR_T（记为 T，预测基于的最近年报年度）；没匹配到时缺失
  matched         是否匹配到本篇预测行
  base            基础样本：匹配到本篇预测 & 有 T+1 年的预测值和实际值 & 研报日早于 T+1 年年报公告日
  TY24、TY25      常数 2024、2025（研究层 src/studies/forecast_accuracy/regressions.py 的 y_sample 按它们设目标年度）
  LNHORIZON24、LNHORIZON25  ln(FY2024 / FY2025 年报公告日 − 研报日 的天数)，天数 ≤ 0 时缺失
  F24、F25        本篇对 FY2024、FY2025 归母净利润的预测（元）
  A24、A25        FY2024、FY2025 归母净利润实际值（元，financial_data 的 np_parent_company_owners_ttm）
  ABSERR24、ABSERR25  本篇绝对预测误差 |F − A|
  ---- 以下比较组变量只对 base 为真的研报有值 ----
  PMAFE_FY25      FY2025、比较组 [d − 90, d] 的相对预测误差 (AFE − MAFE)/MAFE，MAFE 含本篇；同行 < 3 家时缺失
  PMAFE_FY24      同上，FY2024
  SCORE_FY25      FY2025、[d − 90, d] 比较组上的 Hong-Kubik 排名分数（0–100，越大越准）；同行 < 3 家时缺失
  PMAFEW30_FY25   FY2025、比较组窗口向后延伸 30 天 [d − 90, d + 30] 的 PMAFE
  N_PEERS25、N_PEERS24、N_PEERSW30  三个比较组的同行家数（不含本篇；可以小于 3；浮点数，非 base 研报缺失）
  PEERLAG25、PEERLAG24、PEERLAGW30  三个比较组的同行时滞：各同行 (d − 同行预测日) 天数的均值；同行 < 3 家时缺失
  LAGMED25        [d − 90, d]、FY2025 比较组同行时滞的中位数；同行 < 3 家时缺失
  LNNPEERS25      ln(N_PEERS25)；同行 < 3 家时缺失
  ---- 控制变量：研报发布日上的取值，对全部研报计算（缺数据时缺失）----
  SIZE_M          ln(前一交易日总市值)
  BM              T 年末净资产 ÷ 前一交易日总市值
  MOM             前第 21 个交易日收盘价 ÷ 前第 250 个交易日收盘价 − 1（小数）
  VOL             前 120 个交易日的日收益标准差（小数）
  ROA_T           T 年归母净利润 ÷ T 年末总资产（小数）
  LNAGE           ln(上市以来月数)（行情首个交易日起算）

report_peers.parquet：base 研报在 FY2025、[d − 90, d] 比较组里的每个同行一行（PMAFE_FY25、SCORE_FY25、PEERLAG25 用的就是这些同行；
同行不足 3 家的研报也列出）。按 report_accuracy 中 base 研报的顺序，研报内按同行券商、时间的长表顺序
  report_id       研报 ID
  peer_ORG_CL     同行券商编号
  peer_RPT_ID     同行那条预测的 RPT_ID
  peer_dt         同行预测发布日 YYYYMMDD
  LAG             时滞天数 = 研报发布日 − peer_dt（0–90）
  F               同行对 FY2025 归母净利润的预测（元）

pair_accuracy.parquet：每个（公司，券商）一行；范围 = report_accuracy 里有 ORG_CL 的研报出现过的全部对，按 code、ORG_CL 排序
  code、ORG_CL    公司代码、券商编号（ORG_CL 与 report_accuracy 同为浮点数）
  n_reports       研报范围里该对（有 ORG_CL）的研报篇数
  last_deep_dt    其中最晚的发布日
  EVAL_RPT_ID25   被评价的预测（券商 2024 年内对该公司 FY2025 归母净利润的最后一次预测）的 RPT_ID；该券商 2024 年没有这种预测时缺失
  EVAL_DT25       被评价预测的发布日
  EVAL_F25        被评价预测的预测值（元）
  EVAL_T25        被评价预测行的 RPT_YR_T（控制变量里 BM、ROA_T 的年报年度）
  A25             FY2025 归母净利润实际值
  AFE25           被评价预测的绝对误差
  N_GROUP25       比较组家数（2024 年内对该公司 FY2025 做过预测的券商数，含本券商）
  AGE25           被评价预测发布日到 2025-12-31 的天数
  MAFE25          比较组平均绝对误差（含本券商）；比较组 < 4 家或组内有实际值缺失时缺失
  PMAFE_FY25      (AFE25 − MAFE25)/MAFE25；同上条件缺失
  DAGE25          时效差 = AGE25 − 比较组 AGE 的均值；同上条件缺失
  EVAL_IS_DEEP    被评价预测是否正好是该对名下某篇研报本身的预测
  EVAL_GAP        EVAL_DT25 − last_deep_dt（天）
  SIZE_M、BM、MOM、VOL、ROA_T、LNAGE  控制变量在 EVAL_DT25 上的取值（T = EVAL_T25；算法同研报层面，小数、未缩尾）
"""
import pandas as pd

from src.config import out_dir

STEP = 'accuracy'


def _read(name, columns=None, out=None):
    return pd.read_parquet((out or out_dir(STEP)) / name, columns=columns)


def report_accuracy(columns=None, out=None):
    """每篇研报一行的预测准确度、比较组变量和控制变量（列见模块说明）。out：另指定的输出目录。"""
    return _read('report_accuracy.parquet', columns, out)


def report_peers(columns=None, out=None):
    """FY2025、[d − 90, d] 比较组的同行明细（列见模块说明）。"""
    return _read('report_peers.parquet', columns, out)


def pair_accuracy(columns=None, out=None):
    """每个（公司，券商）一行的被评价预测、PMAFE、DAGE 和控制变量（列见模块说明）。"""
    return _read('pair_accuracy.parquet', columns, out)
