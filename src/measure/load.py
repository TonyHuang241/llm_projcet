"""读 measure 步骤的成品：每篇研报的逻辑结构变量表（output/measure/report_measures.parquet，列见 src/measure/__init__.py）。
下游步骤通过这里读，不自己拼路径。"""
import pandas as pd

from src.measure import OUT_PATH


def read_measures(columns=None, report_ids=None, path=OUT_PATH):
    """逻辑结构变量表（每篇研报一行，原始值），返回 pandas.DataFrame；columns 只读这些列（report_id 总会读），
    report_ids 给出时只要这些研报（行序同表里）。"""
    cols = None if columns is None else list(dict.fromkeys(['report_id', *columns]))
    t = pd.read_parquet(path, columns=cols)
    if report_ids is not None:
        t = t[t.report_id.isin(set(report_ids))].reset_index(drop=True)
    return t
