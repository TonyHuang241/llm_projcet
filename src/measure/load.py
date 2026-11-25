"""读 measure 步骤的成品：每篇研报的逻辑结构变量表（各版本一张 output/measure/<版本>/report_measures.parquet，列见
src/measure/__init__.py）。下游步骤通过这里读，不自己拼路径。

按样本读：样本清单决定每篇取自哪个版本的表；各版本按样本登记的版本顺序上下拼接，版本内保持表里的行序（同库的研报表），
与 src/graph/load.py 的 read_table 行序相同。version 参数同 src/samples 的 assignment。
"""
import pandas as pd

from src import samples
from src.config import DEFAULT_SAMPLE
from src.measure import out_path


def read_version(version, columns=None):
    """一个版本的整张变量表（库里每篇一行，不按样本筛选）。"""
    p = out_path(version)
    if not p.exists():
        raise SystemExit(f'缺少 {p}：先运行 python run.py measure --version {version}')
    return pd.read_parquet(p, columns=columns)


def read_measures(columns=None, report_ids=None, sample=DEFAULT_SAMPLE, version=None):
    """样本的逻辑结构变量表（每篇研报一行，原始值），返回 pandas.DataFrame；columns 只读这些列（report_id 总会读；不给时整张表，含 version 列），
    report_ids 给出时只要这些研报（行序同上）。"""
    t, order = samples.assignment(sample, version)
    want = None if report_ids is None else set(report_ids)
    plan = {rid: v for rid, v in zip(t.report_id, t.version) if want is None or rid in want}
    cols = None if columns is None else list(dict.fromkeys(['report_id', *columns]))
    parts = []
    for v in dict.fromkeys(order):
        m = read_version(v, cols)
        parts.append(m[[plan.get(r) == v for r in m.report_id]])
    return pd.concat(parts, ignore_index=True)
