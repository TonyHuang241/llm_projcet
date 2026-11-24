"""研报 CAR（及换手率、异常换手率）与概念 DAG 指标页面：把 src/report_car_metrics.py 算好的逐篇表和自检数字填进
car_metrics_template.html 的 __DATA__，写成一个自包含 html（数据内嵌，双击打开）。由 experiment_car_metrics.py 调用。
目标节点只认名称恰好为“投资评级”的概念（src/valuation_sinks.TARGET）。"""
import json
import math
import time
from pathlib import Path

from src.report_car_metrics import ATURNS, CARS, METRICS, TURNS
from src.valuation_sinks import TARGET

TEMPLATE = (Path(__file__).parent / 'car_metrics_template.html').read_text(encoding='utf-8')


def _clean(v, nd=6):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, float):
        return round(v, nd)
    return v


def main(df, check, page):
    cols = ['report_id', 'code', 'company', 'broker', 'date', 'day0'] + CARS + TURNS + ATURNS + METRICS  # PRETURN 不进页面
    rows = []
    for r in df[cols].astype(object).where(df[cols].notna(), None).to_dict('records'):
        rows.append({k: _clean(int(v) if k in ('NPATH', 'V_NPATH', 'V_HAS') and v is not None else v) for k, v in r.items()})
    data = {'generated': time.strftime('%Y-%m-%d %H:%M'), 'rows': rows, 'target': TARGET,
            'check': {k: int(v) for k, v in check.items()},
            'n_companies': int(df.code.nunique()), 'n_brokers': int(df.broker.nunique()),
            'date_min': str(df.date.min()), 'date_max': str(df.date.max())}
    blob = json.dumps(data, ensure_ascii=False, separators=(',', ':'), allow_nan=False).replace('</', '<\\/')
    page.write_text(TEMPLATE.replace('__DATA__', blob), encoding='utf-8')
    print(f'页面：{page}（{page.stat().st_size / 1024:.0f} KB）')
    return page
