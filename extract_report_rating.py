"""用规则从研报原文识别本次个股评级、评级动作，并按券商自己的评级体系定档位（不调用大模型，可反复运行）。规则见 src/report_rating.py 开头。
样本是 experiment_report_car.py 的 790 篇研报；券商评级体系用语料库（data/hs300_2024_reports.parquet）里这些券商的全部研报汇总。

在仓库根目录运行：
    python -X utf8 extract_report_rating.py

结果在 output_dag/experiments/report_car/rating/：
    report_rating.parquet / .csv     每篇一行：rating_raw、action、tier、TOP、tier_source、tier_fb、rule、evidence、conflict 等
    broker_rating_scale.csv          券商评级体系表：档位顺序、最高档、次高档、来源（评级说明 / 实际用词）、解析篇数、是否一致
    broker_scale_defs.csv            语料库里每篇解析到的评级说明顺序；broker_header_ratings.csv：语料库每篇个股研报识别出的评级（算实际用词）
"""
import sys
import time

from src.config import DATA_DIR, ROOT_DIR

# ==================== 运行参数 ====================
PANEL = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car' / 'report_car_panel.parquet'
META = DATA_DIR / 'hs300_2024_reports.parquet'  # report_id -> 原文文件
TEXT_DIR = DATA_DIR / 'HS300_reports_2024'
OUT_DIR = ROOT_DIR / 'output_dag' / 'experiments' / 'report_car' / 'rating'
# ==================================================

if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, 'reconfigure'):
            stream.reconfigure(encoding='utf-8', errors='replace')
    from src.report_rating import run
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out, st, defs, use = run(PANEL, META, TEXT_DIR)
    out.to_parquet(OUT_DIR / 'report_rating.parquet', index=False)
    out.to_csv(OUT_DIR / 'report_rating.csv', index=False, encoding='utf-8-sig')
    st.to_csv(OUT_DIR / 'broker_rating_scale.csv', index=False, encoding='utf-8-sig')
    defs.to_csv(OUT_DIR / 'broker_scale_defs.csv', index=False, encoding='utf-8-sig')
    use.to_csv(OUT_DIR / 'broker_header_ratings.csv', index=False, encoding='utf-8-sig')
    print(f'{len(out)} 篇；语料库评级说明 {len(defs)} 篇、识别出评级的个股研报 {len(use)} 篇（{time.time() - t0:.0f} 秒）')
    for c in ['action', 'tier', 'TOP', 'tier_source', 'conflict']:
        print(f'  {c}：' + '，'.join(f'{k} {v}' for k, v in out[c].value_counts(dropna=False).items()))
    print('  rating_raw：' + '，'.join(f'{k} {v}' for k, v in out.rating_raw.value_counts(dropna=False).items()))
