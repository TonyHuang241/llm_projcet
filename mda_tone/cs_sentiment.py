"""把 GLM 中性化前后的逐句 FinBERT 情感汇总成公司*年份面板数据。

输入：compare_dir 中 compare_neutral_tone 的结果 tone_by_sentence.csv（逐句概率）、tone_by_doc.csv（公司简称）。
单句情感分数 = P(正面) - P(负面)，取值 -1 ~ 1。
输出 out_path，每家公司每份年报一行（year 为年报所属年份）：
  - n_sentences：句子数
  - sent_mean_orig：原文各句情感的平均值
  - sent_mean_neutral：中性化后各句情感的平均值
  - sent_change_mean：各句情感变化（中性化后 - 原文）的平均值
  - sent_std_orig：原文各句情感的样本标准差（ddof=1），只有一句时为空
  - pos_change_mean：原句被判为正面的句子上，情感变化的平均值（越负下调越多），没有正面句时为空
  - sent_std_neutral：中性化后各句情感的样本标准差（ddof=1），只有一句时为空

用法：在 main.py 中设置 TASK = 'cs_sentiment'
"""
import csv
import itertools
from pathlib import Path

import numpy as np

FIELDS = ['code', 'year', 'company', 'n_sentences', 'sent_mean_orig', 'sent_mean_neutral',
          'sent_change_mean', 'sent_std_orig', 'pos_change_mean', 'sent_std_neutral']
csv.field_size_limit(10 ** 9)  # 个别句子可能超过 csv 默认的字段长度上限


def doc_metrics(rows):
    """由一份年报的逐句结果计算各项指标。"""
    orig = np.array([float(r['pos_orig']) - float(r['neg_orig']) for r in rows])
    neutral = np.array([float(r['pos_neutral']) - float(r['neg_neutral']) for r in rows])
    positive = np.array([r['label_orig'] == 'positive' for r in rows])
    return {
        'n_sentences': len(rows),
        'sent_mean_orig': round(float(orig.mean()), 6),
        'sent_mean_neutral': round(float(neutral.mean()), 6),
        'sent_change_mean': round(float((neutral - orig).mean()), 6),
        'sent_std_orig': round(float(orig.std(ddof=1)), 6) if len(rows) > 1 else '',
        'pos_change_mean': round(float((neutral - orig)[positive].mean()), 6) if positive.any() else '',
        'sent_std_neutral': round(float(neutral.std(ddof=1)), 6) if len(rows) > 1 else '',
    }


def main(compare_dir, out_path):
    compare_dir, out_path = Path(compare_dir), Path(out_path)
    with (compare_dir / 'tone_by_doc.csv').open(encoding='utf-8-sig', newline='') as f:
        companies = {(r['year'], r['code']): r['company'] for r in csv.DictReader(f)}

    panel = []
    # 结果文件中同一篇文章的句子是连续的，可以边读边按文章处理，不占内存
    with (compare_dir / 'tone_by_sentence.csv').open(encoding='utf-8-sig', newline='') as f:
        for (year, code), group in itertools.groupby(csv.DictReader(f), key=lambda r: (r['year'], r['code'])):
            panel.append({'code': code, 'year': year, 'company': companies[year, code], **doc_metrics(list(group))})
    if len(panel) != len(companies):  # 句子不连续会多出重复行，缺文章会少行
        raise SystemExit(f'面板 {len(panel)} 行与 tone_by_doc.csv 的 {len(companies)} 篇不一致，请检查结果文件')

    panel.sort(key=lambda r: (r['code'], r['year']))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8-sig', newline='') as f:  # utf-8-sig：Excel 打开不乱码
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(panel)
    years = sorted({r['year'] for r in panel})
    print(f'共 {len(panel)} 行，{len({r["code"] for r in panel})} 家公司，{years[0]}-{years[-1]} 年，结果保存在 {out_path}')
