"""用中文FinBERT比较MD&A中性化改写前后的语调。

输入：data/mda_neutral/<年份>/*.json，每句含 original（原句）和 neutral（改写句），两者逐句对应。
同一句的原句和改写句都用 finbert_sentiment 中的模型分类，再分别汇总成文档级指标。
输出：
  - tone_by_doc.csv：每篇文章改写前后的各项指标及差值
  - tone_by_sentence.csv：每句改写前后的类别和概率，便于查看具体哪些句子的语调变了
  - 终端打印：文档平均语调、句子类别分布、类别转移矩阵、配对t检验

用法：在 main.py 中设置 TASK = 'compare_neutral_tone'
"""
import csv
import json
import math
import statistics
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.finbert_sentiment import LABELS, MODEL_NAME, classify, summarize

LABEL_CN = {'neutral': '中性', 'positive': '正面', 'negative': '负面'}
METRICS = ['tone', 'pos_ratio', 'neg_ratio', 'neu_ratio', 'avg_pos_prob', 'avg_neg_prob']


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig：Excel 打开不乱码
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def paired_t(diffs):
    """配对t检验：返回 (t值, 双侧p值)，p值用正态近似（样本量较大时与t分布几乎一致）。"""
    n, mean, sd = len(diffs), statistics.mean(diffs), statistics.stdev(diffs)
    t = mean / (sd / math.sqrt(n))
    return t, math.erfc(abs(t) / math.sqrt(2))


def main(neutral_dir, out_dir, batch_size=64):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备：', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU（未检测到可用GPU）')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    doc_rows, sent_rows = [], []
    paths = sorted(Path(neutral_dir).glob('*.json'))
    t0 = time.time()
    for k, path in enumerate(paths, 1):
        code, _, company = path.stem.split('_')[:3]
        pairs = json.loads(path.read_text(encoding='utf-8'))
        p_orig = classify([p['original'] for p in pairs], tokenizer, model, batch_size)
        p_neu = classify([p['neutral'] for p in pairs], tokenizer, model, batch_size)
        s_orig, s_neu = summarize(p_orig), summarize(p_neu)
        row = {'code': code, 'company': company, 'n_sentences': len(pairs)}
        for m in METRICS:
            row[f'{m}_orig'], row[f'{m}_neutral'] = s_orig[m], s_neu[m]
            row[f'{m}_diff'] = round(s_neu[m] - s_orig[m], 4)
        doc_rows.append(row)
        for p, a, b in zip(pairs, p_orig, p_neu):
            sent_rows.append({
                'code': code, 'id': p['id'],
                'label_orig': LABELS[a.index(max(a))], 'label_neutral': LABELS[b.index(max(b))],
                'pos_orig': round(a[1], 4), 'neg_orig': round(a[2], 4),
                'pos_neutral': round(b[1], 4), 'neg_neutral': round(b[2], 4),
                'original': p['original'], 'neutral': p['neutral'],
            })
        if k % 50 == 0 or k == len(paths):  # 每50篇报告一次进度
            sec = time.time() - t0
            print(f'[{k}/{len(paths)}] 已分析 {len(sent_rows)} 句（改写前后各一遍），'
                  f'{2 * len(sent_rows) / sec:.0f} 句/秒，用时 {sec / 60:.1f} 分钟，'
                  f'预计还需 {sec / k * (len(paths) - k) / 60:.1f} 分钟', flush=True)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / 'tone_by_doc.csv', doc_rows)
    write_csv(out / 'tone_by_sentence.csv', sent_rows)

    print(f'\n===== 文档层面（{len(doc_rows)} 篇，各指标为文档平均值）=====')
    print(f'{"指标":<14}{"改写前":>9}{"改写后":>9}{"差值":>9}{"t值":>9}{"p值":>10}')
    for m in METRICS:
        a = [r[f'{m}_orig'] for r in doc_rows]
        b = [r[f'{m}_neutral'] for r in doc_rows]
        t, p = paired_t([r[f'{m}_diff'] for r in doc_rows])
        print(f'{m:<14}{statistics.mean(a):>9.4f}{statistics.mean(b):>9.4f}'
              f'{statistics.mean(b) - statistics.mean(a):>+9.4f}{t:>9.2f}'
              f'{"<0.001" if p < 0.001 else f"{p:.3f}":>10}')  # 正态近似在极小p值处偏小，不报具体数
    tones = sorted(r['tone_diff'] for r in doc_rows)
    print(f'tone 下降的文章 {sum(d < 0 for d in tones)} 篇，上升 {sum(d > 0 for d in tones)} 篇；'
          f'tone 差值的中位数 {statistics.median(tones):+.4f}，范围 {tones[0]:+.4f} ~ {tones[-1]:+.4f}')

    n = len(sent_rows)
    print(f'\n===== 句子层面（{n} 句）=====')
    print('类别分布：' + '；'.join(
        f'{LABEL_CN[l]} {sum(r["label_orig"] == l for r in sent_rows) / n:.1%} -> '
        f'{sum(r["label_neutral"] == l for r in sent_rows) / n:.1%}' for l in LABELS))
    print('类别转移矩阵（行=改写前，列=改写后，单位：句）：')
    print(' ' * 8 + ''.join(f'{LABEL_CN[l]:>8}' for l in LABELS))
    for a in LABELS:
        print(f'{LABEL_CN[a]:<8}' + ''.join(
            f'{sum(r["label_orig"] == a and r["label_neutral"] == b for r in sent_rows):>10}' for b in LABELS))
    print(f'\n结果保存在 {out}')
