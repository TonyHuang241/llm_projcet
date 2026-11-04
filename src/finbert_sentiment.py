"""用中文FinBERT对审计报告和MD&A文本做情感分析。

模型：yiyanghkust/finbert-tone-chinese（bert-base-chinese 在金融文本上微调），对单句分类：中性/正面/负面。
模型一次最多读512个token，所以先把文本切成句子逐句分类，再按文档汇总：
各类句子的占比，以及净语调 tone = (正面句数 - 负面句数) / 全部句数，即正面占比 - 负面占比，取值 -1 ~ 1。
（分母用全部句数而不是正负句数之和：审计报告绝大多数句子是中性，正负句往往只有一两句，后者会被放大成 ±1）

用法：在 main.py 中设置 TASK = 'finbert_sentiment'
"""
import csv
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = 'yiyanghkust/finbert-tone-chinese'
LABELS = ['neutral', 'positive', 'negative']  # 与模型 config 中 id2label 的顺序一致
SENTENCE_RE = re.compile(r'[^。！？；]+[。！？；]')


def split_sentences(text):
    """逐行切句；不以句号等结尾的小标题不参与分析。"""
    return [s for line in text.splitlines() for s in SENTENCE_RE.findall(line) if len(s) >= 5]


@torch.no_grad()
def classify(sentences, tokenizer, model, batch_size):
    """返回每句的三类概率（顺序同LABELS）。先按长度排序再分批，减少padding、加快速度。"""
    order = sorted(range(len(sentences)), key=lambda i: len(sentences[i]))
    probs = [None] * len(sentences)
    for k in range(0, len(order), batch_size):
        idx = order[k:k + batch_size]
        batch = tokenizer([sentences[i] for i in idx], padding=True, truncation=True,
                          max_length=512, return_tensors='pt').to(model.device)
        for i, row in zip(idx, model(**batch).logits.softmax(-1).tolist()):
            probs[i] = row
    return probs


def summarize(probs):
    """把逐句结果汇总成文档级指标。"""
    n = len(probs)
    counts = [0, 0, 0]
    for row in probs:
        counts[row.index(max(row))] += 1
    neu, pos, neg = counts
    return {
        'n_sentences': n,
        'pos_ratio': round(pos / n, 4),
        'neg_ratio': round(neg / n, 4),
        'neu_ratio': round(neu / n, 4),
        'tone': round((pos - neg) / n, 4),
        'avg_pos_prob': round(sum(row[1] for row in probs) / n, 4),
        'avg_neg_prob': round(sum(row[2] for row in probs) / n, 4),
    }


def main(audit_dir, mda_dir, out_path, limit=100, batch_size=64):
    """对 audit_dir 中按文件名排序的前 limit 家公司，分析其审计报告和同名的MD&A文本。"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('使用设备：', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU（未检测到可用GPU）')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()

    names = sorted(p.name for p in Path(audit_dir).glob('*.txt'))[:limit]
    rows, start = [], time.time()
    for k, name in enumerate(names, 1):
        code, _, company = name.split('_')[:3]
        for section, path in (('audit', Path(audit_dir) / name), ('mda', Path(mda_dir) / name)):
            sentences = split_sentences(path.read_text(encoding='utf-8')) if path.exists() else []
            if not sentences:
                print(f'跳过 {name} 的 {section}：无文本')
                continue
            probs = classify(sentences, tokenizer, model, batch_size)
            rows.append({'code': code, 'company': company, 'section': section, **summarize(probs)})
        if k % 10 == 0:
            print(f'已完成 {k} / {len(names)} 家，用时 {time.time() - start:.0f} 秒')

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w', newline='', encoding='utf-8-sig') as f:  # utf-8-sig：Excel 打开不乱码
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'共 {len(rows)} 条结果，保存在 {out}')
