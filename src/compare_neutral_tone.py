"""用中文FinBERT比较MD&A中性化改写前后的语调，所有年份的结果合并输出，支持断点续跑。

输入：压缩包 neutral_zip 中的 <顶层文件夹>/<年份>/*.json（不需要解压），每句含 original（原句）和 neutral（改写句），两者逐句对应。
模型：yiyanghkust/finbert-tone-chinese（bert-base-chinese 在金融文本上微调），对单句分类：中性/正面/负面。
同一句的原句和改写句都逐句分类（模型一次最多读512个token），再分别按文档汇总：
各类句子的占比，以及净语调 tone = (正面句数 - 负面句数) / 全部句数，即正面占比 - 负面占比，取值 -1 ~ 1。
输出（都在 out_dir 中，所有年份合在一起，用 year 列区分）：
  - tone_by_doc.csv：每篇文章改写前后的各项指标及差值
  - tone_by_sentence.csv：每句改写前后的类别和概率，便于查看具体哪些句子的语调变了
  - _progress.jsonl：已完成的文章及当时两个 CSV 的大小
  - 终端打印：文档平均语调、句子类别分布、类别转移矩阵、配对t检验

断点续跑：每分析完一篇就写入磁盘，随时可以停止（包括强制结束），重新运行时自动跳过已完成的文章，
并把两个 CSV 截断到最后一篇完成时的大小，丢弃写了一半的内容。
已有结果：legacy_dir 中没有 year 列的旧 CSV 会被导入（视为 legacy_year 年），不会重新分析，旧文件也不会被修改；
旧结果中的句子必须与当前 JSON 完全一致才会导入，不一致的文章会重新分析。

用法：在 main.py 中设置 TASK = 'compare_neutral_tone'
"""
import csv
import io
import itertools
import json
import math
import os
import statistics
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = 'yiyanghkust/finbert-tone-chinese'
LABELS = ['neutral', 'positive', 'negative']  # 与模型 config 中 id2label 的顺序一致
LABEL_CN ={'neutral': '中性', 'positive': '正面', 'negative': '负面'}
METRICS = ['tone', 'pos_ratio', 'neg_ratio', 'neu_ratio', 'avg_pos_prob', 'avg_neg_prob']
DOC_FIELDS = ['year', 'code', 'company', 'n_sentences'] + [
    f'{m}_{s}' for m in METRICS for s in ('orig', 'neutral', 'diff')]
SENT_FIELDS = ['year', 'code', 'id', 'label_orig', 'label_neutral', 'pos_orig', 'neg_orig',
               'pos_neutral', 'neg_neutral', 'original', 'neutral']
BOM = b'\xef\xbb\xbf'  # utf-8-sig：Excel 打开不乱码
csv.field_size_limit(10 ** 9)  # 个别句子可能超过 csv 默认的字段长度上限


def list_sources(zf):
    """返回 {年份: {股票代码: 压缩包内的 JSON 路径}}。只取直接位于年份文件夹下的 JSON，txt、日志等都忽略。"""
    sources = {}
    for name in sorted(zf.namelist()):
        p = PurePosixPath(name)
        if p.suffix == '.json' and p.parent.name.isdigit():
            sources.setdefault(p.parent.name, {})[p.name.split('_')[0]] = name
    return dict(sorted(sources.items()))


def load_pairs(zf, name):
    return json.loads(zf.read(name).decode('utf-8'))


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


def encode_rows(fields, rows, header=False):
    buf = io.StringIO(newline='')
    writer = csv.DictWriter(buf, fieldnames=fields)
    if header:
        writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode('utf-8')


class ResultStore:
    """合并结果文件。一篇文章的两部分结果都落盘后才写进度记录，所以进度记录之后的内容都可以安全丢弃。"""

    def __init__(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        self.doc_path, self.sent_path = out / 'tone_by_doc.csv', out / 'tone_by_sentence.csv'
        self.log_path = out / '_progress.jsonl'
        if not self.log_path.exists():
            if self.doc_path.exists() or self.sent_path.exists():  # 没有进度记录时无法判断内容是否完整，不动它
                raise SystemExit(f'{out} 中有结果文件但没有 _progress.jsonl，请先确认后移走这些文件')
            self.log_path.touch()
        self.done, doc_end, sent_end, good = set(), 0, 0, 0
        with self.log_path.open('rb') as f:
            for line in f:
                if not line.endswith(b'\n'):  # 中断时写了一半的进度记录
                    break
                rec = json.loads(line)
                self.done.add((rec['year'], rec['code']))
                doc_end, sent_end, good = rec['doc_end'], rec['sent_end'], good + len(line)
        self._truncate(self.log_path, good)
        self._truncate(self.doc_path, doc_end)
        self._truncate(self.sent_path, sent_end)
        self.doc_f, self.sent_f = self.doc_path.open('ab'), self.sent_path.open('ab')
        self.log_f = self.log_path.open('ab')
        if doc_end == 0:
            self.doc_f.write(BOM + encode_rows(DOC_FIELDS, [], header=True))
            self.sent_f.write(BOM + encode_rows(SENT_FIELDS, [], header=True))

    @staticmethod
    def _truncate(path, end):
        size = path.stat().st_size if path.exists() else 0
        if size < end:
            raise SystemExit(f'{path} 比进度记录中的短（{size} < {end}），文件可能被改动过，请先检查')
        if size > end:
            with path.open('r+b') as f:
                f.truncate(end)

    def add(self, doc_row, sent_rows):
        self.sent_f.write(encode_rows(SENT_FIELDS, sent_rows))
        self.doc_f.write(encode_rows(DOC_FIELDS, [doc_row]))
        for f in (self.sent_f, self.doc_f):
            f.flush()
            os.fsync(f.fileno())
        rec = {'year': doc_row['year'], 'code': doc_row['code'],
               'doc_end': self.doc_f.tell(), 'sent_end': self.sent_f.tell()}
        self.log_f.write(json.dumps(rec).encode('utf-8') + b'\n')
        self.log_f.flush()
        os.fsync(self.log_f.fileno())
        self.done.add((doc_row['year'], doc_row['code']))

    def close(self):
        for f in (self.doc_f, self.sent_f, self.log_f):
            f.close()


def import_legacy(store, legacy_dir, legacy_year, sources, zf):
    """导入没有 year 列的旧结果。旧 CSV 中同一篇文章的句子是连续的，所以可以边读边按文章处理。"""
    doc_path, sent_path = Path(legacy_dir) / 'tone_by_doc.csv', Path(legacy_dir) / 'tone_by_sentence.csv'
    if not doc_path.exists() or not sent_path.exists():
        return
    with doc_path.open(encoding='utf-8-sig', newline='') as f:
        docs = {r['code']: r for r in csv.DictReader(f)}
    paths = sources.get(legacy_year, {})
    pending = {c for c in docs if c in paths and (legacy_year, c) not in store.done}
    if not pending:
        return
    t0, n_ok, changed = time.time(), 0, []
    with sent_path.open(encoding='utf-8-sig', newline='') as f:
        for code, group in itertools.groupby(csv.DictReader(f), key=lambda r: r['code']):
            if code not in pending:
                continue
            rows = list(group)
            pairs = load_pairs(zf, paths[code])
            if [(r['original'], r['neutral']) for r in rows] != [(p['original'], p['neutral']) for p in pairs]:
                changed.append(code)  # 改写结果已变化，旧分析不再有效，留给后面重新分析
                continue
            store.add({'year': legacy_year, **docs[code]}, [{'year': legacy_year, **r} for r in rows])
            n_ok += 1
    print(f'导入已有结果：{legacy_year} 年 {n_ok} 篇，用时 {time.time() - t0:.0f} 秒；'
          f'句子与当前 JSON 不一致、需要重新分析 {len(changed)} 篇{changed[:20] if changed else ""}', flush=True)


def analyze(zf, name, year, tokenizer, model, batch_size):
    """分析压缩包中的一篇文章，返回 (文档行, 句子行列表)。"""
    code, _, company = PurePosixPath(name).stem.split('_')[:3]
    pairs = load_pairs(zf, name)
    p_orig = classify([p['original'] for p in pairs], tokenizer, model, batch_size)
    p_neu = classify([p['neutral'] for p in pairs], tokenizer, model, batch_size)
    s_orig, s_neu = summarize(p_orig), summarize(p_neu)
    row = {'year': year, 'code': code, 'company': company, 'n_sentences': len(pairs)}
    for m in METRICS:
        row[f'{m}_orig'], row[f'{m}_neutral'] = s_orig[m], s_neu[m]
        row[f'{m}_diff'] = round(s_neu[m] - s_orig[m], 4)
    sent_rows = [{
        'year': year, 'code': code, 'id': p['id'],
        'label_orig': LABELS[a.index(max(a))], 'label_neutral': LABELS[b.index(max(b))],
        'pos_orig': round(a[1], 4), 'neg_orig': round(a[2], 4),
        'pos_neutral': round(b[1], 4), 'neg_neutral': round(b[2], 4),
        'original': p['original'], 'neutral': p['neutral'],
    } for p, a, b in zip(pairs, p_orig, p_neu)]
    return row, sent_rows


def paired_t(diffs):
    """配对t检验：返回 (t值, 双侧p值)，p值用正态近似（样本量较大时与t分布几乎一致）。"""
    n, mean, sd = len(diffs), statistics.mean(diffs), statistics.stdev(diffs)
    t = mean / (sd / math.sqrt(n))
    return t, math.erfc(abs(t) / math.sqrt(2))


def report(store):
    """从合并后的结果文件计算统计量（逐行读取，句子文件很大时也不占内存）。"""
    with store.doc_path.open(encoding='utf-8-sig', newline='') as f:
        doc_rows = list(csv.DictReader(f))
    if len(doc_rows) < 2:
        print(f'目前只有 {len(doc_rows)} 篇，不做统计')
        return
    years = Counter(r['year'] for r in doc_rows)
    print(f'\n===== 文档层面（{len(doc_rows)} 篇，各指标为文档平均值）=====')
    print('各年篇数：' + '；'.join(f'{y} 年 {years[y]} 篇' for y in sorted(years)))
    print(f'{"指标":<14}{"改写前":>9}{"改写后":>9}{"差值":>9}{"t值":>9}{"p值":>10}')
    for m in METRICS:
        a = [float(r[f'{m}_orig']) for r in doc_rows]
        b = [float(r[f'{m}_neutral']) for r in doc_rows]
        t, p = paired_t([float(r[f'{m}_diff']) for r in doc_rows])
        print(f'{m:<14}{statistics.mean(a):>9.4f}{statistics.mean(b):>9.4f}'
              f'{statistics.mean(b) - statistics.mean(a):>+9.4f}{t:>9.2f}'
              f'{"<0.001" if p < 0.001 else f"{p:.3f}":>10}')  # 正态近似在极小p值处偏小，不报具体数
    tones = sorted(float(r['tone_diff']) for r in doc_rows)
    print(f'tone 下降的文章 {sum(d < 0 for d in tones)} 篇，上升 {sum(d > 0 for d in tones)} 篇；'
          f'tone 差值的中位数 {statistics.median(tones):+.4f}，范围 {tones[0]:+.4f} ~ {tones[-1]:+.4f}')

    trans = Counter()
    with store.sent_path.open(encoding='utf-8-sig', newline='') as f:
        for r in csv.DictReader(f):
            trans[r['label_orig'], r['label_neutral']] += 1
    n = sum(trans.values())
    print(f'\n===== 句子层面（{n} 句）=====')
    print('类别分布：' + '；'.join(
        f'{LABEL_CN[l]} {sum(trans[l, b] for b in LABELS) / n:.1%} -> '
        f'{sum(trans[a, l] for a in LABELS) / n:.1%}' for l in LABELS))
    print('类别转移矩阵（行=改写前，列=改写后，单位：句）：')
    print(' ' * 8 + ''.join(f'{LABEL_CN[l]:>8}' for l in LABELS))
    for a in LABELS:
        print(f'{LABEL_CN[a]:<8}' + ''.join(f'{trans[a, b]:>10}' for b in LABELS))


def main(neutral_zip, out_dir, legacy_dir=None, legacy_year=None, batch_size=64, years=None, limit=None):
    """neutral_zip：中性化结果的压缩包，其中每个以年份命名的文件夹都会被分析（其他内容忽略）。
    legacy_dir、legacy_year：已有的旧结果及其年份，None 表示没有。
    years、limit：试跑时只分析这些年份（如 ['2023']）、每年按文件名排序的前 limit 篇，None 表示全部。
    试跑结果同样写入 out_dir，之后去掉限制重新运行会接着分析其余文章。"""
    store = ResultStore(out_dir)
    with zipfile.ZipFile(neutral_zip) as zf:
        sources = {y: dict(list(files.items())[:limit]) for y, files in list_sources(zf).items()
                   if years is None or y in years}
        if legacy_dir:
            import_legacy(store, legacy_dir, legacy_year, sources, zf)
        todo = [(y, f) for y, files in sources.items() for c, f in files.items() if (y, c) not in store.done]
        print('文章数：' + '；'.join(f'{y} 年 {len(files)} 篇' for y, files in sources.items()) +
              f'；已完成 {len(store.done)} 篇，本次分析 {len(todo)} 篇', flush=True)

        if todo:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print('使用设备：', torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU（未检测到可用GPU）', flush=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device).eval()
            t0, n_sent = time.time(), 0
            for k, (year, name) in enumerate(todo, 1):
                doc_row, sent_rows = analyze(zf, name, year, tokenizer, model, batch_size)
                store.add(doc_row, sent_rows)
                n_sent += len(sent_rows)
                if k % 50 == 0 or k == len(todo):  # 每50篇报告一次进度
                    sec = time.time() - t0
                    print(f'[{k}/{len(todo)}] 已分析 {n_sent} 句（改写前后各一遍），'
                          f'{2 * n_sent / sec:.0f} 句/秒，用时 {sec / 60:.1f} 分钟，'
                          f'预计还需 {sec / k * (len(todo) - k) / 60:.1f} 分钟', flush=True)
    report(store)
    store.close()
    print(f'\n结果保存在 {out_dir}')
