"""把中性化改写前后的 FinBERT 语调变化画成交互式 HTML 报告（单文件，离线可用，不依赖外部库）。

两个分析：
  1. 文章间：每篇文章的平均语调变化 tone_diff（= 改写后 tone - 改写前 tone，tone = 正面句占比 - 负面句占比），
     来自 tone_by_doc.csv，每篇一个值。画分布密度曲线（核密度估计，最高点缩放为 1），
     实线 = 中位数，虚线 = 平均数，浅色区域 = 中间 95% 的数据（2.5% ~ 97.5% 分位数）。
  2. 文章内位置：语调变化发生在文章的什么位置。
     每句的语调变化 = (改写后 正面概率 - 负面概率) - (改写前 正面概率 - 负面概率)，来自 tone_by_sentence.csv。
     第 i 句（从0计）的相对位置 = (i + 0.5) / 句数，把 0~1 等分成 N_BINS 段；每篇文章先在每段内对句子取平均，
     再在每段上汇总所有文章：实线 = 中位数，虚线 = 平均数；浅色带可用按钮切换为
     (a) 各篇文章的中间 95%（2.5% ~ 97.5% 分位数），或 (b) 平均数的 95% 置信区间（平均数 ± 1.96·σ/√n）。

用法：在 main.py 中设置 TASK = 'plot_tone_change'
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BINS = 50  # 文章相对位置分成多少段


def load_docs(compare_dir):
    docs = list(csv.DictReader(open(compare_dir / 'tone_by_doc.csv', encoding='utf-8-sig')))
    return {'codes': [d['code'] for d in docs], 'names': [d['company'] for d in docs],
            'diffs': [float(d['tone_diff']) for d in docs]}


def position_profile(compare_dir, n_bins=N_BINS):
    """返回每段的汇总统计（中位数、平均数、两种95%区间、文章数）。"""
    sents = defaultdict(list)  # 股票代码 -> [(句子id, 语调变化)]
    with open(compare_dir / 'tone_by_sentence.csv', encoding='utf-8-sig') as f:
        for r in csv.DictReader(f):
            d = (float(r['pos_neutral']) - float(r['neg_neutral'])) - (float(r['pos_orig']) - float(r['neg_orig']))
            sents[r['code']].append((int(r['id']), d))

    M = np.full((len(sents), n_bins), np.nan)  # 每篇文章 × 每段的平均语调变化，空段为 NaN
    for k, rows in enumerate(sents.values()):
        d = np.array([v for _, v in sorted(rows)])
        n = len(d)
        b = np.minimum(((np.arange(n) + 0.5) / n * n_bins).astype(int), n_bins - 1)
        cnt = np.bincount(b, minlength=n_bins)
        s = np.bincount(b, weights=d, minlength=n_bins)
        M[k, cnt > 0] = s[cnt > 0] / cnt[cnt > 0]

    out = {k: [] for k in ('x0', 'x1', 'median', 'mean', 'p_lo', 'p_hi', 'ci_lo', 'ci_hi', 'n')}
    for j in range(n_bins):
        v = M[:, j][~np.isnan(M[:, j])]
        mean, sd = v.mean(), v.std(ddof=1)
        half = 1.96 * sd / np.sqrt(len(v))
        for key, val in (('x0', j / n_bins), ('x1', (j + 1) / n_bins), ('median', np.median(v)), ('mean', mean),
                         ('p_lo', np.percentile(v, 2.5)), ('p_hi', np.percentile(v, 97.5)),
                         ('ci_lo', mean - half), ('ci_hi', mean + half), ('n', len(v))):
            out[key].append(round(float(val), 5) if key != 'n' else int(val))
    out['n_docs'] = len(sents)
    out['n_sents'] = sum(len(r) for r in sents.values())
    return out


def main(compare_dir, out_name='tone_change_distribution.html'):
    compare_dir = Path(compare_dir)
    print('读取数据……')
    doc_data, pos_data = load_docs(compare_dir), position_profile(compare_dir)
    html = (HTML_TEMPLATE
            .replace('/*DOC_DATA*/null', json.dumps(doc_data, ensure_ascii=False, separators=(',', ':')))
            .replace('/*POS_DATA*/null', json.dumps(pos_data, ensure_ascii=False, separators=(',', ':'))))
    out = compare_dir / out_name
    out.write_text(html, encoding='utf-8')
    print(f'文章 {len(doc_data["diffs"])} 篇，句子 {pos_data["n_sents"]} 句，位置分 {N_BINS} 段；'
          f'报告已保存到 {out}（{out.stat().st_size / 1e6:.2f} MB）')


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>中性化语调变化分布</title>
<style>
  :root {
    color-scheme: light;
    --page: #f9f9f7; --surface: #fcfcfb; --ink-1: #0b0b0b; --ink-2: #52514e; --ink-muted: #898781;
    --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
    --series: #2a78d6; --band: rgba(42,120,214,0.20); --wash: rgba(42,120,214,0.06);
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page: #0d0d0d; --surface: #1a1a19; --ink-1: #f0efec; --ink-2: #c3c2b7; --ink-muted: #898781;
      --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
      --series: #3987e5; --band: rgba(57,135,229,0.28); --wash: rgba(57,135,229,0.08);
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page: #0d0d0d; --surface: #1a1a19; --ink-1: #f0efec; --ink-2: #c3c2b7; --ink-muted: #898781;
    --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --series: #3987e5; --band: rgba(57,135,229,0.28); --wash: rgba(57,135,229,0.08);
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--page); color: var(--ink-1);
         font: 15px/1.6 system-ui, -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; }
  main { max-width: 1040px; margin: 0 auto; padding: 32px 16px 64px; }
  h1 { font-size: 24px; margin: 0 0 6px; font-weight: 600; }
  h2 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
  p { margin: 0 0 8px; color: var(--ink-2); }
  .lead { margin-bottom: 24px; }
  section { background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
            padding: 20px 20px 16px; margin-bottom: 24px; }
  .controls { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 12px 0 4px; }
  .controls input { flex: 1 1 240px; min-width: 0; padding: 7px 10px; border: 1px solid var(--axis);
                    border-radius: 8px; background: var(--surface); color: var(--ink-1); font: inherit; }
  .controls button { padding: 7px 12px; border: 1px solid var(--axis); border-radius: 8px; cursor: pointer;
                     background: var(--surface); color: var(--ink-1); font: inherit; }
  .controls button:hover { background: var(--wash); }
  .controls button[aria-pressed="true"] { border-color: var(--series); background: var(--wash); font-weight: 600; }
  .key.line-s { border-top: 2px solid var(--series); }
  details { margin-top: 12px; font-size: 13px; color: var(--ink-2); }
  details summary { cursor: pointer; }
  details table th, details table td { width: auto; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 12px 0 4px; font-size: 13px; color: var(--ink-2); }
  .legend span { display: inline-flex; align-items: center; gap: 6px; }
  .key { width: 22px; height: 0; border-top: 2px solid var(--series); }
  .key.median { border-top: 2px solid var(--ink-1); }
  .key.mean { border-top: 2px dashed var(--ink-1); }
  .key.band { height: 12px; border: 0; background: var(--band); border-radius: 2px; }
  .chart { position: relative; width: 100%; }
  .chart svg { display: block; width: 100%; height: auto; overflow: visible; }
  .tip { position: absolute; pointer-events: none; background: var(--surface); border: 1px solid var(--border);
         border-radius: 8px; padding: 6px 10px; font-size: 13px; box-shadow: 0 2px 8px rgba(0,0,0,0.12);
         white-space: nowrap; display: none; }
  .tip b { font-weight: 600; color: var(--ink-1); }
  .tip div { color: var(--ink-2); }
  table { border-collapse: collapse; width: 100%; margin-top: 12px; font-size: 13px; }
  th, td { padding: 6px 8px; border-top: 1px solid var(--grid); text-align: left; }
  th { color: var(--ink-2); font-weight: 500; width: 40%; }
  td { font-variant-numeric: tabular-nums; }
  .tables { display: grid; grid-template-columns: 1fr 1fr; gap: 0 24px; }
  @media (max-width: 640px) { .tables { grid-template-columns: 1fr; } }
  .note { font-size: 12px; color: var(--ink-muted); margin-top: 10px; }
</style>
</head>
<body>
<main>
  <h1>中性化改写前后的语调变化分布</h1>
  <p class="lead">数据：2024 年 A 股年报 MD&amp;A，GLM 中性化改写，中文 FinBERT 逐句分类。
    语调变化 = 改写后 − 改写前，负值表示改写后语调下降。</p>

  <section id="sec-doc">
    <h2>1. 各篇文章平均语调的变化</h2>
    <p>每篇文章一个值：tone_diff = 改写后 tone − 改写前 tone，其中 tone = 正面句占比 − 负面句占比。</p>
    <div class="legend"></div>
    <div class="chart"></div>
    <div class="tables"></div>
  </section>

  <p class="note">第 1 部分：密度曲线为高斯核密度估计（带宽按 1.06·σ·n<sup>−1/5</sup>），并缩放为最高点 = 1，只表示相对密度；
    浅色区域为中间 95% 的文章（2.5% ~ 97.5% 分位数）；平均数的 95% 置信区间 = 平均数 ± 1.96·σ/√n。</p>

  <section id="sec-pos">
    <h2>2. 语调变化发生在文章的什么位置</h2>
    <p>每句的语调变化 = (改写后 正面概率 − 负面概率) − (改写前 正面概率 − 负面概率)。
       按句子顺序把每篇文章的相对位置 0 ~ 1 等分为 <span id="nbins"></span> 段，每篇文章先在各段内对句子取平均，
       再在每一段上汇总全部 <span id="ndocs"></span> 篇文章。</p>
    <div class="controls" role="group" aria-label="浅色带类型">
      <button type="button" data-band="pct" aria-pressed="true">浅色带：各篇文章的中间 95%</button>
      <button type="button" data-band="ci" aria-pressed="false">浅色带：平均数的 95% 置信区间</button>
    </div>
    <div class="legend"></div>
    <div class="chart"></div>
    <details><summary>查看各段数值表</summary><div class="ptable"></div></details>
    <p class="note">相对位置：第 i 句（从 0 计）的位置 = (i + 0.5) / 句数。句子数少于段数的文章只计入有句子的段，各段文章数见数值表。
      “各篇文章的中间 95%”= 该段上各篇文章取值的 2.5% ~ 97.5% 分位数，反映文章之间的差异；
      “平均数的 95% 置信区间”= 平均数 ± 1.96·σ/√n，反映平均数估计的精确程度，两种浅色带的纵轴范围不同。</p>
  </section>
</main>

<script>
const DOC = /*DOC_DATA*/null;
const POS = /*POS_DATA*/null;

// ---------- 统计 ----------
function stats(values) {
  const s = Float64Array.from(values).sort();
  const n = s.length;
  const q = p => { const h = (n - 1) * p, i = Math.floor(h); return s[i] + (h - i) * ((s[Math.min(i + 1, n - 1)]) - s[i]); };
  let sum = 0; for (const v of s) sum += v;
  const mean = sum / n;
  let ss = 0; for (const v of s) ss += (v - mean) ** 2;
  const sd = n > 1 ? Math.sqrt(ss / (n - 1)) : 0;
  let neg = 0, zero = 0, eps = 1e-9; for (const v of s) { if (v < -eps) neg++; else if (v <= eps) zero++; }
  return { s, n, mean, sd, median: q(0.5), lo: q(0.025), hi: q(0.975), min: s[0], max: s[n - 1],
           neg, zero, pos: n - neg - zero, ci: 1.96 * sd / Math.sqrt(n) };
}
function cdf(s, x) {  // ≤ x 的比例
  let lo = 0, hi = s.length; while (lo < hi) { const m = (lo + hi) >> 1; if (s[m] <= x) lo = m + 1; else hi = m; } return lo / s.length;
}
// 分箱高斯核密度估计：任意样本量都很快；返回缩放到最高点=1的密度
function kde(st, x0, x1, m = 600) {
  const step = (x1 - x0) / (m - 1);
  let bw = 1.06 * st.sd * Math.pow(st.n, -0.2);
  if (!(bw > step)) bw = step * 2;
  const counts = new Float64Array(m);
  for (const v of st.s) {
    const t = (v - x0) / step; if (t < 0 || t > m - 1) continue;
    const i = Math.floor(t), f = t - i; counts[i] += 1 - f; if (i + 1 < m) counts[i + 1] += f;
  }
  const K = Math.ceil(4 * bw / step), w = []; for (let k = -K; k <= K; k++) w.push(Math.exp(-0.5 * (k * step / bw) ** 2));
  const y = new Float64Array(m); let ymax = 0;
  for (let i = 0; i < m; i++) { if (!counts[i]) continue;
    for (let k = -K; k <= K; k++) { const j = i + k; if (j >= 0 && j < m) y[j] += counts[i] * w[k + K]; } }
  for (const v of y) if (v > ymax) ymax = v;
  const xs = Array.from({ length: m }, (_, i) => x0 + i * step);
  return { xs, ys: Array.from(y, v => ymax ? v / ymax : 0), bw };
}
function niceTicks(a, b, target = 7) {
  const raw = (b - a) / target, mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map(k => k * mag).find(k => k >= raw);
  const out = []; for (let v = Math.ceil(a / step) * step; v <= b + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
const fmt = (v, d = 3) => (v > 0 ? '+' : v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
const pct = v => (v * 100).toFixed(1) + '%';

// ---------- 绘图 ----------
const NS = 'http://www.w3.org/2000/svg';
function el(tag, attrs, parent) {
  const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e); return e;
}
function legend(box) {
  box.textContent = '';
  for (const [cls, label] of [['', '分布密度（最高点 = 1）'], ['median', '中位数（实线）'], ['mean', '平均数（虚线）'], ['band', '中间 95% 区间']]) {
    const s = document.createElement('span'), k = document.createElement('i');
    k.className = 'key ' + cls; s.appendChild(k); s.appendChild(document.createTextNode(label)); box.appendChild(s);
  }
}
function table(box, st, unitLabel) {
  box.textContent = '';
  const rows1 = [[unitLabel, st.n.toLocaleString()], ['平均数', fmt(st.mean)], ['平均数的 95% 置信区间', `${fmt(st.mean - st.ci)} ~ ${fmt(st.mean + st.ci)}`],
                 ['中位数', fmt(st.median)], ['标准差', st.sd.toFixed(3)]];
  const rows2 = [['中间 95% 区间（2.5% ~ 97.5% 分位）', `${fmt(st.lo)} ~ ${fmt(st.hi)}`], ['最小值 ~ 最大值', `${fmt(st.min)} ~ ${fmt(st.max)}`],
                 ['下降（< 0）', pct(st.neg / st.n)], [`不变（${unitLabel === '句子数' ? '保留3位小数后' : ''}= 0）`, pct(st.zero / st.n)],
                 ['上升（> 0）', pct(st.pos / st.n)]];
  for (const rows of [rows1, rows2]) {
    const t = document.createElement('table');
    for (const [k, v] of rows) { const tr = t.insertRow(), th = document.createElement('th'); th.textContent = k; tr.appendChild(th); tr.insertCell().textContent = v; }
    box.appendChild(t);
  }
}
function draw(box, values, opt) {
  box.textContent = '';
  const st = stats(values);
  let [x0, x1] = opt.range || [st.min, st.max];
  if (!opt.range) { const pad = Math.max((x1 - x0) * 0.08, 1e-3); x0 -= pad; x1 += pad; }
  const dens = kde(st, x0, x1);
  const W = Math.max(box.clientWidth, 320), H = 340, m = { l: 44, r: 16, t: 34, b: 44 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const X = v => m.l + (v - x0) / (x1 - x0) * pw, Y = v => m.t + (1 - v) * ph;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': opt.aria }, box);
  const css = v => `var(${v})`;

  // 网格与坐标轴
  for (const v of [0, 0.25, 0.5, 0.75, 1]) {
    el('line', { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), stroke: css('--grid'), 'stroke-width': 1 }, svg);
    el('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 12, fill: css('--ink-muted') }, svg).textContent = v.toFixed(2).replace(/0$/, '');
  }
  el('line', { x1: m.l, x2: m.l + pw, y1: Y(0), y2: Y(0), stroke: css('--axis'), 'stroke-width': 1 }, svg);
  for (const v of niceTicks(x0, x1)) {
    el('line', { x1: X(v), x2: X(v), y1: Y(0), y2: Y(0) + 4, stroke: css('--axis') }, svg);
    el('text', { x: X(v), y: Y(0) + 18, 'text-anchor': 'middle', 'font-size': 12, fill: css('--ink-muted') }, svg).textContent = fmt(v, opt.tickDigits ?? 2).replace('+', '');
  }
  el('text', { x: m.l + pw / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: css('--ink-2') }, svg).textContent = opt.xLabel;

  // 中间95%区间：整段区间铺一层很淡的底色（密度接近0处也能看出范围），曲线下方再加深
  el('rect', { x: X(st.lo), y: m.t, width: Math.max(X(st.hi) - X(st.lo), 1), height: ph, fill: css('--wash') }, svg);
  const inBand = dens.xs.map((x, i) => [x, dens.ys[i]]).filter(([x]) => x >= st.lo && x <= st.hi);
  if (inBand.length) {
    const yAt = x => { const t = (x - x0) / (x1 - x0) * (dens.xs.length - 1), i = Math.min(Math.floor(t), dens.xs.length - 2); return dens.ys[i] + (t - i) * (dens.ys[i + 1] - dens.ys[i]); };
    const pts = [[st.lo, yAt(st.lo)], ...inBand, [st.hi, yAt(st.hi)]];
    el('path', { d: `M${X(st.lo)},${Y(0)} ` + pts.map(([x, y]) => `L${X(x)},${Y(y)}`).join(' ') + ` L${X(st.hi)},${Y(0)} Z`, fill: css('--band') }, svg);
  }
  // 密度曲线
  el('path', { d: dens.xs.map((x, i) => `${i ? 'L' : 'M'}${X(x).toFixed(1)},${Y(dens.ys[i]).toFixed(1)}`).join(' '),
               fill: 'none', stroke: css('--series'), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
  // 中位数（实线）与平均数（虚线），标签放在两条线的外侧避免重叠
  const lines = [{ v: st.median, dash: null, name: '中位数' }, { v: st.mean, dash: '6 4', name: '平均数' }].sort((a, b) => a.v - b.v);
  lines.forEach((ln, i) => {
    el('line', { x1: X(ln.v), x2: X(ln.v), y1: Y(0), y2: Y(1) - 6, stroke: css('--ink-1'), 'stroke-width': 1.5, ...(ln.dash ? { 'stroke-dasharray': ln.dash } : {}) }, svg);
    const left = i === 0;
    el('text', { x: X(ln.v) + (left ? -6 : 6), y: m.t - 14, 'text-anchor': left ? 'end' : 'start', 'font-size': 12, fill: css('--ink-1') }, svg).textContent = `${ln.name} ${fmt(ln.v)}`;
  });

  // 悬停：十字线 + 提示
  const cross = el('line', { y1: m.t, y2: Y(0), stroke: css('--ink-muted'), 'stroke-width': 1, visibility: 'hidden' }, svg);
  const dot = el('circle', { r: 4, fill: css('--series'), stroke: css('--surface'), 'stroke-width': 2, visibility: 'hidden' }, svg);
  const hit = el('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent' }, svg);
  const tip = document.createElement('div'); tip.className = 'tip'; box.appendChild(tip);
  const move = ev => {
    const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width;
    const x = x0 + (Math.min(Math.max(sx, m.l), m.l + pw) - m.l) / pw * (x1 - x0);
    const i = Math.round((x - x0) / (x1 - x0) * (dens.xs.length - 1)), xv = dens.xs[i], yv = dens.ys[i];
    cross.setAttribute('x1', X(xv)); cross.setAttribute('x2', X(xv)); cross.setAttribute('visibility', 'visible');
    dot.setAttribute('cx', X(xv)); dot.setAttribute('cy', Y(yv)); dot.setAttribute('visibility', 'visible');
    tip.textContent = '';
    const b = document.createElement('b'); b.textContent = `语调变化 ${fmt(xv)}`; tip.appendChild(b);
    for (const t of [`相对密度 ${yv.toFixed(2)}`, `≤ 该值的${opt.unit} ${pct(cdf(st.s, xv))}`]) { const d = document.createElement('div'); d.textContent = t; tip.appendChild(d); }
    tip.style.display = 'block';
    const px = X(xv) / W * r.width, tw = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(px + 12, 0), r.width - tw) + 'px'; tip.style.top = (Y(yv) / H * r.height - 56) + 'px';
  };
  hit.addEventListener('pointermove', move);
  hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dot.setAttribute('visibility', 'hidden'); });
  return st;
}

// ---------- 第1部分：文章间 ----------
const secDoc = document.getElementById('sec-doc');
function renderDoc() {
  legend(secDoc.querySelector('.legend'));
  const st = draw(secDoc.querySelector('.chart'), DOC.diffs,
    { xLabel: '文章平均语调变化 tone_diff（改写后 − 改写前）', unit: '文章比例', aria: '各篇文章平均语调变化的分布密度', tickDigits: 2 });
  table(secDoc.querySelector('.tables'), st, '文章数');
}

// ---------- 第2部分：语调变化的位置分布 ----------
const secPos = document.getElementById('sec-pos');
const NB = POS.median.length;
document.getElementById('nbins').textContent = NB;
document.getElementById('ndocs').textContent = POS.n_docs.toLocaleString();
const BANDS = { pct: { lo: 'p_lo', hi: 'p_hi', name: '各篇文章的中间 95%' }, ci: { lo: 'ci_lo', hi: 'ci_hi', name: '平均数的 95% 置信区间' } };
let band = location.hash === '#ci' ? 'ci' : 'pct';  // 网址加 #ci 可直接打开置信区间图
secPos.querySelectorAll('[data-band]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.band === band)));
const posLabel = j => `${Math.round(POS.x0[j] * 100)}% ~ ${Math.round(POS.x1[j] * 100)}%`;

function renderPos() {
  const B = BANDS[band];
  // 图例
  const lg = secPos.querySelector('.legend'); lg.textContent = '';
  for (const [cls, label] of [['line-s', '中位数（实线）'], ['mean', '平均数（虚线）'], ['band', B.name]]) {
    const s = document.createElement('span'), k = document.createElement('i');
    k.className = 'key ' + cls; s.appendChild(k); s.appendChild(document.createTextNode(label)); lg.appendChild(s);
  }
  // 图
  const box = secPos.querySelector('.chart'); box.textContent = '';
  const xs = POS.x0.map((a, j) => (a + POS.x1[j]) / 2);
  const all = [...POS[B.lo], ...POS[B.hi], ...POS.median, ...POS.mean, 0];
  let y0 = Math.min(...all), y1 = Math.max(...all); const pad = (y1 - y0) * 0.08; y0 -= pad; y1 += pad;
  const W = Math.max(box.clientWidth, 320), H = 360, m = { l: 56, r: 16, t: 16, b: 46 };
  const pw = W - m.l - m.r, ph = H - m.t - m.b;
  const X = v => m.l + v * pw, Y = v => m.t + (y1 - v) / (y1 - y0) * ph;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': '文章内各位置的语调变化：中位数、平均数与95%区间' }, box);
  const css = v => `var(${v})`;
  for (const v of niceTicks(y0, y1, 6)) {
    el('line', { x1: m.l, x2: m.l + pw, y1: Y(v), y2: Y(v), stroke: css(v === 0 ? '--axis' : '--grid'), 'stroke-width': 1 }, svg);
    el('text', { x: m.l - 8, y: Y(v) + 4, 'text-anchor': 'end', 'font-size': 12, fill: css('--ink-muted') }, svg).textContent = fmt(v, 2).replace('+', '');
  }
  for (let v = 0; v <= 1.0001; v += 0.1) {
    el('line', { x1: X(v), x2: X(v), y1: m.t + ph, y2: m.t + ph + 4, stroke: css('--axis') }, svg);
    el('text', { x: X(v), y: m.t + ph + 18, 'text-anchor': 'middle', 'font-size': 12, fill: css('--ink-muted') }, svg).textContent = Math.round(v * 100) + '%';
  }
  el('text', { x: m.l + pw / 2, y: H - 6, 'text-anchor': 'middle', 'font-size': 12, fill: css('--ink-2') }, svg).textContent = '文章内相对位置（0% = 开头，100% = 结尾）';
  el('text', { x: 14, y: m.t + ph / 2, 'text-anchor': 'middle', 'font-size': 12, fill: css('--ink-2'), transform: `rotate(-90 14 ${m.t + ph / 2})` }, svg).textContent = '该段平均语调变化';
  // 浅色带
  const up = xs.map((x, j) => `${X(x)},${Y(POS[B.hi][j])}`), dn = xs.map((x, j) => `${X(x)},${Y(POS[B.lo][j])}`).reverse();
  el('path', { d: 'M' + up.join(' L') + ' L' + dn.join(' L') + ' Z', fill: css('--band') }, svg);
  // 中位数（实线）与平均数（虚线）
  const path = key => xs.map((x, j) => `${j ? 'L' : 'M'}${X(x).toFixed(1)},${Y(POS[key][j]).toFixed(1)}`).join(' ');
  el('path', { d: path('median'), fill: 'none', stroke: css('--series'), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, svg);
  el('path', { d: path('mean'), fill: 'none', stroke: css('--ink-1'), 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, svg);
  // 悬停：十字线吸附到最近的段
  const cross = el('line', { y1: m.t, y2: m.t + ph, stroke: css('--ink-muted'), 'stroke-width': 1, visibility: 'hidden' }, svg);
  const dots = ['median', 'mean'].map(k => el('circle', { r: 4, fill: k === 'median' ? css('--series') : css('--ink-1'), stroke: css('--surface'), 'stroke-width': 2, visibility: 'hidden' }, svg));
  const hit = el('rect', { x: m.l, y: m.t, width: pw, height: ph, fill: 'transparent' }, svg);
  const tip = document.createElement('div'); tip.className = 'tip'; box.appendChild(tip);
  hit.addEventListener('pointermove', ev => {
    const r = svg.getBoundingClientRect(), sx = (ev.clientX - r.left) * W / r.width;
    const j = Math.min(NB - 1, Math.max(0, Math.floor((sx - m.l) / pw * NB)));
    cross.setAttribute('x1', X(xs[j])); cross.setAttribute('x2', X(xs[j])); cross.setAttribute('visibility', 'visible');
    ['median', 'mean'].forEach((k, i) => { dots[i].setAttribute('cx', X(xs[j])); dots[i].setAttribute('cy', Y(POS[k][j])); dots[i].setAttribute('visibility', 'visible'); });
    tip.textContent = '';
    const b = document.createElement('b'); b.textContent = `中位数 ${fmt(POS.median[j])}　平均数 ${fmt(POS.mean[j])}`; tip.appendChild(b);
    for (const t of [`${B.name} ${fmt(POS[B.lo][j])} ~ ${fmt(POS[B.hi][j])}`, `位置 ${posLabel(j)}，${POS.n[j].toLocaleString()} 篇文章`]) {
      const d = document.createElement('div'); d.textContent = t; tip.appendChild(d); }
    tip.style.display = 'block';
    const px = X(xs[j]) / W * r.width, tw = tip.offsetWidth;
    tip.style.left = Math.min(Math.max(px + 12, 0), r.width - tw) + 'px';
    tip.style.top = (Math.min(Y(POS.median[j]), Y(POS.mean[j])) / H * r.height - 70) + 'px';
  });
  hit.addEventListener('pointerleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
  // 数值表
  const tb = secPos.querySelector('.ptable'); tb.textContent = '';
  const t = document.createElement('table'), hr = t.insertRow();
  for (const h of ['位置', '文章数', '中位数', '平均数', '中间 95%（2.5% ~ 97.5%）', '平均数 95% 置信区间']) { const th = document.createElement('th'); th.textContent = h; hr.appendChild(th); }
  for (let j = 0; j < NB; j++) {
    const tr = t.insertRow();
    for (const v of [posLabel(j), POS.n[j].toLocaleString(), fmt(POS.median[j]), fmt(POS.mean[j]),
                     `${fmt(POS.p_lo[j])} ~ ${fmt(POS.p_hi[j])}`, `${fmt(POS.ci_lo[j])} ~ ${fmt(POS.ci_hi[j])}`]) tr.insertCell().textContent = v;
  }
  tb.appendChild(t);
}
secPos.querySelectorAll('[data-band]').forEach(btn => btn.addEventListener('click', () => {
  band = btn.dataset.band;
  secPos.querySelectorAll('[data-band]').forEach(b => b.setAttribute('aria-pressed', String(b === btn)));
  renderPos();
}));

renderDoc(); renderPos();
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { renderDoc(); renderPos(); }, 150); });
</script>
</body>
</html>
'''
