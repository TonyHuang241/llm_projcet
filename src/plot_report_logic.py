"""把 report_logic.py 输出的 report_graphs.json 画成单文件 HTML：每篇研报一张五层逻辑图。

左侧按公司列出研报，点选后在中间画图：公司估值在最上方，往下依次是会计层、经营指标、公司经营情况、外部环境
（左侧标出层名；同一层内有原文关系的，被影响的一方排在上面），箭头由影响因素指向被影响因素：
  边的颜色表示方向：红色为正向、绿色为负向（A 股习惯）；实线是段内原文抽取的关系，点划线是跨段原文
  （悬挂补全或概念补全时找到的，原文引用已核对），虚线是逻辑补全（大模型按定义选择，没有原文），
  灰色细点线是第 3 层的上位指标，灰色细实线是预置的会计关系；
  节点显示全局名称（各研报统一后的叫法），底色表示沿路径相乘后对“公司估值”的净影响：偏红为正、偏绿为负、
  黄色为既有正向路径又有负向路径，灰色为没有连到估值；会计层节点边框加粗，虚线边框的节点没有接入估值。
点击节点会突出它的上游因素和下游路径，右侧显示层级、定义、别名、合并进来的下位指标、原文线索、悬挂记录，以及因另有更长链路而剪掉的直连；
点击边显示原文证据（逻辑补全边显示理由）和段落编号。
分层坐标自己计算（vis 自带的分层布局遇到多个子图时节点间距过大），多个子图按大小排成若干行或列；
布局方向默认自动选择能把图放得更大的一种（估值在上或在右），也可以手动切换。另有“全局名称表”页，用来检查跨研报去重的结果。
需要联网加载 vis-network（jsdelivr CDN）。

用法：report_logic.main 结束时自动调用；只想重画时在 main.py 中设置 TASK = 'report_logic_html'
"""
import json
from pathlib import Path

VIS_URL = 'https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js'

TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>研报逻辑图</title>
<script src="__VIS_URL__"></script>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1f2328; --muted: #656d76; --line: #d8dee4; --accent: #2f6feb;
  --pos: #cf222e; --neg: #1a7f37; --mix: #bf8700; --pos-bg: #ffebe9; --neg-bg: #dafbe1; --mix-bg: #fff8c5;
  --node-bg: #eef1f4; --root-bg: #24292f; --root-ink: #ffffff;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0d1117; --panel: #161b22; --ink: #e6edf3; --muted: #8d96a0; --line: #30363d; --accent: #4493f8;
    --pos: #ff7b72; --neg: #3fb950; --mix: #d29922; --pos-bg: #4a1d1d; --neg-bg: #12361f; --mix-bg: #3d2f05;
    --node-bg: #21262d; --root-bg: #e6edf3; --root-ink: #0d1117;
  }
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.5 -apple-system, "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg); color: var(--ink); height: 100vh; display: flex; flex-direction: column; }
header { display: flex; align-items: center; gap: 16px; padding: 10px 16px; border-bottom: 1px solid var(--line);
  background: var(--panel); flex-wrap: wrap; }
header h1 { font-size: 16px; margin: 0; }
.tabs button { border: 1px solid var(--line); background: var(--bg); color: var(--ink); padding: 4px 12px;
  border-radius: 6px; cursor: pointer; font: inherit; }
.tabs button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin-left: auto; }
.legend span { display: inline-flex; align-items: center; gap: 5px; }
.legend i { display: inline-block; width: 22px; height: 0; border-top: 2px solid; }
.legend b { display: inline-block; width: 12px; height: 12px; border-radius: 3px; border: 1px solid var(--line); }
main { flex: 1; display: flex; min-height: 0; }
#side { width: 270px; border-right: 1px solid var(--line); background: var(--panel); overflow: auto; padding: 8px; }
#side input { width: 100%; padding: 6px 8px; border: 1px solid var(--line); border-radius: 6px; background: var(--bg);
  color: var(--ink); font: inherit; margin-bottom: 6px; }
.company { font-weight: 600; margin: 10px 4px 4px; color: var(--muted); font-size: 12px; }
.rep { padding: 6px 8px; border-radius: 6px; cursor: pointer; font-size: 13px; }
.rep:hover { background: var(--bg); }
.rep.on { background: var(--accent); color: #fff; }
.rep small { display: block; opacity: .75; font-size: 11px; }
#center { flex: 1; display: flex; flex-direction: column; min-width: 0; }
#meta { padding: 8px 14px; border-bottom: 1px solid var(--line); background: var(--panel); font-size: 13px; }
#meta .t { font-weight: 600; font-size: 14px; }
#meta .s { color: var(--muted); }
#tools { display: flex; gap: 14px; align-items: center; padding: 6px 14px; font-size: 12px; color: var(--muted);
  border-bottom: 1px solid var(--line); background: var(--panel); flex-wrap: wrap; }
#tools label { display: inline-flex; gap: 4px; align-items: center; cursor: pointer; }
#tools select { font: inherit; background: var(--bg); color: var(--ink); border: 1px solid var(--line); border-radius: 4px; }
#net { flex: 1; min-height: 0; }
#detail { width: 340px; border-left: 1px solid var(--line); background: var(--panel); overflow: auto; padding: 12px 14px;
  font-size: 13px; }
#detail h3 { margin: 0 0 6px; font-size: 15px; }
#detail .tag { display: inline-block; padding: 0 6px; border-radius: 4px; font-size: 12px; margin-right: 4px; }
.tag.pos { background: var(--pos-bg); color: var(--pos); } .tag.neg { background: var(--neg-bg); color: var(--neg); }
.tag.mix { background: var(--mix-bg); color: var(--mix); } .tag.grey { background: var(--node-bg); color: var(--muted); }
#detail ul { padding-left: 18px; margin: 6px 0; } #detail li { margin: 3px 0; }
#detail .ev { color: var(--muted); font-size: 12px; }
#detail .sec { margin-top: 12px; font-weight: 600; color: var(--muted); font-size: 12px; }
#table { display: none; flex: 1; overflow: auto; padding: 14px; }
#table table { border-collapse: collapse; width: 100%; background: var(--panel); font-size: 13px; }
#table th, #table td { border: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
#table th { position: sticky; top: 0; background: var(--panel); }
@media (max-width: 900px) { main { flex-direction: column; } #side, #detail { width: auto; max-height: 30vh; } #net { min-height: 60vh; } }
</style>
</head>
<body>
<header>
  <h1>研报逻辑图</h1>
  <div class="tabs"><button id="tab-graph" class="on">逻辑图</button> <button id="tab-table">全局名称表</button></div>
  <div class="legend">
    <span><i style="border-color:var(--pos)"></i>正向</span>
    <span><i style="border-color:var(--neg)"></i>负向</span>
    <span data-origin="bridged"><i style="border-top-style:dashed;border-color:var(--ink)"></i>跨段原文（引用已核对）</span>
    <span data-origin="logic"><i style="border-top-style:dashed;border-color:var(--muted);opacity:.7"></i>逻辑补全</span>
    <span data-origin="part_of"><i style="border-top-style:dotted;border-color:var(--muted)"></i>上位指标</span>
    <span data-origin="preset"><i style="border-top-width:1px;border-color:var(--muted)"></i>会计关系</span>
    <span>⚠ 与更长链路方向相反</span>
    <span><b style="background:var(--pos-bg)"></b>对估值净正向</span>
    <span><b style="background:var(--neg-bg)"></b>净负向</span>
    <span><b style="background:var(--mix-bg)"></b>正负皆有</span>
    <span><b style="background:var(--node-bg)"></b>未连到估值</span>
    <span><b style="background:var(--node-bg);border-style:dashed;border-color:var(--muted)"></b>未接入</span>
  </div>
</header>
<main>
  <nav id="side"><input id="q" placeholder="筛选研报或节点名"><div id="list"></div></nav>
  <section id="center">
    <div id="meta"></div>
    <div id="tools">
      <label><input type="checkbox" id="show-evid">边上显示支持次数</label>
      <label>布局 <select id="dir"><option value="auto">自动</option><option value="UD">估值在上</option><option value="RL">估值在右</option></select></label>
      <label><input type="checkbox" id="show-logic" checked>显示逻辑补全边</label>
      <span id="hint">点击节点突出其上游因素和下游路径，点击边查看原文证据，点空白处恢复；滚轮缩放，拖动平移</span>
    </div>
    <div id="net"></div>
  </section>
  <section id="table"></section>
  <aside id="detail"><div class="ev">点击图中的节点或边查看详情。</div></aside>
</main>
<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const ROOT = DATA.root;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
let current = null, network = null;

// 某类边没有出现时，隐藏对应图例
const ORIGINS = new Set(DATA.reports.flatMap(r => r.edges.map(e => e.origin)));
document.querySelectorAll('[data-origin]').forEach(el => { if (!ORIGINS.has(el.dataset.origin)) el.style.display = 'none'; });
const ORIGIN_SHORT = {bridged: '跨段原文', part_of: '上位指标', logic: '逻辑补全', preset: '会计关系'};
const ORIGIN_LONG = {extracted: '段内原文抽取', bridged: '跨段原文（悬挂补全或概念补全找到，原文引用已核对）',
  part_of: '第 3 层上位指标（加总或相乘）', logic: '逻辑补全（大模型按定义和语境选择，没有原文）', preset: '预置的会计关系'};
const LAYERS = DATA.layers || {};
const labelOf = (r, id) => (r.nodes.find(n => n.id === id) || {}).label || id;

const reports = DATA.reports.slice().sort((a, b) => (a.company || '').localeCompare(b.company || '', 'zh') || (b.date || '').localeCompare(a.date || ''));

function renderList() {
  const q = document.getElementById('q').value.trim().toLowerCase();
  const box = document.getElementById('list');
  box.innerHTML = '';
  let company = null;
  for (const r of reports) {
    const hay = [r.company, r.broker, r.title, r.date, ...r.nodes.flatMap(n => [n.id, n.label])].join(' ').toLowerCase();
    if (q && !hay.includes(q)) continue;
    if (r.company !== company) {
      company = r.company;
      box.insertAdjacentHTML('beforeend', `<div class="company">${esc(company || '未知公司')}</div>`);
    }
    const el = document.createElement('div');
    el.className = 'rep' + (current === r ? ' on' : '');
    el.innerHTML = `${esc(r.broker)} ${esc(r.date)}<small>${esc(r.title)}</small>`;
    el.onclick = () => { current = r; renderList(); draw(); };
    box.appendChild(el);
  }
}

function nodeColor(n) {
  if (n.is_root) return {background: css('--root-bg'), border: css('--root-bg'), font: css('--root-ink')};
  const m = {'+': ['--pos-bg', '--pos'], '-': ['--neg-bg', '--neg'], '±': ['--mix-bg', '--mix']}[n.effect] || ['--node-bg', '--line'];
  return {background: css(m[0]), border: css(m[1]), font: css('--ink')};
}

function draw() {
  const r = current;
  if (!r) return;
  const s = r.stats;
  const byLayer = [3, 4, 5].map(k => `${s.n_by_layer[k] || 0}`).join('/');
  document.getElementById('meta').innerHTML =
    `<div class="t">${esc(r.company)}｜${esc(r.broker)} ${esc(r.date)}｜${esc(r.title)}</div>` +
    `<div class="s">${r.n_paragraphs} 段 · 概念 ${s.n_concepts} 个（第 3/4/5 层 ${byLayer}）· ${s.n_relations} 条原文关系` +
    (s.n_reverse ? `（反向 ${s.n_reverse} 条未进图）` : '') +
    ` · 原文边 ${s.n_extracted} 条` + (s.n_bridged ? ` · 跨段原文 ${s.n_bridged} 条` : '') +
    (s.n_logic ? ` · 逻辑补全 ${s.n_logic} 条${s.n_logic_skip ? `（跨两层以上 ${s.n_logic_skip} 条）` : ''}` : '') +
    (s.n_part_of ? ` · 上位指标 ${s.n_part_of} 条` : '') +
    ` · 连到估值的概念 ${Math.round(s.reach_before_completion * 100)}% → ${Math.round(s.reach * 100)}%（补全前→后）` +
    (s.n_unconnected ? ` · 未接入 ${s.n_unconnected} 个` : '') + (s.n_merged ? ` · 下位指标向上合并 ${s.n_merged} 个` : '') +
    (s.n_isolated ? ` · 删除孤立概念 ${s.n_isolated} 个` : '') +
    (s.n_replaced ? ` · 剪掉有更长链路的直连 ${s.n_replaced} 条` : '') +
    (s.n_conflict ? ` · 与更长链路方向相反而保留的直连 ${s.n_conflict} 条（⚠）` : '') + (s.n_pruned ? ` · 删除会计科目 ${s.n_pruned} 个` : '') +
    (s.removed_cycle_edges ? ` · 去环删边 ${s.removed_cycle_edges} 条` : '') + '</div>';
  const showW = document.getElementById('show-evid').checked, showLogic = document.getElementById('show-logic').checked;
  const edges = r.edges.filter(e => showLogic || e.origin !== 'logic');
  const nodes = r.nodes.map(n => {
    const c = nodeColor(n);
    return {id: n.id, label: n.label || n.id, level: n.level, shape: 'box', margin: 8,
      color: {background: c.background, border: c.border, highlight: {background: c.background, border: css('--accent')}},
      font: {color: c.font, size: n.is_root ? 18 : 14, face: 'PingFang SC, Microsoft YaHei, sans-serif', bold: n.is_root || n.layer === 2},
      borderWidth: n.is_root || n.layer === 2 ? 2 : 1, widthConstraint: {maximum: 130},
      borderDashes: !n.is_root && n.connected === false ? [4, 3] : false};
  });
  const DASHES = {bridged: [12, 3, 3, 3], part_of: [2, 4], logic: [8, 6]};
  const grey = e => e.origin === 'part_of' || e.origin === 'preset';
  const visEdges = edges.map((e, i) => ({
    id: i, from: e.source, to: e.target, arrows: {to: {enabled: true, scaleFactor: 0.6}},
    color: {color: css(grey(e) ? '--muted' : e.direction === '正向' ? '--pos' : '--neg'), highlight: css('--accent'),
      opacity: e.origin === 'extracted' || e.origin === 'bridged' ? 1 : e.origin === 'logic' ? 0.6 : 0.7},
    dashes: DASHES[e.origin] || false,
    width: e.origin === 'extracted' ? 1.2 + Math.min(e.weight, 5) * 0.5 : e.origin === 'preset' ? 0.8 : 1.2,
    label: e.origin === 'part_of' ? '⊂' : (e.direction === '正向' ? '+' : '−') + (showW && e.origin === 'extracted' ? ' ×' + e.weight : '') +
      ((e.conflict_path || []).length ? ' ⚠' : ''),
    font: {color: css('--muted'), size: 11, strokeWidth: 3, strokeColor: css('--panel')},
  }));
  // 布局：“自动”时分别算终点在上、终点在右两种坐标，取能把图放得更大的一种
  const box = document.getElementById('net');
  const W = box.clientWidth || 800, H = box.clientHeight || 600;
  let dir = document.getElementById('dir').value, lay;
  if (dir === 'auto') {
    const ud = layout(nodes, visEdges, true, W, H), rl = layout(nodes, visEdges, false, W, H);
    [dir, lay] = ud.scale >= rl.scale ? ['UD', ud] : ['RL', rl];
  } else {
    lay = layout(nodes, visEdges, dir === 'UD', W, H);
  }
  nodes.forEach(n => Object.assign(n, lay.pos[n.id]));
  const vertical = dir === 'UD';
  // 层名标签：放在每层第一行的左侧（估值在右时放在上方）
  const firstRow = {};
  r.nodes.forEach(n => { firstRow[n.layer] = Math.min(firstRow[n.layer] ?? Infinity, n.level); });
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const labels = Object.entries(firstRow).map(([k, lv]) => ({
    id: '__layer' + k, label: `第${k}层\n${LAYERS[k] || ''}`, shape: 'text', chosen: false, fixed: true,
    font: {color: css('--muted'), size: 13, face: 'PingFang SC, Microsoft YaHei, sans-serif'},
    ...(vertical ? {x: Math.min(...xs) - 150, y: lv * lay.gap} : {x: -lv * lay.gap, y: Math.min(...ys) - 70})}));
  visEdges.forEach(e => { e.smooth = {type: 'cubicBezier', forceDirection: vertical ? 'vertical' : 'horizontal', roundness: 0.4}; });
  const options = {
    layout: {hierarchical: false, improvedLayout: false},
    physics: false, interaction: {hover: true, tooltipDelay: 150, navigationButtons: false}, edges: {selectionWidth: 1.5},
  };
  if (network) network.destroy();
  const nodeSet = new vis.DataSet([...nodes, ...labels]), edgeSet = new vis.DataSet(visEdges);
  network = new vis.Network(document.getElementById('net'), {nodes: nodeSet, edges: edgeSet}, options);
  network.once('afterDrawing', () => network.fit({animation: false}));
  network.on('click', p => {
    if (p.nodes.length && String(p.nodes[0]).startsWith('__')) focus(null);
    else if (p.nodes.length) { focus(p.nodes[0]); showNode(r, p.nodes[0]); }
    else if (p.edges.length) { focus(null); showEdge(edges[p.edges[0]]); }
    else focus(null);
  });

  // 点击节点：只突出它的上游因素和下游路径，其余淡化；点空白处恢复
  function focus(id) {
    let keep = null;
    if (id && id !== ROOT) {
      keep = new Set([id]);
      for (const [key, next] of [['to', 'from'], ['from', 'to']]) {
        const stack = [id];
        while (stack.length) {
          const cur = stack.pop();
          visEdges.forEach(e => { if (e[key] === cur && !keep.has(e[next])) { keep.add(e[next]); stack.push(e[next]); } });
        }
      }
    }
    nodeSet.update(nodes.map(n => ({id: n.id, opacity: !keep || keep.has(n.id) ? 1 : 0.15})));
    edgeSet.update(visEdges.map(e => ({id: e.id, color: {...e.color,
      opacity: !keep || (keep.has(e.from) && keep.has(e.to)) ? e.color.opacity : 0.06}})));
  }
  showNode(r, ROOT);
}

// 分层坐标：行号沿用数据里的 level（公司估值为 0），层内按相邻节点的平均位置排几轮以减少交叉；
// 各子图按大小依次排成若干行（估值在上时）或列（估值在右时），行长按画布长宽比取，使整张图接近画布形状。
// 返回 {pos: {节点: {x, y}}, scale: 整张图放进画布时的缩放比例, gap: 行距}
function layout(nodes, edges, vertical, W, H) {
  const GAP_LEVEL = vertical ? 110 : 220, GAP_NODE = vertical ? 170 : 56, GAP_COMP = vertical ? 70 : 50;  // 节点框最宽约 154
  const nbr = new Map(nodes.map(n => [n.id, []]));
  edges.forEach(e => { nbr.get(e.from).push(e.to); nbr.get(e.to).push(e.from); });
  const level = new Map(nodes.map(n => [n.id, n.level]));
  const seen = new Set(), comps = [];
  for (const n of nodes) {
    if (seen.has(n.id)) continue;
    const ids = [], stack = [n.id];
    seen.add(n.id);
    while (stack.length) {
      const u = stack.pop();
      ids.push(u);
      for (const v of nbr.get(u)) if (!seen.has(v)) { seen.add(v); stack.push(v); }
    }
    const layers = new Map();
    ids.forEach(id => { const k = level.get(id); if (!layers.has(k)) layers.set(k, []); layers.get(k).push(id); });
    const ks = [...layers.keys()].sort((a, b) => a - b);
    const order = new Map();
    ks.forEach(k => layers.get(k).forEach((id, i) => order.set(id, i)));
    const bary = id => { const ns = nbr.get(id); return ns.length ? ns.reduce((s, v) => s + order.get(v), 0) / ns.length : order.get(id); };
    for (let sweep = 0; sweep < 6; sweep++) {
      for (const k of sweep % 2 ? [...ks].reverse() : ks) {
        const b = new Map(layers.get(k).map(id => [id, bary(id)]));
        layers.get(k).sort((p, q) => b.get(p) - b.get(q)).forEach((id, i) => order.set(id, i));
      }
    }
    const width = Math.max(...ks.map(k => layers.get(k).length));
    comps.push({layers, ks, width, along: width * GAP_NODE, across: (ks[ks.length - 1] + 1) * GAP_LEVEL});
  }
  comps.sort((a, b) => b.along * b.across - a.along * a.across);
  const area = comps.reduce((s, c) => s + (c.along + GAP_COMP) * (c.across + GAP_COMP), 0);
  const ratio = vertical ? W / H : H / W;  // 层内方向与跨层方向长度之比的目标值
  const lineLen = Math.max(Math.sqrt(area * ratio), ...comps.map(c => c.along));
  const pos = {};
  let a0 = 0, d0 = 0, thick = 0;
  for (const c of comps) {
    if (a0 > 0 && a0 + c.along > lineLen) { a0 = 0; d0 += thick + GAP_COMP; thick = 0; }
    c.ks.forEach(k => c.layers.get(k).forEach((id, i) => {
      const a = a0 + (i + 0.5 + (c.width - c.layers.get(k).length) / 2) * GAP_NODE, d = d0 + k * GAP_LEVEL;
      pos[id] = vertical ? {x: a, y: d} : {x: -d, y: a};
    }));
    a0 += c.along + GAP_COMP;
    thick = Math.max(thick, c.across);
  }
  const xs = Object.values(pos).map(p => p.x), ys = Object.values(pos).map(p => p.y);
  const scale = Math.min(W / (Math.max(...xs) - Math.min(...xs) + 160), H / (Math.max(...ys) - Math.min(...ys) + 80));
  return {pos, scale, gap: GAP_LEVEL};
}

function edgeLine(e, other) {
  const tag = e.direction === '正向' ? '<span class="tag pos">正向</span>' : '<span class="tag neg">负向</span>';
  const org = (e.origin === 'extracted' ? `原文 ${e.weight} 次` : ORIGIN_SHORT[e.origin] || e.origin) +
    ((e.conflict_path || []).length ? '，⚠ 与更长链路方向相反' : '');
  return `<li>${e.origin === 'part_of' ? '<span class="tag grey">组成</span>' : tag}${esc(other)} <span class="ev">（${org}）</span></li>`;
}

function showNode(r, id) {
  const n = r.nodes.find(x => x.id === id);
  if (!n) {
    document.getElementById('detail').innerHTML = '<div class="ev">点击图中的节点或边查看详情。</div>';
    return;
  }
  const eff = {'+': '<span class="tag pos">对估值净正向</span>', '-': '<span class="tag neg">对估值净负向</span>', '±': '<span class="tag mix">正负路径皆有</span>'}[n.effect]
    || '<span class="tag grey">未连到估值</span>';
  const ins = r.edges.filter(e => e.target === id), outs = r.edges.filter(e => e.source === id);
  const alias = (n.aliases || []).filter(a => a !== id);
  const label = n.label || id, L = x => labelOf(r, x);
  const note = x => `<li>第 ${x.para_id} 段（${esc(x.direction)}）${x.note ? '，' + esc(x.note) : ''}<div class="ev">“${esc(x.evidence)}”</div></li>`;
  const pending = n.unresolved || [], attached = n.attached || [], hints = n.hints || [];
  const replaced = (r.replaced_edges || []).filter(e => e.source === id);
  const rev = (r.reverse_relations || []).filter(e => e.source === id || e.cause === id || e.effect === id);
  document.getElementById('detail').innerHTML =
    `<h3>${esc(label)}</h3>${n.is_root ? '<span class="tag grey">估值</span>' : eff}` +
    `<span class="tag grey">第 ${n.layer} 层 ${esc(n.layer_name || '')}</span>${n.kind ? `<span class="tag grey">${esc(n.kind)}</span>` : ''}` +
    (n.connected === false ? '<span class="tag grey">未接入</span>' : '') +
    (label !== id ? `<div class="sec">本篇名称</div><div>${esc(id)}</div>` : '') +
    (n.definition ? `<div class="sec">定义</div><div>${esc(n.definition)}</div>` : '') +
    (alias.length ? `<div class="sec">本篇别名</div><div>${alias.map(esc).join('、')}</div>` : '') +
    ((n.merged || []).length ? `<div class="sec">向上合并进来的下位指标（${n.merged.length}）</div><div>${n.merged.map(esc).join('、')}</div>` : '') +
    (outs.length ? `<div class="sec">影响（${outs.length}）</div><ul>${outs.map(e => edgeLine(e, L(e.target))).join('')}</ul>` : '') +
    (ins.length ? `<div class="sec">受以下因素影响（${ins.length}）</div><ul>${ins.map(e => edgeLine(e, L(e.source))).join('')}</ul>` : '') +
    (hints.length ? `<div class="sec">原文只写了它、没写影响对象的说法（${hints.length}）</div><ul>${hints.map(note).join('')}</ul>` : '') +
    (pending.length ? `<div class="sec">原文未说明影响因素的悬挂记录（${pending.length}）</div><ul>${pending.map(note).join('')}</ul>` : '') +
    (attached.length ? `<div class="sec">已并入的悬挂记录（${attached.length}）</div><ul>${attached.map(note).join('')}</ul>` : '') +
    (rev.length ? `<div class="sec">与层级方向相反、未进图的原文关系（${rev.length}）</div><ul>${rev.map(e =>
      `<li>${esc(L(e.cause))} → ${esc(L(e.effect))}（${esc(e.direction)}）<div class="ev">“${esc(e.evidence[0] || '')}”</div></li>`).join('')}</ul>` : '') +
    (replaced.length ? `<div class="sec">剪掉的直连（另有更长链路，${replaced.length}）</div><ul>${replaced.map(e =>
      `<li>${e.direction === '正向' ? '<span class="tag pos">正向</span>' : '<span class="tag neg">负向</span>'}${esc(L(e.target))}` +
      ` <span class="ev">（${esc(ORIGIN_SHORT[e.origin] || '原文')}）</span>` +
      `<div class="ev">保留的链路（方向一致）：${e.path.map(p => esc(L(p))).join(' → ')}` +
      (e.evidence.length ? `<br>原文：“${esc(e.evidence[0])}”` : '') + '</div></li>').join('')}</ul>` : '');
}

function showEdge(e) {
  const L = x => labelOf(current, x);
  const tag = e.direction === '正向' ? '<span class="tag pos">正向</span>' : '<span class="tag neg">负向</span>';
  const org = ORIGIN_LONG[e.origin] || e.origin;
  const ev = e.evidence.length ? `<ul>${e.evidence.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '<div class="ev">无</div>';
  const ctx = e.context || [];
  document.getElementById('detail').innerHTML =
    `<h3>${esc(L(e.source))} → ${esc(L(e.target))}</h3>${e.origin === 'part_of' ? '' : tag}<span class="tag grey">${esc(org)}</span>` +
    (e.gap >= 2 ? `<span class="tag grey">跨 ${e.gap} 层</span>` : '') +
    ((e.conflict_path || []).length ? `<span class="tag mix">⚠ 与更长链路方向相反</span><div class="sec">方向相反的更长链路（直连因此保留；` +
      `可能是某条边的方向抽错了，也可能是两个不同的渠道）</div><div>${e.conflict_path.map(p => esc(L(p))).join(' → ')}</div>` : '') +
    (e.origin === 'extracted' ? `<div class="sec">支持次数</div><div>${e.weight}（正向 ${e.n_pos}，负向 ${e.n_neg}）</div>` : '') +
    (e.para_ids.length ? `<div class="sec">所在段落</div><div>${e.para_ids.join('、')}</div>` : '') +
    (e.origin === 'logic' || e.origin === 'part_of' ? '' : `<div class="sec">原文证据</div>${ev}`) +
    (ctx.length ? `<div class="sec">${e.origin === 'logic' ? '理由' : '说明'}</div><ul>${ctx.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '');
}

function renderTable() {
  const rows = DATA.global_names || [];
  document.getElementById('table').innerHTML = rows.length
    ? `<table><thead><tr><th>全局名称</th><th>层</th><th>研报数</th><th>各研报中的名称（研报数）</th><th>定义</th></tr></thead><tbody>` +
      rows.map(x => `<tr><td>${esc(x.global)}</td><td>${x.layer ?? ''}</td><td>${x.n_reports}</td>` +
        `<td>${x.members.map(m => `${esc(m.name)}（${m.n_reports}）`).join('、')}</td>` +
        `<td>${x.members.map(m => `<div class="ev">${esc(m.name)}：${esc(m.definition || '—')}</div>`).join('')}</td></tr>`).join('') +
      '</tbody></table>'
    : '<p class="ev">全局去重没有合并任何名称。</p>';
}

function tab(which) {
  const g = which === 'graph';
  document.getElementById('tab-graph').classList.toggle('on', g);
  document.getElementById('tab-table').classList.toggle('on', !g);
  document.getElementById('center').style.display = g ? 'flex' : 'none';
  document.getElementById('detail').style.display = g ? 'block' : 'none';
  document.getElementById('table').style.display = g ? 'none' : 'block';
  if (!g) renderTable();
}

document.getElementById('q').oninput = renderList;
document.getElementById('show-evid').onchange = draw;
document.getElementById('dir').onchange = draw;
document.getElementById('show-logic').onchange = draw;
document.getElementById('tab-graph').onclick = () => tab('graph');
document.getElementById('tab-table').onclick = () => tab('table');
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', draw);
current = reports[0] || null;
renderList();
if (typeof vis === 'undefined') document.getElementById('net').innerHTML = '<p style="padding:16px">vis-network 加载失败，请检查网络连接。</p>';
else draw();
</script>
</body>
</html>
'''


def main(graphs_path, out_html):
    data = json.loads(Path(graphs_path).read_text(encoding='utf-8'))
    payload = json.dumps(data, ensure_ascii=False).replace('</', '<\\/')  # 防止数据中的 </script> 提前结束脚本
    html = TEMPLATE.replace('__VIS_URL__', VIS_URL).replace('__DATA__', payload)
    Path(out_html).write_text(html, encoding='utf-8')
    print(f'已写入 {out_html}（{len(data["reports"])} 篇研报）')
