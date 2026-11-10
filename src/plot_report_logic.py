"""把 report_logic.py 输出的 report_graphs.json 画成单文件 HTML：每篇研报一张以“公司估值”为根的分层逻辑图。

左侧按公司列出研报，点选后在中间画图（vis-network 分层布局，根节点在最上方，箭头由影响因素指向被影响因素）：
  边的颜色表示方向：红色为正向、绿色为负向（A 股习惯）；实线是原文抽取的关系，虚线是连接估值时由大模型补充的关系，
  点线是大模型多次失败后直接连到估值的兜底边（方向未经判断，需要复核）；
  节点底色表示沿路径相乘后对估值的净影响：偏红为正、偏绿为负、黄色为既有正向路径又有负向路径。
点击节点会突出它的上游因素和通往估值的路径，右侧显示合并进来的原始名称；点击边显示原文证据和段落编号。
某一层节点很多时（长篇深度报告）自动改成根节点在右的横向布局，也可以手动切换。另有“节点合并表”页，用来检查去重结果。
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
  <div class="tabs"><button id="tab-graph" class="on">逻辑图</button> <button id="tab-table">节点合并表</button></div>
  <div class="legend">
    <span><i style="border-color:var(--pos)"></i>正向</span>
    <span><i style="border-color:var(--neg)"></i>负向</span>
    <span><i style="border-top-style:dashed;border-color:var(--muted)"></i>大模型补充</span>
    <span><i style="border-top-style:dotted;border-color:var(--muted)"></i>兜底连接</span>
    <span><b style="background:var(--pos-bg)"></b>对估值净正向</span>
    <span><b style="background:var(--neg-bg)"></b>净负向</span>
    <span><b style="background:var(--mix-bg)"></b>正负皆有</span>
  </div>
</header>
<main>
  <nav id="side"><input id="q" placeholder="筛选研报或节点名"><div id="list"></div></nav>
  <section id="center">
    <div id="meta"></div>
    <div id="tools">
      <label><input type="checkbox" id="show-linked" checked>显示补充连接</label>
      <label><input type="checkbox" id="show-evid">边上显示支持次数</label>
      <label>布局 <select id="dir"><option value="auto">自动</option><option value="UD">根节点在上</option><option value="RL">根节点在右</option></select></label>
      <span id="hint">点击节点突出其上游因素和通往估值的路径，点击边查看原文证据，点空白处恢复；滚轮缩放，拖动平移</span>
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
const WIDE_LEVEL = 12;  // “自动”布局下，某一层节点数超过该值就改用根节点在右
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;'}[c]));
let current = null, network = null;

const reports = DATA.reports.slice().sort((a, b) => (a.company || '').localeCompare(b.company || '', 'zh') || (b.date || '').localeCompare(a.date || ''));

function renderList() {
  const q = document.getElementById('q').value.trim().toLowerCase();
  const box = document.getElementById('list');
  box.innerHTML = '';
  let company = null;
  for (const r of reports) {
    const hay = [r.company, r.broker, r.title, r.date, ...r.nodes.map(n => n.id)].join(' ').toLowerCase();
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
  const statusText = {already: '原文逻辑已全部落到估值', linked: '已由大模型补充连接到估值', fallback: '部分终点为兜底连接，需复核', failed: '连接失败'}[s.link_status] || s.link_status;
  document.getElementById('meta').innerHTML =
    `<div class="t">${esc(r.company)}｜${esc(r.broker)} ${esc(r.date)}｜${esc(r.title)}</div>` +
    `<div class="s">${r.n_paragraphs} 段 · ${s.n_triples} 条三元组 · ${s.n_nodes} 个节点 · 原文边 ${s.n_extracted} 条 · 补充边 ${s.n_linked} 条` +
    (s.n_fallback ? ` · 兜底边 ${s.n_fallback} 条` : '') + ` · 连接前 ${s.components_before_link} 个子图 · 最深 ${s.depth} 层` +
    (s.removed_cycle_edges ? ` · 去环删边 ${s.removed_cycle_edges} 条` : '') + ` · ${statusText}</div>`;
  const showLinked = document.getElementById('show-linked').checked;
  const showW = document.getElementById('show-evid').checked;
  const edges = r.edges.filter(e => showLinked || e.origin === 'extracted');
  const used = new Set([ROOT]);
  edges.forEach(e => { used.add(e.source); used.add(e.target); });
  const nodes = r.nodes.filter(n => used.has(n.id)).map(n => {
    const c = nodeColor(n);
    return {id: n.id, label: n.id, level: n.level, shape: 'box', margin: 8,
      color: {background: c.background, border: c.border, highlight: {background: c.background, border: css('--accent')}},
      font: {color: c.font, size: n.is_root ? 18 : 14, face: 'PingFang SC, Microsoft YaHei, sans-serif', bold: n.is_root},
      borderWidth: n.is_root ? 2 : 1, widthConstraint: {maximum: 130}, borderDashes: n.new ? [4, 3] : false};
  });
  const visEdges = edges.map((e, i) => ({
    id: i, from: e.source, to: e.target, arrows: {to: {enabled: true, scaleFactor: 0.6}},
    color: {color: css(e.direction === '正向' ? '--pos' : '--neg'), highlight: css('--accent'), opacity: e.origin === 'extracted' ? 1 : 0.75},
    dashes: e.origin === 'linked' ? [8, 6] : e.origin === 'fallback' ? [2, 5] : false,
    width: e.origin === 'extracted' ? 1.2 + Math.min(e.weight, 5) * 0.5 : 1.2,
    label: (e.direction === '正向' ? '+' : '−') + (showW && e.weight ? ' ×' + e.weight : ''),
    font: {color: css('--muted'), size: 11, strokeWidth: 3, strokeColor: css('--panel')},
  }));
  // 布局：同一层节点不多时根节点在上（树形），某一层节点很多时改成根节点在右，避免图过宽
  let dir = document.getElementById('dir').value;
  if (dir === 'auto') {
    const perLevel = {};
    nodes.forEach(n => { perLevel[n.level] = (perLevel[n.level] || 0) + 1; });
    dir = Math.max(...Object.values(perLevel)) > WIDE_LEVEL ? 'RL' : 'UD';
  }
  const vertical = dir === 'UD';
  visEdges.forEach(e => { e.smooth = {type: 'cubicBezier', forceDirection: vertical ? 'vertical' : 'horizontal', roundness: 0.4}; });
  const options = {
    layout: {hierarchical: {enabled: true, direction: dir, levelSeparation: vertical ? 120 : 230, nodeSpacing: vertical ? 150 : 46,
      treeSpacing: vertical ? 180 : 60, blockShifting: true, edgeMinimization: true, parentCentralization: true}},
    physics: false, interaction: {hover: true, tooltipDelay: 150, navigationButtons: false}, edges: {selectionWidth: 1.5},
  };
  if (network) network.destroy();
  const nodeSet = new vis.DataSet(nodes), edgeSet = new vis.DataSet(visEdges);
  network = new vis.Network(document.getElementById('net'), {nodes: nodeSet, edges: edgeSet}, options);
  network.once('afterDrawing', () => network.fit({animation: false}));
  network.on('click', p => {
    if (p.nodes.length) { focus(p.nodes[0]); showNode(r, p.nodes[0]); }
    else if (p.edges.length) { focus(null); showEdge(edges[p.edges[0]]); }
    else focus(null);
  });

  // 点击节点：只突出它的上游因素和它通往公司估值的路径，其余淡化；点空白处恢复
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

function edgeLine(e, other) {
  const tag = e.direction === '正向' ? '<span class="tag pos">正向</span>' : '<span class="tag neg">负向</span>';
  const org = e.origin === 'extracted' ? `原文 ${e.weight} 次` : e.origin === 'linked' ? '大模型补充' : '兜底连接';
  return `<li>${tag}${esc(other)} <span class="ev">（${org}）</span></li>`;
}

function showNode(r, id) {
  const n = r.nodes.find(x => x.id === id);
  if (!n) return;
  const eff = {'+': '<span class="tag pos">对估值净正向</span>', '-': '<span class="tag neg">对估值净负向</span>', '±': '<span class="tag mix">正负路径皆有</span>'}[n.effect] || '';
  const ins = r.edges.filter(e => e.target === id), outs = r.edges.filter(e => e.source === id);
  const alias = (n.aliases || []).filter(a => a !== id);
  document.getElementById('detail').innerHTML =
    `<h3>${esc(id)}</h3>${n.is_root ? '<span class="tag grey">根节点</span>' : eff}<span class="tag grey">第 ${n.level} 层</span>` +
    (n.new ? '<span class="tag grey">连接时新增</span>' : '') +
    (alias.length ? `<div class="sec">合并进来的原始名称</div><div>${alias.map(esc).join('、')}</div>` : '') +
    (outs.length ? `<div class="sec">影响（${outs.length}）</div><ul>${outs.map(e => edgeLine(e, e.target)).join('')}</ul>` : '') +
    (ins.length ? `<div class="sec">受以下因素影响（${ins.length}）</div><ul>${ins.map(e => edgeLine(e, e.source)).join('')}</ul>` : '');
}

function showEdge(e) {
  const tag = e.direction === '正向' ? '<span class="tag pos">正向</span>' : '<span class="tag neg">负向</span>';
  const org = {extracted: '原文抽取', linked: '连接估值时由大模型补充', fallback: '兜底连接（方向未判断，需复核）'}[e.origin];
  const ev = e.evidence.length ? `<ul>${e.evidence.map(x => `<li>${esc(x)}</li>`).join('')}</ul>` : '<div class="ev">无</div>';
  document.getElementById('detail').innerHTML =
    `<h3>${esc(e.source)} → ${esc(e.target)}</h3>${tag}<span class="tag grey">${org}</span>` +
    (e.origin === 'extracted' ? `<div class="sec">支持次数</div><div>${e.weight}（正向 ${e.n_pos}，负向 ${e.n_neg}）</div>` +
      `<div class="sec">所在段落</div><div>${e.para_ids.join('、')}</div>` : '') +
    `<div class="sec">${e.origin === 'extracted' ? '原文证据' : '补充理由'}</div>${ev}`;
}

function renderTable() {
  const rows = new Map();
  for (const r of DATA.reports) for (const n of r.nodes) {
    const row = rows.get(n.id) || {id: n.id, aliases: n.aliases || [], reports: new Set(), companies: new Set(), isNew: n.new};
    row.reports.add(r.report_id); row.companies.add(r.company);
    rows.set(n.id, row);
  }
  const list = [...rows.values()].sort((a, b) => b.reports.size - a.reports.size || b.aliases.length - a.aliases.length);
  document.getElementById('table').innerHTML = `<table><thead><tr><th>规范节点</th><th>研报数</th><th>涉及公司</th><th>合并的原始名称</th></tr></thead><tbody>` +
    list.map(x => `<tr><td>${esc(x.id)}${x.isNew ? ' <span class="ev">（连接时新增）</span>' : ''}</td><td>${x.reports.size}</td>` +
      `<td>${[...x.companies].map(esc).join('、')}</td><td>${x.aliases.filter(a => a !== x.id).map(esc).join('、')}</td></tr>`).join('') +
    '</tbody></table>';
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
document.getElementById('show-linked').onchange = draw;
document.getElementById('show-evid').onchange = draw;
document.getElementById('dir').onchange = draw;
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
