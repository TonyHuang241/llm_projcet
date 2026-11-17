# llm_projcet

用大语言模型从卖方研报中抽取估值逻辑图，并度量同一家公司不同研报之间的逻辑框架差异。

```
clear_txt.py            第一步入口：研报纯文本清理（不调用大模型）
extract_logic_chain.py  第二步入口：逻辑链抽取；另可用 --task 运行框架对比、生成介绍页
src/                    代码（Python 包）
  config.py               目录约定（DATA_DIR、OUTPUT_DIR、PROMPT_DIR，输出目录的布局 OutDir）
  llm_client.py           大模型接口（兼容 OpenAI Chat Completions）、.env 读取、进度日志
  report_text.py          文本清理：研报纯文本 → 正文段落
  report_logic.py         逻辑链抽取的主流程
  logic/                  逻辑链抽取各环节的实现（base、extract、graph、complete、output）
  fin_items.py            预置的会计层清单
  report_compare.py       同一公司不同研报的逻辑框架对比
  viz/                    页面：逻辑图查看页面（logic_viewer.html）、介绍页（plot_intro.py + intro_template.html）
  prompts/                大模型提示词
docs/                   设计文档、方法调研笔记、介绍页
data/                   研报纯文本与清理后的正文段落（不入 git）
output/                 抽取结果（不入 git），布局见下
literature/             参考文献，按主题分目录：文献梳理和笔记入 git，PDF 不入 git
.env                    API_KEY、BASE_URL、MODEL 等，见 src/llm_client.py 的说明；不入 git
```

## 运行

在仓库根目录运行，使用 `C:\Users\hung\projects\.venv` 里的 Python（系统 PATH 上的 Python 没有 pandas 等依赖）。
两个脚本的输入、输出路径和试跑篇数都在各自开头的“运行参数”里，也可以用命令行参数临时覆盖。

```bash
python clear_txt.py              # 第一步：data/HS300_reports_2024/ 的 txt → data/hs300_2024_paragraphs.parquet，带进度条
python extract_logic_chain.py    # 第二步：正文段落 → output/ 下的逻辑图（调用大模型，已完成的调用走缓存）
```

| 命令 | 功能 | 调用大模型 |
|---|---|---|
| `python clear_txt.py --limit 100` | 只跨月份随机抽 100 篇试跑，写到 `data/hs300_2024_paragraphs_sample100.parquet` | 否 |
| `python extract_logic_chain.py --limit 5` | 只抽取前 5 篇（按 report_id 排序） | 是 |
| `python extract_logic_chain.py --input <段落 parquet> --output <目录>` | 换输入和输出目录（试跑时免得覆盖正式结果） | 是 |
| `python extract_logic_chain.py --task compare` | 框架对比 → `output/data/compare/` | 是（每家公司一次） |
| `python extract_logic_chain.py --task intro_page` | 生成 `docs/逻辑抽取与分歧度量介绍.html`（样例研报写死在 `src/viz/plot_intro.py`） | 否 |

## 输出

```
output/
  研报逻辑图.html   入口：双击打开。左侧选研报，选中时才读取这一篇的数据、计算布局并绘图
  data/             结果数据
    graphs/           每篇研报一个逻辑图数据文件，加上目录 index.js 和全局名称表 global_names.js
    edges.parquet、nodes.parquet、relations.parquet   扁平表
    compare/          框架对比的结果（themes、matrices、pair_metrics、report_metrics）
  cache/            大模型调用缓存（*_cache.jsonl，重跑时跳过已完成的调用）和进度日志 progress.log
```

页面要和 `data/` 放在同一个目录下，需要联网加载 vis-network。`.claude/launch.json` 里的 `report-logic-html`、`intro-page` 可在浏览器预览两个页面。

目前 `output/` 里是旧 PDF 流程那 16 篇研报（`data/report_paragraphs.parquet`）的结果；新 txt 的 3 篇试跑结果在 `output_sample/`。

## 流程

把一篇研报的推理链抽成五层有向图：公司估值 → 会计科目（预置）→ 经营指标 → 公司经营与行业 → 外部环境。

1. **文本清理**（`clear_txt.py`、`src/report_text.py`）：研报纯文本 → 正文段落 parquet，按文字特征剔除图表、表格、侧栏、页眉页脚、风险提示和免责声明。已有处理好的文本时可跳过，自备一份至少含 report_id、para_id、text 三列的 parquet 即可。
2. **逻辑链抽取**（`extract_logic_chain.py`、`src/report_logic.py`、`src/logic/`）：概念分层、原文连边、子图补全、包含关系缩减、全局去重。提示词在 `prompts/report_logic_prompt.md`。
3. **框架对比**（`src/report_compare.py`）：每家公司把概念归成主题，把图转成矩阵，算逐对、逐篇的差异指标。
4. **页面**（`src/viz/`）：交互式逻辑图，以及说明抽取流程和分歧度量的介绍页。

## 文档

- `docs/研报逻辑图抽取框架设计.md`：框架设计、各条规则的来由、各版本的结果。
- `docs/研报因果图合并与逻辑跳跃修正.md` 及 `docs/notes/`：抽取方法的文献调研。
- `docs/逻辑抽取与分歧度量介绍.html`：介绍页（由 `--task intro_page` 生成）。
- `literature/分析师分歧/`、`literature/殊途同归/`：两个方向的文献与梳理；`literature/因果图抽取方法/`：抽取方法的论文。
