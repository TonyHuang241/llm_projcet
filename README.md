# llm_projcet

用大语言模型从卖方研报中抽取估值逻辑图，并度量同一家公司不同研报之间的逻辑框架差异。

```
main.py               启动入口：修改 TASK 选择要运行的功能
src/                  代码（Python 包）
  config.py             目录约定（DATA_DIR、OUTPUT_DIR、PROMPT_DIR）
  llm_client.py         大模型接口（兼容 OpenAI Chat Completions）、.env 读取、进度日志
  report_text.py        第 0 步：研报 PDF → 正文段落
  report_logic.py       第 1 步：逻辑图抽取的主流程
  logic/                第 1 步各环节的实现（base、extract、graph、complete、output）
  fin_items.py          第 1 步预置的会计层清单
  report_compare.py     第 2 步：同一公司不同研报的逻辑框架对比
  viz/                  页面：逻辑图（plot_report_logic.py）、介绍页（plot_intro.py + intro_template.html）
  experiments/          试验：基于 KGGen 框架的研报知识图谱（report_kg.py）
  prompts/              大模型提示词
docs/                 设计文档、方法调研笔记、介绍页
data/                 研报 PDF 与正文段落（不入 git）
output/               抽取与对比结果、大模型调用缓存（不入 git）
literature/           参考文献，按主题分目录：文献梳理和笔记入 git，PDF 不入 git
.env                  API_KEY、BASE_URL、MODEL 等，见 src/llm_client.py 的说明；不入 git
```

## 运行

在仓库根目录运行，使用 `C:\Users\hung\projects\.venv` 里的 Python（系统 PATH 上的 Python 没有 pandas 等依赖）。

```bash
python main.py     # 按 main.py 中的 TASK 运行
```

| TASK | 功能 | 调用大模型 |
|---|---|---|
| `report_text` | 研报 PDF → `data/report_paragraphs.parquet` | 否 |
| `report_logic` | 逻辑图抽取 → `output/report_logic/` | 是（已完成的调用走缓存） |
| `report_logic_html` | 只重画逻辑图页面 | 否 |
| `report_compare` | 框架对比 → `output/report_compare/` | 是（每家公司一次） |
| `intro_page` | 生成 `docs/逻辑抽取与分歧度量介绍.html` | 否 |
| `report_kg_parse` / `report_kg_check` / `report_kg_build` | 知识图谱试验 → `output/knowledge_graph/` | 由 Claude 子代理完成 |

`.claude/launch.json` 里的 `report-logic-html`、`intro-page` 可在浏览器预览两个页面。

## 流程

把一篇研报的推理链抽成五层有向图：公司估值 → 会计科目（预置）→ 经营指标 → 公司经营与行业 → 外部环境。

1. **正文抽取**（`report_text.py`）：研报 PDF → 正文段落 parquet，剔除图表、表格、侧栏和免责声明。
2. **逻辑图抽取**（`report_logic.py`、`logic/`）：概念分层、原文连边、子图补全、包含关系缩减、全局去重。提示词在 `prompts/report_logic_prompt.md`。
3. **框架对比**（`report_compare.py`）：每家公司把概念归成主题，把图转成矩阵，算逐对、逐篇的差异指标。
4. **页面**（`viz/`）：交互式逻辑图，以及说明抽取流程和分歧度量的介绍页。

## 文档

- `docs/研报逻辑图抽取框架设计.md`：框架设计、各条规则的来由、各版本的结果。
- `docs/研报因果图合并与逻辑跳跃修正.md` 及 `docs/notes/`：抽取方法的文献调研。
- `docs/逻辑抽取与分歧度量介绍.html`：介绍页（由 `intro_page` 生成）。
- `literature/分析师分歧/`、`literature/殊途同归/`：两个方向的文献与梳理；`literature/因果图抽取方法/`：抽取方法的论文。
