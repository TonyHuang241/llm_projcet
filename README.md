# llm_projcet

用大语言模型从卖方研报中抽取逻辑结构（概念和概念之间的因果关系构成的有向无环图），计算图结构变量，并检验它和分析师盈利预测准确度的关系。
研究结果见 `docs/研报逻辑结构与分析师预测准确度.md`。样本是 2024 年沪深300 成分股的研报。

## 目录

```
run.py                      唯一入口：python run.py <步骤> [参数]
src/
  config.py                 目录约定、跨步骤共用的数据文件路径
  llm.py                    大模型调用：接口、并发、调用缓存、不合格时重试
  ols.py                    固定效应 OLS、聚类标准误、缩尾（各项研究共用）
  text/                     研报纯文本 → 段落表、元数据表
    clean.py  meta.py  load.py
  graph/                    段落 → 每篇研报的概念和因果边
    extract.py  unify.py  common.py  load.py  prompts/  viewer.html  unify_review.html
  measure/                  边表 → 六个逻辑结构变量
    paths.py  variables.py  load.py
  accuracy/                 预测表 → 预测准确度、同行时滞、控制变量
    forecasts.py  peers.py  controls.py  load.py
  studies/
    forecast_accuracy/      对应 docs/研报逻辑结构与分析师预测准确度.md
      sample.py  regressions.py  tables.py  figures.py
docs/                       核心文档和 figures/；archive/ 是已不维护的旧文档
literature/                 参考文献，按主题分目录：文献梳理和笔记入 git，PDF 不入 git
data/                       输入数据（不入 git）
output/                     各步骤的结果（不入 git）：output/<步骤>/
.env                        API_KEY、BASE_URL、MODEL 等，说明见 src/llm.py 开头；不入 git
```

## 运行

在仓库根目录运行，使用 `C:\Users\hung\projects\.venv` 里的 Python（系统 PATH 上的 Python 没有 pandas 等依赖）。

| 命令 | 做什么 | 读 | 写 | 调用大模型 | 耗时 |
|---|---|---|---|---|---|
| `python run.py text` | 文本清理 | `data/HS300_reports_2024/` | `data/` 的段落表、元数据表 | 否 | — |
| `python run.py graph` | 逻辑结构抽取 | 段落表、元数据表 | `output/graph/<运行名>/` | 是 | 取决于篇数 |
| `python run.py unify` | 跨研报统一概念名称（可选） | `output/graph/<运行名>/` | 同一目录 | 是 | — |
| `python run.py measure` | 逻辑结构变量 | `output/graph/` | `output/measure/` | 否 | 3 秒 |
| `python run.py accuracy` | 预测准确度和控制变量 | `data/` 的预测表、行情、财务 | `output/accuracy/` | 否 | 15 秒 |
| `python run.py study forecast_accuracy` | 样本、回归、文档的表和图 | `output/measure/`、`output/accuracy/` | `output/studies/forecast_accuracy/`、`docs/figures/` | 否 | 30 秒 |

每个步骤的选项用 `python run.py <步骤> --help` 查看，各步骤的口径写在 `src/<步骤>/` 各文件开头。
只想重现文档里的数字时，依次运行后三条即可，不需要重新清理文本和抽取。

## 约定

- **步骤名只定义一次**（`run.py` 的 `STEPS`）：命令名、`src/` 下的目录名、`output/` 下的目录名相同。
- **每个步骤的接口相同**：`src/<步骤>/__init__.py` 提供 `add_args(parser)` 和 `run(args)`，默认运行参数写在本目录里；`load.py` 负责读本步骤的成品，下游通过它读，不自己拼路径。`run.py` 只做分发，选中子命令后才导入对应模块。
- **每一步只读 `data/` 和前面步骤在 `output/` 下的成品。**
- **变量层只出原始值**：`measure`、`accuracy` 对全部研报计算，不缩尾、不标准化、不选样本。选样本、券商分组、缩尾、回归都在 `src/studies/<研究>/` 里做。
- **文件名只写内容**，不带前缀和版本号；版本交给 git。带参数的抽取或训练例外：按运行名分子目录（`output/graph/<运行名>/`），可以多套并存。

加一种新变量：新建 `src/<步骤>/`，在 `run.py` 的 `STEPS` 里加一行。加一项新研究：新建 `src/studies/<名字>/`，用 `python run.py study <名字>` 运行。

## 各步骤

### text：文本清理

把研报纯文本清理成正文段落，按文字特征剔除图表、表格、侧栏、页眉页脚、风险提示和免责声明（规则见 `src/text/clean.py`）。写出两张表，用 `report_id`（txt 文件名里的 GUID）对应：

- `data/hs300_2024_paragraphs.parquet`：段落表，每行一个正文段落（旁边同名 `.txt` 便于人工检查）。
- `data/hs300_2024_reports.parquet`（另有同名 `.csv`）：研报元数据表，每个文件一行，取法见 `src/text/meta.py`。

| 列 | 含义 |
|---|---|
| `date` | 发布日期，取文件所在目录的月/日 |
| `broker` | 券商 |
| `report_type` | 报告类型：公司、行业、策略、宏观、固收、金工、晨报、基金、未知 |
| `is_stock_report` | 是不是个股公司研报 |
| `company`、`stock_code` | 这篇研报讲的那只股票 |
| `analysts`、`analyst_ids` | 分析师姓名、执业证书编号（署名栏在很多纯文本里已经丢了，只有两三成能取到） |
| `status`、`n_paragraphs`、`n_chars` | ok 有正文 / empty 空文件、扫描件、乱码、外文 / failed；保留的段落数和字数 |

`python run.py text --limit 100` 只抽 100 篇试跑，写到 `*_sample100.parquet`，不覆盖全量结果。
**重跑全量会改写段落表**；抽取的存档指纹包含段落文字，文字变了的研报下次抽取时会重新调用大模型。

### graph：逻辑结构抽取

流程在 `src/graph/extract.py`，提示词在 `src/graph/prompts/extract.md`。不预设任何概念，也不做补全，每篇研报做三件事：

1. **合并段落**：正文段落按顺序合并成不超过 `chunk_chars` 字的块，后面每块抽概念、连边各调用一次大模型。
2. **抽概念**：按块顺序通读，滚动维护本篇概念表（规范名、别名、定义、类型：指标 / 事件 / 状态）；每个新概念要给出原文叫法，程序逐字核对，对不上的不收。
3. **连边**：各块并行，在原文里找概念之间“A 影响 B”的关系（只要因果，不要并列、包含、同义），引文逐字核对。各块的边按（A, B）合并，有环时每次删掉环上引文最少、在原文中出现得最晚的一条，得到有向无环图；没有连上边的概念不进图，仍留在概念表里。

每篇研报做完即存档，中断后重新运行接着跑；改了提示词或 `chunk_chars` 会让存档失效、重新处理（调用缓存还在，没变的调用不花钱）。
**不要改 `src/graph/prompts/` 下的文件**，除非确实要重新抽取。

抽取结果按运行名分目录，每次运行的选样和参数写在 `src/graph/__init__.py` 的 `RUNS` 里，`--run` 选一个（默认 `deep`）：

| 运行 | 范围 | 每块字数 | 现状 |
|---|---|---|---|
| `deep` | 全部公司的深度报告（首页有深度、首次覆盖、公司专题等类型标注，且正文超过 5000 字） | 5000 | 选中 975 篇，974 篇有结果、420 家公司 |
| `pilot` | 7 家试跑公司的全部研报（比亚迪只抽深度） | 2500 | 65 篇；试跑之后文本清理改过，按现在的代码已不能完全复现当时的选样 |

```bash
python run.py graph --dry-run          # 只统计选中的研报和块数，不调用大模型
python run.py graph --archived-only    # 只用已有存档重写输出，不调用大模型
python run.py graph --companies 600519.SH,002415.SZ --output output/graph/<新运行名>
```

不要对 `pilot` 跑 `--archived-only`：它只有 56 篇的存档还有效，会把结果表改写成 56 篇。

一次运行的输出（`output/graph/<运行名>/`）：

- `概念DAG.html`：查看页面，双击打开，左侧选研报；点节点突出它的上下游，点边看原文引文和所在段落。
- `data/concepts.parquet`（每篇的每个概念一行，`in_dag` 是否进图）、`edges.parquet`（每条边一行，`in_dag` 为假的是因成环删掉的边，`support` 是引文处数）、`reports.parquet`（每篇的概念数、边数等）。
- `cache/`：调用缓存、各篇存档、出错记录、进度日志。

下游用 `src/graph/load.py` 读：同一篇研报在两次运行里都有时用 `deep` 的。`output/graph/_stability/` 是同一批研报重复抽取的结果，用来衡量抽取噪声。

### unify：跨研报统一概念名称（可选）

同一个量在不同研报里常有不同叫法（营收 / 营业收入）。`unify` 按公司分开，只合并明显完全相同的名称，范围、口径、主体不同的不合并；做法和复核规则见 `src/graph/unify.py`，提示词在 `src/graph/prompts/unify.md`。
**现在的回归不使用统一后的名称**（逻辑结构变量按每篇研报自己的图计算）；保留它是为了以后做跨研报的比较。
`deep` 这次运行的名称统一没有做完：合并提议 7036 组已做完，复核做到 3270 / 5160 组时停下，已完成的都在缓存里；`concept_unify.parquet` 还是文本清理修复之前 939 篇的版本。

### measure：逻辑结构变量

对 `graph` 的全部研报（1018 篇）各算一行，写 `output/measure/report_measures.parquet`。口径见 `src/measure/variables.py`。

- 源点：没有入边的概念；严格汇点：有入边、没有出边的概念；路径：从源点到汇点的有向路径。
- 两套变量各三个：源点数、ln(1 + 路径数)、平均路径长度。
  - 全文逻辑结构（`WG_*`）：整张图。
  - “投资评级”对应的逻辑结构（`RT_*`）：“投资评级”节点及其全部上游，只在它是严格汇点时有定义。

### accuracy：预测准确度和控制变量

对全部研报计算，写 `output/accuracy/` 下三张表，各列说明见 `src/accuracy/load.py`，口径见 `forecasts.py`、`peers.py`、`controls.py`。

- `report_accuracy.parquet`：每篇研报一行。本篇预测（`data/EST_COMP_TN` 里 `RPT_ID` 等于 `report_id` 大写、`SEC_CD` 等于本公司的那一行）、实际值、相对同行的预测误差 PMAFE、排名分数 Score、同行数和同行时滞、六个控制变量、基础样本标记 `base`。
- `report_peers.parquet`：FY2025 比较组的同行明细。
- `pair_accuracy.parquet`：每个（公司，券商）一行，券商 2024 年内最后一次 FY2025 预测的 PMAFE、时效差和控制变量。

### study forecast_accuracy：回归和文档的表、图

样本范围是 `data/hs300_2024_dag_covered.csv` 里 241 家公司、图里有边的研报，共 791 篇；基础样本 786 篇，公司-券商层面 760 对。
券商分组、行业、缩尾和回归设定见 `sample.py`、`regressions.py`。输出在 `output/studies/forecast_accuracy/`：

- `sample_report.parquet`、`sample_pair.parquet`：两个层面的回归样本。
- `reg_all.csv`：全部回归系数的长表。
- `tables/table1_….csv` 到 `table8_….csv`：文档的表 1–8；`tables/text_numbers.csv`：正文、图注和第 4 节出现的每个数字及其算法。
- `docs/figures/` 下的两张图。

文档的每张表、每张图、每一节对应哪个输出文件、由哪个函数生成，列在 `src/studies/forecast_accuracy/__init__.py` 开头。

## 数据

`data/` 不入 git，需要这些文件：

| 路径 | 内容 | 谁用 |
|---|---|---|
| `HS300_reports_2024/` | 研报纯文本，`MM/DD/<GUID>.txt` | text、graph |
| `EST_COMP_TN/<年>/<年>-<月>.parquet` | 分析师盈利预测 | accuracy |
| `market_data/daily_stock_data.parquet` | 个股日行情 | accuracy |
| `market_data/financial_data.parquet` | 财务数据（年报实际值、公告日） | accuracy |
| `market_data/industry.xlsx` | 行业分类（申万一级） | study |
| `hs300_2024_dag_covered.csv` | 样本公司名单（241 家） | study |

## 历史

2026-11-23 精简过一次项目结构。之前的全部代码和文档在标签 `pre-cleanup` 里，包括已移出的几条研究线：五层估值逻辑图和研报分歧度、逻辑结构与 2025 年股价和财务指标的回归、研报发布后的市场反应、预测准确度回归的早期版本。它们的文档在 `docs/archive/`。
