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
  mdpdf.py                  文档 md → HTML → 本机 Chrome / Edge 无头打印成 PDF（各项研究共用）
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
      sample.py  errors.py  regressions.py  tables.py  figures.py
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
| `python run.py study forecast_accuracy` | 样本、描述性统计、方差分解、回归、文档的表和图（分环节，见下文） | `output/graph/`、`output/measure/`、`output/accuracy/`、`data/`（含 `forecast_errors.parquet`） | `output/studies/forecast_accuracy/`、`docs/figures/` | 否 | 1.5 分钟 |
| `python run.py study forecast_accuracy --part pdf` | 文档 md 转 PDF | `docs/` 的 md 和图 | md 旁边同名 `.pdf` | 否 | 几秒 |

每个步骤的选项用 `python run.py <步骤> --help` 查看，各步骤的口径写在 `src/<步骤>/` 各文件开头。
只想重现文档里的数字时，依次运行 measure、accuracy、study 三条即可，不需要重新清理文本和抽取。

### 只跑某一个环节

试跑、补抽或只重做下游某一环时用这些选项（graph、unify 的选项见 `src/graph/__init__.py`、`src/graph/unify.py` 开头）：

| 命令 | 做什么 |
|---|---|
| `python run.py text --limit 100` | 只抽 100 篇试跑，写到 `*_sample100.parquet`，不覆盖全量结果 |
| `python run.py graph --dry-run` | 只统计选中的研报和块数，不调用大模型 |
| `python run.py graph --report-ids <研报ID>` | 只处理这几篇（逗号分隔），结果并入同一个输出目录，其余研报只读存档 |
| `python run.py graph --limit 20 --output output/graph/<试跑名>` | 在选中的研报里按种子随机抽 20 篇试跑；另给 `--output`，免得改写正式结果（同 `--companies`） |
| `python run.py graph --archived-only` | 只用已有存档重写输出，不调用大模型（不要对 `pilot` 用） |
| `python run.py unify --dry-run` | 只统计候选组，不调用大模型 |
| `python run.py unify --companies 600519.SH,002415.SZ` | 只统一这几家公司的研报 |
| `python run.py measure` | 重算逻辑结构变量（3 秒，不调用大模型） |
| `python run.py accuracy` | 重算预测准确度和控制变量（15 秒） |
| `python run.py study forecast_accuracy --part describe,tables` | 只做列出的环节（sample、describe、decomp、regress、tables、figures、pdf），各环节只读前面环节存在输出目录里的成品 |
| `python run.py study forecast_accuracy --part regress --blocks paired,firm_broker:paired` | 只重估这几个回归块，`reg_all.csv` 里其余行不动 |
| `python run.py study forecast_accuracy --part pdf [--doc <md 路径>]` | 文档转 PDF，默认主文档 |

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
段落表里有一处人工替换：视源股份的研报 `5DF25733-0934-58FF-B77D-3F1464B36B8B` 第 32 段的“阿鲁纳恰尔邦”换成了“某邦”（原词会触发模型的内容过滤，整篇抽不出来）。重跑全量后要重新替换这一格，再用 `python run.py graph --report-ids` 补上这一篇。

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
| `deep` | 全部公司的深度报告（首页有深度、首次覆盖、公司专题等类型标注，且正文超过 5000 字） | 5000 | 选中 975 篇，全部有结果、421 家公司 |
| `pilot` | 7 家试跑公司的全部研报（比亚迪只抽深度） | 2500 | 65 篇；试跑之后文本清理改过，按现在的代码已不能完全复现当时的选样 |

```bash
python run.py graph --dry-run          # 只统计选中的研报和块数，不调用大模型
python run.py graph --archived-only    # 只用已有存档重写输出，不调用大模型
python run.py graph --report-ids <研报ID>   # 只处理这一篇（逗号分隔可以给几篇），结果并入同一个输出目录，其余研报只读存档
python run.py graph --companies 600519.SH,002415.SZ --output output/graph/<新运行名>
```

单独重抽或补抽某一篇用 `--report-ids`，不要用 `--companies`：后者不另给 `--output` 时会把结果表改写成只有这几家公司。
`--report-ids` 同时换了 `--chunk-chars` 等参数时是试跑：其余研报的存档按新参数都对不上，所以只抽取这几篇、在日志里报告概念数和边数，不存档、不写输出。

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

对 `graph` 的全部研报（1019 篇）各算一行，写 `output/measure/report_measures.parquet`。口径见 `src/measure/variables.py`。

- 源点：没有入边的概念；严格汇点：有入边、没有出边的概念；路径：从源点到汇点的有向路径。
- 两套变量各三个：源点数、ln(1 + 路径数)、平均路径长度。
  - 全文逻辑结构（`WG_*`）：整张图。
  - “投资评级”对应的逻辑结构（`RT_*`）：“投资评级”节点及其全部上游，只在它是严格汇点时有定义。

### accuracy：预测准确度和控制变量

对全部研报计算，写 `output/accuracy/` 下三张表，各列说明见 `src/accuracy/load.py`，口径见 `forecasts.py`、`peers.py`、`controls.py`。

预测误差按预测窗口定义：H1、H2、H3 是预测行的第 1、2、3 年预测（`PROFIT_T1`–`PROFIT_T3`），目标年度 = 预测行的 `RPT_YR_T` + h，逐篇不同（`RPT_YR_T` = 2022 的研报 H2 是 FY2024，= 2023 的是 FY2025）。窗口相关的列都带后缀 `_H1`、`_H2`、`_H3`；比较组是其他券商对同一公司、同一目标年度的预测。研究层的主窗口由 `src/studies/forecast_accuracy/sample.py` 的 `MAIN_H`（现为 2）决定。

- `report_accuracy.parquet`：每篇研报一行。本篇预测（`data/EST_COMP_TN` 里 `RPT_ID` 等于 `report_id` 大写、`SEC_CD` 等于本公司的那一行）、各窗口的实际值、相对同行的预测误差 PMAFE、排名分数 Score、同行数和同行时滞，六个控制变量、基础样本标记 `base`。
- `report_peers.parquet`：三个窗口的比较组同行明细（列 `h` 区分窗口）。
- `pair_accuracy.parquet`：每个（公司，券商）一行，每个窗口 h 一组列：券商 2024 年内最后一条第 h 年预测的 PMAFE、时效差和该预测发布日上的控制变量。

### study forecast_accuracy：回归和文档的表、图

样本范围是 `data/hs300_2024_dag_covered.csv` 里 241 家公司、图里有边的研报，共 791 篇；基础样本 786 篇，公司-券商层面 760 对。
券商分组、行业、缩尾和回归设定见 `sample.py`、`regressions.py`。

被解释变量取自外部的预测误差文件 `data/forecast_errors.parquet`（读取和整理见 `errors.py`；只用公司代码与研报公司相同的行）：主变量 `PMED180_NP_H2` 是净利润口径、第 2 个预测年度的比例误差 (|A − F| − M180) / M180，M180 是文件给出的 180 天同行窗口内同行绝对误差的中位数（这一比例口径是用文件的中位数派生的，文件没有直接给出；形式上最接近自算的 PMAFE）；主变量的定义名只在 `sample.py` 的 `MAIN_DEF` 一处定义。其余定义（75 天窗口、相对误差 `relative_fe`、相对一致预期的增量误差、原始误差）× 净利润 / EPS 两种口径 × 三个窗口共 54 列，用于稳健性。`accuracy` 步骤自算的 PMAFE、Score 留在样本里作对照（`own_measures` 块），同行时滞仍用自算 90 天比较组的 `PEERLAG_Hh`（数据文件没有同行预测的时间）。公司-券商层面的被解释变量是被评价预测所在研报在数据文件里算出的同一主变量。

分成几个环节，用 `--part` 选（逗号分隔，按下表顺序执行；不给时做前六个，`pdf` 要显式指定）。每个环节只读前面环节存在输出目录里的成品，缺了会报错并提示先跑哪个环节；csv 读回时数值逐位不变，所以分开跑和一次跑完结果相同。

| 环节 | 做什么 | 写（`output/studies/forecast_accuracy/` 内，除注明外） |
|---|---|---|
| `sample` | 选样本、并入数据文件的预测误差、缩尾，图 2 的同行明细 | `scope.parquet`（791 篇研究范围）、`sample_report.parquet`、`sample_pair.parquet`、`forecast_errors_rows.parquet`（数据文件里用到的行）、`figure2_*.parquet` |
| `describe` | 预测误差变量和逻辑结构变量的描述性统计（表 1、表 3），与自算 PMAFE 的一致性（表 2）和中位数核对 | `error_stats.csv`、`descriptive_stats.csv`、`tables/table1_error_descriptive.csv`、`tables/consistency.csv`、`tables/consistency_fields.csv`、`tables/table3_descriptive.csv` |
| `decomp` | 固定效应分解（表 4） | `variance_decomposition.csv`、`tables/table4_fe_decomposition.csv` |
| `regress` | 全部回归；`--blocks` 只重估其中几块 | `reg_all.csv` |
| `tables` | 回归表（表 5–13）和正文数字 | `tables/table5_….csv` 到 `table13_….csv`、`tables/text_numbers.csv` |
| `figures` | 图 1、图 2 | `docs/figures/`（`--fig-dir` 可改） |
| `pdf` | 文档 md 转 PDF（`src/mdpdf.py`，不联网；公式转成 MathML 由浏览器排版） | 中间文件 `<md 文件名>.html`；md 旁边同名 `.pdf` |

`--log1p` 把六个逻辑结构变量都取 ln(1 + x)，只在 `sample` 环节改变变换，输出默认在 `output/studies/forecast_accuracy_log1p/`，单独跑其他环节时也要带上 `--log1p`（或同一个 `--output`）。

主要输出：

- `sample_report.parquet`、`sample_pair.parquet`：两个层面的回归样本。
- `reg_all.csv`：全部回归系数的长表。
- `tables/` 下：文档的表 1–13（表 2 是 `consistency.csv` 的 H2 各行，字段层面的一致性和中位数核对在 `consistency_fields.csv`）；`tables/text_numbers.csv`：正文、图注和第 4 节出现的每个数字及其算法。
- `docs/figures/` 下的两张图；`docs/研报逻辑结构与分析师预测准确度.pdf`。

文档的每张表、每张图、每一节对应哪个输出文件、由哪个函数生成，列在 `src/studies/forecast_accuracy/__init__.py` 开头。

## 数据

`data/` 不入 git，需要这些文件：

| 路径 | 内容 | 谁用 |
|---|---|---|
| `HS300_reports_2024/` | 研报纯文本，`MM/DD/<GUID>.txt` | text、graph |
| `EST_COMP_TN/<年>/<年>-<月>.parquet` | 分析师盈利预测 | accuracy；study（中位数核对） |
| `forecast_errors.parquet` | 外部提供的预测误差（每篇研报 3 个预测年度 × 净利润 / EPS；同行窗口的具体取法文件未附说明） | study（被解释变量） |
| `market_data/daily_stock_data.parquet` | 个股日行情 | accuracy |
| `market_data/financial_data.parquet` | 财务数据（年报实际值、公告日） | accuracy |
| `market_data/industry.xlsx` | 行业分类（申万一级） | study |
| `hs300_2024_dag_covered.csv` | 样本公司名单（241 家） | study |

## 历史

2026-11-23 精简过一次项目结构。之前的全部代码和文档在标签 `pre-cleanup` 里，包括已移出的几条研究线：五层估值逻辑图和研报分歧度、逻辑结构与 2025 年股价和财务指标的回归、研报发布后的市场反应、预测准确度回归的早期版本。它们的文档在 `docs/archive/`。
