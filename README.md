# llm_projcet

用大语言模型从卖方研报中抽取逻辑结构（概念和概念之间的因果关系构成的有向无环图），计算图结构变量，并检验它和分析师盈利预测准确度的关系。
研究结果见 `output/studies/forecast_accuracy/hs300_deep_v1/研报逻辑结构与分析师预测准确度.md`。样本是 2024 年沪深300 成分股的研报。

## 两个概念：抽取版本和样本

- **抽取版本（version）**：一套抽取规则 = 抽取代码 + 提示词 + 影响结果的规则参数（每块字数、`ARCHIVE_VERSION`），代码在 `src/graph/<版本>/`。每个版本一个**库** `output/graph/<版本>/`，同一版本下每篇研报的逻辑结构只存一份；补充抽取、重新抽取都是往同一个库里加或改，不新建目录。只有明确说“这是新版本”时才开新版本（把代码、提示词复制成新的一套，数据写到新的库）；平时的修修补补直接改当前版本的代码。版本登记在 `src/graph/__init__.py` 的 `VERSIONS`。
- **样本（sample）**：分析用哪些研报。样本不存逻辑结构，只是一张“研报 ID → 取自哪个版本的库”的清单 `src/samples/<样本>.csv`（两列 `report_id`、`version`，入 git），登记在 `src/samples/__init__.py` 的 `SAMPLES`（中文说明、来历、版本顺序、研究范围是否另限公司名单）。

现有的版本：

| 版本 | 规则 | 库 | 状态 |
|---|---|---|---|
| `v1` | 现行的代码和提示词（`src/graph/v1/`），每块 5000 字 | `output/graph/v1/`：17956 篇（重构前 `deep` 运行的 975 篇和 4 篇石头科技的短篇深度报告，加上 2026-11-25 补抽的 `hs300_full` 其余研报） | 默认版本，可以继续往里抽 |
| `v0` | 早期试跑：每块 2500 字、重构前的代码（在 git 标签 `pre-cleanup` 里），现在已不能完全复现 | `output/graph/v0/`：65 篇（重构前的 `pilot` 运行，7 家公司） | 只有数据、没有代码，只读 |

现有的样本：

| 样本 | 说明 | 篇数 | 研究范围 |
|---|---|---|---|
| `hs300_deep` | 沪深300 深度研报，现有研究用的研报。重构前分析用的是 `deep`（全部公司的深度报告：首页有深度、首次覆盖、公司专题等类型标注，且正文超过 5000 字）和 `pilot` 两次运行的合并，同一篇取 `deep` 的；2026-11-25 从当时的输出冻结，不能重建 | 1019（v1 975、v0 44） | 另限 `data/hs300_2024_dag_covered.csv` 的 241 家公司，再要求图里有边：791 篇（v1 766、v0 25） |
| `hs300_full` | 沪深300 成分股的全部个股研报：`index_stock.parquet` 里 `000300` 在 20231229、20240628、20241231 三期的并集（328 家）；元数据表里能识别公司的个股研报，代码在成分股里 | 17748（326 家公司，全部取自 v1；2026-11-25 已全部抽完） | 不另限公司 |

读成品时（`src/graph/load.py`、`src/measure/load.py`）清单决定每篇取自哪个版本，各版本按样本登记的版本顺序上下拼接（`hs300_deep`：v1 在前、v0 在后），版本内保持库里文件的行序——回归对求和顺序敏感，行序不能变。研究和读取函数还可以另给 `--version`：给的是样本的主版本（登记的第一个版本）时照清单取；给的是别的版本时，样本的全部研报都从那个版本的库里取（开新版本后比较新旧版本用）。

### 常见操作

| 要做什么 | 命令 |
|---|---|
| 看有哪些样本、某个样本抽了多少 | `python run.py sample`；`python run.py sample hs300_full` |
| **补充抽取**一个样本（只抽缺存档的，已有的原样保留） | `python run.py graph --sample hs300_full --dry-run` 先看篇数和块数，再去掉 `--dry-run`；可以先 `--limit 20` 试 20 篇 |
| **重新抽取**某几篇（如改了段落表里的文字） | `python run.py graph --report-ids <ID1>,<ID2>`（存档指纹对不上的会重抽，结果并入同一个库） |
| 只用存档重写库的输出 | `python run.py graph --archived-only`（不给 `--sample`、`--report-ids` 时本来就不抽取） |
| **开新版本** | ① `cp -r src/graph/v1 src/graph/v2`，改 `v2/` 里的代码、提示词或 `__init__.py` 的规则参数（包内只用相对导入，不用改别处）；② 在 `src/graph/__init__.py` 的 `VERSIONS` 里登记 `'v2': dict(package='src.graph.v2', readonly=False, desc=...)`；③ `python run.py graph --version v2 --sample hs300_full`（版本不是样本的主版本时，样本的全部研报都抽进 `output/graph/v2/`）；④ `python run.py measure --version v2`，`python run.py study forecast_accuracy --sample hs300_full --version v2`（输出在 `hs300_full_v2/`）。要让 v2 成为默认，改 `src/config.py` 的 `DEFAULT_VERSION` |
| **建新样本** | 在 `src/samples/__init__.py` 的 `SAMPLES` 里登记一项（英文名、中文说明、来历、版本顺序、研究范围是否另限公司名单、生成清单的 `build` 函数），`python run.py sample <名字> --build` 生成清单；之后 `graph --sample <名字>` 补抽、`accuracy --sample <名字>`、`study ... --sample <名字>` |
| 把旧版本标为只读 | `VERSIONS` 里设 `readonly=True`：`graph`、`unify` 会拒绝往里写，`measure` 和读取不受影响 |

`--fresh` 会清空一个版本的库（全部存档，所有用到它的样本都要重抽），必须同时给 `--yes`，否则只报错说明后果。

## 目录

```
run.py                      唯一入口：python run.py <步骤> [参数]
src/
  config.py                 目录约定、跨步骤共用的数据文件路径、默认版本和样本、各类输出目录的统一函数
  llm.py                    大模型调用：接口、并发、调用缓存、不合格时重试
  ols.py                    固定效应 OLS、聚类标准误、缩尾（各项研究共用）
  mdpdf.py                  文档 md → HTML → 本机 Chrome / Edge 无头打印成 PDF（各项研究共用）
  text/                     研报纯文本 → 段落表、元数据表
    clean.py  meta.py  load.py
  samples/                  样本：登记表、读清单、按规则建清单、进度统计（python run.py sample）
    __init__.py  hs300_deep.csv  hs300_full.csv
  graph/                    段落 → 每篇研报的概念和因果边
    __init__.py             步骤入口：选版本、选样本，分发到版本包；版本登记表 VERSIONS
    unify_cmd.py            unify 步骤的入口：选版本，分发到版本包的 unify.py
    store.py                各版本共用：库目录的布局（DagOut）、库的全集（in_library）
    load.py                 按样本从各版本的库里读成品（下游只通过它读）
    v1/                     抽取版本 v1 的整套：__init__.py（规则参数和说明）、extract.py、unify.py、common.py、
                            prompts/、viewer.html、unify_review.html
  measure/                  一个版本的库 → 六个逻辑结构变量
    paths.py  variables.py  load.py
  accuracy/                 样本的研报 + 预测表 → 预测准确度、同行时滞、控制变量
    forecasts.py  peers.py  controls.py  load.py
  studies/
    forecast_accuracy/      研报逻辑结构与分析师预测准确度
      sample.py  errors.py  regressions.py  tables.py  figures.py
literature/                 参考文献，按主题分目录：文献梳理和笔记入 git，PDF 不入 git
data/                       输入数据（不入 git）
output/                     各步骤的结果，见下面的“输出目录”
.env                        API_KEY、BASE_URL、MODEL 等，说明见 src/llm.py 开头；不入 git
```

### 输出目录

```
output/
  graph/v1/                                   抽取库（与样本无关）：概念DAG.html、data/、cache/
  graph/v0/                                   早期试跑的库（只读）
  measure/v1/report_measures.parquet          逻辑结构变量（与样本无关，库里每篇一行）
  measure/v0/report_measures.parquet
  accuracy/hs300_deep/                        预测准确度（只和研报范围有关，与抽取版本无关）+ manifest.json
  accuracy/hs300_full/                        同上，hs300_full 的全部研报（不按“投资评级”限定，各研究范围共用）
  studies/forecast_accuracy/hs300_deep_v1/        回归输出 + 文档（md、pdf、figures/）+ manifest.json
  studies/forecast_accuracy/hs300_deep_v1_log1p/  对数变量版，同样带自己的文档和 figures/
  studies/forecast_accuracy/hs300_full_v1_rating/ hs300_full 里“投资评级”是严格汇点的研报（--rating-only），带自己的文档和 figures/
  studies/forecast_accuracy/hs300_full_v1_revenue/ hs300_full 里至少有一个营收类汇点的研报（--target revenue --target-only），带自己的文档和 figures/
  studies/forecast_accuracy/hs300_full_v1_wg/ hs300_full 里图非空的全部研报，只放全文逻辑结构三个变量（--target none），带自己的文档和 figures/
  experiments/graph_stability/                同一批研报重复抽取的结果，用来衡量抽取噪声
  archive/docs/                               已不维护的旧文档
```

研究输出目录名是 `<样本>_<版本>`（样本里个别研报取自别的版本时以清单为准，目录名用主版本），`--target-only` 时加 `_<目标>`（`--rating-only` 即 `_rating`，`--target revenue --target-only` 是 `_revenue`），目标不是 rating 又没给 `--target-only` 时加 `_<目标>_all`，`--target none` 时加 `_wg`，`--log1p` 时再加 `_log1p`。每个研究输出目录和 accuracy 的样本目录里有 `manifest.json`：样本、版本、是否 log1p（给了 `--rating-only` 时另记 `rating_only: true`，目标不是 rating 时另记 `target`、`target_only`）、研报篇数、各版本篇数、生成时间。
**git 只跟踪 `output/` 下的研究文档**（`output/studies/` 下的 md、pdf、`figures/` 里的图）和 `output/archive/` 下的全部；parquet、csv、cache 等其余结果不入 git（见 `.gitignore`）。

### 旧路径到新路径（2026-11-25）

重构前的状态在 git 标签 `pre-layout` 里。

| 旧 | 新 |
|---|---|
| `output/graph/deep/` | `output/graph/v1/` |
| `output/graph/pilot/` | `output/graph/v0/` |
| `output/graph/_stability/` | `output/experiments/graph_stability/` |
| `output/measure/report_measures.parquet`（两次运行合并） | `output/measure/v1/`、`v0/` 各一张；按样本读用 `src/measure/load.py` |
| `output/accuracy/*.parquet` | `output/accuracy/hs300_deep/` |
| `output/studies/forecast_accuracy/` | `output/studies/forecast_accuracy/hs300_deep_v1/` |
| `output/studies/forecast_accuracy_log1p/` | `output/studies/forecast_accuracy/hs300_deep_v1_log1p/` |
| `docs/研报逻辑结构与分析师预测准确度.md`、`.pdf`、`docs/figures/` | `output/studies/forecast_accuracy/hs300_deep_v1/` |
| `docs/研报逻辑结构与分析师预测准确度_对数变量.md` | `output/studies/forecast_accuracy/hs300_deep_v1_log1p/` |
| `docs/archive/` | `output/archive/docs/` |
| `src/graph/extract.py`、`unify.py`、`common.py`、`prompts/`、`*.html` | `src/graph/v1/`（`DagOut` 移到 `src/graph/store.py`） |
| `src/graph/__init__.py` 的 `RUNS`、`--run`、`--companies`、`--deep-only`、`--min-deep-chars`、`--min-per-quarter`、`--all` | 删掉：选哪些研报 → 样本；每块字数 → 版本的规则参数；并发数 → 命令行默认值 |
| `read_table(name, runs=…)` 的 `run` 列（`deep`、`pilot`） | `read_table(name, sample=…, version=…)` 的 `version` 列（`v1`、`v0`） |

## 运行

在仓库根目录运行，使用 `C:\Users\hung\projects\.venv` 里的 Python（系统 PATH 上的 Python 没有 pandas 等依赖）。

| 命令 | 做什么 | 读 | 写 | 调用大模型 | 耗时 |
|---|---|---|---|---|---|
| `python run.py text` | 文本清理 | `data/HS300_reports_2024/` | `data/` 的段落表、元数据表 | 否 | — |
| `python run.py sample [名字] [--build]` | 列出样本、看抽取进度、按规则重建清单 | `src/samples/`、各版本的库 | `src/samples/<名字>.csv`（`--build`） | 否 | 几秒 |
| `python run.py graph [--version v1] [--sample 名字 \| --report-ids …]` | 逻辑结构抽取 | 段落表、元数据表、样本清单 | `output/graph/<版本>/` | 是 | 取决于篇数 |
| `python run.py unify [--version v1]` | 跨研报统一概念名称（可选） | `output/graph/<版本>/` | 同一目录 | 是 | — |
| `python run.py measure [--version v1]` | 逻辑结构变量 | `output/graph/<版本>/` | `output/measure/<版本>/` | 否 | 3 秒 |
| `python run.py accuracy [--sample hs300_deep]` | 预测准确度和控制变量 | 样本的研报表、`data/` 的预测表、行情、财务 | `output/accuracy/<样本>/` | 否 | 20 秒 |
| `python run.py study forecast_accuracy [--sample hs300_deep] [--version v1] [--target rating\|revenue\|none] [--target-only \| --rating-only] [--log1p]` | 样本、描述性统计、方差分解、回归、文档的表和图（分环节，见下文） | graph、measure、accuracy 的成品、`data/`（含 `forecast_errors.parquet`） | `output/studies/forecast_accuracy/<样本>_<版本>[_<目标>[_all] 或 _wg][_log1p]/` | 否 | 1.5 分钟（hs300_full --rating-only 约 8 分钟，其中回归 5 分钟） |
| `python run.py study forecast_accuracy --part pdf` | 文档 md 转 PDF | 输出目录里的 md 和 `figures/` | md 旁边同名 `.pdf` | 否 | 几秒 |
| `python run.py study concept_coverage [--sample hs300_deep] [--concept 投资评级 --tag rating]` | 检查工具：多少篇研报抽到了某个概念、它在图里的位置（见下文） | graph 的成品 | `output/studies/concept_coverage/<样本>_<版本>/<tag>/` | 否 | 几秒 |
| `python run.py study structure_distribution [--sample hs300_deep] [--target rating\|revenue] [--target-only \| --rating-only]` | 检查工具：六个逻辑结构变量（原始值）的分布（见下文） | measure、graph 的成品 | `output/studies/structure_distribution/<样本>_<版本>[_<目标>[_all]]/` | 否 | 几秒 |

每个步骤的选项用 `python run.py <步骤> --help` 查看，各步骤的口径写在 `src/<步骤>/` 各文件开头。
只想重现文档里的数字时，依次运行 `measure`（`--version v1`、`--version v0` 各一次）、`accuracy`、`study` 即可，不需要重新清理文本和抽取。

### 只跑某一个环节

| 命令 | 做什么 |
|---|---|
| `python run.py text --limit 100` | 只抽 100 篇试跑，写到 `*_sample100.parquet`，不覆盖全量结果 |
| `python run.py graph --sample hs300_full --dry-run` | 统计要处理的研报、已有存档和待抽的篇数、块数，不调用大模型 |
| `python run.py graph --report-ids <研报ID>` | 只处理这几篇（逗号分隔），结果并入同一个库，其余研报只读存档 |
| `python run.py graph --sample hs300_full --limit 20` | 在要处理的研报里按种子随机抽 20 篇（`--seed` 换种子，同一个种子下嵌套），抽完照常存档、并入库 |
| `python run.py graph --report-ids <ID> --chunk-chars 3000` | 换每块字数的试跑：其余研报的存档会对不上，所以只在日志里报告，不存档、不写输出 |
| `python run.py graph --archived-only` | 只用已有存档重写库的输出，不调用大模型 |
| `python run.py unify --dry-run` | 只统计候选组，不调用大模型 |
| `python run.py unify --companies 600519.SH,002415.SZ` | 只统一这几家公司的研报（结果表会改写成只有这几家） |
| `python run.py study forecast_accuracy --part describe,tables` | 只做列出的环节（sample、describe、decomp、regress、tables、figures、pdf），各环节只读前面环节存在输出目录里的成品 |
| `python run.py study forecast_accuracy --part regress --blocks paired,firm_broker:paired` | 只重估这几个回归块，`reg_all.csv` 里其余行不动 |
| `python run.py study forecast_accuracy --part pdf [--doc <md 路径>]` | 文档转 PDF，默认输出目录里的主文档（`--log1p` 时是对数变量版） |
| `python run.py study forecast_accuracy --sample hs300_full --rating-only --part tables` | `--rating-only` 范围的输出目录（`hs300_full_v1_rating/`）里只重出表和正文数字；单独跑环节时也要带上 `--rating-only` |

## 约定

- **步骤名只定义一次**（`run.py` 的 `STEPS`）：命令名一般与 `src/` 下的目录名、`output/` 下的目录名相同（`unify` 写在 graph 的库里；`sample` 的代码和清单在 `src/samples/`，没有输出目录）。
- **每个步骤的接口相同**：步骤模块提供 `add_args(parser)` 和 `run(args)`，默认运行参数写在本目录里；`load.py` 负责读本步骤的成品，下游通过它读，不自己拼路径。`run.py` 只做分发，选中子命令后才导入对应模块。输出目录统一由 `src/config.py` 的函数给出（`graph_dir`、`measure_dir`、`accuracy_dir`、`study_dir`）。
- **每一步只读 `data/` 和前面步骤在 `output/` 下的成品。**
- **变量层只出原始值**：`measure` 对一个库的全部研报、`accuracy` 对一个样本的全部研报计算，不缩尾、不标准化、不选样本。选样本、券商分组、缩尾、回归都在 `src/studies/<研究>/` 里做。
- **文件名只写内容**，不带前缀和版本号；抽取版本和样本体现在目录名上（`output/graph/<版本>/`、`output/accuracy/<样本>/`、`<样本>_<版本>/`）。

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

### sample：样本

`python run.py sample` 列出样本；`python run.py sample <名字>` 打印篇数、各版本各多少篇、库里已有结果多少篇、还缺多少篇（准确的待抽篇数和块数用 `graph --sample <名字> --dry-run`）；`python run.py sample <名字> --build` 按登记的规则重建清单（`hs300_deep` 是冻结的，不能重建）。样本的说明见上面“两个概念”和 `src/samples/__init__.py` 开头。

### graph：逻辑结构抽取

v1 的流程在 `src/graph/v1/extract.py`，提示词在 `src/graph/v1/prompts/extract.md`。不预设任何概念，也不做补全，每篇研报做三件事：

1. **合并段落**：正文段落按顺序合并成不超过 `chunk_chars` 字的块（v1 是 5000），后面每块抽概念、连边各调用一次大模型。
2. **抽概念**：按块顺序通读，滚动维护本篇概念表（规范名、别名、定义、类型：指标 / 事件 / 状态）；每个新概念要给出原文叫法，程序逐字核对，对不上的不收。
3. **连边**：各块并行，在原文里找概念之间“A 影响 B”的关系（只要因果，不要并列、包含、同义），引文逐字核对。各块的边按（A, B）合并，有环时每次删掉环上引文最少、在原文中出现得最晚的一条，得到有向无环图；没有连上边的概念不进图，仍留在概念表里。

**库和处理范围**：库的全集是全部能识别公司的个股研报（`src/graph/store.py` 的 `in_library`：个股公司研报，股票代码、公司名、券商都识别出来了，共 18457 篇）。一次运行只处理 `--sample` 给的样本里取自该版本的研报，或 `--report-ids` 给的几篇：缺有效存档的才调用大模型，其余只读存档；写输出时写库里**全部**有有效存档的研报。所以补抽 `hs300_full` 时 `hs300_deep` 的 975 篇原样保留、不重复存。不给 `--sample` 也不给 `--report-ids` 时不抽取，只用存档重写输出。

每篇研报做完即存档，中断后重新运行接着跑；存档指纹 = `ARCHIVE_VERSION` + 每块字数 + 提示词全文 + 研报的公司、标题、段落文字。改了提示词或规则参数会让库里的存档全部失效——这种改动应当开新版本，不要直接改 `src/graph/v1/prompts/` 下的文件。库的现有结果里有研报的存档按这次的参数对不上时（如 `--chunk-chars` 换了），自动改成试跑：只抽取、在日志里报告，不存档、不写输出。

一个库的输出（`output/graph/<版本>/`）：

- `概念DAG.html`：查看页面，双击打开，左侧选研报；点节点突出它的上下游，点边看原文引文和所在段落。
- `data/concepts.parquet`（每篇的每个概念一行，`in_dag` 是否进图）、`edges.parquet`（每条边一行，`in_dag` 为假的是因成环删掉的边，`support` 是引文处数）、`reports.parquet`（每篇的概念数、边数等）。
- `cache/`：调用缓存、各篇存档、出错记录、进度日志。

下游用 `src/graph/load.py` 按样本读（`read_table`、`load_graphs`、`owners`，默认样本 `hs300_deep`），结果多一列 `version`。

### unify：跨研报统一概念名称（可选）

同一个量在不同研报里常有不同叫法（营收 / 营业收入）。`unify` 对一个版本的库按公司分开统一，只合并明显完全相同的名称，范围、口径、主体不同的不合并；做法和复核规则见 `src/graph/v1/unify.py`，提示词在 `src/graph/v1/prompts/unify.md`。
**现在的回归不使用统一后的名称**（逻辑结构变量按每篇研报自己的图计算）；保留它是为了以后做跨研报的比较。
v1 库的名称统一没有做完：合并提议 7036 组已做完，复核做到 3270 / 5160 组时停下，已完成的都在缓存里；`concept_unify.parquet` 还是文本清理修复之前 939 篇的版本。v0 只统一了比亚迪的 16 篇。

### measure：逻辑结构变量

对一个版本的库里的全部研报各算一行（v1 17956 篇——2026-11-25 抽完 hs300_full 后重算，原有 979 篇的值逐位不变；v0 65 篇），写 `output/measure/<版本>/report_measures.parquet`；按样本读用 `src/measure/load.py` 的 `read_measures`（默认 `hs300_deep` 的 1019 篇）。口径见 `src/measure/variables.py`。

- 源点：没有入边的概念；严格汇点：有入边、没有出边的概念；路径：从源点到汇点的有向路径。
- 每套变量三个：源点数、ln(1 + 路径数)、平均路径长度。
  - 全文逻辑结构（`WG_*`）：整张图。
  - “目标”对应的逻辑结构：目标节点集合 T 及其全部上游，T 为空时缺失。目标登记在 `src/measure/variables.py` 的 `TARGETS`（列名前缀、文档符号、中文名、取 T 的函数），以后加目标只加一项、在表的末尾追加列：
    - `rating`（`RT_*`，RateSrc 等）：T = {“投资评级”}，只在它是严格汇点时有定义（标记列 `HAS_RATING_NODE`、`RATING_STRICT_SINK`）。
    - `revenue`（`RV_*`，RevSrc 等；2026-11-25 追加在原有列之后，原有列逐值不变）：T = 名称匹配 `收入|营收`、不匹配 `成本收入比|可支配收入|居民收入|财政收入` 的严格汇点（有出边的营收类节点不算），另有 `RV_N_PATHS`、`N_REVENUE_SINKS`（个数，没有时为 0）、`HAS_REVENUE_SINK`。hs300_full 里 9037 篇（50.9%）有营收类汇点。

### accuracy：预测准确度和控制变量

对一个样本的全部研报计算（只用研报表的代码、日期、券商，与抽取版本无关；样本里还没有抽取结果的研报不在内），写 `output/accuracy/<样本>/` 下三张表和 `manifest.json`，各列说明见 `src/accuracy/load.py`，口径见 `forecasts.py`、`peers.py`、`controls.py`。

预测误差按预测窗口定义：H1、H2、H3 是预测行的第 1、2、3 年预测（`PROFIT_T1`–`PROFIT_T3`），目标年度 = 预测行的 `RPT_YR_T` + h，逐篇不同（`RPT_YR_T` = 2022 的研报 H2 是 FY2024，= 2023 的是 FY2025）。窗口相关的列都带后缀 `_H1`、`_H2`、`_H3`；比较组是其他券商对同一公司、同一目标年度的预测。研究层的主窗口由 `src/studies/forecast_accuracy/sample.py` 的 `MAIN_H`（现为 2）决定。

- `report_accuracy.parquet`：每篇研报一行。本篇预测（`data/EST_COMP_TN` 里 `RPT_ID` 等于 `report_id` 大写、`SEC_CD` 等于本公司的那一行）、各窗口的实际值、相对同行的预测误差 PMAFE、排名分数 Score、同行数和同行时滞，六个控制变量、基础样本标记 `base`。
- `report_peers.parquet`：三个窗口的比较组同行明细（列 `h` 区分窗口）。
- `pair_accuracy.parquet`：每个（公司，券商）一行，每个窗口 h 一组列：券商 2024 年内最后一条第 h 年预测的 PMAFE、时效差和该预测发布日上的控制变量。

### study forecast_accuracy：回归和文档的表、图

`--sample`（默认 `hs300_deep`）、`--version`（默认样本的主版本）决定研究哪些研报、逻辑结构取自哪个库，输出在 `output/studies/forecast_accuracy/<样本>_<版本>/`。`hs300_deep` 的研究范围是 `data/hs300_2024_dag_covered.csv` 里 241 家公司、图里有边的研报，共 791 篇；基础样本 786 篇，公司-券商层面 760 对。
券商分组、行业、缩尾和回归设定见 `sample.py`、`regressions.py`。

被解释变量取自外部的预测误差文件 `data/forecast_errors.parquet`（读取和整理见 `errors.py`；只用公司代码与研报公司相同的行）：主变量 `PMED180_NP_H2` 是净利润口径、第 2 个预测年度的比例误差 (|A − F| − M180) / M180，M180 是文件给出的 180 天同行窗口内同行绝对误差的中位数（这一比例口径是用文件的中位数派生的，文件没有直接给出；形式上最接近自算的 PMAFE）；主变量的定义名只在 `sample.py` 的 `MAIN_DEF` 一处定义。其余定义（75 天窗口、相对误差 `relative_fe`、相对一致预期的增量误差、原始误差）× 净利润 / EPS 两种口径 × 三个窗口共 54 列，用于稳健性。`accuracy` 步骤自算的 PMAFE、Score 留在样本里作对照（`own_measures` 块），同行时滞仍用自算 90 天比较组的 `PEERLAG_Hh`（数据文件没有同行预测的时间）。公司-券商层面的被解释变量是被评价预测所在研报在数据文件里算出的同一主变量。

分成几个环节，用 `--part` 选（逗号分隔，按下表顺序执行；不给时做前六个，`pdf` 要显式指定）。每个环节只读前面环节存在输出目录里的成品，缺了会报错并提示先跑哪个环节；csv 读回时数值逐位不变，所以分开跑和一次跑完结果相同。

| 环节 | 做什么 | 写（输出目录内） |
|---|---|---|
| `sample` | 选样本、并入数据文件的预测误差、缩尾，图 2 的同行明细 | `scope.parquet`（研究范围）、`sample_report.parquet`、`sample_pair.parquet`、`forecast_errors_rows.parquet`（数据文件里用到的行）、`figure2_*.parquet`、`manifest.json` |
| `describe` | 预测误差变量和逻辑结构变量的描述性统计（表 1、表 3），与自算 PMAFE 的一致性（表 2）和中位数核对 | `error_stats.csv`、`descriptive_stats.csv`、`tables/table1_error_descriptive.csv`、`tables/consistency.csv`、`tables/consistency_fields.csv`、`tables/table3_descriptive.csv` |
| `decomp` | 固定效应分解（表 4） | `variance_decomposition.csv`、`tables/table4_fe_decomposition.csv` |
| `regress` | 全部回归；`--blocks` 只重估其中几块 | `reg_all.csv` |
| `tables` | 回归表（表 5–13）和正文数字 | `tables/table5_….csv` 到 `table13_….csv`、`tables/text_numbers.csv` |
| `figures` | 图 1、图 2 | `figures/`（`--fig-dir` 可改） |
| `pdf` | 文档 md 转 PDF（`src/mdpdf.py`，不联网；公式转成 MathML 由浏览器排版） | 中间文件 `<md 文件名>.html`；md 旁边同名 `.pdf` |

`--log1p` 把六个逻辑结构变量都取 ln(1 + x)，只在 `sample` 环节改变变换，输出在 `…/hs300_deep_v1_log1p/`，单独跑其他环节时也要带上 `--log1p`（或同一个 `--output`）。

`--rating-only` 把研究范围再限定在“投资评级”是严格汇点（入度 > 0、出度 = 0，measure 的 `RATING_STRICT_SINK`）的研报，`RT_*` 三个变量在范围内全部有定义；之后的基础样本、券商分组、缩尾、标准化、公司-券商层面的样本对都在这个范围内照原规则做。它同样只在 `sample` 环节起作用，输出目录名加 `_rating`（和 `--log1p` 同时给时是 `_rating_log1p`），单独跑其他环节时也要带上。不给时行为和以前完全相同。现有的一份是 `python run.py study forecast_accuracy --sample hs300_full --rating-only`：hs300_full 里 8468 篇（47.7%）的“投资评级”是严格汇点，另一半研报的“投资评级”没有连进图（原因未查），这份结果不能外推到全部研报。行业表里没有、也不在 `MANUAL_INDUSTRY` 里的公司（hs300_full 里只有 601059 信达证券）行业记缺失，只在含行业固定效应的回归和方差分解的“行业”一列里不用。

`--target {rating,revenue}`（默认 rating）决定“目标对应的逻辑结构”三个变量用 `RT_*` 还是 `RV_*`（全文的 `WG_*` 不变），列名、表头、正文数字的说明和图 1 的 (b) 面板都跟着换；`--target-only` 把范围限定在该目标有定义的研报，`--rating-only` 等于 `--target rating --target-only`。和 `--log1p` 一样只在 `sample` 环节起作用，单独跑其他环节时也要带上。不带这两个开关时输出与以前逐字节相同。现有的一份是 `python run.py study forecast_accuracy --sample hs300_full --target revenue --target-only`（输出在 `hs300_full_v1_revenue/`，另有 `tables/target_nodes.csv`：营收类汇点名称的频数）：营收类节点是汇点说明图里没有抽到它往下（到利润、估值）的边，可能是原文如此，也可能是漏抽，未核查，结论同样不能外推到全部研报。

`--target none` 不用任何目标对应的子图变量，逻辑结构变量只有全文的 `WG_*` 三个（AllSrc、AllPath、AllLen）；研究范围就是原规则（图非空、名单内），不做目标筛选，不能和 `--target-only`、`--rating-only` 同时给。输出目录加 `_wg`，manifest 记 `target: none`。需要子图变量的回归块（成对放入 `paired`、`firm_broker:paired`，`same_sample`）不估计，表 7、13 不输出，图 1 只画全文一个面板；其余照旧。现有的一份是 `python run.py study forecast_accuracy --sample hs300_full --target none`（输出在 `hs300_full_v1_wg/`，文档的表号：标准版表 8–12 依次成为表 7–11）。

主要输出：

- `sample_report.parquet`、`sample_pair.parquet`：两个层面的回归样本。
- `reg_all.csv`：全部回归系数的长表。
- `tables/` 下：文档的表 1–13（表 2 是 `consistency.csv` 的 H2 各行，字段层面的一致性和中位数核对在 `consistency_fields.csv`）；`tables/text_numbers.csv`：正文、图注和第 4 节出现的每个数字及其算法。
- `figures/` 下的两张图；文档 `研报逻辑结构与分析师预测准确度.md` 和 `.pdf`（对数变量版的目录里是 `…_对数变量.md`）。

文档的每张表、每张图、每一节对应哪个输出文件、由哪个函数生成，列在 `src/studies/forecast_accuracy/__init__.py` 开头。

### study concept_coverage：概念覆盖检查（独立工具）

统计一个样本里有多少篇研报的概念表里有某个概念（默认“投资评级”，完全同名）、它是否在图里、是否是严格汇点（口径同 `measure`），以及没有同名概念时的近似名称（`--contains`，默认“评级”）和别名；按正文字数、块数、券商分组。只读 graph 的成品，不调用大模型，不在主流程里，也不被其他步骤调用，需要时手动运行，每次覆盖自己的输出。例：`python run.py study concept_coverage --sample hs300_full`；换概念时要给输出子目录名，如 `--concept 营业收入 --tag revenue`。输出在 `output/studies/concept_coverage/<样本>_<版本>/<tag>/`（`summary.md`、`summary.csv`、`by_*.csv`、`variant_names.csv`、每篇一行的 `reports.parquet`、`manifest.json`），口径和各文件见 `src/studies/concept_coverage/__init__.py` 开头。

### study structure_distribution：逻辑结构变量的分布（独立工具）

统计回归用的六个逻辑结构变量（RT_NSRC、RT_LNPATH、RT_PATHLEN、WG_NSRC、WG_LNPATH、WG_PATHLEN）、两个路径条数和 `--log1p` 用的四个对数变换列的分布（分位数、偏度、峰度、取最小值的占比等），四个整数变量的频数表，按正文字数分组的均值和中位数，以及原始值和对数变换后的直方图，用来判断偏态和取对数是否合适。统计的是 measure 的原始值（不缩尾、不标准化），每个变量在它有定义的研报上统计；默认范围是样本里库中有结果的全部研报，`--rating-only` 只看“投资评级”是严格汇点的研报（不套公司名单，所以可能比 forecast_accuracy 同名开关的范围大）；`--target revenue` 把 RT 三个变量换成 RV 三个，`--target-only` 只看该目标有定义的研报（同 forecast_accuracy）。只读 measure 和 graph 的成品，不调用大模型，不在主流程里，也不被其他步骤调用，每次覆盖自己的输出。例：`python run.py study structure_distribution --sample hs300_full --rating-only`。输出在 `output/studies/structure_distribution/<样本>_<版本>[_<目标>[_all]]/`（`summary.md`、`summary.csv`、`frequency.csv`、`by_length.csv`、`histograms.png`、`histograms_log1p.png`、`manifest.json`），口径和各文件见 `src/studies/structure_distribution/__init__.py` 开头。

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
| `market_data/index_stock.parquet` | 指数成分股 | sample（`hs300_full` 的成分股） |
| `hs300_2024_dag_covered.csv` | `hs300_deep` 研究范围的公司名单（241 家） | study |

## 历史

- 2026-11-23 精简过一次项目结构。之前的全部代码和文档在标签 `pre-cleanup` 里，包括已移出的几条研究线：五层估值逻辑图和研报分歧度、逻辑结构与 2025 年股价和财务指标的回归、研报发布后的市场反应、预测准确度回归的早期版本。它们的文档在 `output/archive/docs/`。
- 2026-11-25 把“抽取版本”和“样本”分开管理，输出按这两个概念重排（对照表见上面“旧路径到新路径”），文档并入研究输出目录。重构前的状态在标签 `pre-layout` 里；重构前后下游结果的数值逐位相同。
