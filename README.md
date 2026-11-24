# llm_projcet

用大语言模型从卖方研报中抽取估值逻辑图，并度量同一家公司不同研报之间的逻辑框架差异。

```
clear_txt.py            第一步入口：研报纯文本清理（不调用大模型）
extract_logic_chain.py  第二步入口：逻辑链抽取；另可用 --task 运行框架对比、生成介绍页
sort_divergence.py      第三步入口：公司-季度分歧度 → 匹配股票数据 → 分组统计和回归（不调用大模型）
extract_concept_dag.py  另一套抽取的入口：概念有向无环图（先抽概念、再连“A 影响 B”的边，不预设概念、不补全），见下文
unify_concept_dag.py    概念 DAG 抽完后，跨研报统一同一个概念的不同叫法（营收 / 营业收入），见下文
src/                    代码（Python 包）
  config.py               目录约定（DATA_DIR、OUTPUT_DIR、PROMPT_DIR，输出目录的布局 OutDir）
  llm_client.py           大模型接口（兼容 OpenAI Chat Completions）、.env 读取、进度日志
  report_text.py          文本清理：研报纯文本 → 正文段落
  report_meta.py          研报元数据：发布日期、券商、报告类型、是否个股研报、公司和代码、分析师
  report_logic.py         逻辑链抽取的主流程
  logic/                  逻辑链抽取各环节的实现（base、extract、graph、complete、output）
  fin_items.py            预置的会计层清单
  report_compare.py       同一公司不同研报的逻辑框架对比
  concept_dag.py          概念有向无环图抽取（和 report_logic.py 互相独立，提示词 prompts/concept_dag_prompt.md）
  concept_unify.py        概念 DAG 的跨研报名称统一（提示词 prompts/concept_unify_prompt.md）
  divergence_sort.py      公司-季度分歧度、匹配行情和财务数据、分组统计和回归
  viz/                    页面：逻辑图查看页面（logic_viewer.html）、介绍页（plot_intro.py + intro_template.html）、
                          抽取进度页（plot_coverage.py + coverage_template.html）、概念 DAG 查看页面（dag_viewer.html）、
                          概念统一核对页面（unify_review.html）
  prompts/                大模型提示词
docs/                   设计文档、方法调研笔记、介绍页
data/                   研报纯文本与清理后的正文段落；market_data/ 下是股票日行情和季度财务数据（不入 git）
output/                 抽取结果（不入 git），布局见下
output_dag/             概念有向无环图的抽取结果（不入 git）
literature/             参考文献，按主题分目录：文献梳理和笔记入 git，PDF 不入 git
.env                    API_KEY、BASE_URL、MODEL 等，见 src/llm_client.py 的说明；不入 git
```

## 运行

在仓库根目录运行，使用 `C:\Users\hung\projects\.venv` 里的 Python（系统 PATH 上的 Python 没有 pandas 等依赖）。
两个脚本的输入、输出路径和试跑篇数都在各自开头的“运行参数”里，也可以用命令行参数临时覆盖。

```bash
python clear_txt.py              # 第一步：data/HS300_reports_2024/ 的 txt → 段落表 + 研报元数据表，多进程并行（默认用满 CPU 核心），带进度条
python extract_logic_chain.py    # 第二步：按元数据表选出个股公司研报 → output/ 下的逻辑图（调用大模型，已完成的研报和调用都跳过）
python extract_logic_chain.py --task compare   # 框架对比：每家公司把概念归成主题（调用大模型）
python sort_divergence.py        # 第三步：公司-季度分歧度 → 匹配股票数据 → 分组统计和回归（不调用大模型）
```

`extract_logic_chain.py` 每篇研报做完逐篇的几步（建表到剪枝）就存档到 `output/cache/reports/`，重跑时存档有效的研报直接读回、不再处理；
待处理的研报逐篇处理：每篇各自走完全部步骤，同时在处理的篇数等于并发数（脚本开头的 `WORKERS`，默认 150），做完一篇即存档并接上下一篇，全量运行中断后重新运行会接着跑。
存档带指纹（段落文字、公司和标题、提示词文件、`src/fin_items.py` 的预置清单、`src/report_logic.py` 的 `ARCHIVE_VERSION`），
其中任何一样变了，这篇就重新处理（调用缓存还在，没变的调用不花钱）；改了流程代码要让旧存档作废时，把 `ARCHIVE_VERSION` 加一。
某一篇出错时只有它不存档、不进输出，记到 `cache/failed_reports.jsonl`，下次运行自动重试；有大模型调用重试用完仍失败的
（各步已按自己的方式兜底），照常输出但不存档，下次运行重试这些调用；鉴权失败、余额不足这类错误会让整个运行停下。
跨研报的全局去重和输出每次对本次选中的、已完成的全部研报统一做一遍。

| 命令 | 功能 | 调用大模型 |
|---|---|---|
| `python clear_txt.py --limit 100` | 只跨月份随机抽 100 篇试跑，写到 `data/hs300_2024_paragraphs_sample100.parquet` | 否 |
| `python extract_logic_chain.py --limit 20` | 在选中的研报里按固定种子随机抽 20 篇试跑；同一个种子下 20 篇 ⊂ 100 篇 ⊂ 全部，放大样本时做完的直接沿用；`--seed 1` 换一组 | 是 |
| `python extract_logic_chain.py --companies 600519.SH,002415.SZ` | 只抽这几只股票的研报（默认是脚本开头 `COMPANIES` 里的三家；`--companies all` 不限公司） | 是 |
| `python extract_logic_chain.py --fresh` | 先清空输出目录里上一次的结果和各篇存档再跑（大模型调用缓存保留） | 是 |
| `python extract_logic_chain.py --all` | 不按元数据表筛选，段落表里的研报全部抽取（没有元数据表的自备段落、旧的 `data/report_paragraphs.parquet` 要加） | 是 |
| `python extract_logic_chain.py --input <段落 parquet> --output <目录>` | 换输入和输出目录（试跑时免得覆盖正式结果） | 是 |
| `python extract_logic_chain.py --task compare` | 框架对比 → `output/data/compare/` | 是（每家公司一次） |
| `python extract_logic_chain.py --task intro_page` | 生成 `docs/逻辑抽取与分歧度量介绍.html`（样例研报写死在 `src/viz/plot_intro.py`） | 否 |
| `python extract_logic_chain.py --task coverage` | 抽取进度页 `docs/研报概念数据整理进度.html`：核对各篇存档，按公司统计做完、没做完的篇数，按最近一次记了花费的运行粗估剩余花费（默认不限公司，`--companies` 只看这几家；抽取在后台运行时也能跑，约一两分钟） | 否 |

## 第三步：分歧度与股票数据

`sort_divergence.py` 不调用大模型，样本变大后直接重跑即可。它做五件事，结果都在 `output/data/divergence/`：

1. **公司-季度分歧度**（`company_quarter.parquet`）：同一家公司、同一个自然季度（按发布日期）里的研报一起比较，
   每篇研报的流量分布对比这些研报的平均分布，得到总分歧 = 主题项 + 关系项（除以 log₂(研报数) 归一到 0–1）。
   没有概念连到估值的研报不参与，不足 2 篇的公司-季度不算。
2. **匹配股票数据**（`panel.parquet`）：股票代码去掉交易所后缀后和 `data/market_data/` 对上，接上当季末总市值、
   下一个季度的股票回报（后复权收盘价，季末到下季末）和当季末已公告的最新一期财务数据。行情里没有的股票（港股等）会去掉。
3. **分组**：先按总分歧、主题项、关系项各自分三组（单变量分组，`sort_single.csv` 每个指标一行，`sort_single_detail.csv`
   是每组的指标均值、季末市值和个数）；再做双重分组（`sort_next_ret.csv`、`sort_mkcap.csv`、`sort_count.csv`）：每个季度内
   按主题项和关系项各分三组。每组（格子）是一个组合，统计它持有下一个季度的日均回报（带“高-低”和 t 值）、季末市值均值和样本数。
4. **逻辑链大小和深度**（`report_sizes.parquet`、`company_quarter_size.parquet`、`size_sort_deep.csv`、`size_sort_earn.csv`、
   `size_sort_detail.csv`）：每篇研报的逻辑图用九种算法量大小和深度，把深度报告和财报点评分开，在公司-季度内取平均，
   每个季度内按每种算法分五组，统计每组组合持有下一个季度的日均回报（带“高-低”和 t 值，算法同双重分组）、季末市值和样本数。
   - 大小：概念数、边数、原文边数（只算有原文证据的边）、链条数（从链条起点走到公司估值的不同路径条数，取对数）、每千字概念数（扣掉篇幅）。
   - 深度：最长链步数（从概念走到公司估值最多几步）、平均链长（全部链条的平均步数）、流量平均步数（每个概念发出一份流量、
     在分叉处平均分开，平均几步到估值）、平均层级（概念平均在第几层，越大越远离财务科目）。

   新数据没有标题，研报类型按正文判断：正文不少于 6000 字（写明“首次覆盖”的 3000 字）算深度报告；其余开头 400 字在讲一期财报的
   算财报点评；剩下的归为其他。这一部分一篇研报也算，不要求公司-季度有两篇以上。逻辑图默认直接读各篇研报的存档（`cache/reports/`），
   抽取还在跑、`data/graphs/` 还没更新时也能用上新做完的研报；分歧度那部分要用框架对比归的主题，仍然读 `data/graphs/`。
5. **回归**（`panel_reg.parquet`、`reg_returns.csv`、`reg_financial.csv`）：因变量分股票收益和财务特征两组，
   分别对总分歧、主题项、关系项各回归一次，再把主题项和关系项放进同一个回归。
   - 股票收益：下季收益、研报发布窗口 [-3,+1] 的累计超额收益（个股减全部 A 股市值加权回报，公司-季度内取平均）、
     季末的动量（前 12 到前 1 个月）和反转（前 1 个月），以及当季和下季的特质波动（市场模型残差）、NCSKEW 和 DUVOL
     （扩展市场模型的日残差，Kim, Li & Zhang 2011）。
   - 财务特征：按季末已公告的最新一期报表计算的 ln 市值、BM、EP、SP、杠杆、ROA、TTM 营收和利润同比增长；
     另加下一季度那期报表的 ROA 变化和同比增长。
   - 回归设定：两个模型，一个只有季度固定效应，一个再加控制变量（股票收益组：ln 市值、BM、ln 研报数；财务特征组：ln 市值、ln 研报数）。
     分歧度先标准化，系数是分歧度高 1 个标准差时因变量的变化。因变量和控制变量按 1%/99% 缩尾，标准误按公司聚类。
     只有 4 个季度，季度内的截面相关没法聚类，t 值可能偏高。

另有 `--task text`（`src/text_graph.py`）：逻辑图去掉大模型自己补全的逻辑边，只留原文边（extracted、bridged）和预置的会计关系
（和框架对比 `edge_set='text'` 同一口径），算每篇研报的图结构指标：概念数、原文边数、每个概念的原文边数、论点模块数（概念层按原文边
弱连通的分量个数）、最大模块占比、走不到估值的子图数（定义同抽取第 3 步）、连到估值的概念占比、会计入口数、原文链条数、原文最长链步数、
平均层级。分两层检验：公司-季度内取平均后，对上面第 5 步的股票收益和财务特征两组因变量回归（控制变量另加 ln 正文字数和深度报告占比），
并按每个指标分五组看下季组合回报；研报层面对发布窗口 [-3,+1] 的累计超额收益和发布后 [+2,+60] 的漂移回归。
结果在 `text_report.parquet`、`text_company_quarter.parquet`、`text_reg_*.csv`、`text_sort*.csv`。

严格原文逻辑图（`python -m src.strict_graph`，`src/strict_graph.py`，直接读存档，约一分钟）：只用抽取第 1 步从原文抽出的因果关系，
不加预置的会计关系、不加任何补全的边、不做概念归并，会计科目和估值只有原文关系里写到时才出现，连不到估值就是连不到。
每个节点另给一个原文字句作标签（变量名或别名在正文里原样出现，否则取引文里最接近的小句）。结果在 `<输出目录>/data/strict_graph/`
的 `strict_nodes.parquet`、`strict_edges.parquet`、`strict_reports.parquet`。

| 命令 | 功能 |
|---|---|
| `python sort_divergence.py --task size` | 只做逻辑链大小和深度的分组；`--size-var depth,depth_mean` 只用这几种算法，`--size-groups 3` 换分组数，`--size-source graphs` 改读 `data/graphs/` |
| `python sort_divergence.py --task divergence` | 只算公司-季度分歧度（要读全部逻辑图，约一分钟） |
| `python sort_divergence.py --task sort` | 分歧度表已经有了，只重做匹配和分组（几秒钟） |
| `python sort_divergence.py --task sort --single-groups 5` | 单变量分组改成五组；`--single rel` 只按关系项分，`--single ""` 不做单变量分组 |
| `python sort_divergence.py --task sort --groups 5 --how dependent` | 双重分组改成五组；条件分组（先按行指标分组，再在组内按列指标分组） |
| `python sort_divergence.py --task sort --weight value` | 组合按季末市值加权（默认等权）；`--nw-lags 10` 指定 Newey-West 滞后阶数 |
| `python sort_divergence.py --task sort --row pair_theme --col pair_rel` | 换成季度内两两平均口径的分歧度 |
| `python sort_divergence.py --task reg` | 分歧度表已经有了，只做回归（要读全部 A 股两年的日行情，约一分钟）；`--reg theme,rel` 换分歧指标，`--car-window=-1,1` 换研报发布窗口 |
| `python sort_divergence.py --task text` | 只用原文边的逻辑图：图结构指标对股票收益、财务特征的回归和分组，以及研报窗口收益的回归（直接读存档，约五分钟） |

回报按 Ang, Hodrick, Xing & Zhang (2006, *The Cross-Section of Volatility and Expected Returns*) 的组合做法算：每组（格子）
是一个组合，在季末建仓、持有下一个季度，下个季末按新的分组换仓；组合每天的回报是成分股当天回报（后复权收盘价）的平均，默认等权，
`--weight value` 按建仓时的季末市值加权。表里是持有期组合日回报的平均，单位是基点（0.01%）。“高-低”是最高组和最低组两个组合
每日回报之差的平均，t 值用 Newey-West (1987) 标准误，滞后阶数默认按 Newey-West (1994) 的经验规则 floor(4·(T/100)^(2/9))，
T 是交易日数（2024 年的研报持有到 2025 年一季度末，约 240 个交易日，取 4 阶）。
各列定义和算法见 `src/divergence_sort.py` 开头，行情和财务数据的路径、接入的财务科目在 `sort_divergence.py` 开头的运行参数里。

## 另一套抽取：概念有向无环图

`extract_concept_dag.py`（流程在 `src/concept_dag.py`，提示词在 `src/prompts/concept_dag_prompt.md`）和五层估值逻辑图是两套独立的代码，
输入同样是 `clear_txt.py` 的段落表和元数据表，结果另放在 `output_dag/`。每篇研报只做三件事，不预设会计科目、估值等固定概念，也不做任何补全：

1. **合并段落**：正文段落按顺序合并成不超过 `CHUNK_CHARS`（默认 2500）字的块，后面每块抽概念、连边各调用一次大模型，多数研报只有一块。
2. **抽概念**：按块顺序通读，滚动维护本篇概念表（规范名、别名、定义、类型：指标/事件/状态）；每个新概念要给出原文叫法，程序逐字核对，对不上的不收。
3. **连边**：各块并行，在原文里找概念表中概念之间“A 影响 B”的关系（只要因果，不要并列、包含、同义），引文逐字核对。各块的边按（A, B）合并，
   有环时每次删掉环上引文最少、在原文中出现得最晚的一条，得到有向无环图；没有连上边的概念不进图，仍留在概念表里。

```bash
python extract_concept_dag.py --dry-run     # 只统计选中的研报、段落合并后的块数，不调用大模型
python extract_concept_dag.py               # 抽取脚本开头 COMPANIES 里的几家公司（默认并发 50），已完成的研报和调用都跳过
python extract_concept_dag.py --companies all --workers 100   # 不限公司
python extract_concept_dag.py --deep-only 002594.SZ           # 这几只股票只抽深度报告（默认用脚本开头的 DEEP_ONLY，现在是比亚迪）
python extract_concept_dag.py --companies all --deep-only all # 全部公司都只抽深度报告
python extract_concept_dag.py --archived-only                 # 只用已有存档写输出、不调用大模型（少数研报一直跑不完时用，没存档的下次再补）
```

元数据表里没有“深度报告”这一列，`DEEP_ONLY` 里的股票会读原文首页，按“公司深度”“深度报告”“深度研究系列”“系列深度二”
一类的类型标注判断，首次覆盖报告（“公司首次覆盖报告”“评级：买入（首次覆盖）”）和“公司专题”也算（`src/report_meta.py` 的
`deep_line`；相关研究列表里“1.XX深度报告：……”这样带编号的条目、“上次评级：首次覆盖”不算）。个股研报里有 1252 篇带这些标注，
其中 277 篇正文不超过 5000 字（多是一两页的首次覆盖点评，或正文没提取出来），只抽深度报告时再按正文字数剔除
（`MIN_DEEP_CHARS = 5000`，`--min-deep-chars` 可改），剩 975 篇、421 家公司。深度报告最多的几家是比亚迪 19 篇、紫金矿业 13 篇、
徐工机械 12 篇、星宇股份 10 篇。（2026-11-20 前只认“深度”字样，是 733 篇、353 家；同时修正了元数据表里的股票代码：
行业研报首页估值表的一行被当成个股研报、指数代码 000001.SH 被当成股票、交易所后缀写错、研报被认成自己券商等，见 `src/report_meta.py` 开头。
2026-11-21 修了两处清理问题后从 944 篇、408 家变成现在的数字：一是页脚提示“请务必阅读正文之后的信息披露和法律声明”“敬请参阅末页
重要声明及评级说明”只删掉前半句，剩下的“和法律声明”“及评级说明”被当成末尾声明的小标题，从那里往后的正文全部丢掉（德邦证券、
华安证券的研报最多），正文里“加强信息披露监管”这样的句子也会被当成声明小标题；二是资料来源和下一段正文抽在同一行时整行当资料来源删掉
（东北、方正、华泰等）。修复后全库正文多了约 98 万字、1894 篇的正文有变化，其中 155 篇已在 DAG 里的重新抽取。另外元数据表里
只写“德邦研究所”的德邦证券研报（281 篇）、资料来源里同时出现“东吴证券”和“预测均为东吴证券”的东吴证券研报（527 篇）原来券商是空的，
抽取要求券商不为空，这些研报一直没有进候选，现在也补上了。）

结果：`output_dag/概念DAG.html` 是入口（双击打开，左侧选研报；点节点突出它的上下游，点边看原文引文和所在段落）；
`output_dag/data/` 下有 `concepts.parquet`（每篇的每个概念一行，`in_dag` 是否进图、`level` 从起点算起的最长步数）、
`edges.parquet`（每条边一行，`in_dag` 为假的是因成环删掉的边，`support` 是原文引文处数）、`reports.parquet`（每篇的概念数、边数、
起点和终点数、子图数、最长链步数等）。调用缓存、各篇存档（JSON，改了提示词或 `CHUNK_CHARS` 会自动重新处理）和进度日志在 `output_dag/cache/`。
`.claude/launch.json` 里的 `concept-dag-html` 可在浏览器预览页面。

并发：同时处理 `--workers` 篇研报，一篇做完马上开始下一篇（不分批，长研报先开始），同时在途的调用也限定为这么多个；
一篇里抽概念要按块顺序做，抽完马上并行连边，不等其他研报。进度日志每分钟记一行在途调用数，除了最后剩下不足
`--workers` 篇的时候都应是满的。名称统一（下文）的“统一”和“复核”两步也在同一个线程池里流水推进，某个候选组统一完，
它的合并组马上排进去复核。中转站对含有某些字眼的请求（如“成人内容”）也回 503“模型繁忙”，这样的块重试 6 次后放弃，
那一篇记进 `cache/failed_reports.jsonl`，不进输出。

### 跨研报统一概念名称

每篇研报各自给概念命名，同一个量在不同研报里常有不同叫法（营收 / 营业收入，PE / 市盈率），研报之间比较概念和边之前，
用 `unify_concept_dag.py`（流程在 `src/concept_unify.py`，提示词在 `src/prompts/concept_unify_prompt.md`）把这些叫法统一。
它只在细节层面合并明显完全相同的名称：范围、口径、主体、指标不同的（销量与海外销量，净利润与归母净利润），以及有因果、
上下位关系或带了方向的（规模效应与成本优势，息差与息差压力），都不合并。统一按公司分开做：同一个说法在不同公司的研报里
可能指不同的东西，所以只在同一家公司的研报之间统一；只有 1 篇研报的公司概念名保持原样。

1. 一家公司各篇的概念名按字面键归并成词条（字面相同的本来就是同一个名称），带上别名、定义和出现的篇数。
2. 同一家公司的两个词条共用一个名称或别名，或者名称的两字片段重合得多，就算候选对；候选对按相似度做单链接聚类，
   每组不超过 40 个名称，没有相近名称的不送模型。
3. 每个候选组调用一次大模型，组内找出完全相同的名称归为一组；组和组之间不再合并。统一名称由程序在组内挑：
   出现篇数最多的名称，带本公司名的排后，一样多时用模型的选择。
4. 复核：每个合并组再调用一次，逐个判断组内名称和统一名称在研报的用法里是不是同一个东西，范围、主体、指标确实不同的剔除，
   剩不到两个名称的组取消；研报里习惯省略的字眼（单车利润 / 单车净利润）、近义说法（上市 / 推出）、时间口径（月销 / 销量）
   算相同。第 3 步一次要看几十个名称，容易放宽（不复核时约一成的组把带限定的和不带的合到了一起，如库存与渠道库存、
   智驾车型销量与高阶智驾车型销量）；复核只会剔除、不会新增。复核要求“任何差别都判为不同”时又太严，把单车利润与单车
   净利润也拦下了，现在的口径介于两者之间。

```bash
python unify_concept_dag.py --dry-run          # 只统计候选组，不调用大模型
python unify_concept_dag.py                    # 只统一脚本开头 COMPANIES 里的公司（现在是比亚迪的 16 篇深度报告），并发 50
python unify_concept_dag.py --companies all    # 概念 DAG 里的研报全部参与统一（各公司分别统一；重跑时没变的候选组用缓存）
python unify_concept_dag.py --task structure   # 不调用大模型：按统一后的名称算各篇的源汇比、唯一有向路径数、平均路径长度，写出 output_dag/概念DAG结构.html
python unify_concept_dag.py --output output_dag/deep --task deep   # 不调用大模型：全样本深度报告的三个指标分布、明细表和每篇的结构图，写出 output_dag/deep/深度报告结构.html
```

全样本深度报告用 `extract_concept_dag.py --companies all --deep-only all --chunk-chars 5000 --workers 100 --output output_dag/deep` 抽取、
`unify_concept_dag.py --output output_dag/deep --companies all --workers 100` 按公司统一，结果在 `output_dag/deep/`（和上面比亚迪 2500 字一块的
结果分开）。`深度报告结构.html`（`src/viz/plot_deep_structure.py`）上面三张直方图是源汇比、唯一有向路径数（对数分箱，每十倍 6 档）、平均路径长度的分布，
下面一排是每家公司这三个指标的平均值的分布（横轴分箱同上，可只看研报多的公司，附组内相关系数 ICC）；点柱子筛选明细表；明细表点一行，下面画出这一篇的结构图：只有文中的概念和连边，从左到右按从源点出发的最长步数分列，源点蓝色、汇点橙色，
悬停概念突出它的上下游、显示经过它的完整路径条数。
2026-11-21 修复清理后重抽：975 篇里 974 篇有结果、420 家公司（正文变了的和新进候选的共 191 篇重新抽取；视源股份 5DF25733 第 2 块的回复连续不是合法 JSON，跳过，记在 `output_dag/deep/data/skipped_reports.csv`）。名称统一这次没有做完：合并提议 7036 组都已做完，复核做到 3270/5160 组时停下，已完成的都在缓存里；`concept_unify.parquet`、`深度报告结构.html` 和下面两项统计还是修复前 939 篇的版本（各篇源汇比中位数 1.27、唯一有向路径数中位数 58、平均路径长度中位数 2.04 条边）。
一次请求的流式返回超过 600 秒（`src/llm_client.py` 的 `STREAM_MAX`）就算这次调用失败、不再重发：上游卡住时会一直极慢地吐内容，
一次调用能拖到一个多小时，重发也常常同样慢。
`python endpoint_overlap.py` 算同一公司不同深度报告两两之间源点、汇点的重合率（Jaccard、重叠系数、去掉通用财务名称后、
统一前、不同公司随机配对的对照、疑似重复研报的上限参照），汇总写到 `output_dag/deep/data/endpoint_overlap.txt`，
每对、每家公司的明细是同目录的 `endpoint_overlap_pairs.parquet`、`endpoint_overlap_companies.parquet`。
`python sink_semantic_overlap.py` 只看汇点：精确名称和句向量（本地 BAAI/bge-base-zh-v1.5，名称 / 名称+定义）相似度匹配在各阈值下的
重合率、各相似度区间的配对例子，以及投资评级、营业收入、归母净利润等通用汇点在每对研报里是否都出现、通往它的路径条数，
写到 `output_dag/deep/data/sink_semantic_overlap.txt` 和 `sink_matches.parquet`。

结果也在 `output_dag/data/`：`concept_unify.parquet`（每篇每个概念的原名 `name` 和统一名称 `unified`）、`unify_groups.parquet`
（每个合并组的公司、统一名称、组内原名、涉及的研报数、理由和复核剔除的名称）、`unified_edges.parquet`（DAG 的边按统一名称改写，同一篇里重复的合并、
变成自环的去掉）。`output_dag/概念统一.html` 是核对页面，列出每一组合并了哪些名称、各出现在哪些研报；`概念DAG.html`
的工具栏多出“统一名称”开关。

## 第一步的两张表

`clear_txt.py` 在 `data/` 下写出两张表，用 `report_id`（txt 文件名里的 GUID）对应：

- `hs300_2024_paragraphs.parquet`：段落表，每行一个正文段落（旁边同名 `.txt` 便于人工检查）。
- `hs300_2024_reports.parquet`（另有同名 `.csv`，可用 Excel 打开）：研报元数据表，输入目录里每个文件一行。

| 列 | 含义 |
|---|---|
| `date` | 发布日期，取文件所在目录的月/日 |
| `broker` | 券商 |
| `report_type` | 报告类型：公司、行业、策略、宏观、固收、金工、晨报、基金、未知 |
| `is_stock_report` | 是不是个股公司研报 |
| `company`、`stock_code` | 这篇研报讲的那只股票（A 股、北交所、港股） |
| `analysts`、`analyst_ids` | 分析师姓名、执业证书编号（署名栏在很多纯文本里已经丢了，只有两三成能取到） |
| `status`、`n_paragraphs`、`n_chars` | ok 有正文 / empty 空文件、扫描件、乱码、外文 / failed；保留的段落数和字数 |

各列的取法见 `src/report_meta.py` 开头。`extract_logic_chain.py` 按这张表筛选要抽取的研报：默认只要个股公司研报，而且股票代码、公司名、券商都明确识别出来了，
并且只要脚本开头 `COMPANIES` 里的股票（列表为空则不限公司）。条件写在该脚本的 `select` 函数里，可以直接改（比如再限定日期、券商）。
同时在途的大模型调用数由脚本开头的 `WORKERS`（默认 100）控制。

## 输出

```
output/
  研报逻辑图.html   入口：双击打开。左侧选研报，选中时才读取这一篇的数据、计算布局并绘图
  data/             结果数据
    graphs/           每篇研报一个逻辑图数据文件，加上目录 index.js 和全局名称表 global_names.js
    edges.parquet、nodes.parquet、relations.parquet   扁平表
    compare/          框架对比的结果（themes、matrices、pair_metrics、report_metrics）
    divergence/       第三步的结果：公司-季度分歧度、匹配股票数据后的面板、分组统计和回归表
  cache/            重跑时用到的中间结果和日志
    *_cache.jsonl     大模型调用缓存（重跑时跳过已完成的调用）
    reports/          每篇研报做完逐篇几步后的存档（pickle，按文件名前两个字符分子目录；重跑时跳过已完成的研报）
    failed_reports.jsonl   出错而没有完成的研报（report_id、出错的步骤和异常、时间），下次运行自动重试
    progress.log      进度日志
```

页面要和 `data/` 放在同一个目录下，需要联网加载 vis-network。`.claude/launch.json` 里的 `report-logic-html`、`intro-page` 可在浏览器预览两个页面。

目前 `output/` 里是旧 PDF 流程那 16 篇研报（`data/report_paragraphs.parquet`）的结果；新 txt 的 3 篇试跑结果在 `output_sample/`。

## 流程

把一篇研报的推理链抽成五层有向图：公司估值 → 会计科目（预置）→ 经营指标 → 公司经营与行业 → 外部环境。

1. **文本清理**（`clear_txt.py`、`src/report_text.py`、`src/report_meta.py`）：研报纯文本 → 正文段落 parquet，按文字特征剔除图表、表格、侧栏、页眉页脚、风险提示和免责声明；同时写出研报元数据表。已有处理好的文本时可跳过，自备一份至少含 report_id、para_id、text 三列的 parquet 即可。
2. **逻辑链抽取**（`extract_logic_chain.py`、`src/report_logic.py`、`src/logic/`）：概念分层、原文连边、子图补全、包含关系缩减、全局去重。提示词在 `prompts/report_logic_prompt.md`。
3. **框架对比**（`src/report_compare.py`）：每家公司把概念归成主题，把图转成矩阵，算逐对、逐篇的差异指标。
4. **分歧度与股票数据**（`sort_divergence.py`、`src/divergence_sort.py`）：公司-季度分歧度，匹配行情和财务数据，分组统计和回归。
5. **页面**（`src/viz/`）：交互式逻辑图，说明抽取流程和分歧度量的介绍页，以及按公司统计的抽取进度页。

## 文档

- `docs/研报逻辑图抽取框架设计.md`：框架设计、各条规则的来由、各版本的结果。
- `docs/研报因果图合并与逻辑跳跃修正.md` 及 `docs/notes/`：抽取方法的文献调研。
- `docs/逻辑抽取与分歧度量介绍.html`：介绍页（由 `--task intro_page` 生成）。
- `docs/研报概念数据整理进度.html`：抽取进度页（由 `--task coverage` 生成，是生成时的快照）。
- `literature/分析师分歧/`、`literature/殊途同归/`：两个方向的文献与梳理；`literature/因果图抽取方法/`：抽取方法的论文。
