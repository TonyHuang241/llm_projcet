# 把段落级 LLM 抽取的小图合并成连通的文档级/跨文档因果图：实体消解、规范化与子图拼接

> 适用场景：中文卖方研报，按段抽取 (影响因素, 正向/负向, 被影响因素)，现有流程为 BGE 嵌入 + k-means + LLM 提议合并 + LLM 复核。
> 说明：每条 Cited Finding 都附来源。"Inferences" 是笔者结合用户场景的推断，不是文献结论。部分来源只看到摘要或第三方综述，已逐条注明。

---

## 问题 1：近期 LLM 建图框架在合并时如何做跨 chunk 的实体/关系消解与规范化？报告了哪些合并精度？

### Takeaway
主流框架分两类。一类是"粗合并"：GraphRAG、LightRAG、CORE-KG/LINK-KG 的下游只按名称（+类型）精确匹配，再用 LLM 汇总描述。另一类是"嵌入召回 + LLM 判定"：EDC、KGGen、iText2KG、Graphiti、RAKG 先用嵌入/BM25 召回候选，再交给 LLM 或阈值判定。真正量化过合并质量的很少：EDC 有人工评估，iText2KG 报了 FDR，KGGen 只报了合并后的规模缩减。用户现有流程（BGE + k-means + LLM）与 KGGen 几乎同构，KGGen 作者自己也承认过合并和欠合并问题仍未解决。

### Cited Findings

**EDC（Extract–Define–Canonicalize，EMNLP 2024，代码 github.com/clear-nus/edc）**
- 三阶段：开放抽取 → LLM 为抽出的每个关系写自然语言定义 → 用句向量对定义做相似检索，再规范化。Define 阶段的定义是规范化的辅助信息 — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)
- 有目标 schema 时：按定义的嵌入检索 top-5 候选，LLM 做多选题，答案可以是"None of the above"；没有合适候选就丢弃该关系，防止过度泛化 — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)
- 无 schema 时（self-canonicalization）：从空 schema 开始，每个新关系先检索已有的规范关系并让 LLM 核对，匹配不上就加入 schema，schema 逐步增长 — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)
- Self-canonicalization 的人工评估（两名标注者，一致率 0.94）：WebNLG 上精度 0.956、关系数 529→200；REBEL 上 0.867、667→225；Wiki-NRE 上 0.898、204→106。对照组 CESI 的精度只有 0.724/0.504/0.753，原因是它把不同关系过度聚到一起。schema 规模缩减约 62%/66%/48% — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)
- 局限：只研究了关系（schema）的规范化，实体去重只做了简单探索。用 GPT-3.5 跑全部组件约 0.009 美元/样本，LLM 调用多 — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)

**KGGen（stair-lab，2025，代码 github.com/stair-lab/kg-gen）**
- 先抽实体，再抽这些实体之间的关系（DSPy）。所有子图合并后统一小写、去掉完全重复项，这一步不用 LLM — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)
- 消解流程：S-BERT 嵌入 → k-means 切成每簇 128 项 → 簇内对每项用 BM25 与语义融合打分取 top-16 → LLM 找出其中的重复项（时态、单复数、大小写、缩写）并为每组选一个规范别名 → 移除已处理项，重复直到簇为空。实体和边分别执行这套流程 — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)
- Prompt 明确要求不合并意义不同的项，例子是"Type 1 diabetes vs Type 2 diabetes"和"MRI vs CT scan" — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)
- 规模与成本：100 万字符语料上实体 4602→3573（−22.4%），边 3180→2448（−23%），约 537 万 token、551 秒、0.84 美元（Gemini 2.0 Flash） — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)
- MINE-1 知识保留率：KGGen 66.07%，GraphRAG 47.80%，OpenIE 29.84%。该基准由作者自建，文章本身也是 LLM 生成的 — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)
- 作者承认过合并和欠合并仍是问题，金融等领域可能超出通用 LLM 的知识。论文内部有数字不一致：GraphRAG 关系复用次数正文写约 22 次，图注写约 2 次 — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)
- 库 API：`cluster=True`（可传 `context`）、`kg.aggregate([g1,g2])` 后再 `kg.cluster()`；`chunk_size`；模型用 LiteLLM 格式，可设 `base_url` — [GitHub stair-lab/kg-gen](https://github.com/stair-lab/kg-gen)

**iText2KG（2024，代码 github.com/AuvaLab/itext2kg）及后续 ATOM**
- 增量构建：第一篇文档的实体作为全局集合，之后每篇文档的局部实体先查是否已在全局集合中；不在的话按余弦相似度找全局匹配，超过阈值就并入，否则新增。关系也走同样的逻辑 — [arXiv 2409.03284](https://arxiv.org/abs/2409.03284)
- 阈值标定：用 text-embedding-3-large 嵌入样本对，实体平均相似度 0.6±0.12，关系 0.56±0.1，最终取 0.7 以保证高精度 — [arXiv 2409.03284](https://arxiv.org/abs/2409.03284)
- 实体 FDR（科研文章/简历/网站）：iText2KG 为 0.01/0/0，OpenAI Function 为 0.11/0/0.31，LangChain 为 0.14/0/0.29。评估规模很小，每类 5 篇文档，只用了 GPT-4 — [arXiv 2409.03284](https://arxiv.org/abs/2409.03284)
- 局限：阈值是超参数；实体类型尚未用作匹配条件；输入文档越多噪声越大 — [arXiv 2409.03284](https://arxiv.org/abs/2409.03284)
- ATOM（EACL 2026 Findings）：把文本拆成原子事实，各自建小图后多线程两两迭代合并。报告穷尽性 +18%、稳定性 +33%、延迟降低 90% 以上；README 称暂不处理时间和逻辑冲突 — [arXiv 2510.22590](https://arxiv.org/abs/2510.22590)；[GitHub AuvaLab/itext2kg](https://github.com/AuvaLab/itext2kg)

**Microsoft GraphRAG**
- 同 title 且同 type 的实体直接合并，描述收集成数组，再让 LLM 写一段涵盖所有不同信息的摘要；同一 source-target 的关系同样处理，重复次数作为边权 — [GraphRAG dataflow 文档](https://microsoft.github.io/graphrag/index/default_dataflow/)；[arXiv 2404.16130](https://arxiv.org/abs/2404.16130)
- 论文说明默认用精确字符串匹配，并声称"GraphRAG is generally resilient to duplicate entities"，理由是重复节点会落进同一社区，但这一点没有实测 — [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)

**LightRAG**
- 描述合并：描述条数少于 `force_llm_summary_on_merge` 且总 token 低于 `summary_max_tokens` 时直接拼接，否则调用 LLM 总结；描述过多时按 `summary_context_size` 分组做 map-reduce 式的递归总结 — [LightRAG operate.py](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py)
- 代码导入了 `normalize_entity_name`，但读到的片段里看不到实体匹配的具体规则（是否只是规范化后精确匹配） — [LightRAG operate.py](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py)

**Graphiti / Zep**
- 抽取时把前 n=4 条消息一并作为上下文。实体消解：名称嵌入（1024 维）做余弦检索，同时对名称和摘要做全文检索，候选连同上下文交给 LLM 判断是否重复；判定重复时由 LLM 给出最完整的名称并更新摘要。Prompt 要求同时参考名称和摘要 — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- 边去重只在同一对实体之间做混合检索，避免跨实体误匹配，也缩小搜索空间 — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- 论文没有任何去重精度或召回数字 — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- 开源代码在 LLM 之前加了一道确定性消解：规范化后精确匹配；名称长度 <6 且 token 数 <2 时要求字符熵 ≥1.5，否则视为低信息名称跳过模糊匹配；3-gram shingle + MinHash（32 个置换，band 大小 4）做 LSH 召回；Jaccard ≥0.9 才判为匹配；其余（多个精确匹配、熵过低、无模糊候选）留给 LLM。模糊匹配的规范化会把 `a-z0-9'` 以外的字符替换为空格 — [graphiti dedup_helpers.py](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/dedup_helpers.py)

**Neo4j LLM Graph Builder / LlamaIndex PropertyGraph**
- 官方功能页只描述了删除"只连到文本块的孤立实体"这一后处理，以及 schema 来源（预设、自定义、读取已有库、"Get Schema From Text"由 LLM 从文本建议 schema） — [Neo4j LLM Graph Builder features](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder-features/)
- README（镜像）列出去重配置 `DUPLICATE_SCORE_VALUE=0.97`、`DUPLICATE_TEXT_DISTANCE=5`、`GRAPH_CLEANUP_MODEL`。这是镜像站内容，未与官方仓库核对 — [llm-graph-builder 镜像](https://gh.r-mj.com/neo4j-labs/llm-graph-builder)
- LlamaIndex/Neo4j 博客的去重方法：向量索引对每个实体取 KNN top-10，保留相似度 >0.9、且（双向子串包含或编辑距离 <5）、且标签相同的邻居，只输出候选组供人工确认，不自动合并，也不用 LLM。示例输出里有误合并"1963 AFL Draft"与"1963 NFL Draft"。schema 约束用 `SchemaLLMPathExtractor(strict=True)` — [LlamaIndex blog](https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex)
- neo4j-graphrag 包提供精确匹配、Fuzzy（Levenshtein，依赖 rapidfuzz）和语义余弦三类 resolver（社区页面，二手来源） — [neo4j skill 页面](https://claudeskills.info/skills/neo4j-contrib/neo4j-skills/neo4j-document-import-skill/)
- 开源库 nodecanon 面向 LLM 抽出的图，综合名称、嵌入和邻居重叠打分，模糊对再交给可选的 LLM 判定 — [PyPI nodecanon](https://pypi.org/project/nodecanon/)

**OneKE / RAKG / CORE-KG / LINK-KG / 金融领域**
- OneKE（浙大 zjunlp，WWW'25 Companion）：Schema Agent、Extraction Agent、Reflection Agent 三个智能体，知识库存放 schema 和历史错误案例；支持 ChatGLM、Qwen 等 — [arXiv 2412.20005](https://arxiv.org/abs/2412.20005)
- RAKG：每个 chunk 先抽 pre-entity（带类型和描述）并以此作 RAG 查询，缓解长上下文遗忘；按向量相似度阈值得到候选，再由 LLM 确认是否合并。MINE 准确率 95.91%，GraphRAG 为 89.71%。消解细节来自第三方综述 — [arXiv 2504.09823](https://arxiv.org/abs/2504.09823)
- CORE-KG：先按实体类型逐类做共指消解，再用改造的 GraphRAG 抽取；节点重复率比 GraphRAG 基线低 33.28% — [arXiv 2506.21607](https://arxiv.org/abs/2506.21607)。消融显示：去掉共指后重复率 20.28%→26.01%，相对上升 28.25%；下游仍按字符串和类型精确合并，"Antonio"与"San Antonio"没能合并 — [arXiv 2510.26512](https://arxiv.org/abs/2510.26512)
- FinKario（基于东方财富研报，ACL 2026）：属性 schema 参考 CFA 和 J.P. Morgan 模板，事件 schema 是两层树（供给、需求、收入、成本效率、战略、技术、政策、宏观等 8 类）；质控三步为实体名规范化（如"BYD Auto→BYD"）、用 Tushare 补全数值属性、对含占位符的三元组回原文用 GPT-4o-mini 修正。论文没有给出归一化算法细节 — [arXiv 2508.00961](https://arxiv.org/abs/2508.00961)
- FinReflectKG（10-K，ICAIF'25）：单遍、多遍（抽取和归一化由两个 LLM 分别负责）、reflection 三种模式，reflection 模式规则合规率 64.8% — [arXiv 2508.17906](https://arxiv.org/abs/2508.17906)；EvalBench 中 reflection 模式的 precision 为 59.49 — [arXiv 2510.05710](https://arxiv.org/abs/2510.05710)

### Inferences
- 用户的"BGE + k-means + LLM 提议 + LLM 复核"与 KGGen 的"S-BERT + k-means(128) + top-16 + LLM"几乎同构，k-means 分区不稳定导致结果漂移是这类结构的固有问题（见问题 3）。
- 各框架共同的做法是：便宜的确定性召回（精确匹配、MinHash、嵌入 kNN）在前，LLM 只判定少量模糊对；判定时附上描述或定义（EDC 的 definition、Graphiti 的 summary、RAKG 的 description），并且允许回答"都不是"。这几点值得直接借鉴。
- Graphiti 的模糊匹配会把中文字符全部替换掉，中文实体名会直接落到"未解决→LLM"分支。中文场景应改用字符 2-gram/3-gram shingle（这是根据代码做的推断）。

### Gaps
- Graphiti、LightRAG、GraphRAG 都没有公开合并精度和召回。KGGen 只报了规模缩减，没有合并正确率。
- AutoKG、DeepKE 本轮未检索到与合并相关的可靠资料。
- Neo4j Graph Builder 后处理是否调用 LLM、以及阈值的确切含义，未在官方仓库核实。
- 没有找到专门针对中文研报"指标型节点"（毛利率、销量等）做实体消解的论文。FinKario 把数值当作属性，没有当作变量节点处理。

---

## 问题 2：如何在抽取阶段就让命名一致，降低后续合并难度？证据有多强？

### Takeaway
最有力的证据来自三项工作，都是在抽取时注入一份不断累积的"已出现实体/别名表"：LINK-KG 按类型维护 prompt cache，节点重复率相对降低约 45%，长文档收益最大；EDC+R 把实体和关系提示连同定义放进 prompt，F1 约 +0.05；iText2KG 把全局实体放进上下文。不过 iText2KG 也显示，把全局实体当作关系候选会降低精度（0.94→0.83）。更稳妥的做法是只把这份表当作"命名词表"使用，并配上定义。

### Cited Findings
- **LINK-KG（ICKG 2025）的三阶段 prompt cache**：(1) 每个 chunk 按实体类型做 NER，并给每个专有名词写简短角色描述；(2) 顺序处理各 chunk，Mapping-LLM 读入当前 cache 和本 chunk 的 NER 结果，输出更新后的 cache。cache 包含 RESOLVED_ENTITIES（别名→规范名）和 AUXILIARY_DESCRIPTIONS（每个规范实体的简短描述），模糊指称映射为 null；可选的 gleaning 第二遍会用完整 cache 回头修正早期映射；(3) Resolve-LLM 带着最终 cache 改写每个 chunk，把别名替换成规范名，禁止超出映射的推断 — [arXiv 2510.26486](https://arxiv.org/abs/2510.26486)
- LINK-KG 的节点重复率：短文档 GraphRAG 27.02% / CORE-KG 17.00% / LINK-KG 10.61%；长文档（>2500 词）36.01% / 26.10% / 17.78%。噪声节点在长文档上为 40.71% / 22.82% / 17.57%。评估只有 16 篇文档，没有金标准，单个专家标注，使用 LLaMA 3.3 70B、温度 0 — [arXiv 2510.26486](https://arxiv.org/abs/2510.26486)
- 作者认为文档较长时单次共指（CORE-KG）受上下文长度和"lost in the middle"影响，而分 chunk 加 cache 的方式更稳 — [arXiv 2510.26486](https://arxiv.org/abs/2510.26486)
- **EDC+R**：refinement 时在抽取 prompt 中加入实体提示（上一轮实体 + LLM 从文本另抽的实体）和关系提示（上一轮关系 + 检索器 top-10，均附定义）。定义能帮 LLM 排除检索到的无关关系。WebNLG 上 F1 0.746→0.794（GPT-4：0.783→0.820），REBEL 0.506→0.559，Wiki-NRE 0.647→0.693；第二轮迭代收益很小 — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)
- **iText2KG 的"全局实体 vs 局部实体"作关系上下文**：计算机领域三元组精度 0.83 vs 0.94，音乐领域 0.81 vs 0.90。全局实体能让图更丰富，但会带入更多无关关系 — [arXiv 2409.03284](https://arxiv.org/abs/2409.03284)
- **先实体后关系的两遍抽取**：KGGen 先抽实体，再只在这些实体之间抽关系 — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)；Graphiti 的事实只在给定实体之间抽取 — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)；LINK-KG 下游也是先按类型抽实体再抽关系 — [arXiv 2510.26486](https://arxiv.org/abs/2510.26486)
- **共指前置**：CORE-KG 先按类型逐类消解共指再抽取，去掉这一步后重复率相对上升 28.25% — [arXiv 2510.26512](https://arxiv.org/abs/2510.26512)
- **chunk 大小与 gleaning**：GraphRAG 发现 600 token 的 chunk 抽到的实体引用几乎是 2400 token 的两倍；gleaning 先用 logit bias 强制 LLM 回答是否有遗漏，再追问补抽，作者称这样可以用更大的 chunk 而不掉质量（只用实体计数佐证） — [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)
- **模板/schema 引导**：FinKario 用机构研报模板和 FIBO 约束抽取 — [arXiv 2508.00961](https://arxiv.org/abs/2508.00961)；OneKE 的 Schema Agent 选择或推断 schema — [arXiv 2412.20005](https://arxiv.org/abs/2412.20005)；Neo4j Graph Builder 支持"Get Schema From Text" — [Neo4j features](https://neo4j.com/labs/genai-ecosystem/llm-graph-builder-features/)；LlamaIndex `strict=False` 时 schema 只是建议 — [LlamaIndex blog](https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex)
- **从语料归纳 schema**：AutoSchemaKG 把实体、事件、关系概念化为更上位的类别，结合邻居节点上下文，形成多层概念 schema；与人工 schema 的语义对齐率在 v1/v3 写 95%，ACL 版写 92%（版本不一致） — [arXiv 2505.23628](https://arxiv.org/abs/2505.23628)
- **原子事实分解**：ATOM 先把文本拆成原子事实再抽取，报告稳定性 +33% — [arXiv 2510.22590](https://arxiv.org/abs/2510.22590)

### Inferences
- 对 60+ 段的研报，可以照搬 LINK-KG 的结构，但把"人物/地点"换成"变量节点"。第一遍逐段只抽变量名，并写一句定义（度量对象、口径、是比率还是绝对量、属于哪个业务）；第二遍顺序维护一份"规范变量表 + 别名 + 定义"（只传表，不传全文，token 可控）；第三遍抽因果三元组时把规范表作为命名词表注入，要求复用已有名称，或明确声明新变量。
- 由于 iText2KG 的精度下降，变量表只应约束"怎么叫"，不应提示"连什么"，以免诱导 LLM 编造跨段因果。
- EDC 的 definition 和 LINK-KG 的 auxiliary description 都说明，"名称 + 一句定义"比单独的名称更利于后续判重。BGE 也可以改为嵌入"名称：定义"。
- 用户不希望硬编码估值骨架。AutoSchemaKG 式的语料驱动概念层（如"收入类/利润率类/销量类"）可以作为折中，它是从数据归纳出来的。

### Gaps
- 没有找到在中文金融文本上量化"注入实体表"对重复率影响的研究。LINK-KG 只在英文法律文书上评估，样本 16 篇。
- cache 随文档变长而膨胀的问题，LINK-KG 没有给出 token 测量。
- 两遍抽取相对一遍抽取的独立消融，未找到严格对照的数字。

---

## 问题 3：如何让 LLM 实体消解更稳定？如何防止"相似但不同"概念（比率 vs 绝对量、部分 vs 整体）被误合并？

### Takeaway
不稳定的根源是：候选集合由 k-means 分区决定，而 LLM 只在局部做判断。改进方向有四点：(1) 用确定性分块（精确匹配 → MinHash/kNN 图 → 连通分量）替代 k-means；(2) 用"从候选中选一个或都不选"（selecting）或小集合聚类（LLM-CER，每组约 9 条、约 4 个实体）替代成对判断，并对每个决策做缓存，使结果与簇的构成无关；(3) 加入全局一致性修复（传递闭包冲突检测、误聚类护栏）；(4) 防误合并靠给出定义、显式提供"都不是"选项、给反例，以及用结构属性（度量类型、口径）做硬约束。

### Cited Findings
- **Match / Compare / Select（COLING 2025）**：逐对独立判断忽视了记录之间的全局一致性。8 个数据集、9 个 LLM 上，"selecting"（从候选列表中选出匹配项，或答 0 表示无）优于逐对 matching，一个摘要版本给出平均 F1 +17.58%。真实匹配项在列表中越靠后，选择准确率下降越多（位置偏差）。据此提出 ComEM：先用小模型粗筛，再对 top 候选用强模型做 select — [arXiv 2405.16884](https://arxiv.org/abs/2405.16884)
- **LLM-CER（SIGMOD'26）**：先分块（默认对 MiniLM 句向量做 LSH，阈值按 F1 调），块内构造约 9 条记录的集合，目标是约 4 个实体、簇大小均衡，相似记录相邻排列；LLM 直接聚类，每个簇用一个代表进入下一轮，同一集合产出的簇不再放到一起比较；误聚类护栏 MDG 把簇内相似度低于簇间相似度的记录标出来，重新生成集合后再聚类，时间开销约 10%，FP-measure 最多提升 75% — [arXiv 2506.02509](https://arxiv.org/abs/2506.02509)
- LLM-CER 的数字：与成对判断（集合大小 2）相比，API 调用少 12–108 倍，成本低 3–22 倍，准确率持平或略高（如 Cora ACC 0.88→0.90，调用 30.23K→0.28K）；集合大小 4–9 时质量稳定，超过 9–10 明显下降；簇大小越均衡越好；每组 4 个实体最佳；batch prompting 基线明显更差（Alaska ACC 0.33 vs 0.82）；gpt-4o-mini、温度 0、3 次平均 — [arXiv 2506.02509](https://arxiv.org/abs/2506.02509)
- **传递一致性**：LLM 记录链接会产生传递性矛盾，现有方法要么局部修复，要么靠聚类结构天然规避；有工作提出把 LLM 置信度作为分区级优化的输入，并提到 TransClean 用传递一致性找出误匹配 — [SBBD 论文](https://sol.sbc.org.br/index.php/sbbd/article/view/44035)
- **廉价 + 昂贵双 oracle 的相关聚类**：KwikBucks 无限次使用嵌入相似度，只在有限预算内调用强模型，查询量减少 3.5 倍以上 — [SustaiNLP 2023](https://preview.aclanthology.org/fix_video/2023.sustainlp-1.1.pdf)；Alper（2026）把 LLM 查询预算优先花在决策边界上的记录，置信记录用加权标签传播解决 — [arXiv 2605.25814](https://arxiv.org/abs/2605.25814)
- **确定性前置**：Graphiti 的精确匹配 + 熵门控 + MinHash/LSH + Jaccard≥0.9，剩下的才交给 LLM — [graphiti dedup_helpers.py](https://github.com/getzep/graphiti/blob/main/graphiti_core/utils/maintenance/dedup_helpers.py)；Graphiti 的边去重只在同一实体对之间做，以限制误匹配 — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- **防过合并**：KGGen 的 prompt 给出不应合并的反例（Type 1/Type 2 diabetes） — [arXiv 2502.09956](https://arxiv.org/abs/2502.09956)；EDC 显式提供"None of the above"，无匹配就新建或丢弃 — [arXiv 2404.03868](https://arxiv.org/abs/2404.03868)；iText2KG 选 0.7 的高阈值换精度 — [arXiv 2409.03284](https://arxiv.org/abs/2409.03284)；纯嵌入 + 编辑距离会误合并"1963 AFL Draft / 1963 NFL Draft"这类字面接近但不同的实体 — [LlamaIndex blog](https://www.llamaindex.ai/blog/customizing-property-graph-index-in-llamaindex)
- **嵌入相似 ≠ 因果结构相同**：先按变量嵌入聚类再分块建因果图，SHD 明显劣于直接提示；作者归因于语义相似和因果结构错位，以及两两合并时误差累积 — [Babakov & Bugarín-Diz, RetroEval 2026](https://citius.gal/en/research/publications/decomposition-does-not-help-evidence-from-semantic-clustering-in-llm-based-causal-graph-discovery/)
- **合并 vs 链接**：跨文档合并因果图时，"summarisation"把相似节点合成一个，图更小但容易被证伪；"analogy"保留原节点，在上方加一个共同的隐祖先节点相连，不破坏各自的机制，代价是节点增多。论文没有评估合并质量 — [arXiv 2410.06392](https://arxiv.org/abs/2410.06392)
- **LLM 匹配器的提示敏感性**：最好的 LLM 不需要或只需要很少示例；不存在通用最优 prompt，需要按模型和数据调；GPT-4 能给出结构化的匹配解释，用于定位错误 — [arXiv 2310.11244](https://arxiv.org/abs/2310.11244)
- **事件/变量名规范化的粒度**：嵌入 + 聚类规范化事件名，由 LLM 给簇命名并裁决边界样本，作者把规范化粒度当作可调的策略参数（只读到检索摘要） — [arXiv 2603.04276](https://arxiv.org/abs/2603.04276)

### Inferences
- **替代 k-means 的确定性方案**（推断）：对所有节点"名称+定义"做 BGE 嵌入，同时计算字符 2-gram Jaccard → 建 kNN 图（相似度 ≥τ 或 Jaccard ≥τ' 才连边）→ 取连通分量作为候选块；大块按 LLM-CER 的方式切成约 9 条一组，相似项相邻。给定嵌入和阈值后结果完全确定，不再受 k 值和随机初始化影响。
- **决策缓存**（推断）：把每次 LLM 的"同一/不同"判定按（规范化名 A，规范化名 B，定义哈希）存入持久表，后续遇到同一对直接复用。这样即使簇的构成变化，已判过的对也不会翻转，这是消除跨批次漂移最直接的方法。再配合温度 0 和 select 式 prompt（候选按相似度排序，或随机打乱多次投票，以缓解位置偏差）。
- **结构签名硬约束**（推断，属于定义性而非估值骨架）：先用规则或 LLM 为每个变量节点标注若干槽位，如度量类型（绝对量/比率/增速/价格/份额）、口径（总量/分地区/分产品/分业务）、主体（公司/行业/宏观）、财务科目归属。槽位不一致时禁止合并，如"毛利率≠毛利"、"出口销量≠销量"，应改为在两者之间建"部分-整体"边（见问题 4）。
- **传递闭包冲突**（推断）：LLM 两两判定为 A=B、B=C 但 A≠C 时，不应直接做闭包，应把 {A,B,C} 作为一个小集合整体交给 LLM 重新聚类（即 LLM-CER 的思路），或用相关聚类按置信度求解。
- 根据"Decomposition does not help"的负面结果，不宜把因果抽取本身按语义簇拆开；簇只用于判重，因果边仍以段落原文为准。

### Gaps
- 没有找到专门测量"同一流程重复运行时合并结果一致率"的研究，大多只报单次准确率。
- 自一致性/多数投票用于实体消解的直接对照数字未检索到。
- 中文短词（如"毛利率/净利率"）在 BGE 下相似度偏高、容易误合并，这只是常识性推断，没有找到实证数据。

---

## 问题 4：如何表示和推断"部分-整体/上下位"关系，让子图连通，又不凭空断言文本没说的因果？

### Takeaway
做法是把部分-整体（组成）和上下位（is-a）作为独立的非因果边类型，并标记来源（文本陈述 / 定义推断 / 规则推断）。这类结构可以来自定义性知识：XBRL calculation linkbase 的 summation-item（带 +1/−1 权重）正是"营业收入 = Σ 分部收入"、"利润 = 收入 − 成本"这类关系的形式化。工具层面可以借鉴 HippoRAG 的"同义边而非合并"，以及因果图合并中的"analogy"（加上位共同节点而不合并）。LLM 的部分-整体知识只是"准语义"水平，推断出的边需要校验和标记。

### Cited Findings
- **XBRL summation-item**：计算关系是 arcrole 为 summation-item 的 calculationArc，"from"端是合计概念，每个子项乘以权重后汇总；权重 +1 表示加入合计，负权重表示扣减 — [Altova XBRL 文档](https://www.altova.com/manual/XMLSpy/spyenterprise/xsxbrl_basics_creatingrelationshipstwo.html)；实际披露中存在 weight=−1.0 的行，把它当作 +1 会破坏汇总 — [EdgarTools 文档](https://edgartools.readthedocs.io/en/latest/xbrl/guides/calculation-linkbase/)；Calculations 1.1 规范把"共享同一合计概念、同一 extended link role、同一 arcrole 的所有 summation-item 关系"定义为一个 calculation — [XBRL Calculations 1.1](https://www.xbrl.org/Specification/calculation-1.1/CR-2022-10-12/calculation-1.1-CR-2022-10-12.html)
- **同义边代替合并**：HippoRAG 2 在嵌入相似度高于阈值（0.8，NV-Embed-v2）的短语节点之间加 synonym edge，不合并节点，用来跨段落连接同义词；段落节点通过"contains"边连到它包含的短语。论文没有单独消融同义边 — [arXiv 2502.14802](https://arxiv.org/abs/2502.14802)
- **加上位节点而非合并**：跨文档合并因果图时，"analogy"策略保留原节点，在上方加一个共同祖先，各图的机制保持不变 — [arXiv 2410.06392](https://arxiv.org/abs/2410.06392)
- **概念层归纳**：AutoSchemaKG 通过概念化把实体和事件组织成多层概念 schema，概念节点由 schema induction 生成，与实体和事件节点分开 — [arXiv 2505.23628](https://arxiv.org/abs/2505.23628)
- **LLM 分类体系归纳**：Chain-of-Layer（CIKM 2024）逐层构建 is-a 层级，每轮选一部分实体挂到已有树上，用分层格式把当前结构展示给 LLM，每层之后用 Ensemble-based Ranking Filter 剪掉幻觉 — [arXiv 2402.07386](https://arxiv.org/abs/2402.07386)；后续 BoostTaxo（2026）提出由粗到细的零样本方案，并以 Chain-of-Layer 为基线 — [arXiv 2605.12520](https://arxiv.org/abs/2605.12520)
- **LLM 的部分-整体能力有限**：用 ConceptNet 和 McRae 特征规范探测，LLM 对 meronymy 只有"准语义"能力，深层推理性质掌握不足 — [arXiv 2504.02395](https://arxiv.org/abs/2504.02395)；有 LLM 流水线从评论中抽 part-of 本体，由于没有金标准，只能用 LLM-as-judge 评估 — [arXiv 2510.13839](https://arxiv.org/abs/2510.13839)；从 LLM 挖掘部件和材料知识的工作，用覆盖率和正确性评估 — [arXiv 2410.12959](https://arxiv.org/abs/2410.12959)
- **推断边单独标记**：ai-knowledge-graph 工具中，推断出的三元组带 `"inferred": true` 和 `"method"`（llm_bridge / taxonomy / transitive / lexical 等），传递推断还记录中间节点 `via`，可视化中用虚线显示；默认开启的 taxonomy 规则会把具体词连到一般词（"quantum computing" is a "computing"）；transitive 和 lexical 规则默认关闭，因为测试中规则推断的边占全部边约 70%、噪声太大 — [GitHub robert-mcdermott/ai-knowledge-graph](https://github.com/robert-mcdermott/ai-knowledge-graph)

### Inferences
- **边类型设计**（推断）：在 (A, 正向/负向, B) 的因果边之外，增加 `PART_OF(部分→整体, weight=+1/−1)`、`IS_A(下位→上位)`、`SAME_AS_CANDIDATE(未确认的同义，带相似度)` 三类，每条边带 `provenance ∈ {text_stated, definitional, rule, llm_inferred}` 和支撑段落 ID。图连通性分析可以把这些边计入，但"因果链是否成立"的判断只走因果边，或者把 PART_OF 视为"定义性同向"并单独标注，不与文本因果混在一起。
- **定义性结构的数据来源**（推断）：(1) 会计恒等式和报表科目层级（营业收入 → 分部收入；归母净利润 = 净利润 − 少数股东损益；毛利 = 收入 − 成本），写成一张小型"科目-口径"词典，相当于 XBRL calculation 的思路，不涉及估值逻辑；(2) 节点名的构词规律，如"出口销量/内销量"共享中心词"销量"并带口径修饰，可以先用规则生成 PART_OF 候选，再交给 LLM 依据定义确认；(3) 研报正文中出现的"其中""占比""拆分"等表述，可抽为 text_stated 的 PART_OF。
- **比率与绝对量**（推断）：毛利率与毛利、收入之间是函数关系，不是部分-整体。可以设 `COMPONENT_OF_RATIO`，或者干脆只在规范表中记录"毛利率=毛利/收入"的定义，不作为因果边，避免把"毛利率↑→利润↑"这类文本没说的因果写进图里。
- 用 synonym edge 代替强制合并，可以缓解"合并不稳定"的影响：不确定的对只连软边，不改变节点身份；需要更强连通时再提升为合并。

### Gaps
- 没有找到在金融指标层面专门评估 LLM 部分-整体推断准确率的研究。
- 中国财政部"企业会计准则通用分类标准"（XBRL）能否直接作为中文科目的 PART_OF 词典，本轮没有检索核实。
- 把 PART_OF 边纳入因果链推理（如符号传播）是否会产生误导，没有找到相关实证。

---

## 问题 5：子图拼接——只给 LLM 某文档的节点和边列表（附带检索的支撑段落），能否识别同一节点、部分-整体和跨段因果？层级式（章节→文档）合并怎么做？

### Takeaway
直接相关的证据是 CrossAug（2026）：针对 chunk 局部抽取遗漏的跨 chunk 关系，先挑出"可能缺边"的局部子图，把子图的实体、已有三元组和证据段落一起交给 LLM 补全，要求每条新三元组引用 chunk ID，不能引用的丢弃。这样在约 8% 的额外 LLM 调用下，对多种 GraphRAG 框架都有小幅稳定的增益，抽检 84% 的新三元组正确。工程上可以按"连通分量 → 分量间候选对 → 带原文段落的 LLM 判定 → 引用校验 → 标记推断边"来实现。层级合并可参考 LINK-KG 的顺序 cache + 回扫、ATOM 的并行两两合并、LightRAG/Zep 的 map-reduce 总结；但"Decomposition does not help"提醒，多级合并会累积误差，合并应保守，并以证据为准。

### Cited Findings
- **CrossAug（arXiv 2605.28004，代码 github.com/DonFinliani/CrossAug）**：问题定义是现有框架只在单个 chunk 内抽取，证据跨段的关系系统性缺失；穷举 chunk 组合因组合爆炸而不可行 — [arXiv 2605.28004](https://arxiv.org/abs/2605.28004)
  - 候选选择：从根实体出发，在实体-实体边上做加权随机游走（事实边权重高于同义边）采样子图；GNN 打"缺失分"，训练是自监督的（正例是遮掉 20% 事实边或删掉 8% 非根实体的子图，负例是完整的更大邻域）；选取 θ≥0.5、与已选子图重叠低的子图，直到用完预算 — [arXiv 2605.28004](https://arxiv.org/abs/2605.28004)
  - LLM 输入为根实体、区域内所有实体、已抽三元组，以及来自事实边出处的证据段落；只能输出"directly stated or unambiguously supported"的内容，并必须引用 chunk ID，否则丢弃；新三元组按原抽取器的 schema 写回 — [arXiv 2605.28004](https://arxiv.org/abs/2605.28004)
  - 成本：LiteraryQA 上额外 1483 次 LLM 调用（基线的约 8.2%），时间增加约 15.3%，prompt token 占大头 — [arXiv 2605.28004](https://arxiv.org/abs/2605.28004)
  - 效果：HippoRAG2 在 MuSiQue 上 EM/F1 为 27.90/39.95→29.30/41.28；LightRAG、GFM-RAG 也有小幅提升；HotpotQA 持平或微降。把 GNN 换成随机选子图后所有指标下降。抽检 1000 条新三元组，84.35% 正确、81.54% 引用准确（以 LLM 为评判，作者人工复核 100 条，κ≈0.77）。论文没有"只去掉跨 chunk 边"的消融 — [arXiv 2605.28004](https://arxiv.org/abs/2605.28004)
- **ai-knowledge-graph 的分量桥接**：识别不连通分量（示例中有 11 个），`llm_bridge` 让 LLM 把孤立分量连到主图（`max_bridge_components=20`，`bridge_groups_per_call=5`）；`llm_hub` 补充中心实体之间的关系；推断边总数不超过抽取边的 `max_inferred_ratio=0.5` 倍，同一对节点不重复连接；推断边加标记，作者提醒需要人工审核 — [GitHub robert-mcdermott/ai-knowledge-graph](https://github.com/robert-mcdermott/ai-knowledge-graph)
- **局部因果图 → 全局图**：HiCaM 为每个 chunk 建局部因果图后合并成全局图（只读到检索摘要，未见规范化细节） — [arXiv 2505.24319](https://arxiv.org/abs/2505.24319)；CausalRAG2 用嵌入相似度识别语义相同的节点，合并描述并汇集证据边（只读到检索摘要） — [arXiv 2602.05143](https://arxiv.org/abs/2602.05143)
- **负面证据**：嵌入聚类 → 分簇建子图 → 两两合并，在 SHD 上明显不如一次性直接提示，原因是语义相似与因果结构错位，以及合并误差累积 — [RetroEval 2026](https://citius.gal/en/research/publications/decomposition-does-not-help-evidence-from-semantic-clustering-in-llm-based-causal-graph-discovery/)
- **顺序式层级合并**：LINK-KG 顺序处理各 chunk 并更新 cache，可选回扫（gleaning）用完整 cache 修正早期映射，再统一改写 — [arXiv 2510.26486](https://arxiv.org/abs/2510.26486)
- **并行两两合并**：ATOM 把原子小图多线程两两迭代合并（README 写 8 线程、批大小 40），延迟比 Graphiti 降 93.8%、比 iText2KG 降 95.3%（README 数字，与摘要的比较口径不同） — [arXiv 2510.22590](https://arxiv.org/abs/2510.22590)；[GitHub AuvaLab/itext2kg](https://github.com/AuvaLab/itext2kg)
- **map-reduce 汇总**：LightRAG 描述过多时按 token 预算分组递归总结 — [LightRAG operate.py](https://github.com/HKUDS/LightRAG/blob/main/lightrag/operate.py)；Zep 社区摘要用迭代 map-reduce，增量更新会逐渐偏离全量重算，需要定期全量刷新 — [arXiv 2501.13956](https://arxiv.org/abs/2501.13956)
- **用 pre-entity 检索相关 chunk**：RAKG 把每个 chunk 的 pre-entity 当作查询，检索其他相关 chunk 再抽关系，以缓解长上下文遗忘 — [arXiv 2504.09823](https://arxiv.org/abs/2504.09823)
- **分块大小权衡**：小 chunk 召回高（600 vs 2400 token，实体引用约 2 倍）— [arXiv 2404.16130](https://arxiv.org/abs/2404.16130)；FinKario 的质控对含占位符的三元组回原文修正 — [arXiv 2508.00961](https://arxiv.org/abs/2508.00961)

### Inferences
- **单篇研报的拼接流程**（推断，可直接实现）：
  1. 段落级抽取，保留段落 ID 和段内节点定义。
  2. 文档内判重：确定性分块 + select 式 LLM 判定 + 决策缓存（问题 3），得到规范节点表。
  3. 计算弱连通分量。对每对分量（或每个小分量对主分量），用节点"名称+定义"的嵌入召回跨分量的 top-k 候选节点对，同时加上构词规则候选（共享中心词）和段落邻近候选（相邻段或同一章节）。
  4. 每次 LLM 调用只传两个分量的节点和边列表（几十行文本），外加候选节点各自的支撑段落原文（2–4 段，CrossAug 式证据），让 LLM 在四类中选择：SAME_AS / PART_OF / 文中明确陈述的跨段因果 / 无关系。每条输出必须引用段落 ID 并摘录原句，引用不到的丢弃。
  5. 推断边单独标记来源，并设置 `max_inferred_ratio` 上限（参考 ai-knowledge-graph 的 0.5）。
  - 这样避免了整篇文档一次性输入 LLM，也不需要硬编码估值骨架，因为新边都必须有原文支持。
- **层级化**（推断）：先在章节内（如"盈利预测""业务分析""估值"）做判重和拼接，再在文档级只处理章节规范节点表。章节规范表通常只有几十到一两百个节点，一次调用放得下。顺序式 cache（LINK-KG）适合抽取期，并行两两合并（ATOM）适合大批量跨文档阶段。
- **跨文档**（推断）：不同研报之间宜先用 SAME_AS_CANDIDATE 软边（HippoRAG 式），只有当多篇报告、多次判定都一致时才提升为合并，以控制全局去重的不稳定传导到单篇图。
- 由于"Decomposition does not help"和 CrossAug 的引用约束，拼接阶段只允许 LLM 提出有原文证据的跨段因果；"定义性"连接（PART_OF/IS_A）可以不要求原文，但必须单独标记，不能当作因果。

### Gaps
- CrossAug 的评估是 QA 指标，不是图的连通性或因果正确率，也没有在金融或中文文本上验证；报告的增益（约 1 个 EM 点）较小。
- 没有找到"只给节点和边列表、不给原文"的拼接方式与"附原文段落"方式的直接对照实验。
- 章节→文档的层级合并相对一次性合并的误差，未找到量化研究；只有因果发现领域的负面结果作为旁证。
- HiCaM、CausalRAG2、Causality Elicitation（2603.04276）只读到检索摘要，未核实全文。
