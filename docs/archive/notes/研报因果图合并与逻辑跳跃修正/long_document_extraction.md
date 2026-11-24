# 长文档 LLM 抽取策略：介于「逐段」与「整篇」之间的上下文管理与多轮流水线（面向研报文档级因果图）

> 说明：本笔记面向报告撰写者，内容自洽、逐条带来源。标注「二手」的数字来自聚合站或摘要，未在原文表格中核对；标注「推断」的内容是基于引用证据的推理，不是已证实结论。用户场景：卖方研报（1 页点评 ~ 40 页深度、60+ 段），逐段抽取 (影响因素, 正向/负向, 被影响因素)，OpenAI 兼容 API（如 GLM flash），Python。

---

## 问题 1：带上下文的分块策略（滑动窗口重叠、按章节分块、给每块前置提纲/摘要/标题、跨块携带「记忆」）——哪些有证据？

### Takeaway
小块抽取召回更高，但会丢跨块联系；目前有证据的折中做法有三种：(a) 给每块前置「整篇文档视角下这块讲什么」的短上下文（Anthropic Contextual Retrieval，检索失败率降 35%–67%）；(b) 先抽实体清单、再以该清单为条件抽关系（KGGen 两步法，用来保证命名一致）；(c) 小块加 gleaning 补漏（GraphRAG）。LLM 在文档级关系抽取（DocRED）上，跨句关系的 F1 只有句内的一半左右，正好对应用户遇到的「跨段丢失」。

### Cited Findings
**块大小与召回（GraphRAG）**
- GraphRAG（Edge et al., 2024）在 HotPotQA 上用 gpt-4-turbo 比较块大小：600-token 块抽到的实体引用数「几乎是」2400-token 块的两倍；论文把原因写成长块会让「出现在块前部的信息召回变差」。正式索引用的是 600-token 块、100-token 重叠 — [GraphRAG arXiv 2404.16130](https://arxiv.org/html/2404.16130v2)
- GraphRAG 的 gleaning（论文称 self-reflection）：抽完后把已抽实体回填给 LLM，先用 logit bias=100 逼出 yes/no 判断「是否全部抽完」；如果答 no，就续写一句「MANY entities were missed in the last extraction」让模型补抽，最多循环到设定次数（论文没给具体次数）。论文称多轮 gleaning 能让大块也基本不掉召回 — [GraphRAG arXiv 2404.16130](https://arxiv.org/html/2404.16130v2)
- 默认参数各处说法不一：一份 GraphRAG best-practices 文档建议从 300-token 块和 1 次 gleaning 起步，并说每次 gleaning 大约让实体抽取成本翻倍 — [GraphRAG best practices（mintlify 托管页，非微软官方站，需谨慎）](https://www.mintlify.com/microsoft/graphrag/guides/best-practices)；Neo4j 的文章则说默认块大小已改为 1200 — [Neo4j 博客](https://neo4j.com/developer-blog/microsoft-graphrag-neo4j)。看起来随版本变化。
- GraphRAG 跨块合并实体只用**精确字符串匹配**。同一实体的多段描述交给 LLM 汇总，重复出现的关系计为边权；论文称系统「对重复实体总体鲁棒」，因为后续社区聚类会把重复项归到一起 — [GraphRAG arXiv 2404.16130](https://arxiv.org/html/2404.16130v2)

**前置文档级上下文（Contextual Retrieval）**
- Anthropic 的 Contextual Retrieval：把整篇文档放进 `<document>`、当前块放进 `<chunk>`，用 Claude 3 Haiku 生成「一段简短上下文，说明这块在整篇文档中的位置」（通常 50–100 token），再前置到块上。在 top-20 检索中，失败率（1−recall@20）的变化是：仅 contextual embeddings 5.7%→3.7%（−35%）；加 contextual BM25 →2.9%（−49%）；再加 rerank →1.9%（−67%）。用 prompt caching 时一次性成本约每百万文档 token $1.02（假设 800-token 块、8k 文档）— [Anthropic: Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- 注意：这组证据衡量的是**检索**，不是关系抽取，但机制相同：让局部块「知道」全局语境。

**先抽实体清单再抽关系（保证命名一致）**
- KGGen（2025）分两次 LM 调用：第一步从原文抽实体；第二步输入「原文 + 实体清单」再输出 (主, 谓, 宾) 三元组。作者明确说两步设计是为了**实体一致**。抽取模型为 Gemini 2.0 Flash，用 DSPy 签名做结构化输出 — [KGGen arXiv 2502.09956](https://arxiv.org/html/2502.09956)

**LLM 在文档级关系抽取（DocRED / Re-DocRED）上的表现**
- LMRC 论文（2024）在 100 篇 DocRED dev 文档上做 3-shot 测试：GPT-3.5-turbo-16k 的 F1 为 6.97；GPT-4-turbo 的 F1 为 10.71，其中**句内 F1 14.52、句间 F1 7.43**。文中引用 Zhu et al. (2024)：零样本 GPT-4 在 DocRED 上 F1 15.58，在 Re-DocRED 上 14.21 — [LMRC arXiv 2408.13889](https://arxiv.org/html/2408.13889)
- LMRC 给出的诊断：候选实体对里 NA 占绝对多数（DocRED dev 共 395,572 对，其中 384,949 对是 NA），LLM 注意力被分散；模型倾向把有关系的对判成 NA，输出三元组明显偏少（例：微调后的 LLaMA2-13B 在 DocRED dev 上抽出 4,925 条，标注为 12,275 条；Re-DocRED 上 precision 69.00、recall 27.43）— [LMRC arXiv 2408.13889](https://arxiv.org/html/2408.13889)
- LMRC 把任务拆成两段：先由 RoBERTa 分类器筛掉 NA 对，再让 LLM 只在候选对上分类。拆分后 GPT-4-turbo 3-shot 的 F1 从 10.71 升到 36.20（句间 F1 从 7.43 升到 27.30）。微调 LLaMA3.1-8B 版本的 DocRED test F1 为 60.69，仍低于 BERT 系 DREEAM 的 67.53 — [LMRC arXiv 2408.13889](https://arxiv.org/html/2408.13889)
- AutoRE（Vicuna-7B 微调）在 Re-DocRED test 上 F1 53.84（摘要级信息）— [AutoRE arXiv 2403.14888](https://arxiv.org/abs/2403.14888)
- 另有研究认为 ChatGPT（无论 GPT-3.5 还是 GPT-4）在 DocRE 上「仍远不及现有基线」，多维度提示可以缩小差距 — [Springer: Refining ChatGPT for DocRE](https://link.springer.com/chapter/10.1007/978-981-97-5669-8_16)

**逐块独立抽取的问题**
- 斯坦福 CS224V 关于 SLIDERS（跨长文档集抽取）的讲义观察到：逐块独立抽取的输出存在重复、信息不完整（课程讲义，非同行评审）— [Stanford CS224V lecture](https://web.stanford.edu/class/cs224v/lectures/l-longdoc-new.pdf)
- 一篇 2026 预印本认为「分块 + top-k」会打断远距离段落之间的联系，导致需要整篇信息的问题被漏答；在其初步实验中，整篇输入基线为 52.9%，其提出的索引方法更高（小规模初步实验）— [Indexing Long Documents for LLM-Based Analysis, arXiv 2608.21237](https://arxiv.org/html/2608.21237v1)

### Inferences
- （推断）DocRED 的「句间 F1 约为句内一半」和 LMRC 的「候选过滤后跨句 F1 大涨」放在一起看，说明跨段关系抽不出来的瓶颈主要在**候选空间过大、注意力分散**，模型本身不是不能做。用户场景的对应做法：不必把整篇喂给模型，而是给每段一个**小而精的全局候选集**（全文概念词表 / 论点提纲），让模型只在「本段内容 × 全局概念」上找关系。
- （推断）用户逐段抽取的四类症状，对应可直接借鉴的手段：
  - **同一概念换名** → KGGen 两步法：每段先给「全局概念清单」，要求优先复用清单名称，必要时再新增；
  - **段落缺语境** → Contextual Retrieval 式前置：给每段加「本段在全文论证中的作用」一句话，外加章节标题路径；
  - **漏掉中间概念** → GraphRAG 式 gleaning：追问「本段中是否还有被跳过的中间变量，如 …的竞争优势、…的增长空间」；
  - **投资建议直连估值** → 第 2 题中的整合与桥接轮。
- （推断）研报天然有章节标题，按「标题路径 + 段落」切分比固定 token 窗口更合适。另外 1 段重叠（把上一段原文或其摘要一并给出）的成本很低，可以缓解「承上启下」类表达（如「受益于上述因素」）指代不清的问题。
- （推断）前置上下文的长度要控制。用户如果用 GLM flash 这类小模型，前缀过长反而会触发第 3 题的长度退化。经验上，提纲加词表控制在几百 token 级别比较稳妥，具体阈值需要在用户数据上实测。

### Gaps
- 没有找到在**关系/因果抽取任务**上直接对比「逐段」「逐段 + 全局提纲」「逐段 + 滑窗」「整篇」四种设置 F1 的公开实验；Contextual Retrieval 的证据来自检索任务。
- 没有核实 GraphRAG 论文 Figure 3 中各块大小、各 gleaning 轮次的具体实体数（论文正文只给了「近两倍」）。
- 没有核实 GLM（智谱）API 是否支持与 Anthropic 类似的 prompt caching 及其计费；这直接影响「每段都前置全局提纲」的成本。
- 搜索摘要中出现过「GPT 在 DocRED dev 预测 3,422 条三元组」的说法，但在 LMRC 原文中没有找到，已剔除。

---

## 问题 2：多轮流水线（略读提纲 → 以提纲为条件的逐段抽取 → 仅在三元组上做文档级整合）；层级 / map-reduce / 阅读代理

### Takeaway
「短上下文 worker + 汇总 manager」这类分层方法在长文档任务上普遍持平或超过整篇输入：Chain-of-Agents 用 8k 窗口胜过 200k 全文输入，ReadAgent 的要点记忆在 QuALITY 上略胜全文输入。整合轮只处理抽取结果、不回读原文，token 消耗会大幅下降（GraphRAG 根层摘要比原文少 97% 以上 token）。实体消歧可用「embedding 召回近邻 + LLM 判重」的迭代聚类（KGGen、EDC）。

### Cited Findings
**Chain-of-Agents（顺序 worker + manager）**
- CoA（Zhang et al., NeurIPS 2024）把输入切段，worker 按顺序阅读，每个 worker 读一段并把累积笔记传给下一个，最后由 manager 汇总；作者称每个 agent 只看短上下文，可以缓解长上下文的注意力聚焦问题。无需训练、与任务无关 — [NeurIPS 2024 论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract.html)；[Google Research 博客](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/)
- 在 QA、摘要、代码补全任务上比 RAG、全文输入、其他多 agent 基线最多高 10%。以 Claude 3 Haiku/Sonnet/Opus 为骨干时，CoA (8k) 显著优于 RAG (8k) 和 Full-Context (200k)；输入越长、RAG 越检索不到金标准时，增益越大 — [NeurIPS 2024 论文页](https://proceedings.neurips.cc/paper_files/paper/2024/hash/ee71a4b14ec26710b39ee6be113d7750-Abstract.html)；[Google Research 博客](https://research.google/blog/chain-of-agents-large-language-models-collaborating-on-long-context-tasks/)
- 消融：去掉 manager 后性能明显下降，MuSiQue 上超过 10%（二手，来自评述站）— [Liner review](https://liner.com/review/chain-of-agents-large-language-models-collaborating-on-longcontext-tasks)

**ReadAgent（要点记忆 + 按需回查）/ MemWalker（摘要树导航）**
- ReadAgent（Lee et al., 2024）是纯提示的阅读代理：先把长文切成语义单元（episode），把每个单元压缩成「gist memory」，需要时再回查原文。QuALITY 上只用 gist 的压缩率为 84.24%，相当于有效上下文能多装约 3 倍原文；ReadAgent-P 的准确率 86.63%，高于 BM25 检索（84.28%）和全文输入（85.83%）（二手，来自聚合站）— [Emergent Mind: ReadAgent](https://www.emergentmind.com/topics/readagent)
- MemWalker（Chen et al., 2023）把长文建成摘要节点树，LLM 通过迭代提示在树上导航找相关段落 — [MemWalker arXiv 2310.05029](https://arxiv.org/abs/2310.05029)（仅引用方法描述，未核对数字）

**GraphRAG 的层级 map-reduce 与 token 成本**
- GraphRAG 先对实体图做社区划分并逐层摘要。回答全局问题时，map 阶段并行处理社区摘要块，每个答案打 0–100 分，0 分丢弃；reduce 阶段按分数降序填入上下文窗口，再生成最终答案。所有阶段都用 8k 窗口 — [GraphRAG arXiv 2404.16130](https://arxiv.org/html/2404.16130v2)
- token 对比（同一查询）：底层社区摘要（C3）比直接对原文做 map-reduce 摘要（TS）少 26–33% token，根层（C0）少 97% 以上。Podcast 数据集上 C0 为 26,657 token，TS 为 1,014,611 token — [GraphRAG arXiv 2404.16130](https://arxiv.org/html/2404.16130v2)

**整合轮：实体 / 关系规范化**
- KGGen 的消歧流程：所有子图合并后先统一小写（不用 LLM）；再用 S-BERT 嵌入，k-means 聚成每簇 128 项；对每一项用 BM25 + 语义融合分数取 top-16 近邻，交给 LLM 判断哪些是**完全重复**（考虑时态、单复数、大小写、缩写）。提示中明确要求**不要合并不同概念**（例如 1 型和 2 型糖尿病）。LLM 为每组选一个规范名并记录别名映射，处理完的项移出簇，循环到簇空为止。论文没有单独的校验步骤 — [KGGen arXiv 2502.09956](https://arxiv.org/html/2502.09956)
- KGGen 在 MINE-1（100 篇文章）上得分：KGGen 66.07%，GraphRAG 47.80%，OpenIE 29.84%。注意：论文 Table 1a 中 Gemini 2.0 Flash 一行为 44%，与「抽取用 Gemini 2.0 Flash」的说法对不上，headline 数字具体出自哪个模型没有说清 — [KGGen arXiv 2502.09956](https://arxiv.org/html/2502.09956)
- EDC（Extract-Define-Canonicalize，EMNLP 2024）分三阶段：开放抽取三元组 → LLM 为抽到的关系/类型写定义 → 用定义做向量检索召回相近的 schema 候选，再由 LLM 决定规范化 — [EDC ACL Anthology](https://aclanthology.org/2024.emnlp-main.548.pdf)；[OpenReview](https://openreview.net/pdf?id=PMa5ef9wxC)
- EDC 官方仓库说明目前只规范化关系，实体类型规范化尚未实现，并建议配合共指消解和实体指称规范化使用 — [clear-nus/edc GitHub](https://github.com/clear-nus/edc)

### Inferences
- （推断）建议的三轮（可扩展为四轮）流水线，与用户约束（不强加估值模板、长文不能整篇抽）对应如下：
  1. **Pass 0（规则，无 LLM）**：解析章节标题树和段落 ID。
  2. **Pass 1 略读**：产出「本篇论证提纲 + 关键变量词表」。长文可以做成 ReadAgent 式两级：先对每章生成 gist（或直接用标题 + 每段首句），再在 gist 上生成提纲；这样任何一次调用都不需要全文。输出包括：
     - 核心结论 / 投资主张；
     - 3–8 条支撑论点；
     - 每条论点涉及的变量及其规范名；
     - 每个变量出现的段落 ID。
     提纲由报告自身归纳，不是预设的估值骨架。
  3. **Pass 2 逐段抽取（可并行）**：输入 = 提纲 + 词表 + 标题路径 + 上一段（或其摘要）+ 本段原文。要求：(a) 优先复用词表名称；(b) 显式保留分析师写出的中间概念；(c) 每条三元组附原文证据片段。之后可接 1 次 gleaning。
  4. **Pass 3 整合（只看三元组 + 词表，不回读全文）**：先做 KGGen 式保守消歧；再做「桥接」检查——对「投资建议 → 估值 / 评级」这类直连边，让 LLM 在已有三元组中寻找能串起来的中间链路，或者标出「缺失中间机制」。若需新增边，必须引用某段原文并回到第 5 题的核验轮，防止凭空补链。
- （推断）CoA 式「顺序 worker + 累积记忆」是另一种实现：每段抽完后更新一份「已抽节点表 / 词表」并传给下一段。优点是命名一致性强、能吸收后文定义；缺点是不能并行（60 段即 60 次串行调用，延迟高），而且早期错误会顺着记忆传下去。Pass 1 提纲 + Pass 2 并行，以 1 次额外调用换取并行，延迟更可控。两者可以结合：在 Pass 1 产生的词表上并行抽取，Pass 3 再统一。
- （推断）成本量级：设 N 为段数，基线为 N 次调用。
  - 方案 A（提纲 + 并行逐段 + 整合）≈ 1~(章节数+1) 次略读调用，加 N 次抽取调用（每次多出提纲 / 词表前缀），再加 1–3 次整合调用。整合输入只有三元组，体量远小于原文，参照 GraphRAG C0 相对原文少 97% token。
  - 每加一轮 gleaning 或核验，约增加 N 次调用。
- （推断）用户之前的观察（Haiku 整篇抽短报告图更连通）与 CoA / ReadAgent 的结论不矛盾：短文本本来就在模型有效范围内，整篇输入天然带全局语境。可以按长度分流：短报告直接整篇抽（或整篇 + 逐段核验），长报告走分层流水线。阈值应在自有标注样本上标定。

### Gaps
- 没有找到名为「LongDocIE」的权威论文或基准，无法引用。
- 没有找到在**关系/因果抽取**任务上直接评估 CoA / ReadAgent / MemWalker 的结果；现有证据来自 QA / 摘要。
- ReadAgent 和 CoA 消融的具体数字来自二手聚合站，没有在原论文表格中核对。
- 没有找到「plan-then-extract」作为固定术语的代表性论文；「先提纲、后条件抽取」的直接证据主要是 KGGen 两步法和 Contextual Retrieval 两类间接证据。

---

## 问题 3：长上下文 LLM 在抽取 / 推理上的表现随长度退化（量化「整篇抽取增加误判」的担忧）

### Takeaway
证据一致：输入越长，性能越差，而且远在标称上下文上限之前就开始下降。约 3k token 时推理准确率已明显下滑（FLenQA）；32k 时 13 个模型中有 11 个跌到短上下文基线的一半以下（NoLiMa）；同一任务约 300 token 的聚焦输入胜过约 113k token 的全量输入，所有模型家族都如此（Context Rot）。中段信息最易被忽略，干扰项会诱发幻觉。大海捞针（NIAH）分数高不代表真实任务好，开源 / 小模型在需要全文推理的任务上退化更快。这支持用户「整篇抽取风险大」的担忧，尤其是 flash 级小模型。

### Cited Findings
- **Lost in the Middle**（Liu et al., TACL 2024）：准确率随相关信息位置呈 U 形，开头和结尾最好、中间最差；即使是标称长上下文的模型，输入变长性能也下降 — [arXiv 2307.03172](https://arxiv.org/abs/2307.03172)；[TACL](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00638/119630/Lost-in-the-Middle-How-Language-Models-Use-Long)
- **Same Task, More Tokens / FLenQA**（Levy, Jacoby, Goldberg, ACL 2024）：对同一推理任务用不同长度、类型、位置的填充文本「拉长」输入，准确率远在技术上限前就下降。一份摘要称平均准确率在约 3,000 token 时从 0.92 降到 0.68（二手）；即使填充的是**相关段落的重复**，多数模型也变差，说明长度本身就是因素；长输入上的 next-word 预测能力与长上下文推理表现不相关 — [ACL Anthology 2024.acl-long.818](https://aclanthology.org/2024.acl-long.818/)；[hyper.ai 摘要](https://hyper.ai/en/papers/acl__2024__2024.acl-long.818)
- **NoLiMa**（2025）：测试 13 个标称支持 ≥128K 的模型，32K 时有 11 个跌到短上下文基线的 50% 以下；表现最好之一的 GPT-4o 从 99.3% 降到 69.7%。作者归因于问题与「针」之间没有字面重合时，注意力机制在长上下文中更难找到相关信息 — [NoLiMa arXiv 2502.05167](https://arxiv.org/abs/2502.05167)
- **Context Rot**（Chroma, 2025 年 7 月，测 18 个模型，包括 GPT-4.1、Claude 4、Gemini 2.5、Qwen3）— [Chroma: Context Rot](https://www.trychroma.com/research/context-rot)
  - 即使是简单任务，可靠性也随输入长度下降；问题与针的语义相似度越低，下降越快。
  - 单个干扰项就会降低准确率，4 个干扰项降得更多，干扰项还会出现在幻觉答案中。
  - 打乱句序的 haystack 反而比连贯文章表现好，18 个模型都如此。
  - LongMemEval 上，约 300 token 的聚焦输入在**所有模型家族**中都胜过约 113k token 的全量输入；thinking 模式能缩小但不能消除差距。
- **HELMET**（Yen et al., ICLR 2025）— [HELMET arXiv 2410.02694](https://arxiv.org/abs/2410.02694)
  - 测 59 个长上下文模型，长度最高 128K。
  - NIAH 等合成任务不能可靠预测下游表现，多数模型 NIAH 接近满分。
  - 在需要全文推理或复杂指令遵循的任务上，开源模型落后闭源模型，且差距随长度扩大。
  - 7 类应用任务之间相关性低。
- **NoCha**（Karpinska et al., 2024）：针对 67 本新出版英文小说构造 1,001 对最小差异真/假命题；没有开源权重模型高于随机水平，最好的 GPT-4o 准确率 55.8%；模型在句子级检索上的表现好于全局推理 — [NoCha arXiv 2406.16264](https://arxiv.org/abs/2406.16264)
- 一篇 2026 年论文转述 Chroma 的结论：18 个模型在多跳任务上的 F1 都随输入长度单调下降（二手转引）— [arXiv 2603.26707](https://arxiv.org/pdf/2603.26707)

### Inferences
- （推断）对研报而言，40 页深度报告整篇输入通常在数万 token 量级（取决于分词器）。按 NoLiMa 和 Context Rot，这个区间的小模型已经明显退化。FLenQA 显示几千 token 就开始掉点，所以即便是 1–3 页的短点评，整篇抽取也并非零风险。不过短文档的退化幅度较小，加上整篇输入带来的跨段语境收益，Haiku 实验中整体为正。
- （推断）Context Rot 的「干扰项诱发幻觉」对研报尤其相关：研报里大量同类数字和指标（营收、毛利率、各业务线增速）彼此就是干扰项，整篇抽取时容易把 A 业务的驱动因素错配到 B 业务上，正是用户担心的「误判」。逐段抽取天然隔离了干扰项。
- （推断）「打乱句序反而更好」说明连贯长文中的论证结构不一定被模型顺利利用。因此不能假设整篇输入会自动抓住论证骨架，显式的提纲抽取（Pass 1）更可靠。
- （推断）实践建议：把**整篇输入只用在输出很小的任务上**（提纲 / 词表 / 主张，属于 gist 型任务，ReadAgent 显示 gist 能接近全文效果），细粒度三元组则坚持短上下文抽取。

### Gaps
- 没有找到专门测「文档级关系 / 因果抽取 F1 随文档长度变化」的长上下文基准数字（DocRED 文档较短，不能回答 40 页尺度的问题）。
- 没有针对 GLM flash 系列的长上下文退化公开评测；上述数字都来自其他模型，用户需要自测。
- 没有找到中文金融长文本的长度退化研究。

---

## 问题 4：面向分析师报告 / 金融文本的论辩挖掘与话语结构（主张–前提作为骨架，不强加估值模板）

### Takeaway
金融论辩挖掘已有数据集，但几乎都不是卖方研报。FinArg（财报电话会，主张/前提 + 支持/攻击关系）已有 GPT-4 等 LLM 的评测；Equity-AMSA 针对股票分析报告，但标的是情景和影响时长，不是主张–前提；AntCritic 是中文金融评论的主张/前提语料。这些数据和标签体系可以借鉴，用来设计 Pass 1 的「报告自身论证骨架」（主张 → 论点 → 证据），而不必套外部估值模板。

### Cited Findings
- **FinArg**：来自 Amazon、Apple、Microsoft、Facebook 2015–2019 年的季度财报电话会，标注句子级的前提 / 主张 / 非论辩标签，以及前提–主张对之间的支持 / 攻击关系 — [Lacuna 对 FinNLP 2024 论文的摘要](https://lacuna.tiptreesystems.com/work/exploring-large-language-models-in-financial-argument)
- FinNLP 2024 的论文用零样本 GPT-4 等通用及金融微调 LLM 测试金融论辩**关系识别** — [ACL Anthology 2024.finnlp-1.12](https://aclanthology.org/2024.finnlp-1.12.pdf)
- FinLLM 共享任务（IJCAI 2024 FinNLP-AgentScen）把 FinArg 类数据改成「前提 / 主张」二分类，训练集 7.75k、测试集 969 条，有参赛系统（BAI-Arg）用 LLM 完成 — [BAI-Arg, ACL Anthology 2024.finnlp-2.18](https://aclanthology.org/2024.finnlp-2.18.pdf)；[共享任务页](https://sites.google.com/nlg.csie.ntu.edu.tw/finnlp-agentscen/shared-task-finllm)
- Springer 的《Financial Argument Mining》一章：把分析师观点分解为前提、情景和前瞻性主张；介绍 Equity-AMSA（股票分析报告，约 7,000 条情景标注和 24,000 条影响时长标注），以及用微调 BERT 按论证强度给 2,280 篇专业分析师报告排序的工作 — [Springer: Financial Argument Mining](https://link.springer.com/chapter/10.1007/978-3-031-94687-5_2)
- **AntCritic**：约 1 万篇中文金融评论（自由格式、带图文），标注主张和前提；不是分析师报告 — [AntCritic arXiv 2208.09612](https://arxiv.org/pdf/2208.09612)
- 细粒度金融观点挖掘综述可作为标签体系参考 — [Fine-grained Financial Opinion Mining survey, arXiv 2005.01897](https://arxiv.org/pdf/2005.01897)

### Inferences
- （推断）可以把论辩挖掘的标签体系直接转成 Pass 1 的输出 schema：`{主张(投资结论/评级理由), 论点[i](支撑主张的机制), 证据[j](数据/事实, 段落ID), 关系: 支持/削弱}`。论点和证据**由报告自身归纳**，因报告而异：有的研报主线是「行业景气 → 量价 → 盈利」，有的是「新产品 → 份额 → 估值重估」。这样既提供跨段骨架，又不预设估值模板。
- （推断）Pass 2 的因果三元组可以挂到论点 ID 上（每条三元组带 `supports_argument=i`）；Pass 3 用「同一论点下的三元组应能连成链」做连通性检查。这能直接定位「投资建议直连估值、跳过中间机制」的问题：某论点下没有三元组覆盖，或者主张节点只连到估值节点。
- （推断）FinArg 上 LLM 的关系识别并不轻松（GPT-4 零样本被单独作为研究对象，说明不是已解决问题），因此 Pass 1 产出的论证骨架也需要人工抽检；不宜让骨架反过来强行约束三元组，只把它当作软提示和整合期的检查依据。
- （推断）研报中的「风险提示」段落对应论辩里的「攻击 / 削弱」关系，可以标为负向或反方论点，避免整合时与主线混在一起。

### Gaps
- 没有找到专门针对**卖方研报**、带主张–前提标注并做 LLM 评测的公开数据集，中文研报更没有。
- 没有取到 FinNLP 2024 论文中 GPT-4 在 FinArg 关系识别上的具体 F1，未能量化 LLM 做金融论辩结构的准确度。
- 没有找到「论辩结构作为因果图骨架」的直接实证研究；以上为推断性设计。

---

## 问题 5：自一致性 / 集成 / 核验轮（多次采样投票、LLM-as-judge 对照原文）以降低抽取不稳定

### Takeaway
多数投票式自一致性在单标签任务上有效，但三元组抽取是「多个合法答案的集合输出」。近期研究指出这种场景下投票会失效，逐候选核验更好。用户可以：先把多次采样结果做规范化（否则同义异名会被当成不同答案），再按「条目级出现次数」投票，并配合「对照原段落逐条核验」的 judge 轮。成本约为 k 倍采样加 1 次核验 / 段。

### Cited Findings
- 自一致性的原始方法（Wang et al.）：多路采样推理，取多数答案 — [Self-Consistency arXiv 2203.11171](https://arxiv.org/abs/2203.11171)
- Universal Self-Consistency 把思路推广到自由文本输出：由 LLM 自己在多个候选中挑出最一致的一个 — [Hugging Face Papers: Universal Self-Consistency](https://huggingface.co/papers?q=Universal+Self-Consistency)
- 一篇 2024 年关系抽取论文把自一致性作为**基线**（多次预测取多数关系标签），并提出「理由验证器 + 反馈修正」框架：检查模型给出的 rationale 是否支持其关系预测，再据此修正 — [Enhancing RE via Supervised Rationale Verification and Feedback, arXiv 2412.07289](https://arxiv.org/html/2412.07289v2)
- 一篇 2026 预印本指出：当合法答案很多时，投票会失效，逐候选核验更好（仅看到摘要级信息）— [When Many Answers Are Valid, Voting Fails, arXiv 2608.03506](https://arxiv.org/pdf/2608.03506)
- 改进投票的方法：按排序投票加权的自一致性 — [Ranked Voting based Self-Consistency, ACL Findings 2025](https://aclanthology.org/2025.findings-acl.744.pdf)；把多数投票视为估计模型最可能答案，并给出决定采样次数的序贯停止规则 — [Certified Self-Consistency, arXiv 2510.17472](https://arxiv.org/pdf/2510.17472)
- Chain-of-Verification（CoVe）：先生成，再规划核验问题、独立回答，然后修订原输出，以减少幻觉 — [CoVe arXiv 2309.11495](https://arxiv.org/abs/2309.11495)
- GraphRAG 把「重复出现次数」作为边权（关系出现越多权重越大），这本身就是一种跨块投票信号 — [GraphRAG arXiv 2404.16130](https://arxiv.org/html/2404.16130v2)
- KGGen 的 LLM 消歧提示中明确要求「不要合并不同概念」，以防过度合并 — [KGGen arXiv 2502.09956](https://arxiv.org/html/2502.09956)

### Inferences
- （推断）用户看到的「同一段多次运行得到不同 target」可以拆成两种：
  - (a) **命名漂移**：同一概念换了说法，可通过词表约束（第 1 题）和消歧（第 2 题）消除，不必投票；
  - (b) **真实不确定**：模型对因果方向或对象判断不稳，适合用多采样加核验处理。
  应先做 (a)，再对 (b) 投票，否则投票会把同义变体当成分歧。
- （推断）可行的稳定化方案（按成本从低到高）：
  1. temperature=0 加结构化输出（JSON schema），并要求每条三元组带原文证据片段（evidence span），缺少证据的三元组直接丢弃；
  2. **逐条核验**：每段 1 次 judge 调用，输入本段原文和该段全部候选三元组，逐条判断「原文是否明确支持该方向与对象」，返回支持 / 不支持 / 部分支持，可选给出修正。对应 CoVe 和「逐候选核验优于投票」的思路，成本约为 +N 次调用；
  3. **k 次采样 + 条目级投票**：k=3，temperature 0.5–0.7，规范化后保留出现 ≥2 次的三元组；只出现 1 次的进入 judge 复核，而不是直接丢弃，以免损失召回。成本约为 k×N 次抽取，加上复核；
  4. **异构集成**：两个不同模型（如 GLM flash 与另一家小模型）各抽一次，取交集作为高置信、差集送核验。
- （推断）对第 2 题 Pass 3 中「桥接补链」新增的边，应强制走第 2 步核验，并要求引用具体段落，防止整合轮「编故事」。这是控制整篇级推理误判风险的关键闸门。
- （推断）可以用重复运行的一致率（例如规范化后三元组集合的 Jaccard）作为流水线改动的回归指标：改提示、换模型、加提纲后，先看稳定性有没有改善，再看人工评估的正确率。

### Gaps
- 没有找到在**三元组 / 因果抽取**任务上系统报告「多次运行一致率」及自一致性改进幅度的论文。
- 「When Many Answers Are Valid, Voting Fails」只看到摘要级信息，具体方法和数字未核实。
- 没有找到 LLM-as-judge 在中文金融因果核验上的准确率数据；judge 本身也可能出错，需要人工抽样校准。
