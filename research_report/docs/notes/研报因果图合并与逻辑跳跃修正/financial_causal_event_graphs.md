# 金融/经济领域的因果图与事理图谱构建：片段级因果关系如何聚合成连贯图、如何补全中间环节

> 研究范围：中文事理图谱（HIT-SCIR 等）、金融因果抽取与因果知识图谱、面向盈利/估值的驱动因素图、跨文档统计、图的逻辑连贯性评估。
> 标注约定：【可实现】指可以只用 OpenAI 兼容 LLM API、本地 embedding（如 BGE）和常规 Python 库（scikit-learn、networkx）复现；【需额外资源】指依赖 GPU 微调、专有本体或未公开数据。
> 检索日期：2026-11-10。下文所有事实都附出处；推断单独放在 Inferences 小节。

## Q1. 中文事理图谱（HIT-SCIR 刘挺/丁效团队等）怎样泛化事件、合并节点、聚合边、形成多跳链？有没有公开系统或代码？

### Takeaway
HIT 一系的事理图谱（ELG、NEEG、抽象因果网络、CausalBank/CEG）都采用"抽象事件 + 计数统计边权"的套路：丢掉事件的具体参与者、时间和地点，用 embedding 相似度或高频词对做泛化，边权取共现或转移概率 P(B|A)=f(A,B)/f(A)。多跳链通常不显式构造，而是在推理时沿图游走或用 GNN 学习。金融版 ELG 规模很大（150 万以上节点），但代码和数据没有公开，只有一个在线 demo 和 NEEG 的代码。2026 年的新工作（AECR）则改用"LLM 多智能体抽象 + 带相似度约束的层次聚类 + 最小簇规模过滤"，把具体因果对蒸馏成抽象规则，可以直接借鉴到 BGE + LLM 管线中。

### Cited Findings
**ELG（事理图谱，Ding, Li, Liu, Liao, HIT, arXiv 2019）**
- 定义：节点是事件，边表示事件之间的顺承、因果、条件、上下位（is-a）关系。论文的出发点是：传统知识图谱只关注实体，忽略了现实世界中的事件 — [arXiv 1907.08015](https://arxiv.org/pdf/1907.08015)
- 事件表示为 (S, P, O) 三元组，必须带触发词，S 或 O 可以省略。候选三元组来自 Open IE，再用 LTP 句法分析得到主谓宾。低频三元组按阈值过滤，"go somewhere"这类过泛事件用正则和词典删除 — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- 泛化方式：忽略具体参与者、地点和时间，让事件保持抽象。由于词袋形式下完全相同的事件很少，作者用 neural tensor network 学习事件 embedding，再按余弦相似度连接语义相近的事件（论文没有给出阈值）— [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- 顺承边的权重是转移概率 P(B|A)=f(A,B)/f(A)，只保留共现 ≥5 的事件对。方向由"Preceding Assumption"决定：如果 A 先于 B 出现的次数多于反向，就连 A→B。分类特征包括频次、比率、上下文和 PMI — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- 因果边用 <Pattern, Constraint, Priority> 规则（如"[cause] leads to [effect]"）或 BERT+BiLSTM+CRF 的 BIO 标注抽取。在 1,000 句人工标注上，规则法 F1 为 72.6（P 93.7 / R 59.3），BERT 法 F1 为 83.6。论文没有给出因果边的强度公式 — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- 金融 ELG：基于腾讯财经和网易财经的 1,362,345 篇中文财经新闻，得到 150 万以上事件节点、180 万以上有向边。另有一个旅游 ELG，约 3 万节点、23.4 万边 — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- 多跳的用法：论文没有给出路径搜索或路径打分算法。在脚本事件预测中，若上下文事件与候选事件构成强连通分量，就视为合理的后继。在线金融 ELG demo 地址为 http://elg.8wss.com/（本次未核实是否仍可访问），论文未提供代码或数据 — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)

**NEEG（叙事事件演化图，Li, Ding, Liu, IJCAI 2018）**
- 从叙事事件链构图，在相关节点的子图上运行 scaled GNN（SGNN）。MCNC 准确率 52.45%，高于 EventComp 的 49.57% 和 PairLSTM 的 50.83%。arXiv 版说明数据和代码发布在 github.com/eecrazy/ 下（具体仓库名本次未确认）— [arXiv 1805.05081](https://arxiv.org/pdf/1805.05081)；[IJCAI 2018](https://www.ijcai.org/proceedings/2018/584)

**抽象事件因果网络（Zhao, Wang, Massung, Qin, Liu, Wang, Zhai, WSDM 2017，HIT+UIUC）**
- 从具体因果网络泛化出抽象因果网络，抽象事件节点是"高频共现词对"，再用 dual cause-effect transition 模型做网络 embedding。链接预测效果优于当时的 SOTA，应用场景包括事件预测、事件聚类和股市涨跌预测 — [Illinois Experts](https://experts.illinois.edu/en/publications/constructing-and-embedding-abstract-event-causality-networks-from/)；[HIT Scholar](https://scholar.hit.edu.cn/en/publications/constructing-and-embedding-abstract-event-causality-networks-from/)

**CausalBank / Cause Effect Graph（Li, Ding, Liu, Hu, Van Durme, IJCAI 2020，HIT+JHU）**
- CausalBank 含 3.14 亿个英文因果对，人工抽检 1,000 句，95% 表达了有意义的因果关系。Cause Effect Graph（CEG）是词汇级因果图，是对 CausalNet 的改进，语料更大更干净，因果关系也比 CausalNet 多。资源发布在 nlp.jhu.edu/causalbank — [IJCAI 2020](https://www.ijcai.org/proceedings/2020/502)；[arXiv 2107.09846](https://arxiv.org/pdf/2107.09846)

**AECR（抽象事件因果规则，Zheng, Chen, Wang, arXiv 2026-08，作者邮箱为 hust.edu.cn）**
- CACI 多智能体管线有五个 LLM 智能体：Event Restatement（结合上下文改写事件）、Gatekeeper（筛掉因果上不成立的对，输出 DISCARD）、Event Abstraction（为每个事件生成多个父概念）、Abstraction Selection（选出一对抽象）、Abstraction Judge（检验抽象是否过泛或过窄，不通过就带反馈重试，有最大重试次数）— [arXiv 2608.05205](https://arxiv.org/abs/2608.05205v1)（PDF 正文）
- 聚类：用冻结的 RoBERTa 编码模板"[CLS] 原因 ⊕ 'causes' ⊕ 结果 [SEP]"并做 mean pooling，距离取 1−cos。采用 complete-linkage 凝聚层次聚类，并设簇内相似度阈值 τ。成员少于 μ 的簇视为泛化性弱，直接删除。每个簇交给 LLM（例如 Gemini-3.1-Pro）蒸馏出一条抽象规则，提示词有三条原则：抓住多数成员共享的主导因果模式、忽略次要差异、既不过具体也不过泛 — [arXiv 2608.05205](https://arxiv.org/pdf/2608.05205)
- 规模：MAVEN-CGEP 的 32,263 个因果对中保留 26,084 个，得到 733 条规则，平均簇大小 35.6。ESC-CGEP 从 6,875 对中保留 6,031 对，得到 180 条规则，平均簇大小 33.5。在罕见和未见事件上的预测提升最大。领域是开放域和新闻，不是金融，论文页面也未提供代码 — [arXiv 2608.05205](https://arxiv.org/pdf/2608.05205)

**其他中文工作与开源项目**
- 刘焕勇的 ComplexEventExtraction：抽取中文因果、条件、顺承、转折等复合事件，用于构建事理图谱。项目比较了三种事件表示：短句（简单但噪声大、难以融合）、词序列和短语（最凝练）。示例是"猪瘟爆发→国内猪肉价格上涨"— [ComplexEventExtraction 文档](https://docsearch.algolia.com/mcp/docs/repo/liuhuanyong/complexeventextraction)
- 新疆财经大学的突发事件事理图谱（《上海理工大学学报》2023）：用规则模板抽取事件关系，事件向量化后用 K-Means++ 聚类得到抽象事件，关系类型以因果、并列、顺承为主 — [上海理工大学学报 2023](https://jns.usst.edu.cn/shlgdxxbzk/article/pdf/20230104)
- 武汉大学（《信息资源管理学报》2022）提出用金融事理知识图谱揭示突发事件的演化机理，辅助应急决策 — [信息资源管理学报 2022](https://jirm.whu.edu.cn/jwk3/xxzyglxb/CN/Y2022/V12/I3/137)
- 中科院软件所薛云志 2018 年做过题为"金融事理图谱构建及应用"的报告，摘要把事理图谱定义为"描述事件之间各种关系的事理演化逻辑有向图"— [中科院数学院报告通知](https://www.cc.ac.cn/Uploads/old/news/2018/20181031-2.pdf)
- 2025 年有研究构建了面向金融领域的中文事件关系语料库，并指出事件关系识别是构建事理图谱的关键 — [计算机应用与软件 2025](https://shcas.net/article/doi/10.3969/j.issn.1000-386x.2025.06.008)

### Inferences
- 事理图谱的"泛化"和用户的"变量节点去重"本质是同一个问题。ELG 用 embedding 余弦连接相似事件，WSDM 2017 用高频词对，AECR 用"聚类 + LLM 蒸馏"，对应用户的"BGE + k-means + LLM 合并"。AECR 有两点值得借鉴：一是把整条因果边（"A 导致 B"作为一个字符串）编码后聚类，而不仅仅是聚类节点，这样可以得到边级别的抽象规则，例如"贸易摩擦类 → 出口销量类"；二是 complete-linkage 加簇内相似度阈值，比 k-means 更不容易把语义跨度大的变量并在一起。【可实现：sklearn AgglomerativeClustering(linkage="complete", distance_threshold=1-τ) + BGE + LLM】
- ELG 的 P(B|A)=f(A,B)/f(A) 和共现 ≥5 的过滤可以直接用在跨研报聚合图上：每条规范化边 (A, 符号, B) 统计出现的研报数，再算条件概率，作为"典型中间环节"的统计依据（见 Q4）。【可实现】
- HIT 的事理图谱主要服务于"下一个事件预测"，并不保证链条在逻辑上完整，因此不能直接解决用户的"层级跳跃"问题。不过"抽象层上位关系（is-a）+ 因果边"的双层结构，可以把"中欧贸易摩擦"挂到"贸易摩擦/关税"的上位节点，再从上位节点继承跨研报统计出的典型下游链。

### Gaps
- 没有找到 HIT 金融 ELG 的公开代码或数据集，在线 demo 是否仍可用也未核实。论文没有给出因果边权重公式，也没有给出多跳路径打分方法。
- 没有找到《中文信息学报》《计算机学报》上专门讲"金融事理图谱 + 事件泛化"的具体论文全文（检索只命中其他期刊）。CEC 语料和 HIT 中文金融因果数据集的细节本次没有核实。
- CausalNet（Luo et al., KR 2016）基于必要性/充分性的因果强度公式本次没有取得原文，因此不作引用。

## Q2. 金融因果关系抽取与因果知识图谱（FinCausal、CausalBank 类挖掘、FinDKG、FinCaKG、LLM 构建的经济因果图）如何处理极性、来源冲突和节点粒度？

### Takeaway
在极性上，近年的 LLM 系统普遍把"增加/减少"编码进关系类型：FinDKG 用 Positive/Negative Impact On、Raise、Decrease，FinCARE 用 INCREASES/DECREASES 等枚举，经济叙事研究则给事件单独标 valence（上升/下降）。在粒度上，常见做法有三种：映射到受控的规范变量表（FinCARE 用 18 个规范列，Schmidt 等用宏观锚点类目）、对齐到本体（FinCaKG-Onto 对齐 FIBO），或者用 embedding 聚类（relatio、FinDKG）。来源冲突的处理是短板：只有 INDRA/World Modelers 一系有系统化的 matching/contradiction/refinement 组装逻辑和 belief 打分，金融领域的系统多数只做去重和阈值过滤。FinCausal 至今仍是片段级任务，不涉及图聚合。

### Cited Findings
**FinCausal 与片段级数据集**
- FinCausal 2025（COLING 2025 FNP 工作坊）从抽取任务改为生成式 QA：上下文是年报段落，问题是抽象式的，答案从原文抽取。评测指标为 EM 和 SAS，数据来自英国和西班牙年报，英文、西班牙文子任务各有 10 支队伍参赛。FinCausal 2026 将在 LREC 的第 7 届 FNP 举办，并更换评测指标 — [ACL Anthology 2025.finnlp-1.21](https://aclanthology.org/2025.finnlp-1.21)；[FinCausal 2026](https://www.lllf.uam.es/wordpress/?p=1790)
- FinReason（EACL 2021）面向公司公告中金融事件的原因抽取，包含 8,794 篇文档、12,861 个金融事件和 11,006 个原因片段 — [EACL 2021](https://preview.aclanthology.org/author-url/2021.eacl-main.175/)
- CCKS 2021 金融事件因果抽取：原因和结果两侧都结构化为"事件类型 + 实体"，例如原因实体为生猪、结果实体为豆粕（引自 CFLLM 论文转述的样例）— [arXiv 2403.06249](https://arxiv.org/pdf/2403.06249)。一篇复现博客称语料来自公开新闻和研报，共 7,000 句（二手来源）— [腾讯云开发者](https://cloud.tencent.cn/developer/article/2473539)

**FinDKG（Li & Sanna Passino, ICAIF 2024）**
- 用 GPT-4 标注约 5,000 篇新闻，微调 Mistral-7B 得到 ICKG，再从约 40 万篇 WSJ 文章（1999–2023）中抽取带时间戳的四元组和五元组。共 15 种关系，其中带极性的有 Impact、Positive Impact On、Negative Impact On、Raise、Decrease；12 种实体类别中包括 ECON IND（经济指标）、SECTOR、CONCEPT — [arXiv 2407.10909](https://arxiv.org/html/2407.10909v1)
- 跨文章的实体消歧用 Sentence-BERT embedding，但论文没有说明聚类方法、阈值或合并规则，也没有报告 ICKG 抽取的精确率和召回率 — [arXiv 2407.10909](https://arxiv.org/html/2407.10909v1)
- 代码在 github.com/xiaohui-victor-li/FinDKG，模型在 huggingface.co/victorlxh/ICKG-v3.2（仅限非商业用途）— [arXiv 2407.10909](https://arxiv.org/html/2407.10909v1)

**FinCaKG / FinCaKG-Onto（东京工业大学，Xu, Takamura, Ichise）**
- 管线由因果检测、实体链接和因果对齐三个模块组成，以 FIBO 本体为指导构建金融因果知识图谱（ICSC 2024）— [FinCaKG 项目页](https://www.ai.iee.e.titech.ac.jp/FinCaKG/)；[AMiner](https://www.aminer.org/pub/65fe674913fb2c6cf61e9e2d/a-framework-to-construct-financial-causality-knowledge-graph-from-text)
- FinCaKG-Onto 指出，原版 FinCaKG 直接依赖原始文本串、没有做本体对齐，在不一致或语义薄弱的情形下会失效。改进版用 BERT 因果检测后，把片段链接到 FIBO 标准概念，在大规模财报上报告的本体一致性为 95.6% — [ITM Web of Conferences 2025](https://www.itm-conferences.org/articles/itmconf/pdf/2025/10/itmconf_keis2025_01029.pdf)

**FinCARE（arXiv 2025-10）：从 10-K 中抽取 LLM 因果三元组，作为因果发现的先验**
- 用 Qwen3-235B-A22B 处理 96 份 10-K，得到 3,419 个三元组。字段包括 cause/effect_entity、*_type（METRIC/EVENT/CONDITION/POLICY/MARKET）、relationship（INCREASES/DECREASES/TRIGGERS/PREVENTS/AMPLIFIES/MODERATES）、strength（STRONG/MODERATE/WEAK）、time_lag 和 evidence（原文引用）。没有单独的符号字段，极性编码在 relationship 枚举中 — [arXiv 2510.20221](https://arxiv.org/html/2510.20221v1)
- 粒度方面，提示词直接要求把短语映射到 18 个规范变量列（例如"Tariffs → Regulatory_Risk_Score"），没有 embedding 或聚类合并步骤 — [arXiv 2510.20221](https://arxiv.org/html/2510.20221v1)

**Eidos + INDRA + Delphi / INDRA World（DARPA World Modelers，社会经济因果模型）**
- 三段式管线：Eidos 是基于规则的开放域因果阅读器；INDRA 聚合多来源的因果信息并做组装，生成连贯的知识库并评估可靠性；Delphi 把量化的因果片段组装成可执行的概率模型 — [NAACL 2019 demo](https://preview.aclanthology.org/teach-a-man-to-fish/N19-4008)；代码分别在 github.com/clulab/eidos、github.com/sorgerlab/indra、github.com/ml4ai/delphi
- INDRA 的组装逻辑会识别陈述之间的 matching、contradiction 和 refinement 关系（refinement 指一条陈述是另一条更一般或更具体的版本），并给每条陈述一个 belief 分数，汇总所有直接和间接支持证据。系统支持增量组装，可以导出 CAG（因果分析图）、TSV 等格式 — [INDRA World 文档](https://indra-world.readthedocs.io/en/latest/general.html)；[INDRA World 模块](https://indra-world.readthedocs.io/en/latest/modules/index.html)
- Eidos 用概念状态（concept states）而不是简单的促进/抑制极性，因为只记极性会丢失信息 — [NSF PAR](https://par.nsf.gov/servlets/purl/10200912)（经检索摘要转述，未细读原文）

**经济叙事研究（经济学界）的极性和粒度处理**
- Schmidt, Lange, Reccius, Müller, Roos, Jentsch（arXiv 2506.15041，2025）的流程：GPT-4o 抽取"事件 A 导致事件 B"；用 GPT-4o-mini 把复合事件拆成原子命题（例如"能源和食品价格上涨"拆成两条）；再用 LLM 给每个事件标 topic 和方向 valence（rising/falling/high），输出 JSON；最后用 all-MiniLM-L6-v2 把 topic 映射到预定义的宏观锚点类目（余弦相似度），并人工修订。他们刻意保留经济学上有意义的区分，例如"通胀预期"和"通胀"不合并 — [arXiv 2506.15041](https://arxiv.org/pdf/2506.15041)
- Huang, Weinig, Fritsche, Usbeck（arXiv 2603.01930，2026）的通胀叙事图标注：节点是事件，边是指向通胀的因果关系，带 Increases/Decreases 极性。事件类目有 26 个子类（需求 5、供给 9、其他 12）。"Pent-up Demand 与 Demand (Residual)"、"政府支出与政府债务"最容易混淆，迫使类目反复拆分和重命名 — [arXiv 2603.01930](https://arxiv.org/html/2603.01930v1)
- relatio（Ash, Gauthier, Widmer，Political Analysis 2024）：用 SRL/SVO 抽取"施事-动词-受事"，对施事和受事联合做 embedding（平均 GloVe）加 K-means 聚类（唯一的超参数是簇数），从而能连成有向多重图。人工评估过 100/500/1000/2000 个簇的质量。可通过 pip install relatio 安装 — [arXiv 2108.01720](https://arxiv.org/pdf/2108.01720)；[GitHub relatio](https://github.com/relatio-nlp/relatio)
- Heddaya 等的"Causal Micro-Narratives"（arXiv 2410.05252）从美国新闻中抽取通胀因果叙事。微调后的小模型优于 GPT-4o 且更便宜，最佳 F1 约 0.71 — [arXiv 2410.05252](https://arxiv.org/html/2410.05252v1)

**日本金融因果链（Sakaji/Izumi，东京大学）**
- 从日文业绩快报（決算短信）和新闻中抽取因果表达，后续有双语（日英）金融因果知识抽取的工作（New Generation Computing 2023）— [ACL Anthology 2021.fnp-1.3](https://aclanthology.org/2021.fnp-1.3)；[检索汇总](https://liner.com/review/economic-causalchain-search-using-text-mining-technology)

### Inferences
- 用户的 (影响因素, 正向/负向, 被影响因素) 与 FinDKG（Positive/Negative Impact On）、FinCARE（INCREASES/DECREASES）、Huang 等（Increases/Decreases）在结构上一致。不过 Schmidt 等和 Eidos 的做法提示还有一种更稳健的表示：把方向写进节点状态（"海外销量↑"），边只表示"导致"。这样能区分"A 上升导致 B 下降"和"A 下降导致 B 下降"这类非对称情形，而单一符号会把两者混为一谈。用户现在的"同向/反向"语义已经隐含线性和对称的假设，这一点需要在报告里写明。
- 冲突处理可以借鉴 INDRA 的三类关系：matching 对应完全重复，合并后累加证据；contradiction 对应同一 (A,B) 符号相反，保留双方证据并按研报数或券商数投票，或者标记为"条件依赖"；refinement 对应一般与具体（如"贸易摩擦"和"中欧贸易摩擦"），建上下位边而不是强行合并。belief 可以先简化为 1−∏(1−p_i)，其中 p_i 是单条证据的可信度（例如 LLM 自评或研报级权重）。这是推断，INDRA 文档没有给出具体公式。【可实现】
- 在粒度上，Schmidt 等的"锚点映射 + 保留经济学区分"和 FinCaKG-Onto 的"对齐 FIBO"说明：只靠无监督聚类会把"毛利率"和"净利率"、"通胀"和"通胀预期"这类变量并在一起。建议在 BGE 聚类之外再维护一份不长的财务变量锚点表（营收、毛利率、费用率、归母净利润、EPS、估值倍数等）。这份表只用于节点规范化，不强制规定链条结构，因此不违背用户"不硬编码估值骨架"的要求。

### Gaps
- 没有核实 CCKS 2021 金融因果数据集的官方字段 schema（reason_type、result_product 等）和许可。
- 没有找到专门处理"跨来源极性冲突"的金融领域论文；INDRA belief 模型的具体公式本次没有取得。
- 没有找到 FinDKG 对 Sentence-BERT 消歧阈值的说明。

## Q3. 有哪些工作从文本中构建通向盈利或估值的"驱动因素树/因子图"？它们如何在没有固定模板的情况下把宏观/行业因素接到公司财务上？

### Takeaway
直接从卖方研报构建"驱动因素 → 财务 → 估值"因果链的公开工作很少。FinKario（ACL 2026）是最接近的：它从东方财富研报中抽取"属性图 + 事件图"，事件按 Supply、Demand、Revenue、Cost、Policy、Macro 等"驱动类别"组织，但没有显式的事件到财务再到估值的链，也没有做跨研报的节点合并。FinCARE 用 18 个规范财务和风险变量作为节点，以跨公司频次给边打分，再交给 PC、GES、NOTEARS，这是"数据驱动骨架"的范例。日本的 Izumi/Sakaji 因果链检索系统通过"前一条因果的结果 ≈ 后一条因果的原因"（embedding 余弦超过阈值）拼接多跳链，从而把宏观事件接到行业和公司业绩上，没有用模板。这正是用户"跨段落/跨研报拼链"的最直接先例。

### Cited Findings
**FinKario（HKUST-GZ 等，ACL 2026；arXiv 2508.00961）**
- 数据为东方财富网研报（2024-08-28 至 2025-02-28），用 MinerU 把 PDF 转成 Markdown 并去掉免责声明。价格数据取自 Tushare，行业分类取自 Wind — [arXiv 2508.00961](https://arxiv.org/html/2508.00961v1)
- 双层结构：属性图记录行业、代码、产品线、风险因素、评级、目标价等相对稳定的属性（11 类属性关系）；事件图记录驱动财务指标的时变因素，按"driven category"组织，包括 Supply、Demand、Revenue、Efficiency Cost、Strategic Action、Technology Innovation、Policy Regulation、Macro，事件子类有"Has increased / decreased"等 — [arXiv 2508.00961](https://arxiv.org/html/2508.00961v1)
- schema 由 LLM 自顶向下生成：属性 schema 参照 CFA 和 J.P. Morgan 报告模板，事件 schema 参照威斯康星大学模板的高层类目，再落到 FIBO 本体。抽取时要求输出主体、客体、时间范围、驱动类别，并附一段推理说明（JSON）— [arXiv 2508.00961](https://arxiv.org/html/2508.00961v1)
- 局限：跨研报只做名称规范化（例如"BYD Inc./BYD Auto → BYD"），没有描述去重或实体合并。论文没有展示"事件 → 财务 → 估值"的显式因果链，也没有评估因果有效性。评估方式是 FinKario-RAG 的周度多头回测（ARR 2.633、SR 4.926、ACC 0.581，均为作者自报）— [arXiv 2508.00961](https://arxiv.org/html/2508.00961v1)
- 三元组数量随版本不同：v1 写 9,625，ACL 版写 210,328；实体数均为 305,360 以上，关系类型 19 种。代码在 github.com/Jackson906E/FinKario — [ACL 2026](https://aclanthology.org/2026.acl-long.446.pdf)；[GitHub](https://github.com/Jackson906E/FinKario)

**FinCARE（arXiv 2510.20221）：规范变量节点 + 跨公司统计 + 因果发现**
- 节点是 18 个规范列（如 Market_Risk_Score、Revenue_Growth_YoY、Regulatory_Risk_Score）。LLM 抽取的边按跨公司频次和覆盖度打分（公式见 Q4），高分边作为"必需边"、低频边作为"禁止边"注入 PC、GES、NOTEARS。另有 MissingEdgeDiscoverer，由 LLM 补充置信度 ≥0.7 的缺失边。去掉这个模块后 F1 下降 0.076±0.018 — [arXiv 2510.20221](https://arxiv.org/html/2510.20221v1)

**Izumi & Sakaji 的经济因果链检索（FinNLP@IJCAI 2019；FNP 2021 扩展）**
- 三步流程：用 SVM 从日文业绩快报中找因果句；用 4 种句法模式抽取原因和结果表达；把一条因果表达的结果与另一条的原因相连。相似度计算方法是：分别对两段表达的词向量取平均，再算余弦相似度，超过阈值就连成链。每条因果表达都带有发布公司和日期元数据 — [Izumi & Sakaji 2019 PDF](https://aclanthology.org/W19-5510.pdf)
- 抽取效果：在业绩快报上 P 0.82、R 0.62、F1 0.71；在业绩相关新闻上 P 0.93、R 0.75、F1 0.83。测试集为 30 份快报，含 478 个因果表达。原型系统（http://socsim.t.u-tokyo.ac.jp/ccs/）可以根据输入词展示因果链，用户能交互式地挑选合适的因果、删除不合适的 — [Izumi & Sakaji 2019 PDF](https://aclanthology.org/W19-5510.pdf)
- 2019 年的论文只评估到抽取步骤，没有评估链条质量。2021 年的扩展把因果链检索用于经济指标预测，2020 年 JSAI 的报告则用于股价分析 — [ACL Anthology 2021.fnp-1.3](https://aclanthology.org/2021.fnp-1.3)；[JSAI 2020](https://www.jstage.jst.go.jp/article/jsaisigtwo/2020/FIN-024/2020_167/_article/-char/en)

**Hashimoto 等的未来情景生成（ACL 2014）：因果链拼接的经典方法**
- 用"AB + BC → ABC"把抽取到的事件因果对拼成情景链，并引入模板的 excitation（兴奋/抑制）语义属性来判断两段因果是否可以衔接，从而放宽拼接条件。要求每条因果"自足"，即脱离原句也能理解为因果。共生成 50,000 条情景，精度 68%；示例链"毁林持续→全球变暖加剧→海水升温→副溶血弧菌污染"并不在原始语料中出现 — [ACL Anthology P14-1093](https://aclanthology.org/P14-1093/)
- 兴奋/抑制方向的定义来自 Hashimoto 等 EMNLP 2012 的"Excitatory or Inhibitory"，该文用它从网络文本中抽取矛盾和因果 — [ACL Anthology D12-1057](https://aclanthology.org/D12-1057)

**其他**
- 有研究用分别基于年报、业绩会纪要和分析师报告的三张因果知识图谱，辅助 LLM 生成业绩会回答。本次只看到二手摘要，原文未核实 — [PromptLayer 摘要](https://www.promptlayer.com/research-papers/rehearsing-answers-to-probable-questions-with-perspective-taking)
- FinReflectKG（arXiv 2508.17906）是 FinCARE 所依赖的抽取框架，从 SEC 10-K 构建知识图谱，包括版面解析、表格感知分块、schema 引导的迭代抽取和反思式反馈循环，并用 LLM-as-a-judge 评估 — [arXiv 2508.17906](https://arxiv.org/html/2508.17906v2)

### Inferences
- 在不用固定模板的前提下，把宏观因素接到公司财务上，文献里有三条可组合的路线：
  1. **Izumi/Hashimoto 式拼链**：在用户的图中，对每个连通分量的"汇点"（没有出边、又不是估值的节点），用 BGE 余弦在其他段落或其他研报的"原因端"节点中检索候选（余弦超过阈值），并要求符号兼容（类似 excitation）。这样可以把断开的子图接起来。【可实现】
  2. **FinCARE 式数据驱动骨架**：先在大量研报上统计规范化变量之间的边频次、研报覆盖度和券商覆盖度，挑出高分边作为"软先验骨架"。它是学出来的，不是硬编码的，符合用户可以接受的边界。【可实现：CompositeScore 只需计数】
  3. **FinKario 式驱动类别**：用 LLM 给每个节点打一个粗粒度"层级/类别"标签（宏观/政策 → 行业供需 → 公司经营量价 → 利润表科目 → 估值），作为节点属性而不是边约束，供后面的层级跳跃检测使用。
- FinKario 回测结果好，并不等于它的因果链合理（论文没有评估因果有效性）。用户如果要在报告里引用它作为"研报因果图"的先例，应当说明它本质上是 RAG 用的知识图谱，不是因果 DAG。

### Gaps
- 没有找到公开的、针对中文卖方研报的"驱动因素 → 盈利 → 估值"因果链数据集或评测集。
- 没有找到专门做 KPI 驱动树（value driver tree）自动抽取的论文；KPI-EDGAR 等 KPI 抽取工作本次没有检索核实。
- 没有找到中国券商或金融科技公司公开的"产业链/事理图谱"技术报告（本次检索未命中）。

## Q4. 这些系统如何用跨文档统计（链条在大量研报或新闻中出现的频次）推断典型中间环节、验证边？

### Takeaway
跨文档统计在文献中主要有四种用法：（a）边权/转移概率，如 ELG 的 f(A,B)/f(A) 加共现阈值；（b）支持度过滤，如 AECR 删除成员少于 μ 的簇、FinCARE 删除提及少于 5 次的边；（c）综合可信度，如 FinCARE 的 CompositeScore =（强度 × 频次 × 公司覆盖度）的几何平均，INDRA 的 belief；（d）角色不对称统计，如 Schmidt 等发现"股市"在 90% 的情况下作为结果、"政府支出"在 91% 的情况下作为原因，这可以用来做数据驱动的变量层级。在"推断中间环节"方面，最接近的方法是 Hashimoto/Izumi 的链拼接，以及 FinCARE 用 LLM 补缺失边。不过有一条重要警告：叙事研究发现，经济叙事长尾极重，409 条叙事大多只出现一次；GPT-4o 还会用自己的经济学知识改写原文的因果链。

### Cited Findings
- **ELG 转移概率**：P(B|A)=f(A,B)/f(A)，只保留共现 ≥5 的事件对，方向按"谁先出现得多"确定 — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- **FinCARE CompositeScore**：CompositeScore(u→v) = (S_strength · S_freq · S_cov)^(1/3)，其中 S_strength=(n_strong+0.5·n_moderate)/n_total，S_freq=n_mentions/max(所有边的提及数)，S_cov=n_companies/max(公司覆盖数)。例如 Market_Risk_Score→Revenue_Growth_YoY 在 96 家公司中的 91 家被提及 317 次，得分 0.876 — [arXiv 2510.20221](https://arxiv.org/html/2510.20221v1)
- **FinCARE 阈值与先验注入**：得分 >0.4 且提及公司数超过 18 家的边记为必需边（原文渲染为">>"，运算符不确定）；提及少于 5 次的边记为禁止边，结果为 11 条必需边、119 条禁止边。PC 中对必需边跳过独立性检验，其余边用 α_adj = α·exp(−w)；GES 用 BIC + λ·R_KG；NOTEARS 出现环时删掉最弱的非必需边。论文内部有若干数字不一致（例如某例子的得分算不出 0.23），而且评估用的合成数据本身由同一套 KG 先验生成，提升可能被高估 — [arXiv 2510.20221](https://arxiv.org/html/2510.20221v1)
- **AECR 支持度过滤**：凝聚聚类后删除成员少于 μ 的簇，理由是这类簇代表泛化性弱的模式；保留下来的规则平均簇大小约 34–36 — [arXiv 2608.05205](https://arxiv.org/pdf/2608.05205)
- **INDRA belief**：每条陈述的 belief 汇总所有直接和间接支持证据，refinement 关系意味着更具体陈述的证据会间接支持更一般的陈述 — [INDRA World 文档](https://indra-world.readthedocs.io/en/latest/general.html)
- **Hashimoto 链拼接（ACL 2014）**：通过 AB+BC→ABC 生成的情景可以包含原始语料中没有出现过的完整链，精度 68% — [ACL Anthology P14-1093](https://aclanthology.org/P14-1093/)
- **Izumi/Sakaji 链拼接**：用结果表达与原因表达的平均词向量余弦是否超过阈值来决定能否连接，并允许用户交互删除不合适的环节 — [Izumi & Sakaji 2019 PDF](https://aclanthology.org/W19-5510.pdf)
- **角色不对称（Schmidt 等 2025）**："股市"作为结果事件的占 90%，"政府支出"作为原因事件的占 91%，作者认为这与经济学中的理论先验一致 — [arXiv 2506.15041](https://arxiv.org/pdf/2506.15041)
- **长尾警告（Schmidt 等 2025）**：80 篇文档在拆分和标准化后得到 409 条不同的叙事（两个事件 × 两个 valence），大多只出现一次，最常见的完整叙事也只有两三次。原因是"类别 × valence"的组合空间太大 — [arXiv 2506.15041](https://arxiv.org/pdf/2506.15041)
- **LLM 改写链条的风险**：原文说的是"硬通货流入推高通胀"，GPT-4o 却编码成"货币供应增长推高通胀"。作者指出模型的经济学知识会和忠实呈现原文相冲突，导致因果焦点偏移 — [arXiv 2506.15041](https://arxiv.org/pdf/2506.15041)
- **FinKario 没有用跨研报统计**：只做名称规范化，没有跨报告的合并或频次验证 — [arXiv 2508.00961](https://arxiv.org/html/2508.00961v1)

### Inferences
- **数据驱动的变量层级（解决层级跳跃的检测问题）**：在全部研报的规范化节点上统计 role_ratio(v) = 出度/(入度+出度)，也可以按研报数计。仿照 Schmidt 等的 90%/91% 统计，"公司估值、归母净利润"会稳定地落在结果端，"贸易摩擦、政策"落在原因端。再用聚合图的拓扑序或最长路径深度给每个节点一个连续的"层级分"。如果一条边 A→D 的层级差远大于该类边在全库中的典型层级差，就标记为疑似跳跃。这样不需要人工写骨架。【可实现：networkx】
- **带符号的中间环节补全**：对疑似跳跃的边 A→D，在跨研报聚合图中搜索 A⇝D 的路径，长度 2–4，每条边的支持度都不低于阈值，并要求路径符号乘积等于原边符号（正×负=负，类似 Hashimoto 的 excitation 兼容）。可以用 ELG 式的转移概率乘积或 FinCARE 式 CompositeScore 的几何平均给路径打分。找到高分路径时，把原边改标为"总效应/捷径边"，并在本研报图中插入中间节点（标记为"由跨研报统计补全"，与原文抽取的边区分）。找不到时保留原边，交给 LLM 在本研报原文中检索中间环节。【可实现】
- **需要防范的副作用**：Schmidt 等的例子说明，LLM 或统计补全的中间链可能"经济学上合理，但不是这份研报说的"。所以补全出来的节点和边必须单独标注来源（原文 / 跨报统计 / LLM 推断），后续分析时也要能把它们剔除。长尾问题意味着：只有在边足够规范化（变量级而非事件级）时，频次统计才会收敛。用户的节点本来就是"变量"而不是"事件"，粒度上比叙事研究更容易收敛，这对用户有利。
- **跨研报图的构建顺序建议**：节点规范化（BGE+聚类+LLM），然后边级聚合（按 (A, 符号, B) 计研报数、券商数、时间跨度），再做冲突分类（matching/contradiction/refinement），然后打分（CompositeScore 类），最后做 DAG 化（删除形成环的最弱边，参照 FinCARE 的 NOTEARS 后处理）。

### Gaps
- 没有找到用"跨文档链频次"专门推断缺失中间环节并做过定量评估的金融论文。Hashimoto 2014 的 68% 是情景整体精度，不是中间环节召回率。
- 没有找到 HIT 金融 ELG 中因果边的跨文档权重公式。
- 没有找到在文本因果图上做（带符号）传递约简的公开方法或评估（见 Q5）。

## Q5. 有哪些方法可以评估这类图的逻辑连贯性和完整性？

### Takeaway
现有评估大致分三类：（1）下游任务代理，如 ELG/NEEG 的 MCNC 脚本事件预测、FinKario 的回测、FinDKG 的链接预测；（2）与金标准图比较，如 FinCARE 在合成 DAG 上算边 F1，叙事研究用节点/边 Jaccard、归一化图编辑距离和图级 Krippendorff α，并区分严重偏差和轻微偏差；（3）一致性指标，如 FinCaKG-Onto 的本体一致性 95.6%。关于"直接边和间接边"有两条直接相关的发现：人工标注时，只标一跳直接原因（Adjacent Story）比多跳链更一致，存在"完整性与一致性的权衡"；LLM 构建因果图的大量假阳性来自分不清直接和间接因果，所以有工作在评估时先对 LLM 图做传递闭包。目前没有专门衡量"链条是否跳级"的标准指标。

### Cited Findings
- **下游代理（ELG/NEEG）**：MCNC 脚本事件预测，SGNN 52.45%，加上 EventComp 和 PairLSTM 后为 54.93%，随机基线为 20% — [ar5iv 1907.08015](https://ar5iv.labs.arxiv.org/html/1907.08015)
- **下游代理（FinKario）**：周度多头回测的 ARR、SR、MDD、ACC 等，消融实验显示去掉事件图后收益下降。文中数字前后不一致（0.336 和 0.386）— [arXiv 2508.00961](https://arxiv.org/html/2508.00961v1)
- **图级一致性（Huang 等 2026）**：把 Krippendorff α 扩展到图，分三档距离：宽松（有重叠即算一致）、中等（节点和边用 Jaccard，整图用按空图距离归一化的 GED）、严格（完全匹配）。All Events 在三档下分别为 0.868、0.481、0.244，说明宽松指标会高估可靠性。只含一跳直接原因的 Adjacent Story 为 0.702/0.441/0.202，含多跳链的 Full Story 为 0.678/0.320/0.127。作者把这称为覆盖面与一致性之间的权衡 — [arXiv 2603.01930](https://arxiv.org/html/2603.01930v1)
- **严重/轻微偏差（Schmidt 等 2025）**：金标准为 100 篇文档中的 291 条叙事。"A 导致 B、C、D 却被合并成 A 导致 (B+C+D)"算轻微偏差；改变含义、虚构或遗漏叙事算严重偏差。GPT-4o 准确率 44%，专家为 67–74%；Jaccard 0.40 对 0.59–0.60；严重偏差率 1.25，专家为 0.35–0.49 — [arXiv 2506.15041](https://arxiv.org/pdf/2506.15041)
- **合成 DAG 上的边 F1（FinCARE）**：29 条边的金标准 DAG。PC 从 0.459 提升到 0.622，GES 从 0.367 提升到 0.735，NOTEARS 从 0.163 提升到 0.759。但合成数据由 KG 先验生成，存在循环论证的风险 — [arXiv 2510.20221](https://arxiv.org/html/2510.20221v1)
- **本体一致性（FinCaKG-Onto）**：大规模财报上为 95.6% — [ITM Web of Conferences 2025](https://www.itm-conferences.org/articles/itmconf/pdf/2025/10/itmconf_keis2025_01029.pdf)
- **直接与间接的混淆**：有研究把 LLM 构建因果图的大量假阳性归因于模型分不清直接因果和间接因果 — [arXiv 2312.14670](https://arxiv.org/pdf/2312.14670)。2025 年一篇生物学研究指出，LLM 预测的边可能混有直接边和祖先边，因此在与间接金标准比较时对 LLM 图取传递闭包 — [arXiv 2503.04347](https://arxiv.org/pdf/2503.04347)
- **逐边加入时检查环**：基于 BFS 的 LLM 因果图构建方法在加入每条新边前都做环检测 — [arXiv 2402.01207](https://arxiv.org/html/2402.01207v5)
- **两两提问的局限**：有来源指出，成对提问无法区分直接效应和间接效应，即使是人类专家也做不到，因此主张改用三元组提问。该说法出自检索结果中的 ICLR 2025 幻灯片，具体论文归属未核实 — [ICLR 2025 slides](https://www.iclr.cc/media/iclr-2025/Slides/30671.pdf)
- **情景链精度（Hashimoto 2014）**：对拼接出的整条链做人工判定，50,000 条中精度为 68% — [ACL Anthology P14-1093](https://aclanthology.org/P14-1093/)
- **FinCausal 2025 的片段级指标**：EM 和 SAS（语义答案相似度），不涉及图结构 — [ACL Anthology 2025.finnlp-1.21](https://aclanthology.org/2025.finnlp-1.21)

### Inferences
- 用户可以组合出一套"结构 + 语义 + 一致性"的评估指标，都可以用 networkx 计算。【可实现】
  - **连通性和估值可达率**：单份研报图中能到达"公司估值"节点的节点占比，以及弱连通分量数。这直接对应用户的"断裂子图"问题。
  - **跳级率**：层级差超过阈值的边占比，层级用 Q4 里的数据驱动 role_ratio 或拓扑深度。另外可以统计"存在 A⇝D 的多跳替代路径、却仍保留 A→D 直接边"的比例，这相当于传递约简前后的边数比。
  - **符号一致性**：同一对节点之间的多条路径符号乘积是否一致；跨研报同一边的符号一致率。
  - **忠实度**：每条边是否都有原文证据（FinCARE 的 evidence 字段），以及补全边占比。
  - **人工抽检**：仿照 Schmidt 等区分严重和轻微偏差；双人标注时用 Huang 等的图级 α（宽松、中等、严格三档都报告）。
- Huang 等发现标一跳直接原因更一致，这对用户的抽取提示词有启发：让 LLM 只抽取"本段明确陈述的直接一跳关系"，并允许它输出"中间变量未在文中说明"的标记，然后在图层面用跨研报统计补全。这比让 LLM 在抽取时自行补链更可控，也能减少 Schmidt 等所说的"模型知识覆盖原文"问题。
- 传递约简要谨慎使用。2503.04347 的做法（对 LLM 图取传递闭包后再评估）说明，抽取出的边本来就混有直接和总效应。所以 A→D 和 A→B→C→D 同时存在时，不应直接删掉 A→D，而应改标为"总效应/捷径"，同时检查符号是否与路径乘积一致；不一致时可能存在另一条机制路径。这一点是推断，没有找到专门的文献。

### Gaps
- 没有找到专门评估文本因果图"层级跳跃/中间环节缺失"的标准指标或基准数据集。
- 没有找到在中文金融因果图上做图级人工一致性评估的公开工作。
- "LLM 分不清直接和间接因果"在金融研报语境下有多严重，没有找到量化数据。
