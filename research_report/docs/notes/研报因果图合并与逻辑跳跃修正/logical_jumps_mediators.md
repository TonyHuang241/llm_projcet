# 研报因果图中“逻辑跳跃”（缺失中介的捷径边）的检测与修复方法

> 适用场景：卖方研报逐段抽取 (影响因素, 正向/负向, 被影响因素) 三元组；问题边如 `中欧贸易摩擦 → 公司估值`（实际链条为 `中欧贸易摩擦 →(-) 海外销量 →(+) 归母净利润 →(+) 公司估值`）。约束：不强加统一估值骨架；补全须有证据或数据支撑；推断出的中介要和原文明说的内容分开标注。工具栈：OpenAI 兼容 LLM API（如 GLM flash）、本地 embedding、Python、networkx。

## 1. 图论检测：传递约简、冗余边、路径符号一致性、直接与间接效应

### Takeaway
图论只能找出“已被其他路径覆盖的边”（传递约简会删掉的边），并检查这条边的符号是否等于替代路径上各边符号的乘积。它无法凭空发现缺失的中介。因果推断理论还提醒：被覆盖的边不一定是错的，它可能是真实的直接效应，也可能是对整条链的总效应概括。所以应当标注它，而不是自动删掉。

### Cited Findings
- 对有限 DAG，传递约简唯一且是原图的子图：它删除所有“能由其他路径推出”的边，保留可达关系不变。这是 Aho、Garey、Ullman (1972) 的经典结果 — [Aho, Garey & Ullman 1972, SIAM J. Comput.](https://doi.org/10.1137/0201008)
- networkx 直接提供 `nx.transitive_reduction(G)`，只接受 DAG，有环时报错。因此对研报图要先用 `nx.condensation` 把强连通分量（反馈环）缩成单点，或先断环 — [networkx transitive_reduction 文档](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.dag.transitive_reduction.html)
- Pearl 区分总效应、直接效应和间接效应（经中介传导的部分）。在因果 DAG 中，X→Y 这条边表示“不经过模型中其他变量的直接效应”。所以一旦插入中介 M，原来的 X→Y 若仍保留，含义就从“总效应”变成“绕开 M 的剩余直接效应” — [Pearl 2001, Direct and Indirect Effects](https://arxiv.org/abs/1301.2300)
- 有符号图的路径检查有现成的工程做法。INDRA 的 `signed_edges_to_signed_nodes()` 把有符号边图展开成“有符号节点图”：每个变量拆成 (x,+)/(x,−) 两个节点，每条边变成两条边。这样普通的有向路径搜索就自动满足“路径符号＝各边符号之积”。INDRA 还提供 `SignedGraphModelChecker` 检查“A 正/负向影响 B”能否由图中路径解释，默认 `max_path_length=5` — [INDRA model_checker 文档](https://indra.readthedocs.io/en/stable/modules/explanation/model_checker.html)；[INDRA explanation 模块索引](https://indra.readthedocs.io/en/latest/modules/explanation/index.html)
- INDRA 的流程是先对多个文本阅读器的抽取结果做预组装（去重、合并、按证据计算 belief，再按阈值如 0.8 过滤），然后再做路径检查。这是“多源文本抽取→去冗余→信度→路径解释”的成熟范式 — [INDRA PyPI 说明](https://pypi.org/project/indra/1.19.0/)
- 在生物领域评估 LLM 生成的因果图时，有研究先对 LLM 图和真值图都取传递闭包再比较。理由是 LLM 输出常把直接边和祖先（间接）边混在一起 — [LLMs for Zero-shot Inference of Causal Structures in Biology, arXiv 2503.04347](https://arxiv.org/pdf/2503.04347)
- 常用的 LLM 因果图评估指标（边 P/R/F1、SHD）都不考虑传递性 — [arXiv 2605.11348](https://arxiv.org/pdf/2605.11348)；[LACR, arXiv 2402.15301](https://arxiv.org/html/2402.15301v1)
- 本轮检索没有找到把传递约简用于清理 LLM 抽取因果图的专门论文（检索结论）— [检索中最接近的工作：arXiv 2608.23660](https://arxiv.org/abs/2608.23660)

### Inferences
- **检测规则 A（被覆盖的边）**：对报告图 G 中每条边 (u,v)，如果去掉它之后 u 仍能到达 v（`nx.has_path(G - e, u, v)`），就标为“被覆盖”。对全图批量计算等价于 `set(G.edges) - set(nx.transitive_reduction(C).edges)`（C 为缩点后的 DAG）。用户之前遇到的“归母净利润→公司估值”和“归母净利润→盈利预测→公司估值”并存，正属于此类。
- **检测规则 B（符号一致性）**：对被覆盖的边 (u,v,s) 和每条替代路径 p，比较 s 与 sign(p)=∏边符号。
  - 一致：大概率是对链条的概括。把它降级为 `summary` 边：可视化时默认隐藏或虚线显示，并记录 `expanded_by=[路径]`。不删除，因为原文确实这么说过。
  - 相反：两种可能。一是真实的对冲机制（如“降价→销量↑”与“降价→毛利率↓”在利润处汇合），二是抽取错误。应进入人工或 LLM 复核队列。
  - 用 INDRA 式“有符号节点图”实现最简单：图中节点为 (x, ±)，正边连 (u,+)→(v,+) 和 (u,−)→(v,−)，负边交叉相连。这样 `nx.has_path(H, (u,'+'), (v,s))` 一步就能判定同号路径是否存在。
- **不要对研报图直接做传递约简后删边**。按 Pearl 的语义，覆盖边可能是独立的直接效应（例如利润同时经 PE 直接影响估值，也经盈利预测上调影响估值）。传递约简只适合用于“推断层”，即只对模型补出的边去冗余，不碰原文显式边。
- **缺失中介本身是图论检测不到的**：如果报告里根本没有“海外销量”节点，`中欧贸易摩擦→公司估值` 在图里就是一条无替代路径的普通边。这类跳跃要靠第 5 节的节点层级/类型规则或语料级证据（第 3 节）来发现。
- 计算成本可忽略：单篇研报几十个节点，`has_path` 或传递约简都是毫秒级；带 `cutoff` 的 `all_simple_paths` 在小图上也安全。

### Gaps
- 没有找到专门讨论“文本抽取因果图中，覆盖边何时是直接效应、何时是冗余概括”的判定准则或标注数据集。目前只能依赖文本证据（原文是否单独陈述该机制）来区分。
- 模糊认知图（FCM）文献中有沿路径的间接/总效应算子，可用于符号与强度的组合。本轮没有检索到可靠的一手链接，未列入引用。

## 2. LLM 因果图补全与修正：现有框架，以及 LLM 提出中间步骤的可靠性

### Takeaway
LLM 擅长判断“A 与 B 是否有因果关系、方向如何”这类常识性问题（Tübingen 成对任务约 97%）。但在区分直接与间接关系上明显不可靠：会把约 40% 的间接（祖先）对误判为直接边，而且置信度很高。在开放式提出中介时，LLM 相对擅长中介类节点（优于源点和汇点），但输出与真值的语义相似度只有中等水平。路径越长，组合推理的错误越多。结论是 LLM 只适合在给定候选和证据的前提下做选择与核验，不适合自由生成链条。

### Cited Findings
- **成对因果方向**：Kıcıman 等报告 GPT-4 在 Tübingen 成对因果方向基准上的加权准确率为 97%，此前最佳的协方差方法为 83%。在全图发现任务上，LLM 只与近期深度学习方法相当。作者把 LLM 定位为对数据驱动方法的补充 — [Kıcıman et al. 2023, arXiv 2305.00050](https://arxiv.org/pdf/2305.00050)
- **直接边与间接边的混淆（与本问题最直接相关）**：Kumar 等 (2026) 用 12 个开源指令模型、6 个基准因果图、5 种提示策略做“是否存在直接边”的成对判别。主要结果如下 — [Kumar et al. 2026, arXiv 2608.23660](https://arxiv.org/abs/2608.23660)
  - 模型把 40.0% 的间接（祖先）非边和 36.0% 的反向非边判成直接边，其他非边的误判率为 28.2%。
  - 这些错误中，80.8% 的间接误报和 84.6% 的反向误报给出的口头置信度 ≥80%。基于 logit 的置信度常塌缩到接近 1.0。
  - LLM 判断以召回为主，生成的图过密。改提示词主要是在 P/R 之间移动，并不能消除过度预测。
  - 跨提示、跨模型的一致率作为置信度，校准更好，但经 Holm 校正后不显著。作者建议把 LLM 输出当作需外部验证的软先验。
- **成对提问在原理上就无法区分直接与间接**：Vashishtha 等指出，即使是完美的专家，逐对回答也无法区分直接与间接效应；LLM 逐对提问还会产生环 — [Vashishtha et al., Causal Order, arXiv 2310.15117](https://arxiv.org/abs/2310.15117)
  - 他们提出“三元组提问”：每对变量附带一个辅助变量，要求三元组内无环，换不同辅助变量多次提问后投票。
  - 并主张输出“因果次序”而非完整图：次序可以正确，即使边不正确。
  - 用三元组法的 Phi-3、Llama-3-8B 胜过用成对提示的 GPT-4，且次序信息降低了下游图发现和效应估计的误差。
- **LLM 假设缺失变量（中介）**：Sheth、Abdelnabi、Fritz 从已知 DAG 删去节点，让 LLM 补出缺失变量。任务包括选择题、开放式单节点假设，以及给定处理与结果时迭代地逐个提出多个中介 — [Sheth et al., arXiv 2409.02604](https://arxiv.org/html/2409.02604)
  - 开放式单节点任务（Task 3）的平均语义相似度（all-mpnet-base-v2）：GPT-4o 0.54、GPT-4 0.50、Mixtral 0.46。LLM-judge 分数分别为 0.76、0.73、0.70。
  - 按节点类型，中介和对撞节点表现较好，源点和汇点较差。
  - 迭代提中介（Task 4）只用语义相似度评估。按中介影响度 |NIE/NDE| 排序后提问通常更好。
  - 作者明确指出 LLM 无法自评合理性或置信度，需要人工监督。
  - 语义相似度会低估同义表述，例如 Dyspnea 与 Shortness of breath 相似度仅 0.57，LLM-judge 却给 9.5。
  - 截至 v2（2025-09）未见同行评审出处。
- **ALCM（数据驱动发现 + LLM 精修）**：结构学习（如 PC）产生初始图，因果包装器为每条边生成提示，LLM 精修器逐边判断有效、反转或删除，还可新增边、推测未观测变量，并给出“Very High/High/Moderate”置信标签 — [ALCM, Khatibi et al., arXiv 2405.01744](https://arxiv.org/html/2405.01744)
  - 置信度未经校准。论文承认存在幻觉风险，缓解措施只有领域知识交叉核对、人工监督和未来的 RAG。
  - 没有单独消融精修器的贡献。
  - 依赖表格数据，不适用于纯文本研报图。
- **组合因果推理随路径变长而退化**：Maasch 等 (ICML 2025) 用 SCM 评估 LLM 能否组合和分解沿路径传播的因果量（ATE、PNS）。除 o1 外，所有模型的错误都随路径复杂度上升 — [Maasch et al., arXiv 2503.04556](https://arxiv.org/html/2503.04556v4)；[PMLR v267](https://proceedings.mlr.press/v267/maasch25a.html)
- **从相关推断因果近乎随机**：Corr2Cause 含 20 万以上样本，测试 17 个 LLM，表现接近随机；对查询做扰动（分布外）后，微调模型也失效 — [Jin et al., Corr2Cause, ICLR 2024, arXiv 2306.05836](https://arxiv.org/abs/2306.05836)
- **CLadder**：按 Pearl 因果阶梯构造约 1 万道题，提出 CausalCoT 提示，结论是 LLM 的形式化因果推理仍然薄弱 — [Jin et al., CLadder, NeurIPS 2023, arXiv 2312.04350](https://arxiv.org/abs/2312.04350)
- **“因果鹦鹉”**：LLM 复述的可能是训练语料中“关于因果的相关性陈述”，而非因果推理能力 — [Zečević et al., Causal Parrots, arXiv 2308.13067](https://arxiv.org/abs/2308.13067)
- **以 BFS 方式降低查询次数**：用 LLM 构图时以广度优先扩展，把查询数从变量数的二次降到线性 — [Jiralerspong et al., arXiv 2402.01207](https://arxiv.org/abs/2402.01207)
- **检索增强构图**：LACR 用 RAG 从文献中检索因果证据来构建或校正因果图 — [LACR, arXiv 2402.15301](https://arxiv.org/html/2402.15301v1)
- **交互式中介建议**：CausalChat (TVCG 2025) 让用户对任一变量对查看 LLM 建议的方向、混杂和中介，再由人决定是否加入图中 — [CausalChat, Stony Brook](https://www3.cs.stonybrook.edu/~mueller/papers/CausalChat%20TVCG.pdf)
- **解释图 / 因果解释基准**：
  - ExplaGraphs 要求模型生成支持立场判断的常识解释图，用结构正确性（连通 DAG、格式）、语义正确性和图编辑距离评估 — [Saha et al., ExplaGraphs, EMNLP 2021, arXiv 2104.07644](https://arxiv.org/abs/2104.07644)
  - e-CARE 为每个因果问答对配一条“概念性解释”，说明为何成立，用于评估生成解释的能力 — [Du et al., e-CARE, ACL 2022, arXiv 2205.05849](https://arxiv.org/abs/2205.05849)
- **中间结构未必是真正的因果瓶颈**：2026 年一项研究发现，LLM 生成的结构化中间表示更像影响输出的上下文，而不是可靠的因果瓶颈。这是关于推理忠实性的研究，不是图补全，但提醒链式输出不保证真实性 — [Breaking the Chain, arXiv 2603.16475](https://arxiv.org/html/2603.16475v1)
- **系统动力学中的文本到因果回路图（带极性）**：与用户的 (因素, ±, 被影响因素) 格式同构。
  - System Dynamics Bot 用 LLM 从文本生成带极性的因果回路图，在两个数据集上约识别出 60% 的变量间链接与反馈回路 — [Hosseinichimeh et al. 2024, System Dynamics Review 40(3), DOI 10.1002/sdr.1782](https://vtechworks.lib.vt.edu/items/fc730454-1744-4c85-9adc-57a61b223d52)
  - sd-ai 基准测试 11 个 LLM 的因果翻译能力：gpt-4.5-preview 总分 92.9%；gpt-4o 找全了链接，但在“减少类”变量名的正极性判断上出错。这是极性错误的直接证据 — [sd-ai benchmark, arXiv 2503.15580](https://web3.arxiv.org/pdf/2503.15580)

### Inferences
- 用户之前的方案（让 LLM 从悬挂节点自由补路径到“公司估值”）出现冗余捷径和跳级，与文献结论吻合：LLM 召回导向、易把间接当直接、置信度虚高（Kumar 2026），且路径越长组合错误越多（Maasch 2025）。因此 LLM 不应担任“自由生成链条”的角色。
- 可行的 LLM 角色是“受约束的选择器与核验器”：
  1. 输入：跳跃边、原句、本报告节点表（封闭词表），以及第 3 节从语料图检索到的候选路径；
  2. 任务：从候选中选择，或回答“原文不足以分解”；
  3. 每一跳必须给出符号，以及本报告中支持该跳的句子编号（找不到则标 `background`）；
  4. 由 Python 复核各跳符号之积等于原边符号，不一致就拒绝；
  5. 多次采样（不同温度或提示）取一致率作为置信度，而不用口头置信（Kumar 2026 与 Vashishtha 2023 的投票思路）。
- “三元组提问”可以直接搬来判断直接或间接：问“在 {中欧贸易摩擦, 海外销量, 公司估值} 三者中，原文所述影响是否经由海外销量？”比逐对问“摩擦是否直接影响估值”更能区分直接与间接。
- Sheth 等的“迭代逐个提中介”适合作为候选生成器的兜底，但应限制在本报告节点表或语料规范变量表中选（封闭词表）。他们的开放式相似度约 0.5，说明自由生成的节点名常与目标不一致，会制造新的未归一化节点。
- 成本：受约束的选择/核验每条跳跃边约 1 次主调用加 2–4 次一致性采样。只对第 1、5 节规则标记出的边调用，不对所有节点对调用。成本与“标记边数”成线性关系，不会出现 BFS 或全对查询那种爆炸。

### Gaps
- 没有找到在“金融或研报文本因果图”上专门评测 LLM 中介补全准确率的工作；上述数字均来自 BNLearn 类基准图或医学、常识领域。
- 本轮未核实 Causal-Copilot（[arXiv 2504.13263](https://arxiv.org/abs/2504.13263)）的细节。按其定位它是面向表格数据的自动因果分析代理，对纯文本研报图适用性预计较低（未验证）。
- CLadder 的具体模型准确率、ExplaGraphs 与 e-CARE 的基线数字，本轮没有逐一核实。

## 3. 语料级知识图谱路径补全：用其他研报中的链条分解跳跃边

### Takeaway
最贴合“证据驱动、不强加骨架”要求的方案，是把所有研报的三元组（经节点归一化后）聚合成全局有符号图。对某篇报告里的跳跃边 (u,v,s)，在全局图中检索长度 2–4、符号积为 s、且被足够多篇报告支持的替代路径作为分解候选。这一思路有直接先例：日本财报因果链检索（Izumi & Sakaji）和哈工大金融事理图谱 / ExCAR。统计打分（支持度、条件概率、PMI、PRA 式随机游走概率）便宜、可解释，足以作为第一版。嵌入或 GNN 链接预测更适合排序和补召回，但可解释性和溯源较弱。

### Cited Findings
- **财报因果链检索（日本）**：Izumi 和 Sakaji 用线索表达式从上市公司决算短信等文本中抽取因果句，再把一个因果对的“结果表达”与另一个因果对的“原因表达”匹配连接，形成跨文档因果链。系统支持正向的波及效应检索和反向的原因检索；2021 年版基于约 2 万份财务摘要，并结合经济指标预测 — [Izumi & Sakaji, FinNLP 2019, W19-5510](https://preview.aclanthology.org/master-new-author-system-docs/W19-5510)；[Izumi, Sano & Sakaji, FNP 2021](https://aclanthology.org/2021.fnp-1.3)
- **金融事理图谱（ELG）**：哈工大构建的事理图谱以事件为节点，边为顺承、因果、条件、上下位关系；金融领域图谱有 150 万以上事件节点、180 万以上边。通过合并语义相似事件实现事件泛化 — [Ding et al., ELG, arXiv 1907.08015](https://arxiv.org/pdf/1907.08015)
- **ExCAR**：从大规模因果事件图中检索“证据事件”（即原因与结果之间的中间事件），以逻辑规则形式支撑并解释因果判断 — [Du et al., ExCAR, ACL 2021](https://preview.aclanthology.org/fix_video/2021.acl-long.183.pdf)
- **路径作为关系证据的经典方法**：PRA 用“受路径类型约束的随机游走概率”作为特征来预测两实体间关系，即用多跳路径解释直接关系 — [Lao & Cohen 2010, Machine Learning](https://doi.org/10.1007/s10994-010-5205-8)
- **规则挖掘式 KG 补全**：AnyBURL 自底向上从路径中学习 `r(X,Y) ← r1(X,Z), r2(Z,Y)` 形式的规则，带置信度，可解释 — [Meilicke et al., AnyBURL, IJCAI 2019](https://www.ijcai.org/proceedings/2019/435)
- **强化学习多跳推理**：MINERVA 在 KG 上学习从查询实体走到答案实体的路径，路径本身就是解释 — [Das et al., MINERVA, arXiv 1711.05851](https://arxiv.org/abs/1711.05851)
- **路径型 GNN**：NBFNet 用广义 Bellman-Ford 聚合两点间所有路径的表示来做链接预测，并能给出重要路径作为解释 — [Zhu et al., NBFNet, arXiv 2106.06935](https://arxiv.org/abs/2106.06935)；[代码](https://github.com/DeepGraphLearning/NBFNet)
- **KG 嵌入工具库**：PyKEEN 汇集 TransE、RotatE 等嵌入模型，可在本地训练 — [PyKEEN GitHub](https://github.com/pykeen/pykeen)
- **常识知识模型生成中间节点**：COMET 用预训练 LM 在 ATOMIC/ConceptNet 上学习生成知识三元组的尾部，属于“生成式补全”路线，会产生新节点 — [Bosselut et al., COMET, arXiv 1906.05317](https://arxiv.org/abs/1906.05317)

### Inferences
- **推荐流程（零 LLM 成本的第一层）**：
  1. **节点归一化**：用本地 embedding（如 [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3)）对变量名聚类，映射为规范变量，例如“海外销量 / 出口量 / 海外出货”。这一步决定路径能否连通，是 Izumi-Sakaji 和 ELG 共同的关键。
  2. **建全局图**：全局有符号多重图的边属性包括 `support`（支持该边的不同报告数）、`sign_votes`（正、负各多少），以及来源 (report_id, para_id, sentence)。符号争议大的边（如正负各半）不参与分解。
  3. **候选检索**：对跳跃边 (u,v,s)，执行 `nx.all_simple_paths(Gglob, u, v, cutoff=4)`。若在有符号节点图上搜 `(u,+)→(v,s)`，则天然只保留符号一致的路径。
  4. **路径打分**，组合以下几项：
     - 最弱一跳的支持度，或各跳支持度的几何平均；
     - PRA 式转移概率乘积 ∏ P(next|cur)，其中 P = 边支持 / 出度支持之和；
     - 各跳 PMI，防止只选“万能枢纽”如“归母净利润”；
     - 长度惩罚；
     - （可选）层级单调性：不回跳，见第 5 节。
  5. **报告内优先**：先在本报告其他段落的边中找替代路径（第 1 节规则 A）。找不到再用全局图，并按“行业相同 > 同公司历史报告 > 全语料”逐级放宽。这样能保留报告独有逻辑，不强加骨架，因为链条来自真实研报写法的频率。
- **统计法与嵌入/GNN 的取舍**：
  - 统计路径法：可解释，可溯源到具体报告句，零训练，适合研报规模（每篇几十条边、上万篇报告）。缺点是对未归一化节点和长尾路径召回低。
  - 规则挖掘（AnyBURL 风格）：可把“X→海外销量→归母净利润”泛化成类型级规则，如“[外部贸易冲击]→[出口/海外销量]→[利润]”，与第 5 节的节点层级结合效果好。
  - 嵌入/NBFNet：适合给候选排序、补长尾，但给出的是分数，不是原文证据。按用户“插入的中介要可区分、可溯源”的要求，只宜作为排序特征，不宜单独作为插入依据。
  - COMET 式生成：会造新节点，与“封闭词表 + 证据驱动”的目标冲突，不推荐。
- **插入后防冗余**：只对新插入的推断边做传递约简。若插入的链 u→m1→m2→v 中某跳已与报告显式边重合，就复用显式边，不新建边，从而避免再次出现“利润→估值”和“利润→盈利预测→估值”的重复。
- 计算与成本：全局图规模预计为数万节点、数十万边。对单条跳跃边做 `cutoff≤4` 的路径枚举，需要先按支持度阈值剪枝（如 support≥3），并限制枢纽节点的扩展；否则 `all_simple_paths` 可能爆炸。也可以改用 k 条最短路（`nx.shortest_simple_paths`，边权取 −log P）。

### Gaps
- 没有找到在中文卖方研报上做跨报告因果路径聚合并量化分解准确率的公开工作；Izumi & Sakaji 的准确率细节本轮未能读到全文。
- PMI 与 PRA 在因果链补全上的效果对比，没有找到直接实证。

## 4. 插入中介的核验与溯源：检索、NLI、自一致性、显式/推断标注、置信度

### Takeaway
核验应分层，并且每一跳单独核验：先在同一报告中检索支持句，用 NLI 或 LLM 判定蕴含；再看语料支持度；最后才接受“背景知识推断”。每条边带上证据等级标签和来源，置信度用多次采样的一致率加支持度，而不用 LLM 口头置信。文本事实核查（MiniCheck、AlignScore）和幻觉检测（SelfCheckGPT、CoVe）都有现成方法可借用。

### Cited Findings
- **MiniCheck**：面向“声明是否被给定文档支持”的轻量核查模型，小模型即可达到接近 GPT-4 的效果，成本低得多 — [Tang, Laban & Durrett, MiniCheck, arXiv 2404.10774](https://arxiv.org/abs/2404.10774)
- **AlignScore**：基于统一信息对齐函数的事实一致性评分，适合判断生成内容与源文本是否一致 — [Zha et al., AlignScore, arXiv 2305.16739](https://arxiv.org/abs/2305.16739)
- **SelfCheckGPT**：对同一问题多次采样，以样本之间的一致性检测幻觉，不需要外部知识库 — [Manakul et al., SelfCheckGPT, arXiv 2303.08896](https://arxiv.org/abs/2303.08896)
- **Self-Consistency**：多条推理路径采样后多数投票，提升推理准确率 — [Wang et al., Self-Consistency, arXiv 2203.11171](https://arxiv.org/abs/2203.11171)
- **Chain-of-Verification**：先起草答案，再自行规划核验问题并独立回答，最后修订，以减少幻觉 — [Dhuliawala et al., CoVe, arXiv 2309.11495](https://arxiv.org/abs/2309.11495)
- **一致率比口头置信更可靠**：对 LLM 直接边判断，跨提示和跨模型一致率的校准优于口头置信与 logit 置信（但未达统计显著）；口头置信在错误上普遍 ≥80% — [Kumar et al. 2026, arXiv 2608.23660](https://arxiv.org/abs/2608.23660)
- **多语言 NLI 模型**：可本地运行，支持中文蕴含判定 — [MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
- **按证据信度过滤的组装范式**：INDRA 先按多源证据计算 belief 再按阈值过滤 — [INDRA PyPI](https://pypi.org/project/indra/1.19.0/)
- **LLM 不能自评合理性**：Sheth 等明确指出，LLM 无法自评所提变量的合理性或置信度 — [Sheth et al., arXiv 2409.02604](https://arxiv.org/html/2409.02604)

### Inferences
- **建议的边属性模式**（满足“推断中介与原文明说可区分”）：
  - `provenance ∈ {explicit, report_supported, corpus_supported, llm_inferred}`，含义如下：
    - `explicit`：原句直接陈述，可附原文 span；
    - `report_supported`：同一报告其他句蕴含；
    - `corpus_supported`：本报告无句子支持，但 ≥k 篇其他报告有此边；
    - `llm_inferred`：只有模型推断。
  - `role ∈ {mechanism, summary}`：被分解的原跳跃边标 `summary`，并带 `expanded_by=[edge_ids]`。
  - `evidence=[(report_id, para_id, sent_id, text)]`、`support_n`、`sign_agreement`、`sc_agreement`（自一致性一致率）。
- **核验流水线（每个插入的跳 m_i→m_{i+1}）**：
  1. 用 bge-m3 在本报告句子中检索 top-k 支持句；
  2. 把“m_i 上升/下降会导致 m_{i+1} 上升/下降”作为假设，跑本地 NLI 或 MiniCheck 类模型；也可用 flash LLM 只输出 {支持, 矛盾, 无关}，并引用句号；
  3. 蕴含则标 `report_supported`；否则查全局图支持度，满足阈值标 `corpus_supported`；
  4. 都不满足就标 `llm_inferred`，可视化时用虚线或灰色显示，且默认不参与下游统计。
- **符号核验必须机器复核**：所有跳的符号乘积等于原跳跃边符号，否则拒绝该分解。sd-ai 基准显示，即便强模型也会在“减少类”变量的极性上出错（见第 2 节），所以变量命名要统一成“水平量”（如用“海外销量”而非“海外销量下滑”），否则极性含义会翻转。
- **成本**：本地 NLI 和 embedding 零 API 成本；LLM 核验只在 NLI 不确定时调用。自一致性 3–5 次采样只用于“选择哪条候选路径”这一决策，不用于每一跳。

### Gaps
- 没有找到在中文金融因果陈述上评测 NLI 或 MiniCheck 类模型准确率的公开结果；多语言 NLI 对“正向/负向共变”这类因果蕴含的判别能力需要用户自建少量标注集验证。
- 一致率与支持度如何组合成单一置信分，没有找到现成的公认公式。

## 5. 抽取时预防：让抽取器标出压缩或隐含的论断，用节点层级检测跳级

### Takeaway
在抽取阶段就让 LLM 为每条三元组附上两类信息。一是“论断强度/类型”（机制陈述、结论判断、风险提示、相关伴随，可参照科学论断强度四分类）。二是节点“层级/类型”（宏观政策与外部事件、行业、公司经营、财务、预测与估值）。这样可以低成本地标记“跨多层直连”的可疑边，而不规定具体链条。这比事后让 LLM 判断“是否直接”更可靠，因为后者正是 LLM 的弱项。

### Cited Findings
- **论断强度四分类**：Yu、Li、Wang 标注 3000 多条 PubMed 结论句，分为“无关系 / 相关 / 条件因果 / 直接因果”四类，BERT 类分类器准确率 0.90、macro-F1 0.88；用于检测把相关说成因果的过度表述 — [Yu et al., EMNLP 2019, D19-1473](https://aclanthology.org/D19-1473/)
- **因果次序比边更稳**：专家（包括 LLM）给出的因果次序比边列表更可靠；次序可以用来约束图发现，降低误差 — [Vashishtha et al., arXiv 2310.15117](https://arxiv.org/abs/2310.15117)
- **隐含论证的重构**：Argument Reasoning Comprehension Task 要求在“理由”与“结论”之间识别并重构隐含的 warrant（推理依据）。这正是“考虑到贸易摩擦→下调目标价”中缺失的机制环节 — [Habernal et al., NAACL 2018, N18-1175](https://aclanthology.org/N18-1175/)
- **警示**：后续研究发现，BERT 在该任务上的高分来自对数据中虚假线索（如 "not"）的利用，去除线索后接近随机。即“补全隐含 warrant”的高分可能并非真实推理 — [Niven & Kao, ACL 2019, P19-1459](https://aclanthology.org/P19-1459/)
- **LLM 擅长常识性的成对方向判断**（97% 加权准确率），但这不等于能区分直接与间接 — [Kıcıman et al., arXiv 2305.00050](https://arxiv.org/pdf/2305.00050)；[Kumar et al., arXiv 2608.23660](https://arxiv.org/abs/2608.23660)
- **极性与变量命名**：CLD 构造中，“减少类”变量名会导致极性判断错误 — [sd-ai benchmark, arXiv 2503.15580](https://web3.arxiv.org/pdf/2503.15580)

### Inferences
- **抽取提示的建议字段**（逐段抽取时一并输出，不增加调用次数）：
  - `evidence_span`：原文逐字片段，便于后续溯源和 NLI。
  - `claim_type`：
    - `机制`：原文含明确传导连接词，如“带动/导致/拖累/从而”，且两端是相邻环节；
    - `判断`：分析师结论，如“考虑到X，下调/维持目标价/评级/盈利预测”；
    - `风险`：风险提示中罗列的因素，无机制；
    - `伴随`：仅共变或相关。
  - `compressed`：bool，是否“明显省略了中间环节”。可选输出 `omitted_hint`（模型猜测的省略环节），但必须标为推测，只作为第 3 节检索的提示词，不直接入图。
  - 节点 `level`：L0 宏观/政策/外部事件，L1 行业供需/价格，L2 公司经营（销量/产能/单价/成本），L3 财务（收入/毛利率/费用/归母净利润），L4 预测与估值（盈利预测/EPS/目标价/估值/评级）。
- **跳级检测规则**：满足任一条件即标为候选跳跃，交给第 1、3、4 节处理：
  - 边 (u,v) 满足 `level(v) − level(u) ≥ 2`（如 L0→L4）；
  - `claim_type ∈ {判断, 风险}`；
  - `compressed=True`。
- 这套层级只用于检测，不规定链条。例如 L0→L2→L4（“贸易摩擦→海外销量→估值”）也属合理压缩，补哪些中介仍由报告内或语料证据决定，从而满足“保留报告独特逻辑”的要求。层级也可作为第 3 节路径打分中的“单调性”软约束（路径不应从 L3 回到 L1）。
- **节点层级分类成本很低**：对去重后的规范变量名做一次批量分类（LLM few-shot 或 embedding 最近邻加少量种子），结果缓存，全语料只需对“新变量名”增量调用。
- **不要依赖“LLM 判断该边是否直接”这一单问题**：成对判直接与间接，在原理上不可靠（Vashishtha），实证上误判约 40% 且过度自信（Kumar）。应由“文本线索（claim_type）+ 层级跨度 + 图上是否存在替代路径”三类可观测信号联合判定。
- **变量命名规范化**：抽取提示中要求变量名为“水平量”（如“海外销量”“毛利率”），方向由 ± 表达，避免“销量下滑”这种把方向写进节点名导致极性翻转的问题。

### Gaps
- 没有找到在金融研报上针对“机制 vs 结论判断”三元组分类的标注数据集或准确率报告；需要用户自标几百条做验证。
- 层级划分（L0–L4）是工程设计，非文献标准；文献中没有找到针对卖方研报变量层级的公认分类体系。

## 6. 金融分析师报告的论证挖掘：区分结论与机制

### Takeaway
金融论证挖掘已有分析师报告的“前提–结论（premise–claim）”标注资源，包括 NTCIR-17 FinArg（含分析师报告）和 Equity-AMSA（英文股票研报，claim/premise/scenario 加情感），以及中文资源 FinGen（研报论点生成）和 AntCritic（中文金融评论）。这些资源证明“结论（评级/目标价调整）”与“支撑前提”可以被区分标注，但它们标的是论证单元，而不是变量级因果链。可以借鉴的是标注框架，以及“隐含 warrant”概念对应的“缺失机制”。

### Cited Findings
- **NTCIR-17 FinArg-1**：任务覆盖分析师报告、业绩说明会和社交媒体。分析师报告部分把文本切分为前提和结论，并标注对各论点的情感。观察到分析师列出的正面前提多于负面前提，看多结论多于看空结论 — [Chen et al., NTCIR-17 FinArg Overview](https://repository.nii.ac.jp/record/2001323/files/01-NTCIR17-OV-FINARG-ChenC.pdf)
- **Equity-AMSA（ACL Findings 2024）**：面向英文股票研究报告，把论证单元分为 claim、premise、scenario，带情感标签，关注前瞻性陈述。作者指出专门针对研报做论证标注的数据集很少 — [Argument-Based Sentiment Analysis on Forward-Looking Statements, ACL Findings 2024](https://aclanthology.org/2024.findings-acl.820)
- **FinGen**：2,004 对中文“新闻–论点”数据，来自 2018-11 至 2020-01 证券公司分析师报告，用于从前提生成前瞻性论点 — [FinGen, arXiv 2405.20708](https://arxiv.org/html/2405.20708v1)
- **AntCritic**：约 1 万条支付宝论坛中文金融评论，约 8.8 万个 claim，标注 claim 及其支撑 premise。不是正式研报 — [AntCritic, arXiv 2208.09612](https://arxiv.org/pdf/2208.09612)
- **业绩说明会论证数据集**：Alhamzeh 等对 earnings call 做论证结构标注 — [FinArg earnings call dataset, FinNLP 2022](https://preview.aclanthology.org/fix_video/2022.finnlp-1.22.pdf)
- **社交媒体与专业报告的差异**：社交媒体用户很少为论断提供前提，专业报告更适合做前提–结论研究 — [Springer 章节, 金融论证挖掘](https://link.springer.com/chapter/10.1007/978-3-031-94687-5_2)
- **隐含 warrant 重构**是论证挖掘的既有任务，但存在虚假线索陷阱 — [Habernal et al. 2018](https://aclanthology.org/N18-1175/)；[Niven & Kao 2019](https://aclanthology.org/P19-1459/)

### Inferences
- 研报中“考虑到中欧贸易摩擦，下调目标价”用论证挖掘术语描述是：premise = 贸易摩擦，claim = 下调目标价，warrant（机制）是隐含的。用户的“逻辑跳跃”问题，本质就是隐含 warrant 的变量级重构。
- 可借鉴的做法：
  - 在抽取时把每段先切成 claim 与 premise 单元，再在 premise 内部抽机制三元组；
  - claim 单元（评级、目标价、盈利预测调整）连到 premise 的边，一律默认 `claim_type=判断`，作为第 5 节候选跳跃；
  - FinArg/Equity-AMSA 的标注规范可作为人工标注指南的参考。
- 这些数据集都没有变量级因果链的金标，不能直接用于评测中介补全准确率。

### Gaps
- 没有找到中文卖方研报的 premise–claim 公开标注数据集（FinGen 是生成任务数据，AntCritic 是论坛评论）；Equity-AMSA 的标注一致性和数据规模本轮未能读到全文（PDF 解析失败）。
- 没有找到把论证挖掘与因果图构建结合、用于补全研报隐含机制的工作。
