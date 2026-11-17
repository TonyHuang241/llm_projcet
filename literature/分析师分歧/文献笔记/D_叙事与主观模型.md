# D 组：叙事、主观模型与因果图

> 本组主题：把"人们用来推理的模型"作为研究对象的文献。
> 页码约定：文中 "p.x" 均指**本地 PDF 的第 x 页**（不是期刊页码）。本地文件多为工作论文版，版本信息在各篇开头注明，数字与正式发表版可能不同。
> 核心论文 8 篇（详写），边缘论文 3 篇（简写）。全部 11 个 PDF 都有文字层，均已读取。

---

## 一、核心论文

### Spiegler（2016）Bayesian Networks and Boundedly Rational Expectations —— `Spiegler_2016_QJE_Bayesian_Networks_Boundedly_Rational_Expectations_WP.pdf`
（本地为 2014 年 7 月工作论文版）

- **研究问题**：怎样刻画一个对环境中变量相关结构理解不完整的决策者？他的信念怎样形成，又怎样影响他的选择？
- **分歧或研报内容怎么度量**：纯理论文章。决策者的"类型"就是一张变量上的有向无环图 R。他的主观信念是把客观联合分布 p 按 R 做贝叶斯网络分解：p_R(x)=∏ p(x_i | x_R(i))（p.2）。R 是线性序时等价于理性预期，R 为空图时信念退化为各变量边际分布的乘积（p.2）。删边、反转边、断开节点、改变边方向这几种操作可以统一现有的有限理性预期模型（粗糙推理、反向因果、遗漏变量、错误归因，第 3 节）。分歧被刻画为**模型（因果结构）不同**：面对同一个客观分布，图不同的人信念也不同，作者称之为"structured belief heterogeneity"（p.6）。
- **主要发现**：
  1. 信念依赖于决策者自己的长期行为，所以最优选择只能用"个人均衡"来定义。财政政策例子中，政府把因果误认为"债务→增长"；当它对增长的权重大于 2 时，理性预期下的唯一最优动作（扩张）就不再是个人均衡（p.13–14）。
  2. 命题 5：一张 DAG 下的预测能被"理性预期 + 信息不全"解释，**当且仅当**这张 DAG 是空图或线性序（p.35）。也就是说，对其他任何图，"因果结构理解错误"和"信息不足"在预测上原则上可以区分（p.36）。
  3. 命题 6：两张互不等价、又都不是线性序的 DAG，不能按期望绩效排出高低（p.36）。
  4. "缺失数据"基础：插补程序得出的图是完美 DAG，其中边的方向无法识别，"causality cannot be inferred from purely statistical data"（p.31）。
- **分歧来自哪里**：来自模型（DAG）不同，信息和先验不在考虑之内。理论上可以区分（命题 5），文中没有用数据。
- **是否观测到推理过程**：不涉及观测。推理结构（DAG）在模型里是给定的原语。
- **作者自己承认的局限和提出的未来研究方向**：绩效排序的标准"very strong"，"It would be interesting to investigate weaker criteria"（p.37）。未来方向是借贝叶斯网络"introduce complexity of statistical inferences into positive economic modeling"（p.39）。原文**没有**提出实证检验方案，也没有谈数据需求。
  - 可从命题推出、但文中没有数据检验的预测：(a) 面对同一客观数据、因果图不同的人，信念差异是确定性的、结构化的（p.6）；(b) 只要主观图既不是空图也不是线性序，其预测偏差的模式与"信息不足"不同（p.35–36）；(c) 决策者自己的历史行为频率会改变他所感知的行动后果（p.10）。
- **与我们方法最相关的一点**：该文的 DAG 表示变量之间的条件依赖结构，边没有正负，也没有强度；信念是用 DAG 去拟合客观联合分布得到的。我们的图由研报文本直接给出，边带正负方向，没有"拟合客观分布"这一步。

---

### Eliaz & Spiegler（2020）A Model of Competing Narratives —— `Eliaz_Spiegler_2020_AER_Model_of_Competing_Narratives_WP.pdf`
（本地为 arXiv 2018 年 11 月版）

- **研究问题**：把"政治分歧源于叙事之争"形式化：什么样的叙事会占上风，它们支撑什么政策，是否导致极化。
- **分歧或研报内容怎么度量**：理论文章。叙事是一对 (p, R)。R 是一张从行动 a 指向后果 y 的 DAG，中间穿插若干选定的变量。信念用 Spiegler（2016）的分解公式拟合长期相关来形成（p.8）。节点代表变量，边代表感知到的直接因果效应，"but not the sign or magnitude of these effects"（p.3）。公众偏好同质，分歧只来自信念（p.3）。公众按"预期效用"（anticipatory utility，相信某个叙事带来的期待性满足）在"叙事—政策"组合之间做选择。均衡指这种组合的分布与历史行动频率自洽（p.10）。
- **主要发现**：
  1. 外交政策例子的唯一均衡里有两组组合：杠杆叙事 a→x₂→y 配鸽派政策，机会叙事 a→y←x₂ 配鹰派政策（Claim 1，p.11–12）。同一个变量放进不同的因果结构，可以服务相反的政策（p.4）。
  2. 命题 3：只允许完美 DAG、叙事集又足够丰富时，任何均衡恰好包含两个政策，分处理想点两侧，即极化（p.21）。
  3. 命题 2：任何完美 DAG 叙事，都存在一个节点数不更多的线性 DAG 叙事，给出同样的条件信念（p.20）。
  4. 只有一个中间变量时，被选中的中间变量要同时与期望结果和所倡导政策高度相关；"边际递减"性质使均衡唯一，并带来居中偏向（p.26–27）。
- **分歧来自哪里**：来自对同一客观数据的因果解读不同。原文："Going beyond differences in preferences or information, divergent opinions emanate from fundamentally different interpretations of reality"（p.2）。区分靠模型构造，没有数据。
- **是否观测到推理过程**：不涉及观测。叙事本身就是推理模型。
- **作者自己承认的局限和提出的未来研究方向**："the term 'narrative' is vague and any formalization inevitably leaves many of its aspects outside the scope"（p.2）。一般 n 且含不完美 DAG 的情形"almost entirely open"；有待研究的问题是 "whether our definition of equilibrium generates a force that favors narratives that involve many variables"（p.33）。作者猜想一般 n 下占优叙事是长度为 n 的线性链（p.33）。
  - 文中命题给出、但没有数据检验的预测：极化，即理想点两侧各有一个主导叙事—政策组合（p.21）；中间变量的选择规律（p.26）；同一变量在对立叙事中扮演不同角色（p.4）；主导叙事随历史行动频率轮替（p.10–11、15）；不扭曲现状的叙事可以不失一般性地取为线性链（p.20）。
- **与我们方法最相关的一点**：该文的叙事 DAG 以单一行动为根、单一后果为终点，叙事之间的差别在于选了哪些中间变量、怎样连接，边没有符号。我们的图以估值为终点，有五层结构，边带正负方向。

---

### Schwartzstein & Sunderam（2021）Using Models to Persuade —— `Schwartzstein_Sunderam_2021_AER_Using_Models_to_Persuade_WP.pdf`
（本地为 NBER WP 26109，2020 年 5 月修订版）

- **研究问题**：说服者不提供新信息，而是提供一个"模型"来解释双方都知道的数据。这种说服在什么条件下有效、能把人误导到什么程度？
- **分歧或研报内容怎么度量**：理论文章。模型是一组似然函数 π_m(h|ω)。接收者在默认模型和说服者的模型之间，选在自身先验下让已观测数据更可能出现的那个，即比较 Pr(h|m,μ₀)（p.4、p.11 式 (1)）。信息和先验固定，只变模型。作者说这与 Kamenica-Gentzkow 正好相反："ours influence by providing models, fixing the information receivers have"（p.10）。
- **主要发现**：
  1. 创业者例子（前两次失败、后三次成功）：默认模型预测下次成功概率 57%，"this time is different"模型预测 80%；数据在两个模型下的概率分别为 1.7% 和 8.3%，错误模型反而胜出（p.5）。
  2. 拟合与移动之间有权衡：模型越贴合"数据 + 先验"，越难推动信念；数据越随机，错误模型越容易胜出（p.7）。
  3. 说服者之间的竞争会推动过拟合，使接收者对证据反应不足，作者称之为"neutralize the data"（p.8）。接收者异质时，私下定制信息更有效（p.8）。
  4. 应用：技术分析中的支撑/阻力模型，使 Amazon 股价数据的似然是随机游走默认模型的 4 倍（p.33）；《Good to Great》的"this time is different"模型是默认模型的 8 倍，样本延长后反而低 25%（p.35）。
- **分歧来自哪里**：来自对同一信息的解读模型不同。数据固定这一点是建模假设，不是实证区分的结果。
- **是否观测到推理过程**：理论文章。应用部分从说服者"自己说了什么"推断他用的模型（"what persuaders themselves say"，p.38）。文中提到股票分析师用技术分析来论证（p.3），并引用调查称超过 30% 的专业投资者使用技术分析（p.32）。
- **作者自己承认的局限和提出的未来研究方向**：假设接收者"do not generate new stories themselves"（p.6）。应用时有五处要靠"analyst judgment"，例如如何区分模型与状态、怎样设定默认模型（p.38）。作者列出的待检验问题（p.38–39）：
  - 与接收者已有知识拟合差的信息，是否确实较少、较不流行？
  - 随机性大、接收者没有现成解释的场景，说服活动是否更多？
  - 揭示真实模型（监管、消费者保护）是否确实无效？
  - 利益对立的说服者竞争时，受众对数据的反应是否被"中和"？
  - 受众异质时，说服者是否在定向信息上投入更多？

  第 9 节还提出：把默认模型内生化，研究接收者何时会自己"看出规律"并过拟合；数据固定的假设抽象掉了贝叶斯说服与模型说服之间的互动（p.43）。
- **与我们方法最相关的一点**：该文的"模型"是对同一组历史数据的解释（似然函数），作者主张可以从说服者的论述中推断。我们直接从研报文本抽出分析师给出的因果解释图，但不计算这张图对数据的拟合度。

---

### Andre, Pizzinelli, Roth & Wohlfart（2022）Subjective Models of the Macroeconomy —— `Andre_Pizzinelli_Roth_Wohlfart_2022_REStud_Subjective_Models_Macroeconomy_WP.pdf`
（本地为 ECONtribute DP 119，2021 年 10 月版）

- **研究问题**：直接度量家庭和专家对宏观冲击传导的主观模型，并检验其异质性是否来自对传导渠道的选择性回忆。
- **分歧或研报内容怎么度量**：
  - **情景题（vignette）**：设计了四种冲击，即油价比当前高 30 美元、联邦基金利率上调 0.5pp、国防支出增速比往年高 2.4pp、所得税税率上调 1pp。受访者先在基线情景下预测 12 个月后的失业率和通胀率，再在冲击情景下预测，两者之差就是他认为的冲击效应（p.3–4、10–12）。给所有人相同的过去数值和冲击参数，作者称这样"shuts down information frictions … to a large extent"（p.13）。
  - **样本**：约 6,500 个美国家庭（Dynata、Lucid）和约 1,500 名专家。Wave 1 是宏观会议的作者和讨论人，共 180 人。Wave 2 嵌入 ifo 世界经济调查，共 908 人，其中 56% 来自学术界、16% 来自政策机构、16% 来自银行或私营公司（p.9）。Wave 3 是 2015–2019 年在 JEL E 类期刊发文的作者等，共 375 人，其中 40% 为博士生（p.18）。文中**没有**单独识别金融分析师。
  - **分歧度量**：一是方向分布，即预测"升 / 不变 / 降"的比例；二是预测效应的标准差，通胀效应的 σ 加权均值专家为 0.27、家庭为 0.60（表 A.5，p.69）。
  - **"模型"度量**：结构化清单，每个情景列 7–8 个传导渠道，受访者勾选做预测时想到了哪些；另有开放文本，做词组计数和双人手工编码（p.19–20、24）。**模型不表示成 DAG**，只记录被想到的渠道集合和方向性预测。
- **主要发现**：
  1. 利率上调情景：72% 的专家预测失业上升，家庭为 51%；对通胀，家庭 57% 预测上升、30% 预测下降，专家分别为 19% 和 72%（p.4）。所有情景中家庭的分歧都大于专家。
  2. 勾选的渠道对预测有解释力：R² 在家庭中为 6–21%，在专家中为 10–44%（p.25）。政府支出情景中，家庭与专家的失业预测差异可被渠道差异完全解释，利率和税收情景约能解释三分之一（p.26）。
  3. 启动实验：先让受访者回答需求相关问题，通胀预测下降 0.057pp（p<0.05）；成本启动 +0.021pp，不显著（p.28）。作者据此认为主观模型是"on the fly"形成的（p.29）。
  4. 曾在军工供应商工作、经历过 1970 年代石油危机等个人经历，与回忆的渠道和预测相关（p.30–32）。22% 的专家明确提到了经济模型（如新凯恩斯模型），家庭中没有人提到（p.23）。
- **分歧来自哪里**：设计上固定信息，把剩余的异质性归于主观模型。但作者承认 "holding different information about the state of the economy could still affect forecasts of the effect of a shock, even under the same subjective model"（p.13）。结论中说主观模型"does not fully substitute but rather complements information frictions"（p.37）。区分手段是"相同信息的情景实验 + 启动实验"。
- **是否观测到推理过程**：部分观测到。看到了"想到哪些传导渠道"和方向判断，但没有观测渠道之间如何连接，也没有观测各渠道的强度。
- **作者自己承认的局限和提出的未来研究方向**：
  - 渠道解释力被低估，原因包括 "we do not measure the perceived strength of the different channels" 以及 "we do not capture all relevant channels"（p.25–26）。
  - 结构化问题可能被用来事后合理化已做的预测（p.20）。
  - 专家内部的异质性没有研究："Our surveys are not tailored to study the drivers of heterogeneity in associations within the expert sample"（p.23 脚注 20）。
  - 自然情境下的分歧"may be even larger than measured"（p.38）。
  - 未来方向：经历和媒体如何触发不同的联想（p.32），以及推广到 TFP、情绪等其他冲击（p.38–39）。
- **与我们方法最相关的一点**：该文逐个冲击度量"冲击→通胀/失业"的方向，相当于我们的方向分歧；被想到的传导渠道集合相当于我们的中间节点或主题集合。但它不记录渠道之间的路径结构，也不记录强度。

---

### Andre, Haaland, Roth & Wohlfart（2022）Narratives about the Macroeconomy —— `Andre_Haaland_Roth_Wohlfart_2022_WP_Narratives_about_the_Macroeconomy.pdf`
（本地为 2022 年 5 月工作论文版）

- **研究问题**：2021–22 年美国通胀飙升期间，人们用什么叙事解释通胀？叙事是否影响通胀预期？媒体是不是叙事的来源？
- **分歧或研报内容怎么度量**：
  - **定义**：叙事是"causal accounts for why a specific economic event occurred"，只看向后解释的叙事，并表示成因果 DAG（p.7）。边都编码为正向因果，即"more A leads to more B"；作者说几乎没有遇到带环的叙事（p.8 脚注 2）。
  - **采集**：先告诉受访者当前通胀率，再开放式提问"Which factors do you think caused the increase in the inflation rate?"（p.11）。
  - **编码**：先定义约 15 个因子，分需求侧（政府支出、货币政策、被压抑需求、需求转移、其他需求）、供给侧（供应链、劳动力短缺、能源、其他供给）和其他（疫情、政府管理不善、通胀预期、基数效应、政府债务、加税、哄抬价格，以及 2022 年 3 月新增的俄乌战争），见表 1（p.12–14）。再由受过训练、不知研究目的的 RA 手工标注因子之间的因果边。描述性波次每条回答由两人独立编码，冲突复核。
  - **信度**：一名编码员标出某因子时，另一人也标出的概率为 87%；标出某条边时为 75%（p.16）。
  - **样本**：四波家庭调查（Lucid，每波约 1,000 人），实验在 Prolific 进行，总计超过 8,000 人。专家为 2015–2019 年在 20 本顶刊发表 JEL E 类论文的学术经济学家，共 111 人参与（摘要写的是"100"）（p.1、10–11）。**不含金融分析师。**
  - **异质性度量**：
    - "平均 DAG"：节点大小 ∝ 提及比例，边粗细 ∝ 边出现比例（p.17）。
    - 两人叙事的距离用两人边集的 Jaccard 距离 D(i,j)=1−|E_i∩E_j|/|E_i∪E_j|，再做层次聚类（平均连接，Silhouette 方法选出 k=15）（p.20）。
    - 复杂度：因子数、边数、因果链长度。
    - 把因子虚拟变量对个人特征回归（p.22）。
- **主要发现**：
  1. 专家的 DAG 平均有 4.3 个因子、3.6 条边，家庭为 3.5 个因子、2.8 条边（p<0.001）。提到供需任一侧的人中，同时提到两侧的比例专家为 77%、家庭为 34%（p.17–18）。"政府管理不善"出现在 32% 的家庭叙事中，专家仅 1%；"哄抬价格"家庭 8%、专家 0%（p.19）。
  2. 倾向共和党的受访者归咎政府管理不善的比例高 37pp，倾向民主党者以疫情为根源的比例高 22pp（p.22）。2022 年 3 月有 28% 的叙事提到俄乌战争（p.24）。
  3. 叙事与预期相关：提到"管理不善"者 12 个月通胀预期高 1.207pp，"被压抑需求"者 5 年预期低 0.606pp（表 3，p.27）。实验中给出"被压抑需求"叙事，通胀预期下降 0.712pp（表 4，p.30）。
  4. 叙事改变人们解读同一信息的方式：先给"政府支出"叙事的组，收到"高支出"预测后通胀预期上升 1.786pp；先给"能源"叙事的组只上升 0.344pp，不显著。IV 弹性分别为 0.378 和 0.051（表 6，p.36）。媒体实验：被激励读新闻的人，叙事中的因子数增加 0.287（表 7，p.43）。
- **分歧来自哪里**：明确指向解读/模型。原文："Different narratives make people draw different conclusions from the same evidence"（p.5），以及家庭"systematically disagree about why a given economic state has been reached"（p.44）。区分方式：2×2 实验固定信息（相同的支出预测），只变叙事（p.33–35）。
- **是否观测到推理过程**：直接观测到。受访者给出的因果解释被编码成 DAG，可以比较不同人群的因子和边。
- **作者自己承认的局限和提出的未来研究方向**：
  - 用人工编码是因为 "artificial intelligence methods still have difficulties detecting (the often implicit) causal structure in human language"；同时承认人工判断有主观性，"in light of the inherent ambiguities of language"（p.13）。
  - 实验只在描述性波次做双人编码，其余实验单人编码（p.13 脚注 9）。
  - 注意力实验对预期的影响 "do not exclusively operate through changes in narratives"（p.32 脚注 20）。
  - 少数解释（移民、气候等）未纳入编码表（p.19 脚注 11）。
  - 只研究向后看、针对单一事件的叙事（p.7）；DAG 只编码正向边（p.8）。
  - 未来方向："investigate economic narratives in other contexts and countries, study which features make narratives popular and persuasive, or explore how heterogeneous narratives feed back into macroeconomic outcomes"（p.44）。
- **与我们方法最相关的一点**：该文用两人边集的 Jaccard 距离度量叙事差异。边没有权重，也不区分正负（全部编码为正向）；图的终点是单一结果变量"通胀"；因子集是研究者预先定义的约 15 个。

---

### Ash, Gauthier & Widmer（2024）RELATIO: Text Semantics Capture Political and Economic Narratives —— `Ash_Gauthier_Widmer_2024_PolAnalysis_Relatio_Text_Semantics_Narratives_WP.pdf`
（本地为 arXiv v3，2022 年 4 月版）

- **研究问题**：怎样无监督地从大规模文本中抽取"谁对谁做了什么"式的叙事结构，并把它们组成网络？
- **分歧或研报内容怎么度量**：
  - **抽取**：用语义角色标注（AllenNLP）把句子拆成"施事—动词—受事"（p.4）。施事和受事做降维：频率最高的 L 个命名实体直接保留，其余短语用加权平均词向量编码后做 K-means 聚类（p.6）。
  - **动词不聚类**：因为嵌入聚类"frequently assigns antonyms (e.g., 'decrease' and 'increase') to the same cluster"，只统一时态，否定用 "not-" 前缀表示（p.6–7）。
  - **组图**：叙事陈述属于 E×V×E，多条陈述组成有向多重图，节点是实体，边是动词（p.7）。
  - **数据**：美国国会记录 1994–2015，1,730 万个句子，保留 1,000 个命名实体和 1,000 个聚类实体、出现至少 50 次的完整叙事，得到 1,638 个独特叙事（p.9）。
  - **异质性度量**：每条叙事的党派对数优势比；实体的"叙事分裂度"= 该实体所在叙事的平均 |log-odds| 减去实体自身的 log-odds（p.12）；再用叙事特征或实体特征分别预测说话人的党派（p.15）。
- **主要发现**：
  1. 叙事随事件变化：例如 9/11 后 "God bless America" 激增，2002 年关于萨达姆威胁的叙事出现高峰（p.11）。
  2. 用叙事特征预测党派的样本外准确率为 81%，用实体特征为 74%；交叉验证均值分别为 79% 和 73%（p.15）。
  3. 同一实体在两党叙事中连接不同，例如民主党说 "oil companies making profits"，共和党说 "Americans rely on oil"（p.15）。
  4. 叙事网络中接近中心度最高的五个实体是 job、family、healthcare、funding、program（p.15）。
- **分歧来自哪里**：没有讨论。这是描述性方法论文，只显示党派差异体现在对同一实体的不同连接上。
- **是否观测到推理过程**：观测到文本中明说的"主体—动作—客体"关系，但边是动词，不是因果关系；方法不区分一句话是不是因果陈述。Schmidt 等（2025）批评它"the causal component … is not considered"（Schmidt 等 p.6）。
- **作者自己承认的局限和提出的未来研究方向**：
  - 抽取质量依赖语义角色标注；如果复杂句更容易出错，而不同群体的用语复杂度不同，会造成选择性样本偏差（p.20）。
  - 聚类太粗时 "'tax hike' and 'tax cut' – will often be combined into the same entity"；实体抽取"is art as much as science"（p.20）。
  - 目前只支持英语（p.21）。
  - 与其他文本特征的系统比较"is a high priority for future work"（p.19）。
- **与我们方法最相关的一点**：该文叙事图的节点是实体、边是动词，没有因果方向，也没有正负含义（只有否定前缀）。我们的图边是因果关系、带符号，并按层级流向估值。

---

### Schmidt, Lange, Reccius, Müller, Roos & Jentsch（2025）Identifying economic narratives in large text corpora —— `Schmidt_etal_2025_arXiv_Identifying_Economic_Narratives_with_LLMs.pdf`
（本地为 arXiv 2506.15041v1）

- **研究问题**：按严格的叙事定义，GPT-4o 能否从报纸文本中抽取经济叙事？与专家标注相比如何？
- **分歧或研报内容怎么度量**：
  - **定义**：叙事是"causal connections between two temporally and semantically distinct events"，形式为 "A causes B" 或 "A is caused by B"（p.5）。
  - **代码本要点**：只编码正向因果（"Positive Causality Only"）；分叉和链条拆成单条；要做指代消解（p.12–13）。
  - **语料**：WSJ 与 NYT 1985–2023 年含 "inflation" 等关键词的文章，取命中句及前后各两句（p.10–11）。
  - **金标准**：随机抽 100 篇，三名专家独立标注后讨论一致，得到 291 条叙事（p.14）。其中 20 篇用于选 few-shot 示例和验证，其余用于测试；最终采用 7-shot 加思维链（CoT）提示（p.17–18）。
  - **后处理**：事件按嵌入与词典归入主题簇，升降用 ↑/↓ 表示，例如 "↑monetary policy → ↑inflation"（p.23–24）。
  - **比较维度**：把专家之间的偏差当作"预期偏差"基准，计算主要/次要偏差率、准确率和 Jaccard 相似度。
- **主要发现**：
  1. 主要偏差率：三名专家分别为 0.40、0.35、0.49，模型为 1.25。相对金标准的准确率：专家 0.72、0.74、0.67，模型 0.44。Jaccard：专家 0.59、0.60、0.59，模型 0.40（表 2，p.27）。
  2. 模型每篇平均抽出 2.32 条（标准差 1.15），专家的标准差约 2.03（p.25–26）。模型难以拆开分叉和链条，会把它们压成一条笼统陈述（p.26–27）。
  3. 聚合后，80 篇测试文档产生 409 个独特叙事；最常见的完整叙事只出现 2–4 次，例如"↑货币政策 → ↑通胀"出现 4 次（p.29–30）。
  4. "政府支出"91% 的时候作为原因出现，"股市"90% 的时候作为结果出现（p.29）。
- **分歧来自哪里**：没有研究经济主体之间的分歧。但发现专家标注者之间差异明显，"pointing to the inherent subjectivity involved in narrative sense-making"（p.3）。
- **是否观测到推理过程**：抽取的是新闻文本中陈述的因果链（媒体叙事），不是个人层面的推理模型，也没有按作者或文档做叙事之间的比较。
- **作者自己承认的局限和提出的未来研究方向**：
  - "Even among expert annotators, we observed substantial variation in annotator opinion about which narratives were identified, how events were segmented, and how causal relations were interpreted"；"fully automating the process remains an ambitious goal"（p.32）。
  - 抽象化会损失信息（例如实际利率与名义利率不分），偶尔产生"rising interest rates cause higher interest rates"这类自指叙事（p.23–24）。
  - 商业模型版本下线后结果无法复现（p.10）。
  - 未来方向：把抽出的叙事与通胀预期、消费者信心、市场波动等指标挂钩（p.33）。
- **与我们方法最相关的一点**：该文的"事件 A 导致事件 B + ↑/↓ 方向 + 主题归并"，在形式上与我们的带符号因果边加主题归类最接近。但它只抽单条二元因果，不组装成多层图，也不在文档之间比较分布。

---

### Bybee, Kelly & Su（2023）Narrative Asset Pricing: Interpretable Systematic Risk Factors from News Text —— `Bybee_Kelly_Su_2023_RFS_Narrative_Asset_Pricing.pdf`
（本地为 RFS 正式版）

- **研究问题**：能否从《华尔街日报》新闻叙事中估计 ICAPM 状态变量，并得到可解释的定价因子？
- **分歧或研报内容怎么度量**：
  - **"叙事"的定义**：就是 LDA 主题，沿用 BKMX 的 180 个主题。每日"叙事注意力"θ_τ 是当日文章主题比例的按词数加权平均，z_τ 是它的创新（p.7–8）。
  - **模型**：设 z_τ = A x_τ + η_τ，把个股收益与 z_τ 的协方差作为 IPCA 的工具变量，用 Sparse IPCA（group lasso）筛选叙事（p.6、9）。
  - **数据**：WSJ 1984:01–2017:06（p.10），样本外期间 2001–2016（p.16）。
  - 不涉及个人之间的分歧，只有市场或媒体层面的单一序列。
- **主要发现**：
  1. 样本外均值-方差有效（MVE）组合的 Sharpe 比率为 1.3，Fama-French 五因子加动量为 0.8；对 78 个异象组合的定价误差更小（p.4）。
  2. MVE 约由十几个叙事线性组合而成。"Recession"的负影响最大，"Record high"和"Optimism"是主要的正向叙事（p.4）。
  3. MVE 正向预测市场收益、GDP 增长，负向预测信用利差和失业，符号与 ICAPM 的约束一致（p.4）。
- **分歧来自哪里**：不涉及。
- **是否观测到推理过程**：没有，只观测到媒体在各主题上的注意力份额。原文："We do not stipulate the causal mechanism for how the two are linked"（p.7）。作者认为新闻里"the hard cognitive work—inferring causes of business events and predicting their subsequent effects—is partially done by the journalists"（p.2），但方法本身不抽取因果关系。
- **作者自己承认的局限和提出的未来研究方向**：原文没有专门的局限讨论，只提到新闻有"sensationalized or biased, or irrelevant"的激励（p.3）。结论称新闻文本是"promising source"（p.27）。没有涉及"无法观测推理"之类的表述。
- **与我们方法最相关的一点**：该文的"叙事"等于主题注意力分布，相当于我们的主题分量，而且是媒体整体的而不是单个分析师的；不包含主题之间、或主题到财务科目的关系。

---

## 二、边缘论文

### Shiller（2017）Narrative Economics —— `Shiller_2017_AER_Narrative_Economics_WP.pdf`
（NBER WP 23075，AEA 主席演讲）

- **研究问题**：主张研究流行叙事怎样像流行病一样传播，以及它们对经济波动的作用。
- **度量**：把叙事定义为"a simple story or easily expressed explanation of events"（p.4）。用 SIR 流行病模型做类比（p.13），用 ProQuest 新闻和 Google Ngrams 的短语频率（如"Laffer Curve"）说明叙事呈驼峰形扩散（p.21）。其余以历史叙述为主。
- **分歧来源 / 推理过程**：没有区分；只有词频，不含因果结构。
- **局限**：
  - 没有对照实验，"We cannot easily prove that any association … is not all reverse causality"（p.3）。
  - 用词频或 n-gram 计数研究叙事有局限（p.8）。
  - "Researchers have trouble dealing with a set of often-conflicting narratives and gradations and superposition of them"（p.49）。
  - 建议由多名 RA 按明确指令阅读、分类叙事（p.50）。作者自 1989 年起在股价、房价问卷中收集开放式回答（p.48–49）。
- **与我们方法最相关的一点**：该文用关键词或短语频率代表叙事，不包含因果关系和结构。

### Flynn & Sastry（2024）The Macroeconomics of Narratives —— `Flynn_Sastry_2024_NBER_Macroeconomics_of_Narratives.pdf`
（NBER WP 32602）

- **研究问题**：会传染的叙事对经济周期有多大影响？
- **度量**：理论上，叙事是"subjective and potentially incorrect models of the world"，形式化为一组可供选择的先验分布（p.3），模型中只有乐观和悲观两种（p.4）。实证上，用 Loughran-McDonald 情绪词典对 10-K 和业绩电话会文本打分，得到"叙事乐观度"；另有两种更细的度量：与 Shiller 的"Perennial Economic Narratives"的文本相似度，以及 LDA 主题（p.5）。
- **主要发现**：乐观度对招聘的局部均衡效应为 3.6pp。乐观度负向预测未来收益和盈利，并预示管理层的销售指引会高于实际值，作者据此认为它不是基本面消息。乐观会在企业之间传染（p.5–6）。量化结果：叙事解释了 2000 年代初衰退 32%、大衰退 18% 的产出下降，以及 19% 的产出方差（p.2、7）。
- **分歧来源 / 推理过程**：分歧来自所持叙事（先验）不同；用"乐观度不预测基本面"来排除"信息"解释（p.6）。只观测到情绪或主题，不包含因果结构。
- **局限 / 未来**：没有研究家庭的叙事；还要研究"what 'makes a narrative a narrative'"，即叙事集合和传染性的微观基础（p.48）。
- **与我们方法最相关的一点**：该文的"叙事"在实证上是企业文本的情绪极性或主题，相当于我们的方向性或主题分量，不包含关系。

### Bybee, Kelly, Manela & Xiu（2024）Business News and Business Cycles —— `Bybee_Kelly_Manela_Xiu_2024_JF_Business_News_and_Business_Cycles_WP.pdf`
（NBER WP 29344，2021 年 10 月版）

- **研究问题**：能否用商业新闻文本度量经济状态？
- **度量**：对 1984–2017 年约 80 万篇 WSJ 文章估计 LDA 主题模型，按贝叶斯因子准则选出 180 个主题；用各主题的新闻注意力占比作为经济状态的度量（p.2–4）。
- **主要发现**：新闻注意力解释了 25% 的股市收益波动（p.5）。"recession"主题的注意力在控制常见 VAR 变量后，仍能显著预测未来产出和就业（p.5–6）。还能检索出影响模型预期最大的文章（p.6）。
- **分歧来源 / 推理过程**：不涉及个人或群体之间的分歧。作者明说与指标相关的主题关键词"are not causal narratives"（p.5）。
- **局限**：本次阅读范围内（摘要、引言）没有看到专门的局限讨论，未能确认全文是否有。
- **与我们方法最相关的一点**：该文的"叙事"是新闻整体在各主题上的注意力份额，相当于主题分布，不包含主题间的因果关系。

---

## 三、本组四个关注点对照（仅汇总上文事实）

| 文献 | "叙事/模型"的定义 | 如何度量 | 人与人之间异质性怎么度量 | 样本是否含分析师/专家 |
|---|---|---|---|---|
| Spiegler 2016 | 变量上的 DAG（贝叶斯网络分解） | 理论 | 不同 DAG 对同一分布给出不同信念 | — |
| Eliaz & Spiegler 2020 | 从行动到后果的因果 DAG（边无符号） | 理论 | 均衡中同时存在的叙事—政策组合 | — |
| Schwartzstein & Sunderam 2021 | 解释数据的似然函数 | 理论；应用中从说服者的话推断 | 不同模型对同一数据的拟合度 | — |
| Andre 等 2022（主观模型） | 对冲击传导的判断，即方向加想到的渠道 | 情景题差分；勾选渠道；开放文本手工编码 | 方向分布、预测标准差、渠道勾选率 | 学术、政策机构、银行/私营专家；未单列金融分析师 |
| Andre 等 2022（叙事） | 解释单一事件的因果 DAG（边只编码为正） | 开放回答 → 人工编码因子与边 | 边集 Jaccard 距离 + 聚类；平均 DAG | 111 名宏观学者；无分析师 |
| Ash 等 2024 | 施事—动词—受事三元组 | 语义角色标注 + 实体聚类 | 党派对数优势比、实体分裂度 | — |
| Schmidt 等 2025 | 两事件间的二元因果（"A causes B"） | GPT-4o 7-shot CoT，以专家金标准对照 | 标注者之间的偏差（不是主体之间） | 专家只作为标注者 |
| Bybee 等 2023 / BKMX / Flynn & Sastry / Shiller | 主题注意力 / 情绪 / 词频 | LDA、词典、Ngram | 无个体层面度量（F&S 有企业层面的乐观度） | — |

理论文章中提出、但缺数据检验的预测，集中在 Spiegler 2016 的命题 5、Eliaz & Spiegler 2020 的极化与中间变量选择，以及 Schwartzstein & Sunderam 2021 第 7.4 节列出的五个问题（详见各篇的"局限"条目）。
