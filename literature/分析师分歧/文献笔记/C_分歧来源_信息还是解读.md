# C 组：分歧的来源——信息不同，还是对同一信息的解读/模型不同

本组 9 篇均为核心论文。版本说明（页码均按所读 PDF）：
- 已发表版：Harris & Raviv (1993, RFS 期刊页码)、Banerjee & Kremer (2010, JF 期刊页码)、Banerjee (2011, RFS 期刊页码)、Giannini, Irvine & Shu (2019, JFM 期刊页码)。
- 工作论文版（页码为 WP 自身印刷页码，数字可能与正式发表版不同）：Bamber, Barron & Stober (1999 年 1 月 WP)、Andrei, Carlin & Hasler (NBER WP 20190, 2014)、Cookson & Niessner (2016 年 6 月 WP)、Cookson, Engelberg & Mullins (2020 年 9 月 WP)、Antweiler & Frank (2002 年 1 月 WP)。
- Harris & Raviv 为扫描件，按页图像阅读。

---

### Harris & Raviv（1993）Differences of Opinion Make a Horse Race —— `Harris_Raviv_1993_RFS_Differences_of_Opinion_Make_a_Horse_Race.pdf`

- **研究问题**：投机市场中，交易者接收相同公开信息但解读不同时，能否在一个框架内解释量价关系和量、价的时间序列性质（p.473–474）。
- **分歧或研报内容怎么度量**：纯理论。终值 R 取 H 或 L，各半概率；每期发布公开信号 s_t，真实似然为指数形式 σ⁰_H(s)=k₀a₀^s（s≥0）、k₀b₀^(−s)（s<0），后验只依赖累积信号 m=s₁+…+s_t，θ₀=b₀/a₀ 是信号质量的反向指标（p.479–480）。两组风险中性的投机者有**共同先验**，各自使用**同函数形式、不同参数**（a_j, b_j, θ_j）的似然函数，θ₁<θ₂：第 1 组（"responsive"）认为信号质量更高、反应更大，第 2 组（"unresponsive"）反应更小（p.480–481）。两组对信号方向（利好/利空）看法一致，只在"程度"上分歧（p.475）。各自模型为共同知识，彼此认为对方模型错误，不从价格学习（p.480）。
- **主要发现**：(1) 绝对价格变化与投机性成交量正相关（定理 1，p.483）；(2) 平均预测（两组期望的平均）的绝对变化与成交量正相关（定理 2，p.486），作者将其对应到 Ziebart (1990) 发现的成交量与分析师平均盈余预测绝对变化正相关（p.485）；(3) 若定价组高估信号质量，相邻价格变化负自相关，低估则正自相关（定理 3，p.486）；成交量正自相关，休市后开盘成交量更大（p.475）。(4) 由于风险中性和卖空限制，只有累积信号由正转负或由负转正、两组"换位"时才发生交易（p.475, 481）。
- **分歧来自哪里**：明确假设是"解读不同"。作者写道投机交易源于对"公告与资产最终表现之间关系"的分歧，这种分歧"can arise either because speculators have different private information or because they simply interpret commonly known data differently"（p.474），本文选择后者。脚注 15 强调两组信息完全相同，"One group simply *believes* the signal is more informative than the other group does"（p.481）。结论部分论证：若只有先验不同（共享似然、仅对状态到终值的映射看法不同），除最后一期外不会发生交易，"to obtain sequential trading over time, one must assume that speculators disagree either about likelihoods or about the fundamental states of nature"（p.491）。无实证，不涉及数据上的区分。
- **"模型"的具体含义**：模型 = 对"信号→终值"关系的似然函数；具体实现为同一函数形式下的不同参数，即对信号质量（θ_j）看法不同。
- **公开信息之后收敛还是扩大**：图 1 中两组后验在 m=0 处重合，随累积信号偏离 0 而分开（p.482），即同一公开信号会使信念差距扩大；成交只发生在信念交叉（换位）时。
- **是否观测到推理过程**：理论文章，不涉及观测；模型中各方似然函数被设定为共同知识。
- **作者自己承认的局限和提出的未来研究方向**：结果"were proved using specific functional forms, we believe them to be robust"，部分推论"remain to be tested"（p.490）；脚注 13 承认忽略了投机者对真实模型的不确定及学习，理由是真实模型变化很快（p.480）；风险中性假设导致"全有或全无"的交易，第 3 节在风险厌恶的简化模型中检验了稳健性（p.490）。未提及分析师模型在数据上无法观测的问题。
- **与我们方法最相关的一点**：该文中的"模型分歧"只是同一似然函数里一个参数（信号质量 θ）的差异，分歧不涉及关注哪些因素或经由何种路径传导。

---

### Bamber, Barron & Stober（1999）Differential Interpretations and Trading Volume —— `Bamber_Barron_Stober_1999_JFQA_Differential_Interpretations_and_Trading_Volume_WP.pdf`

- **研究问题**：对公开信息的"不同解读"是否显著解释交易量（WP p.1）。
- **分歧或研报内容怎么度量**：采用 Kandel & Pearson (1995) 的 KP 指标。数据：I/B/E/S 1984–1994 年，季度盈余公告前 45 天内发布、公告后 30 天内同一分析师修正的**年度** EPS 预测，每个公告至少 5 名分析师，共 1,972 个公告（WP p.5, 9）。对所有分析师两两配对，KP = 满足以下两条的配对比例：(a) 修正方向相反（一升一降），且 (b) 两条预测交叉（flip）或距离拉大（diverge）（WP p.2–3, 8）。理由：这类修正与"所有人似然函数相同"的贝叶斯更新矛盾（WP p.2）。因变量为公告日 [−1,+1] 换手率 VOL；控制平均预测绝对变化 ABFC、市场成交量 MKTVOL、市值 MVE、绝对收益 ABRET（WP p.6–8）。
- **主要发现**：(1) 价格变化最小十分位子样本（N=197）中 ln(KP) 系数 0.031，p=0.02，KP 增加 1 个标准差对应成交量增加 2.5%（WP p.11）；反向回归隐含系数 1.578，作者认为 0.031 是下界（WP p.12）。(2) 全样本 ln(KP) 系数 0.005，p=0.20，不显著（WP p.13, 表 2）。(3) 公告期成交量高于公司非公告期均值的 1,233 个公告中 ln(KP)=0.018，p<0.01；低于均值的 739 个中不显著（WP p.15, 表 3）。(4) KP 均值：小价格变化样本 0.147，全样本 0.124（WP p.9）。
- **分歧来自哪里**：文章明确区分两种来源（WP p.2）：差异化的信念修正可来自 (1) 对新闻的不同解读，或 (2) 披露前私有信息精度不同（即便解读相同，精度高者对新闻赋权更低）。用图 1 的五种情形区分：情形 2、3（同向修正中出现交叉或发散）可由精度不同产生，**不计入**；只有情形 4、5（反向修正且交叉/发散，平均信念基本不变）"arise only from differential interpretations"（WP p.3）。所以 KP 是"解读不同"的保守下界指标。
- **"解读不同"的代理变量与作者承认的不足**：代理变量是分析师年度预测修正方向相反且交叉/发散的配对比例。不足包括：(1) 保守，"it understates the true (unobservable) incidence of differential interpretations"（WP p.4）；情形 2、3 中的解读差异被漏掉，因为"it is not possible to disentangle the extent to which Cases 2 and 3 are attributable to differential interpretations vis-a-vis differential precision of predisclosure information"（WP p.4）。(2) 测量误差随总体信念变化增大而增大，因此先用小价格变化样本（WP p.4）。(3) KP 只基于部分市场参与者（分析师）的预期；且"computed from analysts' expectations about annual earnings rather than investors' expectations about future cash flows"（WP p.12）。(4) 分析师少时 KP 取值离散，所以要求至少 5 人（WP p.10）。(5) 模拟显示即使解读差异是成交量的唯一来源，ln(KP) 平均也只解释约 10% 的 ln(VOL) 变动（WP 附录 p.25）。
- **"模型"的具体含义**：附录沿用 Kandel & Pearson，把"不同解读"刻画为两类交易者对公开信号误差均值的预期不同（μ₂−μ₁），即对同一信号"看到"不同的值；对信号精度看法不同（b₁≠b₂）也算一种解读差异，但模拟中设为相同（WP 附录 p.22–23）。
- **公开信息之后收敛还是扩大**：KP 本身就是盈余公告后分析师预测反向修正并交叉/发散的比例，样本平均约 12–15% 的配对出现这种情况（WP p.9）；作者也指出重大意外后分析师预测往往同向移动（WP p.9）。
- **是否观测到推理过程**：没有。只观测到预测数值的修正方向和幅度，解读差异是从数值模式中推断出来的。
- **作者自己承认的局限和提出的未来研究方向**：见上文测量误差部分；全样本中控制变量不完美（WP p.13）。未提出具体的未来研究方向。
- **与我们方法最相关的一点**：该文从同一盈余公告后两位分析师预测值的反向交叉/发散来推断"解读不同"，没有观测解读的内容（依据哪些因素、如何推导）。

---

### Banerjee & Kremer（2010）Disagreement and Learning: Dynamic Patterns of Trade —— `Banerjee_Kremer_2010_JF_Disagreement_and_Learning_Dynamic_Patterns_of_Trade.pdf`

- **研究问题**：投资者对公开信息解读不同时，分歧过程有哪些性质会产生实证中观察到的成交量与波动率动态（p.1269）。
- **分歧或研报内容怎么度量**：理论加初步实证。两个 CARA 投资者，终值 D=F+d；每期公开信号 s_t，投资者 i 认为 s_t=F+ε_t，ε_t~N(e_{i,t}, q_t)，e_{i,t} 是 i 对该信号的"解读"（e 越大看得越负面），e_{i,t}~N(0, λ_t) 随机，双方都能看到对方的 e，因此没有信息不对称（p.1275）。双方对信号**精度**看法一致，只在**均值**上不同（p.1275）。正文设共同先验，分歧完全来自解读（p.1274，脚注 8）。实证部分：CRSP 1980 年起个股日换手率，按"异常换手率/年内标准差"分十组，看成交量自相关、波动率和未来 10 日收益（p.1291）。
- **主要发现**：(1) 均衡价格等于平均估值，持仓与估值差成正比（引理 1，p.1278）。(2) 成交量 = |信念收敛项 + 特质项|：对当期信号解读一致但对过去信息解读不同时，贝叶斯更新使信念收敛，产生"belief-convergence trade"；对过去一致而对当期信号解读不同时，产生"idiosyncratic trade"（命题 1，p.1270–1271, 1280）。(3) 偶发但重大的分歧使成交量正自相关、成交量与波动率正相关（p.1270）。(4) 预测：异常成交量高时，成交量自相关、波动率和平均收益随之上升；高波动时成交量自相关随成交量上升，低波动时下降；盈余公告日成交量与波动率更高、之后成交量自相关更高（预测 1–3，p.1291），十分组图形与预测 1、2 一致（p.1291–1293）。
- **分歧来自哪里**：假设为解读不同（"agree to disagree" even if they have the same information，p.1270）。文中说明 DO 模型突出的是"aspects of the world that investors still disagree about after they have learned all they can from each other"（p.1275）。实证没有直接区分信息和解读，"Since disagreement across investors is a latent variable, we need to construct a proxy for it"（p.1290），实际用异常成交量和波动率间接代理。
- **"模型"的具体含义**：对公开信号噪声均值（偏差）的主观看法 e_{i,t}，即每个人从同一信号中扣除的偏差不同；这种解读本身是每期的随机冲击。
- **公开信息之后收敛还是扩大**：两种情况都有。当期解读一致时，公开信息使分歧收敛（收敛交易）；当期解读冲击大时，分歧扩大。作者把盈余公告视为"likely to be associated with large jumps in disagreement"（p.1291），引用 Chae (2005)：公告日成交量跳升后逐日衰减（p.1292）。
- **是否观测到推理过程**：没有，模型中解读是外生的随机变量；实证只用成交量和收益。
- **作者自己承认的局限和提出的未来研究方向**：每个投资者都认为对方是错的，这一设定"is made primarily for tractability"（p.1275）；"We are unable to allow for both heterogeneous precision and stochastic disagreement about the mean of the public signal in a tractable manner"（p.1275）；实证检验只是"preliminary"（p.1291）。引用 Brav & Heaton：DO 模型与行为偏差模型"often observationally and mathematically equivalent"（p.1272）。未来方向：交易成本和卖空限制、"the role of different types of information (e.g., 'soft' vs. 'hard') in generating disagreement"、IPO 与并购等高分歧事件、对分歧的学习与投机（p.1294）。
- **与我们方法最相关的一点**：该文把每期分歧拆成"对过去信息的分歧（会收敛）"和"对当期信号的新分歧"两部分，但每个人的解读只是一个标量偏差，不含解读的结构。

---

### Banerjee（2011）Learning from Prices and the Dispersion in Beliefs —— `Banerjee_2011_RFS_Learning_from_Prices_Dispersion_in_Beliefs.pdf`

- **研究问题**：投资者是否用价格更新信念？在同时嵌套理性预期（RE）和意见分歧（DO）的动态模型中，用分歧与收益-成交量特征的相关符号来区分两者（p.3025–3026）。
- **分歧或研报内容怎么度量**：理论：OLG 模型，投资者 i 收到关于下期股利冲击的私有信号 Y_i=δ+s_i；对**他人信号**的信息含量有异质先验，参数 ρ∈[0,1] 表示"他人信号有多大信息量"：ρ=1 为 RE（有效利用价格），ρ=0 为 DO（只用自己的信号）（p.3031）。分歧定义为"the variance in the investors' equilibrium expectations about the asset's payoffs"（p.3027）。实证：I/B/E/S 年度 EPS 预测离散度 AFD（未调整）、AFD/|中位预测|、AFD/滞后股价，以及平均换手率 VOL 作为分歧代理；CRSP，1983.1–2007.9，297 个月，每期 2,000 多家公司，剔除股价<5 美元或分析师<2 人（p.3050–3051）。方法为五分位组合排序和 Fama–MacBeth 回归。
- **主要发现**：(1) RE 下分歧与预期收益、波动率、贝塔、|收益|-成交量相关性正相关，与收益自相关负相关；DO 下符号全部相反；两种情况下分歧都与成交量正相关（p.3026）。(2) 四个代理下波动率、贝塔、|收益|-成交量相关均随分歧上升，例如贝塔 Q5−Q1：AFD 0.145（t=10.99）、VOL 0.560（t=30.97）（表 3，p.3053），总体更支持 RE。(3) 平均收益结果不一致：AFD 下 Q5−Q1=−1.20%（t=−2.62），VOL 下 +1.85%（t=3.38），另两个代理不显著（表 3，p.3053）。(4) 按"成交量与波动率的相关性"把股票分十组（负相关代表 DO），偏 RE 一端的公司更大、M/B 更高、分析师覆盖多约 2 人，两端分析师离散度无显著差异（表 7，p.3057–3058）。
- **分歧来自哪里**：模型中分歧来自**私有信息不同**，加上对他人信号信息量的看法不同（决定是否从价格学习）；作者称该 DO 模型中"disagreement arises endogenously as a result of asymmetric information"（p.3030）。不区分"对同一信息的解读不同"；实证检验的是投资者是否以价格为条件，不是分歧来源。
- **"模型"的具体含义**：ρ，即投资者认为他人信号（从而价格）有多大信息量，可解释为"相对过度自信"（p.3031）。不是对基本面因果关系的模型。
- **公开信息之后收敛还是扩大**：未直接研究；回归中盈余公告月份虚拟变量显示该月波动率更高、|收益|-成交量相关更高（p.3056）。
- **是否观测到推理过程**：没有。只用分析师预测离散度和成交量。
- **作者自己承认的局限和提出的未来研究方向**：RE 与 DO "are extremely difficult to distinguish empirically. The underlying assumptions about beliefs, information, and learning are not observable and cannot be directly tested"（p.3026）。分析师离散度"may also reflect uncertainty about earnings"，且分析师预测有乐观、反应不足、羊群等偏差（p.3050）；不同缩放方式"often lead to opposite implications"（p.3050）；"the results depend on the proxy for belief dispersion and different proxies may lead to conflicting evidence"（p.3057）。高波动均衡"left for future work"（p.3049）；建议用"成交量-波动率相关性"这一不依赖分歧代理的指标重新审视已有文献（p.3059）。
- **与我们方法最相关的一点**：该文明确指出信念、信息和学习方式不可观测，只能从收益-成交量相关性的符号间接推断投资者是否以价格为条件。

---

### Andrei, Carlin & Hasler（2019）Model Disagreement and Economic Outlook —— `Andrei_Carlin_Hasler_2019_MS_Model_Disagreement_Economic_Outlook_WP.pdf`

- **研究问题**：投资者偏好相同、信息相同，但使用不同经济模型（对经济周期长度看法不同）时，资产价格、收益波动和交易如何变化（WP 摘要）。
- **分歧或研报内容怎么度量**：纯理论（连续时间）。总禀赋增长率 f 不可观测，按速度 λ 均值回复；两个投资者都观测禀赋过程 δ，各自用 Kalman 滤波估计 f，但**对均值回复速度取值不同**（λ_A≠λ_B），对长期均值和波动的看法相同，且都不知道真实 λ（WP p.6）。区分两个量：(1) "disagreement over fundamentals" ĝ_t = f̂_B − f̂_A（当前估计之差）；(2) "difference in economic outlooks" η_t（两人概率测度之间的 Radon-Nikodym 导数），取决于未来各期分歧的积分，形成"分歧期限结构"（WP p.3, 8–11）。
- **主要发现**：(1) 若一方相信长期风险，分歧持续；λ 不同使后验方差不同，分歧带随机项；即便关掉随机项，分歧也会围绕随机均值不断再生，不收敛到 0（命题 1，WP p.9）。(2) 当前对基本面毫无分歧（ĝ=0）时，展望不同仍使波动率高于可观测等价的代表性代理人经济，并产生交易：λ_A=0.3 时，λ_B=0.2 成交量 0.085，λ_B=0.1 成交量 0.200；λ 相同时为 0（表 2，WP p.18）。例：同处 f=−1% 的衰退，λ_A=0.3 者认为约 2 年恢复，λ_B=0.1 者认为约 6 年（WP p.11）。(3) 波动水平主要由长期风险决定，波动的变化与持续性（类 GARCH 聚集）主要由分歧驱动（WP p.3, 21）。
- **分歧来自哪里**：明确是**模型/解读不同，信息相同**。脚注 1："This form of disagreement arises if agents are uncertain about the interpretation of public information, even after observing infinitely many signals"（WP p.2）。分歧来源的动机：长期风险模型有争议、使用不同时间序列（消费或产出）、不同估计方法、是否校正均值回复速度估计偏差（WP p.2–3）。
- **"解读不同"如何识别**：本 WP 版本**没有实证识别**。作者承认模型分歧"presumably is not observable by the econometrician"（WP p.17）。唯一的定量依据是附录 A.1 的模拟：同一份模拟季度数据（50 年），一方用 Kalman 滤波加极大似然，另一方用粒子滤波，估出的 λ 绝对差 0.1742，相对真值约 87%，其他参数相对差都小于 25%（表 5，WP p.27–28）。已发表的 MS 2019 版是否加入实证，本文件中未能确认。
- **"模型"的具体含义**：模型 = 数据生成过程的一个结构参数（均值回复速度 λ，即经济周期长度）。同一观测数据通过不同参数的滤波器得到不同的当前估计和不同的未来路径。
- **公开信息之后收敛还是扩大**：模型中只有禀赋过程这一公开信息，持续观测**不会**使分歧消失（WP p.9）；结论中推测，若加入新闻且注意力波动，"spikes in attention will exacerbate the disagreement"（WP p.26）。
- **是否观测到推理过程**：理论文章，不涉及观测。
- **作者自己承认的局限和提出的未来研究方向**："we assume that investors do not change their economic models"，需研究完整学习下的结果（WP p.26）；不考虑额外新闻（报纸、季报、经济数据），学习不确定性被设为常数（WP p.26）；未来可用实证分歧期限结构（Patton & Timmermann 2010；Andrade et al. 2014）"help estimating the magnitude of the difference between these parameters"，并可能解释价值溢价（WP p.26）。
- **与我们方法最相关的一点**：该文区分"当前对基本面的分歧"和"未来展望的分歧"，后者来自模型参数不同，在当前估计一致时依然存在，且被明确说明计量者观测不到。

---

### Cookson & Niessner（2020）Why Don't We Agree? Evidence from a Social Network of Investors —— `Cookson_Niessner_2020_JF_Why_Dont_We_Agree_WP.pdf`

- **研究问题**：投资者分歧有多少来自信息集不同、多少来自投资模型不同（WP 摘要, p.1）。
- **分歧或研报内容怎么度量**：StockTwits 2013.1–2014.9，美国总部公司中讨论量前 100 的股票，只提及单一股票的消息，用户须填写投资方法、持有期和经验，共 1,460,349 条消息、11,874 名用户（WP p.6）。情绪：用户自标 bullish/bearish，未标注的用按投资方法分别训练的最大熵分类器补标，交叉验证准确率 83%（WP p.10）。(1) 组间分歧 CrossDisagreement = 同一公司-日内六种投资方法（基本面、技术、动量、全球宏观、价值、成长）各组平均情绪的标准差（式 4，WP p.16）。(2) 组内分歧 = √(1−AvgSentiment²_{itg})（沿用 Antweiler & Frank），某组当日无消息设为 0（式 6，WP p.18）。作者说明该指标应理解为分歧的**变化**（WP p.16）。(3) ANOVA：情绪对公司、时间、方法固定效应回归。
- **主要发现**：(1) 摘要称组间分歧占公司-日层面总体分歧的 54%，正文表述为"approximately half"（WP p.2, 26）；54% 的具体计算在已读部分未能确认。均值：全体 0.47，组间 0.39，组内平均 0.25（WP p.19）。(2) 公司加时间固定效应解释情绪变动的 10.1%，加入投资方法固定效应再多解释约 1 个百分点，约等于公司加时间所解释部分的 9.9%；持有期和经验的解释力更小（WP p.14–15）。(3) 组间分歧上升 1 个标准差，当日异常成交量增加 4.1%；隔夜（盘前）组间分歧增加 1 个标准差，当日成交量增加 4.2%（WP p.17）；控制组内分歧后组间效应不变（WP p.20）。(4) 与月度分析师离散度相关性不显著；分析师离散度与异常成交量的相关只有 0.0388，本文全体分歧与日异常成交量相关 0.117（WP p.21–22）。组间分歧与分析师离散度、波动率负相关，组内分歧为正相关，作者据此认为"analyst dispersion and return volatility are better measurements of information-driven disagreement"（WP p.21）。
- **分歧来自哪里 / "解读不同"的代理变量**：识别假设："disagreement across investment approaches is more likely to arise due to differing investment models, whereas within-investment approach disagreement is more likely to be related to different information sets"（WP p.1–2）。代理变量即**用户注册时自报的投资方法标签**（6 类）。作者核验标签：各方法最具区分度的高频词与自报方法一致，如基本面投资者常用"eps""announcement"，技术投资者常用"charts""head"（WP p.8，附表 A.2）。
- **作者承认的不足**："our investment approach measures are coarsely defined"，技术派内部还有"主观"和"客观"等子类，"some of the disagreement that we attribute to within-groups, and therefore to differences in information sets, is actually caused by model disagreement. Thus we likely document a lower bound"（WP p.2；p.20 重述）。StockTwits 用户交易不占市场主体（WP p.2）；自报方法可能未被真正理解，因此做了文本核验（WP p.8）；技术投资者组内分歧最大，与"many ways to be a technical investor"一致（WP p.19）。另外，这种划分把组内分歧整体归为信息差异，并没有直接观测到信息集。
- **"模型"的具体含义**：投资风格标签（technical/fundamental/momentum/value/growth/global macro），即一种粗分类的投资哲学，不是具体的参数或因果结构。
- **公开信息之后收敛还是扩大**：盈余公告日异常成交量上升 67%，一周后仍高 39%，三周后仍高 5%（WP p.24–25）；控制分歧后公告日系数由 0.670 降至 0.572，约解释公告日成交量跳升的 1/8；允许分歧效应随日期变化时，组内分歧最多解释 26%；负面盈余意外时约 40%，正面约 20%（WP p.24–25）。引言和结论的说法是"up to one third"（WP p.3, 26），与正文数字口径不同。作者把这一结果视为分歧"rises precisely when information arrives"（WP p.3）的证据，但并未直接展示公告前后分歧水平的变化。
- **是否观测到推理过程**：部分。观测到看多/看空的方向和自报的投资方法；140 字消息文本只用于情绪分类和方法核验，没有提取推理链或关注的因素。
- **作者自己承认的局限和提出的未来研究方向**：除上述"粗分类，下界"外，引言提出分离两种来源时"the researchers would ideally observe investors trading strategies – not just the executed trades"（WP p.1，引 Rothschild & Sethi 2014; Baron et al. 2012）。未单列未来研究方向。
- **与我们方法最相关的一点**：该文以自报投资风格标签作为"模型"的代理，把总分歧拆为组间（模型）和组内（信息）两部分；作者自己承认这种粗分类会把部分模型分歧误记为信息分歧。

---

### Cookson, Engelberg & Mullins（2023）Echo Chambers —— `Cookson_Engelberg_Mullins_2023_RFS_Echo_Chambers_WP.pdf`

- **研究问题**：投资者是否选择性接触能证实自己观点的信息（回音室），这是否维持了分歧并影响收益和交易量（WP 摘要, p.1）。
- **分歧或研报内容怎么度量**：StockTwits 2013–2019.11 全部消息，约 2,500 万条、近 30 万用户、约 1,000 个标的；950 万条关注关系，带时间戳的关注和点赞记录；只用用户自标 bullish/bearish 的消息，剔除日发帖超过 1,000 条的机器人（WP p.1, 7–8）。用户-股票-日内 ≥90% 消息同向即判为 bull 或 bear（WP p.9）。核心回归：Follow Bulls（之后 k 天净关注多头是否多于空头）对 Declare Bull 回归，含用户固定效应和股票×日固定效应（式 2，WP p.13–14）。成交量部分：每个接收者新闻流中情绪的均值 μ̂ 和标准差 σ̂；Received Uncertainty = 各接收者 σ̂ 的平均；Received Disagreement = 各接收者 μ̂ 的跨用户标准差；Sender Disagreement = 发帖情绪的标准差（WP p.25–27）。
- **主要发现**：(1) 自称多头者关注同股多头用户的可能性是空头的 5 倍；同股同日比较，声明多头使关注多头的基准概率提高 41%（WP p.1–2）。(2) 50 天内多头比空头多看到约 70 条看多消息、少看到 15–16 条看空消息；多头新闻流中约 95% 的用户印象为看多，空头约 65%（WP p.2, 12）。(3) 专业投资者中也存在选择性接触；带交易声明的用户选择性接触约为两倍（WP p.2）。(4) 信息"筒仓"：随机分配下预期 38% 的用户收到全同向消息，实际为 50%（WP p.3）；Sender Disagreement 和 Received Disagreement 各增加 1 个标准差，异常换手率约增加均值的 4%，Received Uncertainty 增加 1 个标准差，异常换手率约减少 1.5%（WP p.28）。(5) 关于回音室内形成的信念收益更差：该 WP 版引言中的具体数字仍为占位符（"XXX%""YYYY""ZZZ%"，WP p.3），未能确认。
- **分歧来自哪里**：本文把分歧的持续归因于**信息集不同**，而这种信息差异由投资者自己的选择造成。作者指出已有 DO 文献"often silent about why processing or information sets are persistently different across investors"（WP p.1），并把本文归入"不同信息集"一支，为渐进信息扩散提供一种摩擦（WP p.5–6）。识别方式：同股同日比较多头和空头的关注选择，用股票×日固定效应吸收新闻与注意力冲击（WP p.2）。不检验解读差异；作者把回音室描述为确认偏误中的"信息获取"一面，与已有文献关注的"解读"一面相对（WP p.4）。
- **"模型"的具体含义**：本文不使用"模型"概念；多头或空头身份是持续性立场（声明后 50 天内倾向重复同向情绪，WP p.9）。
- **公开信息之后收敛还是扩大**：**扩大**。"selective exposure to information is nearly twice as large around earnings announcements, when we would expect public news to cause convergence in beliefs"（WP p.2）；对应回归见表 7（WP p.19–20），多头关注多头、空头关注空头都增强，且有股票×日固定效应，不只是活跃度或乐观情绪上升；多头的增强主要出现在正面盈余新闻时（WP p.20）。作者称这是对 Kandel & Pearson (1995) 盈余公告后分歧上升的"complementary mechanism"：KP 认为是解读不同，本文认为是投资者选择接触更极化的信息（WP p.2, 20）。
- **是否观测到推理过程**：没有。观测到的是立场（多/空）和信息获取行为（关注、点赞、新闻流内容），不涉及推理逻辑。
- **作者自己承认的局限和提出的未来研究方向**：StockTwits 用户"not a perfectly-representative sample of investors"（WP p.7）。未来研究："analysts have views on firms and then collect information to make recommendations ... To what extent does selective exposure lead agents to have views which are 'too sticky' in these other settings?"；技术与信息环境变化是否放大选择性接触（WP p.29）。
- **与我们方法最相关的一点**：该文的分歧机制在信息输入端（各人看到的信息集不同且自我选择），推理过程本身未被测量；作者在结论中把分析师先有观点再收集信息列为待研究情境。

---

### Giannini, Irvine & Shu（2019）The Convergence and Divergence of Investors' Opinions around Earnings News —— `Giannini_Irvine_Shu_2019_JFM_Convergence_Divergence_Investors_Opinions_Earnings_News.pdf`

- **研究问题**：盈余公告前后投资者分歧是收敛还是扩大，二者分别如何影响成交量和收益（p.94）。
- **分歧或研报内容怎么度量**：StockTwits 帖子与 Factiva 新闻（PR Newswire、Dow Jones Newswire、Reuters），2009.7–2011.6，2,983 家公司 19,751 个季度盈余公告（p.95, 99–100）。最大熵分类器（手工标注 2,000 条训练，100 条测试准确率 67%）把每帖、每篇新闻标为 −1/0/1（p.101）。IMPACT = Σ(1+粉丝数)×情绪；NEWS = 盈余相关新闻情绪之和；各在公告前（后）两周累计。DIVOP = 1 当且仅当 IMPACT 与 NEWS 符号不同（式 3–4，p.102）。按公告前后的 DIVOP 分四组：AA、AD（前一致后分歧＝发散）、DA（前分歧后一致＝收敛）、DD（p.95–96）。
- **主要发现**：(1) 公告前存在分歧时 [0,1] 累计异常换手率 2.78%，一致时 1.70%，差 1.08%（t=14.11）（p.96）。(2) 收敛（DA）和发散（AD）相对 AA 都提高公告成交量，分别高 1.01% 和 1.07%（p.96）。(3) 公告收益：前分歧组 −0.05%，前一致组 0.33%；DA 组 −0.16%，AD 组 +0.65%，差 0.81%（t=4.48），作者认为支持 Miller (1977)，但只在新信息消解分歧时成立（p.96）。(4) 低机构持股（卖空约束更紧）时分歧效应更强；公告后 [2,20] CAR 分歧组 −0.72%，一致组 −0.27%（p.96）。
- **分歧来自哪里**：作者把成交量机制分成两类理论：Kim & Verrecchia 式的"先验分歧（私有信息精度不同）＋公告后收敛"，与 Karpoff / Kim & Verrecchia (1994) / Kandel & Pearson 式的"对公告本身的不同解读（belief jumbling）＋公告后发散"（p.94–95）。DD/AD/DA 分组只区分公告前后分歧水平的变化方向，不在个人层面识别信息或解读。用于"解释"发散的决定因素：低收益/高成交量哑变量（Bamber et al. 1999 的代理）显著为正（系数 0.532，t=5.96）；机构股东多（先验中含异质信息）、特质波动高、盈余质量差（"could lead investors to interpret earnings information differently"）（p.105）。
- **"解读不同"的代理变量**：社交媒体情绪与新闻情绪的**符号是否相反**，作者把它解释为两个投资者群体之间的分歧（p.95）。作者承认的不足：样本只覆盖部分投资者，脚注 2 写道"we recognize that our sample contains only a subset of investors"，对比新闻是为了纳入更广泛的投资者观点；也可以用 Twitter 用户之间的离散度，结果相似，但要求窗口内有多个用户，样本会变小（p.95）。分类准确率 67%（p.101）。
- **公开信息之后收敛还是扩大**：**两种都有，且发散略多于收敛**。表 5：AA 6,224 个、AD 4,530 个、DA 3,370 个、DD 5,607 个公告（p.107）。结论："while earnings announcements can sometimes increase consensus among investors, they can also produce the opposite effect, a divergence of opinion after the announcement"（p.116）。
- **"模型"的具体含义**：本文不使用"模型"一词；"解读不同"在理论动机中指 belief jumbling（同一盈余新闻引起不同方向的信念修正）。
- **是否观测到推理过程**：没有。只观测帖子和新闻的情绪极性。
- **作者自己承认的局限和提出的未来研究方向**：分析师离散度"captures only the opinions of analysts, who are a small subset of well-informed market participants"，多数已有代理"do not change in a timely manner"（p.95）；本文度量在多元回归中只与换手率、规模、动量稳定相关，与分析师离散度不相关（p.95, 97）。作者给出发散/收敛的可观测决定因素，以"assist future research that will not necessarily have access to a Twitter feed"（p.104）。
- **与我们方法最相关的一点**：该文直接测量了盈余公告前后分歧方向的变化（收敛 3,370 个、发散 4,530 个），但分歧只是二元的"情绪符号是否一致"，且是投资者群体与新闻之间的比较。

---

### Antweiler & Frank（2004）Is All That Talk Just Noise? The Information Content of Internet Stock Message Boards —— `Antweiler_Frank_2004_JF_Is_All_That_Talk_Just_Noise_WP.pdf`

- **研究问题**：网络股票论坛帖子是否含有能预测收益、成交量、波动率和价差的信息（WP p.1）。
- **分歧或研报内容怎么度量**：Yahoo! Finance 和 Raging Bull，2000 年，道琼斯工业指数 30 家加互联网指数 15 家共 45 家公司，1,559,621 条消息；手工标注 1,000 条训练朴素贝叶斯和 SVM，每帖分为买入/持有/卖出（WP p.7, 10）。按 15 分钟、1 小时、1 天汇总：看涨指数 θ_t=(M^B−M^S)/(M^B+M^S)，主用 θ*_t = ln[(1+M^B)/(1+M^S)]（WP p.11）。分歧：令 x_i∈{−1,+1}（忽略持有），期内方差 σ²_t=1−θ²_t，**一致指数** η_t = 1−√(1−θ²_t) ∈[0,1]（式 7–8，WP p.13）。无消息时 θ=0，η=0，"periods without information may be viewed as latent disagreement"（WP p.13）。Cookson & Niessner 的组内分歧指标即来自此式。
- **主要发现**：(1) 发帖数与成交量、波动率正相关；日频上发帖预测次日成交量（1 个标准差冲击使次日成交量增加 0.16 个标准差），对小额交易的作用更强，与"日内交易者"假说一致（WP p.26，结论 p.29）。(2) 帖子不能预测收益（WP p.2, 26）。(3) 一致指数与成交笔数的简单相关为正，控制其他因素后符号反转为负，"consistent with the theoretical expectations such as in Harris and Raviv (1993)"（WP p.22）；但 15 分钟数据中"An increase in agreement predicts an increase in trading volume over the next hour"（WP p.25）；波动率与一致程度简单相关为正，与作者预期相反（WP p.20）。(4) 成交量上升后看涨程度和一致程度随之上升，作者解释为小额交易者买入后发帖（WP p.26，结论 p.29）。
- **分歧来自哪里**：没有区分。只把论坛上看多/看空的离散程度作为分歧，引用 Harris & Raviv (1993) 的"分歧诱发交易"（WP p.12 脚注 12）。
- **"模型"的具体含义**：不涉及。
- **公开信息之后收敛还是扩大**：本 WP 版本未做盈余公告分析。做了《华尔街日报》报道前后的分析：报道前一天发帖增多、略偏看涨，交易增加（尤其大额）；报道前两天市场有异常而论坛没有（WP p.27–28）。未单独报告报道前后一致指数的变化。
- **是否观测到推理过程**：没有。帖子文本只被分类为买入/持有/卖出三类。
- **作者自己承认的局限和提出的未来研究方向**：论坛"probably will not"反映全体市场参与者，而是特定群体，尤其是日内交易者（WP p.7）；无消息时段如何处理是"potential source of concern"（WP p.12）；看涨指数采用何种齐次度的聚合函数没有定论（WP p.11）。未来方向："Given that the messages contain explanations, it might be possible to identify different classes of events that have different stock market effects"，并建议对新闻通讯稿做类似分析（WP p.30）。
- **与我们方法最相关的一点**：该文的分歧度量 1−√(1−θ²) 只使用多/空二元立场的比例，作者在结论中提到帖子含有"explanations"但未加利用。

---

## 本组跨文对照（按任务要求的三个关注点，仅作事实汇总）

| 论文 | 如何识别/刻画"解读不同" | 作者承认的不足（要点） | 公开披露后：收敛还是扩大 | "模型"的含义 |
|---|---|---|---|---|
| Harris & Raviv 1993 | 理论：共同先验，似然参数不同 | 特定函数形式；忽略对模型的学习 | 同一信号下两组后验分开；只在信念交叉时交易 | 似然函数的参数（对信号质量的看法 θ） |
| Bamber et al. 1999 | 分析师年度 EPS 预测反向修正且交叉/发散的配对比例（KP） | 保守下界；无法区分同向交叉中的解读与精度差异；是分析师对年度盈余的预期而非投资者对现金流的预期 | 约 12–15% 的配对在季报后反向交叉/发散 | 对公开信号误差均值的不同预期（μ₁≠μ₂） |
| Banerjee & Kremer 2010 | 理论：对信号噪声均值的随机解读 e_{i,t}；实证无直接代理 | 分歧是潜变量；无法同时允许精度和均值异质 | 都可能：解读一致时收敛，新解读冲击时扩大；盈余公告视为分歧跳升 | 对信号偏差的标量解读 |
| Banerjee 2011 | 不识别解读；区分是否从价格学习（ρ） | 信念、信息、学习不可观测；分析师离散度含不确定性和偏差，不同代理结论冲突 | 未研究 | 认为他人信号有多大信息量（ρ） |
| Andrei et al. 2019（WP） | 无实证；附录模拟：不同估计方法估出不同 λ | 模型分歧"presumably is not observable by the econometrician" | 持续观测不消除分歧；推测注意力高峰会加剧分歧 | 结构参数（均值回复速度 λ，即周期长度） |
| Cookson & Niessner 2020（WP） | 自报投资风格标签：组间＝模型，组内＝信息 | 标签粗分类，组内仍含模型分歧，属下界 | 分歧最多解释公告日成交量跳升的 1/8–26%（引言称最多 1/3）；未直接报告分歧水平变化 | 投资风格标签（6 类） |
| Cookson et al. 2023（WP） | 不识别解读；识别选择性接触（同股同日多/空的关注差异） | 样本不具代表性；收益结果在 WP 中为占位符 | 扩大：公告前后选择性接触约为平时两倍 | 不使用（立场＝多/空身份） |
| Giannini et al. 2019 | 社交媒体情绪与新闻情绪符号相反（DIVOP） | 只覆盖部分投资者；分类准确率 67% | 都有：发散 4,530 个、收敛 3,370 个（另有 AA 6,224、DD 5,607） | 不使用（理论动机为 belief jumbling） |
| Antweiler & Frank 2004（WP） | 不区分；论坛多/空比例的离散度 | 只代表日内交易者等群体；无消息时段处理有争议 | 未做盈余公告分析 | 不涉及 |
