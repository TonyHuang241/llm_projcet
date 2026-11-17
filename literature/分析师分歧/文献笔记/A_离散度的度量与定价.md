# A 组：分析师预测离散度度量的是什么、与收益的关系

> 页码说明：期刊正式版引用期刊页码；工作论文版（文件名带 `_WP`）引用 PDF 页码（记作"PDF p.x"）。
> 全部 13 个 PDF 都有文字层，已读取。DMS（2002）与 Sadka & Scherbina（2007）本地文件是工作论文版（分别为 2001-10-25 版和 2004-06-18 版），数字以工作论文为准，可能与正式发表版不同。

---

## 一、核心论文

### Barron, Kim, Lim & Stevens（1998）Using Analysts' Forecasts to Measure Properties of Analysts' Information Environment —— `Barron_Kim_Lim_Stevens_1998_TAR_Using_Analysts_Forecasts_Information_Environment.pdf`

- **研究问题**：可观测的预测离散度、均值预测误差和分析师人数，各自对应理论上分析师信息环境的哪种性质（不确定性、共识度、公共/私有信息精度）？
- **分歧或研报内容怎么度量**：理论模型。盈利 y ~ N(ȳ, 1/h)；所有分析师共享同一个公共先验（精度 h），分析师 i 另有一个私有信号 zᵢ = y + eᵢ（精度 sᵢ），预测为贝叶斯后验均值 μᵢ = (hȳ + sᵢzᵢ)/(h + sᵢ)。可观测量：离散度 D = E[样本方差]，均值预测误差平方 SE = E[(y − μ̄)²]，人数 N。理论量：不确定性 V（个人预测误差平方的平均）、共同不确定性 C（预测误差的平均两两协方差）、共识度 ρ = C/V。核心关系：V = C + D，SE = C + D/N；因此 D = V(1 − ρ)（p.426）。反解得到 V = (1 − 1/N)D + SE，ρ = (SE − D/N)/[(1 − 1/N)D + SE]（命题 2，p.427）。
- **主要发现**：
  1. 离散度只反映私有信息带来的个体特有误差，是不确定性的增函数、共识度的减函数（式 13）。
  2. 均值预测误差主要反映共同误差，随 N 增大，个体误差被平均掉（式 12）。
  3. 若各分析师私有信息精度相同，ρ = h/(h + s)，即公共信息精度占总精度的比例；h 和 s 都可以由 D、SE、N 反推出来（推论 1，p.428）。
  4. 文中引用 Byard（1998）的应用：年报公告前私有信息占总精度的 56%，一年前为 20%（p.429）。
- **分歧来自哪里**：完全来自**信息不同**（私有信号不同）。模型假设大家共享同一先验、同一套贝叶斯更新规则，不存在"同一信息、不同解读/模型"的情形。附录 B 把 ρ 解释为两种"信息不对称"的综合（私有信息占比和私有信息质量的差异，p.425、p.432）。数据上无法区分信息不同和解读不同，因为后者不在模型里。
- **是否观测到推理过程**：否。只用预测数值和实际盈利。
- **作者自己承认的局限和提出的未来研究方向**：
  - 假设"common information between any two analysts is common among all analysts"（p.423），并称这是唯一的强假设。
  - 分析师偏差会抬高不确定性估计，并使共识度估计偏高或偏低（p.428–429）。
  - 羊群行为会模糊公共/私有信息的区分："The impact of herding behavior on the construct validity of the measures developed in our model is an open issue."（p.429）
  - 结果是无条件期望，实证中需要合并多个观测（p.429）。
  - 术语上提醒：离散度"reflects both uncertainty and a lack of consensus"，不应直接称为"共识"（p.430）。
- **与我们方法最相关的一点**：该文把离散度拆成"不确定性 × (1 − 共识度)"，其中分歧只来源于私有信号差异。框架里没有分析师之间推理模型不同这一维度。

### Lahiri & Sheng（2010）Measuring Forecast Uncertainty by Disagreement: The Missing Link —— `Lahiri_Sheng_2010_JApplEconometrics_Measuring_Forecast_Uncertainty_by_Disagreement.pdf`

- **研究问题**：预测者之间的分歧（离散度）能否代理总体预测不确定性？两者差在哪里？
- **分歧或研报内容怎么度量**：把个人预测误差分解为 eᵢₜₕ = λₜₕ（共同冲击在 h 期内的累积）+ εᵢₜₕ（个体特有误差）。分歧 dₜₕ 为点预测的截面方差，期望值等于个体误差方差的平均；不确定性 Uₜₕ 为个人预测误差方差的平均。推出 **Uₜₕ = σ²_λ + Dₜₕ**（式 8，p.517），即不确定性 = 分歧 + 未来共同冲击的方差。数据来自美国 SPF 的概率（密度）预测：通胀 1969–2007，GDP 1981–2007，预测期 1–4 个季度。直接用个人密度预测的方差作为"真实"不确定性，以此检验分歧是否是好的代理。个人偏差先扣除。
- **主要发现**：
  1. 分歧几乎总小于不确定性，两者之差随预测期变长而扩大。例如 GDP 4 季度预测，不确定性 1.25，分歧 0.37（表 I，p.528）。
  2. 把不确定性对分歧做回归，系数在不稳定期（1969–1983，通胀）为 0.43，稳定期（1984–2007）通胀为 0.76、GDP 为 0.72（表 III，p.530）。结论：环境稳定时分歧是可用的代理。
  3. 直接批评 Barron 等（1998）：其不确定性 = 均值误差平方 + 分歧，本质上是**事后**量。加入均值误差平方后，对调查不确定性的解释力反而下降，R² 从 0.34 降到 0.09（1969–83 通胀），从 0.39 降到 0.30（1984–2007 通胀）（表 IV，p.532）。
  4. 建议的事前度量是"分歧 + GARCH 预测的共同冲击方差"（式 29，p.534）。
- **分歧来自哪里**：模型中的个体误差 ε 被明确归因于"differences in information acquisition and processing, interpretation, judgment, and forecasting models"（p.516），即信息不同和模型/解读不同都装在 ε 里，模型不区分两者。文中设想过分歧高而个人不确定性低的情形："This situation will arise when forecasters disagree on otherwise precise models and scenarios"（p.521）。1979–81 通胀结构突变期，分歧一度超过不确定性，作者解释为预测者"adopting disparate forecasting functions"（p.527）。数据上无法把两种来源分开。
- **是否观测到推理过程**：否。观测到的是点预测和概率直方图，没有所用模型或变量。
- **作者自己承认的局限和提出的未来研究方向**：
  - 模型假设"rule out any feedback from perceived future variability of common shocks to current idiosyncratic individual variances"，并承认"admittedly simple"（p.521）。
  - 分歧与不确定性的关系"lacking any theoretical basis… becomes an empirical issue"，依赖样本期、目标变量和预测期（p.521）。
  - 调查不确定性本身可能不等于客观不确定性：4 季度预测的 90% 区间覆盖率只有 78%/74%（表 II，p.530）。
  - 基于误差的 GARCH 度量没能捕捉 2002 年后缓慢上升的不确定性（p.536），"Further research on refining the GARCH specification may help"（p.536）。
- **与我们方法最相关的一点**：该文的分歧项 dₜₕ 被作者解释为同时包含"idiosyncratic information and diversity in forecasting models"（p.534），但它只是一个标量方差，无法把两者分开。

### Garfinkel（2009）Measuring Investors' Opinion Divergence —— `Garfinkel_2009_JAR_Measuring_Investors_Opinion_Divergence.pdf`

- **研究问题**：常用的意见分歧代理变量（未预期成交量、买卖价差、收益波动率、分析师预测离散度）中，哪个最接近投资者真实私人估值的分歧？
- **分歧或研报内容怎么度量**：用 NYSE 专有订单数据构造新基准 DIVOP。样本为 2002 年 1–3 月、股票代码以 A–D 开头的公司，共 13,017 个公司-日。限价单的报价、市价单的隐含价格（设为上一笔成交价）相对上一笔成交价的偏离为 %Distance = (OrderPrice − PriorTradePrice)/PriorTradePrice；DIVOP 为当日所有订单 %Distance 的标准差（式 1–2，p.1322）。离散度用当月新发布的 IBES 年度 EPS 预测，σ_frcst = 标准差/|均值|，σ_frcst2 = 标准差/股价（式 9，p.1330）。逐个把 DIVOP 对代理变量回归，并用 Vuong 检验比较解释力。
- **主要发现**：
  1. 换手率变化（ΔTO）解释力最强，系数 0.1043，1% 显著；标准化未预期成交量（SUV）次之（表 2，p.1331）。
  2. 分析师离散度与 DIVOP **负相关**：σ_frcst2 的系数为 −0.0001（1% 显著），σ_frcst 不显著。作者称这"calls into question a broad swath of conclusions that presume forecast dispersion proxies for investor opinion divergence"（p.1332）。
  3. 例外：股价较高的股票，离散度与 DIVOP 正相关（脚注 6，p.1320）；没有 IBES 覆盖的公司，买卖价差是最好的代理。
  4. 盈余公告窗口内也只有未预期成交量与 DIVOP 稳定正相关（表 7，p.1344）。
- **分歧来自哪里**：DIVOP 度量的是私人估值的离散，不区分其来源。作者用 Holthausen & Verrecchia（1990）、Kim & Verrecchia（1994）的"信息含量越大的消息，引起的意见分歧越大"来验证 DIVOP：|公告期 CAR| 的系数为 0.0159（t = 2.36）（p.1324）。文中把分析师离散度的问题归因于陈旧预测、偏差、截断（p.1329）。
- **是否观测到推理过程**：否。观测到的是订单价格（估值的结果），比预测值更接近私人估值，但仍不涉及理由。
- **作者自己承认的局限和提出的未来研究方向**：
  - 私人意见"almost inherently unobservable"，公开数据"none of which directly conveys investors' private valuations"（p.1318）。
  - 2002 年 NYSE OpenBook 改革后，限价单可能同时反映观点和"preferences about how much to reveal regarding those opinions"（p.1321）。
  - 样本只有 3 个月、只含 NYSE；建议研究 NASDAQ，以及分析师覆盖、信息不对称和意见分歧三者的关系（p.1345）。
  - "my data necessarily focus attention on investors who believe strongly enough to submit orders"，建议用实验给不同投资者不同信息集来检验（p.1345）。
- **与我们方法最相关的一点**：该文以订单价格为基准检验各种代理，结果是分析师 EPS 预测离散度与投资者估值分歧总体负相关。这一结论针对的是预测数值的离散度。

### Diether, Malloy & Scherbina（2002）Differences of Opinion and the Cross-Section of Stock Returns —— `Diether_Malloy_Scherbina_2002_JF_Differences_of_Opinion_WP.pdf`（2001-10 工作论文版）

- **研究问题**：分析师盈利预测离散度能否预测股票截面收益？方向支持 Miller 的高估假说、Diamond–Verrecchia/Hong–Stein 的无偏假说，还是风险假说？
- **分歧或研报内容怎么度量**：离散度 = 当期财年 EPS 预测标准差 / |均值预测|，要求至少 2 位分析师，均值为 0 时归入最高组；用 IBES 未经拆股调整的原始数据，以避免拆股后四舍五入人为造成零离散度（PDF p.8、p.11）。样本为 CRSP/Compustat/IBES 的交集，1983-01 至 2000-11，股价 ≥ $5。按上月离散度分五组，持有一个月。
- **主要发现**：
  1. 最低离散度组比最高组年化收益高 9.48%，其中 68.4% 来自空头一侧（PDF p.11）；最小市值组的价差为年化 16.4%（PDF p.12）。
  2. 三因子和四因子模型都无法解释（GRS 检验拒绝）；Fama–MacBeth 回归中离散度在控制换手率后仍显著，换手率不显著（PDF p.22）。
  3. 离散度与 β、盈利波动率、收益波动率正相关，作者据此拒绝"风险"解释（PDF p.17–18）。
  4. 离散度越高，一致预期越乐观：季度预测误差对离散度回归，α₁ = −0.727（t = −33.42）（PDF p.26）。作者把分析师自我审查（悲观者停止覆盖）视为与卖空限制并列的摩擦。
- **分歧来自哪里**：作者把离散度解释为投资者意见分歧，并明确称不需要区分来源："our main argument as to why difference of opinions arise and influence prices does not depend on whether investors are asymmetrically informed or simply differ in the way they interpret the same information"（PDF p.21）。引言中也说意见分歧"typically modeled via dogmatic beliefs or asymmetric information sets"（PDF p.3）。没有在数据上做区分。
- **是否观测到推理过程**：否。只用 IBES 预测数值。
- **作者自己承认的局限和提出的未来研究方向**：
  - 卖空成本与收益之间的证据"not as strong"，作者认为可能源于代理变量噪声（PDF p.24）。
  - 1992–2000 子样本中，效应只在最小市值组显著（PDF p.19–20）。
  - 建议比较"whisper"预测出现前后的样本，以分离分析师审查对价格偏差的作用（PDF p.27）。
  - 未就"离散度含义含糊"作专门讨论（除拒绝风险解释外）。
- **与我们方法最相关的一点**：该文明确声明其机制不依赖分歧来自信息不对称还是对同一信息的不同解读，因此没有试图在数据上区分两者。

### Johnson（2004）Forecast Dispersion and the Cross Section of Expected Returns —— `Johnson_2004_JF_Forecast_Dispersion_Expected_Returns.pdf`

- **研究问题**：能否在无摩擦、无非理性的框架中解释 DMS 的"高离散度、低收益"现象？
- **分歧或研报内容怎么度量**：理论部分：真实价值 Vₜ 不可观测；N 个分析师信号 Uₜ⁽ⁿ⁾ = Vₜ·e^{ηₜ⁽ⁿ⁾}，噪声 η 均值回复。噪声波动 σ_η 决定离散度，代表"参数风险"（对当前价值的不确定性），区别于基本面风险 σ_V（p.1960–1962）。股权是杠杆公司资产的看涨期权，因此特质参数风险越大，预期收益越低。实证部分：IBES 1983–2001，DISP1 = 当期财年预测标准差/均值（同 DMS），DISP2 = 标准差/账面总资产，都转为百分位排名；剔除超过 6 个月的旧预测。检验"离散度 × 杠杆"交互项。
- **主要发现**：
  1. 控制规模、BM、动量后，交互项显著为负（DISP1：−0.0090，t = 2.36；DISP2：−0.0131，t = 2.70），离散度单独项不显著（表 II，p.1973）。
  2. 经济量级：低离散度、高杠杆公司比高离散度、低杠杆公司月收益高 60–90 bp；无负债公司在第 99 和第 1 离散度百分位上的预期收益基本相同（p.1973）。
  3. 进一步控制去杠杆后的资产波动率，结论不变（表 III，p.1975）。
- **分歧来自哪里**：采用"多信号"解释：每个分析师的估计是关于真实值的一个独立噪声信号，离散度反映信息噪声，因此是**信息**差异，而非信念或模型差异。作者明确指出 DMS 的行为解释"views dispersion more literally as a measure of heterogeneity of beliefs"（脚注 3，p.1960）。没有在数据上区分两者，而是把离散度代理不确定性作为"maintained hypothesis"（p.1960）。
- **是否观测到推理过程**：否。"How exactly analysts arrive at these numbers is not modeled here."（p.1962）
- **作者自己承认的局限和提出的未来研究方向**：
  - "what is not being measured is a subjective assessment of earnings uncertainty"：分析师可能都同意均值但各自都很不确定，也可能各自很自信但彼此差异很大（p.1959）。
  - 多信号解释留下几个问题：报告的估计是否涵盖全部相关信息？分析师为何不自己汇总彼此的信息？离散度本身是否受激励影响？"This realization highlights the need for a more complete understanding of exactly how forecast differences arise."（p.1959）
  - "even within this model, forecast dispersion is not a perfect proxy for information quality"（p.1962）。
  - 只处理单一异象；信息生产和资本结构外生；是局部均衡；"makes little or no contribution to our understanding of the nature of that risk"（p.1958）。
  - 未来方向：管理层决定披露多少、多准的完整博弈（p.1976）。
- **与我们方法最相关的一点**：该文明确承认离散度既不是主观不确定性，也不能说明"预测差异如何产生"，并把这一点列为需要进一步理解的问题（p.1959）。

### Zhang（2006）Information Uncertainty and Stock Returns —— `Zhang_2006_JF_Information_Uncertainty_and_Stock_Returns.pdf`

- **研究问题**：如果短期价格延续来自投资者对新信息反应不足，那么信息不确定性越大，好消息后收益是否越高、坏消息后收益是否越低？
- **分歧或研报内容怎么度量**：信息不确定性定义为"ambiguity with respect to the implications of new information for a firm's value"，来源是基本面波动和信息质量差（p.105）。用 6 个代理：1/市值、1/上市年限、1/分析师覆盖数、预测离散度 DISP（预测标准差 / 上年末股价）、周收益波动率、5 年经营现金流波动率（p.110–111）。好/坏消息用当月一致预期修正方向或过去 11 个月收益区分。样本 1983–2001，股价 ≥ $5，持有一个月。
- **主要发现**：
  1. 按不确定性单独排序，高减低组收益都不显著；DISP 为 −0.06%（t = −0.25）（表 II，p.114）。不支持"不确定性是被定价的风险"。
  2. 预测修正分组后，多数代理呈现"坏消息时高不确定性组收益更低、好消息时更高"。DISP 例外：坏消息组 D10−D1 = −0.23%（t = −0.77），好消息组 0.56%（t = 1.82）（表 III，p.116）；原文说"The bad-news D10–D1 strategy produces significantly negative returns for all proxies except for DISP"（p.117）。
  3. 动量策略在高 DISP 五分位组月收益 2.29%，全样本为 1.55%（表 VIII，p.130）。
  4. DISP 与市值负相关，但与年限、覆盖数不相关，与收益波动率也不相关；作者认为这些代理"might capture different aspects of information uncertainty"（p.113）。
- **分歧来自哪里**：该文不把离散度当作意见分歧，而当作信息不确定性的代理，与 Barron 等（1998）、Imhoff & Lobo（1992）的用法一致（p.110）。不区分信息不同和解读不同，也不区分基本面波动和信息质量："it is hard to empirically disentangle one from the other"（脚注 2，p.105）。
- **是否观测到推理过程**：否。
- **作者自己承认的局限和提出的未来研究方向**：
  - "Although each proxy might also capture other effects, the common element is their ability to quantify information uncertainty."（p.110）
  - "Because I do not incorporate measures of private information or overconfidence in my empirical analysis, my evidence leaves the door open for other behavioral models"（p.107）。
  - "I cannot definitively rule out the possibility that each information uncertainty proxy may capture other effects"（p.135）。
  - 建议进一步研究信息不确定性能否完全解释规模效应，作者称"beyond the scope of this paper"（p.117）。
- **与我们方法最相关的一点**：该文把离散度放在"信息不确定性"一族代理中，并且在 6 个代理里它与其他代理的相关性最弱、坏消息检验中唯一不显著。

### Miller（1977）Risk, Uncertainty, and Divergence of Opinion —— `Miller_1977_JF_Risk_Uncertainty_and_Divergence_of_Opinion.pdf`

- **研究问题**：投资者对收益估计不同且卖空受限时，证券价格如何决定？能否解释高风险股、新股、封闭式基金折价等现象的低收益？
- **分歧或研报内容怎么度量**：理论文章，没有实证度量。用投资者估值的横截面分布（需求曲线 ABC）刻画分歧：供给固定且只需少数人持有时，价格由最乐观的少数人决定，分布越分散，价格越高（图 1，p.1152–1153）。提到换手率"provides one measure of diversity of opinion"（p.1156）。
- **主要发现**（理论命题）：
  1. 只要少数投资者就能吸收全部供给，价格高于平均估值；分歧越大，价格越高、事后收益越低（p.1153）。
  2. 风险、不确定性和分歧通常相伴，因此高风险股票的事后收益可能更低，以此解释高 β 组合低于 CAPM 预测（p.1155–1157）。
  3. 随时间推移不确定性消解、分歧收窄，乐观者估值下降，以此解释新股长期表现差（p.1156）。
  4. 卖空只能部分缓解，因为卖空者拿不到卖空所得（p.1160）。
- **分歧来自哪里**：主要把分歧归于不确定性本身："the very concept of uncertainty implies that reasonable men may differ in their forecasts"（p.1151），"such uncertainty normally implies divergence of opinion"（p.1155）。同时明确提到**估值方法不同**也是来源："There exist a number of methods of security evaluation… Different investors tend to use different methods. Ownership of a security will gravitate towards the investors who employ the evaluation methodology which is most favorable to it"（p.1163）。此外还有"badly informed"投资者（p.1154）。理论上信息和方法两种来源都有，但没有区分或检验。
- **是否观测到推理过程**：否（理论文章）。不过 p.1163 把"用哪种估值方法"当作投资者之间的差异维度，例如股息率派与成长派。
- **作者自己承认的局限和提出的未来研究方向**：
  - 用"casual observation""casual empiricism"支持部分论点（p.1165），没有做正式检验。
  - "Testing of the above hypothesis is complicated by…"（关于知名度假说，p.1165）。
  - 对计量建模的建议：线性模型可能拟合差，"Better results may come from models which choose the maximum value given by any one of a set of different formulas"，可以先把股票按定价依据分组，再分别估计定价函数（p.1163）。
  - 未提及"离散度含义含糊"，因为文中没有使用离散度。
- **与我们方法最相关的一点**：Miller 把"不同投资者使用不同估值方法、价格由方法最有利者决定"作为分歧机制之一（p.1163），这是文本层面的论述，没有对应的度量。

---

## 二、边缘论文

### Doukas, Kim & Pantzalis（2006）Divergence of Opinion and Equity Returns —— `Doukas_Kim_Pantzalis_2006_JFQA_Divergence_of_Opinion_and_Equity_Returns.pdf`

- **研究问题**：剔除不确定性后，意见分歧是被溢价定价（Miller）还是被折价定价？
- **度量**：用 BKLS（1998）的 D = V(1 − ρ)，以"diversity"(1 − ρ) 作为分歧，V 作为不确定性；两者都用**事后**预测误差估计。原文："The ex post methodology is dictated by the lack of ex ante measures"（p.577）。IBES 1983–2001，35,782 个公司-年。
- **主要发现**：原始离散度高减低为 −0.41%/月（t = −5.98），复现了 DMS（表 2，p.584）；控制 V 后，离散度效应反转为正（表 3，p.587）；diversity 高减低在所有市值组都显著为正（p.588）。作者结论：DMS 的结果"driven by uncertainty"（p.602），并以此"refute the view that dispersion in analysts' forecasts reflects divergence of opinion"（p.573）。
- **分歧来源**：分歧 = 私有信息差异（BKLS 框架）；脚注 8 说"differences of opinion stem from the volatility of a firm's underlying fundamentals rather than poor or limited information"（p.577）。
- **推理过程**：未观测。**局限**：事后度量不能构成交易规则（p.577）。
- **与我们最相关的一点**：该文的"分歧"依赖 BKLS 的信息结构，而 Lahiri & Sheng（2010）批评了 BKLS 用事后误差估计不确定性的做法。

### Sadka & Scherbina（2007）Analyst Disagreement, Mispricing, and Liquidity —— `Sadka_Scherbina_2007_JF_Analyst_Disagreement_Mispricing_Liquidity_WP.pdf`（2004-06 工作论文版）

- **研究问题**：高离散度股票的高估为何长期存在？是否与流动性（交易成本）有关？
- **度量**：离散度 = 当期财年 EPS 预测标准差 / 每股账面净资产（PDF p.13）；流动性用 Sadka（2004）的永久性价格冲击。
- **主要发现**：扣除交易成本后，做空高离散度组合不再盈利，例如最小市值高离散度组的 α 从 −0.74%（t = −2.75）变为 −0.29%（t = −1.06）（PDF p.15）；离散度 × 价格冲击的交互项是最显著的负向预测变量；高离散度股票收益与总流动性创新负相关。
- **分歧来源**：分歧伴随乐观偏差（分析师激励）；交易成本高是因为部分投资者对"how to aggregate analysts' opinions"有更好的知识（PDF p.2），即**对分析师的信息不对称**。脚注 4 记录了另一种解读：离散度代表投资者分歧时，价格冲击可能只是需求曲线变陡（PDF p.7）。
- **推理过程**：未观测。**局限**："it is difficult to distinguish between the liquidity risk interpretation and mispricing-costs of arbitrage explanation"（PDF p.23）。
- **与我们最相关的一点**：该文把离散度同时联系到乐观偏差和"如何汇总不同分析师意见"的信息优势。

### Avramov, Chordia, Jostova & Philipov（2009）Dispersion in Analysts' Earnings Forecasts and Credit Rating —— `Avramov_Chordia_Jostova_Philipov_2009_JFE_Dispersion_Analysts_Earnings_Forecasts_Credit_Rating.pdf`

- **研究问题**：离散度异象是否由财务困境（信用评级下调）驱动？
- **度量**：离散度 = EPS 预测标准差 / |均值|（同 DMS）；样本为 S&P 评级公司，共 3,261 家，1985–2003。
- **主要发现**：离散度策略的收益集中在评级最差、市值合计不到全部评级公司 5% 的公司，且只出现在评级下调期；其余约 92.68% 的观测中离散度与收益关系不显著（p.101）。对评级调整后的离散度或收益不再有预测力。同时反驳了另外两种解释：卖空成本代理捕捉不到这一效应（反驳 DMS），离散度效应"indistinguishable across levered and unlevered firms"（p.84，反驳 Johnson）。
- **分歧来源**：把离散度视为"uncertainty about next year's earnings, which is a single component in asset valuation"，违约风险则涵盖增长率、资本成本等更多估值要素（p.84）。不讨论信息与解读之分。
- **推理过程**：未观测。**局限**：未提及专门局限。
- **与我们最相关的一点**：该文指出 EPS 离散度只覆盖估值中的一个输入（下一年盈利），其余估值要素不在其中。

### Yu（2011）Disagreement and Return Predictability of Stock Portfolios —— `Yu_2011_JFE_Disagreement_Return_Predictability_Stock_Portfolios.pdf`

- **研究问题**：自下而上加总的组合层面分歧能否预测市场和 BM 组合的收益？
- **度量**：个股分歧 = IBES 长期 EPS 增长率（LTG）预测的标准差；市场分歧为其市值加权平均（式 1，p.164）；对比用 S&P 500 指数 EPS 预测离散度的自上而下度量。
- **主要发现**：分歧每升高 1 个标准差，未来一年市场超额收益下降 6.6 个百分点（例如从 9% 降到 2.4%）（p.163）；分歧冲击主要对应贴现率新闻；同样冲击下，成长股未来一年收益下降 8.17%，价值股下降 2.58%（p.163）。
- **分歧来源**：按 Miller 框架解读为信念分歧；控制平均个股方差、特质波动率后结论不变（p.182）。结尾引用 Acemoglu 等（2006）：对"signal interpretation"不确定时，贝叶斯学习者的分歧可能永不消失（p.183）。
- **推理过程**：未观测。**未来方向**：自下而上与自上而下的对比能否推广到其他市场和公司层面，例如有多个子公司的集团（p.182）。
- **与我们最相关的一点**：该文表明同一市场的分歧，按"对哪些成分有分歧"来加总（自下而上）与只看总量（自上而下）结果不同，前者预测力更强。

### Güntay & Hackbarth（2010）Corporate Bond Credit Spreads and Forecast Dispersion —— `Guntay_Hackbarth_2010_JBF_Corporate_Bond_Credit_Spreads_Forecast_Dispersion.pdf`

- **研究问题**：离散度在公司债市场是否也有 Miller 式定价效应？
- **度量**：季度 EPS 预测标准差 / 季末股价（参照 Zhang 2006，p.2330）。附录模型中分析师预测方差分为行为成分（分歧、偏差）和理性成分（现金流不确定性）（p.2328 脚注 2，附录 A）。
- **主要发现**：离散度高的债券信用利差更高，单变量回归解释约 23% 的截面变异；离散度上升 1 个标准差，利差增加 19 bp（样本均值 100 bp）（p.2329）。离散度变化能预测利差变化。在同一批发行人的股票上，仍复现了离散度与股票收益的负相关。
- **分歧来源**：债市结果支持"dispersion appears to proxy largely for future cash flow uncertainty"（摘要），与 Miller 式分歧解释不符（p.2344）。
- **推理过程**：未观测。**未来方向**：股债市场的分割程度（p.2344）。
- **与我们最相关的一点**：同一离散度变量在股市和债市给出方向相反的定价结果，作者据此认为它在两个市场代表不同的东西。

### 朱宏泉、余江、陈林（2016）异质信念、卖空限制与股票收益——基于中国证券市场的分析 —— `朱宏泉等_2016_管理科学学报_异质信念卖空限制与股票收益.pdf`

- **研究问题**：A 股异质信念是否像 Miller 预测的那样"当期收益高、未来收益低"？融资融券放开后效应是否减弱？
- **度量**：不用分析师离散度，理由有二：分析师只代表职业人士且覆盖不全；"分析师预测分歧含有不确定性"和乐观偏差（p.116）。改用换手率分离模型：先用 FF3 拆出市场收益和公司特质收益，再将日换手率对正负两类收益回归（前 36 个月估计），残差（扣除截距代表的流动性需求和信息驱动的交易）即为异质信念 HB（式 1–2，p.117）。样本为沪深 A 股 1997–2014。
- **主要发现**：2000–2009 年，HB 最高组比最低组当月收益高 4.17%、次月低 1.62%（均 1% 显著，表 3，p.119）；FF3 时间序列回归中 HBₜ 系数 0.997、HBₜ₋₁ 系数 −0.510（表 5）；2011–2014 年允许卖空的股票，次月效应不显著（表 8，p.123）。
- **分歧来源**：引言列举先验信念、获取信息的渠道与速度、对信息的解读、风险承受能力（p.115），但度量上不做区分，把不能被流动性和信息解释的交易全部归为分歧。
- **推理过程**：未观测（交易量残差）。**局限**：未提及。
- **与我们最相关的一点**：该文是本组唯一的 A 股样本，它以"含不确定性、偏差、覆盖不全"为由放弃了分析师离散度，改用交易量残差。

---

## 三、本组的争论脉络（按论文内容整理，事实陈述）

1. **离散度是什么——三种解释并存**
   - **意见分歧**：DMS（2002）、Yu（2011），理论基础是 Miller（1977）。DMS 明确声明不区分信息不对称与解读差异。
   - **不确定性 / 信息风险**：Johnson（2004）认为离散度是未定价的参数风险；Zhang（2006）把它作为信息不确定性的代理之一；Güntay & Hackbarth（2010）认为在债市它主要代理现金流不确定性；Avramov 等（2009）称之为"对下一年盈利的不确定性"。
   - **两者的混合**：Barron 等（1998）推出 D = V(1 − ρ)；Lahiri & Sheng（2010）推出 U = 分歧 + 共同冲击方差。Doukas 等（2006）据此主张必须剔除 V。
   - **信息不对称**：Barron 等（1998）附录把 ρ 解释为信息不对称的综合；Sadka & Scherbina（2007）把离散度与"如何汇总分析师意见"的信息优势联系起来，从而与交易成本挂钩。

2. **谁反驳了谁、用什么证据**
   - Johnson（2004）→ DMS：不需要摩擦，杠杆公司的期权凸性就能解释；证据是"离散度 × 杠杆"交互项显著为负，而离散度单独项不显著。
   - Doukas 等（2006）→ DMS：DMS 捕捉的是不确定性而非分歧；证据是控制 BKLS 不确定性 V 后离散度效应反转，diversity 与收益正相关。他们认为这与 Johnson 一致。
   - Lahiri & Sheng（2010）→ Barron 等（1998）：BKLS 用均值误差平方估计共同不确定性，是事后量；证据是 SPF 密度预测，加入该项后对调查不确定性的解释力下降。这一批评同样涉及 Doukas 等所用的事后估计方法（Lahiri & Sheng 原文点名的是 BKLS 及其在会计文献中的应用）。
   - Sadka & Scherbina（2007）↔ Johnson：不否认 Johnson，但认为流动性证据表明至少部分是"costly arbitrage"导致的错误定价。
   - Avramov 等（2009）→ DMS 与 Johnson：卖空成本代理捕捉不到这一效应；离散度效应在有杠杆和无杠杆公司之间没有差异；效应只存在于评级最差公司的下调期。
   - Garfinkel（2009）→ 以离散度代理投资者分歧的整个文献：以订单价格离散度为基准，离散度与之负相关。
   - Zhang（2006）：离散度单独排序与收益无显著关系（−0.06%，t = −0.25），这与 DMS 的显著负关系并列存在。两文样本期相近，但 Zhang 用股价标准化，DMS 用 |均值| 标准化。
   - Güntay & Hackbarth（2010）：股市复现负关系，债市得到正关系，作者以市场制度差异（债券卖空约束较少）和合约差异来解释。

3. **关于"分歧来源能否区分"的共同点**：本组 13 篇没有一篇在数据上把"信息不同"和"同一信息的解读/模型不同"分开。
   - 模型类文章（Barron、Johnson、Lahiri & Sheng）都假设分歧来自私有信号或信号噪声。其中 Lahiri & Sheng 承认个体误差里包含"interpretation, judgment, and forecasting models"，但仍合并为一个方差项。
   - 实证类文章或声明不需要区分（DMS），或承认无法区分（Zhang 脚注 2；Sadka & Scherbina 称两种解释难以分辨）。
   - 只有 Miller（1977, p.1163）和 Lahiri & Sheng（2010, p.521、p.527）在文字上把"不同估值方法 / 不同预测模型"作为分歧来源单独提出。
