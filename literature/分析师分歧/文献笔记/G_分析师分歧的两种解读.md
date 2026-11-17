# G 组：分析师文献里，"预测离散度低（或高）"被怎样解读——与"信息观点/投机观点"的对照

> 调研日期：2026-11-12/13。只调研和记录，不评价我们的方法，不提研究问题。
>
> **依据等级**
> - 【原文】：本次重新读了本地 PDF 的相关段落（页码为该 PDF 的页码；带 `_WP` 的为工作论文版，页码记作"WP p.x"）。
> - 【原文·本地笔记】：数字和引文出自本地 PDF，但本次没有逐字重读，直接沿用本目录 A–F 组笔记里记录的页码（那些笔记是读原文写的）。
> - 【摘要】：只看到期刊页、RePEc/IDEAS、机构库等摘要页（多为工具转述，非逐字），没有读全文。
> - 【二手】：来自其他论文或综述的转述，转述来源在条目里注明。
> - 【未能确认】：没有找到可靠来源。
>
> **检索范围说明**：英文以 WebSearch 检索 RePEc/IDEAS、EconPapers、期刊页、大学机构库为主；中文只做了两次网页检索，知网无法直接检索，中文覆盖很弱（见第 7 节）。所有网页都只读摘要页，没有下载任何 PDF 到项目目录。检索中未遇到网页里要求执行操作的文字。

---

## 1. 分析师预测离散度被解读成什么？

**格局概述**。以分析师 EPS 预测离散度（下称 AFD）为对象的文献里，至少有五种读法并存，而且经常就同一个"高离散度—低收益"事实互相争夺解释权：
(a) **意见分歧 + 卖空约束 → 高估**（Miller 式；DMS 2002、Berkman 等 2009、Yu 2011、Park 2005、Hong & Sraer 2016），其中 Park 和 Hong & Sraer 明确引入了 Harrison–Kreps 式的投机/转售逻辑；
(b) **不确定性 / 参数风险 / 信息不确定性**（Johnson 2004 用杠杆期权凸性解释负收益；Zhang 2006 视为信息不确定性、与反应不足相互作用；Barron, Stanford & Yu 2009 认为离散度"水平"反映特质不确定性）；
(c) **被定价的风险因子 / 风险溢价**（Anderson, Ghysels & Juergens 2005；Qu, Starks & Yan；Doukas, Kim & Pantzalis 2006 JFQA；Barinov 2013 的"总波动率风险对冲"；目标价离散度的 Li 等 2021）——这一支内部对符号也不一致；
(d) **信息不对称**（BKLS 1998 的共识度 ρ；Barron, Stanford & Yu 2009 认为离散度"变化"反映信息不对称；Sadka & Scherbina 2007 的"谁更会汇总分析师意见"）；
(e) **分析师偏差 / 被压住的坏消息**（DMS 的分析师自我审查；Scherbina 2008；Sadka & Scherbina 2007 引 Scherbina 2004 的"离散度与乐观偏差同时出现"）。
各读法之间的检验手段主要是：交互项（离散度×杠杆、×卖空约束、×流动性、×评级）、控制另一读法的代理后看符号是否反转、以及换市场（债券）看符号。没有一篇论文能直接观测分析师为何不同。

### 1.1 核心论文

#### Diether, Malloy & Scherbina（2002）Differences of Opinion and the Cross Section of Stock Returns
- **出处**：Journal of Finance 57(5): 2113–2141。本地 `Diether_Malloy_Scherbina_2002_JF_Differences_of_Opinion_WP.pdf`（2001-10 WP）。https://doi.org/10.1111/0022-1082.00490
- **研究问题**：AFD 能否预测截面收益；方向支持 Miller 高估、无偏分歧，还是风险补偿。
- **数据与变量**：IBES 未经拆股调整的当期财年 EPS 预测；AFD = 标准差/|均值|，至少 2 位分析师；1983-01 至 2000-11，股价 ≥ $5；按上月 AFD 分五组持有一个月。【原文·本地笔记】
- **论证链条**：① 把分析师离散度当作"投资者意见分歧"的代理；② 若存在卖空约束，悲观者的意见进不了价格（Miller 1977），分歧越大价格越偏向乐观者；③ 因此高 AFD 股票当前被高估、未来收益低；④ 若 AFD 是风险，应得到更高而非更低的收益——数据给出更低收益，故拒绝风险解释；⑤ 附带机制：悲观分析师倾向于停止覆盖（自我审查），使一致预期偏乐观。
- **主要结果**：最低减最高 AFD 组年化收益差 9.48%，最小市值组 16.4%；三因子、四因子不能解释；季度预测误差对 AFD 回归系数 −0.727（t = −33.42），离散度越高一致预期越乐观。【原文·本地笔记，WP p.11–12, 26】
- **与其他读法的关系**：作者明确声明不需要区分分歧来源："our main argument as to why difference of opinions arise and influence prices does not depend on whether investors are asymmetrically informed or simply differ in the way they interpret the same information"（WP p.21 附近，本次 grep 核对）。【原文】
- 本文是后续"不确定性/风险"读法（Johnson、Doukas、Avramov、Barinov）共同的反驳对象。

#### Johnson（2004）Forecast Dispersion and the Cross Section of Expected Returns
- **出处**：Journal of Finance 59(5): 1957–1978。本地 `Johnson_2004_JF_Forecast_Dispersion_Expected_Returns.pdf`（正式版）。https://doi.org/10.1111/j.1540-6261.2004.00688.x
- **研究问题**：能否在无摩擦、无非理性的框架里解释 DMS 的"高离散度、低收益"。
- **数据与变量**：IBES 1983–2001；DISP1 = 标准差/均值，DISP2 = 标准差/账面总资产，转为百分位；核心变量是"离散度 × 杠杆"交互项。【原文·本地笔记】
- **论证链条**（本次重读 p.1959–1962）：① 作者先承认 AFD 测的不是主观不确定性："what is not being measured is a subjective assessment of earnings uncertainty… a group of forecasters could all agree about the mean of the distribution while each being hugely uncertain"（p.1959）；② 采用"多信号"解读：观察者把每个分析师的估计当作真实值的独立噪声信号，汇总后后验方差与信号间方差成正比（p.1959）；③ 因此 AFD 代表对公司当前价值的"参数风险"，且是特质性的、本身不被定价；④ 股权是杠杆公司资产的看涨期权，特质参数风险越大，期权的 delta 下降，股权预期收益越低；⑤ 于是"高 AFD → 低预期收益"无需卖空约束或非理性。【原文】
- **主要结果**：交互项显著为负（DISP1：−0.0090，t = 2.36；DISP2：−0.0131，t = 2.70），离散度单独项不显著；无负债公司在离散度第 99 和第 1 百分位的预期收益基本相同。【原文·本地笔记，p.1973】
- **作者自述的疑问**：多信号解读"raises a number of questions… why do the forecasters not themselves aggregate their joint information? … This realization highlights the need for a more complete understanding of exactly how forecast differences arise"（p.1959）。【原文】
- **与本组主题的关系**：这是分析师文献里"把分析师当作独立信号、离散度低 = 参数不确定性低"的最明确表述；但注意其定价含义是"离散度低 → 对杠杆公司预期收益**更高**"，不是"风险更低所以价格更高"。

#### Barron, Kim, Lim & Stevens（1998，BKLS）Using Analysts' Forecasts to Measure Properties of Analysts' Information Environment
- **出处**：The Accounting Review 73(4): 421–433。本地 `Barron_Kim_Lim_Stevens_1998_TAR_...pdf`。
- **研究问题**：可观测的离散度 D、均值预测误差平方 SE、分析师人数 N，各自对应什么理论性质。
- **模型**：所有分析师共享公共信息（精度 h），各有私有信号（精度 s）；预测为贝叶斯后验均值。
- **论证链条**（本次重读 p.425–429）：① 定义总不确定性 V、平均误差协方差 C（共同不确定性）、共识度 ρ = C/V；② 推出 V = C + D、SE = C + D/N（式 11–12）：私有信息带来的个体误差在均值预测里被平均掉，共同误差不会；③ 由此 D = V(1 − ρ)（式 13）：**离散度低既可能是不确定性低，也可能是共识度高（分析师依赖相同的信息）**；④ 作者特意指出与 ALV（1995）的不同："ALV suggested that forecast dispersion indicates a lack of precision in the mean forecast. Equation (13) suggests, however, that forecast dispersion can be caused by low levels of consensus, which equation (14) suggests can be associated with a decrease in the error in the mean forecast"（p.426）——即在 V 给定时，离散度低（ρ 高）反而对应均值预测误差更大；⑤ 等精度特例下 ρ = h/(h + s)，即公共信息在总精度中的占比（p.427）。【原文】
- **局限**：羊群行为会模糊公共/私有信息的区分，"The impact of herding behavior on the construct validity of the measures developed in our model is an open issue"（p.429）。【原文】
- **与本组主题的关系**：这是分析师文献里讨论"分析师信息相关性如何影响共识准确度"的基础框架（见第 2 节），其结论方向是：一致可能来自共同信息，而共同信息不能通过平均消除。

#### Anderson, Ghysels & Juergens（2005）Do Heterogeneous Beliefs Matter for Asset Pricing?
- **出处**：Review of Financial Studies 18(3): 875–924。https://econpapers.repec.org/RePEc:oup:rfinst:v:18:y:2005:i:3:p:875-924 ；本地无。
- **研究问题**：信念异质性是否影响收益、是否是传统定价模型里被定价的因子。【摘要】
- **数据与变量**：用分析师对短期和长期盈利的分歧构造新的异质信念度量，并估计一个以分析师预测代理投资者信念的结构模型。【摘要】
- **论证链条**：① 一般均衡模型中信念离散度作为风险因子进入定价；② 用分析师预测构造的离散度因子检验其是否被定价；③ 检验离散度能否帮助解释资产定价谜题。【摘要】
- **主要结果**：Banerjee（2011, p.3028）转述：该文发现"negative relation between expected returns and short-term dispersion, but a positive relation between returns and long-term dispersion"。【二手，来源 Banerjee 2011 本地 PDF p.3028，本次核对】其他论文把它与 Qu, Starks & Yan 一起列为"离散度越高预期收益越高"的证据。【二手】具体系数【未能确认】。
- **与其他读法的关系**：与 DMS 的 Miller 读法方向相反，属"风险因子"读法。

#### Berkman, Dimitrov, Jain, Koch & Tice（2009）Sell on the News
- **出处**：Journal of Financial Economics 92(3): 376–399。https://econpapers.repec.org/RePEc:eee:jfinec:v:92:y:2009:i:3:p:376-399 ；本地无。
- **研究问题**：如果盈余公告能缩小意见分歧，高分歧且难卖空的股票在公告前后是否出现价格修正。【摘要】
- **论证链条**：① 从 Miller（1977）出发：高分歧 + 卖空约束 → 高估；② 盈余公告降低分歧 → 高估应在公告附近消解；③ 因此高分歧股票公告期收益应更低，且在最难卖空的股票里更低。【摘要】
- **数据与变量**：五种意见分歧代理（摘要未列明，是否全部基于分析师【未能确认】）。
- **主要结果**：高分歧股票公告期收益显著低于低分歧股票，最难卖空子样本更负；卖空约束紧的高分歧股票公告前上涨、公告后下跌更多；控制规模、市净率、杠杆、公告溢价、PEAD、动量和分析师预测偏差后仍成立。【摘要】
- **反驳**：Keskek, Rees & Thomas（2013, JBFA）称控制盈余消息水平后，分歧与收益的关系消失或与 Miller 预测相反，并归因于悲观的管理层指引。【摘要（检索摘要转述）】

#### Barron, Stanford & Yu（2009）Further Evidence on the Relation between Analysts' Forecast Dispersion and Stock Returns
- **出处**：Contemporary Accounting Research 26(2)（页码【未能确认】）。https://doi.org/10.1506/car.26.2.1 ；本地无。
- **研究问题**：离散度"水平"与"变化"是否度量不同的构念。【摘要】
- **论证链条**：① 已有文献：离散度水平与未来收益负相关，离散度变化与同期收益负相关；② 用 BKLS 框架把离散度拆成不确定性成分和信息不对称成分；③ 水平主要反映不确定性，变化主要反映信息不对称；④ 水平中的不确定性成分是特质风险，与未来收益负相关，支持"特质不确定性提高公司期权价值"（即 Johnson 式）而非"水平反映信息不对称"。【摘要】
- **数据**：据 TCU 新闻稿，1986–2003 年美国大公司季度与年度数据。【二手】

### 1.2 次要论文

- **Doukas, Kim & Pantzalis（2006a）Divergence of Opinion and Equity Returns**，JFQA 41(3): 573–606（本地有）。用 BKLS 把 AFD 拆成不确定性 V 与 diversity (1 − ρ)；原始 AFD 高减低 −0.41%/月（t = −5.98），控制 V 后反转为正，diversity 高减低在各市值组显著为正；作者据此"refute the view that dispersion in analysts' forecasts reflects divergence of opinion"（p.573）。注意其 V 与 ρ 都用事后误差估计（p.577）。【原文·本地笔记】
- **Doukas, Kim & Pantzalis（2006b）Divergence of Opinion and Equity Returns under Different States of Earnings Expectations**，Journal of Financial Markets 9(3): 310–331（本地无）。以 AFD 为分歧：分析师乐观时高分歧折价、悲观时高分歧溢价；**高估出现在"离散度低且预测乐观"时**，作者解释为投资者高估自己的预见力，在乐观预期下买入低离散度的"sure winners"。https://researchconnect.stonybrook.edu/en/publications/divergence-of-opinion-and-equity-returns-under-different-states-o/ 【摘要】
- **Sadka & Scherbina（2007）Analyst Disagreement, Mispricing, and Liquidity**，JF 62(5): 2367–2403（本地 WP）。论证链：分析师分歧与异常高的乐观偏差同时出现，有限理性投资者没有充分折扣（引 Scherbina 2004）→ 高估；高估持续是因为"some investors are better informed than the market maker about how to aggregate analysts' opinions"，分歧越大做市商逆向选择越重、交易成本越高（WP 摘要页）；扣除交易成本后做空高离散度组合不再盈利。【原文】
- **Scherbina（2008）Suppressed Negative Information and Future Underperformance**，Review of Finance 12(3): 533–565（本地无）。用停止修正年度预测的分析师比例估计被压住的负面信息，它预测负盈余意外和公告期负反应。该文与 AFD 的直接关系【未能确认】。https://ideas.repec.org/a/oup/revfin/v12y2008i3p533-565.html 【摘要】
- **Avramov, Chordia, Jostova & Philipov（2009）**，JFE 91(1): 83–101（本地有）。离散度效应集中在评级最差、市值合计不到 5% 的公司，且只在评级下调期；反驳 DMS（卖空成本代理捕捉不到）与 Johnson（有无杠杆公司效应无差别，p.84）；把 AFD 视为"对下一年盈利的不确定性"，只是估值中的一个输入。【原文·本地笔记】
- **Zhang（2006）Information Uncertainty and Stock Returns**，JF 61(1): 105–137（本地有）。AFD 作为六个信息不确定性代理之一；单独排序高减低 −0.06%（t = −0.25）；坏消息检验中 AFD 是唯一不显著的代理。【原文·本地笔记】
- **Yu（2011）**，JFE 99(1): 162–183（本地有）。用 LTG 预测离散度自下而上加总市场分歧，1 个标准差上升对应未来一年市场超额收益下降 6.6 个百分点；按 Miller 框架解读，脚注引 Harrison & Kreps、Scheinkman & Xiong，但实证没有检验转售期权本身。【原文·本地笔记；脚注本次 grep 核对】
- **Park（2005）Stock Return Predictability and the Dispersion in Earnings Forecasts**，Journal of Business 78(6): 2351–2376（本地无）。月度分析师预测的离散度在中期限对市场收益有很强预测力；作者认为离散度应解读为投资者预期差异而非风险。https://econpapers.repec.org/RePEc:ucp:jnlbus:v:78:y:2005:i:6:p:2351-2376 【摘要】Banerjee（2011, p.3028）转述：Park "extends the Harrison and Kreps (1978) model to argue that speculative pricing leads to the negative return-disagreement relation"。【二手，本地 PDF 核对】
- **Hong & Sraer（2016）Speculative Betas**，JF 71(5): 2095–2144（本地 WP）。用 IBES LTG 离散度（β 加权）作总体分歧；OLG 扩展中"Investors anticipate that high beta assets are more likely to experience binding short-sales constraints in the future and hence have a potentially higher resale price… (Harrison and Kreps (1978), Morris (1996), Scheinkman and Xiong (2003)…)"（WP p.4）；高分歧期 β 与收益呈倒 U 形。【原文】这是本次找到的、在分析师数据上直接写出转售价格机制的论文。
- **Yin（年份【未能确认】）Resale Value and Security Prices**（Purdue 博士论文"Two essays in behavioral finance"之一）。检验投资者是否外推分歧趋势来估计转售期权价值；AFD 增长最高十分位相对最低十分位在 1、6、12 个月内跑输，多空年化 11.52%。https://docs.lib.purdue.edu/dissertations/AAI3636694 【摘要】
- **Barinov（2013）Analyst Disagreement and Aggregate Volatility Risk**，JFQA 48(6): 1877–1900（本地无）。高 AFD 公司在总波动率上升时跑赢 CAPM，是总波动率风险的对冲，因而预期收益低；总波动率风险因子解释高低分歧公司的异常收益差；效应在实物期权多、卖空约束高的公司更强。https://ideas.repec.org/a/cup/jfinqa/v48y2013i06p1877-1900_00.html 【摘要】
- **Qu, Starks & Yan（2003/2004 WP）Risk, Dispersion of Analyst Forecasts and Stock Returns**。Banerjee（2011, p.3028, 脚注 p.3050）转述：认为 AFD 代理信息不确定性但推出正向关系，用滞后股价标准化后得到正关系。【二手】标题与正式发表情况【未能确认】。
- **Güntay & Hackbarth（2010）**，JBF 34(10)（本地有）。债券市场中 AFD 越高信用利差越高（1 个标准差 +19bp），作者认为主要代理现金流不确定性；同一批发行人的股票仍复现负关系。【原文·本地笔记】
- **Sheng & Thevenot（2012）A New Measure of Earnings Forecast Uncertainty**，JAE 53(1): 21–33（本地无）。仿 Lahiri & Sheng，用"离散度 + GARCH 预测的均值误差方差"度量盈利预测不确定性，理由是不确定性有共同和特质两部分，离散度在共同不确定性大时是噪声代理。https://econpapers.repec.org/RePEc:eee:jaecon:v:53:y:2012:i:1:p:21-33 【摘要】
- **Verardo（2009）Heterogeneous Beliefs and Momentum Profits**，JFQA 44(4): 795–822（本地无）。以 AFD 为信念异质性，高异质性组合动量利润更大，控制信息扩散速度、基本面不确定性、信息精度后仍成立。【摘要】
- **Li, Feng, Yan & Wang（2021）Dispersion in Analysts' Target Prices and Stock Returns**，NAJEF 56: 101385（本地无）。目标价离散度与未来 24 个月收益**正**相关，次月最高减最低十分位可超 2%，伴随更高的未来风险，作者解读为风险。https://pure.psu.edu/en/publications/dispersion-in-analysts-target-prices-and-stock-returns/ 【摘要】
- **Garfinkel & Sokobin（2006）**，JAR 44(1): 85–112。用公告日未被解释的成交量作意见分歧，公告后收益随分歧**上升**，作者联系 Varian（1985）"分歧是额外风险因子"。注意它**不是**分析师离散度。【摘要】
- **专业预测者而非股票分析师：Carlin, Longstaff & Matoba（2014）**，JFE 114(2)（本地 WP）。MBS 交易商提前还款模型的情景斜率离散度作为分歧，分歧变化对 MBS 收益有正向预测（t = 2.88），作者解读为正风险溢价，并基于制度背景认为分歧来自"model choice, underlying assumptions, and interpretation"（WP p.3）。【原文·本地笔记】

### 1.3 各读法如何互相检验（事实汇总）

| 检验 | 谁对谁 | 证据 |
|---|---|---|
| 离散度×杠杆 | Johnson → DMS | 交互项负、单独项不显著【原文·本地笔记】 |
| 控制 BKLS 不确定性 V | Doukas 等 2006a → DMS | 离散度效应反转为正【原文·本地笔记】 |
| 按评级与下调期分样本 | Avramov 等 → DMS、Johnson | 效应只在最差评级下调期；有无杠杆无差别【原文·本地笔记】 |
| 交易成本/流动性 | Sadka & Scherbina | 扣成本后不盈利；离散度×价格冲击最显著【原文】 |
| 总波动率风险因子 | Barinov → DMS | 因子解释收益差【摘要】 |
| 离散度水平 vs 变化 | Barron, Stanford & Yu | 水平 = 不确定性，变化 = 信息不对称【摘要】 |
| 换市场 | Güntay & Hackbarth | 债市正、股市负【原文·本地笔记】 |
| 公告前后 | Berkman 等 vs Keskek 等 | 支持 Miller vs 控制盈余消息后消失【摘要】 |
| 按预期乐观/悲观分组 | Doukas 等 2006b | 低离散度+乐观才高估【摘要】 |
| 标准化方式 | Banerjee 2011 转述 | DMS 用 |均值| 得负，Qu-Starks-Yan 用股价得正【二手】 |

---

## 2. 分析师共识为什么准或不准？

**格局概述**。以分析师为对象讨论"共识质量"的文献，主线是**公共信息 vs 私有信息**，而不是"模型/方法"。BKLS（1998）给出框架：共识误差 = 共同误差 + 私有误差/N，只有私有（独立）部分能被平均掉；Kim, Lim & Shaw（2001）据此证明简单均值高估了共同信息的权重。实证上分三支：(i) 羊群/大胆预测（Hong-Kubik-Solomon、Clement-Tse、Jegadeesh-Kim、Bernhardt 等）——羊群使分析师的误差相关、降低共识信息量；(ii) 众包平台（Da & Huang 2020 用实验证明"独立"预测使共识更准；Jame 等 2016）；(iii) 分析师背景多样性（Merkley, Michaely & Pacelli 2020 文化多样性 → 共识更准；Fang & Hope 2021 团队内部多样性）。**没有找到**一篇以分析师为对象、度量"不同分析师使用的估值方法/推理模型有多不同"并检验其与共识准确度关系的论文；最接近的是 Merkley 等（背景多样性，机制被归为信息流动）、Fang & Hope（团队多样性与更多使用 DCF）和 Cavezzali & Rigoni（单篇研报内部使用多种方法求平均）。

### 2.1 核心论文

#### Kim, Lim & Shaw（2001）The Inefficiency of the Mean Analyst Forecast as a Summary Forecast of Earnings
- **出处**：Journal of Accounting Research 39(2): 329–335。https://ideas.repec.org/a/bla/joares/v39y2001i2p329-335.html ；本地无。
- **研究问题**：简单平均的一致预期是否是对盈利的有效汇总。
- **论证链条**：① 在 BKLS 式框架下，每个分析师的预测 = 公共信息与私有信息的加权；② 简单平均后，公共信息在每个预测里都出现，于是被重复计入；③ 因此均值预测"assigning too much weight to analysts' common information relative to their private information"；④ 更精确的汇总是"当前均值 + 均值变化的正倍数"。【摘要】
- **数据**：分析性论文，是否有实证【未能确认】。
- **与本组主题的关系**：这是分析师文献里关于"分析师信息相关 → 共识不是独立证据的简单合并"的直接分析结论。

#### Da & Huang（2020）Harnessing the Wisdom of Crowds
- **出处**：Management Science 66(5): 1847–1867。https://econpapers.repec.org/RePEc:inm:ormnsc:v:66:y:2020:i:5:p:1847-1867 ；本地无。
- **研究问题**：群体估计什么时候准；以 Estimize 众包盈利预测为对象。【摘要】
- **数据与变量**：Estimize 平台用户活动追踪，度量每个预测者在提交前浏览了多少公共信息（他人预测）；随机实验：Estimize 对部分股票和用户限制信息。【摘要】
- **论证链条**：① 浏览更多公共信息的用户降低了自己私有信息的权重（羊群）；② 个人预测因此更准；③ 但私有信息没有进入平均，共识反而变差；④ 随机实验（限制可见信息）证实"独立"预测带来更准确的共识；⑤ Estimize 于 2015 年 11 月改为盲测平台。【摘要】
- **结论**：鼓励独立意见能更好地利用群体智慧；更多公开信息并不总能改善群体决策。【摘要】
- **注意**：这里的"独立"是**信息独立**（不看他人预测），不是方法或模型不同；样本是众包预测者而非卖方分析师。新闻稿中"盲测共识优于华尔街共识"的比例有 57% 与 64% 两种说法【未能确认】。

#### Merkley, Michaely & Pacelli（2020）Cultural Diversity on Wall Street: Evidence from Consensus Earnings Forecasts
- **出处**：Journal of Accounting and Economics 70(1): 101330。https://ideas.repec.org/a/eee/jaecon/v70y2020i1s016541012030032x.html ；WP：SFI N°19-07。本地无。
- **研究问题**：覆盖同一公司的分析师文化背景差异，是否影响共同预测任务（一致预期）的结果。【摘要】
- **识别**：利用卖方分析师多样性的外生冲击，并辅以面板回归。【摘要】（外生冲击的具体来源与样本规模【未能确认】）
- **论证链条**：① 文化多样性 → 分析师之间带来不同的信息与视角；② 多样性提高个体预测质量并增加电话会议参与和互动（信息流动渠道）；③ 多样性还对应教育和职业兴趣的多样性；④ 一致预期质量提高。【摘要】
- **主要结果**："increases in analyst cultural diversity positively affect the quality of the consensus earnings forecast"；WP 版摘要称提高共识准确度、降低乐观偏差**和离散度**；控制其他维度多样性后稳健。【摘要】
- **注意**：WP 版报告多样性**降低**离散度——即在该文中，多样性与"更准、更不分散"同向出现；作者归因于信息流动，而不是"不同模型相互印证"。

### 2.2 次要论文

- **Barron, Byard & Kim（2002）Changes in Analysts' Information around Earnings Announcements**，TAR 77(4): 821–846（本地无）。用 BKLS 代理，发现盈余公告前后分析师之间的共同信息下降，公告后个体预测中的特质信息上升，修正人数越多上升越大；支持"会计披露触发分析师生产私有信息"（Kim & Verrecchia 1994, 1997）。https://pure.psu.edu/en/publications/changes-in-analysts-information-around-earnings-announcements/ 【摘要】
- **Botosan, Plumlee & Xie（2004）The Role of Information Precision in Determining the Cost of Equity Capital**，RAST 9(2): 233–259（本地无）。用分析师离散度和准确度构造的公共/私有信息精度：公共信息精度越高资本成本越低，但这一效应被"私有信息精度与资本成本正相关"所抵消且有余；两者正相关，遗漏任一会有偏。【摘要】讨论者 Williams 指出私有信息代理度量的是分析师揭示的信息而非内部人信息。【二手】
- **Hong, Kubik & Solomon（2000）Security Analysts' Career Concerns and Herding of Earnings Forecasts**，RAND 31(1): 121–144。经验少的分析师因预测不准更易被解雇，也更倾向向一致预期靠拢（羊群），支持职业关切导致羊群的模型。【摘要】
- **Clement & Tse（2005）Financial Analyst Characteristics and Herding Behavior in Forecasting**，JF 60(1): 307–341。把预测分为羊群与大胆两类；大胆预测比羊群预测更准；羊群修正与预测误差关联更强；作者认为大胆预测"incorporate analysts' private information more completely"。【摘要】
- **Jegadeesh & Kim（2010）Do Analysts Herd? An Analysis of Recommendations and Market Reactions**，RFS 23(2): 901–937。新检验支持评级向共识聚集；偏离共识的修正价格反应更强；**大券商分析师、评级分歧较小的股票上的分析师、修正较少者更易羊群**。【摘要】样本 1993–2005 年【二手，Emory 新闻稿】。
- **Bernhardt, Campello & Kutsoati（2006）Who Herds?**，JFE 80(3): 657–675。摘要页无摘要；他文转述其检验对相关信息与共同行业冲击稳健，并发现"反羊群"（偏离共识）。【二手】
- **Chen & Jiang（2006）Analysts' Weighting of Private and Public Information**，RFS 19(1): 319–355。分析师平均对私有信息赋权过高；比共识乐观时过度赋权更多，比共识悲观时赋权较少甚至不足；作者认为激励比行为偏差更重要。【摘要】
- **Jame, Johnston, Markov & Wolfe（2016）The Value of Crowdsourced Earnings Forecasts**，JAR 54(4): 1077–1110。51,012 条 Estimize 预测，对预测盈利和度量市场预期有增量价值，参与人数越多结果越强，共识修正有显著两日收益。【摘要】与 IBES 共识的准确度比较【未能确认】（SMU 新闻稿称群体价值来自规模与信息多样性【二手】）。
- **Fang & Hope（2021）Analyst Teams**，RAST 26(2): 425–467。手工收集 5 万余份研报：团队比个人预测更准；准确度与团队在卖方经验、教育背景、性别上的多样性正相关；更多样的团队更可能发布现金流预测并使用 DCF。【摘要】这是**团队内部**多样性，不是不同分析师之间的共识。
- **Kumar, Rantala & Xu（2022）Social Learning and Analyst Behavior**，JFE 143(1): 434–461。分析师对某公司的乐观程度与同组合内其他公司"同伴"分析师近期的预测误差负相关，社会学习提高准确度。【摘要】（提示分析师之间并不独立）
- **Palley, Steffen & Zhang（2025）The Effect of Dispersion on the Informativeness of Consensus Analyst Target Prices**，Management Science 71(3): 2264–2288。目标价离散度低时，一致目标价隐含收益与未来收益正相关；离散度高时强烈负相关；作者归因于坏消息后部分分析师因激励迟迟不下调目标价；多空策略年化超 11%，负关系集中在散户关注高的股票。https://ideas.repec.org/a/inm/ormnsc/v71y2025i3p2264-2288.html 【摘要】
- **Brown, Call, Clement & Sharp（2015）**，JAR 53(1)（本地 WP）。问卷中"其他分析师的预测"在 11 项输入中排第 10（2.16 分），"其他分析师的评级"排末位；访谈中分析师称会关注离群同行"如何思考"以复核自己。【原文·本地笔记】
- **Cavezzali & Rigoni（2013）Financial Analysts' Forecast Accuracy: Do Valuation Methods Matter?**，Ca' Foscari 管理系 WP No.9。编码 1,650 份研报的估值方法：目标价只依赖一个主方法时准确度更低；基本面类与乘数类方法准确度相近；作者建议用多种方法求平均。https://econpapers.repec.org/paper/vnmwpdman/45.htm 【摘要】这是**单篇研报内部**的方法多样性，不涉及分析师之间。

### 2.3 对"方法/模型越多样、共识越准"的直接回答
- 以"信息独立性"为对象的直接证据存在：Da & Huang（2020，实验）、Clement & Tse（2005）、Kim, Lim & Shaw（2001，分析性）、BKLS（1998，框架）。
- 以"分析师背景多样性"为对象的证据存在：Merkley 等（2020，文化多样性 → 共识更准，机制归为信息流动）；Fang & Hope（2021，团队内部）。
- 以"分析师之间估值方法/推理模型多样性"为对象、检验共识准确度的论文：**未找到**。搜过的关键词：analyst valuation model diversity consensus accuracy；heterogeneity of analysts' valuation methods consensus；analysts different models same forecast；diversity of analyst methods wisdom of crowds；model diversity sell-side consensus。

---

## 3. 分析师对同一公开信息的不同解读

**格局概述**。这支文献以盈余公告为事件，用分析师预测修正推断"差异化解读"，并主要与**成交量**而不是价格水平联系。代表路径是：Kandel & Pearson（1995）提出用分析师修正方向相反且交叉/发散的配对比例识别"似然函数不同"；Barron（1995）、Bamber, Barron & Stober（1997, 1999）把"事前离散度""离散度变化""信念交错（jumbling）"分开并与成交量回归；Barron 等（2002, 2005, 2017）用 BKLS 框架发现公告后分析师特质信息上升、共识下降，且共识下降伴随成交量上升。价格方面，Berkman 等（2009）与 Giannini 等（2019，非分析师数据）把公告前后分歧变化与公告期收益相连。最新的 Kwon（2025，博士论文）用研报文本显示公告后分析师引用的信息不同。

### 3.1 核心论文

#### Kandel & Pearson（1995）Differential Interpretation of Public Signals and Trade in Speculative Markets
- **出处**：Journal of Political Economy 103(4): 831–872。https://ideas.repec.org/a/ucp/jpolec/v103y1995i4p831-72.html ；本地无（文献清单列为待下载）。
- **研究问题**：公告附近成交量与收益的关系是否与"所有人对公开信息解读相同"的假设相容。【摘要】
- **论证链条**：① 实证事实：公告附近常有大量交易而价格几乎不动；② 在解读相同的模型里，无价格变化时缺乏信息性交易的理由；③ 建立"交易者对公开信号解读不同"的模型，能复制量价关系；④ 用券商分析师的盈利预测修正检验标准的信念修正模型：若似然函数相同，两位分析师不应在同一消息后反向修正并交叉或拉开，数据中这种配对存在，拒绝解读相同。【摘要；第④步的配对定义来自 Bamber 等 1999 本地 WP p.2–3 对 KP p.851–856 的转述，属【二手】】
- **关键数字**：KP 指标均值 13.4%（KP 表 4）。【二手，Bamber 等 1999 WP p.9】
- **作者承认的局限**（经 Bamber 等转述）：指标保守，漏掉同向修正中的解读差异；重大意外后分析师往往同向修正（KP p.860）。【二手】

#### Bamber, Barron & Stober（1999）Differential Interpretations and Trading Volume
- **出处**：JFQA 34(3): 369–386。本地 `Bamber_Barron_Stober_1999_JFQA_..._WP.pdf`（1999-01 WP）。
- **数据与变量**：IBES 1984–1994，季度盈余公告前 45 天内、公告后 30 天内同一分析师修正的年度 EPS 预测，每个公告至少 5 名分析师，1,972 个公告；KP = 两两配对中"反向修正且交叉或发散"的比例；因变量为 [−1,+1] 换手率。【原文】
- **论证链条**（本次重读 WP p.2–5）：① 差异化信念修正有两个来源：对新闻的不同解读，或披露前私有信息精度不同（精度高者给新闻较低权重）；② 同向修正中的交叉/发散可由精度不同产生，不计入；③ 只有反向修正且交叉/发散"arise only from differential interpretations"；④ 测量误差随总体信念变化增大而增大，故先看价格变化最小的样本；⑤ 在价格几乎不动时，解读差异是交易的主要理由（引 Kim & Verrecchia 1991 的反例）。【原文】
- **主要结果**：价格变化最小十分位（N = 197）中 ln(KP) 系数 0.031（p = 0.02），KP 增 1 个标准差对应成交量增 2.5%；全样本不显著（0.005，p = 0.20）；成交量高于公司非公告期均值的 1,233 个公告中 0.018（p < 0.01）。【原文·本地笔记】
- **局限**：KP 基于分析师对年度盈利的预期，而非投资者对现金流的预期（WP p.12）；即使解读差异是成交量唯一来源，模拟中 ln(KP) 也只解释约 10% 的 ln(VOL)（附录）。【原文·本地笔记】

#### Kwon（2025）Forecast Dispersion and Information Selection（芝加哥大学博士论文）
- **出处**：https://knowledge.uchicago.edu/record/15690 （DOI 10.6082/uchicago.15690，受限访问）。导师 Stefan Nagel。本地无。
- **研究问题**：为什么分析师预测离散度在盈余公告后上升、在两次公告之间下降。【摘要】
- **论证链条**：① 公告揭示的公开信息是高维的；② 分析师修正时各自选用公开信息的不同子集（"information selection"）；③ 研报文本显示分析师在公告后修正时**引用的信息不同**；④ elastic net 回归印证这一点，并显示存在一个分析师都拿来对比的共同基准；⑤ 共同基准解释了两次公告之间离散度下降；⑥ 用一个"共同基准下做预测"的简单模型刻画上述动态。【摘要】
- **结论**："信息选择"是分歧的重要来源，尤其在信息高维时。【摘要】
- **注意**：全文受限，文本度量方法与数字【未能确认】。这是本次找到的唯一一篇用**研报文本**说明"同一公开信息、不同分析师关注不同部分"的分析师文献。

### 3.2 次要论文
- **Barron（1995）Trading Volume and Belief Revisions That Differ Among Individual Analysts**，TAR 70(4): 581–597。用分析师当前与之前预测相对位置的相关性度量差异化信念修正，多元回归中它在事前离散度之外解释成交量，支持 Karpoff（1986）"交易来自事前信念差异和差异化修正"。【摘要】
- **Bamber, Barron & Stober（1997）Trading Volume and Different Aspects of Disagreement Coincident with Earnings Announcements**，TAR 72(4): 575–597（卷期页据 Springer 2020 年论文参考文献【二手】）。Dinh & Gajewski（2007, CIRANO）转述其三个分歧维度：公告前信念离散度、公告前后离散度变化、信念交错（某投资者信念相对他人的移动）；2023 年 TAR 论文转述其发现公告期成交量随公告前分歧和公告期分歧增加而上升。【二手】本地无。
- **Barron, Harris & Stanford（2005）Evidence that Investors Trade on Private Event-Period Information around Earnings Announcements**，TAR 80(2): 403–421。用 BKLS 代理：使分析师私有信息增加、共识下降的公告伴随更高成交量，作者解读为投资者交易"只有与公告内容结合才有用"的事件期私有信息。【摘要】
- **Barron, Byard & Yu（2017）Earnings Announcement Disclosures and Changes in Analysts' Information**，CAR 34(1): 343–373。资产负债表和分部披露使分析师预测更多基于私有信息；管理层预测增加共同信息；电话会议问答中这些项目讨论更多，作者解读为分析师在为市场解读披露。【摘要】
- **Atiase & Bamber（1994）**，JAE 17(3): 309–330。成交量反应随价格反应和事前信息不对称增大而增大（Kim & Verrecchia 1991a）。事前信息不对称是否用分析师离散度度量【未能确认】。【摘要】
- **Harris & Raviv（1993）**，RFS 6(3)（本地有，扫描件）。理论：两组交易者对同一公开信号的似然参数不同；推出平均预测绝对变化与成交量正相关，作者对应 Ziebart（1990）"成交量与分析师平均预测绝对变化正相关"（p.485–486）。【原文·本地笔记】
- **非分析师数据的对照**：Giannini, Irvine & Shu（2019, JFM 42，本地有）用 StockTwits 与新闻情绪符号定义分歧，发散（4,530 个公告）多于收敛（3,370 个），公告前分歧组公告期成交量高 1.08%（t = 14.11），"前分歧后一致"组收益低于"前一致后分歧"组 0.81%（t = 4.48），作者解读为支持 Miller。【原文·本地笔记】Cookson, Engelberg & Mullins（2023, RFS）发现盈余公告附近选择性接触约为平时两倍，称这是对 KP"公告后分歧上升"的互补机制。【原文·本地笔记】

### 3.3 解读差异如何与成交量、价格连接（事实汇总）
- 与**成交量**：KP 指标（Bamber 等 1999）、相对位置相关性（Barron 1995）、BKLS 共识下降（Barron 等 2005）都与公告期成交量正相关；Harris & Raviv（1993）理论预测平均预测变化与成交量正相关。
- 与**价格**：分析师数据里的"解读差异"文献几乎不检验价格水平或未来收益；价格方面的分析师离散度证据来自 Miller 支线（Berkman 等 2009），其设定是"公告缩小分歧"，与 KP 一支"公告扩大解读差异"方向不同。两支之间的直接对话【未找到】。

---

## 4. 有没有文献直接涉及"预测一致但依据不同"或"依据相同但预测不同"？

**结论先行**：以分析师为对象、**直接比较"预测值接近而理由/估值模型/关注因素不同"**的论文，**未找到**。
搜过的关键词（英文）：analysts same forecast different reasoning；analysts agree forecasts disagree justification；same target price different valuation method；dispersion of valuation methods analysts；analyst report text dispersion same firm；text-based disagreement among analyst reports；topic dispersion analyst reports；recommendation dispersion vs earnings forecast dispersion；analysts consensus different models；valuation model heterogeneity analysts pricing。中文："研报 逻辑 分歧 一致预期"未在网页检索中得到学术结果。

以下是涉及其中一侧（"依据"或"依据差异"）的最接近文献。

### 4.1 核心论文

#### Décaire, Sosyura & Wittry（2024）Resolving Estimation Ambiguity（依据相同、参数选择不同 → 估值分歧）
- **出处**：WP（Ohio State Dice Center 2024-19；本地 `Decaire_Sosyura_Wittry_2024_WP_Resolving_Estimation_Ambiguity.pdf`）。https://ssrn.com/abstract=4953859
- **数据与变量**：42 家券商提及 DCF/WACC 的研报，2000–2023 年第一季度，同一公司-年至少 2 位分析师，45,992 篇；两两之间 |WACC_A − WACC_B| 等。【原文·本地笔记】
- **论证链条**：① 几乎所有人用同一模型（CAPM）和同一批公开数据；② 但 β 窗口、ERP 等需要估计的输入有多种方法；③ 分析师在可观测输入上趋同（美国 98% 用 10 年期国债），在需估计的输入上分化；④ 由此产生同一公司同一时点的折现率分歧；⑤ 这种分歧与市场分歧（成交量）相关。【原文·本地笔记】
- **主要结果**：两位分析师股权成本平均差 180bp，WACC 差 140bp；股权成本分歧 79% 来自 β×ERP；β 的公司内差异中分析师个人偏好解释 28%、券商 1%；WACC 分歧增 1 个标准差，异常成交量增约 4%。【原文·本地笔记】
- **关于"不同依据是否相互抵消"**（本次 grep 核对 WP p.2–3 与表 2）："higher discount rates are not offset by other modeling choices, such as higher explicit or terminal growth rates. For example, in 45% of the valuation models, an above-consensus estimate of the discount rate is associated with a below-consensus estimate of the terminal growth rate"。【原文】即该文检验过输入差异是否在估值层面相互抵消，结论是不系统抵消。
- **与本节问题的关系**：这是"依据（模型）相同、参数不同 → 结论不同"一侧的直接证据；它没有比较目标价是否一致。

#### Kecskés, Michaely & Womack（2017）Do Earnings Estimates Add Value to Sell-Side Analysts' Investment Recommendations?（同一评级变动、依据不同）
- **出处**：Management Science 63(6): 1855–1871。https://econpapers.repec.org/RePEc:inm:ormnsc:v:63:y:2017:i:6:p:1855-1871 ；本地无。
- **研究问题**：评级变动由盈利预测修正驱动，与由估值方法中的非盈利成分驱动，市场反应和信息含量是否不同。【摘要】
- **论证链条**：① 评级变动反映分析师与市场估值的分歧；② 这种分歧可来自盈利预测不同，也可来自估值方法中的非盈利成分（增长率、折现等）；③ 比较同样的评级变动在"伴随盈利预测修正"与"不伴随"两种情况下的市场反应。【摘要】
- **主要结果**：伴随盈利修正的上调比不伴随者初始反应约多 1.3%，下调约多 −2.8%；后续漂移更大（反应不足）；作为交易策略风险调整后约每月 3%，显著高于非盈利驱动的变动；作者认为盈利驱动的变动较少受认知和激励偏差影响。【摘要】
- **与本节问题的关系**：这是本次找到的唯一一篇按"同一结论（评级变动）的依据类型"区分并检验市场反应的分析师论文；但它比较的是单个分析师的变动，而非多位分析师结论一致时的依据差异。

### 4.2 次要论文（研报依据、估值模型选择）
- **Doukas, Kim & Pantzalis（2006b）**（见 1.2）：低离散度 + 乐观预期时高估，作者解读为投资者把低离散度股票当作"sure winners"。它只观测预测值，不观测依据。【摘要】
- **Kwon（2025）**（见 3.1）：同一公告后分析师引用的信息不同，属"关注因素不同"的文本证据；与预测是否一致的关系【未能确认】。【摘要】
- **Bradshaw（2002）**，Accounting Horizons 16(1)（本地 WP）：人工阅读 103 份研报，P/E 被引用于 3/4 以上报告，LTG 37%；每家公司只取一篇，不比较分析师之间。【原文·本地笔记】
- **Bradshaw（2004）**，TAR 79(1)（本地 WP）：评级与 PEG 启发式估值一致、与 RIM 估值无关或负相关；从一致预期倒推"所用模型"，作者称分析师内在估值"unobservable"（WP p.5）。【原文·本地笔记】
- **Demirakos, Strong & Walker（2004）What Valuation Models Do Analysts Use?**，Accounting Horizons 18(4): 221–240（本地无）：104 份国际投行研报、26 家英国大公司；主导模型为 PE 或显式多期 DCF；饮料业更常用可比公司，电子和制药较少；无人以 P/CF 为主导模型。【摘要】
- **Imam, Barker & Clubb（2008）The Use of Valuation Models by UK Investment Analysts**，European Accounting Review 17(3): 503–535（本地无）：35 名卖方 + 7 名买方访谈、98 份研报内容分析；分析师口头上更重视 DCF，实际仍依赖 P/E 和"价格是否合理"的判断；对买方客户的可信度需求使他们淡化主观方法。【摘要】
- **Gleason, Johnson & Li（2013）Valuation Model Use and the Price Target Performance of Sell-Side Equity Analysts**，CAR 30(1): 80–115（本地无）：First Call 1997–2003 年 45,693 个目标价；用目标价与研究者构造的 RIM 估值和 PEG 估值的相关性推断分析师所用模型；推断为 RIM 者目标价投资价值更高，但盈利预测不准时优势减弱。【摘要】
- **Asquith, Mikhail & Au（2005）**，JFE 75(2)（本地 WP）：研报论证强度（正面理由数 − 负面理由数）在评级、目标价、盈利预测之外有信息含量；编码 16 类理由。【原文·本地笔记】
- **Joos, Piotroski & Srinivasan（2016）Can Analysts Assess Fundamental Risk and Valuation Uncertainty?**，JFE 121(3): 645–663（本地无）：摩根士丹利分析师 2007 年起须给出牛/基准/熊三情景估值；**同一分析师**情景估值的跨度反映经营风险，预测长期估值误差绝对值和未来基本面变化；分析师越乐观，风险评估越小。【摘要】这是"单个分析师内部不确定性"的直接度量，与分析师之间的离散度不同。
- **Huang, Lehavy, Zang & Zheng（2018）**，MS 64(6)（本地 WP）：LDA 比较研报与电话会议主题；同一电话会议后的多篇研报被合并成一份，不比较分析师之间的主题差异。【原文·本地笔记】
- **Palley, Steffen & Zhang（2025）**（见 2.2）：一致目标价的信息含量取决于离散度。【摘要】
- **非分析师对照**：Carlin 等（2014）MBS 交易商"模型"差异的斜率离散度（见 1.2）；Bali, Kelly, Mörke & Rahman（2023, NBER WP 31583，本地有）用不同特征子集上的随机森林合成"机器投资者"，其分歧 MFD 与 AFD 平均截面秩相关 0.23，模型差异与信息差异被一起随机化而未分解。【原文·本地笔记】

### 4.3 "依据相同但预测不同"一侧
- Décaire, Sosyura & Wittry（2024）：同一 CAPM、同一公开数据，参数估计选择不同 → WACC 分歧。【原文·本地笔记】
- Kandel & Pearson（1995）/ Bamber 等（1999）：同一盈余公告后两位分析师反向修正，被解释为似然函数（解读）不同。【二手/原文】
- 这两类都只在数值层面推断"依据"，没有观测研报中的论证内容。

---

## 5. 分析师的分歧和投资者的分歧是什么关系？

**格局概述**。早期会计文献（Ajinkya & Gift 1985；Ajinkya, Atiase & Gift 1991）发现 AFD 与期权隐含波动率、成交量正相关，DMS 等据此把 AFD 当作投资者分歧的代理。随后有两类质疑：理论上 Abarbanell, Lanen & Verrecchia（1995）与 BKLS（1998）指出 AFD 同时混有不确定性与信息结构；实证上，以更直接的投资者分歧为基准时，AFD 的表现普遍很弱——Garfinkel（2009）用 NYSE 订单价格离散度发现 AFD 与之负相关，Cookson & Niessner（2020）发现 AFD 与 StockTwits 分歧不显著相关、与异常成交量相关仅 0.0388，Goulding 等（2026）发现 AFD 与卖空比率相关仅 0.07。也有研究把 AFD 解读为"信息驱动"的分歧而非"模型驱动"的分歧（Cookson & Niessner）。

### 5.1 核心论文

#### Garfinkel（2009）Measuring Investors' Opinion Divergence
- **出处**：JAR 47(5)。本地 `Garfinkel_2009_JAR_Measuring_Investors_Opinion_Divergence.pdf`。
- **数据与变量**：NYSE 专有订单数据，2002 年 1–3 月、代码 A–D 开头公司，13,017 个公司-日；DIVOP = 当日所有订单价格相对上一笔成交价偏离的标准差；AFD 用当月新发布的 IBES 年度 EPS 预测（标准差/|均值|、标准差/股价）。【原文·本地笔记】
- **论证链条**：① 投资者私人估值不可观测；② 限价单与市价单的价格比预测值更接近私人估值；③ 以 DIVOP 为基准，逐一回归各常用代理并用 Vuong 检验比较；④ 用"信息含量大的消息引起更大分歧"验证 DIVOP（|公告期 CAR| 系数 0.0159，t = 2.36）。【原文·本地笔记】
- **主要结果**：换手率变化解释力最强；AFD（标准差/股价）系数 −0.0001（1% 显著），作者称"the negative correlations between DIVOP and the analysts' forecast dispersion measures calls into question a broad swath of conclusions that presume forecast dispersion proxies for investor opinion divergence"（本次 grep 核对）；高股价股票例外。【原文】

#### Cookson & Niessner（2020）Why Don't We Agree?
- **出处**：JF 75(1): 173–228。本地 WP（2016-06）。
- **相关内容**（本次重读 WP 第 5.1 节）：月度 AFD 与 StockTwits 分歧"do not correlate with one another significantly"；跨投资风格的组间分歧与 AFD、收益波动率负相关，组内分歧正相关，作者推论"analyst dispersion and return volatility are better measurements of information-driven disagreement"；月度 AFD 与异常成交量相关 0.0388，不显著；引言称 AFD"has been questioned as a reliable measure of market-wide disagreement (Atiase and Bamber, 1994; Bamber et al., 2011)"。【原文】
- **论证链条**：① 以自报投资风格划分：组间分歧 ≈ 模型差异，组内分歧 ≈ 信息差异；② 比较各分歧成分与 AFD 的相关符号；③ AFD 只与组内（信息）成分同向。【原文】

### 5.2 次要论文
- **Abarbanell, Lanen & Verrecchia（1995）Analysts' Forecasts as Proxies for Investor Beliefs in Empirical Research**，JAE 20(1): 31–60（本地无）。IDEAS 无摘要；检索摘要转述其建模分析师预测与投资者信念的关系及对盈余意外量价反应的含义，并指出预测离散度不能完全反映投资者不确定性。【二手】BKLS（1998, p.426, 428）把它称作 ALV，并说明 ALV 的预测是外生给定的，ALV 认为离散度表示均值预测精度不足。【原文，BKLS 本地 PDF】
- **Ajinkya & Gift（1985）**，JF 40(5): 1353–1365。期权隐含标准差在历史收益之外增量反映同期 AFD。【摘要】
- **Ajinkya, Atiase & Gift（1991）Volume of Trading and the Dispersion in Financial Analysts' Earnings Forecasts**，TAR 66(2): 389–401。以年度 EPS 预测离散度代理投资者信念差异，控制月度均值修正后，与成交量显著正相关；检验 Varian（1985）、Karpoff（1986）的理论。【摘要】
- **Banerjee（2011）**，RFS 24(9)（本地有）。四个分歧代理中，AFD 与平均换手率给出相反的收益符号（Q5−Q1：AFD −1.20%，t = −2.62；VOL +1.85%，t = 3.38）；作者称 AFD"may also reflect uncertainty about earnings"，且"different proxies may lead to conflicting evidence"（p.3050, 3057）。【原文·本地笔记】
- **Goulding, Harvey & Kurtović（2026）Disagreement of Disagreement**，NBER WP 35049（本地有）。AFD–IVOL 相关 0.08，AFD–SIR 0.07；模型中 AFD 只"放大"投资者信念异质性的规模，"analyst dispersion in the data is an imperfect proxy for disagreement"（p.19）。【原文·本地笔记】
- **Giannini, Irvine & Shu（2019）**（本地有）：其社交媒体/新闻分歧度量在多元回归中与 AFD 不相关；作者称 AFD 只反映"a small subset of well-informed market participants"（p.95）。【原文·本地笔记】
- **Bamber, Barron & Stevens（2011）**，CAR 28(2): 431–471。综述公告期成交量文献，讨论个体投资者信念代理的度量难题。【摘要】对 AFD 的具体评价【未能确认】。
- **Bamber, Barron & Stober（1999）**：作者自承 KP 只基于分析师对年度盈利的预期（WP p.12）。【原文·本地笔记】
- **DMS（2002）**：直接假设 AFD 代理投资者分歧；Fama–MacBeth 中控制换手率后 AFD 仍显著，换手率不显著。【原文·本地笔记】
- **朱宏泉、余江、陈林（2016）**，管理科学学报 19(7)（本地有）：以"分析师只代表职业人士且覆盖不全""含不确定性和乐观偏差"为由不用 AFD，改用换手率残差（p.116）。【原文·本地笔记】
- **Li & Luan（2025）News-based Investor Disagreement and Stock Returns**，RAST 30(3): 2312–2375。用新闻附近成交量对波动率的弹性度量分歧（弹性越高分歧越小），分歧负向预测收益，结果与 Atmaz & Basak（2018）一致。【摘要】文献清单所写"指出分析师离散度不能代表全体投资者的分歧"在摘要中**未能确认**。
- **Huang, Li & Wang（2021）Are Disagreements Agreeable?**，JFE 141(1): 83–101。用偏最小二乘把多个分歧代理（会议版称 20 个）聚合成市场分歧指数，单个代理通常不能预测市场收益，聚合指数可以。AFD 是否在其中【未能确认】。【摘要】
- **Bali 等（2023）**（本地有）：MFD 与 AFD 平均截面秩相关 0.23。【原文·本地笔记】

---

## 6. 与"信息观点/投机观点"二分法的对应情况（只陈述事实）

### 6.1 "信息观点"（不同模型得出相同结论 = 独立证据相互印证，共识更准、风险更低）
- **分析师文献里有直接出处的部分**：
  - "分析师预测可视为对真实值的多个独立噪声信号，离散度低 = 参数不确定性低"：Johnson（2004, p.1959）明确采用这一"multiple-signal interpretation"。但其定价含义是：对杠杆公司，离散度低对应**更高**的预期收益（期权凸性），并非"风险更低所以价格更高"；且 Johnson 自己提出"为什么分析师不自行汇总信息""离散度本身是否受激励影响"等疑问。【原文】
  - "独立性提高共识准确度"：Da & Huang（2020，随机实验，信息独立）、Clement & Tse（2005，大胆预测更准）、Kim, Lim & Shaw（2001，均值高估共同信息）、BKLS（1998，只有私有误差能被平均掉）。【摘要/原文】
  - "多样性提高共识准确度"：Merkley, Michaely & Pacelli（2020，文化多样性；机制归为信息流动）。【摘要】
- **分析师文献里没有找到的部分**：
  - "**同一结论由不同推理模型得出**，因而构成相互印证"的说法：未找到。上述文献里的"独立"都是**信息**（私有信号）独立，不是模型或方法独立；Merkley 等的"多样性"是背景多样性。Bates & Granger 式"组合不同模型的预测"的逻辑，在分析师文献中对应的是 BKLS/KLS 的公共-私有信息框架，而非模型异质性。
  - "这类共识的股票风险更低、价格更高"的定价检验：未找到。
- **分析师文献中与之方向相反或使之复杂化的事实**：
  - BKLS（1998, p.426）：在不确定性给定时，离散度低可能意味着共识度 ρ 高（分析师依赖共同信息），对应均值预测误差**更大**。
  - Jegadeesh & Kim（2010）：在评级分歧较小的股票上，分析师更易羊群。【摘要】
  - Doukas, Kim & Pantzalis（2006b）：离散度低且预期乐观时出现高估，作者解读为投资者把低离散度股票当作"sure winners"。【摘要】
  - Palley, Steffen & Zhang（2025）：低离散度时一致目标价有正向信息含量、高离散度时为负——这与"低离散度共识更可信"同向，但作者的机制是高离散度来自部分分析师迟迟不下调，而不是模型相互印证。【摘要】

### 6.2 "投机观点"（模型不同 → 未来消息到来时分歧，卖空约束下有转售期权溢价，价格偏高）
- **分析师文献里有直接出处的部分**：
  - Miller 式"**当前**分歧 + 卖空约束 → 高估"：DMS（2002）、Berkman 等（2009）、Yu（2011）、Sadka & Scherbina（2007）等，均用当前 AFD。
  - Harrison–Kreps / 转售期权逻辑在分析师数据上的应用：Park（2005，据 Banerjee 2011 转述为扩展 Harrison & Kreps 模型）；Hong & Sraer（2016，用分析师 LTG 离散度，投资者预期高 β 资产未来更可能遇到卖空约束、转售价格更高）；Yin（Purdue 博士论文，AFD 增长与转售期权的外推误估）。
  - 盈余公告后分析师分歧扩大（差异化解读）：Kandel & Pearson（1995）、Bamber 等（1999）、Barron 等（2002）、Kwon（2025）。这些论文把它与**成交量**相连，没有与转售期权溢价相连。
- **分析师文献里没有找到的部分**：
  - "**当前预测一致，但由于模型不同而在未来消息到来时会分歧**，因此现在就有转售期权溢价"：未找到。上面的投机类论文都以**当前**离散度（或其变化）为输入，没有以"推理依据的差异"为输入。结构上最接近的是 Hong & Sraer（2016）：资产对共同因子的敏感度（β）越高，宏观分歧上升时投资者对该资产现金流的分歧越大——即"未来分歧的潜力"由暴露程度决定，但其度量是 β 而非分析师模型差异。
  - "模型分歧在当前估计一致时仍然存在、并推高波动与交易"的论证只出现在一般理论（Andrei, Carlin & Hasler 2019，宏观周期长度参数）和宏观专业预测者（Andrade, Crump, Eusepi & Moench 2016, JME 83: 106–128，长期限分歧持续且时变【摘要】）里，不在股票分析师文献里。
- **分析师文献中使之复杂化的事实**：离散度与收益的符号本身不稳定——AGJ（2005，长期离散度正）、Qu-Starks-Yan（正）、Doukas 等 2006a（控制不确定性后正）、Li 等 2021（目标价离散度正）、Carlin 等 2014（MBS 交易商，正风险溢价）。

### 6.3 分析师文献实际的划分方式与该二分法的差异
1. 分析师文献争论的核心变量是**离散度的高低**，而不是"离散度低时依据是否相同"。二分法关心的"殊途同归"在分析师文献里没有对应变量。
2. 分析师文献对离散度的划分是"意见分歧/错误定价 vs 不确定性/风险 vs 信息不对称 vs 偏差"（见第 1 节），这四者都是关于"离散度是什么"，与"一致是否可信"是不同的问题。
3. 讨论"一致是否可信"时，分析师文献的划分是"**公共信息 vs 私有信息**"（BKLS、KLS、Barron 等 2002/2005、Botosan 等 2004）和"**羊群 vs 独立/大胆**"（Hong-Kubik-Solomon、Clement-Tse、Jegadeesh-Kim、Da-Huang）。在这一划分下，一致的风险是"来自共同信息或羊群"，而不是"来自不同模型"。
4. 讨论"分歧来源"时，"信息不同 vs 模型/解读不同"的划分在一般理论（Hong & Stein 2007 的渐进信息/有限注意/异质先验）和投资者社交数据（Cookson & Niessner 2020）里明确，在分析师数据上只有 Kandel & Pearson 一支（解读）、Décaire, Sosyura & Wittry（参数估计方法）和 Kwon（信息选择）三处做了区分，且都未与"殊途同归"的定价相连。
5. "信息观点"所需的"不同模型 → 独立证据"在分析师文献中缺少直接出处；"投机观点"所需的"Miller/Harrison–Kreps 机制"在分析师文献中有出处，但所依赖的输入是当前离散度，而非依据差异。

---

## 7. 本地文件夹里没有的论文（便于以后补下载）

按本笔记出现顺序：
1. Anderson, Ghysels & Juergens（2005）RFS 18(3): 875–924
2. Berkman, Dimitrov, Jain, Koch & Tice（2009）JFE 92(3): 376–399
3. Barron, Stanford & Yu（2009）CAR 26(2)（文献清单已列为待下载）
4. Doukas, Kim & Pantzalis（2006b）JFM 9(3): 310–331
5. Scherbina（2008）Review of Finance 12(3): 533–565
6. Park（2005）Journal of Business 78(6): 2351–2376
7. Yin（Purdue 博士论文）Two Essays in Behavioral Finance / Resale Value and Security Prices
8. Barinov（2013）JFQA 48(6): 1877–1900
9. Qu, Starks & Yan（2003/2004 WP）——题名与发表情况待核实
10. Sheng & Thevenot（2012）JAE 53(1): 21–33
11. Verardo（2009）JFQA 44(4): 795–822
12. Li, Feng, Yan & Wang（2021）NAJEF 56: 101385（文献清单已列）
13. Garfinkel & Sokobin（2006）JAR 44(1): 85–112
14. Keskek, Rees & Thomas（2013）JBFA
15. Kim, Lim & Shaw（2001）JAR 39(2): 329–335
16. Da & Huang（2020）MS 66(5): 1847–1867
17. Merkley, Michaely & Pacelli（2020）JAE 70(1): 101330
18. Barron, Byard & Kim（2002）TAR 77(4): 821–846（文献清单已列）
19. Botosan, Plumlee & Xie（2004）RAST 9(2): 233–259
20. Hong, Kubik & Solomon（2000）RAND 31(1): 121–144
21. Clement & Tse（2005）JF 60(1): 307–341
22. Jegadeesh & Kim（2010）RFS 23(2): 901–937
23. Bernhardt, Campello & Kutsoati（2006）JFE 80(3): 657–675
24. Chen & Jiang（2006）RFS 19(1): 319–355
25. Jame, Johnston, Markov & Wolfe（2016）JAR 54(4): 1077–1110
26. Fang & Hope（2021）RAST 26(2): 425–467
27. Kumar, Rantala & Xu（2022）JFE 143(1): 434–461
28. Palley, Steffen & Zhang（2025）MS 71(3): 2264–2288
29. Cavezzali & Rigoni（2013）Ca' Foscari WP No.9
30. Kandel & Pearson（1995）JPE 103(4): 831–872（文献清单已列）
31. Kwon（2025）芝加哥大学博士论文（受限访问）
32. Barron（1995）TAR 70(4): 581–597
33. Bamber, Barron & Stober（1997）TAR 72(4): 575–597
34. Barron, Harris & Stanford（2005）TAR 80(2): 403–421
35. Barron, Byard & Yu（2017）CAR 34(1): 343–373
36. Atiase & Bamber（1994）JAE 17(3): 309–330
37. Kecskés, Michaely & Womack（2017）MS 63(6): 1855–1871
38. Demirakos, Strong & Walker（2004）Accounting Horizons 18(4): 221–240（文献清单已列）
39. Imam, Barker & Clubb（2008）European Accounting Review 17(3): 503–535
40. Gleason, Johnson & Li（2013）CAR 30(1): 80–115（文献清单已列）
41. Joos, Piotroski & Srinivasan（2016）JFE 121(3): 645–663
42. Abarbanell, Lanen & Verrecchia（1995）JAE 20(1): 31–60
43. Ajinkya & Gift（1985）JF 40(5): 1353–1365
44. Ajinkya, Atiase & Gift（1991）TAR 66(2): 389–401
45. Bamber, Barron & Stevens（2011）CAR 28(2): 431–471
46. Li & Luan（2025）RAST 30(3): 2312–2375（文献清单已列）
47. Huang, Li & Wang（2021）JFE 141(1): 83–101（文献清单已列）
48. Andrade, Crump, Eusepi & Moench（2016）JME 83: 106–128

**中文文献**：本次网页检索未找到以分析师预测分歧为对象、检验"共识可信度"或"转售期权"的中文期刊论文；仅有本地已读的朱宏泉等（2016）。建议在知网以"分析师 预测分歧 定价""分析师 羊群 一致预期 准确度""研报 估值方法 目标价"检索《经济研究》《管理世界》《金融研究》《会计研究》《管理科学学报》。

**本次检索中无法确认、未列入正文的线索**：一篇耶鲁管理学院作者用 110 余万份研报文本（LLM + 检索增强）研究分析师信念、称"信息与意见差异共同导致盈利预测分歧"的工作论文，检索摘要中出现但题名与作者无法确认。
