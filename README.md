# llm_projcet

用大语言模型做金融文本分析的两条研究线，各自独立成一个顶层目录，只共用大模型调用层和 `.env`。

```
common/             两条线共用：大模型接口（兼容 OpenAI Chat Completions）、.env 读取、进度日志
mda_tone/           主线一：年报 MD&A 语调中性化 → 情感打分 → 事件研究与假说检验
research_report/    主线二：卖方研报估值逻辑图 / 知识图谱抽取
literature_review/  其他选题的文献综述（投资者场外生活与投资行为），不属于以上两条线
.env                API_KEY、BASE_URL、MODEL 等，见 common/llm_client.py 的说明；不入 git
```

每条研究线目录下的约定相同：

| 子目录 / 文件 | 内容 | 入 git |
|---|---|---|
| `*.py` | 代码（本目录是一个 Python 包） | 是 |
| `main.py` | 启动入口，修改 `TASK` 选择功能 | 是 |
| `config.py` | `DATA_DIR`、`OUTPUT_DIR`、`PROMPT_DIR` 等路径约定 | 是 |
| `prompts/` | 大模型提示词 | 是 |
| `docs/` | 研究报告、设计文档、笔记 | 是 |
| `data/` | 原始数据与中间数据 | 否 |
| `output/` | 分析结果 | 否 |
| `literature/` | 参考文献 PDF | 否 |

## 运行

一律在仓库根目录运行，使用 `C:\Users\hung\projects\.venv` 里的 Python（系统 PATH 上的 Python 没有 pandas 等依赖）。

```bash
python mda_tone/main.py            # 主线一：按 main.py 中的 TASK 运行（也可 python -m mda_tone.main）
python research_report/main.py     # 主线二：同上
```

主线一的假说检验脚本不走 `main.py`，按顺序直接运行（先用 main.py 跑完 `cs_sentiment`、`event_study`）：

```bash
python -m mda_tone.construct_hypothesis_vars
python -m mda_tone.run_hypothesis_regressions
python -m mda_tone.tone_decomposition
python -m mda_tone.tone_decomposition_reg
```

## 主线一：年报 MD&A 语调（`mda_tone/`）

研究问题：管理层在年报“管理层讨论与分析”中的语调修饰，是否影响公告后的股价反应。

1. `extract_mda.py`、`clean_mda_scope.py`、`extract_audit.py`：从年报 PDF 压缩包抽取 MD&A（及审计报告）文字。
2. `neutralize_llm.py`：用大模型把每句话改写成只保留事实的中性版本（提示词 `prompts/mda_neutralize_prompt.md`）。
3. `compare_neutral_tone.py`、`cs_sentiment.py`、`plot_tone_change.py`：原文与中性化版本逐句情感打分，汇总为公司 × 年份面板。
4. `event_study.py`、`plot_std_car.py`：公告后 CAR 与情感指标的回归。
5. `construct_hypothesis_vars.py`、`run_hypothesis_regressions.py`、`tone_decomposition*.py`：事前假说与多重检验。

结论见 `docs/sent_change_hypotheses.md`、`docs/tone_decomposition_tests.md`。

## 主线二：研报估值逻辑图（`research_report/`）

把一篇研报的推理链抽成五层有向图：公司估值 → 会计科目（`fin_items.py` 预置）→ 经营指标 → 公司经营与行业 → 外部环境。

1. `report_text.py`：研报 PDF → 正文段落 parquet。
2. `report_logic.py`：概念分层、原文连边、子图补全、去重（提示词 `prompts/report_logic_prompt.md`）。
3. `plot_report_logic.py`：交互式 HTML；`.claude/launch.json` 里的 `report-logic-html` 可在浏览器预览。
4. `report_kg.py`：基于 KGGen 框架的研报知识图谱试验（提示词 `prompts/report_kg_prompt.md`）。

框架设计见 `docs/研报逻辑图抽取框架设计.md`。
