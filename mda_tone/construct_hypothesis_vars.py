"""围绕 sent_change_mean 构造事前假说变量（只构造变量，不做任何收益回归）。

输入：
  - tone_by_sentence.csv：逐句分数和文本，同一 (year, code) 的句子连续，流式读取
  - cs_sentiment.csv：全部公司*年份的 sent_change_mean（用于公司自身历史）
  - event_panel.csv：只读取 ALLOWED_PANEL_COLS 中的列，不读 car_* 和 beta
  - financial_data / daily_stock_data / stock_forecast_means

输出（out_dir）：
  - variables.csv：code, year + 所有新变量，与 event_panel 一一对应，缺失写空
  - hypotheses.json：假说清单

无前视偏差的约定（t0 为事件日在交易日历上的位置）：
  - 行情变量只用 [t0-60, t0-1]
  - 分析师数据是月度表（create_date 为月初，可能包含当月全部研报），只用事件月之前的完整月份；
    数据只到 2023-08，最后一个完整月份取 2023-07，事件月晚于 2023-08 的事件记为缺失
  - 财务变量用与年报同时披露的当期值
  - 同年横截面分组只用事件日严格更早的同年事件

交互项：两个成分先在事件面板上 1%/99% 缩尾，连续成分再去均值，哑变量不去均值。
sent_change_mean（下称 sc）通常为负，越负粉饰越多，因此“粉饰有害”对应 sc 的系数为正。

用法：在项目根目录运行 python -m mda_tone.construct_hypothesis_vars
"""
import csv
import itertools
import json
import math
from bisect import insort
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .config import DATA_DIR, OUTPUT_DIR
from .event_study import load_returns, winsorize

csv.field_size_limit(10 ** 9)  # 个别句子可能超过 csv 默认的字段长度上限

ALLOWED_PANEL_COLS = ['code', 'year', 'event_date', 'announce_date', 'ln_assets', 'roe', 'leverage', 'sue',
                      'sent_change_mean', 'sent_mean_orig', 'sent_mean_neutral', 'sent_std_orig']
SENTENCE_COLS = ['year', 'code', 'id', 'label_orig', 'label_neutral', 'pos_orig', 'neg_orig',
                 'pos_neutral', 'neg_neutral', 'original', 'neutral']

BIG_CUT = -0.5  # 单句情感变化不高于该值视为被大幅下调
MIN_LABEL_GROUP = 5  # 计算翻转占比时，分母句数的下限
MIN_POSITION_SENT = 8  # 计算首尾差异时，全文句数的下限
MIN_CORR_SENT = 10  # 计算相关系数时，全文句数的下限
PRE_WINDOW = 60  # 行情窗口 [t0-60, t0-1]
MIN_TURNOVER_OBS, MIN_RUNUP_OBS = 20, 40
ANALYST_MONTHS = 12  # 分析师覆盖回看的完整月数
CONSENSUS_MONTHS = 6  # 一致预期回看的完整月数
LAST_ANALYST_MONTH = 2023 * 12 + 7  # 分析师数据最后一个完整月份（2023-07）
MIN_HISTORY = 2  # 过去 3 年均值至少需要的年数
MIN_PEERS = 50  # 同年事件日更早的同行数下限
SMALL_PROFIT_ROA = 0.01  # 微利：0 <= 归母净利润/总资产 < 1%
NON_FINANCIAL_TYPE = 99  # financial_data 的 company_type：99 为非金融，13/31/33/35/39 为银行、证券、信托、保险、其他金融

SENTENCE_VARS = ['big_cut_ratio', 'cut_conc_top10', 'pos_flip_ratio', 'neg_soft_ratio', 'unchanged_rate',
                 'len_compress', 'front_back_gap', 'neutral_neg_share', 'hard_pos_share', 'tone_retention_corr']
MODERATORS = ['sue_pos', 'loss', 'miss_consensus', 'small_profit', 'accruals', 'ln_analyst_12m',
              'ln_turnover_60', 'runup_60']
HISTORY_VARS = ['sc_yoy', 'sc_dev_3y', 'sc_extreme_q5']
# 交互项名 -> (调节变量列, 是否哑变量)
INTERACTIONS = {
    'sc_x_sue_pos': ('sue_pos', True),
    'sc_x_loss': ('loss', True),
    'sc_x_miss': ('miss_consensus', True),
    'sc_x_small_profit': ('small_profit', True),
    'sc_x_accruals': ('accruals', False),
    'sc_x_analyst': ('ln_analyst_12m', False),
    'sc_x_turnover': ('ln_turnover_60', False),
    'sc_x_runup': ('runup_60', False),
    'sc_x_size': ('ln_assets', False),
    'sc_x_std_orig': ('sent_std_orig', False),
    'sc_x_neutral': ('sent_mean_neutral', False),
}
VARIABLES = SENTENCE_VARS + MODERATORS + HISTORY_VARS + list(INTERACTIONS)

_SC_NOTE = 'sc 越负粉饰越多；交互项系数为正表示调节变量越大（或哑变量为 1）时粉饰的负面作用越强，为负表示越弱。'
_BREAK_NOTE = '2021 年起 MD&A 并入业务概要，2011-2014 为清理后文本，跨断点比较需依赖年份固定效应。'


def _h(hid, name, cn, family, hypothesis, construction, sign, extra, window, caveats):
    return {'id': hid, 'name': name, 'name_cn': cn, 'family': family, 'hypothesis': hypothesis,
            'construction': construction, 'expected_sign': sign, 'extra_terms': extra, 'focal_term': name,
            'expected_window': window, 'caveats': caveats}


HYPOTHESES = [
    _h('H1', 'big_cut_ratio', '大幅下调句占比', 'A',
       '被中性化大幅下调的句子越多，文本中可被去掉的强修饰越多。MD&A 篇幅长、信息消化慢，市场会逐步折价，公告后收益更低。',
       '句子 change = (pos_neutral-neg_neutral)-(pos_orig-neg_orig)；big_cut_ratio = mean(change <= -0.5)。',
       '-', [], '漂移[0,20]', '阈值 -0.5 为经验设定；' + _BREAK_NOTE),
    _h('H2', 'cut_conc_top10', '下调集中度', 'A',
       '给定总体粉饰水平，下调集中在少数句子说明是有针对性的修饰，而非全文统一的文风。针对性修饰更可能是在掩饰特定问题，公告后收益更低。',
       'cut = max(-change, 0)；cut_conc_top10 = 下调最大的 ceil(10%*n) 句的 cut 之和 / 全部 cut 之和；全部 cut 为 0 时缺失。',
       '-', ['sent_change_mean'], '漂移[0,20]', '依赖句数，句数少的年报机械偏高，2021 年起句数大增（中位数约 150 到 230）会压低集中度，依赖年份固定效应；与 sc 的 FE 后相关约 0.84（总下调越小集中度越高），VIF 约 3.5，系数需结合 sc 解读。'),
    _h('H3', 'pos_flip_ratio', '正面句翻转占比', 'A',
       '原文正面的句子改写后不再正面的比例，度量正面语气中缺少事实支撑的部分。该比例高说明好消息的实质含量低，市场逐步修正后收益更低。',
       '原文标签 positive 的句子中，改写后标签不是 positive 的比例；原文正面句少于 5 句时缺失。',
       '-', [], '漂移[0,20]', '标签由 FinBERT 给出，分数接近分类边界的句子可能因轻微改写翻转。'),
    _h('H4', 'neg_soft_ratio', '负面句软化占比', 'A',
       '原文负面的句子改写后不再负面，说明负面措辞是渲染而非事实（如夸大外部困难、为未来业绩铺垫）。实际情况好于字面，市场修正后收益更高。',
       '原文标签 negative 的句子中，改写后标签不是 negative 的比例；原文负面句少于 5 句时缺失。',
       '+', [], '漂移[0,20]', '原文负面句较少，分母小的年报噪声大；已设 5 句下限。'),
    _h('H5', 'unchanged_rate', '未改写率', 'A',
       '改写前后完全相同的句子说明表述本身就是平实事实。未改写率高的年报信息更硬、可信度更高，公告后收益更高。',
       '去除全部空白后 original == neutral 的句子占比。',
       '+', [], '漂移[0,20]', '必须去空白比较，否则深市 2022 年有空格假象；未改写也可能来自模板化套话；' + _BREAK_NOTE),
    _h('H6', 'len_compress', '字符压缩比', 'A',
       '改写后保留的字符比例越低，文本中的修辞和冗余越多。这一指标不依赖 FinBERT，是对粉饰的独立度量，比例高的年报实质内容多，公告后收益更高。',
       'len_compress = 全文去空白后 neutral 字符数之和 / original 字符数之和。',
       '+', [], '漂移[0,20]', 'LLM 偶尔补充主语（如年份），比值可能略大于句子层面的真实压缩。'),
    _h('H7', 'front_back_gap', '首尾粉饰差', 'A',
       '投资者注意力有限，读者更关注 MD&A 开头。给定总体粉饰水平，把粉饰前置（开头 change 比结尾更负）是策略性的版面安排，短期可能奏效但随后回吐，较长窗口收益更低。',
       '按句序取前 25% 与后 25% 的句子（各 floor(n/4) 句），front_back_gap = 前段 change 均值 - 后段 change 均值；句数少于 8 时缺失。',
       '+', ['sent_change_mean'], '漂移[0,20]', '值越负表示粉饰越前置，正系数表示前置粉饰收益更低；开头与结尾的章节内容差异也会带来结构性差值；2021 年起业务概要并入后开头内容改变，该变量跨断点不可比，依赖年份固定效应，建议分段检验。'),
    _h('H8', 'neutral_neg_share', '中性化后负面句占比', 'A',
       '去掉修辞后仍为负面的句子代表公司披露的实质性负面信息，负面事实较多的公司公告后收益更低。',
       '改写后标签为 negative 的句子占全部句子的比例。',
       '-', [], '短期[0,3]', '与 sent_mean_neutral 相关度较高，不要同时放入。'),
    _h('H9', 'sc_x_sue_pos', '粉饰×盈余好消息', 'B',
       '好消息配合粉饰容易被视为确认信号，粉饰的负面作用较弱；坏消息配合粉饰更像掩饰。粉饰的不利影响在 SUE 为正时更小。',
       'sue_pos = (sue > 0)；sc_x_sue_pos = sc_dm * sue_pos，sc_dm 为缩尾去均值后的 sc。',
       '-', ['sent_change_mean', 'sue_pos'], '短期[0,3]', _SC_NOTE),
    _h('H10', 'sc_x_loss', '粉饰×亏损', 'B',
       '亏损公司的粉饰更可能是掩盖坏消息，亏损本身又容易引起关注，投资者惩罚更重。粉饰的负面作用在亏损公司更强。',
       'loss = (归母净利润TTM < 0)；sc_x_loss = sc_dm * loss。',
       '+', ['sent_change_mean', 'loss'], '短期[0,5]', _SC_NOTE + '亏损样本占比较小。'),
    _h('H11', 'sc_x_miss', '粉饰×未达预期', 'B',
       '业绩低于分析师一致预期时，粉饰更像为不及预期找说辞，投资者对这类公司的文字更不信任，粉饰的负面作用更强。',
       'miss_consensus = (归母净利润 - 事件月前最近一致预期*1e4 < 0)；sc_x_miss = sc_dm * miss_consensus。',
       '+', ['sent_change_mean', 'miss_consensus'], '短期[0,3]',
       _SC_NOTE + '一致预期回看 6 个完整月，当年月份用 fy0，下一年月份用 fyneg1；单位按万元换算（已用财务数据核对）；2023 年报缺失；约 70% 的样本低于一致预期（分析师乐观偏差），哑变量为 0 的组是少数。'),
    _h('H12', 'sc_x_small_profit', '粉饰×微利', 'B',
       '微利公司存在盈余管理嫌疑，文字粉饰与盈余管理叠加是更强的危险信号。粉饰的负面作用在微利公司更强。',
       'small_profit = (0 <= 归母净利润TTM/总资产 < 1%)，金融企业（company_type != 99）记缺失；sc_x_small_profit = sc_dm * small_profit。',
       '+', ['sent_change_mean', 'small_profit'], '漂移[0,20]', _SC_NOTE + '微利阈值为经验值；金融企业 ROA 天然低于 1%，已记缺失。'),
    _h('H13', 'sc_x_accruals', '粉饰×应计', 'B',
       '高应计说明利润缺少现金支撑。文字粉饰与高应计同时出现时，粉饰更可能是在掩盖盈余质量问题，粉饰的负面作用更强。',
       'accruals = (净利润TTM - 经营现金流TTM)/总资产，金融企业（company_type != 99）记缺失；sc_x_accruals = sc_dm * accruals_dm。',
       '+', ['sent_change_mean', 'accruals'], '漂移[0,20]', _SC_NOTE + '金融企业的经营现金流含义不同，已记缺失。'),
    _h('H14', 'sc_x_analyst', '粉饰×分析师覆盖', 'B',
       '分析师多的公司信息环境好，文字粉饰容易被识别和过滤，定价影响更弱。粉饰的负面作用随覆盖增加而减弱。',
       'ln_analyst_12m = ln(1+事件月前 12 个完整月研报数之和)；sc_x_analyst = sc_dm * analyst_dm。',
       '-', ['sent_change_mean', 'ln_analyst_12m'], '漂移[0,20]',
       _SC_NOTE + '分析师表到 2023-08 为止，2023 年报（事件在 2024 年）缺失；中性化样本本身要求 Y-1 有覆盖，变异主要来自覆盖强度。'),
    _h('H15', 'sc_x_turnover', '粉饰×换手率', 'B',
       '高换手股票散户和短线交易占比高，容易按字面接受乐观文字，短期内粉饰的负面作用被抵消甚至反转。',
       'ln_turnover_60 = ln([t0-60,t0-1] 日均 成交额/总市值)；sc_x_turnover = sc_dm * turnover_dm。',
       '-', ['sent_change_mean', 'ln_turnover_60'], '短期[0,3]',
       _SC_NOTE + '预期只针对短窗口，长窗口可能反转；分母为总市值而非流通市值（数据中没有流通市值）。'),
    _h('H16', 'sc_x_runup', '粉饰×公告前涨幅', 'B',
       '公告前已大幅上涨的股票，预期已被推高，此时的粉饰更容易在信息消化后被回吐。粉饰的负面作用在高涨幅股票中更强。',
       'runup_60 = sum(r_i - r_m)，[t0-60,t0-1]，r_m 为市值加权市场收益，至少 40 个有效日；sc_x_runup = sc_dm * runup_dm。',
       '+', ['sent_change_mean', 'runup_60'], '漂移[0,20]', _SC_NOTE + '公告前涨幅部分反映业绩预告信息。'),
    _h('H17', 'sc_x_size', '粉饰×规模', 'B',
       '小公司媒体和机构关注少，文字粉饰被发现得晚，大公司的粉饰更快被识别和定价，影响更弱。粉饰的负面作用随规模增加而减弱。',
       'sc_x_size = sc_dm * ln_assets_dm。',
       '-', ['sent_change_mean'], '漂移[0,20]', _SC_NOTE + 'ln_assets 主效应已在基准控制变量中。'),
    _h('H18', 'sc_yoy', '粉饰同比变化', 'C',
       '相对公司自身上年，粉饰程度上升（sc 更负）意味着管理层的表述风格转向乐观修饰，往往出现在基本面转弱时，公告后收益更低。',
       'sc_yoy = sc(Y) - sc(Y-1)，历史值取自 cs_sentiment，Y-1 缺失时为空。',
       '+', [], '漂移[0,20]', _BREAK_NOTE + '2021 年的同比跨越了口径断点，各公司受影响程度不同，建议加做剔除 2021 年的稳健性检验。2015 年同比跨越 2011-2014 清理文本口径。'),
    _h('H19', 'sc_dev_3y', '粉饰相对历史偏离', 'C',
       '相对公司过去 3 年均值的偏离度量异常粉饰，比单年同比噪声更小。异常粉饰高（偏离更负）的年报公告后收益更低。',
       'sc_dev_3y = sc(Y) - mean(sc(Y-1), sc(Y-2), sc(Y-3))，至少需要 2 年历史。',
       '+', [], '漂移[0,20]', _BREAK_NOTE + '2021-2023 年的窗口横跨断点。样本从 2013 年开始有值。'),
    _h('H20', 'sc_extreme_q5', '极端粉饰组', 'C',
       '粉饰程度处于同年最高五分之一的年报更容易被市场注意和警惕，存在超出线性关系的额外折价。',
       '在事件日严格早于本事件的同年事件中取 sc 的 20% 分位数，sc 不高于该值时记 1；同行少于 50 家时缺失。',
       '-', ['sent_change_mean'], '漂移[0,20]', '只用已发布的同年同行，因此每年最早发布的事件缺失；与 sc 同时放入以检验非线性。'),
    _h('H21', 'tone_retention_corr', '语气保留度', 'D',
       '原文语气与中性化后语气的句子层面相关度高，说明原文语气确实反映事实；相关度低说明语气与事实脱节。保留度高的年报信息含量高，公告后收益更高。',
       'tone_retention_corr = corr(orig_score, neutral_score)，句数少于 10 或方差为 0 时缺失。',
       '+', [], '漂移[0,20]', '相关系数对两极化分数的分布较敏感。'),
    _h('H22', 'hard_pos_share', '实质正面句占比', 'D',
       '改写前后都为正面的句子代表有事实支撑的好消息，该比例高说明好消息确实存在，市场定价为正。',
       '原文标签与改写后标签都是 positive 的句子占全部句子的比例。',
       '+', [], '短期[0,3]', '与 sent_mean_neutral 相关，不要同时放入。'),
    _h('H23', 'sc_x_std_orig', '粉饰×语气离散', 'D',
       '原文语气离散度高说明文本内部好坏信息并存，此时粉饰可能是在冲淡负面内容。粉饰的负面作用在内部不一致的年报中更强。',
       'sc_x_std_orig = sc_dm * sent_std_orig_dm。',
       '+', ['sent_change_mean', 'sent_std_orig'], '漂移[0,20]', _SC_NOTE),
    _h('H24', 'sc_x_neutral', '粉饰×实质语气', 'D',
       '中性化后的语气代表事实层面的好坏。事实较差却仍粉饰，是最典型的掩饰，粉饰的负面作用应在实质语气低时更强，因此交互项为负。',
       'sc_x_neutral = sc_dm * sent_mean_neutral_dm。',
       '-', ['sent_change_mean', 'sent_mean_neutral'], '漂移[0,20]',
       _SC_NOTE + '只放 sc 和 sent_mean_neutral，不放 sent_mean_orig，避免恒等式共线。'),
]


def fmt(x):
    """把数值格式化成 csv 字段，缺失写空。"""
    return '' if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), 6) + 0.0  # + 0.0 把 -0.0 变成 0.0


def num(s):
    return float(s) if s not in ('', None) else math.nan


def load_panel(panel_path):
    """只取 ALLOWED_PANEL_COLS，其余列（含 car_* 和 beta）在解析后立刻丢弃。"""
    with Path(panel_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        index = [header.index(c) for c in ALLOWED_PANEL_COLS]
        return [dict(zip(ALLOWED_PANEL_COLS, (row[i] for i in index))) for row in reader]


def doc_stats(rows, col):
    """由一份年报的全部句子计算句子层面变量。"""
    rows = sorted(rows, key=lambda r: int(r[col['id']]))
    orig = np.array([float(r[col['pos_orig']]) - float(r[col['neg_orig']]) for r in rows])
    neutral = np.array([float(r[col['pos_neutral']]) - float(r[col['neg_neutral']]) for r in rows])
    label_o = np.array([r[col['label_orig']] for r in rows])
    label_n = np.array([r[col['label_neutral']] for r in rows])
    stripped = [(''.join(r[col['original']].split()), ''.join(r[col['neutral']].split())) for r in rows]
    change, n = neutral - orig, len(rows)

    cuts = np.maximum(-change, 0.0)
    top_k = math.ceil(0.1 * n)
    total_cut = cuts.sum()
    is_pos_o, is_neg_o = label_o == 'positive', label_o == 'negative'
    is_pos_n, is_neg_n = label_n == 'positive', label_n == 'negative'
    chars_orig = sum(len(a) for a, _ in stripped)
    quarter = n // 4
    corr_ok = n >= MIN_CORR_SENT and orig.std() > 0 and neutral.std() > 0
    return {
        'big_cut_ratio': float((change <= BIG_CUT).mean()),
        'cut_conc_top10': float(np.sort(cuts)[-top_k:].sum() / total_cut) if total_cut > 0 else None,
        'pos_flip_ratio': float((~is_pos_n[is_pos_o]).mean()) if is_pos_o.sum() >= MIN_LABEL_GROUP else None,
        'neg_soft_ratio': float((~is_neg_n[is_neg_o]).mean()) if is_neg_o.sum() >= MIN_LABEL_GROUP else None,
        'unchanged_rate': sum(a == b for a, b in stripped) / n,
        'len_compress': sum(len(b) for _, b in stripped) / chars_orig if chars_orig else None,
        'front_back_gap': float(change[:quarter].mean() - change[-quarter:].mean()) if n >= MIN_POSITION_SENT else None,
        'neutral_neg_share': float(is_neg_n.mean()),
        'hard_pos_share': float((is_pos_o & is_pos_n).mean()),
        'tone_retention_corr': float(np.corrcoef(orig, neutral)[0, 1]) if corr_ok else None,
    }


def sentence_vars(sentence_path, keys, max_docs=None):
    """流式读取逐句文件，只计算 keys 中的年报，返回 {(code, year): 变量字典}。"""
    result, seen = {}, set()
    with Path(sentence_path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        col = {c: header.index(c) for c in SENTENCE_COLS}
        groups = itertools.groupby(reader, key=lambda r: (r[col['code']], r[col['year']]))
        for n_docs, (key, group) in enumerate(groups, 1):
            if key in seen:
                raise SystemExit(f'年报 {key} 的句子不连续，请检查结果文件')
            seen.add(key)
            if key in keys:
                result[key] = doc_stats(list(group), col)
            if n_docs % 5000 == 0:
                print(f'已处理 {n_docs} 篇年报', flush=True)
            if max_docs and len(result) >= max_docs:
                break
    return result


def load_fin(fin_path):
    """返回 {(code, year): 年报当期财务字段}（年报无重复键），NaN 视同缺失。"""
    columns = ['code', 'report_date', 'company_type', 'np_parent_company_owners_ttm', 'net_profit_ttm',
               'net_operate_cash_flow_ttm', 'total_assets']
    fin = {}
    for r in pq.read_table(fin_path, columns=columns).to_pylist():
        if r['report_date'].endswith('1231'):
            fin[r['code'], r['report_date'][:4]] = {k: None if isinstance(v, float) and math.isnan(v) else v
                                                    for k, v in r.items()}
    return fin


def load_analyst(path):
    """返回 {股票: {月序号: (研报数, fy0 净利润, fyneg1 净利润)}}，月序号 = 年*12+月。"""
    columns = ['stock_code', 'create_date', 'report_count', 'forecast_np_mean_fy0_q4', 'forecast_np_mean_fyneg1_q4']
    table = {}
    for r in pq.read_table(path, columns=columns).to_pylist():
        d = r['create_date']
        fy0, fyneg1 = r['forecast_np_mean_fy0_q4'], r['forecast_np_mean_fyneg1_q4']
        table.setdefault(r['stock_code'], {})[d.year * 12 + d.month] = (
            r['report_count'] or 0, None if fy0 is None or math.isnan(fy0) else fy0,
            None if fyneg1 is None or math.isnan(fyneg1) else fyneg1)
    return table


def analyst_vars(code, year, event_date, table):
    """事件月之前的分析师覆盖和年报年度的一致预期，返回 (覆盖, 一致预期万元)。"""
    month = event_date // 10000 * 12 + event_date // 100 % 100
    if month - 1 > LAST_ANALYST_MONTH:
        return None, None
    records = table.get(code, {})
    count = sum(records[m][0] for m in range(month - ANALYST_MONTHS, month) if m in records)
    consensus = None
    for m in range(month - 1, month - CONSENSUS_MONTHS - 1, -1):
        if m not in records:
            continue
        calendar_year = (m - 1) // 12  # 月序号减 1 后整除得到自然年
        # 预测年份对齐：fy0 为当期自然年，fyneg1 为上一年，已用实际净利润核对
        value = records[m][1] if calendar_year == year else records[m][2] if calendar_year == year + 1 else None
        if value is not None:
            consensus = value
            break
    return math.log1p(count), consensus


def load_turnover(daily_path, days, code_col):
    """返回 [日, 股] 的换手率矩阵（成交额/总市值），轴与 load_returns 一致。"""
    t = pq.read_table(daily_path, columns=['date', 'code', 'money', 'mkcap'])
    day_idx = np.searchsorted(days, pc.cast(t['date'], 'int32').to_numpy())
    encoded = t['code'].combine_chunks().dictionary_encode()
    mapping = np.array([code_col[c] for c in encoded.dictionary.to_pylist()])
    stock_idx = mapping[encoded.indices.to_numpy(zero_copy_only=False)]
    money = t['money'].to_numpy(zero_copy_only=False).astype(float)
    cap = t['mkcap'].to_numpy(zero_copy_only=False).astype(float)
    ok = (money > 0) & (cap > 0)
    turnover = np.full((len(days), len(code_col)), np.nan, dtype=np.float32)
    turnover[day_idx[ok], stock_idx[ok]] = money[ok] / cap[ok]
    return turnover


def market_vars(code, event_date, days, code_col, ret, market, turnover):
    """[t0-60, t0-1] 的对数日均换手率和市场调整累计涨幅。"""
    empty = None, None
    t0 = int(np.searchsorted(days, event_date))
    if t0 >= len(days) or days[t0] != event_date or code not in code_col or t0 < PRE_WINDOW:
        return empty
    j, lo = code_col[code], t0 - PRE_WINDOW
    turn = turnover[lo:t0, j]
    turn = turn[~np.isnan(turn)]
    excess = ret[lo:t0, j] - market[lo:t0]
    excess = excess[~np.isnan(excess)]
    return (math.log(float(turn.mean())) if len(turn) >= MIN_TURNOVER_OBS else None,
            float(excess.sum()) if len(excess) >= MIN_RUNUP_OBS else None)


def history_vars(panel, cs_path):
    """公司自身历史偏离和同年横截面极端组，返回与 panel 同序的列表。"""
    with Path(cs_path).open(encoding='utf-8-sig', newline='') as f:
        history = {(r['code'], int(r['year'])): float(r['sent_change_mean']) for r in csv.DictReader(f)}
    output = []
    for r in panel:
        code, year, sc = r['code'], int(r['year']), float(r['sent_change_mean'])
        last = history.get((code, year - 1))
        past = [history[code, year - k] for k in (1, 2, 3) if (code, year - k) in history]
        output.append({'sc_yoy': sc - last if last is not None else None,
                       'sc_dev_3y': sc - float(np.mean(past)) if len(past) >= MIN_HISTORY else None,
                       'sc_extreme_q5': None})
    # 同年按事件日排序，同一天发布的事件互不作为同行，只用严格更早的事件
    by_year = {}
    for i, r in enumerate(panel):
        by_year.setdefault(r['year'], []).append(i)
    for indices in by_year.values():
        indices.sort(key=lambda i: int(panel[i]['event_date']))
        peers = []
        for _, day_group in itertools.groupby(indices, key=lambda i: panel[i]['event_date']):
            day_group = list(day_group)
            if len(peers) >= MIN_PEERS:
                cutoff = peers[math.ceil(0.2 * len(peers)) - 1]
                for i in day_group:
                    output[i]['sc_extreme_q5'] = float(float(panel[i]['sent_change_mean']) <= cutoff)
            for i in day_group:
                insort(peers, float(panel[i]['sent_change_mean']))
    return output


def add_interactions(panel_rows, base):
    """在事件面板上缩尾、去均值后构造交互项，直接写入 panel_rows。"""
    def to_array(values):
        return np.array([np.nan if v is None else float(v) for v in values], dtype=float)

    sc = winsorize(base['sent_change_mean'])
    sc_dm = sc - np.nanmean(sc)
    for name, (moderator, is_dummy) in INTERACTIONS.items():
        m = to_array(r[moderator] for r in panel_rows) if moderator in MODERATORS else base[moderator]
        if not is_dummy:
            m = winsorize(m)
            m = m - np.nanmean(m)
        for r, value in zip(panel_rows, sc_dm * m):
            r[name] = None if math.isnan(value) else float(value)


def main(sentence_path, cs_path, panel_path, fin_path, daily_path, analyst_path, out_dir, max_docs=None):
    """构造全部变量，写出 variables.csv 和 hypotheses.json。max_docs 仅用于小样本调试。"""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel(panel_path)
    keys = {(r['code'], r['year']) for r in panel}
    if len(keys) != len(panel):
        raise SystemExit('event_panel 中 (code, year) 有重复')

    docs = sentence_vars(sentence_path, keys, max_docs)
    print(f'句子层面变量完成：{len(docs)} 篇年报', flush=True)

    fin = load_fin(fin_path)
    analyst = load_analyst(analyst_path)
    days, code_col, ret, market = load_returns(daily_path)
    turnover = load_turnover(daily_path, days, code_col)
    print('行情和财务数据载入完成', flush=True)

    history = history_vars(panel, cs_path)
    rows = []
    for r, hist in zip(panel, history):
        code, year, event_date = r['code'], int(r['year']), int(r['event_date'])
        row = {'code': code, 'year': r['year'], **docs.get((code, r['year']), {}), **hist}
        sue = num(r['sue'])
        row['sue_pos'] = None if math.isnan(sue) else float(sue > 0)
        f = fin.get((code, r['year']), {})
        np_parent, assets = f.get('np_parent_company_owners_ttm'), f.get('total_assets')
        net_profit, ocf = f.get('net_profit_ttm'), f.get('net_operate_cash_flow_ttm')
        row['loss'] = None if np_parent is None else float(np_parent < 0)
        non_financial = f.get('company_type') == NON_FINANCIAL_TYPE  # 金融企业的 ROA 和经营现金流不可比，两个变量记缺失
        row['small_profit'] = float(0 <= np_parent / assets < SMALL_PROFIT_ROA) if non_financial and np_parent is not None and assets else None
        row['accruals'] = (net_profit - ocf) / assets if non_financial and None not in (net_profit, ocf) and assets else None
        coverage, consensus = analyst_vars(code, year, event_date, analyst)
        row['ln_analyst_12m'] = coverage
        row['miss_consensus'] = float(np_parent - consensus * 1e4 < 0) if consensus is not None and np_parent is not None else None
        row['ln_turnover_60'], row['runup_60'] = market_vars(code, event_date, days, code_col, ret, market, turnover)
        rows.append(row)

    base = {c: np.array([num(r[c]) for r in panel], dtype=float)
            for c in ['sent_change_mean', 'ln_assets', 'sent_std_orig', 'sent_mean_neutral']}
    add_interactions(rows, base)

    with (out_dir / 'variables.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['code', 'year'] + VARIABLES)
        for r in rows:
            writer.writerow([r['code'], r['year']] + [fmt(r.get(v)) for v in VARIABLES])

    known = set(VARIABLES) | set(ALLOWED_PANEL_COLS)
    for h in HYPOTHESES:
        missing = [t for t in h['extra_terms'] + [h['focal_term']] if t not in known]
        if missing:
            raise SystemExit(f"{h['id']} 引用了不存在的列：{missing}")
        if {'sent_mean_orig', 'sent_mean_neutral', 'sent_change_mean'} <= set(h['extra_terms'] + [h['focal_term']]):
            raise SystemExit(f"{h['id']} 同时包含恒等式的三项")
    with (out_dir / 'hypotheses.json').open('w', encoding='utf-8') as f:
        json.dump(HYPOTHESES, f, ensure_ascii=False, indent=2)
    print(f'共 {len(rows)} 行，{len(VARIABLES)} 个变量，{len(HYPOTHESES)} 条假说，结果保存在 {out_dir}')


if __name__ == '__main__':
    main(sentence_path=OUTPUT_DIR / 'sentiment_analysis/neutral_compare_glm_all/tone_by_sentence.csv',
         cs_path=OUTPUT_DIR / 'sentiment_data/cs_sentiment.csv',
         panel_path=OUTPUT_DIR / 'event_study/event_panel.csv',
         fin_path=DATA_DIR / 'market_data/financial_data.parquet',
         daily_path=DATA_DIR / 'market_data/daily_stock_data.parquet',
         analyst_path=DATA_DIR / 'analyst_forecast/stock_forecast_means.parquet',
         out_dir=OUTPUT_DIR / 'event_study/hypotheses')
