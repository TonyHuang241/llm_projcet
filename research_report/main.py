"""研报估值逻辑图研究的启动入口：修改下面的 TASK 选择要运行的功能，然后在仓库根目录执行 python research_report/main.py
（或 python -m research_report.main）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 让 python research_report/main.py 也能找到本包和 common 包

from research_report import plot_report_logic, report_kg, report_logic, report_text  # noqa: E402
from research_report.config import DATA_DIR, OUTPUT_DIR  # noqa: E402

KG_PDF = DATA_DIR / 'reports' / '20260807-中金公司-比亚迪（002594.SZ）比亚迪股份（1211.HK）：全球化拓展增长边界，技术底座构筑发展基石.pdf'
KG_DIR = OUTPUT_DIR / 'knowledge_graph' / 'byd_cicc_20260807'
REPORT_PDF_DIR = DATA_DIR / 'reports'  # 研报 PDF
REPORT_TEXT = DATA_DIR / 'report_paragraphs.parquet'  # 研报正文段落（推理链抽取的输入）
REPORT_LOGIC_DIR = OUTPUT_DIR / 'report_logic'  # 研报推理链（逻辑图）结果

# ==================== 运行参数 ====================
TASK = 'report_logic'  # 要运行的功能，可选值见下方 TASKS

# 所有可运行的功能，新增功能时在这里加一项
TASKS = {
    # 研报知识图谱（KGGen 框架）：parse 切块 -> Claude 子代理抽取 -> check 校验聚合 -> Claude 子代理去重 -> build 生成 HTML
    'report_kg_parse': lambda: report_kg.parse(KG_PDF, KG_DIR),
    'report_kg_check': lambda: report_kg.check(KG_DIR),
    'report_kg_build': lambda: report_kg.build(KG_PDF, KG_DIR),
    # 研报推理链第一步：PDF 正文抽取（剔除图表、表格、侧栏和免责声明），每行一个段落；已有处理好的文本时可跳过，
    # 直接把 REPORT_TEXT 换成自己的 parquet（至少含 report_id、para_id、text 三列）
    'report_text': lambda: report_text.main(
        pdf_dir=REPORT_PDF_DIR,
        out_path=REPORT_TEXT,
    ),
    # 研报估值逻辑图（框架见 docs/研报逻辑图抽取框架设计.md，会计层清单见 fin_items.py）：概念分层填表 -> 原文连边
    # -> 上位指标向上合并、删除孤立概念 -> 概念补全（先找原文、再按逻辑）-> 剪掉有更长链路的直连、删除没有概念连入的会计科目
    # -> 全局去重 -> JSON/parquet + HTML
    'report_logic': lambda: report_logic.main(
        paras_path=REPORT_TEXT,
        out_dir=REPORT_LOGIC_DIR,
        limit=None,  # 试跑时只处理前多少篇研报（按 report_id 排序），None 表示全部；已完成的大模型调用会自动跳过
    ),
    # 只重画研报逻辑图 HTML（改了样式时用，不调用大模型）
    'report_logic_html': lambda: plot_report_logic.main(
        graphs_path=REPORT_LOGIC_DIR / 'report_graphs.json',
        out_html=REPORT_LOGIC_DIR / 'report_logic_graphs.html',
    ),
}
# ==================================================

if __name__ == '__main__':
    if TASK not in TASKS:
        raise SystemExit(f'未知的 TASK: {TASK}，可选: {list(TASKS)}')
    TASKS[TASK]()
