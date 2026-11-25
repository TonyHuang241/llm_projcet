"""目录约定和跨步骤共用的数据文件路径：src/ 只放代码、提示词和样本清单；数据、输出、文献和 .env 都在仓库根目录。

各步骤自己的默认运行参数写在该步骤的目录里（src/<步骤>/），不放这里；下游步骤读上游的成品时通过上游目录里的 load.py，
不自己拼路径。

两个概念（详见 README）：
  抽取版本（version）  一套抽取规则（代码 + 提示词 + 每块字数等规则参数），每个版本一个抽取库 output/graph/<版本>/，
                       同一版本下每篇研报的逻辑结构只存一份。版本登记在 src/graph/__init__.py 的 VERSIONS。
  样本（sample）       分析用哪些研报：一张“研报 ID → 取自哪个版本的库”的清单，登记在 src/samples/__init__.py 的 SAMPLES。
输出目录按这两个概念组织，统一由下面的函数给出：
  output/graph/<版本>/                     抽取库（与样本无关）
  output/measure/<版本>/                   逻辑结构变量（与样本无关，库里每篇一行）
  output/accuracy/<样本>/                  预测准确度（只和研报范围有关，与抽取版本无关）
  output/studies/<研究>/<样本>_<版本>[_后缀]/   专题研究的输出和文档
  output/experiments/<名字>/               一次性的实验（如抽取稳定性）
  output/archive/                          已不维护的旧文档
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / 'data'  # 原始数据与中间数据（不入 git）
OUTPUT_DIR = ROOT_DIR / 'output'  # 各步骤的结果（只有研究文档和 archive/ 入 git，见 .gitignore）

# 跨步骤共用的数据文件
TXT_DIR = DATA_DIR / 'HS300_reports_2024'  # 研报纯文本（MM/DD/<GUID>.txt）
PARAGRAPHS = DATA_DIR / 'hs300_2024_paragraphs.parquet'  # text 步骤写出的正文段落表（旁边有同名 .txt 便于人工检查）
REPORTS = DATA_DIR / 'hs300_2024_reports.parquet'  # text 步骤写出的研报元数据表（旁边有同名 .csv）
EST_DIR = DATA_DIR / 'EST_COMP_TN'  # 分析师盈利预测，按发布日期分月：<年>/<年>-<月>.parquet
FINANCIAL = DATA_DIR / 'market_data' / 'financial_data.parquet'  # 财务数据（年报实际值、公告日、总资产、负债）
DAILY = DATA_DIR / 'market_data' / 'daily_stock_data.parquet'  # 个股日行情（收盘价、总市值、停牌）
INDEX_STOCK = DATA_DIR / 'market_data' / 'index_stock.parquet'  # 指数成分股（date、code、index_code、index_name）

DEFAULT_VERSION = 'v1'  # 默认的抽取版本（graph、unify、measure 和研究的默认值）
DEFAULT_SAMPLE = 'hs300_deep'  # 默认的样本（graph 读成品、accuracy 和研究的默认值）


def out_dir(step, sub=None):
    """步骤的输出目录 output/<步骤>/，给了 sub 时是 output/<步骤>/<sub>/。"""
    d = OUTPUT_DIR / step
    return d / sub if sub else d


def graph_dir(version):
    """抽取版本的库 output/graph/<版本>/（布局见 src/graph/store.py 的 DagOut）。"""
    return out_dir('graph', version)


def measure_dir(version):
    """抽取版本的逻辑结构变量 output/measure/<版本>/。"""
    return out_dir('measure', version)


def accuracy_dir(sample):
    """样本的预测准确度 output/accuracy/<样本>/。"""
    return out_dir('accuracy', sample)


def study_dir(study, sample, version, suffix=''):
    """专题研究的输出目录 output/studies/<研究>/<样本>_<版本><后缀>/（如 hs300_deep_v1、hs300_deep_v1_log1p）。"""
    return out_dir('studies', study) / f'{sample}_{version}{suffix}'


EXPERIMENTS_DIR = OUTPUT_DIR / 'experiments'  # 一次性的实验：graph_stability/ 是同一批研报重复抽取的结果（衡量抽取噪声）
ARCHIVE_DIR = OUTPUT_DIR / 'archive'  # 已不维护的旧文档（原先文档目录下的 archive/ 搬来的，2026-11-23 之前的文档），入 git
