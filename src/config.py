"""目录约定：本包只放代码和提示词；数据、输出、文档、文献和 .env 都在仓库根目录。"""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
DATA_DIR = ROOT_DIR / 'data'  # 原始数据与中间数据（不入 git）
OUTPUT_DIR = ROOT_DIR / 'output'  # 分析结果（不入 git）
PROMPT_DIR = PKG_DIR / 'prompts'


class OutDir:
    """输出目录内部的布局：最外层只有一个入口页面，结果数据都在 data/ 下，大模型缓存和进度日志在 cache/ 下。"""

    def __init__(self, root):
        self.root = Path(root)
        self.viewer = self.root / '研报逻辑图.html'  # 入口：双击打开，选中一篇研报时才读取它的数据并绘图
        self.data = self.root / 'data'  # edges / nodes / relations.parquet
        self.graphs = self.data / 'graphs'  # 每篇研报一个逻辑图数据文件，加上目录 index.js 和全局名称表 global_names.js
        self.compare = self.data / 'compare'  # 框架对比的结果
        self.cache = self.root / 'cache'  # 各步大模型调用的缓存（重跑时跳过已完成的调用）
        self.log = self.cache / 'progress.log'

    def make(self):
        for d in (self.data, self.cache):
            d.mkdir(parents=True, exist_ok=True)
        return self
