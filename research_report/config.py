"""本研究线的目录约定：代码、提示词、数据、输出都放在本包目录下；.env 在仓库根目录。"""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
DATA_DIR = PKG_DIR / 'data'  # 原始数据与中间数据（不入 git）
OUTPUT_DIR = PKG_DIR / 'output'  # 分析结果（不入 git）
PROMPT_DIR = PKG_DIR / 'prompts'
