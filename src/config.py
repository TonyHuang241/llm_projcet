"""目录约定：本包只放代码和提示词；数据、输出、文档、文献和 .env 都在仓库根目录。"""
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent
ROOT_DIR = PKG_DIR.parent
DATA_DIR = ROOT_DIR / 'data'  # 原始数据与中间数据（不入 git）
OUTPUT_DIR = ROOT_DIR / 'output'  # 分析结果（不入 git）
PROMPT_DIR = PKG_DIR / 'prompts'
