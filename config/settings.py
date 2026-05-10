"""
项目全局配置
"""
from pathlib import Path

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# 数据目录
DATA_DIR = PROJECT_ROOT / "data"                    # 数据目录
RAW_DATA_DIR = DATA_DIR / "raw"                     # 原始数据目录
INTERMEDIATE_DATA_DIR = DATA_DIR / "intermediate"   # 中间数据目录
PROCESSED_DATA_DIR = DATA_DIR / "processed"         # 处理后数据目录
COOKIES_DIR = DATA_DIR / "cookies"                  # 浏览器cookies目录

# 模型目录
MODEL_DIR = PROJECT_ROOT / "models"
