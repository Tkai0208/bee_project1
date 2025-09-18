# -*- coding: utf-8 -*-
"""项目配置和路径设置"""

import os

# 1. 项目根目录（固定到 D:/bee_project1，无需自动推断）
PROJECT_PATH = r'D:/bee_project1'

# 2. 数据集根目录
DATASETS_PATH = os.path.join(PROJECT_PATH, 'datasets')

# 3. 图像处理模式列表
PROCESSING_MODES = [
    'original',      # 原始图像
    'histogram_eq',  # 直方图均衡化
    'sharpening',    # 锐化
    'contrast',      # 对比度增强
    'hsv',           # HSV色彩空间
    'lab'            # LAB色彩空间
]

# 4. 当前要使用的模式（用索引 3 → contrast）
PROCESSING_MODE = PROCESSING_MODES[3]

# 5. 创建必要的输出目录（不存在则自动创建）
os.makedirs(os.path.join(PROJECT_PATH, 'models', 'trained'), exist_ok=True)
os.makedirs(os.path.join(PROJECT_PATH, 'results'), exist_ok=True)
os.makedirs(DATASETS_PATH, exist_ok=True)