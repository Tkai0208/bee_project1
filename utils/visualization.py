# -*- coding: utf-8 -*-
"""可视化功能"""

import os
from processing.image_processing import AdvancedImageProcessing


def visualize_processing_examples(datasets_path, project_path):
    """可视化处理步骤示例"""
    processor = AdvancedImageProcessing()
    examples_dir = os.path.join(project_path, 'results', 'processing_examples')
    os.makedirs(examples_dir, exist_ok=True)

    # 查找示例图像
    splits = ['train', 'valid', 'test']
    example_image = None

    for split in splits:
        input_dir = os.path.join(datasets_path, 'images', split)
        if os.path.exists(input_dir):
            image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if image_files:
                example_image = os.path.join(input_dir, image_files[0])
                break

    if example_image:
        print(f"使用示例图像: {example_image}")
        processor.visualize_processing_steps(example_image, examples_dir)
        return True
    else:
        print("未找到示例图像")
        return False