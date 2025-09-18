# -*- coding: utf-8 -*-
"""蜜蜂检测项目主程序"""

import os
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入自定义模块
from config.settings import PROJECT_PATH, DATASETS_PATH, PROCESSING_MODE
from data.conversion import process_all_coco_annotations
from processing.image_processing import batch_process_images
from analysis.dataset_analysis import analyze_dataset, create_dataset_yaml
from model.training import train_yolov8_model, evaluate_model
from utils.visualization import visualize_processing_examples


def main():
    """主函数"""
    print("蜜蜂检测项目启动...")

    # 检查数据格式转换
    print("检查数据格式转换...")
    labels_exist = all(
        os.path.exists(os.path.join(DATASETS_PATH, 'labels', split)) and
        len(os.listdir(os.path.join(DATASETS_PATH, 'labels', split))) > 0
        for split in ['train', 'valid', 'test']
    )

    if not labels_exist:
        print("未找到标签文件，开始转换COCO格式到YOLO格式...")
        total_img, total_box, total_skip = process_all_coco_annotations(DATASETS_PATH)

        if total_img == 0:
            print("错误: 没有成功转换任何标注文件，请检查数据集路径和文件")
            return
    else:
        print("标签文件已存在，跳过转换步骤")

    # 数据集分析
    print("分析数据集统计信息...")
    dataset_stats = analyze_dataset(DATASETS_PATH, PROJECT_PATH)

    # 可视化处理步骤
    print("可视化处理步骤...")
    visualize_processing_examples(DATASETS_PATH, PROJECT_PATH)

    # 处理图像
    print(f"开始处理图像，模式: {PROCESSING_MODE}")
    batch_process_images(PROCESSING_MODE, DATASETS_PATH)

    # 创建数据集配置文件
    yaml_path = create_dataset_yaml(PROCESSING_MODE, DATASETS_PATH, PROJECT_PATH)

    # 训练模型
    print(f"开始训练模型，处理模式: {PROCESSING_MODE}")
    best_model_path, results = train_yolov8_model(yaml_path, PROCESSING_MODE, PROJECT_PATH)
    print(f"训练完成，最佳模型保存在: {best_model_path}")

    # 评估模型
    eval_results = evaluate_model(best_model_path, yaml_path, PROCESSING_MODE, PROJECT_PATH)

    print("项目执行完成!")


if __name__ == "__main__":
    main()