# -*- coding: utf-8 -*-
"""模型训练和评估功能"""

import os
import torch
from datetime import datetime
from ultralytics import YOLO


def train_yolov8_model(yaml_path, processing_mode, project_path):
    """训练YOLOv8模型"""
    # 检查GPU可用性
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {'GPU' if device == '0' else 'CPU'}")

    # 创建模型（从零开始，不使用预训练权重）
    model = YOLO('yolov8m.yaml')  # 使用yolov8m.yaml架构

    # 设置项目名称和路径
    project_name = f'bee_training_{processing_mode}'
    train_project_path = os.path.join(project_path, 'models', 'trained')

    # 训练参数 - 从零开始训练需要更多epoch
    train_args = {
        'data': yaml_path,  # 数据集配置文件路径
        'epochs': 100,  # 训练的总轮数，从零开始训练需要更多轮次来学习特征
        'imgsz': 640,  # 输入图像的尺寸，YOLOv8支持多种尺寸，640是常用尺寸
        'batch': 16,  # 每个批次的图像数量，根据GPU内存调整
        'name': project_name,  # 训练任务的名称，用于标识不同的训练运行
        'project': train_project_path,  # 训练结果保存的项目路径
        'device': device,  # 使用的设备，'0'表示GPU，空字符串表示CPU

        # 优化器参数 - 从零开始训练需要更小的学习率
        'lr0': 0.001,  # 初始学习率，从零开始训练需要较小的学习率以避免梯度爆炸
        'lrf': 0.01,  # 最终学习率 = 初始学习率 × lrf，用于学习率衰减
        'momentum': 0.937,  # 动量参数，帮助优化器在相关方向上加速并抑制振荡
        'weight_decay': 0.0005,  # 权重衰减(L2正则化)系数，防止过拟合
        'warmup_epochs': 5.0,  # 热身训练的轮数，逐渐增加学习率到初始值

        # 数据增强参数 - 这些是在训练过程中实时应用的图像变换
        'hsv_h': 0.015,  # 图像色调(Hue)变化幅度，增强模型对颜色变化的鲁棒性
        'hsv_s': 0.7,  # 图像饱和度(Saturation)变化幅度
        'hsv_v': 0.4,  # 图像亮度(Value)变化幅度
        'translate': 0.2,  # 图像平移幅度，增强模型对位置变化的鲁棒性
        'scale': 0.5,  # 图像缩放幅度，增强模型对尺度变化的鲁棒性
        'fliplr': 0.5,  # 图像水平翻转的概率，增强模型对镜像变化的鲁棒性
        'mosaic': 1.0,  # 使用马赛克数据增强的概率，将4张图像拼接为1张

        # 记录详细日志
        'verbose': True,  # 是否显示详细的训练进度和信息
    }

    print("开始训练YOLOv8模型...")
    results = model.train(**train_args)

    # 返回最佳模型路径
    best_model_path = os.path.join(train_project_path, project_name, 'weights', 'best.pt')
    return best_model_path, results


def evaluate_model(model_path, yaml_path, processing_mode, project_path):
    """评估训练好的模型"""
    model = YOLO(model_path)

    # 设置评估结果保存路径
    eval_name = f'model_evaluation_{processing_mode}'
    eval_path = os.path.join(project_path, 'results')

    # 在测试集上评估
    print("在测试集上评估模型...")
    results = model.val(
        data=yaml_path,
        split='test',
        name=eval_name,
        project=eval_path
    )

    # 保存评估结果到文本文件
    eval_results_path = os.path.join(eval_path, eval_name, 'evaluation_results.txt')
    with open(eval_results_path, 'w') as f:
        f.write(f"评估时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"模型路径: {model_path}\n")
        f.write(f"处理模式: {processing_mode}\n")
        f.write(f"测试集评估结果:\n")
        f.write(f"mAP50: {results.box.map50:.4f}\n")
        f.write(f"mAP50-95: {results.box.map:.4f}\n")
        f.write(f"精确度: {results.box.mp:.4f}\n")
        f.write(f"召回率: {results.box.mr:.4f}\n")
        f.write(f"F1分数: {2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16):.4f}\n")

    print(f"测试集评估结果:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    print(f"精确度: {results.box.mp:.4f}")
    print(f"召回率: {results.box.mr:.4f}")
    print(f"F1分数: {2 * (results.box.mp * results.box.mr) / (results.box.mp + results.box.mr + 1e-16):.4f}")
    print(f"评估结果已保存到: {eval_results_path}")

    return results