# -*- coding: utf-8 -*-
"""数据集分析功能"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def analyze_dataset(datasets_path, project_path):
    """分析数据集统计信息"""
    stats_dir = os.path.join(project_path, 'results', 'dataset_analysis')
    os.makedirs(stats_dir, exist_ok=True)

    splits = ['train', 'valid', 'test']
    stats = {}

    for split in splits:
        labels_dir = os.path.join(datasets_path, 'labels', split)
        images_dir = os.path.join(datasets_path, 'images', split)

        if not os.path.exists(labels_dir) or not os.path.exists(images_dir):
            print(f"跳过 {split} 集，目录不存在")
            continue

        # 统计标注文件
        label_files = [f for f in os.listdir(labels_dir) if f.endswith('.txt')]
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # 统计边界框
        bbox_counts = []
        bbox_sizes = []

        for label_file in label_files:
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
                bbox_count = len([line for line in lines if line.strip()])
                bbox_counts.append(bbox_count)

                # 解析边界框尺寸
                for line in lines:
                    if line.strip():
                        _, xc, yc, w, h = map(float, line.strip().split())
                        bbox_sizes.append((w, h))

        # 转换为numpy数组以便计算统计信息
        bbox_counts = np.array(bbox_counts)
        bbox_sizes = np.array(bbox_sizes)

        stats[split] = {
            'image_count': len(image_files),
            'label_count': len(label_files),
            'total_bboxes': np.sum(bbox_counts),
            'avg_bbox_per_image': np.mean(bbox_counts) if len(bbox_counts) > 0 else 0,
            'bbox_size_mean': np.mean(bbox_sizes, axis=0) if len(bbox_sizes) > 0 else [0, 0],
            'bbox_size_std': np.std(bbox_sizes, axis=0) if len(bbox_sizes) > 0 else [0, 0]
        }

    # 保存统计结果
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(os.path.join(stats_dir, 'dataset_statistics.csv'))

    # 可视化统计信息
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # 图像数量
    splits_list = list(stats.keys())
    image_counts = [stats[split]['image_count'] for split in splits_list]
    axes[0, 0].bar(splits_list, image_counts, color=['blue', 'green', 'red'])
    axes[0, 0].set_title('图像数量')
    axes[0, 0].set_ylabel('数量')

    # 边界框数量
    bbox_counts = [stats[split]['total_bboxes'] for split in splits_list]
    axes[0, 1].bar(splits_list, bbox_counts, color=['blue', 'green', 'red'])
    axes[0, 1].set_title('边界框数量')
    axes[0, 1].set_ylabel('数量')

    # 每图像平均边界框数
    avg_bbox = [stats[split]['avg_bbox_per_image'] for split in splits_list]
    axes[1, 0].bar(splits_list, avg_bbox, color=['blue', 'green', 'red'])
    axes[1, 0].set_title('每图像平均边界框数')
    axes[1, 0].set_ylabel('平均数量')

    # 边界框尺寸分布
    all_sizes = []
    for split in splits_list:
        if 'train' in stats and len(stats['train'].get('bbox_sizes', [])) > 0:
            all_sizes.extend(stats['train'].get('bbox_sizes', []))

    if all_sizes:
        all_sizes = np.array(all_sizes)
        axes[1, 1].scatter(all_sizes[:, 0], all_sizes[:, 1], alpha=0.5)
        axes[1, 1].set_xlabel('宽度 (归一化)')
        axes[1, 1].set_ylabel('高度 (归一化)')
        axes[1, 1].set_title('边界框尺寸分布')

    plt.tight_layout()
    plt.savefig(os.path.join(stats_dir, 'dataset_statistics.png'), dpi=300, bbox_inches='tight')
    plt.show()

    print("数据集统计分析完成")
    return stats


def create_dataset_yaml(processing_mode, datasets_path, project_path):
    """创建YOLO数据集配置文件"""
    if processing_mode == 'original':
        # 原始模式：图像和标签都在 datasets/ 目录下
        dataset_path = os.path.abspath(datasets_path)
        yaml_content = {
            'path': dataset_path,
            'train': 'images/train',  # 相对于 datasets_path
            'val': 'images/valid',  # 相对于 datasets_path
            'test': 'images/test',  # 相对于 datasets_path
            'nc': 1,
            'names': ['bee']
        }
    else:
        # 处理模式：图像在 processed_images/{mode}/ 目录，标签在 labels/ 目录
        dataset_path = os.path.abspath(datasets_path)
        yaml_content = {
            'path': dataset_path,
            'train': f'processed_images/{processing_mode}/train',  # 相对于 datasets_path
            'val': f'processed_images/{processing_mode}/valid',  # 相对于 datasets_path
            'test': f'processed_images/{processing_mode}/test',  # 相对于 datasets_path
            'nc': 1,
            'names': ['bee']
        }

    yaml_path = os.path.join(project_path, f'bee_{processing_mode}.yaml')
    with open(yaml_path, 'w') as f:
        import yaml
        yaml.dump(yaml_content, f, default_flow_style=False)

    print(f"数据集配置文件已创建: {yaml_path}")

    # 验证数据集配置
    if not validate_dataset_config(yaml_path):
        print(f"警告: 数据集配置验证失败，请检查路径: {yaml_path}")

    return yaml_path


def validate_dataset_config(yaml_path):
    """验证数据集配置是否正确"""
    try:
        with open(yaml_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)

        # 检查路径是否存在
        base_path = config['path']

        # 使用配置中的实际名称
        splits = {
            'train': config.get('train', 'train'),
            'val': config.get('val', 'valid'),
            'test': config.get('test', 'test')
        }

        # 检查图像目录
        for split_name, split_dir in splits.items():
            image_path = os.path.join(base_path, split_dir)
            if not os.path.exists(image_path):
                print(f"错误: {split_name} 图像路径不存在: {image_path}")
                return False

            # 检查图像文件是否存在
            image_files = [f for f in os.listdir(image_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(image_files) == 0:
                print(f"警告: {split_name} 图像目录中没有图像文件: {image_path}")

        # 检查标签目录（对于所有模式，标签都在 datasets/labels/ 目录下）
        for split_name in splits.keys():
            label_path = os.path.join(base_path, 'labels', split_name)
            if not os.path.exists(label_path):
                print(f"错误: {split_name} 标签路径不存在: {label_path}")
                return False

            # 检查标签文件是否存在
            label_files = [f for f in os.listdir(label_path) if f.endswith('.txt')]
            if len(label_files) == 0:
                print(f"警告: {split_name} 标签目录中没有标签文件: {label_path}")

        print("数据集配置验证通过")
        return True
    except Exception as e:
        print(f"数据集配置验证失败: {e}")
        return False