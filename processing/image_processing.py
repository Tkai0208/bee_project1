# -*- coding: utf-8 -*-
"""图像处理功能"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class AdvancedImageProcessing:
    """增强的图像处理算法实现"""

    def __init__(self):
        self.results = {}

    def load_image(self, image_path):
        """加载图像"""
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 转换为RGB格式用于显示
        self.rgb_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        return self.original_image

    def histogram_equalization(self):
        """直方图均衡化 - 增强对比度"""
        # 转换到YUV色彩空间，对Y通道进行均衡化
        yuv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2YUV)
        yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
        equalized = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return equalized

    def sharpening_filter(self):
        """锐化滤波 - 增强边缘"""
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(self.original_image, -1, kernel)
        return sharpened

    def contrast_enhancement(self):
        """对比度增强 - 使用CLAHE"""
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced

    def hsv_conversion(self):
        """HSV色彩空间转换"""
        hsv = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
        # 增加饱和度以增强颜色
        hsv[:, :, 1] = hsv[:, :, 1] * 1.2
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        # 转换回BGR
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return enhanced

    def lab_conversion(self):
        """LAB色彩空间转换"""
        lab = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2LAB)
        # 增强亮度通道
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced

    def process_for_training(self, image_path, output_path, processing_mode):
        """为训练处理单张图像"""
        self.load_image(image_path)

        if processing_mode == 'original':
            processed_image = self.original_image
        elif processing_mode == 'histogram_eq':
            processed_image = self.histogram_equalization()
        elif processing_mode == 'sharpening':
            processed_image = self.sharpening_filter()
        elif processing_mode == 'contrast':
            processed_image = self.contrast_enhancement()
        elif processing_mode == 'hsv':
            processed_image = self.hsv_conversion()
        elif processing_mode == 'lab':
            processed_image = self.lab_conversion()
        else:
            processed_image = self.original_image

        # 确保图像是3通道的BGR格式
        if len(processed_image.shape) == 2:
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_GRAY2BGR)
        elif processed_image.shape[2] == 4:
            processed_image = processed_image[:, :, :3]

        cv2.imwrite(output_path, processed_image)
        return processed_image

    def visualize_processing_steps(self, image_path, output_dir):
        """可视化所有处理步骤"""
        os.makedirs(output_dir, exist_ok=True)
        self.load_image(image_path)

        # 创建对比图
        plt.figure(figsize=(15, 10))

        # 原始图像
        plt.subplot(2, 3, 1)
        plt.imshow(self.rgb_image)
        plt.title('原始图像')
        plt.axis('off')

        # 直方图均衡化
        hist_eq = self.histogram_equalization()
        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(hist_eq, cv2.COLOR_BGR2RGB))
        plt.title('直方图均衡化')
        plt.axis('off')

        # 锐化
        sharpened = self.sharpening_filter()
        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))
        plt.title('锐化滤波')
        plt.axis('off')

        # 对比度增强
        contrast = self.contrast_enhancement()
        plt.subplot(2, 3, 4)
        plt.imshow(cv2.cvtColor(contrast, cv2.COLOR_BGR2RGB))
        plt.title('对比度增强')
        plt.axis('off')

        # HSV转换
        hsv = self.hsv_conversion()
        plt.subplot(2, 3, 5)
        plt.imshow(cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB))
        plt.title('HSV色彩空间')
        plt.axis('off')

        # LAB转换
        lab = self.lab_conversion()
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(lab, cv2.COLOR_BGR2RGB))
        plt.title('LAB色彩空间')
        plt.axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processing_steps.png'), dpi=300, bbox_inches='tight')
        plt.show()

        return True


def batch_process_images(processing_mode, datasets_path):
    """批量处理所有训练、验证和测试图像"""
    splits = ['train', 'valid', 'test']
    processor = AdvancedImageProcessing()

    for split in splits:
        # 创建输出目录
        output_dir = os.path.join(datasets_path, 'processed_images', processing_mode, split)
        os.makedirs(output_dir, exist_ok=True)

        # 获取原始图像目录
        input_dir = os.path.join(datasets_path, 'images', split)

        if not os.path.exists(input_dir):
            print(f"警告: {input_dir} 不存在，跳过")
            continue

        # 处理所有图像
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        print(f"处理 {split} 集的图像，模式: {processing_mode}")
        for img_file in tqdm(image_files):
            input_path = os.path.join(input_dir, img_file)
            output_path = os.path.join(output_dir, img_file)

            # 如果输出文件已存在，则跳过
            if os.path.exists(output_path):
                continue

            processor.process_for_training(input_path, output_path, processing_mode)

    print(f"所有图像已处理完成，模式: {processing_mode}")
    print(f"处理后的图像保存在: {os.path.join(datasets_path, 'processed_images', processing_mode)}")