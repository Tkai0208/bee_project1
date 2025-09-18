# -*- coding: utf-8 -*-
"""
蜜蜂检测系统图形界面程序 - 优化版
使用PyQt5和YOLOv8训练模型
"""

import sys
import os
import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from collections import defaultdict

from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QPushButton, QFileDialog, QListWidget, QTabWidget,
                             QGroupBox, QSpinBox, QDoubleSpinBox, QComboBox, QMessageBox,
                             QProgressBar, QSlider, QSplitter, QTextEdit, QTableWidget,
                             QTableWidgetItem, QHeaderView, QProgressDialog, QDialog,
                             QDialogButtonBox, QSizePolicy, QScrollArea)
from PyQt5.QtCore import Qt, QTimer, QSize
from PyQt5.QtGui import QPixmap, QImage, QIcon, QFont, QPalette, QColor

from ultralytics import YOLO
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BeeDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.model = None
        self.current_image = None
        self.current_image_path = None
        self.dataset_path = None
        self.image_files = []
        self.current_index = 0
        self.batch_results = []  # 存储批量处理结果

        self.init_ui()
        self.load_settings()

    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("蜜蜂检测系统")
        self.setGeometry(100, 100, 1400, 900)

        # 中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 主布局
        main_layout = QHBoxLayout(central_widget)

        # 左侧面板 - 图像选择和控件
        left_panel = QWidget()
        left_panel.setMaximumWidth(350)
        left_layout = QVBoxLayout(left_panel)

        # 模型加载部分
        model_group = QGroupBox("模型设置")
        model_layout = QVBoxLayout(model_group)

        self.btn_load_model = QPushButton("加载训练模型")
        self.btn_load_model.clicked.connect(self.load_model)
        model_layout.addWidget(self.btn_load_model)

        self.model_status = QLabel("未加载模型")
        model_layout.addWidget(self.model_status)

        # 数据集选择
        dataset_group = QGroupBox("数据集")
        dataset_layout = QVBoxLayout(dataset_group)

        self.btn_load_dataset = QPushButton("选择数据集文件夹")
        self.btn_load_dataset.clicked.connect(self.load_dataset)
        dataset_layout.addWidget(self.btn_load_dataset)

        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self.select_image)
        dataset_layout.addWidget(self.image_list)

        # 检测参数
        params_group = QGroupBox("检测参数")
        params_layout = QVBoxLayout(params_group)

        # 置信度阈值
        conf_layout = QHBoxLayout()
        conf_label = QLabel("置信度阈值:")
        conf_label.setToolTip("只显示置信度高于此值的检测结果。值越高，检测越严格，但可能漏检。")
        conf_layout.addWidget(conf_label)
        self.conf_threshold = QDoubleSpinBox()
        self.conf_threshold.setRange(0.0, 1.0)
        self.conf_threshold.setValue(0.25)
        self.conf_threshold.setSingleStep(0.05)
        self.conf_threshold.valueChanged.connect(self.update_params_info)
        conf_layout.addWidget(self.conf_threshold)
        params_layout.addLayout(conf_layout)

        # IoU阈值
        iou_layout = QHBoxLayout()
        iou_label = QLabel("IoU阈值:")
        iou_label.setToolTip("用于非极大值抑制(NMS)，值越高，保留的重叠检测框越多。")
        iou_layout.addWidget(iou_label)
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.0, 1.0)
        self.iou_threshold.setValue(0.7)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.valueChanged.connect(self.update_params_info)
        iou_layout.addWidget(self.iou_threshold)
        params_layout.addLayout(iou_layout)

        # 参数说明
        self.params_info = QLabel("")
        self.params_info.setWordWrap(True)
        self.params_info.setStyleSheet("background-color: #f8f8f8; padding: 5px; border: 1px solid #ddd;")
        self.update_params_info()
        params_layout.addWidget(self.params_info)

        # 添加到左侧布局
        left_layout.addWidget(model_group)
        left_layout.addWidget(dataset_group)
        left_layout.addWidget(params_group)
        left_layout.addStretch()

        # 右侧面板 - 图像显示和结果
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 图像显示区域
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(600, 400)
        self.image_label.setText("请加载图像")
        self.image_label.setStyleSheet("border: 1px solid gray; background-color: #f0f0f0;")
        image_scroll.setWidget(self.image_label)
        right_layout.addWidget(image_scroll, 3)  # 图像区域占用3份空间

        # 控制按钮
        control_layout = QHBoxLayout()
        self.btn_prev = QPushButton("上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        control_layout.addWidget(self.btn_prev)

        self.btn_next = QPushButton("下一张")
        self.btn_next.clicked.connect(self.next_image)
        control_layout.addWidget(self.btn_next)

        self.btn_detect = QPushButton("检测蜜蜂")
        self.btn_detect.clicked.connect(self.detect_bees)
        self.btn_detect.setEnabled(False)
        control_layout.addWidget(self.btn_detect)

        self.btn_batch = QPushButton("批量检测")
        self.btn_batch.clicked.connect(self.batch_detect)
        self.btn_batch.setEnabled(False)
        control_layout.addWidget(self.btn_batch)

        self.btn_stats = QPushButton("查看统计")
        self.btn_stats.clicked.connect(self.show_statistics)
        self.btn_stats.setEnabled(False)
        control_layout.addWidget(self.btn_stats)

        right_layout.addLayout(control_layout)

        # 结果标签
        result_group = QGroupBox("检测结果")
        result_layout = QVBoxLayout(result_group)
        self.result_label = QLabel("检测结果将显示在这里")
        self.result_label.setWordWrap(True)
        result_layout.addWidget(self.result_label)
        right_layout.addWidget(result_group, 1)  # 结果区域占用1份空间

        # 统计信息标签
        stats_group = QGroupBox("统计信息")
        stats_layout = QVBoxLayout(stats_group)
        self.stats_label = QLabel("统计信息将显示在这里")
        self.stats_label.setWordWrap(True)
        stats_layout.addWidget(self.stats_label)
        right_layout.addWidget(stats_group, 1)  # 统计区域占用1份空间

        # 添加到主布局
        main_layout.addWidget(left_panel)
        main_layout.addWidget(right_panel)

        # 状态栏
        self.statusBar().showMessage("就绪")

        # 应用样式
        self.apply_styles()

    def apply_styles(self):
        """应用样式表"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #cccccc;
                border-radius: 5px;
                margin-top: 1ex;
                padding-top: 10px;
                background-color: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 8px 16px;
                text-align: center;
                text-decoration: none;
                font-size: 14px;
                margin: 4px 2px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:disabled {
                background-color: #cccccc;
            }
            QLabel {
                padding: 5px;
            }
            QListWidget {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QDoubleSpinBox {
                padding: 3px;
                border: 1px solid #cccccc;
                border-radius: 3px;
                background-color: white;
            }
        """)

    def update_params_info(self):
        """更新参数说明"""
        conf = self.conf_threshold.value()
        iou = self.iou_threshold.value()

        info_text = (
            f"<b>当前参数说明:</b><br>"
            f"• <b>置信度阈值 ({conf})</b>: 只显示置信度高于此值的检测结果<br>"
            f"• <b>IoU阈值 ({iou})</b>: 用于非极大值抑制，值越高保留的重叠检测框越多"
        )
        self.params_info.setText(info_text)

    def load_settings(self):
        """加载应用程序设置"""
        # 这里可以添加设置加载逻辑
        pass

    def save_settings(self):
        """保存应用程序设置"""
        # 这里可以添加设置保存逻辑
        pass

    def load_model(self):
        """加载训练好的模型"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择模型文件", "", "模型文件 (*.pt)"
        )

        if file_path:
            try:
                self.model = YOLO(file_path)
                self.model_status.setText(f"已加载模型: {os.path.basename(file_path)}")
                self.btn_detect.setEnabled(True)
                self.btn_batch.setEnabled(True)
                self.statusBar().showMessage("模型加载成功")
            except Exception as e:
                QMessageBox.critical(self, "错误", f"加载模型失败: {str(e)}")

    def load_dataset(self):
        """加载数据集"""
        folder_path = QFileDialog.getExistingDirectory(
            self, "选择数据集文件夹"
        )

        if folder_path:
            self.dataset_path = folder_path
            self.scan_image_files()

    def scan_image_files(self):
        """扫描图像文件"""
        if not self.dataset_path:
            return

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        self.image_files = []

        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.lower().endswith(image_extensions):
                    self.image_files.append(os.path.join(root, file))

        self.image_list.clear()
        for file_path in self.image_files:
            self.image_list.addItem(os.path.basename(file_path))

        self.statusBar().showMessage(f"找到 {len(self.image_files)} 张图像")

    def select_image(self, index):
        """选择图像"""
        if index < 0 or index >= len(self.image_files):
            return

        self.current_index = index
        self.current_image_path = self.image_files[index]

        # 加载并显示图像
        self.load_and_display_image()

    def load_and_display_image(self):
        """加载并显示图像"""
        if not self.current_image_path:
            return

        pixmap = QPixmap(self.current_image_path)
        if not pixmap.isNull():
            # 缩放图像以适应标签大小
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)
            self.current_image = cv2.imread(self.current_image_path)
            self.statusBar().showMessage(f"已加载: {os.path.basename(self.current_image_path)}")

            # 重置结果标签
            self.result_label.setText("点击'检测蜜蜂'按钮开始检测")
        else:
            self.image_label.setText("无法加载图像")

    def prev_image(self):
        """显示上一张图像"""
        if self.image_files and self.current_index > 0:
            self.current_index -= 1
            self.image_list.setCurrentRow(self.current_index)

    def next_image(self):
        """显示下一张图像"""
        if self.image_files and self.current_index < len(self.image_files) - 1:
            self.current_index += 1
            self.image_list.setCurrentRow(self.current_index)

    def detect_bees(self):
        """检测当前图像中的蜜蜂"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        if self.current_image is None:
            QMessageBox.warning(self, "警告", "请先加载图像")
            return

        # 获取参数
        conf = self.conf_threshold.value()
        iou = self.iou_threshold.value()

        # 执行检测
        try:
            results = self.model.predict(
                source=self.current_image,
                conf=conf,
                iou=iou,
                verbose=False
            )

            # 处理结果
            self.process_detection_results(results[0])

        except Exception as e:
            QMessageBox.critical(self, "错误", f"检测过程中出错: {str(e)}")

    def process_detection_results(self, result):
        """处理检测结果"""
        # 绘制检测结果
        plotted_image = result.plot()

        # 转换OpenCV图像为QPixmap
        rgb_image = cv2.cvtColor(plotted_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # 缩放并显示图像
        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

        # 显示检测统计信息
        num_bees = len(result.boxes) if result.boxes is not None else 0
        confidences = [f"{box.conf.item():.2f}" for box in result.boxes] if result.boxes else []

        self.result_label.setText(
            f"检测到 {num_bees} 只蜜蜂\n"
            f"置信度: {', '.join(confidences) if confidences else '无检测结果'}"
        )

        self.statusBar().showMessage(f"检测完成，找到 {num_bees} 只蜜蜂")

    def batch_detect(self):
        """批量检测多张图像"""
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先加载模型")
            return

        # 选择输出文件夹
        output_dir = QFileDialog.getExistingDirectory(
            self, "选择结果保存文件夹"
        )

        if not output_dir:
            return

        # 选择要处理的图像
        image_paths, _ = QFileDialog.getOpenFileNames(
            self, "选择要处理的图像", "", "图像文件 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )

        if not image_paths:
            return

        # 创建进度对话框
        progress = QProgressDialog("处理中...", "取消", 0, len(image_paths), self)
        progress.setWindowTitle("批量处理")
        progress.setWindowModality(Qt.WindowModal)

        # 获取参数
        conf = self.conf_threshold.value()
        iou = self.iou_threshold.value()

        # 处理每张图像
        self.batch_results = []
        for i, image_path in enumerate(image_paths):
            if progress.wasCanceled():
                break

            progress.setValue(i)
            progress.setLabelText(f"处理中: {os.path.basename(image_path)} ({i + 1}/{len(image_paths)})")
            QApplication.processEvents()

            try:
                # 执行检测
                image = cv2.imread(image_path)
                results = self.model.predict(
                    source=image,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )

                result = results[0]
                num_bees = len(result.boxes) if result.boxes is not None else 0

                # 保存结果图像
                output_path = os.path.join(output_dir, f"detected_{os.path.basename(image_path)}")
                plotted_image = result.plot()
                cv2.imwrite(output_path, plotted_image)

                # 记录结果
                self.batch_results.append({
                    'image': os.path.basename(image_path),
                    'bees_detected': num_bees,
                    'confidences': [box.conf.item() for box in result.boxes] if result.boxes else [],
                    'image_path': image_path,
                    'output_path': output_path
                })

            except Exception as e:
                self.batch_results.append({
                    'image': os.path.basename(image_path),
                    'error': str(e),
                    'image_path': image_path
                })

        progress.setValue(len(image_paths))

        # 显示批量处理结果
        self.show_batch_results(output_dir)

        # 启用统计按钮
        self.btn_stats.setEnabled(True)

    def show_batch_results(self, output_dir):
        """显示批量处理结果"""
        total_images = len(self.batch_results)
        successful = sum(1 for r in self.batch_results if 'error' not in r)
        total_bees = sum(r['bees_detected'] for r in self.batch_results if 'bees_detected' in r)

        # 创建结果对话框
        result_dialog = QMessageBox(self)
        result_dialog.setWindowTitle("批量处理结果")
        result_dialog.setText(
            f"处理完成!\n\n"
            f"处理图像: {total_images}\n"
            f"成功处理: {successful}\n"
            f"检测到蜜蜂总数: {total_bees}\n"
            f"结果保存到: {output_dir}"
        )
        result_dialog.setStandardButtons(QMessageBox.Ok)
        result_dialog.exec_()

        # 更新统计信息
        self.update_statistics()

    def update_statistics(self):
        """更新统计信息"""
        successful_results = [r for r in self.batch_results if 'error' not in r]

        if not successful_results:
            self.stats_label.setText("无有效统计信息")
            return

        total_bees = sum(r['bees_detected'] for r in successful_results)
        avg_bees = total_bees / len(successful_results)

        # 计算置信度统计
        all_confidences = []
        for r in successful_results:
            all_confidences.extend(r['confidences'])

        avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
        min_confidence = min(all_confidences) if all_confidences else 0
        max_confidence = max(all_confidences) if all_confidences else 0

        # 计算蜜蜂数量分布
        bee_counts = [r['bees_detected'] for r in successful_results]
        count_distribution = defaultdict(int)
        for count in bee_counts:
            count_distribution[count] += 1

        # 格式化分布信息
        distribution_text = "蜜蜂数量分布:\n"
        for count in sorted(count_distribution.keys()):
            distribution_text += f"  {count}只: {count_distribution[count]}张图像\n"

        self.stats_label.setText(
            f"统计信息:\n"
            f"处理图像: {len(successful_results)}\n"
            f"蜜蜂总数: {total_bees}\n"
            f"平均每图像蜜蜂数: {avg_bees:.2f}\n"
            f"平均置信度: {avg_confidence:.2f}\n"
            f"最小置信度: {min_confidence:.2f}\n"
            f"最大置信度: {max_confidence:.2f}\n\n"
            f"{distribution_text}"
        )

    def show_statistics(self):
        """显示详细的统计图表"""
        if not self.batch_results:
            QMessageBox.information(self, "提示", "没有可用的统计数据，请先进行批量检测")
            return

        successful_results = [r for r in self.batch_results if 'error' not in r]
        if not successful_results:
            QMessageBox.information(self, "提示", "没有成功的检测结果可供统计")
            return

        # 创建统计对话框
        stats_dialog = QDialog(self)
        stats_dialog.setWindowTitle("蜜蜂检测统计图表")
        stats_dialog.setMinimumSize(800, 600)

        layout = QVBoxLayout(stats_dialog)

        # 创建标签页
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 蜜蜂数量分布图
        bee_counts = [r['bees_detected'] for r in successful_results]

        # 创建分布图
        dist_tab = QWidget()
        dist_layout = QVBoxLayout(dist_tab)

        # 创建matplotlib图形
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        dist_layout.addWidget(canvas)

        # 绘制柱状图
        ax = fig.add_subplot(111)
        counts, bins, patches = ax.hist(bee_counts, bins=range(0, max(bee_counts) + 2),
                                        align='left', rwidth=0.8, color='skyblue')

        ax.set_xlabel('蜜蜂数量')
        ax.set_ylabel('图像数量')
        ax.set_title('蜜蜂数量分布')
        ax.set_xticks(range(0, max(bee_counts) + 1))

        # 在柱子上添加数值标签
        for i, (count, patch) in enumerate(zip(counts, patches)):
            if count > 0:
                ax.text(patch.get_x() + patch.get_width() / 2, count + 0.05,
                        str(int(count)), ha='center', va='bottom')

        canvas.draw()
        tab_widget.addTab(dist_tab, "蜜蜂数量分布")

        # 置信度分布图
        conf_tab = QWidget()
        conf_layout = QVBoxLayout(conf_tab)

        fig2 = Figure(figsize=(10, 6))
        canvas2 = FigureCanvas(fig2)
        conf_layout.addWidget(canvas2)

        # 绘制置信度直方图
        all_confidences = []
        for r in successful_results:
            all_confidences.extend(r['confidences'])

        ax2 = fig2.add_subplot(111)
        ax2.hist(all_confidences, bins=20, color='lightcoral', alpha=0.7)
        ax2.set_xlabel('置信度')
        ax2.set_ylabel('频率')
        ax2.set_title('检测置信度分布')

        canvas2.draw()
        tab_widget.addTab(conf_tab, "置信度分布")

        # 添加关闭按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok)
        button_box.accepted.connect(stats_dialog.accept)
        layout.addWidget(button_box)

        stats_dialog.exec_()

    def closeEvent(self, event):
        """应用程序关闭事件"""
        self.save_settings()
        event.accept()


class MplCanvas(FigureCanvas):
    """Matplotlib画布组件"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


def main():
    """主函数"""
    app = QApplication(sys.argv)

    # 设置应用程序信息
    app.setApplicationName("蜜蜂检测系统")
    app.setApplicationVersion("1.0")

    window = BeeDetectionApp()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()