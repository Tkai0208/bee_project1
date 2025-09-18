# bee_project1
数字图像处理课程设计
bee_project1/
├── BeeDetectorGUI.py        # 蜜蜂检测系统图形化界面
├── config/
│   └── settings.py          # 项目配置和路径设置
├── data/
│   └── conversion.py        # 数据格式转换功能
├── processing/
│   └── image_processing.py  # 图像处理功能
├── analysis/
│   └── dataset_analysis.py  # 数据集分析功能
├── model/
│   └── training.py          # 模型训练和评估功能
├── utils/
│   └── visualization.py     # 可视化功能
├── main.py                  # 主程序入口
└── requirements.txt         # 项目依赖
目录结构说明：

BeeDetectorGUI.py - 位于项目根目录，是蜜蜂检测系统的主要图形用户界面

config/ - 配置文件目录

settings.py - 包含项目路径、参数设置等配置信息

data/ - 数据处理目录

conversion.py - 数据格式转换功能（如COCO转YOLO格式）

processing/ - 图像处理目录

image_processing.py - 图像增强、预处理等功能

analysis/ - 数据分析目录

dataset_analysis.py - 数据集统计分析功能

model/ - 模型相关目录

training.py - 模型训练和评估功能

utils/ - 工具函数目录

visualization.py - 结果可视化功能

main.py - 项目主入口文件

requirements.txt - 项目依赖包列表
