# 目标检测与实例分割模型对比

本项目在VOC数据集上训练并测试Mask R-CNN和Sparse R-CNN两个目标检测/实例分割模型，并对比它们的性能。

## 项目结构

```
├── data/                  # 数据集相关代码
├── models/                # 模型定义
├── utils/                 # 辅助工具函数
├── scripts/               # 训练和测试脚本
├── results/               # 实验结果
├── requirements.txt       # 依赖包列表
└── README.md              # 项目说明
```

## 环境配置

```bash
# 安装依赖
pip install -r requirements.txt

# 安装Detectron2（如果上面安装失败）
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

## 数据准备

1. 下载VOC数据集：
```bash
python scripts/download_voc.py
```

## 模型训练

```bash
# 训练Mask R-CNN
python scripts/train_mask_rcnn.py

# 训练Sparse R-CNN
python scripts/train_sparse_rcnn.py
```

## 模型测试和可视化

```bash
# 测试并可视化模型结果
python scripts/test_and_visualize.py
```

## 实验结果

实验结果将保存在`results/`目录下，包括：
- 模型权重
- 训练日志
- 可视化图像
- 性能指标 
