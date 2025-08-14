# Weather Chart Classification Trainer

基于文档 `docs/train.md` 第3.2节设计的CNN-RNN统一框架训练模块，用于气象图表的多标签分类任务。

## 架构概述

```
图像 → CNN编码器 → 特征投影 → 联合嵌入空间 ← 标签嵌入
                                     ↓
              RNN序列解码器 → {序列头, 并行头} → 要素预测
```

### 核心特性

- **CNN主干**: 使用ImageNet预训练的ResNet-50进行特征提取
- **RNN解码器**: 双层LSTM建模标签依赖关系
- **双重预测**: 序列预测头和并行BCE头结合
- **分阶段训练**: 预热阶段(冻结CNN) + 端到端微调
- **多任务损失**: BCE + 序列交叉熵 + 覆盖惩罚

## 安装依赖

```bash
pip install torch torchvision transformers
pip install scikit-learn pillow tqdm wandb
pip install pandas numpy
```

## 快速开始

### 1. 准备数据

数据格式支持两种方式：

**方式1: JSON格式**
```json
[
  {
    "image_path": "/path/to/image1.jpg",
    "labels": ["temperature", "pressure", "wind_speed"],
    "metadata": {}
  },
  {
    "image_path": "/path/to/image2.jpg", 
    "labels": ["humidity", "precipitation"],
    "metadata": {}
  }
]
```

**方式2: 目录结构**
```
dataset/
├── train/
│   ├── image1.jpg
│   ├── image1.json  # {"labels": ["temperature", "pressure"]}
│   ├── image2.jpg
│   └── image2.json
├── val/
└── test/
```

### 2. 准备标签映射

创建 `label_mapping.json` 文件:
```json
{
  "temperature": 0,
  "pressure": 1,
  "wind_speed": 2,
  "humidity": 3,
  "precipitation": 4
}
```

### 3. 配置训练参数

复制示例配置文件并修改:
```bash
cp traniner/examples/config_example.json my_config.json
```

主要配置项:
- `data.train_data_path`: 训练数据路径
- `data.val_data_path`: 验证数据路径  
- `data.label_mapping_path`: 标签映射文件路径
- `model.num_labels`: 标签数量
- `training.num_epochs`: 训练轮数
- `training.batch_size`: 批大小

### 4. 开始训练

```bash
# 使用配置文件训练
python -m traniner.train --config my_config.json

# 使用命令行参数训练
python -m traniner.train \
    --train_data_path ./dataset/train \
    --val_data_path ./dataset/val \
    --label_mapping_path ./dataset/label_mapping.json \
    --output_dir ./outputs \
    --num_epochs 50 \
    --batch_size 32
```

## 训练策略

### 分阶段训练

**阶段1: 预热训练 (Epochs 1-5)**
- 冻结CNN主干网络
- 仅优化RNN和投影层
- 使用较高学习率 (2e-3)

**阶段2: 端到端微调 (Epochs 6+)**
- 解冻CNN网络
- 差异化学习率: CNN (1e-4), RNN (5e-4)
- 精细调整整个网络

### 损失函数

多任务损失组合 (α=1.0, β=0.5, γ=0.1):
```
L = α·L_BCE + β·L_seq + γ·L_coverage
```

- **L_BCE**: 并行二元交叉熵，提供强监督信号
- **L_seq**: 序列交叉熵，建模标签依赖关系  
- **L_coverage**: 覆盖惩罚，抑制重复预测

### Teacher Forcing调度

训练期间逐渐减少真实标签的使用比例:
- 初始: 100% 使用真实标签
- 最终: 70% 使用真实标签
- 缓解训练与推理的分布差异

## 模型推理

### 生成预测

```python
from traniner import WeatherChartModel, LabelProcessor, create_dataloaders

# 加载模型
model = WeatherChartModel.from_pretrained("./outputs/best_model")

# 加载数据
test_loader, _, _ = create_dataloaders(config.data, label_processor)

# 生成预测
predictions = model.generate(
    images=batch["images"],
    max_length=10,
    beam_width=3,
    early_stopping=True
)
```

### Beam Search解码

使用集束搜索维护多个候选路径:
- 束宽度: 3-5 (平衡搜索质量与计算成本)
- 自然终止: 生成 `<eos>` 符号
- 长度限制: 最大序列长度10
- 早停策略: 概率阈值判断

## 评估指标

### 多标签分类指标
- **Macro/Micro F1**: 宏观和微观平均F1分数
- **Subset Accuracy**: 完全匹配准确率  
- **Hamming Loss**: 汉明损失
- **Per-label Metrics**: 每个标签的精确率、召回率、F1

### 序列特定指标
- **Sequence Accuracy**: 序列完全匹配率
- **Average Length**: 平均序列长度
- **Length Difference**: 长度差异统计

## 高级配置

### 类别不平衡处理

```json
{
  "training": {
    "use_focal_loss": true,
    "focal_alpha": 0.25,
    "focal_gamma": 2.0
  }
}
```

### 数据增强

```json
{
  "data": {
    "random_rotation": 10,
    "random_horizontal_flip": 0.5,
    "color_jitter_brightness": 0.1,
    "color_jitter_contrast": 0.1
  }
}
```

### Weights & Biases集成

```json
{
  "use_wandb": true,
  "wandb_project": "weather_chart_classification",
  "wandb_entity": "your_entity",
  "wandb_tags": ["cnn-rnn", "multi-label"]
}
```

## 分布式训练

```bash
# 单机多卡训练
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    -m traniner.train \
    --config config.json
```

## 故障排除

### 常见问题

1. **内存不足**
   - 减小 `batch_size`
   - 启用 `gradient_accumulation_steps`
   - 使用 `fp16` 混合精度

2. **标签序列过长**
   - 增加 `max_sequence_length` 
   - 调整 `max_labels_per_sample`

3. **收敛困难**
   - 延长 `warmup_epochs`
   - 降低学习率
   - 增加 `teacher_forcing_end` 比例

4. **类别不平衡**
   - 启用 `use_focal_loss`
   - 调整 `focal_alpha` 和 `focal_gamma`
   - 使用样本权重

### 性能优化

- 使用SSD存储数据
- 增加 `num_workers` 数量
- 启用 `pin_memory`
- 使用适当的 `batch_size`

## 目录结构

```
traniner/
├── __init__.py          # 模块导出
├── config.py            # 配置类定义
├── model.py             # CNN-RNN统一框架模型
├── trainer.py           # 训练器类
├── utils.py             # 工具函数
├── dataset.py           # 数据加载器
├── train.py             # 训练脚本
├── examples/            # 示例配置
│   ├── config_example.json
│   └── label_mapping_example.json
└── README.md            # 本文档
```

## 参考文献

详细理论背景和架构设计请参考项目文档 `docs/train.md` 第3.2节。
